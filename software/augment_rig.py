"""augment_rig.py — the "turn people into monkeys" runtime: a world-locked AR overlay on moving
people, on top of the calibrated rig.

Per frame it chains everything that was built:

    SyncBank (synchronized grab)                     [sync_capture]
      -> AutoExposureBank (keep the image usable)     [autoexpose]
      -> WorldTracker/WorldMesh (head pose in world)  [world_mesh]
      -> PeopleTracker (each person's 3D + stable id)  [people_track]
      -> AnchorProjector (calibrated display map)      [anchor]
      -> MonkeyAvatar.render_all + compose             [avatar]
      -> Overlay onto the AR display                   [overlay]

Because each monkey's billboard corners are WORLD points re-projected through the current head
pose + the calibrated display map, the monkeys stay glued to the people as BOTH the people and the
wearer's head move — the whole point.

PRECONDITION: the glasses must already be CALIBRATED (the human-in-the-loop loop has trained the
pixel map, and connect.py has mapped the cameras). This runtime consumes that calibration; it does
not create it. In `--run` the display map is the trained calibrator; in `--selftest` it is the
synthetic device model, so the full people->monkeys->locked pipeline is proven with NO hardware.
"""
import argparse
import sys
import time

import numpy as np

from anchor import AnchorProjector
from people_track import PeopleTracker
from avatar import MonkeyAvatar, compose, RenderSmoother


class AugmentTelemetry:
    def __init__(self):
        self.frames = 0
        self.t0 = time.monotonic()
        self.people = 0
        self.monkeys = 0
        self.jitter_ms = 0.0
        self.pose_C = np.zeros(3)

    @property
    def fps(self):
        dt = time.monotonic() - self.t0
        return self.frames / dt if dt > 0 else 0.0

    def line(self):
        return ("frame %5d  fps %5.1f  jitter %.2f ms  people %d  monkeys-drawn %d  head=(%+5.0f %+5.0f %+5.0f)"
                % (self.frames, self.fps, self.jitter_ms, self.people, self.monkeys,
                   self.pose_C[0], self.pose_C[1], self.pose_C[2]))


class MonkeyAugmenter:
    """Owns the people tracker, the monkey avatar, and the world-locked projector; folds one
    frame (world detections + head pose) into a display canvas of monkeys."""

    def __init__(self, display_map, display_w=1920, display_h=1080, texture=None,
                 smooth_alpha=0.5):
        self.projector = AnchorProjector(display_map)
        self.people = PeopleTracker()
        self.avatar = MonkeyAvatar(texture=texture)
        self.smoother = RenderSmoother(alpha=smooth_alpha)   # EMA the drawn quad -> less jitter
        self.DW, self.DH = display_w, display_h
        self.tele = AugmentTelemetry()
        self.render_scale = 1.0          # perf.QualityController lowers this under load
        self.detect_stride = 1           # ...and raises this (tracker dead-reckons between detects)

    def apply_quality(self, settings):
        """Adopt a perf.QualityController level. `render_scale` shrinks the compositing buffer
        (cost scales with its square); `detect_stride` is read by the caller, which owns the
        detector — PeopleTracker already dead-reckons through gaps, so skipping detections
        degrades latency-to-new-person, not track continuity."""
        self.render_scale = float(settings.get("render_scale", 1.0))
        self.detect_stride = max(1, int(settings.get("detect_stride", 1)))
        return self.render_scale

    def step(self, detL, detR, pose, jitter_ms=0.0):
        """detL/detR: people_track.Detection lists from the two world cams (on hardware these come
        from the detector); pose: (R_cw, C) from the world mesh. Returns (canvas, tracks)."""
        t = self.tele
        t.frames += 1
        t.jitter_ms = jitter_ms
        t.pose_C = np.asarray(pose[1], float)
        tracks = self.people.update(detL, detR, pose)
        items = self.avatar.render_all(tracks, self.projector, pose)
        # keep only monkeys that actually intersect the display window (clip the rest)
        items = [it for it in items if _quad_hits_display(it.quad)]
        for it in items:                                     # temporal smoothing per identity
            it.quad = self.smoother.smooth(it.id, it.quad)
        self.smoother.prune({it.id for it in items})
        canvas = compose(items, self.DW, self.DH, self.render_scale)
        t.people = len(tracks)
        t.monkeys = len(items)
        return canvas, tracks


def _quad_hits_display(quad):
    """True if the (possibly off-screen) billboard quad overlaps [0,1]^2 at all."""
    q = np.asarray(quad)
    return not (q[:, 0].max() < 0 or q[:, 0].min() > 1 or q[:, 1].max() < 0 or q[:, 1].min() > 1)


# --------------------------------------------------------------------------
#  Hardware run
# --------------------------------------------------------------------------
def run(fps=100, use_imu=False, verbose=True):
    """Live: calibrated map + detector + world mesh + people -> monkeys on the AR display. Needs
    opencv, the cameras, a saved role map (connect.py), AND a trained calibrator (the calibration
    loop must have been run first)."""
    import cv2
    from connect import load_map, validate_map
    from sync_capture import SyncBank
    from autoexpose import AutoExposureBank, make_cv2_apply
    from world_mesh import WorldTracker
    from people_track import CvPeopleDetector
    from anchor import calibrator_display_map
    from config import Config
    from calibrator import Calibrator
    from dataset import Dataset

    role_index = load_map()
    if not role_index:
        print("no role map — run:  python3 connect.py --auto"); return 1
    if validate_map(role_index):
        print("role map invalid — re-run connect.py"); return 1
    cfg = Config()

    # load the trained calibrator (the glasses must already be calibrated)
    ds = Dataset(cfg.db_path, cfg.feature_names)
    X, Y, Wt = ds.load()
    if len(X) < cfg.min_samples_for_model:
        print("glasses not calibrated (%d samples) — run the calibration loop first "
              "(main.py --calibrate-corners, then collect samples)." % len(X))
        return 1
    cal = Calibrator(cfg.n_features, degree=cfg.poly_degree,
                     min_samples=cfg.min_samples_for_model)
    cal.fit(X, Y, Wt)

    bank = SyncBank(role_index, fps=fps).start()
    caps = {r: bank.cams[r].cap.cap for r in role_index}
    expo = AutoExposureBank(list(role_index), apply=make_cv2_apply(caps))
    world = WorldTracker()
    detector = CvPeopleDetector()

    # display map = the trained calibrator, folding in the LIVE eye-corner features each frame
    latest_eye = {"feat": np.zeros(cfg.n_features - 4)}
    disp_map = calibrator_display_map(cal, cfg, lambda: latest_eye["feat"])
    aug = MonkeyAugmenter(disp_map, cfg.display_w, cfg.display_h)

    imu = None
    if use_imu:
        try:
            from imu_serial import GyroIntegrator
            imu = GyroIntegrator(port="/dev/tty.usbmodem*").start()
        except Exception as e:
            print("IMU unavailable (%s) — vision-only" % e)

    win = "monkey-overlay"                       # black = transparent on the see-through display
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("MONKEYS LIVE — q to stop")
    try:
        while True:
            fs = bank.sync_frame()
            expo.update(fs.frames)
            imu_dR = imu.consume_rotation() if imu is not None else None
            wl, wr = fs.get("worldL"), fs.get("worldR")
            if wl is None or wr is None:
                continue
            world.track(wl, wr, imu_dR)
            pose = world.mesh.pose()
            # TODO(hardware): populate latest_eye["feat"] from the eye-corner trackers each frame
            detL, detR = detector.detect(wl), detector.detect(wr)
            canvas, _ = aug.step(detL, detR, pose, fs.jitter_ms)
            cv2.imshow(win, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            if verbose and aug.tele.frames % 15 == 0:
                print("  " + aug.tele.line())
    except KeyboardInterrupt:
        pass
    finally:
        bank.close(); cv2.destroyAllWindows()
    return 0


# ==========================================================================
#  Self-test (no hardware): synthetic moving people + moving head; assert each
#  visible person is covered by a WORLD-LOCKED monkey that tracks them frame to
#  frame, IDs stay stable, and monkeys depth-sort. The whole pipeline, headless.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== augment_rig self-test (moving people + moving head -> locked monkeys) ==")
    from anchor import _make_synthetic_device
    from people_track import _project_box, UP
    from world_mesh import DEFAULT_F, DEFAULT_B
    from rig import WORLD_RES

    display_map, gt = _make_synthetic_device()
    aug = MonkeyAugmenter(display_map)
    f = DEFAULT_F; B = DEFAULT_B; W = WORLD_RES; H = int(W * 3 // 4); cx = W / 2; cy = H / 2

    # people walking across the scene, 3.5-6 m out (so they fit the display FOV), + a moving head
    people = [dict(pos=np.array([-800.0, 0, 4200.0]), vel=np.array([30.0, 0, 0.0]), h=1700.0),
              dict(pos=np.array([600.0, 0, 5200.0]), vel=np.array([-22.0, 0, 8.0]), h=1600.0)]

    def head_pose(k):
        yaw = np.radians(7 * np.sin(k * 0.2))
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        return R, np.array([15.0 * np.sin(k * 0.25), 0.0, 0.0])

    checks = []
    id_sets = [set(), set()]
    lock_err_px = []
    monkeys_each = []
    for k in range(24):
        R_cw, C = head_pose(k)
        detL, detR, truth = [], [], []
        for p in people:
            p["pos"] = p["pos"] + p["vel"]
            truth.append(p["pos"].copy())
            dl = _project_box(p["pos"], p["h"], R_cw, C, f, cx, cy, W, H, 0.0)
            dr = _project_box(p["pos"], p["h"], R_cw, C, f, cx, cy, W, H, B)
            if dl and dr:
                detL.append(dl); detR.append(dr)
        canvas, tracks = aug.step(detL, detR, (R_cw, C), jitter_ms=0.3)
        monkeys_each.append(aug.tele.monkeys)
        # each person should have a monkey whose centre sits on the person's display projection
        for pi, tp in enumerate(truth):
            near = min(tracks, key=lambda t: np.linalg.norm(t.pos - tp), default=None)
            if near is None:
                continue
            id_sets[pi].add(near.id)
            monkey = aug.avatar.render(near, aug.projector, (R_cw, C))
            person_uv = gt(tp + UP * 0, (R_cw, C))          # where the person's centre truly is
            if monkey is not None and person_uv is not None:
                center = monkey.quad.mean(0)
                lock_err_px.append(np.hypot((center[0] - person_uv[0]) * 1920,
                                            (center[1] - person_uv[1]) * 1080))

    # (1) after the tracker's brief confirmation warmup, a monkey is drawn for each person
    checks.append(("a monkey is drawn for each visible person post-warmup (min %d after frame 2)"
                   % min(monkeys_each[2:]), min(monkeys_each[2:]) >= 2))
    # (2) WORLD-LOCK: each monkey's centre stays on its person through people+head motion
    checks.append(("monkeys stay locked on moving people (median centre err %.1f px)"
                   % np.median(lock_err_px), np.median(lock_err_px) < 40))
    # (3) stable IDs -> a monkey doesn't jump identity between people
    checks.append(("each person keeps one monkey identity (id counts %s)"
                   % [len(s) for s in id_sets], all(len(s) == 1 for s in id_sets)))
    # (4) canvas is a valid overlay: lit on people, otherwise transparent (black)
    lit_frac = (canvas.sum(axis=2) > 0).mean()
    checks.append(("overlay canvas is mostly transparent (%.1f%% lit), only monkeys drawn"
                   % (100 * lit_frac), 0.0 < lit_frac < 0.5))
    # (5) telemetry renders
    checks.append(("telemetry line renders", isinstance(aug.tele.line(), str)
                   and "monkeys" in aug.tele.line()))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  final:", aug.tele.line())
        print("  =>", "AUGMENT RIG OK — moving people become world-locked monkeys ✅"
              if ok else "PROBLEM ⚠️")
        print("  --run needs: calibrated glasses (trained calibrator) + connect.py map + cameras.")
        print("  swap MonkeyAvatar texture / detector for realism; the lock is calibration-driven.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="turn moving people into world-locked monkeys")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--imu", action="store_true")
    args = ap.parse_args()
    if args.run:
        sys.exit(run(use_imu=args.imu))
    sys.exit(selftest())
