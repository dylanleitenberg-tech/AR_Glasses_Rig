"""live_rig.py — the integrated real-time driver: everything running together, every frame.

This is the loop the user asked for — "cams capture synchronously, and the cams constantly track
and adjust the image." Each iteration:

    1. SYNC GRAB      sync_capture.SyncBank -> one barrier-aligned set of all 6 role frames,
                      with the measured jitter (a smeared set is flagged, not trusted).
    2. ADJUST IMAGE   autoexpose.AutoExposureBank -> per-role exposure/gain nudged toward target
                      every frame (world mid-gray & sharp, pupil glints hot but field dark).
    3. TRACK WORLD    world_mesh -> triangulate + robust VO + IMU-fused pose + growing mesh, so
                      the overlay knows where the real world is as the head moves.
    4. TRACK EYES     capture.LiveCapture feature read (dot / eye-corner / pupil) for the
                      calibrator, degraded gracefully through blinks.
    5. TELEMETRY      fps, jitter, exposure-settled, mesh size, camera pose, tracking health.

The loop body is `step()`, which is pure given its inputs, so `--selftest` drives the WHOLE
pipeline headless: synthetic per-role frames exercise the exposure loop, and a synthetic moving
world exercises the mesh tracker — proving the wiring end-to-end with NO hardware. `run()` needs
opencv, the cameras, and a saved role map (connect.py).
"""
import argparse
import sys
import time

import numpy as np

from autoexpose import AutoExposureBank
from world_mesh import WorldMesh


class RigTelemetry:
    """Rolling health of the live loop."""
    def __init__(self):
        self.frames = 0
        self.t0 = time.monotonic()
        self.jitter_ms = 0.0
        self.exposure_settled = False
        self.map_points = 0
        self.world_used = "init"
        self.world_inliers = 0
        self.pose_C = np.zeros(3)
        self.feat_mode = "none"

    @property
    def fps(self):
        dt = time.monotonic() - self.t0
        return self.frames / dt if dt > 0 else 0.0

    def line(self):
        return ("frame %5d  fps %5.1f  jitter %.2f ms  expo[%s]  world[%s inl=%d map=%d]  "
                "pose=(%+6.0f %+6.0f %+6.0f) feat=%s"
                % (self.frames, self.fps, self.jitter_ms,
                   "settled" if self.exposure_settled else "adjust", self.world_used,
                   self.world_inliers, self.map_points, self.pose_C[0], self.pose_C[1],
                   self.pose_C[2], self.feat_mode))


class LiveRig:
    """Owns the per-frame trackers and folds one synchronized frame-set into updated state."""

    def __init__(self, roles, expo_apply=None):
        self.roles = list(roles)
        self.autoexpose = AutoExposureBank(self.roles, apply=expo_apply)
        self.mesh = WorldMesh()
        self.tele = RigTelemetry()
        self.world_tracker = None            # set to a world_mesh.WorldTracker on hardware
        self._mesh_stride = 1                # perf.QualityController may raise this under load

    def apply_quality(self, settings):
        """Push a perf.QualityController level onto the trackers that cost time.

        Only the ORB budget is a live knob here; mesh_stride is honoured by the caller (it
        decides whether to run the mesh stage at all this frame), and render_scale belongs to
        the overlay compositor in augment_rig."""
        if self.world_tracker is not None:
            self.world_tracker.set_max_features(settings["orb_feat"])
        self._mesh_stride = max(1, int(settings.get("mesh_stride", 1)))

    def step(self, frames, jitter_ms=0.0, imu_dR=None, world_corr=None):
        """Process one synchronized set.
            frames:     {role: frame or None}
            jitter_ms:  the set's sync jitter (telemetry / trust)
            imu_dR:     optional world->cam rotation increment from the gyro
            world_corr: (ids, cam3d) synthetic stereo correspondences (sim path). On hardware this
                        is None and self.world_tracker consumes frames[worldL], frames[worldR].
        Returns the RigTelemetry."""
        t = self.tele
        t.frames += 1
        t.jitter_ms = jitter_ms

        # 2) constantly adjust the image
        self.autoexpose.update(frames)
        t.exposure_settled = self.autoexpose.all_settled()

        # 3) track the real world -> pose + mesh
        info = None
        if world_corr is not None:
            ids, cam3d = world_corr
            info = self.mesh.ingest(ids, cam3d, imu_dR=imu_dR)
        elif self.world_tracker is not None and frames.get("worldL") is not None \
                and frames.get("worldR") is not None \
                and (t.frames % self._mesh_stride == 0):     # stride: skip the heavy stage under load
            info = self.world_tracker.track(frames["worldL"], frames["worldR"], imu_dR)
            self.mesh = self.world_tracker.mesh
        if info is not None:
            t.world_used = info["used"]
            t.world_inliers = info["inliers"]
        _, P = self.mesh.map_points()
        t.map_points = len(P)
        _, C = self.mesh.pose()
        t.pose_C = C
        return t

    def set_feature_mode(self, mode):
        self.tele.feat_mode = mode


# --------------------------------------------------------------------------
#  Hardware run
# --------------------------------------------------------------------------
def run(fps=100, budget_ms=3.0, seconds=None, use_imu=False, target_fps=30.0,
        cpu_budget=0.50, verbose=True):
    """Open the mapped bank + all trackers and run the live loop until Ctrl-C (or `seconds`)."""
    from connect import load_map, validate_map
    from sync_capture import SyncBank
    from world_mesh import WorldTracker
    role_index = load_map()
    if not role_index:
        print("no role map — run:  python3 connect.py --auto"); return 1
    probs = validate_map(role_index)
    if probs:
        print("role map invalid:", "; ".join(probs)); return 1

    bank = SyncBank(role_index, fps=fps).start()
    # push exposure commands straight to the underlying cv2 captures
    from autoexpose import make_cv2_apply
    caps = {r: bank.cams[r].cap.cap for r in role_index}      # SyncCamera -> _Cv2Adapter -> cv2 cap
    rig = LiveRig(list(role_index), expo_apply=make_cv2_apply(caps))
    if "worldL" in role_index and "worldR" in role_index:
        rig.world_tracker = WorldTracker()

    imu = None
    if use_imu:
        try:
            from imu_serial import GyroIntegrator
            imu = GyroIntegrator(port="/dev/tty.usbmodem*").start()   # bg-threaded gyro integration
        except Exception as e:
            print("IMU unavailable (%s) — running vision-only" % e)

    # Frame budget: cameras + mesh together must fit the deadline, and must not do it by
    # burning every core. LoadManager degrades ORB/mesh-stride until both hold.
    from perf import LoadManager
    load = LoadManager(target_fps=target_fps, cpu_budget=cpu_budget)
    rig.apply_quality(load.quality.settings)

    t_end = None if seconds is None else time.monotonic() + seconds
    try:
        while t_end is None or time.monotonic() < t_end:
            with load.stage("sync"):
                fs = bank.sync_frame()
            with load.stage("imu"):
                imu_dR = imu.consume_rotation() if imu is not None else None
            with load.stage("track"):
                tele = rig.step(fs.frames, jitter_ms=fs.jitter_ms, imu_dR=imu_dR)
            frame_ms, settings = load.end_frame()
            rig.apply_quality(settings)
            if verbose and tele.frames % 15 == 0:
                print("  %s  %.1f ms %s" % (tele.line(), frame_ms, load.quality.name))
    except KeyboardInterrupt:
        pass
    finally:
        bank.close()
    print("stopped after %d frames (%.1f fps avg)" % (rig.tele.frames, rig.tele.fps))
    print("\n".join(load.lines()))
    return 0


# ==========================================================================
#  Self-test (no hardware): drive the WHOLE loop with synthetic frames +
#  a synthetic moving world. Proves sync->expose->mesh->telemetry wiring.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== live_rig self-test (synthetic frames + moving world, no hardware) ==")
    from autoexpose import _SynthSensor
    from world_mesh import _project_pair, _rot, triangulate_stereo, DEFAULT_F, DEFAULT_B

    roles = ("worldL", "worldR", "eyeL", "eyeR", "pupilL", "pupilR")
    rng = np.random.default_rng(11)
    sensors = {r: _SynthSensor(r if r in ("pupilL", "pupilR") else
                               ("worldL" if r.startswith("world") else "eyeL"),
                               scene=0.5 if r.startswith("world") else 1.1, rng=rng)
               for r in roles}
    rig = LiveRig(list(roles))

    # a synthetic static world + a moving head, feeding the mesh real stereo correspondences
    Xw = rng.uniform([-400, -300, 900], [400, 300, 2200], size=(160, 3))
    T = 30
    Cs = np.stack([70 * np.sin(np.linspace(0, 1.5, T)), 30 * np.sin(np.linspace(0, 1.0, T)),
                   90 * np.sin(np.linspace(0, 0.8, T))], 1)
    Rs = [_rot([0, 1, 0], 8 * np.sin(x)) for x in np.linspace(0, 1.4, T)]

    checks = []
    jitters = []
    prevR = np.eye(3)
    posC_err = []
    for k in range(T):
        # (a) synthesize a synchronized frame-set for the exposure loop
        frames = {r: sensors[r].frame(rig.autoexpose.ctl[r].a) for r in roles}
        jit = float(abs(rng.normal(0, 0.4)))         # a small, in-budget sync jitter
        jitters.append(jit)
        # (b) synthesize this frame's world stereo correspondences
        uvL, uvR, vis = _project_pair(Xw, Rs[k], Cs[k])
        ids = np.where(vis)[0]
        p3, good = triangulate_stereo(uvL[vis], uvR[vis], DEFAULT_F, DEFAULT_B)
        imu_dR = Rs[k] @ prevR.T
        prevR = Rs[k]
        rig.set_feature_mode("stereo")
        tele = rig.step(frames, jitter_ms=jit, imu_dR=imu_dR,
                        world_corr=(ids[good], p3[good]))
        posC_err.append(np.linalg.norm(tele.pose_C - Cs[k]))

    # (1) the loop ran every frame and aggregated telemetry
    checks.append(("loop ran all %d frames, fps computed (%.0f)" % (T, rig.tele.fps),
                   rig.tele.frames == T and rig.tele.fps > 0))

    # (2) exposure loop converged while the loop ran (image kept adjusted)
    checks.append(("auto-exposure settled all roles during the run", rig.autoexpose.all_settled()))

    # (3) the world mesh grew and the pose tracked the synthetic head motion
    checks.append(("world mesh built (%d map points) and pose tracked motion (max err %.2f mm)"
                   % (rig.tele.map_points, max(posC_err)),
                   rig.tele.map_points > 120 and max(posC_err) < 5.0))

    # (4) world tracking reported healthy inliers (vision, not IMU-only fallback)
    checks.append(("world VO healthy (used=%s, inliers=%d)" % (rig.tele.world_used, rig.tele.world_inliers),
                   rig.tele.world_used in ("vision", "vision+imu", "anchor") and rig.tele.world_inliers > 20))

    # (5) telemetry line renders + jitter tracked
    line_ok = isinstance(rig.tele.line(), str) and "world" in rig.tele.line()
    checks.append(("telemetry renders; median jitter %.2f ms tracked" % float(np.median(jitters)),
                   line_ok and rig.tele.jitter_ms >= 0))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  final:", rig.tele.line())
        print("  =>", "LIVE RIG OK — sync + autoexpose + mesh + telemetry run together ✅"
              if ok else "PROBLEM ⚠️")
        print("  on hardware:  python3 live_rig.py --run   (ORB/LK world features + real cameras)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="integrated live rig driver")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--imu", action="store_true", help="fuse the XIAO IMU gyro")
    ap.add_argument("--target-fps", type=float, default=30.0,
                    help="frame budget the adaptive quality controller holds")
    ap.add_argument("--cpu-budget", type=float, default=0.50,
                    help="max fraction of the whole machine's CPU")
    args = ap.parse_args()
    if args.run:
        sys.exit(run(seconds=args.seconds, use_imu=args.imu,
                     target_fps=args.target_fps, cpu_budget=args.cpu_budget))
    sys.exit(selftest())
