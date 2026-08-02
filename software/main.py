#!/usr/bin/env python3
"""AR eye-corner pixel calibration — real-time loop.

Modes:
  python3 main.py --selftest            headless convergence test (numpy only)
  python3 main.py --simulate            interactive GUI sim (align red onto green)
  python3 main.py --world-cam-left 0 --world-cam-right 1 --eye-cam-left 2 --eye-cam-right 3
  python3 main.py --list-cams           probe which camera indices are connected
  python3 main.py --calibrate-corners   capture the eye-corner templates (do once)

The live loop each frame:
  1. read features  = [worldL_dot, worldR_dot, eyeL_corner, eyeR_corner] (8, norm [0,1])
  2. predicted pixel = calibrator.predict(features)
  3. shown pixel     = predicted + your accumulated joystick/key nudge
  4. render red dot (sim also renders the green truth target)
  5. on APPROVE: high-speed snapshot the eye corners, store
       (features -> shown pixel), retrain, reset nudge, next dot.
"""
import argparse
import sys
import time

import numpy as np

from config import Config
from dataset import Dataset
from calibrator import Calibrator
import autosim
import rig


# ----------------------------------------------------------------------
#  Shared: load dataset + (re)fit a calibrator
# ----------------------------------------------------------------------
def make_lambdas(cfg: Config) -> np.ndarray:
    return np.logspace(np.log10(cfg.lambda_lo), np.log10(cfg.lambda_hi),
                       cfg.lambda_steps)


def new_calibrator(cfg: Config) -> Calibrator:
    return Calibrator(cfg.n_features, cfg.poly_degree, cfg.min_samples_for_model,
                      make_lambdas(cfg), cfg.robust_iters, cfg.huber_k)


def build_calibrator(cfg: Config, ds: Dataset) -> Calibrator:
    cal = new_calibrator(cfg)
    X, Y, W = ds.load()
    cal.fit(X, Y, W)
    return cal


# ----------------------------------------------------------------------
#  Headless self-test: prove the learning loop converges, no hardware/GUI
# ----------------------------------------------------------------------
def run_selftest(cfg: Config, iterations: int = 240, seed: int = 0) -> int:
    print("== headless self-test (anatomy simulator, one subject) ==")
    print("Closed-loop calibration with the glasses slipping naturally each step and")
    print("~12% GROSS outliers injected (mis-tracked corners / fat-fingered approves)")
    print("to prove robustness. 'residual' = the overlay miss BEFORE the correction.\n")
    sim = autosim.Simulator(seed)
    subject = sim.new_subject()
    dev = sim.seat()
    rng = np.random.default_rng(seed + 1)
    cal = new_calibrator(cfg)
    X = np.empty((0, cfg.n_features)); Y = np.empty((0, 2)); Wt = np.empty((0,))
    window = []
    outliers = 0
    print("%6s | %12s | %9s | %8s | %s" %
          ("sample", "residual px", "median px", "lambda", "model"))
    print("-" * 60)
    for i in range(1, iterations + 1):
        dev = sim.slip(dev)                           # glasses move, naturally
        obs = sim.observe(subject, dev, sim.world_point())
        if obs is None:
            continue
        f, label, truth = obs
        residual = float(np.linalg.norm(cal.predict(f) - truth))   # real overlay miss
        window.append(residual)
        if len(window) > 25:
            window.pop(0)
        feat, approved = f, label                     # train on the human answer
        conf = float(np.clip(rng.normal(0.85, 0.1), 0.2, 1.0))
        if rng.random() < 0.12:                       # inject a gross outlier
            outliers += 1
            if rng.random() < 0.5:
                approved = rng.uniform(0, 1, 2)        # fat-fingered approve
            else:
                feat = np.clip(f + rng.normal(0, 0.25, f.shape), 0, 1)  # bad track
            conf = min(conf, float(rng.uniform(0.2, 0.45)))
        X = np.vstack([X, feat]); Y = np.vstack([Y, approved])
        Wt = np.append(Wt, conf)
        cal.fit(X, Y, Wt)
        if i % 20 == 0:
            avg = float(np.mean(window))
            he = cal.holdout_error(X, Y, Wt)
            med = he[1] * 1080 if he else float("nan")
            print("%6d | %12.1f | %9.1f | %8.3f | %s" %
                  (i, avg * 1080, med, cal.lambda_ or 0,
                   "trained" if cal.is_trained else "warming up"))
    final = float(np.mean(window))
    he = cal.holdout_error(X, Y, Wt)
    print("-" * 60)
    print("injected outliers : %d of %d samples (%.0f%%)"
          % (outliers, iterations, 100.0 * outliers / iterations))
    print("final residual    : %.1f px @1080p (auto-prediction vs truth)"
          % (final * 1080))
    if he is not None:
        print("holdout error     : %.1f px RMS / %.1f px median @1080p"
              % (he[0] * 1080, he[1] * 1080))
    auto_med = he[1] if he else 0.0
    ok = final < 0.025 and auto_med < 0.025
    print("\nRESULT:", "PASS — learned an accurate map despite outliers ✅" if ok
          else "WEAK — residual still high ⚠️")
    return 0 if ok else 1


# ----------------------------------------------------------------------
#  Camera helpers (import cv2-dependent modules lazily)
# ----------------------------------------------------------------------
def run_list_cams() -> int:
    from cameras import list_cameras
    print("Probing camera indices 0..7 ...")
    for idx, ok in list_cameras():
        print("  index %d : %s" % (idx, "OK, frames flowing" if ok else
                                    "opens but no frame"))
    print("Map these to --world-cam-left/right and --eye-cam-left/right "
          "(world + eye-corner base of the 6-cam binocular build: 2 world + 2 eye-corner;")
    print("the 2 NIR pupil cams attach via the pupil/live_features path, +2 stereo = 8-cam FULL)")
    return 0


# Minimum eye-corner template size, in CAPTURE pixels. Measured on this rig 2026-08-02: the
# cross-frame localisation margin is dominated by template SIZE, not by placement or contrast
# enhancement. Sweep at the eyeR corner, cropping one frame and matching into a frame 1.2 s later:
#     30px 0.041  45px 0.049  60px 0.057  80px 0.064  110px 0.066  150px 0.091  200px 0.207
# Enforcing 150 took the two real captures from 0.046/0.021 to 0.121/0.057, confirming size is the
# lever; 200 is where eyeR clears the bar with room to spare.
#
# TRADE-OFF, deliberately taken: 200 px is a quarter of the frame height, so the patch is as much
# "face around the corner" as it is the canthus, and rig.py models the tracked point as the canthus
# ITSELF. A constant offset is absorbed by the calibrator (it learns a mapping), and the patch
# still MOVES with the corner, which is what the features encode. What it costs is sensitivity to
# soft-tissue deformation -- and SOFT_TISSUE_SD=0.15mm in rig.py is a "holding still" placeholder
# that squinting/talking already blows past. So: hold a neutral face when calibrating. A lock that
# wanders is useless; a slightly impure landmark is merely biased, and bias is learnable.
TEMPLATE_MIN_PX = 200


def run_calibrate_corners(cfg: Config, only=None) -> int:
    """Capture the eye-corner templates.

    LIVE preview first, then freeze and draw. The original version grabbed ONE frame per eye and
    made you draw on whatever it happened to catch — blink, look away, or catch a bad exposure and
    your only recourse was to re-run the whole thing and redo the eye you had already got right.
    That cost two runs on 2026-08-02. Now: SPACE freezes the frame you like, ENTER accepts the
    box, and `only` re-does a single eye without touching the other.

    Templates need STRUCTURE, not just position: a crop of smooth skin correlates ~1.0 with any
    other smooth patch, so it scores perfectly and localises nowhere (see
    calib_preflight.template_margin). The live view prints a running texture reading so you can
    see whether what you are about to box has any edges in it at all."""
    import cv2
    import numpy as np
    from cameras import Camera
    from eye_tracker import EyeCornerTracker

    targets = [t for t in (("eyeL", cfg.eye_cam_left), ("eyeR", cfg.eye_cam_right))
               if only is None or t[0].lower() == ("eye" + only.lower())]
    if not targets:
        print("  !! nothing to do for --eye %s" % only)
        return 1

    print("Capturing eye-corner templates (%s)." % ", ".join(n for n, _ in targets))
    print("  LIVE view per eye:  SPACE = freeze this frame   Q = skip this eye")
    print("  then drag a box CENTRED on the corner (lid margin / lash roots / caruncle), ENTER.")
    print("  DRAW IT BIG — roughly 150-200 px, not a tight little crop. MEASURED on this rig")
    print("  2026-08-02, cross-frame localisation margin vs box size:")
    print("      30px 0.038   45px 0.058   |   60px 0.122   80px 0.203   150px 0.211   200px 0.368")
    print("  Under ~60 px the template does not localise at all: this eye is low-contrast enough")
    print("  that a small crop matches everywhere. Bigger wins until the box stops being about")
    print("  the corner. Trade-off: a big box takes in more skin, which deforms when you squint")
    print("  or talk, so keep the CORNER at its centre and hold a neutral face when calibrating.")

    for name, idx in targets:
        cam = Camera(idx, cfg.cam_width, cfg.cam_height, name=name)
        win = "%s — SPACE freezes, Q skips" % name
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        frame, frozen = None, None
        while True:
            frame = cam.read()
            if frame is None:
                print("  !! no frame from %s (index %d)" % (name, idx)); cam.release()
                cv2.destroyWindow(win)
                return 1
            shown = frame.copy()
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            tex = cv2.Laplacian(g, cv2.CV_64F).var()
            sat = float((g >= 250).mean() * 100)
            cv2.putText(shown, "texture %6.1f   saturated %4.1f%%" % (tex, sat), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, shown)
            k = cv2.waitKey(30) & 0xFF
            if k == 32:                      # SPACE — freeze this one
                frozen = frame.copy()
                break
            if k in (ord("q"), 27):
                break
        cv2.destroyWindow(win)
        if frozen is None:
            cam.release()
            print("  %s: skipped (no frame frozen)" % name)
            continue

        tracker = EyeCornerTracker("%s/%s.png" % (cfg.template_dir, name))
        # You place the CENTRE; min_px guarantees the size. selectROI shows a scaled window with
        # no size readout, so judging 150 image-pixels by eye is not a thing a person can do.
        ok = tracker.calibrate(frozen, "select %s corner" % name, min_px=TEMPLATE_MIN_PX)
        cam.release()
        if not ok:
            print("  %s: skipped (box too small)" % name)
            continue
        # Report the template's own texture immediately — a flat one is worth knowing about NOW,
        # not 30 corrections into a run when the tracker starts drifting.
        t = cv2.imread("%s/%s.png" % (cfg.template_dir, name), cv2.IMREAD_GRAYSCALE)
        if t is not None:
            tv = cv2.Laplacian(t, cv2.CV_64F).var()
            flag = "" if tv >= 20 else "   <-- LOW TEXTURE, expect a weak lock; re-grab on edges"
            print("  %s: saved  %dx%d px, texture %.1f%s" % (name, t.shape[1], t.shape[0], tv, flag))
        else:
            print("  %s: saved" % name)
    return 0


# ----------------------------------------------------------------------
#  The live loop (real hardware OR interactive --simulate)
# ----------------------------------------------------------------------
def run_loop(cfg: Config, simulate: bool) -> int:
    # GUI + input always needed here
    from overlay import Overlay
    from input_ctl import InputController

    cams = []
    if not simulate:
        from cameras import Camera
        from dot_detector import DotDetector
        from eye_tracker import EyeCornerTracker
        world_L = Camera(cfg.world_cam_left, cfg.cam_width, cfg.cam_height, name="worldL")
        world_R = Camera(cfg.world_cam_right, cfg.cam_width, cfg.cam_height, name="worldR")
        eyeL_cam = Camera(cfg.eye_cam_left, cfg.cam_width, cfg.cam_height, name="eyeL")
        eyeR_cam = Camera(cfg.eye_cam_right, cfg.cam_width, cfg.cam_height, name="eyeR")
        cams = [world_L, world_R, eyeL_cam, eyeR_cam]
        detector = DotDetector()
        trk_L = EyeCornerTracker("%s/eyeL.png" % cfg.template_dir, cfg.search_margin)
        trk_R = EyeCornerTracker("%s/eyeR.png" % cfg.template_dir, cfg.search_margin)
        if not (trk_L.ready and trk_R.ready):
            print("Eye-corner templates missing. Run --calibrate-corners first.")
            return 1
    else:
        world = autosim.Simulator(int(time.time()) % 10000)
        sim_subject = world.new_subject()
        sim_dev = world.seat()
        sim_pts = _serpentine_points()                 # ordered (snake) practice targets
        sim_i = 0
        sim_features, sim_truth, sim_i = _sim_next_ordered(
            world, sim_subject, sim_dev, sim_pts, sim_i)

    ds = Dataset(cfg.db_path, cfg.feature_names)
    cal = build_calibrator(cfg, ds)
    overlay = Overlay(cfg.display_w, cfg.display_h, cfg.overlay_window,
                      cfg.fullscreen, cfg.dot_radius_px)
    inp = InputController(cfg.nudge_gain, cfg.deadzone, cfg.approve_button,
                          cfg.quit_button)

    nudge = np.zeros(2)
    last_features = np.full(cfg.n_features, 0.5)
    smooth_feat = None             # EMA of live eye features (real mode)
    pred_s = None                  # EMA of the predicted pixel (reduces jitter)
    conf = 1.0
    n_saved = 0
    print("Loop running. %s. Q/ESC quits."
          % ("joystick + keyboard" if inp.has_joystick else "keyboard (no pad)"))

    try:
        while True:
            green = None
            if simulate:
                features, conf = sim_features, 1.0
                green = tuple(sim_truth)
            else:
                features, conf, ok = read_real_features(
                    world_L, world_R, eyeL_cam, eyeR_cam, detector, trk_L, trk_R,
                    last_features)
                if not ok:
                    key = overlay.render(tuple(np.clip(
                        cal.predict(last_features) + nudge, 0, 1)),
                        hud="searching for dot / eye corners...")
                    if inp.poll(key).quit:
                        break
                    continue
                # EMA-smooth the live eye features to steady the prediction
                smooth_feat = (features if smooth_feat is None else
                               cfg.feat_smooth * features +
                               (1 - cfg.feat_smooth) * smooth_feat)
                features = smooth_feat
                last_features = features

            predicted = cal.predict(features)
            pred_s = predicted if pred_s is None else (
                cfg.pred_smooth * predicted + (1 - cfg.pred_smooth) * pred_s)
            shown = np.clip(pred_s + nudge, 0.0, 1.0)

            hud = make_hud(ds.count(), cal, simulate, shown, green, conf)
            key = overlay.render(tuple(shown), green, hud)
            act = inp.poll(key)

            if act.quit:
                break
            if act.recalibrate and not simulate:
                print("Re-run --calibrate-corners for fresh templates.")
            if act.reset:                          # RESET fallback: cancel a mis-nudge
                nudge[:] = 0.0
                pred_s = None
                print("  reset — current nudge cancelled (nothing stored)")
            if act.undo:                           # UNDO fallback: drop the last bad sample
                removed = ds.undo_last()
                if removed is None:
                    print("  undo — nothing to undo (no samples stored)")
                else:
                    cal = build_calibrator(cfg, ds)   # retrain without the removed sample
                    nudge[:] = 0.0
                    pred_s = None
                    print("  undo — removed sample #%d (%d left), retrained"
                          % (ds.count() + 1, ds.count()))
            m = overlay.take_mouse()               # click/drag places the dot (fast, X+Y at once)
            if m is not None:
                nudge = np.clip(np.array(m) - pred_s, -1.0, 1.0)
            nudge = np.clip(nudge + np.array([act.dx, act.dy]), -1, 1)

            if act.approve:
                if simulate:
                    snap_features, snap_conf = features, 1.0
                else:
                    snap_features, snap_conf = snapshot_features(
                        world_L, world_R, eyeL_cam, eyeR_cam, detector,
                        trk_L, trk_R, last_features)
                if snap_conf < cfg.eye_conf_min:
                    print("  eye confidence %.2f < %.2f — hold steady, not stored"
                          % (snap_conf, cfg.eye_conf_min))
                else:
                    approved = np.clip(pred_s + nudge, 0.0, 1.0)
                    ds.add(snap_features, approved, weight=snap_conf)
                    cal = build_calibrator(cfg, ds)    # retrain on all samples
                    nudge[:] = 0.0
                    pred_s = None
                    print("  approved #%d  (conf %.2f, %s, lambda %.3f)" %
                          (ds.count(), snap_conf,
                           "trained" if cal.is_trained else "warming up",
                           cal.lambda_ or 0))
                    if simulate:
                        sim_dev = world.slip(sim_dev)
                        if world.rng.random() < 0.1:    # occasionally a new subject (re-seat)
                            sim_subject = world.new_subject(); sim_dev = world.seat()
                        sim_features, sim_truth, sim_i = _sim_next_ordered(
                            world, sim_subject, sim_dev, sim_pts, sim_i)
    finally:
        n_saved = ds.count()                 # read BEFORE closing the DB
        overlay.close()
        inp.close()
        ds.close()
        for c in cams:
            c.release()
    print("Done. %d samples stored in %s" % (n_saved, cfg.db_path))
    return 0


def _features_from(wlf, wrf, lf, rf, detector, trk_L, trk_R, last):
    """Assemble 8 features from the four camera frames. (features, conf, ok)."""
    if wlf is None or wrf is None or lf is None or rf is None:
        return last, 0.0, False
    dL, _ = detector.detect(wlf)
    dR, _ = detector.detect(wrf)
    if dL is None or dR is None:
        return last, 0.0, False
    lc, sl = trk_L.track(lf)
    rc, sr = trk_R.track(rf)
    if lc is None or rc is None:
        return last, 0.0, False
    feats = np.array([dL[0], dL[1], dR[0], dR[1], lc[0], lc[1], rc[0], rc[1]])
    return feats, float(min(sl, sr)), True


def read_real_features(world_L, world_R, eyeL_cam, eyeR_cam, detector,
                       trk_L, trk_R, last):
    """Live read of all four cameras -> (features, confidence, ok)."""
    return _features_from(world_L.read(), world_R.read(), eyeL_cam.read(),
                          eyeR_cam.read(), detector, trk_L, trk_R, last)


def snapshot_features(world_L, world_R, eyeL_cam, eyeR_cam, detector,
                      trk_L, trk_R, last):
    """High-speed capture at approval: freshest frames, no blur. (features, conf)."""
    feats, conf, ok = _features_from(
        world_L.snapshot(), world_R.snapshot(), eyeL_cam.snapshot(),
        eyeR_cam.snapshot(), detector, trk_L, trk_R, last)
    return (feats, conf) if ok else (last, 0.0)


def _sim_next(world, subject, dev):
    """Next interactive-sim scenario: (features, true pixel), retrying off-frame draws."""
    obs = None
    while obs is None:
        obs = world.observe(subject, dev, world.world_point())
    return obs[0], obs[2]      # features, geometric truth (the green target)


def _serpentine_points(nx=5, ny=4):
    """World dots laid out in a SNAKE order across the page so each practice target is right
    next to the previous one (short travel = faster calibration), instead of random jumps."""
    xs = np.linspace(-rig.DOT_X * 0.8, rig.DOT_X * 0.8, nx)
    ys = np.linspace(-rig.DOT_Y * 0.8, rig.DOT_Y * 0.8, ny)
    pts = []
    for j, y in enumerate(ys):
        for x in (xs if j % 2 == 0 else xs[::-1]):     # reverse every other row = serpentine
            pts.append(np.array([x, y, rig.DOT_Z_MEAN]))
    return pts


def _sim_next_ordered(world, subject, dev, pts, i):
    """Walk the serpentine list from index i; return (features, truth, next_i). Retries a point
    a few times (a random blink can spoil a capture) and skips points off-screen for this pose."""
    n = len(pts)
    for _ in range(n):
        for _try in range(6):
            obs = world.observe(subject, dev, pts[i % n])
            if obs is not None:
                return obs[0], obs[2], (i + 1) % n
        i += 1                                          # this point off-page for this pose -> next
    f, t = _sim_next(world, subject, dev)               # fallback: any on-screen point
    return f, t, i % n


def make_hud(n, cal, simulate, shown, green, conf) -> str:
    lines = ["samples: %d   model: %s   lambda: %.3f" %
             (n, "trained" if cal.is_trained else "warming up", cal.lambda_ or 0),
             "drag/click = move dot   wasd/WASD = nudge   ENTER = approve   U = undo   Z = reset   Q = quit"]
    if not simulate:
        lines.append("eye-corner confidence: %.2f" % conf)
    if simulate and green is not None:
        resid = float(np.linalg.norm(np.array(shown) - np.array(green)))
        lines.append("align RED onto GREEN  |  residual: %.1f px"
                     % (resid * 1080))
    return "\n".join(lines)


# ----------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulate", action="store_true",
                   help="interactive GUI sim, no hardware (align red onto green)")
    p.add_argument("--selftest", action="store_true",
                   help="headless convergence test (numpy only, no GUI/hardware)")
    p.add_argument("--list-cams", action="store_true")
    p.add_argument("--calibrate-corners", action="store_true")
    p.add_argument("--eye", choices=("L", "R", "l", "r"), default=None,
                   help="with --calibrate-corners: re-do ONE eye only (L or R), leaving the "
                        "other template untouched")
    p.add_argument("--world-cam-left", type=int)
    p.add_argument("--world-cam-right", type=int)
    p.add_argument("--eye-cam-left", type=int)
    p.add_argument("--eye-cam-right", type=int)
    p.add_argument("--fullscreen", action="store_true",
                   help="push the overlay onto your AR monitor fullscreen")
    p.add_argument("--db", type=str, help="override sample DB path")
    p.add_argument("--reset-db", action="store_true",
                   help="RESET fallback: wipe all stored calibration samples (start clean) "
                        "then exit. Use when the sample DB is full of misinputted data.")
    p.add_argument("--iterations", type=int, default=240,
                   help="self-test iterations")
    p.add_argument("--auto-train", type=int, metavar="N",
                   help="self-play train across N synthetic subjects (anatomy sim), "
                        "build the meta-database + prior, and benchmark warm vs cold")
    p.add_argument("--benchmark-users", type=int, default=24,
                   help="held-out new users for the warm-start benchmark")
    p.add_argument("--identify", action="store_true",
                   help="train + evaluate the few-shot facial-geometry identifier "
                        "(on the meta-database; build it first with --auto-train)")
    p.add_argument("--preset", action="store_true",
                   help="evaluate per-user eye presets: identify geometry from a few "
                        "answers, build a pose-invariant calibration, vs baselines")
    p.add_argument("--megarun", type=int, metavar="N",
                   help="large-scale run across N synthetic eyes (streaming prior + "
                        "poly identifier + warm-start); saves prior/identifier artifacts")
    p.add_argument("--pixel-map", action="store_true",
                   help="ground-truth display-pixel map: where the pixel must be for 100 dot "
                        "positions x each glasses position x every eye; saves data/pixel_map.npz")
    p.add_argument("--pixel-sweep", action="store_true",
                   help="pixel-by-pixel correction sweep: 100 faces x 100 glasses positions, "
                        "a guessing-AI guesses a point on each pixel's chief ray and a user-AI "
                        "corrects it until right twice in a row; counts guesses. Validates the "
                        "sim first. Add --every-pixel-full for native 1920x1080.")
    p.add_argument("--every-pixel-full", action="store_true",
                   help="with --pixel-sweep: sweep every native display pixel (1920x1080)")
    p.add_argument("--match", action="store_true",
                   help="identify eye geometry by matching a new user's (glasses position, "
                        "guess inaccuracy) fingerprint to a database of simulated faces; "
                        "includes the IMU/level and pupil/iris-sensor ablations")
    p.add_argument("--eyetracker", action="store_true",
                   help="quantify what a full NIR eye-tracker (measured gaze / pupil centre) "
                        "does for registration accuracy vs eye-corner cameras only")
    p.add_argument("--accuracy-map", action="store_true",
                   help="pixel accuracy for each glasses position on each face: population "
                        "prior vs geometry preset vs pupil-aided preset, error vs the oracle")
    p.add_argument("--kappa-hybrid", action="store_true",
                   help="DC-removed (bias-free) geometry ID + offset stage — the combination v2 missed")
    p.add_argument("--kappa-sdsweep", action="store_true",
                   help="correction-SD requirement sweep: what vernier precision buys under-3px")
    p.add_argument("--kappa-resid", action="store_true",
                   help="under-3px stage: per-user residual MODEL sweep (const/affine/quad, "
                        "MAD-gated) on vernier corrections, scored on PERCEIVED error")
    p.add_argument("--kappa-followup", action="store_true",
                   help="kappa_boost follow-up arms: per-user 2D offset stage + fair no-IR build")
    p.add_argument("--kappa-boost", action="store_true",
                   help="kappa-precision chain: honest label-based geometry ID at dot-nudge vs "
                        "VERNIER correction noise + averaging -> deployed stereo px (kappa_boost.py)")
    p.add_argument("--binocular-test", action="store_true",
                   help="validate the LEFT-eye oracle: coverage, true vertical disparity vs "
                        "fusion limits, vergence sanity (binocular.py)")
    p.add_argument("--vernier-test", action="store_true",
                   help="sim: overlay error vs #corrections for dot-nudge vs vernier-level "
                        "correction noise (the Phase-2 kappa lever, vernier.py)")
    p.add_argument("--vernier-demo", action="store_true",
                   help="pygame alignment practice + measure YOUR correction noise "
                        "(run on the XREAL display; writes data/vernier_noise.json)")
    p.add_argument("--sbs", action="store_true",
                   help="with --vernier-demo: side-by-side DICHOPTIC mode (One Pro in SBS 3D) "
                        "for inter-eye vertical-disparity alignment")
    p.add_argument("--kappa", action="store_true",
                   help="angle-kappa study (see KAPPA.md): convergence (perceived vs geometric "
                        "error vs corrections) + multi-vergence separation (a way around the "
                        "kappa/perceptual-bias confound)")
    p.add_argument("--kappa-live", action="store_true",
                   help="end-to-end: PCCR/objective kappa into the per-user preset, per-position "
                        "accuracy vs the oracle (with a true-geometry floor diagnostic)")
    p.add_argument("--calibrate", action="store_true",
                   help="production calibration: match a user to the nearest premade eye model "
                        "from their (glasses position, guess inaccuracy); builds/loads the DB "
                        "and demos on held-out users")
    p.add_argument("--use-pupil", action="store_true",
                   help="with --calibrate: build the per-user preset on the 10-feature "
                        "(incl. NIR pupil-centre) signal for sub-pixel accuracy")
    p.add_argument("--physics-preset", action="store_true",
                   help="white-box physics preset (estimate dot+pose -> forward-compute pixel): "
                        "polynomial vs physics, 8 vs 10 features; physics+pupil reaches sub-0.5 px")
    p.add_argument("--complete-geometry", action="store_true",
                   help="identify the COMPLETE 10-param geometry (+ eye_y, cfwd) -> physics preset; "
                        "best deployed systematic (~1.26 px)")
    p.add_argument("--realtime", action="store_true",
                   help="real-life deviation: per-user calibration (K corrections) x N-fixation "
                        "averaging, measured vs perceived alignment")
    p.add_argument("--stereo-test", action="store_true",
                   help="does a 2nd eye-corner camera (stereo) break under 1 px? mono vs stereo "
                        "complete-geometry physics preset")
    p.add_argument("--contract-test", action="store_true",
                   help="Phase-0 foundations: feature-order parity (live vector == "
                        "Config.feature_names) + device-calibration round-trip selftests")
    p.add_argument("--capture-test", action="store_true",
                   help="Phase-1 capture: blink-detector + live capture pipeline selftests "
                        "(--simulate mirror: blink discounting, stereo->mono fallback, both-cams-valid)")
    p.add_argument("--imu-test", action="store_true",
                   help="IMU drift filter: prove the 2-state Kalman (angle+gyro-bias) tilt filter "
                        "cancels gyro drift and smooths accel noise (software/imu.py)")
    p.add_argument("--estimator-test", action="store_true",
                   help="deployable physics+pupil estimator: same low median as raw physics, "
                        "with the degenerate-gaze tail cut by the corner-only fallback (estimator.py)")
    args = p.parse_args(argv)

    cfg = Config()
    if args.world_cam_left is not None:
        cfg.world_cam_left = args.world_cam_left
    if args.world_cam_right is not None:
        cfg.world_cam_right = args.world_cam_right
    if args.eye_cam_left is not None:
        cfg.eye_cam_left = args.eye_cam_left
    if args.eye_cam_right is not None:
        cfg.eye_cam_right = args.eye_cam_right
    if args.fullscreen:
        cfg.fullscreen = True
    if args.db:
        cfg.db_path = args.db

    import os
    meta_path = os.path.join(os.path.dirname(cfg.db_path), "meta.db")
    if args.megarun:
        import megarun
        megarun.run(n_subjects=args.megarun)
        return 0
    if args.pixel_map:
        import pixel_map
        pixel_map.build_map()
        return 0
    if args.pixel_sweep:
        import pixel_sweep
        if args.every_pixel_full:
            pixel_sweep.run_every_pixel()
        else:
            pixel_sweep.run_dense_grid()
        return 0
    if args.match:
        import match
        match.evaluate()
        return 0
    if args.eyetracker:
        import eyetracker
        eyetracker.compare()
        return 0
    if args.accuracy_map:
        import accuracy_map
        accuracy_map.evaluate()
        return 0
    if args.kappa:
        import kappa
        kappa.run_all()
        return 0
    if args.kappa_boost:
        import kappa_boost
        kappa_boost.evaluate()
        return 0
    if args.kappa_followup:
        import kappa_boost
        kappa_boost.evaluate(followup=True)
        return 0
    if args.kappa_resid:
        import kappa_boost
        kappa_boost.evaluate(followup="resid")
        return 0
    if args.kappa_sdsweep:
        import kappa_boost
        kappa_boost.evaluate(followup="sdsweep")
        return 0
    if args.kappa_hybrid:
        import kappa_boost
        kappa_boost.evaluate(followup="hybrid")
        return 0
    if args.binocular_test:
        import binocular
        return 0 if binocular.evaluate() else 1
    if args.vernier_test:
        import vernier
        vernier.vernier_test()
        return 0
    if args.vernier_demo:
        import vernier
        vernier.demo(sbs=args.sbs)
        return 0
    if args.calibrate:
        import calibrate
        calibrate.demo(use_pupil=args.use_pupil)
        return 0
    if args.physics_preset:
        import physics_preset
        physics_preset.evaluate()
        return 0
    if args.complete_geometry:
        import complete_geometry
        complete_geometry.evaluate()
        return 0
    if args.realtime:
        import realtime
        realtime.real_life_curve()
        return 0
    if args.stereo_test:
        import stereo_test
        stereo_test.evaluate()
        return 0
    if args.contract_test:
        import live_features
        import device
        rc_feat = live_features.selftest()   # run both so the user sees both reports
        rc_dev = device.selftest()
        return 1 if (rc_feat or rc_dev) else 0
    if args.capture_test:
        import blink
        import capture
        rc_blink = blink.selftest()
        print()
        rc_cap = capture.selftest()
        print()
        rc_fb = capture.validate_fallback()
        return 1 if (rc_blink or rc_cap or rc_fb) else 0
    if args.imu_test:
        import imu
        return imu.selftest()
    if args.estimator_test:
        import estimator
        return estimator.selftest()
    if args.auto_train:
        import autotrain
        prior, _, dbp = autotrain.auto_train(n_subjects=args.auto_train, seed=0)
        autotrain.benchmark(prior, n_test=args.benchmark_users)
        print("\nmeta-database written to %s" % dbp)
        if args.identify:
            print(); import identify; identify.evaluate(dbp)
        if args.preset:
            print(); import preset; preset.evaluate(dbp)
        return 0
    if args.identify or args.preset:
        if args.identify:
            import identify; identify.evaluate(meta_path)
        if args.preset:
            print(); import preset; preset.evaluate(meta_path)
        return 0
    if args.reset_db:
        ds = Dataset(cfg.db_path, cfg.feature_names)
        n = ds.clear()
        ds.close()
        print("reset-db: cleared %d stored samples from %s" % (n, cfg.db_path))
        return 0
    if args.selftest:
        return run_selftest(cfg, args.iterations)
    if args.list_cams:
        return run_list_cams()
    if args.calibrate_corners:
        return run_calibrate_corners(cfg, only=args.eye)
    return run_loop(cfg, simulate=args.simulate)


if __name__ == "__main__":
    sys.exit(main())
