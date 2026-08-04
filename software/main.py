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
        if getattr(cfg, "use_model", False):
            # LEARNED landmark instead of a hand-drawn template. The template had to be
            # re-captured every session and localised whatever matched rather than a canthus;
            # the model is trained once from human seed points and carries its own plausibility
            # gates (anatomical prior + jump test). eyeL is the MIRRORED camera.
            from canthus_net import CanthusTracker
            trk_L = CanthusTracker(mirrored=True)
            trk_R = CanthusTracker(mirrored=False)
            print("eye tracking: LEARNED canthus model (data/canthus_net.npz)")
        else:
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
                      cfg.fullscreen, cfg.dot_radius_px,
                      win_x=getattr(cfg, "overlay_x", None))
    inp = InputController(cfg.nudge_gain, cfg.deadzone, cfg.approve_button,
                          cfg.quit_button)

    nudge = np.zeros(2)
    last_features = np.full(cfg.n_features, 0.5)
    smooth_feat = None             # EMA of live eye features (real mode)
    _last_key = [None]             # most recent raw cv2 keycode, surfaced on the HUD
    pred_s = None                  # EMA of the predicted pixel (reduces jitter)
    conf = 1.0
    n_saved = 0
    print("Loop running. %s. Q/ESC quits."
          % ("joystick + keyboard" if inp.has_joystick else "keyboard (no pad)"))

    try:
        while True:
            green = None
            live = True                # are this frame's features real? (gates APPROVE only)
            why = ""
            if simulate:
                features, conf = sim_features, 1.0
                green = tuple(sim_truth)
            else:
                features, conf, ok, why = read_real_features(
                    world_L, world_R, eyeL_cam, eyeR_cam, detector, trk_L, trk_R,
                    last_features)
                if not ok:
                    live = False
                    features = last_features
                else:
                    # EMA the EYE features only. The world dot drives DIRECTION and must pass
                    # through unsmoothed -- filtering it is what made the overlay lag behind the
                    # head. See config.feat_smooth_world for the measured lag budget.
                    if smooth_feat is None:
                        smooth_feat = features.copy()
                    else:
                        aw, ae = cfg.feat_smooth_world, cfg.feat_smooth
                        smooth_feat[:4] = aw * features[:4] + (1 - aw) * smooth_feat[:4]
                        smooth_feat[4:] = ae * features[4:] + (1 - ae) * smooth_feat[4:]
                    features = smooth_feat.copy()
                    last_features = features

            predicted = cal.predict(features)
            pred_s = predicted if pred_s is None else (
                cfg.pred_smooth * predicted + (1 - cfg.pred_smooth) * pred_s)
            shown = np.clip(pred_s + nudge, 0.0, 1.0)

            hud = make_hud(ds.count(), cal, simulate, shown, green, conf)
            # SHOW THE RAW KEY CODE. "WASD isn't working" has several possible causes -- the
            # window not holding focus, a platform keycode difference, another process eating the
            # event -- and they are indistinguishable from the outside. Displaying what
            # cv2.waitKey actually returned separates "no key arrived" from "key arrived and was
            # ignored" in one glance, which is the difference between a focus problem and a code
            # problem.
            hud += ("\nkeys: WASD nudge (shift=coarse) | ENTER approve | Z cancel | U undo | Q quit"
                    "\nlast key: %s" % ("none yet" if _last_key[0] is None
                                        else "%d (%s)" % (_last_key[0],
                                                          chr(_last_key[0])
                                                          if 32 <= _last_key[0] < 127 else "?")))
            if not live:
                # Approve is NOT hard-blocked here: it re-reads the cameras via snapshot_features,
                # so it can still succeed if the frame recovers in between. It prints its own
                # reason when it doesn't.
                hud += "\nNOT LIVE -- %s  (nudge/undo/quit still work)" % why
            key = overlay.render(tuple(shown), green, hud)
            if key != -1 and key != 255:
                _last_key[0] = key
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
                snap_why = ""
                if simulate:
                    snap_features, snap_conf = features, 1.0
                else:
                    snap_features, snap_conf, snap_why = snapshot_features(
                        world_L, world_R, eyeL_cam, eyeR_cam, detector,
                        trk_L, trk_R, last_features)
                if snap_why:
                    # Say WHICH input is missing. A silent no-op here is what made the loop look
                    # like it was ignoring ENTER.
                    print("  not stored — %s" % snap_why)
                elif snap_conf < cfg.eye_conf_min:
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


# Largest plausible frame-to-frame move of the world dot, in normalised units.
#
# A dot on a page is a physical object: it cannot teleport, and neither can your head within one
# frame. Measured on this rig, dot_detector reported 0.136, then 0.202, then 0.417 for worldR on a
# STATIONARY target -- it was jumping between the dot, dark furniture and frame-edge artefacts. Once
# the prediction started following the world cameras (geometric_bootstrap) that instability became
# visible immediately: Dylan's "dot is flying across screen".
#
# Same reasoning as the canthus jump gate: physics constrains what a real detection can do between
# frames, and physics does not depend on how confident a detector claims to be. A jump beyond this
# is a mis-detection by construction, so hold the previous value rather than propagate it.
DOT_MAX_JUMP = 0.12
_dot_prev = {"L": None, "R": None}


# Frames of SUSTAINED disagreement after which a "jump" is accepted as real motion.
#
# THE GATE USED TO HAVE NO ESCAPE, AND IT SILENTLY RUINED A 35-SAMPLE CALIBRATION.
# On a jump it returned `prev` WITHOUT updating `_dot_prev`, so the next frame compared against
# the same stale position, exceeded the threshold again, and rejected again -- permanently. Move
# the target further than DOT_MAX_JUMP once and the feature froze for the rest of the session.
# Measured consequence (2026-08-03, 35 approved samples): every one of the 8 features spanned
# 0.013-0.052 of frame, the dot barely moving, so the fit had almost no information and behaved
# like a constant -- which on the glasses reads as "the overlay moves WITH my head".
#
# The canthus tracker already solved this exact problem with MAX_HELD: a one-frame flick is a
# mis-detection, but a position that PERSISTS is the world having genuinely moved. Physics
# constrains how fast the dot can move between frames; it does not forbid it from being somewhere
# new a moment later. This gate now gets the same escape, and deliberately reuses the same
# reasoning rather than inventing a second policy.
DOT_MAX_HELD = 8                 # ~0.6 s at the measured 13 fps
_dot_held = {"L": 0, "R": 0}


def _gate_dot(d, side):
    """Reject a world-dot detection that moved impossibly far since the last accepted one.

    A SUSTAINED new position is accepted after DOT_MAX_HELD frames -- moving the calibration
    target IS the point of the exercise, so a gate with no escape defeats the whole session.
    """
    prev = _dot_prev[side]
    if d is None:
        return prev
    if prev is not None:
        if float(np.hypot(d[0] - prev[0], d[1] - prev[1])) > DOT_MAX_JUMP:
            _dot_held[side] += 1
            if _dot_held[side] <= DOT_MAX_HELD:
                return prev                  # transient -> a mis-detection, hold the last good
            # held long enough: the target really did move. Accept and resync.
    _dot_held[side] = 0
    _dot_prev[side] = (float(d[0]), float(d[1]))
    return _dot_prev[side]


def _features_from(wlf, wrf, lf, rf, detector, trk_L, trk_R, last):
    """Assemble 8 features from the four camera frames. (features, conf, ok, why).

    `why` NAMES EVERY FAILING INPUT, and that is the point of it. This used to return a bare
    False, and the loop rendered one string -- "searching for dot / eye corners..." -- for five
    genuinely different faults: a dead camera, either world dot missing, either eye landmark
    missing. On 2026-08-03 the rig sat in this state for a whole session and the message was read
    as "the dot is hard to find", when `eyeR`'s plausibility gate was rejecting every frame (see
    HANDOFF open item 2). One summary string covering several distinct failures is the same
    lumping mistake as a pass rate covering several distinct inputs.
    """
    missing = [n for n, f in (("worldL", wlf), ("worldR", wrf),
                              ("eyeL", lf), ("eyeR", rf)) if f is None]
    if missing:
        return last, 0.0, False, "no frame from: " + ", ".join(missing)
    # JOINT stereo pick, not one argmax per camera.
    #
    # This was still the OLD independent path long after calib_preflight had been switched over,
    # which meant preflight could report a clean stereo pair while the LIVE loop was locking each
    # camera onto its own dark object. The single-camera score is circularity*sqrt(area)*contrast,
    # so it rewards AREA, and a real room always contains a bigger dark round thing than a dot on
    # a page -- measured on this rig, the true dot ranked 2nd in worldL and 3rd in worldR. When the
    # two cams lock onto different furniture the dot feature stops tracking the target altogether,
    # and the overlay then appears glued to the head, which is exactly what Dylan reported:
    # "it is following my head movements... it should be going the opposite direction."
    dL = dR = None
    try:
        pair = detector.detect_pair(wlf, wrf)
        if pair[0] is not None:
            dL, dR = pair
    except Exception:
        pass
    if dL is None:
        dL, _ = detector.detect(wlf)
        dR, _ = detector.detect(wrf)
    dL = _gate_dot(dL, "L")
    dR = _gate_dot(dR, "R")
    nodot = [n for n, d in (("worldL", dL), ("worldR", dR)) if d is None]
    if nodot:
        return last, 0.0, False, "no world dot in: " + ", ".join(nodot)
    lc, sl = trk_L.track(lf)
    rc, sr = trk_R.track(rf)
    noeye = [n for n, c in (("eyeL", lc), ("eyeR", rc)) if c is None]
    if noeye:
        return last, 0.0, False, "no eye landmark in: " + ", ".join(noeye)
    feats = np.array([dL[0], dL[1], dR[0], dR[1], lc[0], lc[1], rc[0], rc[1]])
    return feats, float(min(sl, sr)), True, ""


def selftest_input() -> int:
    """Pin the two failures that cost the 2026-08-03 session.

    BOTH are written as the failing case first, per the project rule: a check that goes green for
    a reason unrelated to what it claims to measure is worse than no check.

    1. A not-live frame must still process nudge/quit keys. The loop used to `continue` out of the
       frame when any input was missing, polling the InputController only for `.quit` -- so every
       WASD press and every ENTER was consumed and dropped. With eyeR's plausibility gate rejecting
       every frame (HANDOFF open item 2) the rig sat in that branch permanently, which is why the
       symptom read as "WASD isn't working" and "no samples stored" at the same time. The raw
       keycode HUD added to diagnose it was itself only in the live branch, so it showed "none yet"
       forever and pointed at a focus problem that did not exist.
    2. The not-live reason must NAME the failing input. One string for five faults sent a whole
       session hunting the world dot while the actual failure was eyeR.
    """
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               ("  — " + detail) if detail else ""))

    frame = np.zeros((8, 8), dtype=np.uint8)

    class _Det:
        def __init__(self, hit=True):
            self.hit = hit

        def detect(self, f):
            return ((0.5, 0.5), 1.0) if self.hit else (None, 0.0)

    class _Trk:
        def __init__(self, hit=True):
            self.hit = hit

        def track(self, f):
            return ((0.5, 0.5), 0.9) if self.hit else (None, 0.0)

    last = np.full(8, 0.5)

    # -- 2. the reason names the failing input, and ONLY the failing input --
    _dot_prev["L"] = _dot_prev["R"] = None
    _, _, ok, why = _features_from(frame, frame, frame, frame,
                                   _Det(), _Trk(), _Trk(False), last)
    check("eyeR-only failure is named", (not ok) and "eyeR" in why, why)
    check("eyeR-only failure does NOT blame eyeL", "eyeL" not in why, why)
    check("eyeR-only failure does NOT blame the world dot",
          "world dot" not in why, why)

    _dot_prev["L"] = _dot_prev["R"] = None
    _, _, ok, why = _features_from(frame, None, frame, frame,
                                   _Det(), _Trk(), _Trk(), last)
    check("dead camera is named", (not ok) and "worldR" in why, why)

    _dot_prev["L"] = _dot_prev["R"] = None
    _, _, ok, why = _features_from(frame, frame, frame, frame,
                                   _Det(False), _Trk(), _Trk(), last)
    check("missing world dot is named", (not ok) and "world dot" in why, why)

    _dot_prev["L"] = _dot_prev["R"] = None
    feats, conf, ok, why = _features_from(frame, frame, frame, frame,
                                          _Det(), _Trk(), _Trk(), last)
    check("all-good frame is live with no reason", ok and why == "" and conf > 0,
          "conf %.2f" % conf)
    check("all-good frame yields 8 features", ok and feats.shape == (8,))

    # -- 3. THE DOT GATE MUST LET THE TARGET ACTUALLY MOVE --
    # Written as the failing case first: before DOT_MAX_HELD existed, a jump returned `prev`
    # WITHOUT updating it, so the very next frame compared against the same stale position and
    # rejected again, forever. Moving the calibration target -- the entire point of the exercise --
    # froze the feature for the rest of the session and produced a 35-sample set whose features
    # spanned 0.013-0.052 of frame, i.e. almost no information.
    _dot_prev["L"] = None
    _dot_held["L"] = 0
    check("gate accepts the first detection", _gate_dot((0.20, 0.50), "L") == (0.20, 0.50))
    check("gate accepts a small move", _gate_dot((0.24, 0.50), "L") == (0.24, 0.50))
    held = [_gate_dot((0.80, 0.50), "L") for _ in range(DOT_MAX_HELD)]
    check("a big jump is HELD at first (a one-frame flick is a mis-detection)",
          all(h == (0.24, 0.50) for h in held), "held %d frames" % len(held))
    after = _gate_dot((0.80, 0.50), "L")
    check("but a SUSTAINED new position is finally ACCEPTED (the target really moved)",
          after == (0.80, 0.50), "got %s" % (after,))
    check("and the gate resyncs there rather than re-freezing",
          _gate_dot((0.82, 0.50), "L") == (0.82, 0.50))
    _dot_prev["L"] = None
    _dot_held["L"] = 0

    # -- 1. keys survive when nothing is being tracked --
    # The InputController is stateless w.r.t. liveness, so the property to pin is that the loop's
    # key handling is reachable at all: poll() must map every documented key, and must ignore the
    # two "no key" sentinels rather than treating them as a keypress.
    from input_ctl import InputController
    inp = InputController(use_joystick=False)
    check("'w' nudges up", inp.poll(ord('w')).dy < 0)
    check("'s' nudges down", inp.poll(ord('s')).dy > 0)
    check("'a' nudges left", inp.poll(ord('a')).dx < 0)
    check("'d' nudges right", inp.poll(ord('d')).dx > 0)
    check("shift = coarse", abs(inp.poll(ord('W')).dy) > abs(inp.poll(ord('w')).dy))
    check("ENTER approves (13 and 10)",
          inp.poll(13).approve and inp.poll(10).approve)
    check("'u' undoes", inp.poll(ord('u')).undo)
    check("'z' resets", inp.poll(ord('z')).reset)
    check("'q'/ESC quit", inp.poll(ord('q')).quit and inp.poll(27).quit)
    # 255 is what `cv2.waitKey(1) & 0xFF` returns for "no key" (-1 & 0xFF). Treating it as a
    # keypress would make the dot drift on its own every frame.
    #
    # HONESTY NOTE: this pair is currently satisfied for a reason other than the guard it names --
    # chr(255) is 'ÿ' and chr(-1) is not a letter, so neither would hit the WASD branch even with
    # the `key != 255` guard deleted. It is a regression pin (it would catch someone mapping a
    # default/fallthrough action), NOT evidence the guard works. Do not read it as the latter.
    for sentinel in (-1, 255):
        a = inp.poll(sentinel)
        check("sentinel %d is not a keypress" % sentinel,
              a.dx == 0 and a.dy == 0 and not a.approve and not a.quit)
    inp.close()

    print("INPUT PLUMBING OK ✅" if ok_all else "INPUT PLUMBING FAILED ❌")
    return 0 if ok_all else 1


def read_real_features(world_L, world_R, eyeL_cam, eyeR_cam, detector,
                       trk_L, trk_R, last):
    """Live read of all four cameras -> (features, confidence, ok, why)."""
    return _features_from(world_L.read(), world_R.read(), eyeL_cam.read(),
                          eyeR_cam.read(), detector, trk_L, trk_R, last)


def snapshot_features(world_L, world_R, eyeL_cam, eyeR_cam, detector,
                      trk_L, trk_R, last):
    """High-speed capture at approval: freshest frames, no blur. (features, conf, why)."""
    feats, conf, ok, why = _features_from(
        world_L.snapshot(), world_R.snapshot(), eyeL_cam.snapshot(),
        eyeR_cam.snapshot(), detector, trk_L, trk_R, last)
    return (feats, conf, "") if ok else (last, 0.0, why)


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
    p.add_argument("--use-model", action="store_true",
                   help="track the canthus with the trained model instead of hand-drawn "
                        "templates (needs data/canthus_net.npz)")
    p.add_argument("--eye", choices=("L", "R", "l", "r"), default=None,
                   help="with --calibrate-corners: re-do ONE eye only (L or R), leaving the "
                        "other template untouched")
    p.add_argument("--world-cam-left", type=int)
    p.add_argument("--world-cam-right", type=int)
    p.add_argument("--eye-cam-left", type=int)
    p.add_argument("--eye-cam-right", type=int)
    p.add_argument("--fullscreen", action="store_true",
                   help="push the overlay onto your AR monitor fullscreen")
    p.add_argument("--overlay-x", type=int, default=None,
                   help="desktop x-coordinate of the AR display's left edge; the overlay window "
                        "is moved there BEFORE going fullscreen (without it, fullscreen fills the "
                        "primary monitor and nothing reaches the glasses)")
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
    p.add_argument("--input-test", action="store_true",
                   help="calibration-loop input plumbing: keys reach the loop when nothing is "
                        "being tracked, and the not-live reason names the failing camera")
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
    if args.input_test:
        return selftest_input()
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
    cfg.use_model = bool(getattr(args, "use_model", False))
    cfg.overlay_x = getattr(args, "overlay_x", None)
    if args.calibrate_corners:
        return run_calibrate_corners(cfg, only=args.eye)
    return run_loop(cfg, simulate=args.simulate)


if __name__ == "__main__":
    sys.exit(main())
