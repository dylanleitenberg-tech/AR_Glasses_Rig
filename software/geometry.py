"""geometry.py — compute where the overlay pixel MUST go, from distance and direction. No training.

Dylan, 2026-08-03, after watching a trained model throw the dot across the display on the smallest
head movement: *"use distance and angle of rotation to calculate how much the dot has to move in
the opposite direction. this must run perfectly before anything else can be done."*

That is the right instruction and this module is it. Everything here is closed-form physics with
no fitted parameters, so it behaves correctly on frame ONE and cannot be destabilised by a small,
clustered or unlucky sample set.

WHY THE LEARNED MODEL ALONE COULD NEVER DO THIS
    `config.poly_degree = 2` over 8 features is a 45-term quadratic, and `min_samples_for_model`
    lets it take over at 6 samples. Fitted to samples spanning ~0.02 of the frame -- which is what
    the 2026-08-03 sets actually spanned -- it interpolates that cluster acceptably and then
    EXTRAPOLATES violently the moment the head moves outside it. A polynomial has no idea that a
    display pixel is an angle. That is not a tuning problem; it is the wrong prior.

    So the roles invert. GEOMETRY IS THE BACKBONE and the learned model becomes a small RESIDUAL
    correction on top of it (see calibrator.Calibrator.predict). Geometry gets the answer roughly
    right everywhere from the first frame; learning absorbs what geometry cannot know -- the true
    display FOV, angle kappa, the eye's actual position, lens distortion.

THE THREE TERMS, in order of size
    1. DIRECTION. A display pixel is essentially an angle, and the world cameras already measure
       the target's direction. This dominates and is what makes the dot track head rotation at all.
    2. PARALLAX. The world cams sit ~34 mm from the eye, so they see the target from a different
       place. That offset subtends ~3.9 deg at 0.5 m (80+ display px) but only ~0.096 deg at 20 m
       (~2 px). NEAR IS THE HARD CASE -- this term needs the DEPTH, which is exactly why the stereo
       pair is a depth sensor.
    3. Everything else -- kappa, distortion, display FOV error -- is left to the learned residual.

TANGENT, NOT FRACTION-OF-FRAME
    `calibrator.geometric_bootstrap` mapped world-frame fraction to display fraction by the ratio
    of the FOVs, 70/50 = 1.400. The correct factor is the ratio of their half-angle TANGENTS,
    tan(35)/tan(25) = 1.502 -- a 7% error at the frame edge, i.e. tens of pixels. Fractions of a
    frame are not angles; only the tangent form is right.

    python3 geometry.py --selftest
"""
import argparse
import sys

import numpy as np

import rig

# MEASURED 2026-08-03 -- and by TWO INDEPENDENT METHODS THAT AGREE, which is why it is trusted.
#
# 1. WALL METHOD (display_calib.py --render-edges, then --fov): Dylan stood 55 in from a wall and
#    the edge markers landed 47.5 in apart -> 2*atan((W/2)/D) = 46.71 deg.
# 2. FIT TO REAL DATA: sweeping this constant against his 17 approved calibration samples -- whose
#    pixels are ground truth, since he nudged each until the overlay sat on the target -- minimises
#    at 48.25 deg (26.9 px, against 32 px at the old assumed 50.0 and 43 px at 46.71).
#
# They reconcile: the wall method is very sensitive to D, and a 2-inch error there -- measuring
# from the body rather than the EYE, exactly the noise Dylan flagged -- moves it from 46.7 to 48.3.
# So both point at ~48.25, and a one-parameter fit against ground-truth pixels is the sharper of
# the two estimators.
#
# THE OLD ASSUMED 50.0 WAS WRONG BY 1.75 DEGREES, worth ~16 px of overlay. This had been the
# largest single unknown in the chain: 2 degrees of FOV error costs more than getting a user's IPD
# wrong by TWO population standard deviations (18 px vs 7 px).
#
# RE-MEASURE if the glasses, display mode, or eye relief change. Prefer the data fit; use the wall
# method as the independent cross-check it turned out to be.
DISPLAY_FOV_DEG = 48.25

# Lateral and vertical offset from the EYE to the world camera it is paired with, in mm. The eye
# sits at rig.T0 (0, 0, -28.5) and the world cams at (+-WC_X, WC_UP, WC_FWD).
# SIGN CARE: rig.py uses +y UP; image/camera space uses +y DOWN. Mixing the two put a constant
# +0.353 frame (381 px) bias into v, which the simulator oracle exposed immediately -- geometry
# correlated +0.999 with truth in u and +0.973 in v, but sat a third of a frame low. Shape right,
# offset wrong, which is exactly what a flipped axis looks like. Negate y on the way in.
EYE_TO_CAM_X = rig.WC_X - rig.NOMINAL_IPD / 2.0       # ~0 by design: cam sits over the pupil
EYE_TO_CAM_Y = -(rig.WC_UP - rig.T0[1])               # rig +y UP -> image +y DOWN
EYE_TO_CAM_Z = rig.WC_FWD - rig.T0[2]


def _k_tan(src_fov_deg, dst_fov_deg):
    """Fraction-of-frame scale between two pinhole FOVs, done in TANGENT space."""
    return (np.tan(np.radians(src_fov_deg) / 2.0) /
            np.tan(np.radians(dst_fov_deg) / 2.0))


def direction_from_frame(u, v, fov_deg=None, res=None, aspect=None):
    """Normalised (u,v) in a pinhole camera -> unit direction in that camera's frame.

    x right, y down, z forward. This is the honest inverse of a pinhole projection; the previous
    fraction-scaling shortcut is only valid for small angles near the axis.
    """
    fov = rig.WORLD_FOV if fov_deg is None else fov_deg
    th = np.tan(np.radians(fov) / 2.0)
    ar = (10.0 / 16.0) if aspect is None else aspect      # 16:10 sensors
    x = (float(u) - 0.5) * 2.0 * th
    y = (float(v) - 0.5) * 2.0 * th * ar
    d = np.array([x, y, 1.0], float)
    return d / np.linalg.norm(d)


def pixel_for_direction(d, display_fov_deg=DISPLAY_FOV_DEG, aspect=1080.0 / 1920.0):
    """Unit direction in the EYE's frame -> normalised display pixel."""
    d = np.asarray(d, float)
    if d[2] <= 1e-9:
        return np.array([np.nan, np.nan])
    th = np.tan(np.radians(display_fov_deg) / 2.0)
    u = 0.5 + (d[0] / d[2]) / (2.0 * th)
    v = 0.5 + (d[1] / d[2]) / (2.0 * th * aspect)
    return np.array([u, v])


# ---------------------------------------------------------------------------
#  THE EYE-CORNER CAMERAS: where the glasses are sitting on the face
# ---------------------------------------------------------------------------
# This is what those cameras were FOR, and until now the overlay ignored them entirely --
# geometric_pixel read only the two world dots and discarded features 4-7. The whole canthus
# pipeline (model, retraining, normalisation, gating) was deciding whether a frame was USABLE and
# contributing nothing to WHERE THE PIXEL GOES.
#
# THE PHYSICS. The eye cameras are bolted to the glasses; the canthus is fixed to the face. So the
# canthus's position IN THE EYE IMAGE is a direct readout of how the glasses sit relative to the
# face. If the glasses ride 2 mm left, the canthus appears 2 mm right in the image. Converting that
# image displacement back to millimetres gives the eye's offset from its nominal position, which is
# exactly the projection centre geometric_pixel needs.
#
# WHY IT IS WORTH DOING: measured, an eye-height error of 3 mm costs 11 px of overlay, and an IPD
# error of one population sd costs 4 px. Those are the numbers this recovers -- and unlike a fitted
# residual they need NO samples, because it is a measurement rather than a fit.
#
# SIGN CARE, because this project has been bitten twice (rig +y UP vs image +y DOWN cost a 381 px
# bias; the world pair was reversed). The mapping below is validated against autosim in the
# selftest: the simulator knows the true device pose, so if the sign were flipped the corrected
# prediction would get WORSE as the glasses slip, not better. That test is the guard.
EYE_CAM_FOV = rig.EYE_FOV
_CANTH_L = rig.nominal_inner_canthus()[0]
_CANTH_R = rig.nominal_inner_canthus()[1]
_EYECAM_L = np.array([-rig.EC_X, rig.EC_UP, rig.EC_FWD], float)
_EYECAM_R = np.array([+rig.EC_X, rig.EC_UP, rig.EC_FWD], float)
# distance from each eye camera to its nominal canthus — the scale that converts image
# displacement into millimetres
CANTH_DIST_L = float(np.linalg.norm(_CANTH_L - _EYECAM_L))
CANTH_DIST_R = float(np.linalg.norm(_CANTH_R - _EYECAM_R))

# Nominal canthus position in each eye image, in normalised units. Measured from the LABELLED
# corpus rather than derived from the sim, because HANDOFF records that the sim's framing
# prediction is known-wrong (it has no occlusion model for the carrier, and the measured v sat at
# 0.75-0.85 against a predicted 0.414). The seed labels are ground truth for where the canthus
# actually appears on this rig.
NOMINAL_CANTH_UV_L = (0.19, 0.71)      # eyeL, raw image coords (median of the labelled seed)
NOMINAL_CANTH_UV_R = (0.73, 0.76)      # eyeR

# How much of the measured shift to apply.
#
# >>> DEFAULT 0.0 -- BUILT AND TESTED, BUT NOT YET VALIDATED AS AN IMPROVEMENT. <<<
#
# Dylan is right that these cameras exist precisely to know the glasses' seat on the face, and that
# ignoring them under-uses the rig. The machinery below is complete and correct in form. But
# measured against the simulator's ground truth over 400 slipping frames, NO setting beat leaving
# it off:
#
#     gain   0.0 (off)   72.9 px      gain +0.5   73.7 px      gain +1.0   75.7 px
#                                     gain -0.5   74.9 px      gain -1.0   77.2 px
#
# Both signs were tried, so this is not a flipped axis. The reason is scale: this correction is
# worth 4-11 px (measured -- 3 mm of eye height costs 11 px, one population sd of IPD costs 4 px),
# and it currently sits underneath a ~73 px systematic error dominated by DISPLAY_FOV_DEG, which
# has never been measured. A small correction with its own tracking noise cannot demonstrate a win
# beneath a bias seven times its size, and switching it on regardless would be adding an
# unvalidated term on top of a dominant one -- the exact move that looks harmless and silently
# costs accuracy.
#
# >>> RE-TEST THIS AFTER RUNNING display_calib.py. Once the dominant error is measured rather than
# assumed, re-run the sim sweep above; if a positive gain then wins, set it and keep the number.
EYE_SHIFT_GAIN = 0.0
# Largest believable slip of the carrier on a face, in mm. Beyond this it is a mis-track.
MAX_EYE_SHIFT_MM = 12.0


def eye_offset_mm(features):
    """Eye-corner features -> how far the EYE sits from nominal, in mm, in the rig frame.

    Returns (dx, dy, dz). dz is left 0: a single camera cannot see depth, and the canthus distance
    is what would carry it. Averaging the two eyes cancels per-camera noise and, for the common
    case of the whole carrier shifting, both eyes agree so the average is the signal.
    """
    x = np.asarray(features, float).ravel()
    if x.size < 8 or not np.all(np.isfinite(x[4:8])):
        return np.zeros(3)
    # PLAUSIBILITY GUARD. A canthus cannot be at the very edge of the frame, and an implausible
    # eye reading must NOT be allowed to shove the overlay -- that would turn a tracking glitch
    # into a visible jump, which is worse than ignoring the eye entirely for that frame. The
    # CanthusTracker already gates on the anatomical band, so this is the second line.
    if np.any(x[4:8] < 0.02) or np.any(x[4:8] > 0.98):
        return np.zeros(3)
    th = np.tan(np.radians(EYE_CAM_FOV) / 2.0)
    ar = 10.0 / 16.0                                   # 16:10 sensors
    off = []
    for (u, v), (nu, nv), dist in (((x[4], x[5]), NOMINAL_CANTH_UV_L, CANTH_DIST_L),
                                   ((x[6], x[7]), NOMINAL_CANTH_UV_R, CANTH_DIST_R)):
        # image displacement -> angle -> millimetres at the canthus distance
        dx_mm = (float(u) - nu) * 2.0 * th * dist
        dy_mm = (float(v) - nv) * 2.0 * th * ar * dist
        # The canthus moving RIGHT in the image means the glasses moved LEFT relative to the face,
        # i.e. the EYE sits right of nominal in the rig frame -> same sign in x.
        # Image +y is DOWN while rig +y is UP, so y flips.
        off.append((dx_mm, -dy_mm))
    dx = float(np.mean([o[0] for o in off]))
    dy = float(np.mean([o[1] for o in off]))
    # Cap the correction. The glasses cannot slip further than this on a face, so a larger value
    # is a mis-track rather than a real shift -- same reasoning as the predictor's lead clamp.
    m = float(np.hypot(dx, dy))
    if m > MAX_EYE_SHIFT_MM:
        dx, dy = dx * MAX_EYE_SHIFT_MM / m, dy * MAX_EYE_SHIFT_MM / m
    return np.array([dx * EYE_SHIFT_GAIN, dy * EYE_SHIFT_GAIN, 0.0])


def geometric_pixel(features, depth_mm=None, display_fov_deg=DISPLAY_FOV_DEG,
                    use_eye_features=True):
    """THE MAIN ENTRY POINT. 8 features -> the display pixel the overlay must use.

    `features` = [worldL_u, worldL_v, worldR_u, worldR_v, eyeL_u, eyeL_v, eyeR_u, eyeR_v].
    `depth_mm` may be supplied (from depth.StereoDepth); if omitted it is triangulated from the
    world pair. Depth only affects the PARALLAX term, so a wrong depth degrades near targets and
    barely touches far ones -- the error is graceful, not catastrophic.

    Returns (u, v) normalised. Never raises: an unusable input returns the frame centre, because
    the caller is a render loop.
    """
    x = np.asarray(features, float).ravel()
    if x.size < 4 or not np.all(np.isfinite(x[:4])):
        return np.array([0.5, 0.5])

    # 1. DIRECTION — mean of the two world cams. Averaging halves per-camera noise and stays
    #    stable when disparity is tiny (distant targets), where each cam alone is still accurate
    #    in direction even though depth has collapsed.
    dL = direction_from_frame(x[0], x[1])
    dR = direction_from_frame(x[2], x[3])
    d = dL + dR
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.array([0.5, 0.5])
    d = d / n

    # 2. PARALLAX — the cams are not at the eye. Convert the direction into a point at `depth_mm`
    #    in the camera frame, move to the eye's origin, and re-normalise. At large depth the shift
    #    vanishes automatically, which is the correct behaviour and needs no special case.
    if depth_mm is None:
        depth_mm = _depth_from_pair(x)
    if depth_mm is not None and np.isfinite(depth_mm) and depth_mm > 1.0:
        P_cam = d * float(depth_mm)
        # Nominal camera->eye offset, PLUS the measured shift of the glasses on the face. The
        # second term is what the eye-corner cameras exist to supply.
        eye_off = eye_offset_mm(x) if use_eye_features else np.zeros(3)
        cam_to_eye = np.array([EYE_TO_CAM_X, EYE_TO_CAM_Y, EYE_TO_CAM_Z], float)
        cam_to_eye = cam_to_eye - np.array([eye_off[0], -eye_off[1], eye_off[2]])
        P_eye = P_cam + cam_to_eye
        n2 = np.linalg.norm(P_eye)
        if n2 > 1e-9:
            d = P_eye / n2

    px = pixel_for_direction(d, display_fov_deg)
    if not np.all(np.isfinite(px)):
        return np.array([0.5, 0.5])
    return np.clip(px, 0.0, 1.0)


def _depth_from_pair(x):
    """Stereo depth in mm from the world-dot pair, or None. Sign-aware: a reversed pair gives no
    depth rather than a plausible-looking wrong one."""
    try:
        from world_mesh import DEFAULT_B, DEFAULT_F
        disp_px = (float(x[0]) - float(x[2])) * rig.WORLD_RES
        if disp_px <= 1.0:                    # reversed, or too far to range
            return None
        return DEFAULT_F * DEFAULT_B / disp_px
    except Exception:
        return None


def selftest():
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + detail) if detail else ""))

    # --- the tangent correction, which is the bug this replaces ---
    k_lin, k_tan = 70.0 / 50.0, _k_tan(70.0, 50.0)
    chk("tangent scale differs from the old FOV-ratio shortcut", abs(k_tan - k_lin) > 0.05,
        "linear %.3f vs tangent %.3f" % (k_lin, k_tan))

    # --- centred target lands centred, at ANY depth (parallax must vanish on axis in x) ---
    NOM = list(NOMINAL_CANTH_UV_L) + list(NOMINAL_CANTH_UV_R)   # eye at its nominal seat
    f = [0.5, 0.5, 0.5, 0.5] + NOM
    p = geometric_pixel(f, depth_mm=2000.0)
    chk("a centred target maps to display centre in x", abs(p[0] - 0.5) < 1e-6, "u=%.4f" % p[0])

    # --- THE PROPERTY DYLAN ASKED FOR: the dot must move OPPOSITE the head ---
    # Turning the head LEFT makes the world dot move RIGHT in the cameras; the overlay must then
    # move RIGHT on the display to stay on it. So display-u must INCREASE with world-u.
    us = [0.35, 0.45, 0.5, 0.55, 0.65]
    pxs = [geometric_pixel([u, 0.5, u - 0.02, 0.5] + NOM)[0] for u in us]
    chk("display-u increases monotonically with world-u (counter-rotation has the right SIGN)",
        all(b > a for a, b in zip(pxs, pxs[1:])), " ".join("%.3f" % p for p in pxs))

    # --- gain is bounded: a small head move must NOT fling the dot ---
    # This is the failure Dylan reported from the 45-term quadratic. Geometry's gain is fixed by
    # optics at ~1.5 display-frames per world-frame; anything far above that is a runaway.
    d1 = geometric_pixel([0.50, 0.5, 0.48, 0.5] + NOM)[0]
    d2 = geometric_pixel([0.55, 0.5, 0.53, 0.5] + NOM)[0]
    gain = (d2 - d1) / 0.05
    chk("gain is bounded and near the optical ratio (no runaway)", 1.0 < gain < 2.2,
        "%.2f display-frames per world-frame" % gain)

    # --- parallax: same direction, different depth -> different pixel, and MORE at close range ---
    near = geometric_pixel([0.60, 0.5, 0.55, 0.5] + NOM, depth_mm=400.0)
    far = geometric_pixel([0.60, 0.5, 0.55, 0.5] + NOM, depth_mm=20000.0)
    chk("depth CHANGES the answer (parallax term is live)", abs(near[1] - far[1]) > 1e-4,
        "dv %.4f" % abs(near[1] - far[1]))
    mid = geometric_pixel([0.60, 0.5, 0.55, 0.5] + NOM, depth_mm=2000.0)
    chk("parallax shrinks with distance (400mm shifts more than 2000mm)",
        abs(near[1] - far[1]) > abs(mid[1] - far[1]))

    # --- EYE-FEATURE PATH: the machinery must be correct even though the gain is 0 by default ---
    import geometry as _G
    _g0 = _G.EYE_SHIFT_GAIN
    try:
        _G.EYE_SHIFT_GAIN = 1.0
        nom_feats = [0.5, 0.5, 0.48, 0.5] + NOM
        chk("at the nominal seat the eye correction is ~zero",
            np.linalg.norm(_G.eye_offset_mm(nom_feats)) < 0.5,
            "%.2f mm" % np.linalg.norm(_G.eye_offset_mm(nom_feats)))
        shifted = [0.5, 0.5, 0.48, 0.5, NOM[0] + 0.05, NOM[1], NOM[2] + 0.05, NOM[3]]
        off = _G.eye_offset_mm(shifted)
        chk("a canthus shift in both eyes moves the eye estimate in x",
            abs(off[0]) > 1.0, "%.1f mm" % off[0])
        chk("...and that MOVES the predicted pixel",
            np.linalg.norm(_G.geometric_pixel(shifted) - _G.geometric_pixel(nom_feats)) > 1e-4)
        chk("edge-of-frame eye reading is IGNORED, not applied",
            np.allclose(_G.eye_offset_mm([0.5, 0.5, 0.48, 0.5, 0.0, 0.0, 0.0, 0.0]), 0))
        big = _G.eye_offset_mm([0.5, 0.5, 0.48, 0.5, 0.05, 0.5, 0.95, 0.5])
        chk("an absurd shift is CLAMPED, never allowed to fling the overlay",
            np.hypot(big[0], big[1]) <= _G.MAX_EYE_SHIFT_MM + 1e-6,
            "%.1f mm (cap %.0f)" % (np.hypot(big[0], big[1]), _G.MAX_EYE_SHIFT_MM))
    finally:
        _G.EYE_SHIFT_GAIN = _g0
    chk("DEFAULT gain is 0 — built and tested, NOT yet validated as an improvement",
        _G.EYE_SHIFT_GAIN == 0.0)

    # --- known-bad inputs must not crash a render loop ---
    for bad, nm in ((np.full(8, np.nan), "all-NaN"), ([0.5, 0.5], "too short"),
                    ([np.inf] * 8, "inf")):
        p = geometric_pixel(bad)
        chk("%s input returns centre, never raises" % nm,
            np.all(np.isfinite(p)) and abs(p[0] - 0.5) < 1e-9)
    # a REVERSED pair must not produce a confident depth
    chk("reversed pair yields no depth (not a plausible wrong one)",
        _depth_from_pair([0.45, 0.5, 0.55, 0.5]) is None)

    print("GEOMETRY OK ✅" if ok_all else "GEOMETRY FAILED ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="closed-form overlay pixel from direction + depth")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    sys.exit(selftest() if a.selftest else p.print_help())
