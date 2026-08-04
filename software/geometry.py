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

# Display half-angle. NOT MEASURED -- `display_calib.py` exists to measure it and has never been
# run, so this is the single largest known-unknown in the chain. The learned residual absorbs a
# constant scale error here, which is one more reason geometry must be the backbone and not the
# whole answer.
DISPLAY_FOV_DEG = 50.0

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


def geometric_pixel(features, depth_mm=None, display_fov_deg=DISPLAY_FOV_DEG):
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
        P_eye = P_cam + np.array([EYE_TO_CAM_X, EYE_TO_CAM_Y, EYE_TO_CAM_Z], float)
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
    f = [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]
    p = geometric_pixel(f, depth_mm=2000.0)
    chk("a centred target maps to display centre in x", abs(p[0] - 0.5) < 1e-6, "u=%.4f" % p[0])

    # --- THE PROPERTY DYLAN ASKED FOR: the dot must move OPPOSITE the head ---
    # Turning the head LEFT makes the world dot move RIGHT in the cameras; the overlay must then
    # move RIGHT on the display to stay on it. So display-u must INCREASE with world-u.
    us = [0.35, 0.45, 0.5, 0.55, 0.65]
    pxs = [geometric_pixel([u, 0.5, u - 0.02, 0.5, 0, 0, 0, 0])[0] for u in us]
    chk("display-u increases monotonically with world-u (counter-rotation has the right SIGN)",
        all(b > a for a, b in zip(pxs, pxs[1:])), " ".join("%.3f" % p for p in pxs))

    # --- gain is bounded: a small head move must NOT fling the dot ---
    # This is the failure Dylan reported from the 45-term quadratic. Geometry's gain is fixed by
    # optics at ~1.5 display-frames per world-frame; anything far above that is a runaway.
    d1 = geometric_pixel([0.50, 0.5, 0.48, 0.5, 0, 0, 0, 0])[0]
    d2 = geometric_pixel([0.55, 0.5, 0.53, 0.5, 0, 0, 0, 0])[0]
    gain = (d2 - d1) / 0.05
    chk("gain is bounded and near the optical ratio (no runaway)", 1.0 < gain < 2.2,
        "%.2f display-frames per world-frame" % gain)

    # --- parallax: same direction, different depth -> different pixel, and MORE at close range ---
    near = geometric_pixel([0.60, 0.5, 0.55, 0.5, 0, 0, 0, 0], depth_mm=400.0)
    far = geometric_pixel([0.60, 0.5, 0.55, 0.5, 0, 0, 0, 0], depth_mm=20000.0)
    chk("depth CHANGES the answer (parallax term is live)", abs(near[1] - far[1]) > 1e-4,
        "dv %.4f" % abs(near[1] - far[1]))
    mid = geometric_pixel([0.60, 0.5, 0.55, 0.5, 0, 0, 0, 0], depth_mm=2000.0)
    chk("parallax shrinks with distance (400mm shifts more than 2000mm)",
        abs(near[1] - far[1]) > abs(mid[1] - far[1]))

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
