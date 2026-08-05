"""anchor.py, WORLD-LOCKED projection: keep a virtual thing glued to a real 3D location.

This is the glue between "track the world" (world_mesh) and "draw on the display" (overlay). A
virtual entity (a monkey on a person, a label on a wall) lives at a 3D point in the WORLD frame.
Every frame the head has moved, so to keep the entity visually locked to the real spot we must
RE-PROJECT it: world 3D point -> where it lands on the AR display right now.

Two stages, and the second one is why naive AR drifts:

  1. GEOMETRY (world_mesh): project the 3D world point into BOTH world cameras at the current head
     pose (R_cw, C). The stereo pair (uL, uR) encodes the point's DEPTH, essential, because the
     world cameras and the display sit at slightly different places on the head, so the display
     pixel for a given camera pixel depends on how far away the thing is (parallax). A mono pixel
     alone cannot place it; the stereo observation can.

  2. CALIBRATED MAP (calibrator): map that stereo observation (+ the current eye-corner features,
     i.e. how the glasses sit on the face) to a DISPLAY pixel. This is exactly the mapping the
     human-in-the-loop calibration loop already learns. `display_map` is that learned function in
     production; in `--selftest` it is a synthetic device model so the world-lock math is proven.

Result: as the head turns, (R_cw, C) change, (uL, uR) change, and the composed display pixel
tracks the real point, the entity stays put in the world. A fixed display pixel would smear off.

Pure geometry + an injected `display_map`, so `--selftest` runs headless with NO hardware.
"""
import argparse
import sys

import numpy as np

from world_mesh import DEFAULT_F, DEFAULT_B
from rig import WORLD_RES


def project_into_cam(world_pt, R_cw, C, f, cx, cy, cam_dx=0.0):
    """World 3D point -> (u, v) pixel in a world camera. `cam_dx` shifts to the right cam
    (its centre is +baseline along the left-cam x). Returns None if behind the camera."""
    x = R_cw @ (np.asarray(world_pt, float) - np.asarray(C, float))   # left-cam coords
    x = x - np.array([cam_dx, 0.0, 0.0])                              # right cam = left shifted +B
    if x[2] <= 1e-6:
        return None
    return np.array([f * x[0] / x[2] + cx, f * x[1] / x[2] + cy])


class AnchorProjector:
    """Projects world 3D points to display pixels through the mesh pose + the calibrated map.

    display_map(obs) -> (u, v) normalized display pixel in [0,1]^2, or None if off-display.
      obs = (uL, vL, uR, vR) world-camera stereo pixels. In production this is a thin wrapper over
      calibrator.predict() that also folds in the current eye-corner features; in tests it's a
      synthetic device model."""

    def __init__(self, display_map, f=DEFAULT_F, B=DEFAULT_B, res=WORLD_RES):
        self.display_map = display_map
        self.f = f
        self.B = B
        self.W = res
        self.H = int(res * 3 // 4)
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

    def stereo_obs(self, world_pt, pose):
        """(uL, vL, uR, vR) or None if the point isn't seen by both world cameras."""
        R_cw, C = pose
        L = project_into_cam(world_pt, R_cw, C, self.f, self.cx, self.cy, 0.0)
        R = project_into_cam(world_pt, R_cw, C, self.f, self.cx, self.cy, self.B)
        if L is None or R is None:
            return None
        if not (0 <= L[0] <= self.W and 0 <= L[1] <= self.H and 0 <= R[0] <= self.W):
            return None                      # outside the world-camera image -> not observable
        return (L[0], L[1], R[0], R[1])

    def project(self, world_pt, pose):
        """World 3D point -> normalized display pixel (u, v) in [0,1], or None if not displayable."""
        obs = self.stereo_obs(world_pt, pose)
        if obs is None:
            return None
        return self.display_map(obs)

    def project_many(self, world_pts, pose):
        """Vectorized-ish helper: list of world points -> list of (uv or None)."""
        return [self.project(p, pose) for p in world_pts]

    @staticmethod
    def on_display(uv, margin=0.0):
        """True if a projected pixel falls within the AR display window [0,1]^2 (optionally a
        margin, so a partly-off billboard whose centre is near the edge still counts)."""
        return uv is not None and -margin <= uv[0] <= 1 + margin and -margin <= uv[1] <= 1 + margin


class Anchor:
    """A virtual entity pinned to the world. `points` are named 3D world coordinates (e.g. a
    person's head/feet/shoulders) so an avatar can be posed to them; `meta` carries id/kind."""

    def __init__(self, anchor_id, points, meta=None):
        self.id = anchor_id
        self.points = {k: np.asarray(v, float) for k, v in points.items()}
        self.meta = meta or {}

    def project(self, projector, pose):
        """{name: display_uv or None} for each anchor point at the current head pose."""
        return {k: projector.project(v, pose) for k, v in self.points.items()}


# --------------------------------------------------------------------------
#  Production display_map: chain the mesh stereo obs -> the trained calibrator.
# --------------------------------------------------------------------------
def calibrator_display_map(calibrator, cfg, eye_features_fn, res=WORLD_RES):
    """Build a display_map from the trained calibrator. `eye_features_fn()` returns the current
    eye-corner (and optional stereo/pupil) features so the map reflects how the glasses sit RIGHT
    NOW. Feature order MUST match config.feature_names (world dots first, then eye corners)."""
    def dm(obs):
        uL, vL, uR, vR = obs
        # normalize world-cam pixels to the calibrator's convention (0..1)
        W = res; H = res * 3 // 4
        wl = np.array([uL / W, vL / H]); wr = np.array([uR / W, vR / H])
        eye = eye_features_fn()                       # e.g. [eyeL_x,eyeL_y,eyeR_x,eyeR_y, ...]
        feats = np.concatenate([wl, wr, np.asarray(eye, float)])
        px = calibrator.predict(feats)                # -> normalized display pixel
        if px is None:
            return None
        u, v = float(px[0]), float(px[1])
        return (u, v) if (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0) else None
    return dm


# ==========================================================================
#  Self-test (no hardware): a synthetic device (world cams + a display with a
#  fixed offset/rotation on the head) proves the world-lock is parallax-correct
#  and that re-projection is REQUIRED (a fixed display pixel drifts).
# ==========================================================================
def _make_synthetic_device(f=DEFAULT_F, B=DEFAULT_B, res=WORLD_RES,
                           disp_f=None, disp_res=1080, offset=(6.0, 4.0, 2.0),
                           roll_deg=1.5):
    """Returns (display_map, ground_truth_display_project). The display is a pinhole rigidly fixed
    to the head at `offset` (mm, in left-cam frame) with a small rotation R_dc, so mapping a
    world-cam pixel to a display pixel genuinely depends on depth (parallax), exactly like the
    real optic vs the world cameras."""
    disp_f = disp_f or (disp_res / 2) / np.tan(np.radians(50) / 2)   # ~50 deg display FOV
    W = res; H = int(res * 3 // 4); cx = W / 2; cy = H / 2
    dcx = disp_res / 2; dcy = (disp_res * 9 / 16) / 2                # 16:9 display
    th = np.radians(roll_deg)
    R_dc = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    offset = np.asarray(offset, float)

    def _disp_px_from_cam(p_cam):
        p = R_dc @ (p_cam - offset)
        if p[2] <= 1e-6:
            return None                        # behind the display -> genuinely not displayable
        u = disp_f * p[0] / p[2] + dcx
        v = -disp_f * p[1] / p[2] + dcy        # world +y is UP (rig frame); screen +v is DOWN -> flip
        # NOTE: no [0,1] gate here: a value outside is OFF the small AR display; callers use
        # AnchorProjector.on_display() and the compositor CLIPS the overlay to the display window
        # (a close, tall person's head/feet exceed the ~50 deg display FOV but the body still shows).
        return (u / disp_res, v / (disp_res * 9 / 16))             # normalized display pixel

    def display_map(obs):
        uL, vL, uR, vR = obs
        disp = uL - uR
        if disp <= 1e-6:
            return None
        Z = f * B / disp                                            # stereo depth -> full 3D
        p_cam = np.array([(uL - cx) * Z / f, (vL - cy) * Z / f, Z])
        return _disp_px_from_cam(p_cam)

    def ground_truth(world_pt, pose):
        R_cw, C = pose
        p_cam = R_cw @ (np.asarray(world_pt) - C)                   # left-cam-frame 3D
        return _disp_px_from_cam(p_cam)

    return display_map, ground_truth


def _rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def selftest(verbose=True):
    if verbose:
        print("== anchor self-test (synthetic device, no hardware) ==")
    display_map, gt = _make_synthetic_device()
    proj = AnchorProjector(display_map)
    rng = np.random.default_rng(0)
    checks = []

    # world points spread in front of the head, and a series of head poses (turn + translate)
    pts = rng.uniform([-300, -200, 900], [300, 200, 1800], size=(40, 3))
    poses = [( _rot([0, 1, 0], yaw) @ _rot([1, 0, 0], pit),
              np.array([tx, ty, tz]) )
             for yaw, pit, tx, ty, tz in
             [(0, 0, 0, 0, 0), (6, -3, 30, 10, 20), (-8, 4, -25, -15, 40), (12, 2, 50, 5, -10)]]

    # (1) WORLD-LOCK: anchored projection matches the ground-truth display projection at every
    #     head pose (for points visible to both cams + display), to sub-pixel.
    max_err = 0.0; n_seen = 0
    for pose in poses:
        for p in pts:
            a = proj.project(p, pose); g = gt(p, pose)
            if a is not None and g is not None:
                n_seen += 1
                max_err = max(max_err, np.hypot((a[0] - g[0]) * 1920, (a[1] - g[1]) * 1080))
    checks.append(("anchored point stays locked across head poses (max %.4f px, %d obs)"
                   % (max_err, n_seen), max_err < 1e-3 and n_seen > 60))

    # (2) re-projection is NECESSARY: a display pixel fixed at pose0 DRIFTS as the head moves
    p = pts[0]
    fixed = proj.project(p, poses[0])
    moved = proj.project(p, poses[2])
    drift_px = np.hypot((fixed[0] - moved[0]) * 1920, (fixed[1] - moved[1]) * 1080)
    checks.append(("a NON-reprojected pixel would drift (%.0f px head-move), re-proj required"
                   % drift_px, drift_px > 50))

    # (3) PARALLAX: because cam != display, the display pixel depends on DEPTH (not just direction)
    near = np.array([0.0, 0.0, 900.0]); far = np.array([0.0, 0.0, 1800.0])   # same bearing, diff depth
    dn = proj.project(near, poses[0]); df = proj.project(far, poses[0])
    par_px = np.hypot((dn[0] - df[0]) * 1920, (dn[1] - df[1]) * 1080)
    checks.append(("depth changes the display pixel (parallax %.1f px), needs stereo, not mono"
                   % par_px, par_px > 2))

    # (4) visibility gating: a point BEHIND the head projects to None
    behind = np.array([0.0, 0.0, -500.0])
    checks.append(("point behind the head -> not displayable (None)",
                   proj.project(behind, poses[0]) is None))

    # (5) an Anchor with named points projects each (a person's head + feet)
    anc = Anchor("p1", {"head": [0, 150, 1200], "feet": [0, -150, 1200]}, meta=dict(kind="person"))
    got = anc.project(proj, poses[0])
    checks.append(("Anchor projects named points (head/feet) to display pixels",
                   got["head"] is not None and got["feet"] is not None
                   and got["head"][1] < got["feet"][1]))   # head above feet on screen

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "ANCHOR OK, parallax-correct world-lock via mesh pose + calibrated map ✅"
              if ok else "PROBLEM ⚠️")
        print("  in production display_map = calibrator_display_map(trained calibrator); the stereo")
        print("  world obs carries depth so the lock holds as the head and the target both move.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="world-locked display projection")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    sys.exit(selftest())
