"""glasses_fit.py — does the mount physically fit the REAL XREAL One Pro? (collision check)

cad_fit.py checks the see-through cone + eyeball; wearable.py checks the face. This checks the
GLASSES themselves: a measured-dimension keep-out model of the One Pro (front frame body with the
two see-through lens apertures + nose cutout removed, the brow the clip intentionally grips, and the
temple arms) — and reports how far each camera's solid (board + M12 lens) + its boom sits from a
collision. Negative = the part is INSIDE the glasses (clash). The brow grip-zone is excluded (the
clip is supposed to touch it). Sim frame: origin at the optic centre, +x right, +y up, +z forward.

One Pro measured specs: front 151.6 x 50.5 mm, 57 deg FOV, IPD M = 63 (sim uses 64), brow ~11 mm
(MEASURE), temples 148 mm, pantoscopic +-3.5 deg.  Lens aperture / lower-rim extents are estimates
until calipered on the real unit — this is a first-order screen; the real test is a dry-fit.
"""
import sys
import numpy as np

import rig
import wearable as w

DROP = rig.OPTIC_DROP if hasattr(rig, "OPTIC_DROP") else 17.0   # brow rail is +DROP above optic centre
# --- One Pro keep-out geometry (sim mm) ---
G_HW = 151.6 / 2          # half width
G_TOP = 17.0              # brow rail (top of frame) above the optic centre
G_BOT = G_TOP - 50.5      # frame bottom  (= -33.5)
G_FZ0, G_FZ1 = 0.0, 11.0  # front frame depth: optic plane -> forward
LENS_HW, LENS_HH = 28.0, 16.0   # see-through aperture half-w / half-h per eye (estimate; CALIPER)
NOSE_HW, NOSE_TOP = 13.0, -6.0  # nose cutout half-width, and it opens below this y
BROW_GRIP_Y = 8.0         # above this y the front frame is the brow the CLIP grips -> exclude
TEMPLE_X = 70.0           # temple hinge |x|
TEMPLE_R = 6.0            # temple-arm radius


def _in_lens_aperture(x, y):
    for cx in (-rig.NOMINAL_IPD/2, rig.NOMINAL_IPD/2):
        if abs(x - cx) <= LENS_HW and abs(y) <= LENS_HH:
            return True
    return False


PRISM_TH = 4.0     # the flat X-prism optic is solid glass ~4 mm thick in the see-through aperture


def _frame_depth(x, y, z):
    """How deep a point is inside the glasses solid (>0 = inside the clash); else <=0."""
    if not (-G_HW <= x <= G_HW and G_BOT <= y <= G_TOP and z <= G_FZ1):
        return -1.0
    if abs(x) <= NOSE_HW and y <= NOSE_TOP:   # nose cutout -> no material
        return -1.0
    if _in_lens_aperture(x, y):        # see-through hole: EMPTY except the PRISM glass at z in [0,PRISM_TH]
        if 0.0 <= z <= PRISM_TH:
            return min(z - 0.0, PRISM_TH - z)
        return -1.0
    if y >= BROW_GRIP_Y:               # the brow the clip grips on purpose -> not a clash
        return -1.0
    if z < G_FZ0:                      # behind the front frame (in the eye-relief gap)
        return -1.0
    return min(z - G_FZ0, G_FZ1 - z, y - G_BOT)   # inside the solid frame rim


def _temple_gap(pts):
    """Min distance from points to either temple-arm capsule (negative inside)."""
    g = np.inf
    for s in (-1, 1):
        a = np.array([s*TEMPLE_X, 12.0, 0.0]); b = np.array([s*(TEMPLE_X-6), 5.0, -148.0])
        ab = b - a
        for p in pts:
            t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
            g = min(g, np.linalg.norm(p - (a + t*ab)) - TEMPLE_R)
    return g


def glasses_gap(pts):
    """Min clearance of points to the glasses (frame body + temples). <0 = collision."""
    pts = np.atleast_2d(pts)
    fc = np.inf
    for p in pts:
        d = _frame_depth(p[0], p[1], p[2])
        if d > 0:
            fc = min(fc, -d)                      # inside the solid frame -> negative clearance
    tc = _temple_gap(pts)                         # signed distance to the temple arms
    return min(fc if np.isfinite(fc) else 99.0, tc)


import cad_fit as cf


def _solid_oriented(C, aim, board, deg):
    a, u0, v0 = cf._basis(np.asarray(aim) - np.asarray(C))
    th = np.radians(deg); u = np.cos(th)*u0 + np.sin(th)*v0; v = -np.sin(th)*u0 + np.cos(th)*v0
    sensor = np.asarray(C) - cf.BACK*a
    pts = []
    ts = np.linspace(-board/2, board/2, 6)
    for su in ts:
        for sv in ts:
            pts.append(sensor + su*u + sv*v)
    for s in np.linspace(0, cf.BACK + cf.FRONT, 4):
        c = sensor + s*a
        for k in range(6):
            ph = k*np.pi/3
            pts.append(c + cf.RLENS*(np.cos(ph)*u + np.sin(ph)*v))
    return np.array(pts)


def _all_clear(C, aim, board):
    """Best board orientation that clears cone>=1, eyeball>=2, face>=2, glasses>=1. Returns the
    margin (min of the four at the best orientation) or None."""
    best = None
    for deg in range(0, 180, 30):
        pts = _solid_oriented(C, aim, board, deg)
        m = min(cf.raw_cone_clear(pts) - 0, cf.eyeball_gap(pts) - 1,
                w.face_gap(pts) - 1, glasses_gap(pts) - 0)
        if best is None or m > best:
            best = m
    return best


def find_all_clear(name, board, aim, span=16, step=2):
    C0 = cf.CAMS[name][0]; best = None
    for dx in range(-span, span+1, step):
        for dy in range(-span, span+1, step):
            for dz in range(-span, span+1, step):
                C = C0 + np.array([dx, dy, dz], float)
                m = _all_clear(C, aim, board)
                if m is not None and m >= 0:
                    disp = np.linalg.norm([dx, dy, dz])
                    if best is None or disp < best[0]:
                        best = (disp, C, m)
    return best


def search_all_clear():
    print("\n== search: clears ALL of cone + eyeball + face + GLASSES (real 36 mm boards) ==")
    for name, aim in [("eyeR", cf.CANTH_R), ("eye2R", cf.CANTH_R), ("pupil", w.EYES["R"])]:
        r = find_all_clear(name, 36, aim)
        if r is None:
            print("  %-7s NO position within +-16mm clears all four at 36mm" % name)
        else:
            disp, C, m = r
            print("  %-7s -> %-18s (move %.0fmm, margin %+.1f)" % (name, np.round(C, 1).tolist(), disp, m))


def report():
    boards = {"worldR": 38, "eyeR": 36, "eye2R": 36, "pupil": 36}
    print("== glasses fit: camera solids + booms vs the XREAL One Pro (collision) ==")
    print("  (negative = the part is INSIDE the glasses; >=2 mm = clear). Brow grip-zone excluded.")
    worst = np.inf
    for name in ("worldR", "eyeR", "eye2R", "pupil"):
        pts = w.camera_solid(name, boards[name])      # board + lens + boom (sim coords)
        g = glasses_gap(pts); worst = min(worst, g)
        v = "CLEAR" if g >= 2 else ("TIGHT" if g >= 0 else "CLASH")
        print("  %-7s %+7.1f mm   %s" % (name, g, v))
    print("  => worst %+.1f mm  %s" % (worst, "FITS THE GLASSES" if worst >= 0 else "CLASH — fix"))
    return worst


if __name__ == "__main__":
    sys.exit(0 if report() >= 0 else 1)
