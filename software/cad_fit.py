"""cad_fit.py — does a REAL camera module physically fit at each rig.py position?

The occlusion guard in rig.py only checks camera CENTRES. But the parts we actually order are
36-38 mm PCBs with an M12 lens, not points. This models each camera as its true solid — a square
board behind the lens + the M12 lens cylinder in front — and checks, for both eyes:
  * clearance to the see-through CONE (negative = blocks the view), and
  * gap to the EYEBALL sphere (negative = the board would hit the eye/face).
The board is rotated about the optical axis to the BEST orientation (we'd design it that way), so
the reported clearance is best-case: if it's still negative, the module genuinely does not fit.

Real module sizes (measured from vendor pages, see ORDER_LIST.md):
  ELP AR0234 USB (world)      : 38 x 38 mm board + M12
  InnoMaker OV9281 USB (eye)  : 32 x 32 mm board + M12 (measured 2026-07-19)
  Arducam Mini OV9281 (MIPI)  : 24 x 25 mm board (smaller; needs Jetson, not USB)
"""
import sys
import numpy as np

import rig   # positions come from the single source of truth, so this can never drift

IPD = rig.NOMINAL_IPD; EYE_BEHIND = rig.EYE_BEHIND; GLOBE_R = 12.0
TAN = np.tan(np.radians(rig.CONE_HALF_DEG))
EYES = {"L": np.array([-IPD/2, 0, -EYE_BEHIND]), "R": np.array([IPD/2, 0, -EYE_BEHIND])}

# camera optical centre (from rig.py) + what it looks at (aim), in sim frame
CANTH_R = rig.nominal_outer_canthus()[1]; CANTH_L = rig.nominal_outer_canthus()[0]
CAMS = {
    "worldL": (np.array([-rig.WC_X, rig.WC_UP, rig.WC_FWD]), np.array([-rig.WC_X, rig.WC_UP, 200.])),
    "worldR": (np.array([rig.WC_X, rig.WC_UP, rig.WC_FWD]),  np.array([rig.WC_X, rig.WC_UP, 200.])),
    "eyeL":   (np.array([-rig.EC_X, rig.EC_UP, rig.EC_FWD]), CANTH_L),
    "eyeR":   (np.array([rig.EC_X, rig.EC_UP, rig.EC_FWD]),  CANTH_R),
    "eye2L":  (np.array([-rig.EC2_X, rig.EC2_UP, rig.EC2_FWD]), CANTH_L),
    "eye2R":  (np.array([rig.EC2_X, rig.EC2_UP, rig.EC2_FWD]),  CANTH_R),
    "pupil":  (rig.PUPIL_POS.copy(), EYES["R"]),                    # images the right pupil/CoR
}

# M12 lens + board mounting geometry (mm)
BACK = 10.0     # sensor/board plane sits this far behind the optical centre (M12 BFL + board)
FRONT = 7.0     # lens barrel protrudes this far in FRONT of the optical centre
RLENS = 9.0     # M12 lens holder radius (~18 mm dia)


def raw_cone_clear(pts):
    """Min clearance of sample points to the RAW see-through cone (both eyes). <0 = blocks view."""
    best = np.inf
    for e in EYES.values():
        d = pts[:, 2] - e[2]                      # forward distance from the eye CoR
        off = np.hypot(pts[:, 0] - e[0], pts[:, 1] - e[1])
        best = min(best, np.where(d <= 0, np.inf, off - TAN * d).min())
    return best


def eyeball_gap(pts):
    """Min gap from sample points to either eyeball sphere surface. <0 = hits the eye."""
    g = np.inf
    for e in EYES.values():
        g = min(g, (np.linalg.norm(pts - e, axis=1) - GLOBE_R).min())
    return g


def _basis(axis):
    a = axis / np.linalg.norm(axis)
    t = np.array([0, 0, 1.]) if abs(a[2]) < 0.9 else np.array([0, 1., 0])
    u = np.cross(a, t); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def module_points(C, aim, board, n=11):
    """Sample the board (square `board` mm, behind the lens) + the M12 lens cylinder, oriented to
    maximize cone clearance (search the in-plane rotation). Returns the best (clearance, gap, pts)."""
    a, u0, v0 = _basis(aim - C)
    sensor = C - BACK * a
    best = None
    for deg in range(0, 90, 15):                  # board can be rotated about the optical axis
        th = np.radians(deg)
        u = np.cos(th) * u0 + np.sin(th) * v0
        v = -np.sin(th) * u0 + np.cos(th) * v0
        ts = np.linspace(-board/2, board/2, n)
        grid = [sensor + su * u + sv * v for su in ts for sv in ts]   # full board (catches the eye-side)
        # M12 lens cylinder samples (rim) from sensor to front
        for s in np.linspace(0, BACK + FRONT, 6):
            c = sensor + s * a
            for k in range(8):
                ph = k * np.pi / 4
                grid.append(c + RLENS * (np.cos(ph) * u + np.sin(ph) * v))
        pts = np.array(grid)
        cl, gp = raw_cone_clear(pts), eyeball_gap(pts)
        score = min(cl, gp)
        if best is None or score > best[0]:
            best = (score, cl, gp)
    return best[1], best[2]


def report(board_by_cam, label):
    print("== %s ==" % label)
    print("  %-7s %6s %6s %6s   %-s" % ("camera", "board", "cone", "eye", "verdict"))
    allfit = True
    for name, (C, aim) in CAMS.items():
        b = board_by_cam[name]
        cl, gp = module_points(C, aim, b)
        fit = cl >= 0 and gp >= 1.0
        allfit = allfit and fit
        print("  %-7s %5.0fmm %+5.1f %+5.1f   %s"
              % (name, b, cl, gp, "FITS" if fit else ("CONE!" if cl < 0 else "HITS EYE!")))
    print("  =>", "ALL FIT" if allfit else "DOES NOT FIT — see flags", "\n")
    return allfit


def _obb(C, aim, board):
    """Oriented bounding box of a camera board (perpendicular to its optical axis)."""
    a, u, v = _basis(np.asarray(aim, float) - np.asarray(C, float))
    return dict(c=np.asarray(C, float), ax=[u, v, a], h=[board / 2, board / 2, (BACK + FRONT) / 2])


def _sat_clear(A, B):
    """Separating-axis clearance between two OBBs: >0 = gap (mm), <=0 = penetration."""
    axes = A["ax"] + B["ax"] + [np.cross(a, b) for a in A["ax"] for b in B["ax"]]
    t = B["c"] - A["c"]; best = -1e9
    for L in axes:
        n = np.linalg.norm(L)
        if n < 1e-9:
            continue
        L = L / n
        ra = sum(A["h"][i] * abs(np.dot(A["ax"][i], L)) for i in range(3))
        rb = sum(B["h"][i] * abs(np.dot(B["ax"][i], L)) for i in range(3))
        best = max(best, abs(np.dot(t, L)) - ra - rb)
    return best


def camera_clearance(board_by_cam):
    """INTER-CAMERA board clearance — the check cad_fit was missing. The per-camera fit above
    verifies each board clears the eye/cone/face; this verifies the boards don't collide with EACH
    OTHER (two boards can each clear the eye yet interpenetrate). Prints every pair under 3 mm;
    returns True if no CORE (non-stereo) pair overlaps."""
    import itertools
    O = {n: _obb(C, aim, board_by_cam[n]) for n, (C, aim) in CAMS.items() if n in board_by_cam}
    print("== inter-camera board clearance (boards must not collide with each other) ==")
    core = ("worldL", "worldR", "eyeL", "eyeR", "pupil")
    ok = True
    for a, b in itertools.combinations(list(O), 2):
        g = _sat_clear(O[a], O[b])
        if g < 3.0:
            stereo = a.startswith("eye2") or b.startswith("eye2")
            print("  %-7s <-> %-7s : %+6.1f mm  %s%s" % (a, b, g,
                  "OVERLAP" if g <= 0 else "tight", "  (FULL/stereo upgrade)" if stereo else ""))
            if g <= 0 and not stereo and a in core and b in core:
                ok = False
    print("  =>", "CORE 6-cam boards CLEAR — buildable ✅" if ok
          else "CORE BOARDS OVERLAP — reposition needed ⚠️", "\n")
    return ok


def standoff_needed(name, board, target_clear=2.0):
    """How far OUT along the lens->cam direction must the optical centre move for `board` to clear?"""
    C, aim = CAMS[name]
    a = (aim - C); a = a / np.linalg.norm(a)
    for push in np.arange(0, 60, 1.0):
        C2 = C - push * a                          # move the cam away from its target (more standoff)
        cl, gp = module_points(C2, aim, board)
        if cl >= target_clear and gp >= 2.0:
            return push, np.round(C2, 1)
    return None, None


def _aim_for(name, C):
    """World cams look +z from wherever they are; eye cams look at their fixed canthus/CoR target."""
    if name.startswith("world"):
        return C + np.array([0, 0, 100.])
    return CAMS[name][1]


def parallax(name, C):
    """For a stereo cam: angle between the primary-eye-cam view and this view of the same canthus."""
    if name not in ("eye2L", "eye2R"):
        return None
    prim = CAMS["eyeR" if name == "eye2R" else "eyeL"][0]
    canth = CAMS[name][1]
    v1, v2 = canth - prim, canth - C
    return np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


def find_fit(name, board, cone_min=1.5, eye_min=3.0, span=16, step=2):
    """Search a local grid for the nearest position that fits the real module with margin."""
    C0 = CAMS[name][0]; best = None
    for dx in range(-span, span + 1, step):
        for dy in range(-span, span + 1, step):
            for dz in range(-span, span + 1, step):
                C2 = C0 + np.array([dx, dy, dz], float)
                cl, gp = module_points(C2, _aim_for(name, C2), board)
                if cl >= cone_min and gp >= eye_min:
                    disp = np.linalg.norm([dx, dy, dz])
                    if best is None or disp < best[0]:
                        best = (disp, C2, cl, gp)
    return best


def search():
    print("== nearest FITTING position per camera (real USB modules, margin cone>=1.5 eye>=3.0) ==")
    boards = {"worldL": 38, "worldR": 38, "eyeL": 32, "eyeR": 32,
              "eye2L": 32, "eye2R": 32, "pupil": 32}
    for name in ("worldR", "eyeR", "eye2R", "pupil"):
        b = boards[name]; C0 = CAMS[name][0]
        res = find_fit(name, b)
        if res is None:
            print("  %-7s (%dmm) — no fit within +-16mm" % (name, b)); continue
        disp, C2, cl, gp = res
        px = parallax(name, C2)
        extra = "  parallax %.0f deg (was %.0f)" % (px, parallax(name, C0)) if px else ""
        moved = "KEEP (already fits)" if disp < 1e-6 else "move %.1fmm -> %s" % (disp, np.round(C2, 1).tolist())
        print("  %-7s (%dmm)  %s  [cone %+.1f eye %+.1f]%s" % (name, b, moved, cl, gp, extra))


def main():
    usb = {"worldL": 38, "worldR": 38, "eyeL": 32, "eyeR": 32,
           "eye2L": 32, "eye2R": 32, "pupil": 32}
    mini = dict(usb); mini.update({k: 24 for k in ("eyeL", "eyeR", "eye2L", "eye2R", "pupil")})
    report(usb, "ORDER-LIST USB modules (ELP 38mm world, InnoMaker OV9281 32mm eye/pupil)")
    camera_clearance(usb)                       # <- inter-camera collision check (boards vs boards)
    report(mini, "world ELP 38mm USB + eye/pupil Mini-OV9281 24mm (MIPI, needs Jetson)")
    search()
    return 0


if __name__ == "__main__":
    sys.exit(main())
