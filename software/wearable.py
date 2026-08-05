"""wearable.py, would the cameras stop the glasses resting on the face in normal position?

Bulky is fine (it's a prototype) but it must be WEARABLE: nothing on the carrier may push into the
face when the glasses sit at their normal vertex distance (held by nose pads + temples). cad_fit.py
only checked the eyeball; this adds a coarse FACE model (paraboloid skin that recedes laterally +
vertically, a forward NOSE ridge, open eye sockets) and checks every camera's solid (case + M12
lens) AND its boom against it. A negative gap = the part is BEHIND the facial surface there = it
would hold the glasses off the face. This is a first-order screen; the real test is a
dry-fit on the wearer's face. Sim frame: origin at the optic centre, +z forward (world), mm.
"""
import sys
import numpy as np

import rig
import cad_fit as cf   # reuse the camera table + the board/lens solid sampler

GLOBE_R = 12.0
EYES = {"L": np.array([-rig.NOMINAL_IPD/2, 0, -rig.EYE_BEHIND]),
        "R": np.array([rig.NOMINAL_IPD/2, 0, -rig.EYE_BEHIND])}
LENS_PLANE_Z = 0.0      # the glasses lens sits here; the face is behind it (more negative z)


def face_depth(x, y):
    """The most-FORWARD (largest z) the facial skin reaches at (x, y), away from the open sockets.
    Paraboloid that recedes laterally + vertically, plus a forward NOSE ridge near the midline."""
    base = -9.0 - 0.0022 * x * x - 0.0015 * y * y           # orbital/cheek/brow/temple skin
    # nose: protrudes forward near the midline, more as you go down toward the tip
    nb = max(0.0, 1.0 - (x / 17.0) ** 2)
    yprof = np.clip((12.0 - y) / 40.0, 0.0, 1.0)            # 0 at the bridge top, 1 lower down
    nose = nb * (3.0 + 7.0 * yprof)
    return base + nose


def in_socket(x, y):
    """Inside an open eye socket: the eyeball sphere (not a wall) is the keep-out there."""
    return min(np.hypot(x - rig.NOMINAL_IPD/2, y), np.hypot(x + rig.NOMINAL_IPD/2, y)) < 15.0


def face_gap(pts):
    """Min clearance of sample points to the FACE. In a socket -> distance to the eyeball surface;
    otherwise -> how far IN FRONT of the facial skin (z - face_depth). <0 means it hits the face."""
    g = np.inf
    for p in np.atleast_2d(pts):
        if in_socket(p[0], p[1]):
            d = min(np.linalg.norm(p - EYES["L"]), np.linalg.norm(p - EYES["R"])) - GLOBE_R
        else:
            d = p[2] - face_depth(p[0], p[1])
        g = min(g, d)
    return g


def camera_solid(name, board):
    """Sample points on a camera's CASE (board behind the lens + the M12 lens cylinder) + its BOOM,
    in sim coords, at the rig position. Mirrors cad_fit's optical model + the CAD boom routing."""
    C, aim = cf.CAMS[name]
    a, u, v = cf._basis(aim - C)
    pts = []
    ts = np.linspace(-board/2, board/2, 7)
    sensor = C - cf.BACK * a
    for su in ts:                                   # the PCB / case back plane
        for sv in ts:
            pts.append(sensor + su * u + sv * v)
    for s in np.linspace(0, cf.BACK + cf.FRONT, 5):  # the M12 lens cylinder rim
        c = C - cf.BACK * a + s * a
        for k in range(8):
            ph = k * np.pi / 4
            pts.append(c + cf.RLENS * (np.cos(ph) * u + np.sin(ph) * v))
    # boom: from the rail anchor to the case back (sim coords), matching xreal_one_mount.scad
    anchors_cad = {"worldR": [rig.NOMINAL_IPD/2 + 10, 0, -2], "worldL": [-(rig.NOMINAL_IPD/2 + 10), 0, -2],
                   "eyeR": [56-14, 0, -2], "eyeL": [-(56-14), 0, -2],
                   "eye2R": [rig.NOMINAL_IPD/2 + 13, -2, -2], "eye2L": [-(rig.NOMINAL_IPD/2 + 13), -2, -2],
                   "pupil": [8, -2, -2]}
    def cad2sim(c): return np.array([c[0], c[2] + 17.0, c[1]])
    anc = cad2sim(anchors_cad[name])
    casebk = C - (cf.BACK + 2) * a
    for t in np.linspace(0, 1, 18):
        pts.append(anc * (1 - t) + casebk * t)
    return np.array(pts)


def _solid_pts(C, aim, board):
    a, u, v = cf._basis(np.asarray(aim) - np.asarray(C))
    sensor = np.asarray(C) - cf.BACK * a
    pts = []
    ts = np.linspace(-board/2, board/2, 7)
    for su in ts:
        for sv in ts:
            pts.append(sensor + su*u + sv*v)
    for s in np.linspace(0, cf.BACK + cf.FRONT, 5):
        c = sensor + s*a
        for k in range(8):
            pts.append(c + cf.RLENS*(np.cos(k*np.pi/4)*u + np.sin(k*np.pi/4)*v))
    return np.array(pts)


def _clear(C, aim, board):                 # (cone, eyeball, face) clearances for a solid
    pts = _solid_pts(C, aim, board)
    return cf.raw_cone_clear(pts), cf.eyeball_gap(pts), face_gap(pts)


def _parallax(C, prim, canth):
    v1, v2 = np.asarray(canth) - np.asarray(prim), np.asarray(canth) - np.asarray(C)
    return np.degrees(np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))


def find_wearable(name, board, aim, span=20, step=2, want_parallax=False):
    """Nearest (or max-parallax) position that is WEARABLE + cone-clear + eyeball-clear."""
    C0 = cf.CAMS[name][0]; prim = cf.CAMS["eyeR" if "R" in name else "eyeL"][0]
    best = None
    for dx in range(-span, span+1, step):
        for dy in range(-span, span+1, step):
            for dz in range(-span, span+1, step):
                C = C0 + np.array([dx, dy, dz], float)
                cone, eye, face = _clear(C, aim, board)
                if cone >= 1.0 and eye >= 2.0 and face >= 2.0:
                    if want_parallax:
                        score = _parallax(C, prim, aim)           # maximize canthus-depth parallax
                    else:
                        score = -np.linalg.norm([dx, dy, dz])     # minimize the move
                    if best is None or score > best[0]:
                        best = (score, C, cone, eye, face)
    return best


def search_wearable():
    canth = cf.CANTH_R; cor = EYES["R"]
    print("\n== search: WEARABLE + cone-clear + eyeball-clear positions (real 36-38mm boards) ==")
    jobs = [("worldR", 38, np.array([32, 28, 200.]), False),   # world looks forward
            ("eyeR",   36, canth, False),
            ("eye2R",  36, canth, True),                        # stereo: maximize parallax
            ("pupil",  36, cor,   False)]
    for name, board, aim, wp in jobs:
        r = find_wearable(name, board, aim, want_parallax=wp)
        if r is None:
            print("  %-7s NO wearable position within +-20mm at this board size" % name); continue
        _, C, cone, eye, face = r
        px = "  parallax %.0f deg" % _parallax(C, cf.CAMS["eyeR"][0], aim) if wp else ""
        print("  %-7s -> %-18s cone %+.1f eye %+.1f face %+.1f%s"
              % (name, np.round(C, 1).tolist(), cone, eye, face, px))


def report():
    boards = {"worldL": 38, "worldR": 38, "eyeL": 36, "eyeR": 36,
              "eye2L": 36, "eye2R": 36, "pupil": 36}
    print("== wearability: camera solids vs a face model (gap to the facial surface) ==")
    print("  (negative = the part is behind the face -> would hold the glasses off; >=2mm = clear)")
    print("  %-7s %10s   %s" % ("camera", "face gap", "verdict"))
    worst = np.inf
    for name in ("worldR", "eyeR", "eye2R", "pupil"):     # mirror covers L
        pts = camera_solid(name, boards[name])
        g = face_gap(pts); worst = min(worst, g)
        v = "CLEAR" if g >= 2.0 else ("TIGHT" if g >= 0 else "HITS FACE")
        print("  %-7s %8.1f mm   %s" % (name, g, v))
    print("  => worst face gap %.1f mm  %s" % (worst, "WEARABLE" if worst >= 0 else "NOT WEARABLE, fix"))
    return worst


if __name__ == "__main__":
    sys.exit(0 if report() >= 0 else 1)
