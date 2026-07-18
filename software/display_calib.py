"""display_calib.py — measure the REAL XREAL optics (FOV + K1/K2) so the sim stops guessing.

`rig.py` currently uses placeholder display optics: fov_deg=50, k1=0.06, k2=0.02. Now that the
glasses are in hand you can MEASURE them. This is the device-side counterpart to the caliper
measurements, and it lowers the systematic floor (a known display map = less residual).

Two measurements:

  1. FOV (wall method, no photo math):
     - Render two vertical edge markers at the far left/right of the display (`--render-edges`).
     - Stand a measured distance D from a wall; mark where the left and right markers land on
       the wall; measure the width W between the marks.
     - `python3 display_calib.py --fov D_mm W_mm`  ->  horizontal FOV in degrees.

  2. DISTORTION K1/K2 (photo method):
     - `--render-grid` shows a checkerboard fullscreen on the glasses (do it in a DARK room so
       the photo is mostly the pattern). Photograph it straight-on through the optic with a phone
       held at the eye position.
     - `python3 display_calib.py --analyze photo.jpg`  detects the checkerboard, fits the radial
       distortion the optic added, and prints k1/k2 + the exact `rig.py` lines to change.

`--selftest` validates the distortion FIT with a synthetic grid of KNOWN k1/k2 (no photo needed).
"""
import argparse
import json
import os
import sys
import numpy as np
import cv2

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BOARD = (9, 6)        # inner corners (cols, rows) of the rendered checkerboard
FILL = 0.85           # board spans this fraction of the display (big = good distortion leverage;
#                       your phone's ~70 deg FOV still contains the ~50 deg display with margin if centered)
#  Distortion is fit in DISPLAY-normalized coords ([-1,1] over the full display per axis, y DOWN to
#  match image/photo order) so the recovered k1/k2 are directly rig.py's DisplayOptics units.


# ----------------------------------------------------------------------
#  FOV — wall method
# ----------------------------------------------------------------------
def fov_from_wall(distance_mm, width_mm):
    """Horizontal FOV (deg) from the display's full width W projected on a wall at distance D."""
    return float(2.0 * np.degrees(np.arctan((width_mm / 2.0) / distance_mm)))


# ----------------------------------------------------------------------
#  Distortion fit — recover (k1, k2) + (scale, center) mapping ideal grid -> observed
# ----------------------------------------------------------------------
def ideal_grid(cols, rows):
    """Inner-corner positions in DISPLAY-normalized coords [-1,1]^2 (x right, y DOWN to match the
    image/photo order), at the SAME fractions of the display that render_grid draws them. Fitting
    in these coords yields k1/k2 directly in rig.py's DisplayOptics units. Row-major (cv2 order)."""
    lo = (1 - FILL) / 2.0
    fx = lo + FILL * (np.arange(cols) + 1) / (cols + 1)    # fractions of display width
    fy = lo + FILL * (np.arange(rows) + 1) / (rows + 1)    # fractions of display height
    gx, gy = np.meshgrid(fx * 2 - 1, fy * 2 - 1)           # -> [-1,1], y DOWN
    return np.column_stack([gx.ravel(), gy.ravel()])


def fit_distortion(observed, cols, rows, iters=80):
    """Fit observed = scale*ideal*(1 + k1 r2 + k2 r2^2) + center for the optic's radial distortion.

    JOINT Gauss-Newton over (scale, cx, cy, k1, k2) — a joint fit is essential: an alternating
    fit lets `scale` absorb the average radial expansion and biases k1 toward 0. Returns
    dict(k1, k2, scale, cx, cy, rms_px). numpy only."""
    ideal = ideal_grid(cols, rows)
    obs = np.asarray(observed, float)
    r2 = np.sum(ideal ** 2, axis=1)

    def model(p):
        scale, cx, cy, k1, k2 = p
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        m = scale * ideal * d[:, None]
        return m + np.array([cx, cy])

    scale0 = (obs.max(0) - obs.min(0)).mean() / (ideal.max(0) - ideal.min(0)).mean()
    p = np.array([scale0, obs[:, 0].mean(), obs[:, 1].mean(), 0.0, 0.0])
    h = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3])
    for _ in range(iters):
        r = (model(p) - obs).ravel()
        J = np.zeros((len(r), 5))
        for k in range(5):
            pp = p.copy(); pp[k] += h[k]
            J[:, k] = ((model(pp) - obs).ravel() - r) / h[k]
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.linalg.norm(step[3:]) < 1e-8 and np.linalg.norm(step[:3]) < 1e-4:
            break
    m = model(p)
    rms = float(np.sqrt(np.mean(np.sum((m - obs) ** 2, axis=1))))
    return dict(scale=float(p[0]), cx=float(p[1]), cy=float(p[2]),
                k1=float(p[3]), k2=float(p[4]), rms_px=rms)


def detect_checkerboard(photo_path):
    """Find the checkerboard corners (raster order). Downscales big phone images for speed, and
    tries the robust SB detector first. Returns (cols,rows,pts) or None. Detection works in the
    (downscaled) photo's pixels; the distortion fit is scale-invariant so that's fine."""
    img = cv2.imread(photo_path)
    if img is None:
        print("could not read", photo_path)
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    s = 1600.0 / max(H, W)
    if s < 1.0:
        gray = cv2.resize(gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):        # robust to blur/uneven light, and fast
        ok, c = cv2.findChessboardCornersSB(gray, BOARD, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = c.reshape(-1, 2)
    if corners is None:
        ok, c = cv2.findChessboardCorners(
            gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            cv2.cornerSubPix(gray, c, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            corners = c.reshape(-1, 2)
    if corners is None:
        print("checkerboard NOT found — the WHOLE board (all corners) must be in frame with a "
              "black margin. Retake: center it, back off a bit, straight-on, sharp.")
        return None
    return BOARD[0], BOARD[1], corners


# ----------------------------------------------------------------------
#  Renderers (display these on the glasses)
# ----------------------------------------------------------------------
def render_grid(w=1920, h=1080):
    cols, rows = BOARD
    bx = (np.linspace((1 - FILL) / 2, (1 + FILL) / 2, cols + 2) * w).astype(int)  # square boundaries
    by = (np.linspace((1 - FILL) / 2, (1 + FILL) / 2, rows + 2) * h).astype(int)
    board = np.zeros((h, w, 3), np.uint8)
    for j in range(rows + 1):
        for c in range(cols + 1):
            if (j + c) % 2 == 0:
                cv2.rectangle(board, (bx[c], by[j]), (bx[c + 1], by[j + 1]), (255, 255, 255), -1)
    cv2.namedWindow("display-calib", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("display-calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Checkerboard on the glasses. Photograph it straight-on through the optic (dark room). q to close.")
    while True:
        cv2.imshow("display-calib", board)
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


def render_edges(w=1920, h=1080):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.line(img, (2, 0), (2, h), (0, 255, 0), 4)
    cv2.line(img, (w - 3, 0), (w - 3, h), (0, 255, 0), 4)
    cv2.putText(img, "mark where these LEFT/RIGHT edges hit a wall at known distance D; measure width W",
                (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.namedWindow("display-calib", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("display-calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        cv2.imshow("display-calib", img)
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


def write_calib(fov=None, k1=None, k2=None):
    path = os.path.join(_DATA, "display_calib.json")
    cur = {}
    if os.path.exists(path):
        cur = json.load(open(path))
    if fov is not None:
        cur["fov_deg"] = fov
    if k1 is not None:
        cur["k1"] = k1
        cur["k2"] = k2
    os.makedirs(_DATA, exist_ok=True)
    json.dump(cur, open(path, "w"), indent=2)
    print("\nsaved ->", path)
    print("To use it, set these in rig.py's `DisplayOptics(...)` / DISPLAY_K2:")
    if "fov_deg" in cur:
        print("    DisplayOptics(fov_deg=%.2f, ...)" % cur["fov_deg"])
    if "k1" in cur:
        print("    DisplayOptics(..., k1=%.4f)   # was 0.06" % cur["k1"])
        print("    DISPLAY_K2 = %.4f              # was 0.02" % cur["k2"])


# ----------------------------------------------------------------------
#  Self-test: recover KNOWN distortion from a synthetic grid (no photo)
# ----------------------------------------------------------------------
def selftest(verbose=True):
    cols, rows = BOARD
    ideal = ideal_grid(cols, rows)
    true_k1, true_k2, scale, cen = 0.08, -0.03, 420.0, np.array([960.0, 540.0])
    r2 = np.sum(ideal ** 2, axis=1)
    obs = scale * ideal * (1 + true_k1 * r2 + true_k2 * r2 * r2)[:, None] + cen
    obs = obs + np.random.default_rng(0).normal(0, 0.4, obs.shape)   # ~sub-px detection noise
    fit = fit_distortion(obs, cols, rows)
    ok = abs(fit["k1"] - true_k1) < 0.01 and abs(fit["k2"] - true_k2) < 0.01 and fit["rms_px"] < 1.0
    if verbose:
        print("== display-calib distortion fit self-test ==")
        print("  true  k1=%.4f k2=%.4f" % (true_k1, true_k2))
        print("  fit   k1=%.4f k2=%.4f   (scale=%.1f, center=%.0f,%.0f, rms=%.2f px)"
              % (fit["k1"], fit["k2"], fit["scale"], fit["cx"], fit["cy"], fit["rms_px"]))
        print("  =>", "FIT OK — recovers known distortion ✅" if ok else "FAIL ⚠️")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render-grid", action="store_true", help="checkerboard fullscreen (photograph it)")
    p.add_argument("--render-edges", action="store_true", help="left/right edge markers (FOV wall method)")
    p.add_argument("--analyze", type=str, metavar="PHOTO", help="fit K1/K2 from a checkerboard photo")
    p.add_argument("--fov", type=float, nargs=2, metavar=("D_mm", "W_mm"),
                   help="horizontal FOV from wall distance D and projected width W")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.render_grid:
        render_grid(); return 0
    if a.render_edges:
        render_edges(); return 0
    if a.fov:
        f = fov_from_wall(a.fov[0], a.fov[1])
        print("Horizontal FOV = %.2f deg" % f); write_calib(fov=f); return 0
    if a.analyze:
        det = detect_checkerboard(a.analyze)
        if det is None:
            return 1
        cols, rows, pts = det
        # the board can be detected from either end -> try both orders, keep the better fit
        f0 = fit_distortion(pts, cols, rows)
        f1 = fit_distortion(pts[::-1], cols, rows)
        fit = f0 if f0["rms_px"] <= f1["rms_px"] else f1
        print("Detected %d corners. Fitted optic distortion:" % len(pts))
        print("  k1=%.4f  k2=%.4f  (fit rms %.2f px)" % (fit["k1"], fit["k2"], fit["rms_px"]))
        if fit["rms_px"] > 4.0:
            print("  ⚠️ high rms — retake the photo (straight-on, board upright, fill the frame, sharp).")
        write_calib(k1=fit["k1"], k2=fit["k2"]); return 0
    p.print_help(); return 0


if __name__ == "__main__":
    sys.exit(main())
