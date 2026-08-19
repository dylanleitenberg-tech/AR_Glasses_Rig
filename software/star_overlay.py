"""star_overlay.py, TRACE a real star shape on the wall and hold the outline on it as the
head moves.

Rebuild of the demo lost with the 2026-08-16 machine (README: "draw an outline around a star
shape on my wall, matching its orientation and size, and the outline held on the shape as I
moved my head"). This version sits on the 2026-08-18 restored stack: rot180 world cameras at
native 1920x1200, the joint stereo detection idea from dot_detector, and geometry.geometric_pixel
with the session display constants.

HOW IT LOCKS. No world map and no persistence: the star's boundary is re-detected in BOTH world
cameras every frame and re-projected POINT BY POINT through the calibrated geometry. The head
moves -> the star moves in the camera frames -> every outline point lands somewhere new on the
display -> the outline stays on the star. Orientation and size come for free because the outline
IS the detected contour, not a fitted template. Depth enters through the centroid disparity
(the star is planar and small against its distance, so one disparity serves every point; depth
only drives the parallax term, so the approximation degrades gracefully, see geometry.py).

RUN (worn, calibrated session constants):

    AR_DISPLAY_FOV=22.0 AR_DISPLAY_OFF=-0.006,-0.466 \
    python3 star_overlay.py --run --world-cam-left 3 --world-cam-right 2 \
        --world-rot180 --world-res 1920x1200 --overlay-x 1512

    python3 star_overlay.py --selftest     # headless, no cameras

Draw/print a solid DARK star on light paper, a hand-width across, and put it on the wall.
"""
import argparse
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
#  Star detection: the same physics-of-the-target reasoning as dot_detector,
#  with star-shaped gates instead of round ones.
# --------------------------------------------------------------------------
class StarDetector:
    """A solid dark star on light paper: big enough to trace, spiky (low solidity, deep
    convexity defects), on a bright surround. Everything else in a room fails one of those."""

    min_area = 600
    max_area_frac = 0.20
    solidity_lo, solidity_hi = 0.25, 0.90   # 6-point hexagram is ~0.67 solid
    approx_eps = 0.02              # approxPolyDP epsilon as a fraction of perimeter
    verts_lo, verts_hi = 8, 16     # 5-point star ~10 vertices, 6-point ~12
    min_deep_defects = 4           # 5-point has 5 concave notches, 6-point has 6
    min_surround = 90.0            # median grey of the ring around the blob
    min_contrast = 0.30

    def candidates(self, frame):
        """Every star-like blob, ranked. Each is dict(score, contour Nx2 px, centroid uv,
        area) with centroid normalized to [0,1] of the frame."""
        import cv2
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(cv2.bitwise_not(gray), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area_frac * w * h:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if not (self.solidity_lo <= solidity <= self.solidity_hi):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_eps * peri, True)
            if not (self.verts_lo <= len(approx) <= self.verts_hi):
                continue
            # deep convexity defects = the notches between the star's points
            hull_idx = cv2.convexHull(c, returnPoints=False)
            deep = 0
            if hull_idx is not None and len(hull_idx) > 3:
                try:
                    defects = cv2.convexityDefects(c, hull_idx)
                except Exception:
                    defects = None
                if defects is not None:
                    depth_gate = 0.08 * np.sqrt(hull_area)      # scale-free notch depth
                    deep = int((defects[:, 0, 3] / 256.0 > depth_gate).sum())
            if deep < self.min_deep_defects:
                continue
            # on a bright surround, like the dot: rejects dark clutter as a class. The DARK
            # side is measured ON THE STROKE (median grey along the contour itself), not over
            # the bounding rect: a star drawn as LINE ART has a white interior, and a rect
            # median would call it bright and reject the real target.
            x, y, bw, bh = cv2.boundingRect(c)
            pad = int(max(bw, bh) * 0.6) + 4
            patch = gray[max(0, y - pad):min(h, y + bh + pad),
                         max(0, x - pad):min(w, x + bw + pad)]
            if patch.size == 0:
                continue
            surround = float(np.median(patch))
            if surround < self.min_surround:
                continue
            cpts = c.reshape(-1, 2)
            stroke = float(np.median(gray[cpts[:, 1].clip(0, h - 1),
                                          cpts[:, 0].clip(0, w - 1)]))
            contrast = (surround - stroke) / max(surround, 1.0)
            if contrast < self.min_contrast:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            out.append(dict(score=deep * (1.0 - solidity) * np.sqrt(area) * contrast,
                            contour=c.reshape(-1, 2).astype(float),
                            centroid=(M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h),
                            area=area))
        out.sort(key=lambda d: -d["score"])
        return out

    # joint stereo gates, same physics as dot_detector.detect_pair
    max_row_offset = 0.030
    min_disp = 0.004
    max_disp = 0.300

    def detect_pair(self, frameL, frameR):
        """Pick the star in BOTH frames JOINTLY (epipolar row + plausible disparity + similar
        area) -> (candL, candR) or None. Signed disparity: worldL's u must be RIGHT of
        worldR's, anything else is a mis-pairing (see main.py's signed-disparity gate)."""
        candL, candR = self.candidates(frameL), self.candidates(frameR)
        best, best_score = None, -1.0
        for a in candL[:6]:
            for b in candR[:6]:
                row = abs(a["centroid"][1] - b["centroid"][1])
                if row > self.max_row_offset:
                    continue
                disp = a["centroid"][0] - b["centroid"][0]      # SIGNED
                if not (self.min_disp <= disp <= self.max_disp):
                    continue
                ratio = a["area"] / b["area"]
                if not (0.5 <= ratio <= 2.0):
                    continue
                s = (a["score"] + b["score"]) * (1.0 - row / self.max_row_offset)
                if s > best_score:
                    best_score, best = s, (a, b)
        return best


def densify_closed(pts, n_target=160):
    """Insert points along each edge of a closed polygon, KEEPING every original vertex.

    Spacing adapts to hit ~n_target points total; vertices are never dropped, so a star's
    tips survive exactly. (Uniform resampling alone rounds them: a sample grid has no reason
    to land on a corner.)"""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    total = seg.sum()
    if total <= 0:
        return pts.copy()
    step = total / max(n_target, len(pts))
    out = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        out.append(a)
        k = int(seg[i] // step)
        for j in range(1, k + 1):
            out.append(a + (b - a) * (j / (k + 1)))
    return np.array(out)


def resample_closed(pts, n):
    """Uniform arc-length resampling of a closed polyline (Nx2 -> nx2)."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, total, n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)


def project_outline(contour_px, frame_wh, disp, row_off, n_points=72,
                    fov=None, center_off=None):
    """LEFT-cam contour (pixels) + centroid disparity -> display outline (nx2 normalized).

    Per point: normalize, synthesize the right-cam observation via the centroid disparity
    (planar target: one disparity serves the outline; depth only moves the parallax term),
    assemble the 8-feature vector with NOMINAL eye features (EYE_SHIFT_GAIN is 0.0, so
    nominal is exactly neutral), and run the SAME geometric_pixel as the dot loop.

    `fov` / `center_off` override the session display constants (the live adjustment keys);
    None keeps geometry's values."""
    import geometry
    from geometry import geometric_pixel_raw, NOMINAL_CANTH_UV_L, NOMINAL_CANTH_UV_R
    if fov is None:
        fov = geometry.DISPLAY_FOV_DEG
    off = geometry.DISPLAY_CENTER_OFF if center_off is None else np.asarray(center_off)
    w, h = frame_wh
    # TIPS ARE VERTICES, NOT SAMPLES (2026-08-18). Uniform arc-length resampling of the raw
    # contour lands ~a perimeter/n apart and walks straight past the star's sharp corners --
    # "isn't getting points". So: simplify the contour to its polygon (approxPolyDP keeps the
    # tips as exact vertices), then densify ALONG THE EDGES while keeping every vertex.
    import cv2
    cnt = np.asarray(contour_px, np.float32)
    peri = cv2.arcLength(cnt, True)
    ap = cv2.approxPolyDP(cnt, 0.008 * peri, True).reshape(-1, 2).astype(float)
    poly = ap if len(ap) >= 6 else np.asarray(contour_px, float)
    pts = densify_closed(poly, n_target=n_points) / np.array([w, h], float)
    eyes = [NOMINAL_CANTH_UV_L[0], NOMINAL_CANTH_UV_L[1],
            NOMINAL_CANTH_UV_R[0], NOMINAL_CANTH_UV_R[1]]
    out = np.empty((len(pts), 2))
    for i, (u, v) in enumerate(pts):
        feats = np.array([u, v, u - disp, v - row_off] + eyes)
        # geometric_pixel_raw folds in geometry.DISPLAY_CENTER_OFF; swap it for the live one
        out[i] = geometric_pixel_raw(feats, display_fov_deg=fov) \
            - geometry.DISPLAY_CENTER_OFF + off
    return out


# --------------------------------------------------------------------------
#  Live run
# --------------------------------------------------------------------------
def run(args):
    import cv2
    from cameras import Camera
    wl = Camera(args.world_cam_left, args.res_w, args.res_h, name="worldL",
                rot180=args.world_rot180)
    wr = Camera(args.world_cam_right, args.res_w, args.res_h, name="worldR",
                rot180=args.world_rot180)
    det = StarDetector()
    win = "star-overlay"                       # black = transparent on the see-through optic
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.overlay_x is not None:
        cv2.moveWindow(win, int(args.overlay_x), 0)   # MOVE BEFORE FULLSCREEN, see overlay.py
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    DW, DH = 1920, 1080
    import geometry
    fov = float(geometry.DISPLAY_FOV_DEG)              # live-adjustable display constants
    off = np.array(geometry.DISPLAY_CENTER_OFF, float).copy()
    held = None                                # last good outline, held through brief dropouts
    held_since = 0.0
    hold_s = 0.6
    quit_arm = 0.0
    n = hits = 0
    t0 = time.time()
    # macOS cv2.waitKeyEx arrow codes (waitKey&0xFF masks arrows to junk, hence Ex)
    K_UP, K_DOWN, K_LEFT, K_RIGHT = 63232, 63233, 63234, 63235
    print("STAR OVERLAY LIVE. Arrows nudge, WASD = coarse nudge, -/= FOV, q twice to stop")
    while True:
        fl, fr = wl.read(), wr.read()
        if fl is None or fr is None:
            continue
        n += 1
        pair = det.detect_pair(fl, fr)
        canvas = np.zeros((DH, DW, 3), np.uint8)
        hud = []
        if pair is not None:
            a, b = pair
            disp = a["centroid"][0] - b["centroid"][0]
            row_off = a["centroid"][1] - b["centroid"][1]
            outline = project_outline(a["contour"], (fl.shape[1], fl.shape[0]),
                                      disp, row_off, args.n_points,
                                      fov=fov, center_off=off)
            held, held_since = outline, time.time()
            hits += 1
            try:
                from world_mesh import DEFAULT_F, DEFAULT_B
                hud.append("depth %.0f mm" % (DEFAULT_F * DEFAULT_B / (disp * fl.shape[1])))
            except Exception:
                pass
        elif held is not None and time.time() - held_since > hold_s:
            held = None                        # stale, stop drawing rather than lie
        if held is not None:
            poly = (np.clip(held, -0.2, 1.2) * [DW, DH]).astype(np.int32)
            fresh = pair is not None
            cv2.polylines(canvas, [poly], True,
                          (255, 255, 0) if fresh else (120, 120, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "LOOK AT THE STAR", (DW // 2 - 260, DH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)
        fps = n / max(time.time() - t0, 1e-6)
        hud.append("fps %.1f  lock %d/%d" % (fps, hits, n))
        hud.append("fov %.2f  off (%+.3f, %+.3f)   arrows nudge, wasd coarse, -/= fov"
                   % (fov, off[0], off[1]))
        if time.time() - quit_arm < 2.5:
            hud.append("PRESS Q AGAIN TO QUIT")
        for i, line in enumerate(hud):
            cv2.putText(canvas, line, (16, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        k = cv2.waitKeyEx(1)
        if k == -1:
            continue
        fine, coarse = 0.004, 0.016
        if k == K_LEFT:
            off[0] -= fine
        elif k == K_RIGHT:
            off[0] += fine
        elif k == K_UP:
            off[1] -= fine
        elif k == K_DOWN:
            off[1] += fine
        elif k == ord("a"):
            off[0] -= coarse
        elif k == ord("d"):
            off[0] += coarse
        elif k == ord("w"):
            off[1] -= coarse
        elif k == ord("s"):
            off[1] += coarse
        elif k in (ord("-"), ord("_")):
            fov = max(5.0, fov - 0.5)
        elif k in (ord("="), ord("+")):
            fov = min(60.0, fov + 0.5)
        elif k in (ord("q"), ord("Q"), 27):
            now = time.time()
            if now - quit_arm < 2.5:
                break
            quit_arm = now
    wl.release(); wr.release(); cv2.destroyAllWindows()
    print("done: %d frames, star locked on %d (%.0f%%)" % (n, hits, 100.0 * hits / max(n, 1)))
    print("tuned constants:  AR_DISPLAY_FOV=%.2f AR_DISPLAY_OFF=%.3f,%.3f" % (fov, off[0], off[1]))
    try:
        import json, os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "display_session.json")
        with open(path, "w") as f:
            json.dump({"display_fov_deg": fov, "center_off": [float(off[0]), float(off[1])],
                       "tuned": "star_overlay live adjustment"}, f, indent=2)
        print("saved -> data/display_session.json")
    except Exception as e:
        print("could not save display_session.json: %s" % e)
    return 0


# --------------------------------------------------------------------------
#  Self-test: synthetic star frames, no cameras.
# --------------------------------------------------------------------------
def _star_poly(cx, cy, r_out, r_in, rot=0.0, n=5):
    ang = rot + np.arange(2 * n) * np.pi / n
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)


def selftest(verbose=True):
    import cv2
    if verbose:
        print("== star_overlay self-test (synthetic frames, no cameras) ==")
    checks = []
    W, H = 1920, 1200
    det = StarDetector()

    def render(shapes):
        f = np.full((H, W, 3), 235, np.uint8)
        for poly in shapes:
            cv2.fillPoly(f, [np.asarray(poly, np.int32)], (40, 40, 40))
        return f

    star = _star_poly(1100, 620, 130, 52, rot=0.35)
    fL = render([star])
    cand = det.candidates(fL)
    got = cand and np.hypot(cand[0]["centroid"][0] - 1100 / W,
                            cand[0]["centroid"][1] - 620 / H) < 0.01
    checks.append(("detects a synthetic star, centroid on target", bool(got)))

    # the REAL target is SIX-pointed (hexagram, r_in/r_out = 1/sqrt(3)), solid or line-art
    star6 = _star_poly(900, 500, 140, 140 / np.sqrt(3), rot=0.1, n=6)
    c6 = det.candidates(render([star6]))
    f6line = np.full((H, W, 3), 235, np.uint8)
    cv2.polylines(f6line, [star6.astype(np.int32)], True, (40, 40, 40), 9)
    c6l = det.candidates(f6line)
    checks.append(("detects the SIX-pointed star, solid AND line-art",
                   bool(c6) and bool(c6l)))

    # specificity: a filled circle and a square must NOT pass the star gates
    circ = _star_poly(700, 400, 120, 118, n=24)
    sq = np.array([[300, 800], [520, 800], [520, 1020], [300, 1020]])
    checks.append(("rejects circle and square (star-shaped gates)",
                   not det.candidates(render([circ])) and not det.candidates(render([sq]))))

    # joint pair with clutter: star + a dark blob; right frame shifted by a known disparity
    disp_px = 90.0
    starR = star - [disp_px, 0]
    blob = _star_poly(400, 300, 90, 88, n=24)
    pair = det.detect_pair(render([star, blob]), render([starR, blob - [5, 0]]))
    ok_pair = pair is not None and abs((pair[0]["centroid"][0] - pair[1]["centroid"][0]) * W
                                       - disp_px) < 6
    checks.append(("joint stereo pick finds the star pair at the true disparity", ok_pair))

    # projection: outline is a closed finite ring, and shifting the star in BOTH cams
    # (same disparity = same depth, new direction) shifts the outline the SAME direction --
    # the sign that, wrong, reads as "it moves with my head".
    if pair is not None:
        o1 = project_outline(pair[0]["contour"], (W, H),
                             disp_px / W, 0.0, n_points=48)
        shifted = det.detect_pair(render([star + [260, 0]]), render([starR + [260, 0]]))
        o2 = project_outline(shifted[0]["contour"], (W, H), disp_px / W, 0.0, n_points=48)
        finite = np.all(np.isfinite(o1)) and np.all(np.isfinite(o2))
        moved_right = (o2[:, 0].mean() - o1[:, 0].mean()) > 0.01
        span = np.ptp(o1[:, 0]) > 0.005 and np.ptp(o1[:, 1]) > 0.005
        checks.append(("outline projects finite, spans area, and moves WITH the target "
                       "direction (sign check)", finite and moved_right and span))
    else:
        checks.append(("outline projection (skipped, no pair)", False))

    # resampler: uniform closed resampling preserves the shape's extent
    rs = resample_closed(star, 72)
    checks.append(("closed resampling preserves extent",
                   abs(np.ptp(rs[:, 0]) - np.ptp(star[:, 0])) < 8
                   and abs(np.ptp(rs[:, 1]) - np.ptp(star[:, 1])) < 8))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "STAR OVERLAY OK, boundary-traced world lock via per-point geometry ✅"
              if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="trace a real star and hold the outline on it")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--world-cam-left", type=int, default=3)
    ap.add_argument("--world-cam-right", type=int, default=2)
    ap.add_argument("--world-rot180", action="store_true")
    ap.add_argument("--world-res", type=str, default="1920x1200")
    ap.add_argument("--overlay-x", type=int, default=None)
    ap.add_argument("--n-points", type=int, default=72)
    args = ap.parse_args()
    args.res_w, args.res_h = (int(s) for s in args.world_res.lower().split("x"))
    sys.exit(run(args) if args.run else selftest())
