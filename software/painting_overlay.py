"""painting_overlay.py, REPLACE a framed painting on the wall with another image, live.

The star overlay traced a boundary; this fills one. Per frame, both world cameras find the
painting's canvas as a convex quadrilateral, the four corners are projected through the same
calibrated geometry as the dot and the star, and the replacement image is perspective-warped
into the projected quad on the display. The real gilt frame stays visible around it; only the
canvas is replaced. Head movement re-projects the corners every frame, so the new painting
hangs where the old one does.

QUAD SELECTION. A room is full of quadrilaterals, and the biggest one in the first live frame
was the CEILING. Three gates carry the selection, none of them tuned to this wall:
  * border rejection: a hung painting does not touch the frame edge (the ceiling does),
  * aspect and area bands: paintings are between 1:2 and 2.5:1 and take 1-30% of the frame,
  * nesting: a framed painting detects as 2-3 concentric quads (outer frame, inner frame,
    canvas); the INNERMOST of the dominant nest is the canvas. Covering the canvas and leaving
    the real frame is also what looks right through the optic.

DISPLAY REALITY. The optic is additive: black pixels are transparent, so dark paint renders
dim and the real painting shows through it. --gain lifts the replacement's brightness; a
demo-quality replacement wants a bright image or a dim wall, and that is physics, not a bug.

RUN (worn, session display constants):

    AR_DISPLAY_FOV=22.0 AR_DISPLAY_OFF=-0.006,-0.470 \
    python3 painting_overlay.py --run --world-cam-left 3 --world-cam-right 2 \
        --world-rot180 --overlay-x 1512 --image ../data/night_watch.jpg
"""
import argparse
import sys
import time

import numpy as np


def order_corners(q):
    """4x2 points -> (tl, tr, br, bl), stable under strong PERSPECTIVE.

    The first version sorted by angle around the centroid and then rotated to the smallest
    u+v: at oblique viewing angles the angular order can start on a diagonal and the artwork
    renders rotated ("it needs to know the orientation of the painting"). The classic
    sum/difference rule labels each corner independently of the others, so foreshortening
    cannot scramble it: tl minimises x+y, br maximises it, tr minimises y-x, bl maximises."""
    q = np.asarray(q, float)
    s = q.sum(1)
    d = q[:, 1] - q[:, 0]
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]])


class PaintingDetector:
    min_area_frac = 0.01
    max_area_frac = 0.30
    border_frac = 0.02
    aspect_lo, aspect_hi = 0.4, 2.6

    def quads(self, frame):
        """Convex 4-gons passing the physics gates, largest first."""
        import cv2
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        out = []
        for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, 75, 8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not (self.min_area_frac * w * h <= area <= self.max_area_frac * w * h):
                    continue
                ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
                if len(ap) != 4 or not cv2.isContourConvex(ap):
                    continue
                q = ap.reshape(4, 2).astype(float)
                bx, by = self.border_frac * w, self.border_frac * h
                if (q[:, 0].min() <= bx or q[:, 1].min() <= by
                        or q[:, 0].max() >= w - bx or q[:, 1].max() >= h - by):
                    continue                       # touches the frame edge -> ceiling/door class
                ww = q[:, 0].max() - q[:, 0].min()
                hh = q[:, 1].max() - q[:, 1].min()
                if hh <= 0 or not (self.aspect_lo <= ww / hh <= self.aspect_hi):
                    continue
                out.append((area, order_corners(q)))
        out.sort(key=lambda t: -t[0])
        return out

    def canvas(self, frame):
        """The painting's CANVAS: innermost quad of the BEST nest. (quad, centroid) or None.

        The nest is chosen by MEMBER COUNT, not by area: a framed painting is the only thing
        in a room that detects as two-plus concentric quads (outer frame, inner frame,
        canvas), while a doorway or a ceiling panel detects once. Anchoring on the largest
        single quad let a doorway hijack the pick in one camera and the stereo pairing died."""
        import cv2
        qs = self.quads(frame)
        if not qs:
            return None
        h, w = frame.shape[:2]
        used = [False] * len(qs)
        best = None                                  # (members, inner_area, inner_quad)
        for i, (ai, qi) in enumerate(qs):
            if used[i]:
                continue
            ci = qi.mean(0)
            group = [(ai, qi)]
            for j in range(i + 1, len(qs)):
                if used[j]:
                    continue
                if np.hypot(*(qs[j][1].mean(0) - ci)) < 0.03 * w:
                    group.append(qs[j]); used[j] = True
            inner = min(group, key=lambda t: cv2.contourArea(t[1].astype(np.float32)))
            key = (len(group), inner[0])
            if best is None or key > best[0]:
                best = (key, inner[1])
        # A nest (2+ concentric quads) is the strongest structural evidence, but a thin frame
        # at range merges frame and canvas into ONE quad -- measured live: the real painting
        # detected as a single clean quad and the >=2 rule refused it. A single quad is
        # therefore ACCEPTED as a candidate; the coplanarity and matte-interior gates in
        # detect_pair are what actually separate a painting from a window, and they run on
        # every candidate regardless of nest depth.
        inner_q = best[1]
        return inner_q, inner_q.mean(0) / [w, h]

    max_row_offset = 0.030
    min_disp = 0.004
    max_disp = 0.300

    def detect_pair(self, fL, fR):
        """Joint stereo pick: canvas in both frames at a physical disparity, or None.

        Two PAINTING-specific gates beyond the quad structure, because the room contains a
        framed WINDOW that is structurally identical (nested rectangles on a wall):

        COPLANARITY. A painting's canvas lies in its frame's plane, so the disparity of the
        interior equals the disparity of the frame. A window's interior is the world far
        behind it (interior disparity collapses toward zero) and a mirror's is the room
        reflected (disparity beyond the frame's). Measured by template-matching an interior
        patch from worldL into worldR along the epipolar row.

        MATTE INTERIOR. A painting sits at mid-brightness; a daylit window saturates."""
        import cv2
        a = self.canvas(fL)
        b = self.canvas(fR)
        if a is None or b is None:
            return None
        (qL, cL), (qR, cR) = a, b
        if abs(cL[1] - cR[1]) > self.max_row_offset:
            return None
        disp = cL[0] - cR[0]                      # SIGNED, worldL right of worldR
        if not (self.min_disp <= disp <= self.max_disp):
            return None
        h, w = fL.shape[:2]
        gL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY) if fL.ndim == 3 else fL
        gR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY) if fR.ndim == 3 else fR
        cx, cy = int(cL[0] * w), int(cL[1] * h)
        ph = max(24, int(0.10 * h))                # interior patch, safely inside the canvas
        pw = max(24, int(0.10 * w))
        y0, y1 = max(0, cy - ph // 2), min(h, cy + ph // 2)
        x0, x1 = max(0, cx - pw // 2), min(w, cx + pw // 2)
        patch = gL[y0:y1, x0:x1]
        if patch.size == 0 or float(np.median(patch)) > 220.0:
            return None                            # blown interior -> a window, not a painting
        # search worldR along the row for the patch; the strip spans plausible disparities
        sx0 = max(0, x0 - int(self.max_disp * w))
        sx1 = min(w, x1 + int(0.02 * w))
        strip = gR[max(0, y0 - 6):min(h, y1 + 6), sx0:sx1]
        if strip.shape[0] < patch.shape[0] or strip.shape[1] < patch.shape[1]:
            return None
        res = cv2.matchTemplate(strip, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < 0.5:
            return None                            # interior does not even match -> not planar art
        disp_interior = (x0 - (sx0 + loc[0])) / float(w)
        if abs(disp_interior - disp) > 0.02:
            return None                            # interior off the frame's plane -> window/mirror
        return qL, qR, disp, cL[1] - cR[1]


def project_quad(quadL_px, quadR_px, frame_wh, fov=None, center_off=None):
    """Painting corners seen in BOTH cams -> display corners, via the calibrated geometry.

    PER-CORNER STEREO, not a shared centroid disparity. Viewed obliquely, the near edge of
    the painting carries a larger disparity than the far edge; collapsing that to one number
    projected every corner at the same depth, so the parallax term (camera-to-eye offset)
    shifted all four corners equally and the rendered plane read as flat -- "the left side
    should be closer than the right. it has to know angles." Each corner is now its own
    stereo observation, so each gets its own depth and its own parallax."""
    import geometry
    from geometry import geometric_pixel_raw, NOMINAL_CANTH_UV_L, NOMINAL_CANTH_UV_R
    if fov is None:
        fov = geometry.DISPLAY_FOV_DEG
    off = geometry.DISPLAY_CENTER_OFF if center_off is None else np.asarray(center_off)
    w, h = frame_wh
    eyes = [NOMINAL_CANTH_UV_L[0], NOMINAL_CANTH_UV_L[1],
            NOMINAL_CANTH_UV_R[0], NOMINAL_CANTH_UV_R[1]]
    qL = order_corners(quadL_px) / [w, h]
    qR = order_corners(quadR_px) / [w, h]
    out = np.empty((4, 2))
    for i in range(4):
        feats = np.array([qL[i, 0], qL[i, 1], qR[i, 0], qR[i, 1]] + eyes)
        out[i] = geometric_pixel_raw(feats, display_fov_deg=fov) \
            - geometry.DISPLAY_CENTER_OFF + off
    return out


def run(args):
    import cv2
    import geometry
    from cameras import Camera
    art = cv2.imread(args.image)
    if art is None:
        print("!! cannot read %s" % args.image)
        return 1
    if args.gain != 1.0:
        art = cv2.convertScaleAbs(art, alpha=args.gain, beta=0)
    ah, aw = art.shape[:2]
    src = np.array([[0, 0], [aw, 0], [aw, ah], [0, ah]], np.float32)

    wl = Camera(args.world_cam_left, 1920, 1200, name="worldL", rot180=args.world_rot180)
    wr = Camera(args.world_cam_right, 1920, 1200, name="worldR", rot180=args.world_rot180)
    det = PaintingDetector()
    win = "painting-overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.overlay_x is not None:
        cv2.moveWindow(win, int(args.overlay_x), 0)   # move BEFORE fullscreen, see overlay.py
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    DW, DH = 1920, 1080
    fov = float(geometry.DISPLAY_FOV_DEG)
    off = np.array(geometry.DISPLAY_CENTER_OFF, float).copy()
    inset = float(args.inset)
    # per-corner display-space trim (tl, tr, br, bl): the user drags each artwork corner
    # onto the real frame corner, capturing local calibration error (size AND orientation)
    # exactly where the painting is. 'c' cycles which corner the arrows move; OFF = arrows
    # move the whole image via the global offset as before.
    corner_delta = np.zeros((4, 2))
    sel = -1                                       # -1 = global, 0..3 = tl,tr,br,bl
    held = None
    held_since = 0.0
    quit_arm = 0.0
    n = hits = coasts = 0
    # optical-flow carry: when the contour detector misses (motion blur), the last vetted
    # quad's corners are TRACKED into the new frame with LK, which survives exactly the blur
    # that breaks thresholded contours. Detection re-vets the object whenever it succeeds;
    # the carry only bridges gaps, capped so a stale quad cannot wander forever.
    gL_prev = gR_prev = None
    qL_last = qR_last = None
    carry_age = 0
    CARRY_MAX = 45                                 # ~1.5 s of coasting at 30 fps
    lk = dict(winSize=(31, 31), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    t0 = time.time()
    K_UP, K_DOWN, K_LEFT, K_RIGHT = 63232, 63233, 63234, 63235
    print("PAINTING OVERLAY LIVE. Arrows nudge, WASD coarse, -/= FOV, q twice to stop")
    while True:
        fl, fr = wl.read(), wr.read()
        if fl is None or fr is None:
            continue
        n += 1
        # HALF-RES detection: two full-res adaptive thresholds ran the loop at ~10 fps,
        # which reads as flicker on a worn display. Contours at 960x600 lose <2 px of corner
        # precision (well under the projection error) and cost a quarter as much. LK carry
        # runs at half res too; corners are scaled back to full-res coordinates throughout.
        fs_l = cv2.resize(fl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        fs_r = cv2.resize(fr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gL = cv2.cvtColor(fs_l, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fs_r, cv2.COLOR_BGR2GRAY)
        got = det.detect_pair(fs_l, fs_r)
        if got is not None:
            got = (got[0] * 2.0, got[1] * 2.0, got[2], got[3])
        if got is None and qL_last is not None and gL_prev is not None and carry_age < CARRY_MAX:
            pL, sL, _ = cv2.calcOpticalFlowPyrLK(
                gL_prev, gL, (qL_last * 0.5).astype(np.float32).reshape(-1, 1, 2), None, **lk)
            pR, sR_, _ = cv2.calcOpticalFlowPyrLK(
                gR_prev, gR, (qR_last * 0.5).astype(np.float32).reshape(-1, 1, 2), None, **lk)
            if sL is not None and sR_ is not None and sL.all() and sR_.all():
                got = (pL.reshape(4, 2).astype(float) * 2.0,
                       pR.reshape(4, 2).astype(float) * 2.0, None, None)
                carry_age += 1
                coasts += 1
        elif got is not None:
            carry_age = 0
        gL_prev, gR_prev = gL, gR
        canvas = np.zeros((DH, DW, 3), np.uint8)
        if got is not None:
            qL, qR = got[0], got[1]
            qL_last, qR_last = qL.copy(), qR.copy()
            # INSET: at range the canvas and frame detect as ONE quad (the frame's outer
            # edge), so painting into it covers the real gilt frame. Shrinking both camera
            # quads toward their centroids is exact on the painting's plane; '['/']' tune it
            # live until the artwork sits inside the frame.
            if inset < 0.999:
                qL = qL.mean(0) + (qL - qL.mean(0)) * inset
                qR = qR.mean(0) + (qR - qR.mean(0)) * inset
            dst = project_quad(qL, qR, (fl.shape[1], fl.shape[0]),
                               fov=fov, center_off=off)
            # EMA on the projected corners: per-frame independent detection jitters, and a
            # painting does not move. A real jump (new pose after a dropout) resets instead
            # of gliding, gated on corner displacement.
            if held is not None and np.abs(dst - held).max() < 0.05:
                dst = 0.6 * dst + 0.4 * held
            held, held_since = dst, time.time()
            hits += 1
        elif held is not None and time.time() - held_since > 0.6:
            held = None
        if held is not None:
            shown_q = held + corner_delta
            dst_px = (np.clip(shown_q, -0.5, 1.5) * [DW, DH]).astype(np.float32)
            M = cv2.getPerspectiveTransform(src, dst_px)
            warped = cv2.warpPerspective(art, M, (DW, DH))
            np.copyto(canvas, warped, where=warped > 0)
            if sel >= 0:
                p = dst_px[sel].astype(int)
                cv2.circle(canvas, tuple(p), 24, (0, 255, 255), 3)
                cv2.putText(canvas, "corner %s selected, arrows move IT ('c' cycles)"
                            % ["TL", "TR", "BR", "BL"][sel], (16, DH - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "LOOK AT THE PAINTING", (DW // 2 - 320, DH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)
        fps = n / max(time.time() - t0, 1e-6)
        hud = ("fps %.1f  lock %d/%d (+%d coast)   fov %.2f off (%+.3f,%+.3f) inset %.2f  ([/] adjust)"
               % (fps, hits, n, coasts, fov, off[0], off[1], inset))
        if time.time() - quit_arm < 2.5:
            hud += "   PRESS Q AGAIN TO QUIT"
        cv2.putText(canvas, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        if n % 100 == 0:
            print("STATS frame %d fps %.1f lock %d coast %d drawing %s"
                  % (n, fps, hits, coasts, held is not None), flush=True)
        k = cv2.waitKeyEx(1)
        if k == -1:
            continue
        fine, coarse = 0.004, 0.016
        if k == ord("c"):
            sel = sel + 1 if sel < 3 else -1
        elif k == K_LEFT:
            (corner_delta[sel].__setitem__(0, corner_delta[sel][0] - fine)
             if sel >= 0 else off.__setitem__(0, off[0] - fine))
        elif k == K_RIGHT:
            (corner_delta[sel].__setitem__(0, corner_delta[sel][0] + fine)
             if sel >= 0 else off.__setitem__(0, off[0] + fine))
        elif k == K_UP:
            (corner_delta[sel].__setitem__(1, corner_delta[sel][1] - fine)
             if sel >= 0 else off.__setitem__(1, off[1] - fine))
        elif k == K_DOWN:
            (corner_delta[sel].__setitem__(1, corner_delta[sel][1] + fine)
             if sel >= 0 else off.__setitem__(1, off[1] + fine))
        elif k == ord("a"): off[0] -= coarse
        elif k == ord("d"): off[0] += coarse
        elif k == ord("w"): off[1] -= coarse
        elif k == ord("s"): off[1] += coarse
        elif k in (ord("-"), ord("_")): fov = max(5.0, fov - 0.5)
        elif k in (ord("="), ord("+")): fov = min(60.0, fov + 0.5)
        elif k == ord("["): inset = max(0.40, inset - 0.02)
        elif k == ord("]"): inset = min(1.00, inset + 0.02)
        elif k in (ord("q"), ord("Q"), 27):
            now = time.time()
            if now - quit_arm < 2.5:
                break
            quit_arm = now
    wl.release(); wr.release(); cv2.destroyAllWindows()
    print("done: %d frames, painting locked on %d (%.0f%%)" % (n, hits, 100.0 * hits / max(n, 1)))
    print("tuned:  AR_DISPLAY_FOV=%.2f AR_DISPLAY_OFF=%.3f,%.3f  --inset %.2f" % (fov, off[0], off[1], inset))
    if np.abs(corner_delta).max() > 0:
        print("corner deltas (tl,tr,br,bl):", np.round(corner_delta, 4).tolist())
    return 0


def selftest(verbose=True):
    import cv2
    if verbose:
        print("== painting_overlay self-test (synthetic frames, no cameras) ==")
    checks = []
    W, H = 1920, 1200
    det = PaintingDetector()

    def render(quads, edge=False):
        f = np.full((H, W, 3), 225, np.uint8)
        for q in quads:
            if edge:
                cv2.polylines(f, [np.asarray(q, np.int32)], True, (60, 60, 60), 10)
            else:
                cv2.fillPoly(f, [np.asarray(q, np.int32)], (70, 70, 70))
        return f

    outer = np.array([[700, 400], [1150, 380], [1160, 700], [690, 690]])
    inner = outer + np.array([[40, 35], [-40, 35], [-40, -35], [40, -35]])
    # a framed painting = dark frame band with lighter canvas: draw outer solid, inner lighter
    f = render([outer])
    cv2.fillPoly(f, [inner.astype(np.int32)], (150, 150, 150))
    got = det.canvas(f)
    ok = got is not None and np.hypot(*(got[0].mean(0) - inner.mean(0))) < 25
    checks.append(("finds the framed painting and picks the INNER canvas quad", bool(ok)))

    # the ceiling class: a huge quad touching the border must be rejected
    ceil = np.array([[100, -50], [1800, -50], [1600, 300], [300, 300]])
    f2 = render([ceil])
    checks.append(("border-touching ceiling quad is rejected", det.canvas(f2) is None))

    # stereo: right frame shifted by a known disparity. The canvas needs TEXTURE (real art
    # has it) or the coplanarity template-match is degenerate on a flat fill.
    def paint(f, shift):
        cv2.fillPoly(f, [(inner - [shift, 0]).astype(np.int32)], (150,) * 3)
        c = inner.mean(0) - [shift, 0]
        for k, (dx, dy, r) in enumerate(((-60, -20, 22), (10, 30, 30), (70, -35, 16))):
            cv2.circle(f, (int(c[0] + dx), int(c[1] + dy)), r, (60 + 25 * k,) * 3, -1)
        cv2.line(f, (int(c[0] - 90), int(c[1] + 60)), (int(c[0] + 90), int(c[1] - 55)),
                 (90, 90, 90), 5)
        return f
    fL = paint(render([outer]), 0)
    sR = 80
    fR = paint(render([outer - [sR, 0]]), sR)
    pair = det.detect_pair(fL, fR)
    ok2 = pair is not None and abs(pair[2] * W - sR) < 8
    checks.append(("joint stereo pick at the true disparity", bool(ok2)))

    if pair is not None:
        d1 = project_quad(pair[0], pair[1], (W, H))
        finite = np.all(np.isfinite(d1))
        # convexity preserved through projection (a quad must stay a quad)
        hull = cv2.convexHull(d1.astype(np.float32))
        checks.append(("projected corners finite and still convex",
                       finite and len(hull) == 4))
    else:
        checks.append(("projection (skipped, no pair)", False))

    ok_all = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "PAINTING OVERLAY OK ✅" if ok_all else "PROBLEM ⚠️")
    return 0 if ok_all else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="replace a framed painting with another image")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--image", default="../data/night_watch.jpg")
    ap.add_argument("--inset", type=float, default=0.75,
                    help="shrink the detected quad toward its centre before painting: the "
                         "detected quad is usually the frame's OUTER edge, and the artwork "
                         "belongs on the canvas inside it. Tune live with [ and ]")
    ap.add_argument("--gain", type=float, default=1.6,
                    help="brightness lift for the replacement; the optic is additive and "
                         "cannot draw black, so dark art needs help")
    ap.add_argument("--world-cam-left", type=int, default=3)
    ap.add_argument("--world-cam-right", type=int, default=2)
    ap.add_argument("--world-rot180", action="store_true")
    ap.add_argument("--overlay-x", type=int, default=None)
    args = ap.parse_args()
    sys.exit(run(args) if args.run else selftest())
