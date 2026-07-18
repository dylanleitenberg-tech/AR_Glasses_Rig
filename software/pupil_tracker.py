"""IR pupil + corneal-glint tracker -> per-eye gaze features (OpenCV + numpy).

Dark-pupil method: under near-infrared illumination the pupil is the darkest region of the
eye image, and the reflections of the IR LEDs off the cornea ("glints", a.k.a. first
Purkinje images) are tiny, very bright spots. We:
  1. threshold for the dark pupil, clean it up, and fit an ELLIPSE (an ellipse, not a
     circle, because the nose-bridge camera views the eye obliquely),
  2. find the bright glints,
  3. output a compact gaze feature: the pupil center AND the pupil-center -> glint vector
     (the "pupil-glint vector" used in video-oculography — it tracks gaze and is robust to
     small shifts of the camera on the face).

These per-eye features replace the eye-corner features in the calibration pipeline; the
same calibrator/prior/preset then learns the display-pixel mapping from a stronger signal.

Run `python3 pupil_tracker.py` for a synthetic self-test (no hardware needed).
"""
import argparse
import sys

import numpy as np
import cv2


class PupilResult:
    def __init__(self, ok, pupil=(0.5, 0.5), axes=(0.0, 0.0), angle=0.0,
                 glints=None, conf=0.0):
        self.ok = ok
        self.pupil = np.asarray(pupil, float)      # normalized (x, y) in [0,1]
        self.axes = axes                           # ellipse (major, minor) normalized
        self.angle = angle                         # ellipse angle, deg
        self.glints = glints if glints is not None else []   # list of normalized (x,y)
        self.conf = conf


class PupilTracker:
    def __init__(self, min_area_frac=0.004, max_area_frac=0.4,
                 dark_pct=4.0, glint_pct=99.6, max_glints=4,
                 min_round=0.28, max_dark_mean=95.0):
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.dark_pct = dark_pct       # pupil = darkest this-% of pixels
        self.glint_pct = glint_pct     # glints = brightest this-% of pixels
        self.max_glints = max_glints
        # robustness gates (lessons from EllSeg/Swirski: reject non-pupil blobs)
        self.min_round = min_round         # reject ellipses too elongated to be a pupil
        self.max_dark_mean = max_dark_mean # the pupil must actually be dark inside

    def detect(self, gray):
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        area = float(H * W)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---- pupil: the darkest blob, ellipse-fitted ----
        # RETR_EXTERNAL + a CLOSE first means glints *inside* the pupil (bright holes) don't
        # break the outer contour or the ellipse fit (a known failure mode otherwise).
        thr = max(8.0, float(np.percentile(blur, self.dark_pct)))
        _, dark = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)
        k = np.ones((5, 5), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)   # fill glint holes first
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, -1.0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_area_frac * area or a > self.max_area_frac * area or len(c) < 5:
                continue
            mask = np.zeros((H, W), np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            if cv2.mean(blur, mask=mask)[0] > self.max_dark_mean:   # must be dark inside
                continue
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * a / (peri * peri) if peri > 0 else 0
            score = circ * np.sqrt(a)          # round + sizeable
            if score > best_score:
                best_score, best = score, c
        if best is None:
            return PupilResult(False)
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(best)
        round_ratio = (min(MA, ma) / max(MA, ma)) if max(MA, ma) > 0 else 0
        if round_ratio < self.min_round:       # too elongated (occlusion/eyelid) -> reject
            return PupilResult(False)

        # ---- glints: tiny bright spots ----
        gthr = max(200.0, float(np.percentile(blur, self.glint_pct)))
        _, bright = cv2.threshold(blur, gthr, 255, cv2.THRESH_BINARY)
        gc, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glints = []
        for c in gc:
            a = cv2.contourArea(c)
            if a < 1 or a > 0.01 * area:        # glints are small
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            # keep glints near the pupil (on the cornea)
            if np.hypot(gx - cx, gy - cy) < 1.5 * max(MA, ma):
                glints.append((gx / W, gy / H))
        glints.sort(key=lambda g: np.hypot(g[0] - cx/W, g[1] - cy/H))
        glints = glints[:self.max_glints]

        conf = float(round_ratio) * (1.0 if glints else 0.4)
        return PupilResult(True, (cx/W, cy/H), (MA/W, ma/H), ang, glints, conf)

    def features(self, res):
        """Per-eye gaze feature: [pupil_x, pupil_y, pcg_x, pcg_y].

        pcg = pupil-center minus the mean glint (the pupil-glint vector). If no glint is
        found we fall back to the pupil center alone (pcg = 0)."""
        if not res.ok:
            return None
        if res.glints:
            g = np.mean(np.array(res.glints), axis=0)
            pcg = res.pupil - g
        else:
            pcg = np.zeros(2)
        return np.array([res.pupil[0], res.pupil[1], pcg[0], pcg[1]])

    def draw(self, bgr, res):
        if not res.ok:
            return bgr
        H, W = bgr.shape[:2]
        c = (int(res.pupil[0]*W), int(res.pupil[1]*H))
        axes = (int(res.axes[0]*W/2), int(res.axes[1]*H/2))
        cv2.ellipse(bgr, c, axes, res.angle, 0, 360, (0, 255, 0), 1)
        cv2.circle(bgr, c, 2, (0, 255, 0), -1)
        for g in res.glints:
            cv2.circle(bgr, (int(g[0]*W), int(g[1]*H)), 3, (0, 0, 255), 1)
        return bgr


# ----------------------------------------------------------------------
#  Synthetic NIR eye (for the self-test) — dark pupil, gray iris, bright glints
# ----------------------------------------------------------------------
def synth_eye(W=400, H=300, pupil=(0.5, 0.5), pupil_r=0.11, iris_r=0.26,
              glints=((0.47, 0.47), (0.54, 0.49)), tilt_deg=0.0, lid=0.0,
              glint_on_pupil=False, rng=None):
    """Synthetic NIR eye. tilt_deg foreshortens it (off-axis nose-bridge view); lid
    occludes from the top (fraction of the iris); glint_on_pupil puts a glint on the pupil.
    These are the cases the literature (EllSeg/NVGaze) flags as where naive trackers break."""
    rng = rng or np.random.default_rng(0)
    M = min(W, H)
    cf = float(np.cos(np.radians(tilt_deg)))         # horizontal foreshortening
    img = np.full((H, W), 60, np.uint8)
    cx, cy = int(pupil[0]*W), int(pupil[1]*H)
    cv2.ellipse(img, (cx, cy), (int(iris_r*M*cf), int(iris_r*M)), 0, 0, 360, 110, -1)
    cv2.ellipse(img, (cx, cy), (int(iris_r*M*cf), int(iris_r*M)), 0, 0, 360, 150, 8)
    cv2.ellipse(img, (cx, cy), (int(pupil_r*M*cf), int(pupil_r*M)), 0, 0, 360, 14, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for gx, gy in glints:
        cv2.circle(img, (int(gx*W), int(gy*H)), 3, 255, -1)
    if glint_on_pupil:
        cv2.circle(img, (cx + 2, cy - 2), 2, 255, -1)            # specular on the pupil
    if lid > 0:                                                   # upper-eyelid occlusion
        ey = cy - int(iris_r*M) + int(2*iris_r*M*lid)
        cv2.ellipse(img, (cx, ey - int(iris_r*M)), (W, int(iris_r*M*1.3)), 0, 0, 360, 95, -1)
    img = np.clip(img.astype(int) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img


def _battery(trk, n, rng, **kw):
    errs, found = [], 0
    for _ in range(n):
        px, py = rng.uniform(0.38, 0.62), rng.uniform(0.40, 0.60)
        gj = lambda: (px + rng.uniform(-0.05, 0.05), py + rng.uniform(-0.05, 0.05))
        res = trk.detect(synth_eye(pupil=(px, py), glints=(gj(), gj()), rng=rng, **kw))
        if res.ok:
            found += 1
            errs.append(np.hypot(res.pupil[0]-px, res.pupil[1]-py))
    med = float(np.median(errs)) if errs else 1.0
    return found, med


def run_selftest(n=200, seed=0):
    rng = np.random.default_rng(seed)
    trk = PupilTracker()
    print("== pupil-tracker self-test (%d synthetic NIR eyes per case) ==" % n)
    cases = [
        ("clean (on-axis)",            dict()),
        ("off-axis 40° + glint-on-pupil", dict(tilt_deg=40, glint_on_pupil=True)),
        ("off-axis 55°",               dict(tilt_deg=55)),
        ("eyelid occlusion 30%",       dict(lid=0.30)),
    ]
    clean_ok = False
    for name, kw in cases:
        found, med = _battery(trk, n, rng, **kw)
        flag = "ok" if (found >= 0.9*n and med < 0.02) else "degrades (deep-seg territory)"
        print("  %-30s detected %3d/%d  median %.4f norm (~%.1f px)  %s"
              % (name, found, n, med, med*400, flag))
        if name.startswith("clean"):
            clean_ok = found >= 0.95*n and med < 0.01
    print("  RESULT:", "PASS ✅ (clean solid; hard cases honestly flagged)" if clean_ok
          else "WEAK ⚠️")
    print("  note: heavy occlusion is where EllSeg/RITnet-class CNNs are needed — see RESEARCH.md")
    return 0 if clean_ok else 1


def run_cam(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print("could not open camera", idx); return 1
    trk = PupilTracker()
    print("live pupil tracking on cam %d — press Q to quit" % idx)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = trk.detect(frame)
        cv2.imshow("pupil", trk.draw(frame, res))
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    cap.release(); cv2.destroyAllWindows()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", type=int, help="run live on a camera index")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.cam is not None:
        return run_cam(a.cam)
    return run_selftest()


if __name__ == "__main__":
    sys.exit(main())
