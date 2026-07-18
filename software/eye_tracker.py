"""Track the corner of the eye (canthus) in a glasses-mounted inward camera.

A close-up glasses-mounted camera sees only a patch of eye, not a whole face, so
face-mesh trackers don't apply. Instead we use a rock-simple, robust method: you
capture a small template of your eye corner once (--calibrate-corners), and each
frame we template-match to locate it. The corner's position in the camera frame is
exactly the signal we want — it moves when the glasses shift on your face, which is
the whole point.

Returns normalized (x, y) in [0, 1] of the frame. Two of these (left + right eye)
give the 4 numbers that encode the glasses-on-face pose.
"""
import os
from typing import Optional, Tuple
import cv2


class EyeCornerTracker:
    def __init__(self, template_path: str, search_margin: float = 0.35):
        self.template_path = template_path
        self.search_margin = search_margin
        self.template = None
        self.tw = self.th = 0
        self.last = None            # last normalized hit, to constrain the search
        if os.path.exists(template_path):
            self._load()

    def _load(self) -> None:
        tmpl = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            return
        self.template = tmpl
        self.th, self.tw = tmpl.shape[:2]

    @property
    def ready(self) -> bool:
        return self.template is not None

    def calibrate(self, frame, window: str = "select eye corner") -> bool:
        """Interactively grab the eye-corner template from a frame. Returns ok."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
        x, y, w, h = [int(v) for v in roi]
        if w < 4 or h < 4:
            return False
        os.makedirs(os.path.dirname(self.template_path) or ".", exist_ok=True)
        cv2.imwrite(self.template_path, gray[y:y + h, x:x + w])
        self._load()
        return True

    def track(self, frame) -> Tuple[Optional[Tuple[float, float]], float]:
        """Return (normalized (x,y), score). (None, 0) if not found/ready."""
        if not self.ready:
            return None, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        # restrict the search to a window around the last hit (faster + steadier)
        if self.last is not None:
            cx, cy = self.last[0] * W, self.last[1] * H
            mx, my = self.search_margin * W, self.search_margin * H
            x0 = max(0, int(cx - mx)); y0 = max(0, int(cy - my))
            x1 = min(W, int(cx + mx)); y1 = min(H, int(cy + my))
            roi = gray[y0:y1, x0:x1]
            off = (x0, y0)
        else:
            roi, off = gray, (0, 0)

        if roi.shape[0] < self.th or roi.shape[1] < self.tw:
            roi, off = gray, (0, 0)
        if roi.shape[0] < self.th or roi.shape[1] < self.tw:
            return None, 0.0

        res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        cx = off[0] + max_loc[0] + self.tw / 2.0
        cy = off[1] + max_loc[1] + self.th / 2.0
        nx, ny = cx / W, cy / H
        self.last = (nx, ny)
        return (nx, ny), float(max_val)
