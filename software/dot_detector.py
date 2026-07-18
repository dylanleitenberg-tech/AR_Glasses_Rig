"""Find the hand-drawn dot on a blank page in the world-camera frame.

Assumption (matches your workflow): a dark dot drawn with pen/marker on a light,
mostly blank page. We find the most dot-like dark blob and return its centroid as
normalized (x, y) in [0, 1] of the frame, plus the annotated frame for display.

If you later want AI/segmentation instead of classic CV, swap detect() — the rest
of the pipeline only needs a normalized (x, y).
"""
from typing import Optional, Tuple
import cv2
import numpy as np


class DotDetector:
    def __init__(self, min_area: int = 8, max_area_frac: float = 0.05,
                 min_circularity: float = 0.6):
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.min_circularity = min_circularity

    def detect(self, frame) -> Tuple[Optional[Tuple[float, float]], object]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # dot is darker than the page -> invert so the dot is bright, then Otsu
        inv = cv2.bitwise_not(gray)
        _, mask = cv2.threshold(inv, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        max_area = self.max_area_frac * w * h
        best = None
        best_score = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > max_area:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < self.min_circularity:
                continue
            # prefer round + reasonably sized blobs
            score = circularity * np.sqrt(area)
            if score > best_score:
                best_score = score
                best = c

        annotated = frame.copy()
        if best is None:
            return None, annotated
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None, annotated
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        cv2.circle(annotated, (int(cx), int(cy)), 10, (0, 255, 0), 2)
        return (cx / w, cy / h), annotated
