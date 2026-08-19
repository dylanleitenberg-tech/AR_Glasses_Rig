"""Find the hand-drawn dot on a blank page in the world-camera frame.

Assumption (matches your workflow): a dark dot drawn with pen/marker on a light,
mostly blank page. We find the most dot-like dark blob and return its centroid as
normalized (x, y) in [0, 1] of the frame, plus the annotated frame for display.

If you later want AI/segmentation instead of classic CV, swap detect(), the rest
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

    # A dot on white paper is enclosed by page. These encode that, and are the difference
    # between "darkest round thing in the room" and "the target".
    min_surround = 90.0      # median grey of the ring around the blob (0-255)
    min_contrast = 0.35      # (surround - dot) / surround
    # A dot on paper NEVER touches the frame edge -- its page margin encloses it. Furniture,
    # door frames and frame-corner artefacts routinely do (2026-08-18: a corner blob scored
    # 64.6 against the real dot's 5 and won the JOINT pick 263/263 frames; 2026-08-04 fault
    # #4 was the same class at u=0.987). Class-level gate, not a tuned threshold.
    border_frac = 0.02       # candidates whose bbox comes this close to any edge are rejected

    def detect(self, frame) -> Tuple[Optional[Tuple[float, float]], object]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # dot is darker than the page -> LOCALLY-dark threshold. Global Otsu broke on the
        # rig's own stereo pair (2026-08-18): the two cameras auto-expose differently, and on
        # the brighter frame Otsu merged ceiling+laptop+target into one 1.8 Mpx blob. The
        # target's physics is LOCAL contrast (dark ink on the white page around it), which is
        # exactly what an adaptive threshold measures, and it is exposure-invariant.
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 75, 8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        # CLOSE bridges thin white gaps so a segmented target (a glyph with white slits,
        # 2026-08-18) contours as ONE blob; the circularity/surround gates still apply to
        # the fused shape, so this does not admit new clutter classes.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
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
            # PREFER A DOT THAT SITS ON WHITE PAPER, not merely a dark round blob.
            #
            # The target is defined as "a dark round dot on white paper", and the second half of
            # that carries most of the information. Scoring only on roundness and size makes any
            # dark object in the room a candidate: on this rig it picked laptop keys, then dark
            # furniture, then a frame-corner artefact -- each time confidently, and each time the
            # stereo checks had to catch it after the fact (epipolar violation 0.286, implied
            # depth 150 mm). Measured in a cluttered room, the real dot never won on size alone.
            #
            # So require the SURROUND to be bright. A dot on paper is enclosed by page; a chair
            # leg or a keyboard key is enclosed by more dark. This is a property of the target
            # itself rather than a tuned threshold, and it rejects the whole class of distractor
            # rather than one instance of it.
            x, y, bw, bh = cv2.boundingRect(c)
            bx = int(self.border_frac * w); by = int(self.border_frac * h)
            if x <= bx or y <= by or x + bw >= w - bx or y + bh >= h - by:
                continue                     # touches the frame border -> not on a page
            pad = int(max(bw, bh) * 1.6) + 4
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            inner = gray[y:y + bh, x:x + bw]
            inner_mask = mask[y:y + bh, x:x + bw] > 0
            surround = float(np.median(patch))
            # the INK's darkness, not the rect's: a glyph with internal white segments is
            # still a dark target; the rect median called it grey and rejected it
            dot_val = (float(np.median(inner[inner_mask])) if inner_mask.any()
                       else (float(np.median(inner)) if inner.size else 255.0))
            if surround < self.min_surround:          # not on a bright page -> not our target
                continue
            contrast = (surround - dot_val) / max(surround, 1.0)
            if contrast < self.min_contrast:
                continue
            score = circularity * np.sqrt(area) * contrast
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


    def candidates(self, frame):
        """Every blob that passes ALL the single-camera gates, as (score, u, v). Ranked.

        Exposed because the single-camera argmax is not enough on a cluttered scene, and the way
        it fails is systematic rather than unlucky. See detect_pair.
        """
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 75, 8)   # local, exposure-invariant (see detect)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        max_area = self.max_area_frac * w * h
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > max_area:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < self.min_circularity:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            bx = int(self.border_frac * w); by = int(self.border_frac * h)
            if x <= bx or y <= by or x + bw >= w - bx or y + bh >= h - by:
                continue                     # touches the frame border -> not on a page
            pad = int(max(bw, bh) * 1.6) + 4
            patch = gray[max(0, y - pad):min(h, y + bh + pad),
                         max(0, x - pad):min(w, x + bw + pad)]
            inner = gray[y:y + bh, x:x + bw]
            if patch.size == 0:
                continue
            inner_mask = mask[y:y + bh, x:x + bw] > 0
            surround = float(np.median(patch))
            dot_val = (float(np.median(inner[inner_mask])) if inner_mask.any()
                       else (float(np.median(inner)) if inner.size else 255.0))
            if surround < self.min_surround:
                continue
            contrast = (surround - dot_val) / max(surround, 1.0)
            if contrast < self.min_contrast:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            out.append((circ * np.sqrt(area) * contrast,
                        (M["m10"] / M["m00"]) / w, (M["m01"] / M["m00"]) / h,
                        float(np.sqrt(area / np.pi)) / w))
        out.sort(reverse=True)
        return out

    # Joint stereo selection tolerances.
    max_row_offset = 0.030   # |vL - vR| for one physical point on a roughly-aligned pair
    min_disp = 0.004         # below this the point is too far to trust (and near zero disparity)
    max_disp = 0.300         # above this it is closer than the rig can be looking at

    def detect_pair(self, frameL, frameR):
        """Pick the dot in BOTH frames JOINTLY. -> ((uL,vL), (uR,vR)) or (None, None).

        WHY NOT ARGMAX PER CAMERA. Each camera choosing its own best blob is what produced every
        world-dot failure on this rig: the two cameras confidently lock onto DIFFERENT objects and
        the stereo checks then reject the pair after the fact -- epipolar violation 0.154, implied
        depth 246 mm, worldR pinned against a frame edge. The single-camera score is
        `circularity * sqrt(area) * contrast`, so it rewards AREA, and in a real room there is
        always a bigger dark round thing than a dot on a page. MEASURED on this rig with the real
        target in view: the true dot ranked 2nd in worldL (score 4.99 at u=0.508) and 3rd in worldR
        (5.07 at u=0.536), beaten by larger blobs at 6.15 and 6.64.

        But the dot is the only thing in the room that appears in BOTH cameras at the SAME
        epipolar row with a PHYSICALLY PLAUSIBLE disparity. That is a property of the target, not a
        threshold to tune, and it is exactly the information a per-camera argmax throws away. So
        score every (L, R) pairing and let geometry break the tie: the true pair above sits at a
        row offset of 0.012 against the impostor's 0.020, and both must clear the same bar.

        Deliberately does NOT check the disparity SIGN -- a reversed world pair must still be
        caught loudly by calib_preflight.dot_geometry rather than silently accommodated here.
        """
        candL, candR = self.candidates(frameL), self.candidates(frameR)
        best, best_score = None, -1.0
        self.last_radii = None       # (rL, rR) normalized blob radii of the winning pair
        for sL, uL, vL, rL in candL[:12]:
            for sR, uR, vR, rR in candR[:12]:
                row = abs(vL - vR)
                if row > self.max_row_offset:
                    continue
                disp = abs(uL - uR)
                if not (self.min_disp <= disp <= self.max_disp):
                    continue
                # appearance score, penalised by how badly the pair violates the epipolar row
                score = (sL + sR) * (1.0 - row / self.max_row_offset)
                if score > best_score:
                    best_score, best = score, ((uL, vL), (uR, vR))
                    self.last_radii = (rL, rR)
        return best if best else (None, None)
