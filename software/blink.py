"""blink.py — detect eye closure (blinks) so the live pipeline DISCOUNTS bad eye reads.

A blink or a half-closed eye gives garbage canthus + pupil readings (the lids and lashes
occlude the eye), so feeding them into the feature vector would corrupt the calibration.
This mirrors what the simulator already does: autosim.observe() drops a capture with
rig.BLINK_PROB. Here we detect the closure on the REAL eye images and flag the affected
eye's reads invalid; the capture layer then discounts that frame (and counts it, for the
both-cams-valid robustness metric).

Two signals (measured on the synthetic NIR eye, see the probe in blink.py --selftest):
  * PUPIL camera (robust, primary): the dark-pupil tracker fails (no round dark pupil) once
    the eye is more than ~half closed -> a lost pupil == a blink. Reuses pupil_tracker.
  * EYE-CORNER cameras (heuristic): the fraction of 'pupil-dark' pixels collapses from
    ~0.025 (open) to ~0.000 (closed). The dark threshold is the pupil-vs-rest NIR cut and
    MUST be calibrated to the camera's exposure (exposed as `dark_thresh`); the default is
    the synthetic-eye value used by the self-test.
"""
import sys

import numpy as np
import cv2


class BlinkDetector:
    def __init__(self, dark_thresh: float = 40.0, open_frac: float = 0.006):
        # dark_thresh: NIR level below which a pixel is 'pupil-dark'. CALIBRATE per camera
        #              (set it between the dark pupil and everything brighter under your IR).
        # open_frac:   minimum dark-pixel fraction for an eye to count as open.
        self.dark_thresh = dark_thresh
        self.open_frac = open_frac

    def openness(self, gray) -> float:
        """Proxy for the palpebral aperture: fraction of pupil-dark pixels (0 = closed)."""
        if gray is None:
            return 0.0
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return float((blur < self.dark_thresh).mean())

    def is_blink_region(self, gray) -> bool:
        """Blink/closure decision for an eye-corner camera crop (heuristic)."""
        return self.openness(gray) < self.open_frac

    @staticmethod
    def is_blink_pupil(pupil_result) -> bool:
        """Blink decision for the NIR pupil camera: a lost/!ok pupil == a closed eye."""
        return (pupil_result is None) or (not getattr(pupil_result, "ok", False))


# ----------------------------------------------------------------------
#  Self-test: open eyes pass, closed eyes are flagged (both signals)
# ----------------------------------------------------------------------
def selftest(n=60, seed=0, verbose=True):
    from pupil_tracker import synth_eye, PupilTracker
    rng = np.random.default_rng(seed)
    det = BlinkDetector()
    trk = PupilTracker()
    if verbose:
        print("== blink-detector self-test (synthetic NIR eyes) ==")
        print("  %-16s %10s %12s %12s" % ("lid (closure)", "openness", "region-blink%", "pupil-blink%"))
    rows = []
    for lid in (0.0, 0.3, 0.6, 0.9):
        opens, rb, pb = [], [], []
        for _ in range(n):
            img = synth_eye(pupil=(rng.uniform(.42, .58), rng.uniform(.44, .56)), lid=lid, rng=rng)
            opens.append(det.openness(img))
            rb.append(det.is_blink_region(img))
            pb.append(BlinkDetector.is_blink_pupil(trk.detect(img)))
        rows.append((lid, np.mean(opens), np.mean(rb), np.mean(pb)))
        if verbose:
            print("  %-16.2f %10.4f %11.0f%% %11.0f%%"
                  % (lid, np.mean(opens), 100 * np.mean(rb), 100 * np.mean(pb)))
    # open eyes (lid 0) must NOT be flagged; clearly closed eyes (lid 0.9) MUST be flagged
    open_row = rows[0]; closed_row = rows[-1]
    region_ok = open_row[2] < 0.1 and closed_row[2] > 0.8
    pupil_ok = open_row[3] < 0.1 and closed_row[3] > 0.5
    ok = region_ok and pupil_ok
    if verbose:
        print("  region heuristic: open<10%% & closed>80%%  -> %s" % ("PASS" if region_ok else "FAIL"))
        print("  pupil signal:     open<10%% & closed>50%%  -> %s" % ("PASS" if pupil_ok else "FAIL"))
        print("  =>", "BLINK DETECTOR OK — closures flagged, open eyes pass ✅" if ok
              else "WEAK ⚠️")
        print("  note: dark_thresh is the pupil-vs-rest NIR cut — CALIBRATE on real footage; the")
        print("  pupil-camera signal (tracker !ok) is the robust primary, the region heuristic is")
        print("  the fallback for corner cams that don't image the pupil.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest())
