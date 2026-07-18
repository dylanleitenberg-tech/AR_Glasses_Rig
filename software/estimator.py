"""estimator.py — the deployable PHYSICS + PUPIL pixel estimator (the "better estimator").

physics_preset.py proved the white-box approach (triangulate the dot, fit the 6-DOF pose,
forward-compute the pixel) reaches the ~0.5 px true-geometry floor with the pupil feature —
far better than the polynomial. But run raw it has an ugly tail: a few degenerate-gaze solves
diverged (max ~51 px) because the pupil term makes the Gauss-Newton chase a bad reading.

This packages that solver into a clean, reusable estimator with two robustness guards, so it's
deployable in the live loop instead of being an experiment:
  1. QUALITY CHECK — after the solve, re-predict the eye features at the recovered pose; if the
     fit residual is large or the pixel lands off-screen, the solve is untrustworthy.
  2. ROBUST FALLBACK — on a bad solve, fall back to the CORNER-ONLY physics solve (no pupil),
     which under-determines the pose a little (higher median) but is far more stable (tiny tail).
  Net: the low median of physics+pupil with the small tail of corner-only.

Usage (deployment):
    est = PhysicsEstimator(device, user_geometry)   # device from device.build_device(); geometry
    px = est.predict(features)                       # from the per-user identify/calibrate step

Run:  python3 estimator.py        (or:  python3 main.py --estimator-test)
"""
import sys
import numpy as np

import autosim
import pixel_map
from physics_preset import (triangulate_dot, estimate_pose, _eye_pred, _unq,
                            physics_predict)
from preset import build_preset


class PhysicsEstimator:
    """Robust physics+pupil pixel predictor. `device` is a Simulator carrying the (calibrated)
    device constants; `subject` is the per-user eye geometry (identified or measured)."""

    def __init__(self, device, subject, use_pupil=True, use_stereo=False,
                 resid_thresh=0.012, margin=0.01):
        self.dev = device
        self.subject = subject
        self.use_pupil = use_pupil
        self.use_stereo = use_stereo
        self.resid_thresh = resid_thresh        # normalized eye-feature RMS above which we distrust
        self.margin = margin
        self._last = np.array([0.5, 0.5])       # temporal hold for a total failure
        self.n_fallback = 0                     # diagnostics: how often the robust path fired

    def _neye(self, use_pupil, use_stereo):
        return 4 + (4 if use_stereo else 0) + (2 if use_pupil else 0)

    def _solve(self, features, use_pupil, use_stereo):
        """One physics solve -> (pixel or None, normalized pose-fit residual)."""
        cams = _unq(self.dev)
        P = triangulate_dot(self.dev, features[0:4])
        eye = np.asarray(features[4:4 + self._neye(use_pupil, use_stereo)], float)
        dev = estimate_pose(self.dev, self.subject, eye, P, use_pupil, use_stereo)
        pred_eye = _eye_pred(cams, self.subject, dev, P, use_pupil, use_stereo)
        resid = (np.inf if pred_eye is None
                 else float(np.linalg.norm(eye - pred_eye) / np.sqrt(len(eye))))
        g = self.dev.ground_truth(self.subject, dev, P)
        return (None if g is None else g[1]), resid

    def _onscreen(self, px):
        return px is not None and np.all(px >= -0.05) and np.all(px <= 1.05)

    def predict(self, features):
        """features -> display pixel [0,1]^2, robustly."""
        features = np.asarray(features, float)
        px, resid = self._solve(features, self.use_pupil, self.use_stereo)
        good = px is not None and resid < self.resid_thresh and self._onscreen(px)
        if not good and self.use_pupil:                 # robust corner-only fallback
            px2, resid2 = self._solve(features, False, False)
            if px2 is not None and (px is None or not self._onscreen(px) or resid2 < resid):
                px, self.n_fallback = px2, self.n_fallback + 1
        if px is None or not self._onscreen(px):
            return self._last.copy()                    # total failure -> hold last good
        px = np.clip(px, self.margin, 1.0 - self.margin)
        self._last = px
        return px


# ----------------------------------------------------------------------
#  Self-test: prove it keeps the physics+pupil MEDIAN while cutting the TAIL
# ----------------------------------------------------------------------
def selftest(n_test=10, seed=900, verbose=True):
    sim = autosim.Simulator(seed, use_pupil=True)       # 10-feature oracle
    poses, dots = pixel_map.pose_grid(), pixel_map.dot_grid()
    err = {"polynomial+pupil": [], "physics+pupil (raw)": [], "PhysicsEstimator (robust)": []}
    fellback = 0
    if verbose:
        print("== PhysicsEstimator self-test (robust physics+pupil) ==")
        print("  %d eyes x %d poses x %d dots ...\n" % (n_test, len(poses), len(dots)))
    for i in range(n_test):
        subj = sim.new_subject()
        p10 = build_preset(subj.descriptor(), n_sweep=4000, use_pupil=True)
        est = PhysicsEstimator(sim, subj, use_pupil=True)
        for dev in poses:
            for P in dots:
                g = sim.ground_truth(subj, dev, P)
                if g is None:
                    continue
                f, truth = g
                err["polynomial+pupil"].append(np.linalg.norm(p10.predict(f) - truth))
                raw = physics_predict(sim, subj, f, use_pupil=True)
                if raw is not None:
                    err["physics+pupil (raw)"].append(np.linalg.norm(raw - truth))
                err["PhysicsEstimator (robust)"].append(np.linalg.norm(est.predict(f) - truth))
        fellback += est.n_fallback
        if verbose and (i + 1) % 5 == 0:
            print("    ... %d/%d eyes" % (i + 1, n_test))

    px = 1080.0
    if verbose:
        print("\n  overlay error (px @1080p):       median     95th      max")
        for name, e in err.items():
            e = np.array(e) * px
            print("    %-28s %7.3f  %7.3f  %7.3f"
                  % (name, np.median(e), np.percentile(e, 95), e.max()))
    raw = np.array(err["physics+pupil (raw)"]) * px
    rob = np.array(err["PhysicsEstimator (robust)"]) * px
    # robust keeps the median (within 10%) and meaningfully cuts the worst-case tail
    median_ok = np.median(rob) < 1.10 * np.median(raw) + 0.05
    tail_ok = rob.max() < 0.6 * raw.max()
    ok = median_ok and tail_ok
    if verbose:
        print("\n  robust fallback fired on %d samples (the degenerate-gaze solves)" % fellback)
        print("  median preserved: %s   |   tail cut >40%%: %s"
              % ("PASS" if median_ok else "FAIL", "PASS" if tail_ok else "FAIL"))
        print("  =>", "ESTIMATOR OK — physics+pupil accuracy, robust tail ✅" if ok
              else "review ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest())
