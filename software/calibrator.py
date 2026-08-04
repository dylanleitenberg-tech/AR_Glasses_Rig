"""The learning core: map (world dot + eye-corner geometry) -> display pixel.

Model:  display_pixel = f(world_dot_x, world_dot_y, eyeL_x, eyeL_y, eyeR_x, eyeR_y)

f is a polynomial regression. The eye->display->world relationship is mildly
nonlinear (lens distortion + parallax as the glasses shift on the face), and a
degree-2 polynomial captures that with few parameters — which matters because every
sample costs you a manual nudge + approve.

What makes this version reliable rather than just functional:

  * Standardization — raw features and polynomial terms are centered/scaled, so the
    fit is well-conditioned and the regularization strength means the same thing
    regardless of where on the screen / how your glasses sit.
  * Auto-regularization (GCV) — the ridge penalty is chosen automatically by
    Generalized Cross-Validation each refit, via one SVD. No hand-tuned constant,
    and it self-adjusts from 6 samples up to thousands.
  * Robustness (Huber IRLS) — a single mis-tracked eye corner or a fat-fingered
    approve no longer warps the whole map; gross outliers are down-weighted.
  * Confidence weighting — samples carry the eye-tracker's match confidence, so
    crisp captures count more than marginal ones.

Depends only on numpy + stdlib, so it runs headless with no cameras/display.
"""
from typing import Optional
import itertools
import numpy as np


def _poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Expand X (n, d) into monomials of degree 1..`degree` (NO bias column).

    The bias/offset is handled by centering Y, so we don't include a constant
    column here. Degree 2 over 6 inputs -> 27 columns.
    """
    X = np.atleast_2d(X)
    n, d = X.shape
    cols = []
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            term = np.ones(n)
            for idx in combo:
                term = term * X[:, idx]
            cols.append(term)
    return np.column_stack(cols) if cols else np.empty((n, 0))


class Calibrator:
    """Incrementally fittable predictor of the display pixel (normalized [0,1])."""

    def __init__(self, n_features: int, degree: int = 2, min_samples: int = 6,
                 lambdas: Optional[np.ndarray] = None, robust_iters: int = 2,
                 huber_k: float = 1.5):
        self.n_features = n_features
        self.degree = degree
        self.min_samples = min_samples
        self.lambdas = (np.asarray(lambdas, float) if lambdas is not None
                        else np.logspace(-4.0, 1.5, 18))
        self.robust_iters = robust_iters
        self.huber_k = huber_k
        # learned state
        self.W = None                       # (m, 2) on standardized poly features
        self.x_mean = self.x_std = None     # raw-feature standardization
        self.p_mean = self.p_std = None     # poly-term standardization
        self.y_mean = None                  # output offset
        self.lambda_ = None                 # GCV-selected ridge strength
        self._fallback = np.array([0.5, 0.5])

    @property
    def is_trained(self) -> bool:
        return self.W is not None

    # ---- internals -----------------------------------------------------
    def _design(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.atleast_2d(X) - self.x_mean) / self.x_std
        P = _poly_features(Xs, self.degree)
        return (P - self.p_mean) / self.p_std

    def _ridge_gcv(self, P: np.ndarray, Y: np.ndarray, w: np.ndarray):
        """Weighted ridge whose penalty is picked by GCV. Returns (W, lambda)."""
        sw = np.sqrt(w)[:, None]
        Pw, Yw = P * sw, Y * sw
        # economy SVD once; evaluate every lambda cheaply off the spectrum
        U, s, Vt = np.linalg.svd(Pw, full_matrices=False)
        UtY = U.T @ Yw
        n = Pw.shape[0]
        s2 = s ** 2
        best = None
        for lam in self.lambdas:
            f = s2 / (s2 + lam)                      # influence per singular dir
            HY = U @ (f[:, None] * UtY)
            rss = float(np.sum((Yw - HY) ** 2))
            tr = n - float(np.sum(f))                # trace(I - H)
            gcv = n * rss / max(tr * tr, 1e-12)
            if best is None or gcv < best[0]:
                best = (gcv, lam, f)
        lam = best[1]
        coef = s / (s2 + lam)
        W = Vt.T @ (coef[:, None] * UtY)
        return W, lam

    # ---- public API ----------------------------------------------------
    def fit(self, X, Y, weights=None) -> None:
        """Fit on all samples so far. X (n,d), Y (n,2), optional weights (n,)."""
        X = np.atleast_2d(np.asarray(X, float))
        Y = np.atleast_2d(np.asarray(Y, float))
        n = X.shape[0]
        if n == 0:
            self.W = None
            return
        w = (np.ones(n) if weights is None
             else np.clip(np.asarray(weights, float).ravel(), 1e-3, None))
        self._fallback = np.average(Y, axis=0, weights=w)
        if n < self.min_samples:
            self.W = None                    # not enough to trust a poly fit yet
            return

        # TRAIN ON THE RESIDUAL, not the absolute pixel — predict() adds geometry back, so
        # fitting Y directly here would double-count it. Subtracting geometry first also makes the
        # learning problem far easier: the target becomes a small, smooth correction (kappa, real
        # display FOV, distortion) instead of the full nonlinear direction mapping, which is
        # exactly the part a 45-term quadratic had no business trying to reproduce from a handful
        # of clustered samples.
        Y_target = Y
        if self.residual_mode:
            base = np.array([self.geometric_pixel(xi) for xi in X], float)
            if base.shape == Y.shape and np.all(np.isfinite(base)):
                Y_target = Y - base

        self.x_mean = X.mean(0)
        self.x_std = X.std(0); self.x_std[self.x_std < 1e-9] = 1.0
        Xs = (X - self.x_mean) / self.x_std
        P = _poly_features(Xs, self.degree)
        self.p_mean = P.mean(0)
        self.p_std = P.std(0); self.p_std[self.p_std < 1e-9] = 1.0
        Ps = (P - self.p_mean) / self.p_std
        self.y_mean = np.average(Y_target, axis=0, weights=w)
        Yc = Y_target - self.y_mean

        # Robust IRLS: refit, down-weight gross residuals (Huber), repeat.
        rw = w.copy()
        W, lam = self._ridge_gcv(Ps, Yc, rw)
        for _ in range(max(0, self.robust_iters)):
            resid = np.linalg.norm(Yc - Ps @ W, axis=1)
            sigma = 1.4826 * np.median(resid) + 1e-9      # robust scale (MAD)
            t = self.huber_k * sigma
            hub = np.where(resid <= t, 1.0, t / np.maximum(resid, 1e-9))
            rw = w * hub
            W, lam = self._ridge_gcv(Ps, Yc, rw)
        self.W, self.lambda_ = W, lam

        # THE RESIDUAL MUST EARN ITS PLACE. Cross-validate it against doing nothing.
        #
        # Measured on the first good 17-sample set (2026-08-03): pure geometry scored 32 px @1080
        # held-out -- an HONEST number, since geometry has zero fitted parameters -- while every
        # learned residual was WORSE: degree 2 at 66 px, degree 2 with 100x regularisation at 61,
        # degree 1 at 85. The learning curve also RÖSE with more samples (58 -> 62 px from n=6 to
        # n=14), which is the signature of fitting noise rather than signal.
        #
        # A 45-term quadratic over 8 features cannot be constrained by seventeen samples, several
        # of which carry a mis-detected dot. So rather than trusting the fit, MEASURE it: k-fold
        # the residual against the geometry-only baseline on the same folds, and if it does not
        # beat geometry, discard it. That turns "is the calibration helping?" from a judgement call
        # into a number the code checks every time it fits.
        if self.residual_mode and n >= self.min_samples + 4:
            try:
                self._residual_helps = self._beats_geometry(X, Y, w)
            except Exception:
                self._residual_helps = False
            if not self._residual_helps:
                self.W = None          # fall back to pure geometry

    # GEOMETRY IS THE BACKBONE; LEARNING IS A RESIDUAL ON TOP.
    #
    # This used to be a switch: geometric_bootstrap until the model trained, then the polynomial
    # INSTEAD. That is what threw the dot across the display. poly_degree=2 over 8 features is a
    # 45-term quadratic taking over at min_samples_for_model=6, and the real sample sets spanned
    # ~0.02 of frame -- so it interpolated the cluster and extrapolated violently outside it.
    # Dylan on hardware: "dot flies off with the smallest movement."
    #
    # A polynomial does not know a display pixel is an angle. Geometry does, needs no data, and is
    # right everywhere from frame one. So geometry always predicts, and the learned model corrects
    # only what geometry cannot know -- true display FOV, kappa, real eye position, distortion.
    # `residual_mode` is what `fit` trains against, so the two halves cannot disagree about which
    # quantity is being learned.
    residual_mode = True

    _residual_helps = None       # None = not yet assessed; False = geometry was better

    def _beats_geometry(self, X, Y, w, folds=4):
        """k-fold: does the learned residual beat geometry alone on held-out samples?"""
        n = X.shape[0]
        idx = np.random.default_rng(0).permutation(n)
        e_res, e_geo = [], []
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            if len(tr) < self.min_samples or len(te) == 0:
                continue
            probe = Calibrator(self.n_features, self.degree, self.min_samples,
                               self.lambdas, self.robust_iters, self.huber_k)
            probe.residual_mode = False          # avoid recursing into this check
            base = np.array([self.geometric_pixel(X[i]) for i in tr])
            probe.fit(X[tr], Y[tr] - base, w[tr])
            for i in te:
                g = self.geometric_pixel(X[i])
                r = probe.predict_raw(X[i]) if probe.W is not None else np.zeros(2)
                r = np.clip(r, -self.max_residual, self.max_residual)
                e_res.append(np.linalg.norm(np.clip(g + r, 0, 1) - Y[i]))
                e_geo.append(np.linalg.norm(g - Y[i]))
        if not e_res:
            return False
        return float(np.median(e_res)) < float(np.median(e_geo))

    def predict(self, x) -> np.ndarray:
        """Predict a normalized display pixel for one feature vector (d,)."""
        base = self.geometric_pixel(x)
        if self.W is None:
            return np.clip(base, 0.0, 1.0)
        corr = self.y_mean + (self._design(np.asarray(x, float)) @ self.W)[0]
        if not self.residual_mode:
            return np.clip(corr, 0.0, 1.0)
        # The residual is a CORRECTION, and a correction that exceeds the whole display is a
        # symptom of extrapolation rather than a real offset. Bounding it keeps a data-starved or
        # badly-conditioned fit from undoing geometry that was already approximately right.
        corr = np.clip(np.asarray(corr, float), -self.max_residual, self.max_residual)
        return np.clip(base + corr, 0.0, 1.0)

    # Largest correction the learned residual may apply, in normalised display units. 0.25 = a
    # quarter of the display; anything beyond that is not a lens/kappa correction.
    max_residual = 0.25

    def geometric_pixel(self, x) -> np.ndarray:
        """Closed-form pixel from direction + stereo depth. Falls back to the old FOV-ratio
        bootstrap only if geometry.py is unavailable, so this file keeps working standalone."""
        try:
            from geometry import geometric_pixel as _g
            return _g(x)
        except Exception:
            return self.geometric_bootstrap(x)

    # Display and world-camera fields of view, used only by the untrained bootstrap below.
    # Kept as plain numbers rather than importing rig, so calibrator.py stays dependency-free.
    _WORLD_FOV = 70.0        # rig.WORLD_FOV, horizontal
    _DISPLAY_FOV = 50.0      # rig display fov_deg, horizontal (57 deg diagonal on 16:9)

    def geometric_bootstrap(self, x) -> np.ndarray:
        """Where the dot should appear, from world-camera direction alone. No training needed.

        WHY THIS EXISTS. Before the first sample the model has no weights, and predict() used to
        return a FIXED fallback point. Dylan caught what that means on hardware: "when i move my
        head up, the dot should move down to counteract and stay on the dot. dot is just staying in
        position." Exactly so -- a constant cannot track anything, so the very first corrections
        were made against a marker that ignored the world entirely.

        The world cameras already measure the dot's DIRECTION, and direction is most of the answer:
        a display pixel is essentially an angle. So map the dot's angular offset from the world
        cams' optical axis into display coordinates by the ratio of their fields of view
        (70 deg world vs 50 deg display). That is a first-order estimate -- it ignores eye position,
        parallax and distortion, which is precisely what the learned model exists to add -- but it
        MOVES CORRECTLY with head motion from the first frame, so the human is nudging a marker
        that already tracks rather than one nailed to the screen.

        Uses the mean of the two world cams, which averages out per-camera noise and is stable even
        when stereo disparity is small (distant targets)."""
        x = np.asarray(x, float).ravel()
        if x.size < 4:
            return self._fallback.copy()
        u = 0.5 * (x[0] + x[2])          # worldL_x, worldR_x
        v = 0.5 * (x[1] + x[3])          # worldL_y, worldR_y
        k = self._WORLD_FOV / self._DISPLAY_FOV
        return np.clip(np.array([0.5 + (u - 0.5) * k, 0.5 + (v - 0.5) * k]), 0.0, 1.0)

    def predict_raw(self, x) -> np.ndarray:
        """Unclipped prediction (for residual/warm-start models on top of a prior).

        Returns the running mean before training (≈0 for a residual model)."""
        if self.W is None:
            return self._fallback.copy()
        return self.y_mean + (self._design(np.asarray(x, float)) @ self.W)[0]

    def holdout_error(self, X, Y, weights=None, folds: int = 5):
        """k-fold CV error in normalized units. Returns (rms, median) or None."""
        X = np.atleast_2d(np.asarray(X, float))
        Y = np.atleast_2d(np.asarray(Y, float))
        n = X.shape[0]
        if n < self.min_samples + folds:
            return None
        w = np.ones(n) if weights is None else np.asarray(weights, float).ravel()
        idx = np.random.default_rng(0).permutation(n)
        dists = []
        for fold in range(folds):
            te = idx[fold::folds]
            tr = np.setdiff1d(idx, te)
            probe = Calibrator(self.n_features, self.degree, self.min_samples,
                               self.lambdas, self.robust_iters, self.huber_k)
            probe.fit(X[tr], Y[tr], w[tr])
            for i in te:
                dists.append(np.linalg.norm(probe.predict(X[i]) - Y[i]))
        d = np.asarray(dists)
        return float(np.sqrt(np.mean(d ** 2))), float(np.median(d))

    # kept for backward compatibility (RMS only)
    def cv_error(self, X, Y, weights=None):
        r = self.holdout_error(X, Y, weights)
        return None if r is None else r[0]
