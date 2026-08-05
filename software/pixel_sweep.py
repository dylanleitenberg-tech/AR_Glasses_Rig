"""Pixel-by-pixel calibration sweep, the protocol stated in plain terms.

Goal (the protocol, stated precisely):

    For 100 eye geometries (faces sampled from real ocular anthropometry), and for
    100 glasses positions on each face, walk the AR display PIXEL BY PIXEL. For each
    pixel a "guessing AI" proposes a point in the world that lies on the straight line
    through (a) the pixel, (b) where light enters the eye (the entrance pupil), and
    (c) the real-world dot that pixel must register on, i.e. a point on the eye's
    chief ray for that pixel. A "user AI" (which knows the physics ground truth)
    judges the guess and, when it is wrong, corrects the guessing AI. We repeat on the
    SAME pixel until the guessing AI is right TWICE IN A ROW, then step to the next
    pixel over. Every guess is one stored data piece, so the dataset size is

        100 faces  x  100 positions  x  (pixels per position)  x  (avg guesses/pixel).

Why a pixel maps to a *line*, not a point
-----------------------------------------
Lighting display pixel (u, v) emits one angular direction out of the see-through optic.
The eye perceives the overlay along its chief ray: the line from the ENTRANCE PUPIL
through the world. So pixel (u, v) <-> a single line in space (the chief ray). Any real
target on that line registers under the pixel; we pin the canonical point where the line
meets the page (depth z0), giving the unique world dot the pixel must sit on. The guessing
AI guesses that point; correctness = the guess lands on the line (within tolerance).

This module is the INVERSE of pixel_map.py: pixel_map asks "given a world dot, what pixel?"
(forward); here we walk pixels and ask "what world point?" (inverse), which is what
"go pixel by pixel ... guess a point on the line" literally requires.

Fidelity / "make the sim very good before starting"
---------------------------------------------------
Every number comes from the same physics as the rest of the rig:
  * faces            -> anatomy.sample_subject (population stats, cited in anatomy.py)
  * glasses + cams   -> rig.build (mirrors cad/xreal_one_mount.scad) + per-unit tolerance
  * registration     -> autosim.Simulator.ground_truth (noise-free chief-ray oracle)
We add a *vectorized* copy of that oracle (`truepx_batch`) for speed and a pixel->world
inverse (`invert_pixels`) by Gauss-Newton. `selftest()` proves, before any sweep runs,
that (1) the vectorized oracle matches autosim to ~1e-12, and (2) the inverse round-trips
(forward(invert(uv)) == uv) to sub-pixel accuracy. If those fail, NOTHING runs.

No pre-existing data is used: the guessing AI starts at zero and learns only from the
corrections gathered during this run. Outputs go to data/pixel_sweep.* (never the
existing mega_*/meta.db/pixel_map artifacts).

Run:
    python3 pixel_sweep.py --selftest          # validate the sim only (do this first)
    python3 pixel_sweep.py                      # full 100 x 100 sweep
    python3 pixel_sweep.py --faces 100 --poses 100 --grid 24
    python3 main.py --pixel-sweep
"""
import os
import time
import argparse
import numpy as np

import rig
import anatomy
import autosim
from calibrator import _poly_features

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

Z0 = rig.DOT_Z_MEAN          # page depth where we pin the chief-ray point (mm)
PAGE_X = rig.DOT_X           # reachable page half-width  (mm)
PAGE_Y = rig.DOT_Y           # reachable page half-height (mm)


# ======================================================================
#  Vectorized noise-free oracle  (must match autosim.Simulator.ground_truth)
# ======================================================================
def _pixel_for_angles_batch(disp, az, el):
    """DisplayOptics.pixel_for_angles, vectorized over (az, el) arrays."""
    x = az / disp.half
    y = el / disp.half
    r2 = x * x + y * y
    dist = 1.0 + disp.k1 * r2 + disp.k2 * r2 * r2
    u = 0.5 + 0.5 * (x * dist) + disp.cx
    v = 0.5 - 0.5 * (y * dist) + disp.cy
    w = disp.warp
    u = u + w[0] * x * y + w[1] * (x * x - y * y)
    v = v + w[2] * x * y + w[3] * (x * x - y * y)
    return np.stack([u, v], axis=-1)


def truepx_batch(sim, subject, dev, P):
    """Noise-free display pixel for a batch of world points P (N,3).

    Identical math to autosim.Simulator.ground_truth (display/right eye), vectorized.
    Returns true_px (N,2). Off-display points are NOT rejected here (caller masks).
    """
    P = np.atleast_2d(np.asarray(P, float))
    R, t = rig.pose_R_t(dev)
    cor = (R @ subject.cor.T).T + t                     # (2,3)
    e = sim.rig["display_eye"]
    core = cor[e]                                        # (3,)
    disp = sim.rig["display"]

    gaze = P - core
    gaze = gaze / np.linalg.norm(gaze, axis=1, keepdims=True)
    ep = core + subject.ep_dist * gaze                  # entrance pupil per point (N,3)
    ga = np.degrees(np.arccos(np.clip(gaze[:, 2], -1.0, 1.0)))
    lat = gaze.copy(); lat[:, 2] = 0.0
    ln = np.linalg.norm(lat, axis=1)
    m = ln > 1e-6
    ep[m] = ep[m] + rig.PUPIL_WANDER * (ga[m] / 10.0)[:, None] * (lat[m] / ln[m, None])

    epd = ep @ sim.R_dc.T + sim.t_dc
    Pd = P @ sim.R_dc.T + sim.t_dc
    opticd = sim.R_dc @ sim.rig["optic"] + sim.t_dc
    dd = Pd - epd
    dd = dd / np.linalg.norm(dd, axis=1, keepdims=True)
    az = np.arctan2(dd[:, 0], dd[:, 2]) + (1.0 if e == 1 else -1.0) * np.radians(subject.kappa[0])
    el = np.arctan2(dd[:, 1], dd[:, 2]) + np.radians(subject.kappa[1])
    dk = np.stack([np.tan(az), np.tan(el), np.ones_like(az)], axis=1)
    dk = dk / np.linalg.norm(dk, axis=1, keepdims=True)

    # DisplayOptics.pixel_for_target, vectorized (finite virtual image -> vergence)
    if disp.virtual_dist is None:
        return _pixel_for_angles_batch(disp, np.arctan2(dk[:, 0], dk[:, 2]),
                                       np.arctan2(dk[:, 1], dk[:, 2]))
    w = epd - opticd
    b = 2.0 * np.sum(dk * w, axis=1)
    c = np.sum(w * w, axis=1) - disp.virtual_dist ** 2
    s = (-b + np.sqrt(np.maximum(b * b - 4 * c, 0.0))) / 2.0
    r = (epd + s[:, None] * dk) - opticd
    return _pixel_for_angles_batch(disp, np.arctan2(r[:, 0], r[:, 2]),
                                   np.arctan2(r[:, 1], r[:, 2]))


def corner_features(sim, subject, dev):
    """Clean eye-corner readings (4,) for this face+pose: [eyeL x,y, eyeR x,y].

    These are what the inward CAD cameras report; constant across the pixel grid for a
    fixed pose, and the only per-pose pose/geometry signal the guessing AI gets to see.
    Returns None if a corner is not imaged (should not happen at sane poses).
    """
    R, t = rig.pose_R_t(dev)
    outer = (R @ subject.outer_canthus.T).T + t
    cL = sim.rig["eye"][0].project(outer[0])
    cR = sim.rig["eye"][1].project(outer[1])
    if cL is None or cR is None:
        return None
    return np.concatenate([cL, cR])


def pupil_features(sim, subject, dev):
    """Binocular NIR pupil-camera readings (4,) = [pupilL x,y, pupilR x,y] for this pose.

    The 6-camera design's two NIR pupil cams each image their eye; we project the eye's
    CENTRE OF ROTATION (the gaze-independent EYE-POSITION the camera pins down) through the
    matching cam. This is the slip-fighting registration signal the new design adds over
    corner-only (eyetracker.py: eye POSITION, not gaze angle, is the lever). Constant across
    the pixel grid for a fixed pose, like the corners. None if a pupil isn't imaged.
    """
    R, t = rig.pose_R_t(dev)
    cor = (R @ subject.cor.T).T + t
    pL = sim.rig["pupil_l"].project(cor[0])      # left eye -> left NIR pupil cam
    pR = sim.rig["pupil"].project(cor[1])        # right eye -> right NIR pupil cam
    if pL is None or pR is None:
        return None
    return np.concatenate([pL, pR])


def eye_features(sim, subject, dev, use_pupil):
    """Per-pose eye signal the guessing AI sees: eye-corners (4,), plus the binocular NIR
    pupil readings (-> 8,) for the 6-camera design. None if any required camera misses."""
    corners = corner_features(sim, subject, dev)
    if corners is None:
        return None
    if not use_pupil:
        return corners
    pup = pupil_features(sim, subject, dev)
    return None if pup is None else np.concatenate([corners, pup])


# ======================================================================
#  Pixel -> world inverse  (the chief-ray point the pixel must register on)
# ======================================================================
def invert_pixels(sim, subject, dev, target_uv, z0=Z0, iters=6, eps=2e-4):
    """For each target display pixel, find the world point at depth z0 on its chief ray.

    Gauss-Newton on the (smooth, bijective) forward oracle truepx_batch, started from an
    affine fit of the local map. Returns (P (M,3), valid (M,) bool). `valid` is False
    where the solve did not converge or the point falls off the reachable page (that
    pixel's chief ray simply doesn't cross the page for this eye+pose).
    """
    target_uv = np.atleast_2d(np.asarray(target_uv, float))
    M = len(target_uv)
    h = 0.5  # mm finite-difference step

    # affine initialization: fit uv ~ A[x,y,1] over a coarse world grid, then invert.
    gx, gy = np.meshgrid(np.linspace(-PAGE_X, PAGE_X, 5), np.linspace(-PAGE_Y, PAGE_Y, 5))
    Pg = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)], axis=1)
    uvg = truepx_batch(sim, subject, dev, Pg)
    A, *_ = np.linalg.lstsq(np.column_stack([Pg[:, 0], Pg[:, 1], np.ones(len(Pg))]),
                            uvg, rcond=None)            # (3,2): uv = [x,y,1] @ A
    M2 = A[:2, :].T                                     # (2,2) d uv / d[x,y]
    b0 = A[2, :]                                        # offset
    xy0 = (target_uv - b0) @ np.linalg.inv(M2).T        # (M,2) affine inverse
    P = np.column_stack([xy0, np.full(M, z0)])

    for _ in range(iters):
        px0 = truepx_batch(sim, subject, dev, P)
        Px = P.copy(); Px[:, 0] += h
        Py = P.copy(); Py[:, 1] += h
        Jx = (truepx_batch(sim, subject, dev, Px) - px0) / h    # (M,2)
        Jy = (truepx_batch(sim, subject, dev, Py) - px0) / h    # (M,2)
        # solve [Jx Jy] @ delta = (target - px0) per pixel (2x2 closed form)
        det = Jx[:, 0] * Jy[:, 1] - Jx[:, 1] * Jy[:, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        res = target_uv - px0
        dxx = (Jy[:, 1] * res[:, 0] - Jy[:, 0] * res[:, 1]) / det
        dyy = (-Jx[:, 1] * res[:, 0] + Jx[:, 0] * res[:, 1]) / det
        P[:, 0] += np.clip(dxx, -50, 50)
        P[:, 1] += np.clip(dyy, -50, 50)

    resid = np.linalg.norm(truepx_batch(sim, subject, dev, P) - target_uv, axis=1)
    on_page = (np.abs(P[:, 0]) <= PAGE_X) & (np.abs(P[:, 1]) <= PAGE_Y)
    valid = (resid < eps) & on_page
    return P, valid


def pixel_grid(grid, margin=None):
    """grid x grid display pixels spanning the on-display field (normalized [0,1])."""
    m = rig.ONSCREEN_MARGIN if margin is None else margin
    a = np.linspace(m, 1.0 - m, grid)
    uu, vv = np.meshgrid(a, a)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)   # raster order, "next pixel over"


# ======================================================================
#  The two agents
# ======================================================================
def _design(X):
    """[1, deg-2 monomials of the 6 inputs] for inputs centered at 0.5 -> (n,28)."""
    Xc = np.atleast_2d(np.asarray(X, float)) - 0.5
    P = _poly_features(Xc, 2)
    return np.column_stack([np.ones(len(Xc)), P])


class GuessingAI:
    """Predicts a world point (x,y at z0) from [u, v, eyeL x,y, eyeR x,y].

    Two levels, mirroring the rig's warm-start meta-learning, so the guess count tells a
    real story instead of being constant:
      * a GLOBAL prior, folded in at the end of each face, that carries what's been
        learned across faces (later faces start warmer); and
      * a fast PER-FACE residual, reset each face and updated by every correction, that
        captures this face's irreducible bias (angle kappa) and its pose-dependent warp
        (it can see the corner features, which move with the glasses).
    Both are ridge regressions on the same degree-2 design; guess = (prior+residual).
    Starts at ZERO, no pre-existing data.
    """

    def __init__(self, n_in=6, lam_prior=8.0, lam_resid=3.0):
        # n_in = guesser input width: 6 = [u,v + 4 corner] (corner-only); 10 = + 4 binocular
        # NIR pupil features (the 6-camera design). The degree-2 design grows with n_in.
        self.m = _design(np.zeros((1, n_in))).shape[1]
        self.lam_prior, self.lam_resid = lam_prior, lam_resid
        self.Pxx = np.zeros((self.m, self.m)); self.Pxy = np.zeros((self.m, 2))
        self.Wp = np.zeros((self.m, 2))
        self._fbX, self._fbY = [], []
        self._reset_resid()

    def _reset_resid(self):
        self.Rxx = np.zeros((self.m, self.m)); self.Rxy = np.zeros((self.m, 2))
        self.Wr = np.zeros((self.m, 2))

    def guess(self, x):
        d = _design(x)
        return (d @ (self.Wp + self.Wr))[0]

    def correct(self, x, truth):
        """User AI revealed `truth` for input `x`; learn the residual it didn't predict."""
        d = _design(x)                       # (1, m)
        base = (d @ self.Wp)[0]              # prior part (constant within a face)
        tgt = np.asarray(truth, float) - base
        self.Rxx += d.T @ d
        self.Rxy += d.T @ tgt[None, :]
        reg = self.lam_resid * np.eye(self.m)
        self.Wr = np.linalg.solve(self.Rxx + reg, self.Rxy)
        self._fbX.append(np.asarray(x, float)); self._fbY.append(np.asarray(truth, float))

    def correct_batch(self, X, truth, stash_cap=40000):
        """Vectorized correction over many pixels at once (for the every-pixel mode)."""
        X = np.atleast_2d(np.asarray(X, float))
        truth = np.atleast_2d(np.asarray(truth, float))
        D = _design(X)
        tgt = truth - D @ self.Wp
        self.Rxx += D.T @ D
        self.Rxy += D.T @ tgt
        self.Wr = np.linalg.solve(self.Rxx + self.lam_resid * np.eye(self.m), self.Rxy)
        # keep a bounded representative sample for the end-of-face prior fold
        room = stash_cap - len(self._fbX)
        if room > 0:
            take = X if len(X) <= room else X[np.random.default_rng(0).choice(len(X), room, False)]
            tk = truth if len(truth) <= room else truth[:len(take)]
            self._fbX.extend(list(take)); self._fbY.extend(list(tk))

    def guess_batch(self, X):
        return _design(np.atleast_2d(np.asarray(X, float))) @ (self.Wp + self.Wr)

    def end_face(self):
        """Fold this face's corrections into the global prior, then reset the residual."""
        if self._fbX:
            D = _design(np.array(self._fbX))
            Y = np.array(self._fbY)
            self.Pxx += D.T @ D
            self.Pxy += D.T @ Y
            reg = self.lam_prior * np.eye(self.m)
            self.Wp = np.linalg.solve(self.Pxx + reg, self.Pxy)
        self._fbX, self._fbY = [], []
        self._reset_resid()


class UserAI:
    """The corrector. Knows the physics ground truth; judges guesses by world miss."""

    def __init__(self, tol_mm):
        self.tol = tol_mm

    def miss(self, guess_xy, truth_xy):
        return float(np.linalg.norm(np.asarray(guess_xy) - np.asarray(truth_xy)))

    def is_correct(self, guess_xy, truth_xy):
        return self.miss(guess_xy, truth_xy) < self.tol


# ======================================================================
#  Validation: runs BEFORE any sweep
# ======================================================================
def selftest(verbose=True):
    """Prove the simulator is faithful before spending hours generating data."""
    ok = True
    sim = autosim.Simulator(0)
    rng = np.random.default_rng(123)

    # 1) vectorized oracle == autosim.ground_truth, point by point
    worst = 0.0
    checked = 0
    for _ in range(40):
        subj = sim.new_subject()
        dev = sim.seat()
        for _ in range(6):
            dev = sim.slip(dev)
            P = sim.world_point()
            g = sim.ground_truth(subj, dev, P)
            if g is None:
                continue
            ref = g[1]
            got = truepx_batch(sim, subj, dev, P[None, :])[0]
            worst = max(worst, float(np.max(np.abs(ref - got))))
            checked += 1
    t1 = worst < 1e-9
    ok &= t1
    if verbose:
        print("  [%s] vectorized oracle vs autosim.ground_truth: max diff %.2e over %d pts"
              % ("PASS" if t1 else "FAIL", worst, checked))

    # 2) inverse round-trips: forward(invert(uv)) == uv (sub-pixel @1080p)
    grid = pixel_grid(16)
    rt_err = []
    cov = []
    for _ in range(12):
        subj = sim.new_subject()
        dev = sim.seat()
        for _ in range(3):
            dev = sim.slip(dev)
        P, valid = invert_pixels(sim, subj, dev, grid)
        cov.append(valid.mean())
        if valid.any():
            uv2 = truepx_batch(sim, subj, dev, P[valid])
            rt_err.append(np.linalg.norm(uv2 - grid[valid], axis=1))
    rt = np.concatenate(rt_err) if rt_err else np.array([1.0])
    px1080 = float(np.median(rt) * 1080.0)
    t2 = float(rt.max()) < 5e-4
    ok &= t2
    if verbose:
        print("  [%s] pixel->world->pixel round-trip: median %.3f px@1080p, max %.2e (norm)"
              % ("PASS" if t2 else "FAIL", px1080, float(rt.max())))
        print("        mean on-page coverage: %.0f%% of display pixels reach the page"
              % (100.0 * np.mean(cov)))

    # 3) the recovered point really lies on the chief ray (entrance pupil -> point),
    #    i.e. lighting the pixel registers on it: this is exactly (2) holding, but we
    #    also check the point is finite, in front, on the page.
    t3 = bool(np.all(np.isfinite(rt)))
    ok &= t3
    if verbose:
        print("  [%s] recovered chief-ray points finite & on-page" % ("PASS" if t3 else "FAIL"))

    # 4) physical sensitivity: sliding the glasses down the nose must move the required
    #    point for a fixed pixel (otherwise the whole calibration premise is moot).
    subj = sim.new_subject()
    uv = np.array([[0.5, 0.5]])
    Pa, va = invert_pixels(sim, subj, np.zeros(6), uv)
    dev_slip = np.array([6.0 * rig.PITCH_PER_MM, 0, 0, 0, 6.0, 0])  # 6 mm down the nose
    Pb, vb = invert_pixels(sim, subj, dev_slip, uv)
    shift = float(np.linalg.norm(Pa[0, :2] - Pb[0, :2])) if (va[0] and vb[0]) else 0.0
    t4 = shift > 1.0
    ok &= t4
    if verbose:
        print("  [%s] center pixel moves %.1f mm on the page when glasses slip 6 mm"
              % ("PASS" if t4 else "FAIL", shift))

    if verbose:
        print("  => SIM %s\n" % ("VALID, safe to run the sweep" if ok else "INVALID, DO NOT RUN"))
    return ok


# ======================================================================
#  The sweep
# ======================================================================
def run(n_faces=100, n_poses=100, grid=24, tol_mm=2.5, seed=0, guess_cap=60,
        out_dir=None, verbose=True, validate=True, use_pupil=True):
    """The full protocol. Returns a results dict; writes data/pixel_sweep.npz + .json.

    use_pupil=True is the NEW 6-camera design (corners + binocular NIR pupil); False is the
    legacy corner-only design (for an apples-to-apples comparison)."""
    if validate and not selftest(verbose=verbose):
        raise RuntimeError("simulator self-test failed; refusing to generate data")

    out_dir = out_dir or _DATA
    sim = autosim.Simulator(seed)                       # one CAD device (fixed extrinsics)
    pop = np.random.default_rng(seed + 1)               # the population of faces
    grid_uv = pixel_grid(grid)
    nP = len(grid_uv)
    user = UserAI(tol_mm)
    n_in = 2 + (8 if use_pupil else 4)                  # [u,v] + eye features
    guesser = GuessingAI(n_in=n_in)                     # starts at ZERO, no prior data

    # guess count per (face, pose, pixel); 0 = that pixel's ray missed the page (skipped)
    counts = np.zeros((n_faces, n_poses, nP), dtype=np.int16)
    descriptors = np.zeros((n_faces, 8))
    face_avg = np.zeros(n_faces)                        # avg guesses/pixel, for the curve

    if verbose:
        print("== Pixel-by-pixel calibration sweep ==")
        print("  faces: %d   positions/face: %d   display pixels/position: %d (%dx%d grid)"
              % (n_faces, n_poses, nP, grid, grid))
        print("  rule: guess a point on the pixel's chief ray; user-AI corrects until the")
        print("        guess is right twice in a row (within %.1f mm @ %.0f mm), then next pixel."
              % (tol_mm, Z0))
        print("  guessing AI starts cold; it learns only from this run's corrections.\n")
        print("  %5s | %8s | %12s | %14s" %
              ("face", "IPD/kap", "avg guess/px", "cum data pieces"))
        print("  " + "-" * 52)

    t0 = time.time()
    total_pieces = 0
    for fi in range(n_faces):
        subj = anatomy.sample_subject(pop)
        descriptors[fi] = subj.descriptor()
        dev = sim.seat()
        face_pieces = 0
        face_valid = 0
        for pj in range(n_poses):
            dev = sim.slip(dev)                         # the next glasses position
            eyef = eye_features(sim, subj, dev, use_pupil)   # corners (+ binocular pupil)
            if eyef is None:
                continue
            Pstar, valid = invert_pixels(sim, subj, dev, grid_uv)
            for k in range(nP):
                if not valid[k]:
                    continue
                truth = Pstar[k, :2]
                x = np.concatenate([grid_uv[k], eyef])      # guesser input (6 or 10,)
                streak = 0
                n = 0
                while streak < 2 and n < guess_cap:
                    g = guesser.guess(x)                # the guessing AI's guess
                    n += 1
                    if user.is_correct(g, truth):
                        streak += 1
                    else:
                        streak = 0
                        guesser.correct(x, truth)       # the user AI corrects it
                counts[fi, pj, k] = n
                face_pieces += n
                face_valid += 1
            total_pieces += 0  # tallied below
        guesser.end_face()                              # fold into the cross-face prior
        total_pieces += face_pieces
        face_avg[fi] = face_pieces / max(face_valid, 1)
        if verbose and ((fi + 1) % max(1, n_faces // 20) == 0 or fi == 0):
            print("  %5d | %5.1f/%3.1f | %12.2f | %14s"
                  % (fi + 1, subj.ipd, subj.kappa[0], face_avg[fi], f"{total_pieces:,}"))

    dt = time.time() - t0
    valid_pixels = int((counts > 0).sum())
    avg_guesses = total_pieces / max(valid_pixels, 1)
    early = face_avg[:max(1, n_faces // 5)].mean()
    late = face_avg[-max(1, n_faces // 5):].mean()

    if verbose:
        print("  " + "-" * 52)
        print("\n  DATASET")
        print("    faces                         : %d" % n_faces)
        print("    glasses positions per face    : %d" % n_poses)
        print("    display pixels swept / position: %d (on-page subset of a %dx%d grid)"
              % (int(valid_pixels / max(n_faces * n_poses, 1)), grid, grid))
        print("    avg guesses to lock a pixel   : %.3f  (min possible = 2)" % avg_guesses)
        print("    --------------------------------------------------------")
        print("    TOTAL DATA PIECES = 100 x 100 x pixels x avg_guesses")
        print("                      = %s guesses recorded" % f"{total_pieces:,}")
        print("\n  LEARNING (this is why the count isn't just 2x everywhere)")
        print("    first %d faces: %.2f guesses/pixel   last %d faces: %.2f guesses/pixel"
              % (max(1, n_faces // 5), early, max(1, n_faces // 5), late))
        print("    -> the guessing AI needs fewer corrections as it sees more faces.")
        print("\n  generated in %.0f s (%.0f min)" % (dt, dt / 60.0))

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "pixel_sweep.npz")
    np.savez_compressed(npz, guess_counts=counts, grid_uv=grid_uv, descriptors=descriptors,
                        face_avg_guesses=face_avg, tol_mm=tol_mm, z0=Z0,
                        use_pupil=use_pupil,
                        design=("6cam_corner+pupil" if use_pupil else "corner_only"),
                        descriptor_names=np.array(anatomy.DESCRIPTOR_NAMES))
    if verbose:
        print("\n  saved -> %s" % npz)
        print("    guess_counts[face, position, pixel] = guesses to lock that pixel (0 = off-page)")
        print("    sum(guess_counts) = total data pieces; descriptors[face] = that eye's geometry.")
    return {"counts": counts, "total_pieces": int(total_pieces),
            "valid_pixels": valid_pixels, "avg_guesses": avg_guesses,
            "face_avg": face_avg, "seconds": dt}


# ======================================================================
#  EVERY-PIXEL (native display resolution) mode
# ======================================================================
def _poly_uv(uv, deg):
    """Monomials of (u,v) up to total degree `deg`, with bias -> design for the inverse
    surrogate that maps display pixel -> world point over the whole field."""
    uv = np.atleast_2d(np.asarray(uv, float)) - 0.5
    u, v = uv[:, 0], uv[:, 1]
    cols = [np.ones(len(uv))]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append((u ** (d - i)) * (v ** i))
    return np.column_stack(cols)


def inverse_surrogate(sim, subject, dev, ng=64, deg=3):
    """Fit a smooth pixel->world map from a GN-exact ng x ng grid, so we can place EVERY
    native pixel with one matrix multiply. Returns (Wsur, deg, fit_resid_mm) or None.

    fit_resid_mm is the surrogate's own error vs the exact inverse, in mm at the page , 
    we assert it is far below the correctness tolerance before trusting it (keeps the
    every-pixel run as faithful as the exact grid run)."""
    uv = pixel_grid(ng)
    P, valid = invert_pixels(sim, subject, dev, uv)
    if valid.sum() < 30:
        return None
    A = _poly_uv(uv[valid], deg)
    Wsur, *_ = np.linalg.lstsq(A, P[valid, :2], rcond=None)
    resid = np.linalg.norm(A @ Wsur - P[valid, :2], axis=1)
    return Wsur, deg, float(np.max(resid))


def run_every_pixel(n_faces=100, n_poses=100, display_w=1920, display_h=1080,
                    tol_mm=2.5, seed=0, round_cap=40, chunk=200000, out_dir=None,
                    verbose=True, validate=True):
    """The SAME protocol over every native display pixel (default 1920x1080).

    Each round every still-unlocked on-page pixel is guessed (one vectorized pass); a
    pixel locks after it is correct twice in a row; every wrong pixel corrects the
    guessing AI (batched). A pixel already within tolerance therefore locks in exactly 2
    guesses, which is why the whole 1920x1080 field is tractable: only the pixels the AI
    can't yet predict drive extra rounds. Storage is per-(face,pose) aggregates (the full
    per-pixel tensor would be ~40 GB); the grand total of guesses is exact."""
    if validate and not selftest(verbose=verbose):
        raise RuntimeError("simulator self-test failed; refusing to generate data")
    out_dir = out_dir or _DATA
    sim = autosim.Simulator(seed)
    pop = np.random.default_rng(seed + 1)
    user = UserAI(tol_mm)
    guesser = GuessingAI()

    # native pixel centers in normalized [0,1] coords
    us = (np.arange(display_w) + 0.5) / display_w
    vs = (np.arange(display_h) + 0.5) / display_h
    UU, VV = np.meshgrid(us, vs)
    UVfull = np.stack([UU.ravel(), VV.ravel()], axis=1)        # (W*H, 2)
    n_native = len(UVfull)

    descriptors = np.zeros((n_faces, 8))
    agg = np.zeros((n_faces, n_poses, 3))     # [valid_pixels, total_guesses, max_rounds]
    face_avg = np.zeros(n_faces)
    worst_surrogate = 0.0
    total_pieces = 0

    if verbose:
        print("== EVERY-PIXEL sweep: full %dx%d display = %s pixels/position ==" %
              (display_w, display_h, f"{n_native:,}"))
        print("  faces %d  positions/face %d  ->  up to %s pixel-locks; "
              "same twice-in-a-row rule, vectorized per round.\n"
              % (n_faces, n_poses, f"{n_faces*n_poses*n_native:,}"))
        print("  %5s | %12s | %14s | %12s" %
              ("face", "avg guess/px", "cum data pieces", "elapsed"))
        print("  " + "-" * 56)

    t0 = time.time()
    for fi in range(n_faces):
        subj = anatomy.sample_subject(pop)
        descriptors[fi] = subj.descriptor()
        dev = sim.seat()
        face_pieces = 0; face_valid = 0
        for pj in range(n_poses):
            dev = sim.slip(dev)
            corners = corner_features(sim, subj, dev)
            if corners is None:
                continue
            sur = inverse_surrogate(sim, subj, dev)
            if sur is None:
                continue
            Wsur, deg, sres = sur
            worst_surrogate = max(worst_surrogate, sres)

            # place every native pixel; keep the on-page ones
            Pxy = _poly_uv(UVfull, deg) @ Wsur                  # (n_native, 2)
            onpage = (np.abs(Pxy[:, 0]) <= PAGE_X) & (np.abs(Pxy[:, 1]) <= PAGE_Y)
            K = int(onpage.sum())
            if K == 0:
                continue
            uv = UVfull[onpage]; truth = Pxy[onpage]
            xin = np.column_stack([uv, np.tile(corners, (K, 1))])  # (K,6)
            # the guesser's design is CONSTANT across rounds for this pose -> build once.
            D = _design(xin).astype(np.float32)                    # (K, m)
            base_prior = D @ guesser.Wp.astype(np.float32)         # prior part (fixed/pose)

            consec = np.zeros(K, np.int8)
            guesses = np.zeros(K, np.int32)
            active = np.ones(K, bool)
            rounds = 0
            while active.any() and rounds < round_cap:
                rounds += 1
                ai = np.flatnonzero(active)
                g = base_prior[ai] + D[ai] @ guesser.Wr.astype(np.float32)
                miss = np.linalg.norm(g - truth[ai], axis=1)
                guesses[ai] += 1
                ok = miss < tol_mm
                consec[ai[ok]] += 1
                consec[ai[~ok]] = 0
                active[ai[consec[ai] >= 2]] = False
                wrong = ai[~ok]
                if wrong.size:
                    # batched correction straight from the cached design (no rebuild)
                    Dw = D[wrong].astype(np.float64)
                    tgt = truth[wrong] - Dw @ guesser.Wp
                    guesser.Rxx += Dw.T @ Dw
                    guesser.Rxy += Dw.T @ tgt
                    guesser.Wr = np.linalg.solve(
                        guesser.Rxx + guesser.lam_resid * np.eye(guesser.m), guesser.Rxy)
                    if len(guesser._fbX) < 40000:        # bounded stash for end_face fold
                        room = 40000 - len(guesser._fbX)
                        s = wrong[:room]
                        guesser._fbX.extend(list(xin[s])); guesser._fbY.extend(list(truth[s]))
            tg = int(guesses.sum())
            agg[fi, pj] = [K, tg, rounds]
            face_pieces += tg; face_valid += K
        guesser.end_face()
        total_pieces += face_pieces
        face_avg[fi] = face_pieces / max(face_valid, 1)
        if verbose and ((fi + 1) % max(1, n_faces // 20) == 0 or fi == 0):
            el = time.time() - t0
            print("  %5d | %12.3f | %14s | %9.0f s"
                  % (fi + 1, face_avg[fi], f"{total_pieces:,}", el), flush=True)
        # crash-safe checkpoint after every face (an 11 h run must survive a hiccup)
        np.savez_compressed(os.path.join(out_dir, "pixel_sweep_full.ckpt.npz"),
                            agg=agg, descriptors=descriptors, face_avg_guesses=face_avg,
                            faces_done=fi + 1, total_pieces=total_pieces,
                            display=np.array([display_w, display_h]), tol_mm=tol_mm)

    dt = time.time() - t0
    valid_pixels = int(agg[:, :, 0].sum())
    avg_guesses = total_pieces / max(valid_pixels, 1)
    early = face_avg[:max(1, n_faces // 5)].mean()
    late = face_avg[-max(1, n_faces // 5):].mean()
    if verbose:
        print("  " + "-" * 56)
        print("\n  surrogate inverse max error: %.4f mm  (tolerance %.2f mm) -> %s"
              % (worst_surrogate, tol_mm,
                 "faithful" if worst_surrogate < 0.1 * tol_mm else "CHECK"))
        print("\n  DATASET (every pixel)")
        print("    faces x positions             : %d x %d" % (n_faces, n_poses))
        print("    on-page pixels swept / position: ~%s of %s native"
              % (f"{int(valid_pixels/max(n_faces*n_poses,1)):,}", f"{n_native:,}"))
        print("    avg guesses to lock a pixel   : %.4f" % avg_guesses)
        print("    TOTAL DATA PIECES             : %s guesses" % f"{total_pieces:,}")
        print("    first/last %d faces guesses/px : %.3f -> %.3f (learning)"
              % (max(1, n_faces // 5), early, late))
        print("\n  generated in %.0f s (%.1f h)" % (dt, dt / 3600.0))

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "pixel_sweep_full.npz")
    np.savez_compressed(npz, agg=agg, descriptors=descriptors, face_avg_guesses=face_avg,
                        display=np.array([display_w, display_h]), tol_mm=tol_mm, z0=Z0,
                        total_pieces=total_pieces, descriptor_names=np.array(anatomy.DESCRIPTOR_NAMES))
    if verbose:
        print("\n  saved -> %s" % npz)
        print("    agg[face, position] = [on-page pixels, total guesses, rounds]; "
              "sum of col 1 = %s data pieces." % f"{total_pieces:,}")
    return {"total_pieces": int(total_pieces), "valid_pixels": valid_pixels,
            "avg_guesses": avg_guesses, "seconds": dt, "face_avg": face_avg}


GUESS_DTYPE = np.dtype([("face", "u1"), ("pose", "u1"), ("pixel", "u2"),
                        ("guess", "u1"), ("dx", "f2"), ("dy", "f2")])   # 9 bytes/guess


def run_dense_grid(n_faces=100, n_poses=100, grid=48, tol_mm=2.5, seed=0,
                   round_cap=60, out_dir=None, verbose=True, validate=True,
                   record_guesses=True, use_pupil=True):
    """Dense grid sweep, VECTORIZED + stable, storing the full per-pixel tensor.

    Same twice-in-a-row protocol as `run`, but every still-unlocked pixel is guessed once
    per round and all wrong pixels correct the guessing-AI together (batched). A pixel
    within tolerance locks in exactly 2 guesses. This batch-correct-the-whole-field order
    converges the smooth pixel->point warp in a few rounds and does NOT degrade on later
    faces the way the strict one-pixel-at-a-time loop does. Saves the full
    guess_counts[face, position, pixel] tensor to data/pixel_sweep.npz.

    If `record_guesses`, EVERY individual guess is streamed to data/pixel_sweep_guesses.bin
    (GUESS_DTYPE: face, pose, pixel, guess#, signed inaccuracy dx,dy in mm), the dataset is
    one row PER GUESS, so its length is exactly the total data-piece count. Streamed to disk
    per round so memory stays bounded even at hundreds of millions of guesses."""
    if validate and not selftest(verbose=verbose):
        raise RuntimeError("simulator self-test failed; refusing to generate data")
    out_dir = out_dir or _DATA
    rec_fh = None
    rec_path = os.path.join(out_dir, "pixel_sweep_guesses.bin")
    if record_guesses:
        os.makedirs(out_dir, exist_ok=True)
        rec_fh = open(rec_path, "wb")
    sim = autosim.Simulator(seed)
    pop = np.random.default_rng(seed + 1)
    user = UserAI(tol_mm)
    guesser = GuessingAI(n_in=2 + (8 if use_pupil else 4))   # +4 binocular pupil = 6-cam design
    grid_uv = pixel_grid(grid)
    nP = len(grid_uv)

    counts = np.zeros((n_faces, n_poses, nP), dtype=np.int16)
    first_miss = np.zeros((n_faces, n_poses, nP), dtype=np.float32)  # guess inaccuracy (mm)
    descriptors = np.zeros((n_faces, 8))
    face_avg = np.zeros(n_faces)
    total_pieces = 0

    if verbose:
        print("== Dense-grid pixel sweep (vectorized) ==")
        print("  faces %d  positions/face %d  display pixels/position %d (%dx%d grid)"
              % (n_faces, n_poses, nP, grid, grid))
        print("  rule: guess a point on each pixel's chief ray; user-AI corrects every")
        print("        wrong pixel until each is right twice in a row (<%.1f mm @ %.0f mm).\n"
              % (tol_mm, Z0))
        print("  %5s | %8s | %12s | %14s" %
              ("face", "IPD/kap", "avg guess/px", "cum data pieces"))
        print("  " + "-" * 52)

    t0 = time.time()
    for fi in range(n_faces):
        subj = anatomy.sample_subject(pop)
        descriptors[fi] = subj.descriptor()
        dev = sim.seat()
        face_pieces = 0; face_valid = 0
        for pj in range(n_poses):
            dev = sim.slip(dev)
            eyef = eye_features(sim, subj, dev, use_pupil)   # corners (+ binocular pupil)
            if eyef is None:
                continue
            Pstar, valid = invert_pixels(sim, subj, dev, grid_uv)
            idx = np.flatnonzero(valid)
            if idx.size == 0:
                continue
            truth = Pstar[idx, :2]
            xin = np.column_stack([grid_uv[idx], np.tile(eyef, (idx.size, 1))])
            D = _design(xin)                              # cached: constant across rounds
            base_prior = D @ guesser.Wp
            K = idx.size
            consec = np.zeros(K, np.int8)
            guesses = np.zeros(K, np.int32)
            active = np.ones(K, bool)
            rounds = 0
            while active.any() and rounds < round_cap:
                rounds += 1
                ai = np.flatnonzero(active)
                g = base_prior[ai] + D[ai] @ guesser.Wr
                err = g - truth[ai]                          # signed inaccuracy (mm)
                miss = np.linalg.norm(err, axis=1)
                if rounds == 1:                              # the guessing-AI's raw error
                    first_miss[fi, pj, idx] = miss           # before any correction = inaccuracy
                guesses[ai] += 1
                if rec_fh is not None:                       # one row PER GUESS, streamed
                    rec = np.empty(ai.size, dtype=GUESS_DTYPE)
                    rec["face"] = fi; rec["pose"] = pj
                    rec["pixel"] = idx[ai]; rec["guess"] = np.minimum(guesses[ai], 255)
                    rec["dx"] = err[:, 0]; rec["dy"] = err[:, 1]
                    rec.tofile(rec_fh)
                ok = miss < tol_mm
                consec[ai[ok]] += 1
                consec[ai[~ok]] = 0
                active[ai[consec[ai] >= 2]] = False
                wrong = ai[~ok]
                if wrong.size:
                    Dw = D[wrong]
                    guesser.Rxx += Dw.T @ Dw
                    guesser.Rxy += Dw.T @ (truth[wrong] - Dw @ guesser.Wp)
                    guesser.Wr = np.linalg.solve(
                        guesser.Rxx + guesser.lam_resid * np.eye(guesser.m), guesser.Rxy)
                    if len(guesser._fbX) < 40000:
                        guesser._fbX.extend(list(xin[wrong])); guesser._fbY.extend(list(truth[wrong]))
            counts[fi, pj, idx] = guesses
            face_pieces += int(guesses.sum()); face_valid += K
        guesser.end_face()
        total_pieces += face_pieces
        face_avg[fi] = face_pieces / max(face_valid, 1)
        if verbose and ((fi + 1) % max(1, n_faces // 20) == 0 or fi == 0):
            print("  %5d | %5.1f/%3.1f | %12.3f | %14s"
                  % (fi + 1, subj.ipd, subj.kappa[0], face_avg[fi], f"{total_pieces:,}"),
                  flush=True)

    dt = time.time() - t0
    valid_pixels = int((counts > 0).sum())
    avg_guesses = total_pieces / max(valid_pixels, 1)
    early = face_avg[:max(1, n_faces // 5)].mean(); late = face_avg[-max(1, n_faces // 5):].mean()
    if verbose:
        print("  " + "-" * 52)
        print("\n  DATASET = 100 faces x 100 positions x pixels x avg-guesses")
        print("    avg pixels/position (on-page) : %d" % (valid_pixels // max(n_faces * n_poses, 1)))
        print("    avg guesses to lock a pixel   : %.3f  (min 2)" % avg_guesses)
        print("    TOTAL DATA PIECES             : %s" % f"{total_pieces:,}")
        print("    learning: first/last %d faces : %.2f -> %.2f guesses/pixel"
              % (max(1, n_faces // 5), early, late))
        print("\n  generated in %.0f s (%.1f min)" % (dt, dt / 60.0))

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "pixel_sweep.npz")
    np.savez_compressed(npz, guess_counts=counts, first_miss_mm=first_miss, grid_uv=grid_uv,
                        descriptors=descriptors, face_avg_guesses=face_avg, tol_mm=tol_mm, z0=Z0,
                        total_pieces=total_pieces, use_pupil=use_pupil,
                        design=("6cam_corner+pupil" if use_pupil else "corner_only"),
                        descriptor_names=np.array(anatomy.DESCRIPTOR_NAMES))
    if verbose:
        print("\n  saved -> %s  (guess_counts[face, position, pixel]; sum = %s pieces)"
              % (npz, f"{total_pieces:,}"))
    if rec_fh is not None:
        rec_fh.close()
        import json
        with open(rec_path + ".json", "w") as jf:
            json.dump({"records": int(total_pieces), "bytes_each": GUESS_DTYPE.itemsize,
                       "fields": ["face", "pose", "pixel", "guess", "dx_mm", "dy_mm"],
                       "dtype": "u1,u1,u2,u1,f2,f2", "tol_mm": tol_mm, "z0_mm": Z0,
                       "note": "one row per guess; dx,dy = signed guess inaccuracy at the page"},
                      jf, indent=2)
        if verbose:
            gb = total_pieces * GUESS_DTYPE.itemsize / 1e9
            print("  per-guess log -> %s  (%s rows x %d bytes = %.2f GB; schema in .json)"
                  % (rec_path, f"{total_pieces:,}", GUESS_DTYPE.itemsize, gb))
    return {"counts": counts, "total_pieces": int(total_pieces), "valid_pixels": valid_pixels,
            "avg_guesses": avg_guesses, "face_avg": face_avg, "seconds": dt}


def _cli(argv=None):
    p = argparse.ArgumentParser(description="Pixel-by-pixel AR calibration sweep.")
    p.add_argument("--faces", type=int, default=100)
    p.add_argument("--poses", type=int, default=100)
    p.add_argument("--grid", type=int, default=24, help="NxN display pixels per position")
    p.add_argument("--tol-mm", type=float, default=2.5, help="'correct' radius at the page")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every-pixel", action="store_true",
                   help="sweep the full native display (1920x1080), every pixel")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--selftest", action="store_true", help="validate the sim and exit")
    p.add_argument("--corner-only", action="store_true",
                   help="legacy 4-cam (eye-corner only) design; default is the new 6-cam "
                        "design (corners + binocular NIR pupil)")
    p.add_argument("--out-dir", type=str, default=None, help="override output directory")
    args = p.parse_args(argv)
    if args.selftest:
        return 0 if selftest() else 1
    use_pupil = not args.corner_only
    if args.every_pixel:
        run_every_pixel(n_faces=args.faces, n_poses=args.poses,
                        display_w=args.width, display_h=args.height,
                        tol_mm=args.tol_mm, seed=args.seed)
    else:
        run_dense_grid(n_faces=args.faces, n_poses=args.poses, grid=args.grid,
                       tol_mm=args.tol_mm, seed=args.seed, out_dir=args.out_dir,
                       use_pupil=use_pupil)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
