"""reseat.py, did the glasses actually MOVE during the run, and does knowing that help?

THE PROBLEM THIS EXISTS TO STOP US REPEATING.

On 2026-08-03 the eye-corner correction (`geometry.eye_offset_mm`, `EYE_SHIFT_GAIN`) was built,
guarded, and swept against the 17 real calibration samples. Every gain lost to leaving it off:

    gain 0.00 -> 26.9 px (best)      gain 0.50 -> 27.3 px      gain 1.00 -> 27.6 px

It would have been easy to write that down as "the eye-corner correction does not help". It is not
what the data says. The eye-feature spread across those 17 samples was sd 0.032 / 0.0085 of frame
-- THE GLASSES NEVER MOVED. A correction for how the glasses sit on the face cannot be shown to
help by a run in which how the glasses sit on the face was constant; all it could contribute was
its own tracking noise, so of course it lost.

So the sweep was not a null result. It was an UNDERPOWERED one, and the two look identical if you
only report the winning gain. This module refuses to let them look identical:

  1. It measures whether the seat ACTUALLY varied -- between-seat vs within-seat spread of the
     eye features, in mm of eye displacement, which is the unit the correction works in.
  2. It measures the EFFECT SIZE the correction can possibly have on THIS dataset, in display
     pixels, before looking at whether it wins. If turning the term on moves the overlay by less
     than a pixel across every stored sample, no sweep over that data can decide anything.
  3. Only then does it sweep, and it reports a paired bootstrap interval rather than a bare
     argmin, so "0.5 beat 0.0 by 0.4 px" cannot be mistaken for a result.

The verdict is one of three: SET IT, LEAVE IT OFF, or NOT ENOUGH SEAT VARIATION TO DECIDE. The
third is the one the old sweep could not say.

    python3 reseat.py --check                 # the real samples in data/samples.db
    python3 reseat.py --check --joint         # also re-fit DISPLAY_FOV at each gain (fairer)
    python3 reseat.py --selftest              # positive AND negative controls

HOW TO PRODUCE DATA THIS CAN DECIDE ON:
    python3 main.py ... --reseat-every 8      # take the glasses off and back on every 8 samples
"""
import argparse
import os
import sys

import numpy as np

import geometry
import rig

DISPLAY_H = 1080.0          # the project's px convention: normalised error x 1080
EYE = slice(4, 8)           # eyeL_u, eyeL_v, eyeR_u, eyeR_v

# The 2026-08-03 run, kept as the reference for "the glasses did not move".
BASELINE_EYE_SD = (0.032, 0.0085)



# ---------------------------------------------------------------------------
#  measurement
# ---------------------------------------------------------------------------
def shifts_mm(X):
    """Per-sample eye displacement from nominal, in mm, at UNIT gain.

    This is the raw measurement the correction is built on, with `EYE_SHIFT_GAIN` divided back
    out, so the spread reported here does not depend on whether the term is currently switched on.
    """
    X = np.atleast_2d(np.asarray(X, float))
    g0 = geometry.EYE_SHIFT_GAIN
    try:
        geometry.EYE_SHIFT_GAIN = 1.0
        return np.array([geometry.eye_offset_mm(x)[:2] for x in X], float)
    finally:
        geometry.EYE_SHIFT_GAIN = g0


def seat_report(X, seats):
    """Between-seat vs within-seat spread. Did the re-seating actually happen?

    Between-seat spread is the signal the correction feeds on; within-seat spread is head motion,
    facial movement and tracker noise. If between is not clearly larger than within, the run did
    not exercise the thing under test -- whatever the sweep then says.
    """
    X = np.atleast_2d(np.asarray(X, float))
    seats = np.asarray(seats, int).ravel()
    mm = shifts_mm(X)
    uniq = np.unique(seats)

    # per-seat means of the eye displacement, and the pooled within-seat scatter
    means = np.array([mm[seats == s].mean(axis=0) for s in uniq])
    within = []
    for s in uniq:
        m = mm[seats == s]
        if len(m) >= 2:
            within.append(m - m.mean(axis=0))
    within = np.vstack(within) if within else np.zeros((0, 2))

    between_sd = means.std(axis=0, ddof=1) if len(uniq) >= 2 else np.zeros(2)
    within_sd = within.std(axis=0, ddof=1) if len(within) >= 2 else np.zeros(2)
    eye_sd = X[:, EYE].std(axis=0, ddof=1) if len(X) >= 2 else np.zeros(4)

    # Spread of the seat means, as a single number in mm: the range the correction had to work
    # across. This is what "the glasses moved" means quantitatively.
    span_mm = float(np.hypot(*(means.max(axis=0) - means.min(axis=0)))) if len(uniq) >= 2 else 0.0

    return {
        "n": int(len(X)),
        "n_seats": int(len(uniq)),
        "per_seat_n": {int(s): int(np.sum(seats == s)) for s in uniq},
        "seat_means_mm": means,
        "between_sd_mm": between_sd,
        "within_sd_mm": within_sd,
        "span_mm": span_mm,
        "eye_feature_sd": eye_sd,
    }


def _pixels(X, gain, fov=None):
    """Geometry-only overlay pixels for every sample at a given eye-shift gain."""
    X = np.atleast_2d(np.asarray(X, float))
    fov = geometry.DISPLAY_FOV_DEG if fov is None else fov
    g0 = geometry.EYE_SHIFT_GAIN
    try:
        geometry.EYE_SHIFT_GAIN = float(gain)
        return np.array([geometry.geometric_pixel(x, display_fov_deg=fov) for x in X], float)
    finally:
        geometry.EYE_SHIFT_GAIN = g0


def effect_px(X, gain=1.0):
    """How far turning the term ON moves the overlay, over THIS dataset, in display px.

    The power question, asked before the accuracy question. A term that moves the overlay by
    0.2 px cannot be validated or refuted by samples whose residual is 27 px -- and reporting it
    as "does not help" would be a claim the data cannot support.
    """
    d = np.linalg.norm(_pixels(X, gain) - _pixels(X, 0.0), axis=1) * DISPLAY_H
    return float(np.median(d)), float(np.max(d))


def errors_px(X, Y, gain, fov=None):
    """Per-sample overlay error against the approved pixels, in display px.

    The approved pixels ARE ground truth: each was nudged by hand until the overlay sat on the
    target.
    """
    p = _pixels(X, gain, fov)
    return np.linalg.norm(p - np.atleast_2d(np.asarray(Y, float)), axis=1) * DISPLAY_H


def _refit_fov(X, Y, gain, lo=42.0, hi=56.0, steps=57):
    """Re-fit DISPLAY_FOV_DEG at this gain. Returns (fov, median px).

    DISPLAY_FOV_DEG is an EFFECTIVE scale that absorbs whatever else is mis-scaled (see
    geometry.py), and it was fitted with the eye term OFF. Sweeping the gain against that frozen
    fit charges the eye term for any change in mean bias that the scale would have absorbed --
    which biases the comparison towards the status quo. Re-fitting both is the fair test.
    """
    grid = np.linspace(lo, hi, steps)
    med = [float(np.median(errors_px(X, Y, gain, f))) for f in grid]
    i = int(np.argmin(med))
    return float(grid[i]), med[i]


def estimate_gain(X, Y, nuisance=True):
    """THE ESTIMATOR. Fit the eye-shift gain directly, as a regression coefficient.

    Sweeping a grid of gains and comparing MEDIAN ERRORS is the obvious approach and it is far too
    blunt for this term. Errors are unsigned and dominated by a ~27 px systematic; a 3 px
    correction moves the median by a fraction of a pixel, so the sweep needs thousands of samples
    to see something a regression sees in dozens. Measured in the controls below: the grid sweep
    cannot recover a gain of 1.0 that was PLANTED in 40 samples; this estimator recovers it.

    The model, per sample, in display px:

        Y - pixel(gain=0)  =  A . [1, world_u, world_v]  +  g . (pixel(gain=1) - pixel(gain=0))

    The first term is the nuisance: every registration error that is a smooth function of where
    the target is -- display-FOV scale error, distortion, kappa, world-camera error. That is
    exactly the ~27 px this project has been unable to see past. It is absorbed here rather than
    fought, with its own coefficients for u and for v.

    `g` is then identified ONLY by variation that the direction basis cannot explain -- i.e. by
    two samples pointing the same way whose eye features differ, which is precisely what
    re-seating the glasses creates and what a single-seating run does not contain.

    WHY THIS IS NOT CHEATING: g is a single shared scalar across both axes and all samples, and
    the shift regressor is a MEASUREMENT (mm from the eye cameras through fixed optics), not a
    free fit. The nuisance basis is affine, so it cannot mimic the shift's sample-to-sample
    structure unless the seat is aliased with direction -- which `collinearity` below detects.

    Returns dict(gain, se, t, n, collinearity, sigma_px).
    """
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    p0 = _pixels(X, 0.0)
    p1 = _pixels(X, 1.0)
    resid = (Y - p0) * DISPLAY_H                      # px
    shift = (p1 - p0) * DISPLAY_H                     # px, the regressor
    n = len(X)
    wu = (X[:, 0] + X[:, 2]) / 2.0                    # direction proxy: mean world-dot position
    wv = (X[:, 1] + X[:, 3]) / 2.0

    # stack the u and v components into one system so a single g serves both
    rows, ys = [], []
    for i in range(n):
        base = [1.0, wu[i], wv[i]] if nuisance else [1.0]
        k = len(base)
        for c in (0, 1):
            r = np.zeros(2 * k + 1)
            r[c * k:(c + 1) * k] = base               # per-component nuisance
            r[-1] = shift[i, c]                       # shared gain
            rows.append(r)
            ys.append(resid[i, c])
    A = np.array(rows)
    b = np.array(ys)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    g = float(coef[-1])

    dof = max(len(b) - A.shape[1], 1)
    sigma = float(np.sqrt(np.sum((b - A @ coef) ** 2) / dof))
    cov = np.linalg.pinv(A.T @ A) * sigma ** 2
    se = float(np.sqrt(max(cov[-1, -1], 0.0)))

    # COLLINEARITY: how much of the shift regressor survives after the nuisance basis has taken
    # what it can? If the seat happened to change only when the target also moved, the two are
    # aliased and no amount of data separates them. 1.0 = fully independent, 0 = fully explained.
    s = A[:, -1]
    N = A[:, :-1]
    resid_s = s - N @ np.linalg.lstsq(N, s, rcond=None)[0]
    denom = float(np.sum(s ** 2))
    collin = float(np.sum(resid_s ** 2) / denom) if denom > 0 else 0.0

    return {"gain": g, "se": se, "t": (g / se if se > 0 else 0.0), "n": n,
            "collinearity": collin, "sigma_px": sigma}


GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def sweep(X, Y, gains=GAINS, joint=False):
    rows = []
    for g in gains:
        e = errors_px(X, Y, g)
        row = {"gain": float(g), "median": float(np.median(e)),
               "p95": float(np.percentile(e, 95))}
        if joint:
            row["fov"], row["median_refit"] = _refit_fov(X, Y, g)
        rows.append(row)
    return rows


# A gain estimate is only worth acting on if its standard error is small enough to tell a real
# correction from none at all. With SE 0.5 you cannot distinguish gain 1.0 from gain 0.0 at two
# sigma, and setting a physically-derived constant on a coin flip is how an unvalidated term gets
# switched on -- the thing EYE_SHIFT_GAIN = 0.0 was protecting against in the first place.
MAX_USABLE_SE = 0.35
# Below this, the seat regressor is so tangled with target direction that the fit is degenerate
# rather than merely imprecise.
#
# DELIBERATELY LOW, and the reason matters. Measured over 8 seeds per design: mixed targets give
# independence 0.77 and gain 0.94 +- 0.06; one target per seating gives 0.24 and 0.27; three
# seatings with one target each gives 0.01 and 1.21. Aliasing does NOT bias the estimate -- it
# inflates its variance, and the standard error already measures that, so SE is the real gate and
# this is the diagnostic that says WHY the SE is large and what to change in the protocol. Gating
# on collinearity as well would just reject the same runs twice, in a less principled unit.
MIN_COLLINEARITY = 0.10


def verdict(X, Y, seats, joint=False, gains=GAINS):
    """The whole decision, as text plus a machine-readable dict."""
    rep = seat_report(X, seats)
    eff_med, eff_max = effect_px(X, 1.0)
    rows = sweep(X, Y, gains, joint)
    key = "median_refit" if joint else "median"
    best = min(rows, key=lambda r: r[key])
    est = estimate_gain(X, Y)

    moved = rep["n_seats"] >= 2 and rep["span_mm"] >= 1.0 and (
        np.max(rep["between_sd_mm"]) > np.max(rep["within_sd_mm"]))
    aliased = est["collinearity"] < MIN_COLLINEARITY
    powered = est["se"] <= MAX_USABLE_SE and not aliased
    g, se = est["gain"], est["se"]

    if not powered:
        call = "CANNOT DECIDE"
        if aliased:
            why = ("the seat is ALIASED with target direction (only %.0f%% of the eye-shift "
                   "regressor survives the direction basis), so no sample count separates them. "
                   "Cover the same part of the field at every seating."
                   % (100 * est["collinearity"]))
        else:
            why = ("gain = %.2f +- %.2f, the interval spans both 0 and 1, so this data cannot "
                   "tell the correction from nothing. This is NOT evidence it is useless."
                   % (g, se))
            why += (" The term is worth a median %.1f px here; per-sample residual scatter is "
                    "%.1f px." % (eff_med, est["sigma_px"]))
            if not moved:
                why += (" The seat barely varied: %d seating(s), span %.2f mm."
                        % (rep["n_seats"], rep["span_mm"]))
            why += (" To fix it: re-seat DELIBERATELY DIFFERENTLY (%.0f mm+), and calibrate CLOSE "
                    ",  the term is parallax and scales as 1/distance." % RECOMMENDED_SEAT_MM)
    elif g - 2 * se > 0.0:
        call = "SET EYE_SHIFT_GAIN = %.2f" % round(g, 2)
        why = ("regression gain %.2f +- %.2f (%.1f sigma from zero), effect %.1f px, residual "
               "scatter %.1f px over %d samples." % (g, se, est["t"], eff_med,
                                                     est["sigma_px"], est["n"]))
    else:
        call = "LEAVE EYE_SHIFT_GAIN = 0.0"
        why = ("gain = %.2f +- %.2f, resolvable and consistent with zero. This IS a real null "
               "result: the measured eye shift does not predict the residual."
               % (g, se))

    return {"call": call, "why": why, "seat": rep, "rows": rows, "best": best, "est": est,
            "effect_px": (eff_med, eff_max), "moved": bool(moved), "powered": bool(powered),
            "aliased": bool(aliased)}


def contamination(X):
    """Samples whose target was outside the display entirely, see geometry.offscreen.

    Reported FIRST, because everything downstream is a statistic over these samples and 4 of the
    17 in the only real set were unstorable. A spread figure computed over poisoned samples is
    what made a run covering 5.6% of the world field read as covering 47%.
    """
    X = np.atleast_2d(np.asarray(X, float))
    bad = np.array([geometry.offscreen(x) for x in X], bool)
    return bad, np.array([geometry.offaxis_deg(x) for x in X], float)


def print_verdict(v, X=None):
    if X is not None:
        bad, ang = contamination(X)
        if bad.any():
            print("== CONTAMINATION ==")
            print("  %d of %d samples had the target OUTSIDE the %.1f deg display "
                  "(off-axis up to %.0f deg), these cannot be alignments."
                  % (bad.sum(), len(bad), geometry.DISPLAY_FOV_DEG, ang[bad].max()))
            good = ~bad
            if good.sum() >= 2:
                print("  world-dot span WITH them %.3f, WITHOUT them %.3f"
                      % (X[:, 0].max() - X[:, 0].min(),
                         X[good, 0].max() - X[good, 0].min()))
            print("  main.py now refuses to store these; older rows predate that guard.")
    r = v["seat"]
    print("== SEATING ==")
    print("  %d samples across %d seating(s): %s"
          % (r["n"], r["n_seats"], r["per_seat_n"]))
    print("  eye-feature sd        %s   (2026-08-03 'never moved' run: %.3f / %.4f)"
          % (" ".join("%.4f" % s for s in r["eye_feature_sd"]), *BASELINE_EYE_SD))
    print("  eye displacement mm   between-seat sd %.2f, %.2f | within-seat sd %.2f, %.2f"
          % (r["between_sd_mm"][0], r["between_sd_mm"][1],
             r["within_sd_mm"][0], r["within_sd_mm"][1]))
    print("  seat span             %.2f mm   -> glasses %s"
          % (r["span_mm"], "MOVED" if v["moved"] else "did NOT move enough"))
    print("== EFFECT SIZE (what the term can possibly be worth on this data) ==")
    print("  gain 1.0 moves the overlay by median %.2f px, max %.2f px"
          % v["effect_px"])
    e = v["est"]
    print("== GAIN, ESTIMATED BY REGRESSION (the sensitive test) ==")
    print("  gain %.2f +- %.2f  (%.1f sigma)   residual scatter %.1f px over %d samples"
          % (e["gain"], e["se"], e["t"], e["sigma_px"], e["n"]))
    print("  seat/direction independence %.0f%%  (below %.0f%% = aliased, unusable)"
          % (100 * e["collinearity"], 100 * MIN_COLLINEARITY))
    print("== SWEEP (median-error grid, kept for continuity with 2026-08-03, and far blunter) ==")
    for row in v["rows"]:
        line = "  gain %.2f   median %6.1f px   p95 %6.1f px" % (
            row["gain"], row["median"], row["p95"])
        if "median_refit" in row:
            line += "   | re-fit FOV %.2f deg -> %.1f px" % (row["fov"], row["median_refit"])
        print(line)
    print("== VERDICT ==")
    print("  %s" % v["call"])
    print("  %s" % v["why"])


# ---------------------------------------------------------------------------
#  synthetic data, used only by the controls below
# ---------------------------------------------------------------------------
def _canthus_uv(dx_mm, dy_mm):
    """Inverse of geometry.eye_offset_mm: an eye offset in mm -> the 4 canthus features.

    Both eyes are displaced identically, which is the common case (the whole carrier shifting) and
    the one the correction averages to.
    """
    th = np.tan(np.radians(geometry.EYE_CAM_FOV) / 2.0)
    ar = 10.0 / 16.0
    out = []
    for (nu, nv), dist in ((geometry.NOMINAL_CANTH_UV_L, geometry.CANTH_DIST_L),
                           (geometry.NOMINAL_CANTH_UV_R, geometry.CANTH_DIST_R)):
        out += [nu + dx_mm / (2.0 * th * dist),
                nv - dy_mm / (2.0 * th * ar * dist)]
    return out


def make_synth(n_seats, per_seat, seat_mm=4.0, true_gain=1.0, uv_noise=0.008,
               align_px=4.0, depth_mm=(700.0, 3000.0), fov_err_deg=1.5,
               one_target_per_seat=False, seed=0):
    """Samples whose TRUE pixel was generated with a KNOWN eye-shift gain.

    The point of controls built this way: if the estimator cannot recover a gain that was planted
    in the data, it can never be trusted to find one that was not.

    `fov_err_deg` plants the thing that actually makes this hard -- a direction-dependent
    systematic, here a mis-scaled display FOV, which is the dominant real error (the constant was
    wrong by 1.75 deg until 2026-08-03). Without it the controls would be far easier than reality
    and would validate nothing.
    """
    rng = np.random.default_rng(seed)
    from world_mesh import DEFAULT_B, DEFAULT_F
    X, Y, seats = [], [], []
    for s in range(n_seats):
        # seat 0 is nominal; later seatings are displaced. seat_mm = 0 means the glasses never
        # moved, which is the negative control.
        off = np.zeros(2) if s == 0 else rng.uniform(-seat_mm, seat_mm, 2)
        # ALIASING CASE: each seating only ever looks at ONE target. The seat then varies exactly
        # where the direction varies and the two cannot be told apart by any estimator.
        fixed = (rng.uniform(0.30, 0.70), rng.uniform(0.35, 0.65),
                 rng.uniform(*depth_mm)) if one_target_per_seat else None
        for _ in range(per_seat):
            uL = fixed[0] if fixed else rng.uniform(0.30, 0.70)
            vL = fixed[1] if fixed else rng.uniform(0.35, 0.65)
            depth = fixed[2] if fixed else rng.uniform(*depth_mm)
            disp = (DEFAULT_F * DEFAULT_B / depth) / rig.WORLD_RES
            eye = np.array(_canthus_uv(*off)) + rng.normal(0, uv_noise, 4)
            x = np.array([uL, vL, uL - disp, vL] + list(eye))
            # truth is rendered at the TRUE display scale; the estimator will use the assumed one
            y = _pixels(x[None, :], true_gain,
                        fov=geometry.DISPLAY_FOV_DEG + fov_err_deg)[0]
            y = y + rng.normal(0, align_px / DISPLAY_H, 2)
            X.append(x); Y.append(np.clip(y, 0, 1)); seats.append(s)
    return np.array(X), np.array(Y), np.array(seats, int)


# What to actually do on the rig, DERIVED from power_table() below rather than guessed.
RECOMMENDED_SEAT_MM = 6.0
RECOMMENDED_DEPTH_MM = (400.0, 1000.0)


def power_table(seeds=12, verbose=True):
    """How much data, at what distance, with how big a re-seat, before this is decidable?

    This is the question that should have been asked before the 2026-08-03 run, and it costs
    nothing to answer offline. The eye-shift term is PARALLAX -- it scales as 1/distance -- so the
    same protocol that is decisive at 0.5 m is hopeless at 3 m, and no sample count fixes that.
    """
    grid = [
        # label,                     n_seats, per_seat, seat_mm, depth
        ("2026-08-03 run (no re-seat, 1-3 m)", 1, 17, 0.0, (700.0, 3000.0)),
        ("re-seat only, 1-3 m, 32 samples",    4,  8, 3.0, (700.0, 3000.0)),
        ("re-seat only, 1-3 m, 64 samples",    8,  8, 3.0, (700.0, 3000.0)),
        ("re-seat BIG, 1-3 m, 32 samples",     4,  8, 6.0, (700.0, 3000.0)),
        ("re-seat BIG, CLOSE 0.4-1 m, 32",     4,  8, 6.0, (400.0, 1000.0)),
        ("re-seat BIG, CLOSE 0.4-1 m, 48",     6,  8, 6.0, (400.0, 1000.0)),
    ]
    rows = []
    for label, ns, per, mm, depth in grid:
        ses, gains_, effs, hits = [], [], [], 0
        for sd in range(seeds):
            X, Y, seats = make_synth(ns, per, seat_mm=mm, true_gain=1.0,
                                     depth_mm=depth, seed=100 + sd)
            est = estimate_gain(X, Y)
            ses.append(est["se"]); gains_.append(est["gain"])
            effs.append(effect_px(X, 1.0)[0])
            if est["se"] <= MAX_USABLE_SE and est["gain"] - 2 * est["se"] > 0:
                hits += 1
        rows.append({"label": label, "n": ns * per, "se": float(np.mean(ses)),
                     "gain": float(np.mean(gains_)), "effect": float(np.median(effs)),
                     "detect": hits / float(seeds)})
    if verbose:
        print("== POWER: can this protocol resolve a TRUE gain of 1.0? ==")
        print("  %-38s %5s %8s %9s %8s %8s" %
              ("protocol", "n", "effect", "gain", "SE", "detect"))
        for r in rows:
            print("  %-38s %5d %6.1f px %9.2f %8.2f %7.0f%%" %
                  (r["label"], r["n"], r["effect"], r["gain"], r["se"], 100 * r["detect"]))
        print("  (detect = fraction of runs that would correctly say SET, at 2 sigma)")
    return rows


# ---------------------------------------------------------------------------
def load_real(db_path=None):
    from config import Config
    from dataset import Dataset
    cfg = Config()
    path = db_path or cfg.db_path
    if not os.path.exists(path):
        raise SystemExit("no sample DB at %s" % path)
    ds = Dataset(path, cfg.feature_names)
    X, Y, W = ds.load()
    seats = ds.load_seats()
    ds.close()
    return X, Y, W, seats


def selftest(verbose=True):
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               (", " + detail) if detail else ""))

    # --- NEGATIVE CONTROL 1: the glasses never moved -----------------------------------------
    # This is the 2026-08-03 dataset reproduced deliberately. A gain of 1.0 was planted in the
    # truth, so the tool is being handed a correction that IS real -- and it still must refuse to
    # claim it, because the data cannot show it. Answering "LEAVE IT OFF" here would be the exact
    # wrong conclusion, written down as a finding, which is what nearly happened.
    Xn, Yn, sn = make_synth(n_seats=4, per_seat=6, seat_mm=0.0, true_gain=1.0, seed=1)
    vn = verdict(Xn, Yn, sn)
    chk("glasses that never moved -> CANNOT DECIDE (not a null result)",
        vn["call"].startswith("CANNOT DECIDE"), vn["call"])
    chk("...and the seat report says the seat did not vary", not vn["moved"],
        "span %.2f mm" % vn["seat"]["span_mm"])
    chk("...and the gain is unresolvable (SE spans 0 and 1)",
        vn["est"]["se"] > MAX_USABLE_SE,
        "gain %.2f +- %.2f" % (vn["est"]["gain"], vn["est"]["se"]))

    # --- NEGATIVE CONTROL 2: seats vary, but there is NO correction to find -------------------
    # Truth generated at gain 0: the eye features carry only tracking noise. A tool that always
    # finds a gain is worthless, so this must come back LEAVE IT OFF -- and, unlike the case
    # above, that null is real and is allowed to be reported as one.
    Xz, Yz, sz = make_synth(n_seats=4, per_seat=8, seat_mm=RECOMMENDED_SEAT_MM, true_gain=0.0,
                            depth_mm=RECOMMENDED_DEPTH_MM, seed=2)
    vz = verdict(Xz, Yz, sz)
    chk("seat varies but truth has no eye term -> LEAVE IT OFF",
        vz["call"].startswith("LEAVE"),
        "%s (gain %.2f +- %.2f)" % (vz["call"], vz["est"]["gain"], vz["est"]["se"]))
    chk("...and that null IS resolvable (so it may be reported as a null)", vz["powered"],
        "SE %.2f" % vz["est"]["se"])

    # --- NEGATIVE CONTROL 3: the seat is ALIASED with target direction ------------------------
    # Each seating looks at one target only. The seat then varies exactly where the direction
    # varies, so no estimator can separate the eye term from a direction-dependent error -- and
    # the failure is silent unless it is checked for. This is the protocol trap the recommendation
    # "cover the same part of the field at every seating" exists to avoid.
    Xa, Ya, sa = make_synth(n_seats=3, per_seat=8, seat_mm=RECOMMENDED_SEAT_MM, true_gain=1.0,
                            depth_mm=RECOMMENDED_DEPTH_MM, one_target_per_seat=True, seed=7)
    va = verdict(Xa, Ya, sa)
    chk("one target per seating -> ALIASED, refuses to answer",
        va["aliased"] and va["call"].startswith("CANNOT DECIDE"),
        "independence %.0f%%, call: %s" % (100 * va["est"]["collinearity"], va["call"]))

    # --- POSITIVE CONTROL: a KNOWN gain, under the RECOMMENDED protocol -----------------------
    Xp, Yp, sp = make_synth(n_seats=4, per_seat=8, seat_mm=RECOMMENDED_SEAT_MM, true_gain=1.0,
                            depth_mm=RECOMMENDED_DEPTH_MM, seed=3)
    vp = verdict(Xp, Yp, sp)
    chk("re-seated run with a planted gain -> the seat report sees the movement", vp["moved"],
        "span %.2f mm, between sd %.2f vs within %.2f"
        % (vp["seat"]["span_mm"], np.max(vp["seat"]["between_sd_mm"]),
           np.max(vp["seat"]["within_sd_mm"])))
    chk("...and the estimator RECOVERS it (within 0.3 of the planted 1.0)",
        abs(vp["est"]["gain"] - 1.0) <= 0.3,
        "gain %.2f +- %.2f" % (vp["est"]["gain"], vp["est"]["se"]))
    chk("...and says SET, with the interval clear of zero",
        vp["call"].startswith("SET"), "%s | %s" % (vp["call"], vp["why"]))
    chk("...and the recommended protocol keeps seat and direction well separated",
        vp["est"]["collinearity"] > 0.5,
        "independence %.0f%%" % (100 * vp["est"]["collinearity"]))

    # --- WHY THE REGRESSION REPLACED THE GRID SWEEP, pinned as a check ------------------------
    # The MARGINAL protocol: a plain off-and-on re-seat (~3 mm) at the 1-3 m distances this rig
    # has been calibrated at. A gain of 1.0 is genuinely present in every one of these datasets.
    #
    # ACROSS SEEDS, not on one -- a single run proves nothing here, and an earlier version of this
    # check passed or failed purely on which seed it drew. The grid is also given every advantage:
    # its cells include the planted value exactly, so it can be right by construction.
    def _recover(seat_mm, depth, seeds=10, base=400):
        gg, rg = [], []
        for sd in range(seeds):
            Xm, Ym, _ = make_synth(n_seats=4, per_seat=8, seat_mm=seat_mm, true_gain=1.0,
                                   depth_mm=depth, seed=base + sd)
            gg.append(min(sweep(Xm, Ym), key=lambda r: r["median"])["gain"])
            rg.append(estimate_gain(Xm, Ym)["gain"])
        gg, rg = np.array(gg), np.array(rg)
        return (float(np.sqrt(np.mean((gg - 1) ** 2))), float(gg.mean()),
                float(np.sqrt(np.mean((rg - 1) ** 2))), float(rg.mean()))

    g_rms, g_mean, r_rms, r_mean = _recover(RECOMMENDED_SEAT_MM, RECOMMENDED_DEPTH_MM)
    chk("recommended protocol: the REGRESSION recovers the planted gain far better than the GRID",
        r_rms < g_rms / 2.0,
        "rms error: grid %.2f (mean %.2f) vs regression %.2f (mean %.2f) over 10 seeds"
        % (g_rms, g_mean, r_rms, r_mean))

    # The grid does not merely scatter -- it reads LOW, because a term worth a few px cannot move
    # the median of a distribution whose per-sample scatter is many times larger, so the sweep
    # keeps landing nearer gain 0 than the truth. That is the shape of the 2026-08-03 result.
    mg_rms, mg_mean, mr_rms, mr_mean = _recover(3.0, (700.0, 3000.0), base=500)
    chk("marginal protocol (plain off-and-on at 1-3 m): the GRID reads LOW, towards zero",
        mg_mean < 0.85, "grid mean %.2f vs planted 1.00 (regression mean %.2f)"
        % (mg_mean, mr_mean))

    # --- the mm<->uv inverse used by the controls must round-trip through the real function ---
    # If it did not, every control above would be testing its own arithmetic instead of
    # geometry.eye_offset_mm.
    g0 = geometry.EYE_SHIFT_GAIN
    try:
        geometry.EYE_SHIFT_GAIN = 1.0
        want = np.array([3.0, -2.0])
        got = geometry.eye_offset_mm([0.5, 0.5, 0.48, 0.5] + _canthus_uv(*want))[:2]
        chk("mm -> canthus uv -> mm round-trips through the production function",
            np.allclose(got, want, atol=1e-6), "want %s got %s" % (want, np.round(got, 4)))
    finally:
        geometry.EYE_SHIFT_GAIN = g0

    # --- the contamination report must flag the real poisoned samples, and only those ---------
    poisoned = np.array([[0.972, 0.072, 0.948, 0.070] + list(geometry.NOMINAL_CANTH_UV_L)
                         + list(geometry.NOMINAL_CANTH_UV_R),
                         [0.531, 0.376, 0.497, 0.365] + list(geometry.NOMINAL_CANTH_UV_L)
                         + list(geometry.NOMINAL_CANTH_UV_R)])
    bad, _ = contamination(poisoned)
    chk("contamination report flags the off-screen sample and spares the good one",
        bad.tolist() == [True, False], "flags %s" % bad.tolist())

    # --- PRODUCTION PATH: a real Dataset, written and read the way main.py does ---------------
    # Built through the real constructor with the real schema, not hand-assembled: the seat column
    # is the whole point of the protocol, and a migration that silently dropped it would make
    # every run above look like a single seating.
    import tempfile
    from config import Config
    with tempfile.TemporaryDirectory() as td:
        from dataset import Dataset
        cfg = Config()
        ds = Dataset(os.path.join(td, "s.db"), cfg.feature_names)
        for i, (x, y, s) in enumerate(zip(Xp[:6], Yp[:6], sp[:6])):
            ds.add(x, y, weight=1.0, seat=int(s))
        X2, Y2, _ = ds.load()
        s2 = ds.load_seats()
        ds.close()
        chk("Dataset round-trips the seat index in load() order",
            len(s2) == 6 and np.array_equal(s2, sp[:6]) and np.allclose(X2, Xp[:6]),
            "seats %s" % s2.tolist())

    print("RESEAT OK ✅" if ok_all else "RESEAT FAILED ❌")
    return 0 if ok_all else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="analyse the stored calibration samples")
    p.add_argument("--db", type=str, default=None, help="sample DB path override")
    p.add_argument("--joint", action="store_true",
                   help="also re-fit DISPLAY_FOV_DEG at each gain (the fair comparison)")
    p.add_argument("--power", action="store_true",
                   help="how much data, at what distance, with how big a re-seat, before this "
                        "is decidable at all")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.power:
        power_table()
        return 0
    if a.check:
        X, Y, W, seats = load_real(a.db)
        if len(X) < 4:
            raise SystemExit("only %d samples, run a calibration first" % len(X))
        print_verdict(verdict(X, Y, seats, joint=a.joint), X)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
