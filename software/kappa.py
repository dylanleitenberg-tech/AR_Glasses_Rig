"""Two measured answers about angle kappa (see KAPPA.md for the full study).

1) convergence() — "why can't we have 3 px?"  Overlay error vs the number of the user's OWN
   corrections, reported TWO ways:
     * vs geometric truth  — floored by the perceptual bias (kappa/bias confound);
     * vs perceived alignment (truth + the user's own consistent offset) — floored only by
       how well the model fits, so it drops toward 0 as corrections accumulate.
   The honest point: against what the user actually SEES on target, sub-3 px is reachable;
   the geometric-truth number is inflated by a bias the user doesn't perceive as error.

2) vergence_separation() — a concrete WAY AROUND kappa (KAPPA.md §5 method 7). The
   perceptual offset has two pieces with different signatures: angle kappa (constant with
   viewing distance) and the vergence-accommodation bias (scales with 1/d - 1/d_virtual,
   and is ZERO at the virtual image plane). Calibrating at ONE distance cannot tell them
   apart, so it fails to transfer to other distances. Calibrating across distances (or at
   the virtual plane) regresses them apart and restores accuracy everywhere — kappa
   separated from the bias with NO new hardware.

Run:  python3 kappa.py        (or:  python3 main.py --kappa)
"""
import numpy as np

import rig
import autosim
import pixel_map
from calibrator import Calibrator
from autotrain import WarmCalibrator
from config import Config

DEG_PER_PX = 50.0 / 1080.0            # display: 50 deg FOV over 1080 px


# ----------------------------------------------------------------------
# 1) convergence: perceived vs geometric error as corrections accumulate
# ----------------------------------------------------------------------
def convergence(n_users=30, Ks=(2, 4, 8, 16, 32, 64), seed_base=6000, verbose=True):
    if verbose:
        print("   (training the warm-start prior the user starts from ...)")
    prior = pixel_map.train_prior(Config())     # population features->pixel map
    poses = pixel_map.pose_grid()
    dots = pixel_map.dot_grid()[::5]           # 20 gaze dots for the clean eval
    rows = []
    for K in Ks:
        geo, perc = [], []
        for i in range(n_users):
            s = autosim.Simulator(seed_base + i); subj = s.new_subject()
            # the user's own corrections (noisy, perceptually biased — all reality gives)
            X, Y, dev, guard = [], [], s.seat(), 0
            while len(X) < K and guard < K * 6:
                guard += 1; dev = s.slip(dev)
                o = s.observe(subj, dev, s.world_point())
                if o is not None:
                    X.append(o[0]); Y.append(o[1])
            cal = WarmCalibrator(prior, Config())   # warm start + per-user residual
            cal.fit(np.array(X), np.array(Y))
            # clean systematic eval: predict vs geometric truth AND vs perceived alignment
            ge, pe = [], []
            for dev in poses:
                for P in dots:
                    g = s.ground_truth(subj, dev, P)
                    if g is None:
                        continue
                    f, truth = g
                    pred = cal.predict(f)
                    ge.append(np.linalg.norm(pred - truth))
                    pe.append(np.linalg.norm(pred - (truth + subj.human_bias)))
            if ge:
                geo.append(np.median(ge) * 1080.0); perc.append(np.median(pe) * 1080.0)
        rows.append((K, np.median(geo), np.median(perc)))
    if verbose:
        print("== convergence: overlay error vs number of the user's own corrections ==")
        print("   (systematic error vs the oracle; per-attempt fixation noise ~%.1f px adds on"
              % (rig.FIXATION_JITTER_DEG / DEG_PER_PX))
        print("    top at runtime and averages as 1/sqrt(N))\n")
        print("   %5s | %18s | %20s" % ("K", "vs geometric truth", "vs perceived alignment"))
        print("   " + "-" * 50)
        for K, g, p in rows:
            print("   %5d | %10.1f px      | %12.1f px" % (K, g, p))
        print("   " + "-" * 50)
        gfloor = rows[-1][1]; pfloor = rows[-1][2]
        print("\n   vs geometric truth floors at ~%.1f px (the perceptual bias = kappa confound)."
              % gfloor)
        print("   vs PERCEIVED alignment reaches ~%.1f px (%s 3 px) — what the user sees on target."
              % (pfloor, "below" if pfloor < 3 else "approaching"))
        print("   So 'within 3 px' is gated by the user's own corrections + fixation noise,")
        print("   NOT by the database; the bias the geometric number penalizes isn't perceived.")
    return rows


# ----------------------------------------------------------------------
# 2) vergence separation: a way around the kappa/bias confound (no new HW)
# ----------------------------------------------------------------------
def vergence_separation(n_users=200, seed=0, verbose=True):
    """The per-user alignment offset = a (kappa + constant bias, CONSTANT in distance) +
    b * x(d) (vergence-accommodation bias, x(d) = 1/d - 1/d_virtual, ZERO at the virtual
    plane). Compare recovering the correct offset at every distance from (a) a single
    near-distance calibration that assumes the offset is constant, vs (b) a multi-distance
    calibration that regresses a and b apart."""
    rng = np.random.default_rng(seed)
    vid = rig.VIRTUAL_DIST
    x = lambda d: 1.0 / d - 1.0 / vid
    cal_d_single = 400.0                       # one near distance
    cal_d_multi = np.array([350.0, 700.0, 1500.0])
    test_d = np.linspace(300.0, 1800.0, 8)
    noise = 0.004                              # per-distance alignment noise (norm), few reps

    err_single, err_multi, err_vid = [], [], []
    for _ in range(n_users):
        a = rng.normal(0, 0.006)               # kappa + constant bias (norm; ~6.5 px SD)
        b = rng.normal(9.0, 3.0)               # per-user VAC gain (norm offset per 1/mm)
        true_off = lambda d: a + b * x(d)
        # (a) single-distance: measure once, assume constant
        est_single = true_off(cal_d_single) + rng.normal(0, noise)
        # (b) multi-distance: regress offset = a_hat + b_hat * x over the cal distances
        xs = x(cal_d_multi)
        ys = np.array([true_off(d) + rng.normal(0, noise) for d in cal_d_multi])
        A = np.column_stack([np.ones_like(xs), xs])
        a_hat, b_hat = np.linalg.lstsq(A, ys, rcond=None)[0]
        # (c) calibrate AT the virtual plane (x=0 -> pure constant), then it's still missing b
        est_vid = true_off(vid) + rng.normal(0, noise)   # = a (vergence-clean), assumed constant
        for d in test_d:
            t = true_off(d)
            err_single.append(abs(t - est_single))
            err_multi.append(abs(t - (a_hat + b_hat * x(d))))
            err_vid.append(abs(t - est_vid))
    es, em, ev = (np.array(e) * 1080.0 for e in (err_single, err_multi, err_vid))
    if verbose:
        print("\n== a way around kappa: multi-vergence separation (KAPPA.md method 7) ==")
        print("   registration error across target distances 300-1800 mm (px @1080p):")
        print("   %-34s %8s %8s" % ("calibration strategy", "median", "max"))
        print("   " + "-" * 52)
        print("   %-34s %8.1f %8.1f" % ("single distance (assume constant)", np.median(es), es.max()))
        print("   %-34s %8.1f %8.1f" % ("at virtual plane (constant)", np.median(ev), ev.max()))
        print("   %-34s %8.1f %8.1f" % ("multi-distance (regress a,b)", np.median(em), em.max()))
        print("   " + "-" * 52)
        print("\n   single-distance calibration carries a big error to OTHER distances (the")
        print("   vergence-accommodation bias it mistook for a constant kappa); regressing across")
        print("   distances separates kappa (the constant) from the vergence bias (the slope) and")
        print("   stays accurate everywhere. The confound is broken by VARYING viewing distance —")
        print("   no extra sensor. (A PCCR/imaging measurement removes the residual constant too.)")
    return es, em, ev


def run_all():
    convergence()
    vergence_separation()


if __name__ == "__main__":
    run_all()
