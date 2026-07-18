"""Real-life deviation: calibrate a user, then run the LIVE noisy loop with multi-fixation
averaging, and measure the deviation a real wearer would see.

Real-life error = sqrt( systematic_calibration_error^2 + per_attempt_noise^2 / N_avg ).
Averaging N fixations beats the per-attempt noise (microsaccades, tracker, head motion) as
1/sqrt(N), but CANNOT touch the systematic calibration error. So this both (a) shows how many
fixations buy how much, and (b) exposes the systematic floor we must lower to reach a target.

Goal of the campaign: get the real-life deviation under 0.75 px. That needs the SYSTEMATIC
floor well under 0.75 px (averaging handles the rest) — pursued in successive experiments.

Run:  python3 realtime.py        (or:  python3 main.py --realtime)
"""
import numpy as np

import autosim
import match
import pixel_map
from accuracy_map import _identifier
from preset import build_preset
from calibrator import Calibrator
from config import Config


def calibrate_user(sim, subj, prior, K, degree=3):
    """Realistic per-user calibration: collect K of the user's own corrections (noisy,
    perceptually biased), sampled ACROSS the operating envelope (so the fit doesn't have to
    extrapolate), and fit a flexible per-user residual on the population prior. As K grows it
    converges toward the MODEL floor, absorbing the user's session + perceptual biases."""
    rng = sim.rng
    X, Y, guard = [], [], 0
    while len(X) < K and guard < K * 4:
        guard += 1
        dty = rng.uniform(-2.0, 8.0)                    # cover the slip envelope (the eval range)
        dev = np.array([dty * 0.25, 0.0, rng.uniform(-1.0, 1.0), rng.uniform(-2.5, 2.5), dty, 0.0])
        o = sim.observe(subj, dev, sim.world_point())   # dots cover the field
        if o is not None:
            X.append(o[0]); Y.append(o[1])
    X, Y = np.array(X), np.array(Y)
    base = np.array([prior.predict(x) for x in X])
    resid = Calibrator(Config().n_features, degree=degree, min_samples=12)
    resid.fit(X, Y - base)                              # flexible per-user residual on the prior

    def predict(f):
        r = resid.predict_raw(f) if resid.is_trained else 0.0
        return np.clip(prior.predict(f) + r, 0.0, 1.0)
    return predict


def real_life_curve(n_users=12, Ks=(20, 60, 150, 400), Ns=(1, 8, 32), seed=0, verbose=True):
    """Real-life deviation (vs PERCEIVED alignment) for per-user calibration with K corrections
    and N-fixation averaging. cdrift reset per target (a ~1 s dwell, not a whole session)."""
    prior = match.train_prior()
    fsim = autosim.Simulator(2024)
    ev = autosim.Simulator(2024)
    eval_poses, eval_dots = pixel_map.pose_grid(), pixel_map.dot_grid()[::5]
    nmax = max(Ns)
    grid = {(K, N): [] for K in Ks for N in Ns}
    syst = {K: [] for K in Ks}
    if verbose:
        print("== real-life deviation: per-user calibration (K corrections) + N-fixation averaging ==")
        print("  measured vs PERCEIVED alignment (what the wearer sees on target)\n")
    for u in range(n_users):
        subj = fsim.new_subject()
        preds = {K: calibrate_user(fsim, subj, prior, K) for K in Ks}
        for dev in eval_poses:
            for P in eval_dots:
                g = ev.ground_truth(subj, dev, P)
                if g is None:
                    continue
                perceived = g[1] + subj.human_bias            # where the wearer perceives alignment
                subj.cdrift = np.zeros(4)                      # ~1 s dwell: no long-session drift
                obs = []
                guard = 0
                while len(obs) < nmax and guard < nmax * 3:
                    guard += 1
                    o = ev.observe(subj, dev, P)
                    if o is not None:
                        obs.append(o[0])
                if len(obs) < nmax:
                    continue
                obs = np.array(obs)
                for K in Ks:
                    pr = np.array([preds[K](f) for f in obs])
                    syst[K].append(np.linalg.norm(preds[K](g[0]) - perceived) * 1080.0)
                    for N in Ns:
                        grid[(K, N)].append(np.linalg.norm(pr[:N].mean(0) - perceived) * 1080.0)
    if verbose:
        print("  systematic (clean features) vs corrections K:")
        for K in Ks:
            print("    K=%4d : %6.3f px" % (K, np.median(np.array(syst[K]))))
        print("\n  real-life deviation (median px) by [K corrections] x [N averaged fixations]:")
        print("    %6s | " % "K\\N" + " | ".join("N=%-4d" % N for N in Ns))
        print("    " + "-" * (9 + 9 * len(Ns)))
        for K in Ks:
            row = " | ".join("%6.3f" % np.median(np.array(grid[(K, N)])) for N in Ns)
            print("    %6d | %s" % (K, row))
        best = min(np.median(np.array(grid[(K, N)])) for K in Ks for N in Ns)
        print("    " + "-" * (9 + 9 * len(Ns)))
        tag = "UNDER 0.75 px" if best < 0.75 else "best %.2f px (target 0.75)" % best
        print("\n  best real-life deviation: %s" % tag)
    return grid, syst


def setup(n_train=800, seed=0):
    """Population prior + an 8-feature geometry identifier (matches the deployed pipeline)."""
    prior = match.train_prior()
    poses, dots = match.probe_poses(), match.probe_dots()
    cam = match.pupil_sensor.build_camera()
    sim = autosim.Simulator(seed); rng = np.random.default_rng(seed + 100)
    INA, EYE, DESC = [], [], []
    for _ in range(n_train):
        s = sim.new_subject()
        ia, ey, _ir, _ic, _pu = match.fingerprint(sim, s, prior, poses, dots, 4, rng, cam)
        INA.append(ia); EYE.append(ey); DESC.append(s.descriptor())
    ident = _identifier([np.array(INA), np.array(EYE)], np.array(DESC))
    return prior, ident, poses, dots, cam, rng


def averaging_curve(n_users=15, Ns=(1, 2, 4, 8, 16, 32, 64), seed=0, verbose=True):
    prior, ident, poses, dots, cam, rng = setup(seed=seed)
    fsim = autosim.Simulator(2024)                 # fingerprint (8-feature)
    ev = autosim.Simulator(2024)                   # live noisy observe (8-feature)
    fsim.rng = np.random.default_rng(7)
    eval_poses, eval_dots = pixel_map.pose_grid(), pixel_map.dot_grid()[::4]
    res = {N: [] for N in Ns}
    nmax = max(Ns)
    if verbose:
        print("== real-life deviation vs multi-fixation averaging (deployed poly preset) ==")
        print("  %d users; averaging N noisy fixations per target\n" % n_users)
    for _ in range(n_users):
        subj = fsim.new_subject()
        ia, ey, _ir, _ic, _pu = match.fingerprint(fsim, subj, prior, poses, dots, 3, rng, cam)
        preset = build_preset(ident([ia[None], ey[None]])[0])
        for dev in eval_poses:
            for P in eval_dots:
                g = ev.ground_truth(subj, dev, P)
                if g is None:
                    continue
                truth = g[1]
                preds = []
                guard = 0
                while len(preds) < nmax and guard < nmax * 3:
                    guard += 1
                    o = ev.observe(subj, dev, P)
                    if o is not None:
                        preds.append(preset.predict(o[0]))
                if len(preds) < nmax:
                    continue
                preds = np.array(preds)
                for N in Ns:
                    res[N].append(np.linalg.norm(preds[:N].mean(0) - truth) * 1080.0)
    if verbose:
        print("    %4s | %10s | %10s" % ("N", "median px", "95th px"))
        print("    " + "-" * 32)
        for N in Ns:
            a = np.array(res[N])
            print("    %4d | %10.3f | %10.3f" % (N, np.median(a), np.percentile(a, 95)))
        floor = np.median(np.array(res[max(Ns)]))
        print("    " + "-" * 32)
        print("\n  averaging drives the per-attempt noise down ~1/sqrt(N) to the SYSTEMATIC floor")
        print("  (~%.2f px here). Real-life < 0.75 px therefore needs that systematic floor lowered" % floor)
        print("  below ~0.6 px — the focus of the next experiments (complete geometry + physics).")
    return res


if __name__ == "__main__":
    averaging_curve()
