"""kappa_boost.py — improve kappa precision: honest label-based ID + vernier corrections.

THE GAP THIS CLOSES: stereo_test fingerprints the user with `prior.predict(f) - TRUTH`, but
reality never shows the truth — only the USER'S LABEL (truth + per-session perceptual bias +
per-correction alignment noise + fat-fingers). Kappa is identified FROM those corrections, so
correction quality gates kappa precision, and kappa gates the deployed floor (stereo+good-kappa
was the proven <1 px recipe). This module:

  1. re-runs the complete-geometry stereo pipeline with LABEL-based fingerprints at the
     current dot-nudge noise (the honest deployed number),
  2. swaps in VERNIER-grade corrections (vernier.py UI: ~3x lower SD, gross errors killed by
     confirm-twice; uses YOUR measured noise from data/vernier_noise.json when present),
  3. adds multi-fixation AVERAGING at test (more repeats per probe = 1/sqrt(N) noise),
  4. keeps the truth-fingerprint arm as the optimistic upper bound for reference.

Each arm's identifier is TRAINED the way it is TESTED (matched noise), so ridge absorbs what
it can. Reported per arm: kappa recovery (deg + %popSD) and deployed STEREO physics-preset
systematic error (px @1080p) — directly comparable to stereo_test's 1.077 px.

NOTE on multi-DISTANCE separation: the autosim perceptual bias is constant per session, so
distance sweeps separate nothing HERE; the multi-vergence protocol's 2.4 px cross-distance
result stands from kappa.py and applies at the real-device protocol level.

Run:  python3 main.py --kappa-boost
"""
import json
import os

import numpy as np

import anatomy
import autosim
import complete_geometry as cg
import match
import pixel_map
import pupil_sensor
import physics_preset as pp
import rig
from accuracy_map import _identifier

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

VERNIER_SD_DEFAULT = 0.0013
VERNIER_GROSS = 0.002


def _vernier_sd():
    p = os.path.join(_DATA, "vernier_noise.json")
    if os.path.exists(p):
        with open(p) as f:
            return float(json.load(f)["noise_sd_norm"]), True
    return VERNIER_SD_DEFAULT, False


class _correction_noise:
    """Temporarily set the human per-correction model (what the UI determines)."""
    def __init__(self, sd, gross):
        self.v = (sd, gross)

    def __enter__(self):
        self.saved = (rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB)
        rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB = self.v

    def __exit__(self, *a):
        rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB = self.saved


def _fingerprint(sim, subj, prior, poses, dots, repeats, rng, cam, use_label,
                 dc_remove=False, use_pupil=True):
    """stereo_test._fingerprint, with the reference switchable: TRUTH (optimistic) or the
    user's LABEL (what a real calibration actually gets). dc_remove subtracts the user's
    MEAN correction from the inaccuracy pattern — the constant per-session perceptual bias
    is DC in pixels while kappa's effect varies across the field, so removing DC makes the
    identification bias-invariant (at the cost of kappa's own DC component). use_pupil=False
    = the no-IR build (no NIR pupil cams: no pupil features, no pupil-sensor fingerprint)."""
    inacc, eyestereo, pup = [], [], []
    for dev in poses:
        ps, pp_ = [], []
        for P in dots:
            ms = []
            for _ in range(repeats):
                o = sim.observe(subj, dev, P)
                if o is None:
                    continue
                f, label, truth = o
                ref = label if use_label else truth
                ms.append(prior.predict(f[:8]) - ref)
                ps.append(f[4:12])
            inacc.append(np.mean(ms, axis=0) if ms else np.zeros(2))
            if use_pupil:
                r = pupil_sensor.pupil_read(subj, dev, P, rng, cam)
                pp_.append(np.concatenate([r[0], r[1]]) if r is not None else np.zeros(3))
        eyestereo.append(np.mean(ps, axis=0) if ps else np.zeros(8))
        if use_pupil:
            pup.append(np.mean(pp_, axis=0))
    ia = np.array(inacc)
    if dc_remove:
        ia = ia - ia.mean(axis=0)
    return (ia.ravel(), np.concatenate(eyestereo),
            np.concatenate(pup) if use_pupil else np.zeros(0))


def _refine_geometry(asim, sub_th, samples, param_idx, pupil, iters=4, lam=1e-3):
    """Per-user Gauss-Newton REFINEMENT of the identified geometry on pose-diverse vernier
    corrections: adjust the selected descriptor params to minimize |label - physics_predict|
    across seats. The kappa/bias confound leaks bias INTO kappa here — which is exactly right
    for the PERCEIVED objective (the kappa+bias SUM is what places pixels where the user
    wants; KAPPA.md's 'product reframe'). Returns the refined descriptor."""
    import complete_geometry as cg
    import physics_preset as pp
    import numpy as np
    th = sub_th.copy()
    feats = [f for f, _ in samples]
    labels = np.array([l for _, l in samples])

    def resid(t):
        sub = cg.reconstruct_full(t)
        out = []
        for f, l in zip(feats, labels):
            b = pp.physics_predict(asim, sub, f, use_pupil=pupil, use_stereo=True)
            out.append((l - b) if b is not None else np.zeros(2))
        return np.concatenate(out)

    steps = {i: 0.05 for i in param_idx}               # finite-diff step per param (deg/mm scale)
    for _ in range(iters):
        r0 = resid(th)
        J = np.zeros((len(r0), len(param_idx)))
        for j, i in enumerate(param_idx):
            tp = th.copy(); tp[i] += steps[i]
            J[:, j] = (resid(tp) - r0) / steps[i]
        A = J.T @ J + lam * np.eye(len(param_idx))
        d = np.linalg.solve(A, J.T @ r0)
        for j, i in enumerate(param_idx):
            th[i] += np.clip(d[j], -1.0, 1.0)
    return th


def _fit_residual(kind, samples):
    """Per-user residual model from K (position, label-pred) pairs: MAD-gate the outliers
    (a wild pose-fit must not poison the model — the no-IR offset arm blew up exactly this
    way), then fit constant / affine / quadratic-in-position. Returns f(pos)->resid."""
    P = np.array([p for p, _ in samples])
    R = np.array([r for _, r in samples])
    med = np.median(R, axis=0)
    mad = np.median(np.abs(R - med), axis=0) * 1.4826 + 1e-9
    keep = np.all(np.abs(R - med) < 4 * mad, axis=1)
    P, R = P[keep], R[keep]
    if kind == "affine" and len(P) >= 10:
        A = np.c_[np.ones(len(P)), P]
        W = np.linalg.lstsq(A, R, rcond=None)[0]
        return lambda p: np.array([1.0, p[0], p[1]]) @ W
    if kind == "quad" and len(P) >= 16:
        A = np.c_[np.ones(len(P)), P, P**2, P[:, :1] * P[:, 1:]]
        W = np.linalg.lstsq(A, R, rcond=None)[0]
        return lambda p: np.array([1.0, p[0], p[1], p[0]**2, p[1]**2, p[0]*p[1]]) @ W
    m = R.mean(axis=0) if len(R) else med          # const (or fallback when too few kept)
    return lambda p: m


def _new_user(pop, sim):
    s = anatomy.sample_subject(pop)
    s.canthus_bias = sim.rng.normal(0, rig.CANTHUS_BIAS_SD, 4)
    s.human_bias = sim.rng.normal(0, rig.HUMAN_BIAS_SD, 2)
    return s


def evaluate(n_train=600, n_test=12, seed=0, verbose=True, followup=False):
    prior = match.train_prior()
    poses, dots = cg.dense_poses(), cg.dense_dots()
    cam = pupil_sensor.build_camera()
    sim = autosim.Simulator(0, use_pupil=True, use_stereo=True)
    rng = np.random.default_rng(100)
    ep_, ed = pixel_map.pose_grid(), pixel_map.dot_grid()[::2]
    vsd, measured = _vernier_sd()

    arms = [
        # name                        use_label  noise                 reps  dc     pupil  offK
        ("truth-fp (optimistic)",     False,     None,                 3,    False, True,  None, False, None),
        ("label dot-nudge",           True,      None,                 3,    False, True,  None, False, None),
        ("label VERNIER",             True,      (vsd, VERNIER_GROSS), 3,    False, True,  None, False, None),
        ("label vernier + avg6",      True,      (vsd, VERNIER_GROSS), 6,    False, True,  None, False, None),
        ("label nudge  + DC-removed", True,      None,                 3,    True,  True,  None, False, None),
        ("label vern+avg6 + DC-rm",   True,      (vsd, VERNIER_GROSS), 6,    True,  True,  None, False, None),
        ("NO-IR: vern+avg6+DC-rm",    True,      (vsd, VERNIER_GROSS), 6,    True,  False, None, False, None),
    ]
    if followup == "hybrid":
        # THE MISSED COMBINATION: v2's "DC-removal failed" arms had NO offset stage. But
        # kappa's ID error is ~pure DC in pixel space (a constant offset absorbs it), while
        # the 4.1->2.3 gap is the OTHER params' pose-varying error — which bias-free
        # (DC-removed) fingerprints should identify BETTER. Test DC-removed ID + offset.
        arms = [
            ("plain ID   + const8",    True, (vsd, VERNIER_GROSS), 6, False, True, ("const", 8), False, None),
            ("DC-rm ID   + const8",    True, (vsd, VERNIER_GROSS), 6, True,  True, ("const", 8), False, None),
            ("DC-rm ID   + const16",   True, (vsd, VERNIER_GROSS), 6, True,  True, ("const", 16), False, None),
        ]
    elif followup == "sdsweep":
        # WHAT DOES UNDER-3PX REQUIRE? Every label-driven trick failed; the gate is the
        # user's correction SD (kappa ID ~ linear in it). Sweep SD -> perceived px, so the
        # --vernier-demo measurement directly predicts the deployable number.
        arms = [
            ("SD 0.0013 (3x) + const8",  True, (0.0013, VERNIER_GROSS), 6, False, True, ("const", 8), False, None),
            ("SD 0.0008 (5x) + const8",  True, (0.0008, VERNIER_GROSS), 6, False, True, ("const", 8), False, None),
            ("SD 0.0004 (10x) + const8", True, (0.0004, VERNIER_GROSS), 6, False, True, ("const", 8), False, None),
        ]
    elif followup == "resid":
        # UNDER-3PX stage: the 4.3 px perceived residual is the FIELD-VARYING part of the
        # geometry-ID error (the constant part is already absorbed). Fit a richer per-user
        # residual model on vernier-grade corrections; score PERCEIVED (the target).
        # arms: (name, use_label, noise, reps, dc, pupil, resid=(kind,K), true_geom)
        # arms: (..., resid, true_geom, refine) — refine = (param names, n_seats, per_seat)
        arms = [
            ("vern+avg6 + const8",       True, (vsd, VERNIER_GROSS), 6, False, True, ("const", 8),   False, None),
            ("refine(kappa) + const8",   True, (vsd, VERNIER_GROSS), 6, False, True, ("const", 8),   False, (("kappa_x", "kappa_y"), 4, 8)),
            ("refine(k+geo) + const8",   True, (vsd, VERNIER_GROSS), 6, False, True, ("const", 8),   False, (("kappa_x", "kappa_y", "IPD", "cor_z"), 4, 8)),
            ("TRUE-GEOM + affine24",     True, (vsd, VERNIER_GROSS), 6, False, True, ("affine", 24), True,  None),
        ]
    elif followup:
        # FOLLOW-UP (after v2): DC-removal FAILED (kappa signal lives in the DC too), so the
        # deployable recipe = vernier ID + a per-user 2D OFFSET stage (a constant px shift
        # represents the bias exactly, which geometry cannot); plus the FAIR no-IR arms
        # (v2 ran no-IR under the failed DC protocol).
        arms = [
            ("vern+avg6 + offset(8)",  True, (vsd, VERNIER_GROSS), 6, False, True,  ("const", 8), False, None),
            ("NO-IR: vern+avg6",       True, (vsd, VERNIER_GROSS), 6, False, False, None,         False, None),
            ("NO-IR: vern+avg6+off8",  True, (vsd, VERNIER_GROSS), 6, False, False, ("const", 8), False, None),
        ]
    if verbose:
        print("== kappa-precision chain: correction quality -> kappa ID -> deployed px ==")
        print("   vernier SD = %.5f norm (%s)" % (vsd, "MEASURED user" if measured else
                                                  "assumed 3x; run --vernier-demo to measure"))
        print("   training %d eyes/arm, %d held-out users, stereo physics pipeline\n" % (n_train, n_test))

    kx = cg.FULL_NAMES.index("kappa_x")
    ky = cg.FULL_NAMES.index("kappa_y")
    sim_np = autosim.Simulator(0, use_pupil=False, use_stereo=True)   # the no-IR build
    identifiers = {}
    results = []
    for name, use_label, noise, reps, dc, pupil, resid, true_geom, refine in arms:
        asim = sim if pupil else sim_np
        key = (use_label, noise, dc, pupil)
        if true_geom:
            key = "TRUE"
            identifiers.setdefault(key, (None, np.ones(len(cg.FULL_NAMES))))
        if key not in identifiers:                     # train matched to the arm's protocol
            ctx = _correction_noise(*noise) if noise else _correction_noise(
                rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB)
            pop = np.random.default_rng(7)
            FP, FULL = [], []
            with ctx:
                for _ in range(n_train):
                    s = _new_user(pop, asim)
                    FP.append(_fingerprint(asim, s, prior, poses, dots, 4, rng, cam,
                                           use_label, dc, pupil))
                    FULL.append(cg.full_descriptor(s))
            FULLa = np.array(FULL)
            blocks = [np.array([f[i] for f in FP]) for i in (range(3) if pupil else range(2))]
            identifiers[key] = (_identifier(blocks, FULLa), FULLa.std(0))
        ident, pop_sd = identifiers[key]

        ctx = _correction_noise(*noise) if noise else _correction_noise(
            rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB)
        tpop = np.random.default_rng(54321)            # same users across arms (paired)
        errs, perc, krec = [], [], []
        with ctx:
            for _ in range(n_test):
                s = _new_user(tpop, asim)
                if true_geom:
                    th = cg.full_descriptor(s)
                else:
                    fp = _fingerprint(asim, s, prior, poses, dots, reps, rng, cam,
                                      use_label, dc, pupil)
                    th = ident([b[None] for b in (fp if pupil else fp[:2])])[0]
                if refine:                     # pose-diverse per-user geometry refinement
                    pnames, n_seats, per_seat = refine
                    pidx = [cg.FULL_NAMES.index(n) for n in pnames]
                    S = []
                    for _ in range(n_seats):       # "re-seat the glasses" between batches
                        dev, guard = asim.seat(), 0
                        while True:
                            got = sum(1 for _ in S)
                            if got >= per_seat * (len(S) // max(1, per_seat) + 1) or guard > per_seat * 6:
                                break
                            guard += 1
                            dev = asim.slip(dev)
                            o = asim.observe(s, dev, asim.world_point())
                            if o is not None:
                                S.append((o[0], o[1]))
                            if len(S) % per_seat == 0 and len(S) > 0:
                                break
                    if len(S) >= 8:
                        th = _refine_geometry(asim, th, S, pidx, pupil)
                krec.append(np.abs(th[[kx, ky]] - cg.full_descriptor(s)[[kx, ky]]))
                sub = cg.reconstruct_full(th)
                rmodel = lambda p: np.zeros(2)
                if resid:                      # per-user residual model from K extra corrections
                    kind, K = resid
                    dev, S, guard = asim.seat(), [], 0
                    while len(S) < K and guard < K * 6:
                        guard += 1
                        dev = asim.slip(dev)
                        o = asim.observe(s, dev, asim.world_point())
                        if o is None:
                            continue
                        b = pp.physics_predict(asim, sub, o[0], use_pupil=pupil, use_stereo=True)
                        if b is not None:
                            S.append((b, o[1] - b))         # (position, label - prediction)
                    if S:
                        rmodel = _fit_residual(kind, S)
                for dev in ep_:
                    for P in ed:
                        g = asim.ground_truth(s, dev, P)
                        if g is None:
                            continue
                        b = pp.physics_predict(asim, sub, g[0], use_pupil=pupil, use_stereo=True)
                        if b is not None:
                            bc = b + rmodel(b)
                            errs.append(np.linalg.norm(bc - g[1]))
                            perc.append(np.linalg.norm(bc - (g[1] + s.human_bias)))
        e = np.array(errs) * 1080
        pe = np.array(perc) * 1080
        kr = np.array(krec).mean(0)
        results.append((name, np.median(e), np.percentile(e, 95), kr, np.median(pe)))
        if verbose:
            print("  %-26s kappa %.3f/%.3f deg (%3.0f%%/%3.0f%%)  vs-truth %6.3f px (95th %6.2f)  PERCEIVED %6.3f px"
                  % (name, kr[0], kr[1], 100 * kr[0] / pop_sd[kx], 100 * kr[1] / pop_sd[ky],
                     np.median(e), np.percentile(e, 95), np.median(pe)))
    if verbose and not followup:
        base = next(r for r in results if r[0] == "label dot-nudge")
        best = min((r for r in results if "DC-rm" in r[0] and "NO-IR" not in r[0]),
                   key=lambda r: r[1])
        print("\n  => honest baseline %.3f px -> best protocol %.3f px vs truth / %.3f px "
              "perceived (truth-bound %.3f px)" % (base[1], best[1], best[4], results[0][1]))
    return results


if __name__ == "__main__":
    evaluate()
