"""Large-scale run (100k+ eyes), numpy + stdlib, memory- and crash-safe.

Generation (the expensive part) is checkpointed to disk before any analysis, so a
crash in analysis never costs the 16-minute regen. Data is held as compact stacked
arrays (not 100k tuples), the prior is fit by STREAMING sufficient statistics, and the
identifier subsamples training subjects + uses batched prediction so memory stays bounded
(an unbounded identify step is what OOM-killed the first 100k attempt).

We give the extra data something to bite on: degree-3 prior (vs 2) and a polynomial-
expanded identifier, then report honestly whether 100k actually beats the 1000 run.
"""
import os
import time
import numpy as np

import autosim
import anatomy
from config import Config
from calibrator import _poly_features
from identify import MultiRidge
from autotrain import WarmCalibrator, train_subject
from main import new_calibrator


class StreamingPrior:
    """Polynomial ridge from streamed normal equations. Scales to millions of samples."""

    def __init__(self, degree=3, lam=2.0):
        self.degree = degree
        self.lam = lam
        self.lambda_ = lam

    def _design(self, X):
        Xs = (np.asarray(X, float) - self.xm) / self.xs
        P = _poly_features(Xs, self.degree)
        return np.column_stack([np.ones(P.shape[0]), P])

    def fit(self, X, Y, chunk=50000):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        self.xm = X.mean(0); self.xs = X.std(0); self.xs[self.xs < 1e-9] = 1.0
        m = self._design(X[:1]).shape[1]
        XtX = np.zeros((m, m)); XtY = np.zeros((m, Y.shape[1]))
        for i in range(0, len(X), chunk):
            P = self._design(X[i:i+chunk])
            XtX += P.T @ P; XtY += P.T @ Y[i:i+chunk]
        reg = self.lam * np.eye(m); reg[0, 0] = 0.0
        self.W = np.linalg.solve(XtX + reg, XtY)
        self.n_params = m
        return self

    def predict(self, x):
        return np.clip((self._design(np.atleast_2d(x)) @ self.W)[0], 0.0, 1.0)

    def predict_batch(self, X, chunk=100000):
        out = np.empty((len(X), self.W.shape[1]))
        for i in range(0, len(X), chunk):
            out[i:i+chunk] = self._design(X[i:i+chunk]) @ self.W
        return np.clip(out, 0.0, 1.0)


def _summary_poly(s, degree):
    return s if degree <= 1 else np.concatenate([s, _poly_features(s[None, :], degree)[0]])


def generate(n_subjects, m, seed, verbose=True):
    """Vary facial geometry (new_subject) x glasses pose (slip) x dot (point).
    Returns compact stacked arrays + per-subject offsets."""
    sim = autosim.Simulator(seed); t0 = time.time()
    Xs, YLs, YTs, descs, lengths = [], [], [], [], []
    for s in range(n_subjects):
        subj = sim.new_subject(); dev = sim.seat()
        xs, yl, yt = [], [], []
        tries = 0
        while len(xs) < m and tries < m * 5:
            tries += 1
            dev = sim.slip(dev); o = sim.observe(subj, dev, sim.world_point())
            if o is not None:
                xs.append(o[0]); yl.append(o[1]); yt.append(o[2])
        if len(xs) >= 8:
            Xs.append(np.array(xs)); YLs.append(np.array(yl)); YTs.append(np.array(yt))
            descs.append(subj.descriptor()); lengths.append(len(xs))
        if verbose and (s + 1) % 10000 == 0:
            print("  generated %6d / %d   (%.0fs)" % (s + 1, n_subjects, time.time() - t0))
    X = np.vstack(Xs); YL = np.vstack(YLs); YT = np.vstack(YTs)
    descs = np.array(descs); lengths = np.array(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    print("usable subjects: %d   total samples: %d   (%.0fs)"
          % (len(lengths), len(X), time.time() - t0))
    return X, YL, YT, descs, offsets


def analyze(X, YL, YT, descs, offsets, prior_degree=3, id_poly=2,
            identify_shots=(4, 8, 16), bench_users=30, max_id_train=20000,
            out_dir=".", verbose=True):
    t0 = time.time(); cfg = Config()
    prior = StreamingPrior(prior_degree, lam=2.0).fit(X, YL)
    print("prior: degree %d, %d params, %d samples" % (prior_degree, prior.n_params, len(X)))
    resids = YL - prior.predict_batch(X)          # once, batched
    n_sub = len(offsets) - 1
    pop_sd = descs.std(0)
    rng = np.random.default_rng(7); idx = rng.permutation(n_sub)
    ntr = int(0.8 * n_sub)
    tr = idx[:ntr][:max_id_train]                 # cap training subjects (memory + ample)
    te = idx[ntr:][:10000]

    def summ(i, K):
        a, e = offsets[i], offsets[i+1]
        b = min(a + K, e)
        return np.concatenate([X[a:b].mean(0), X[a:b].std(0), YL[a:b].mean(0),
                               resids[a:b].mean(0), resids[a:b].std(0)])

    names, units = anatomy.DESCRIPTOR_NAMES, anatomy.DESCRIPTOR_UNITS
    print("\n== geometry identification (%d eyes, poly-%d summary) ==" % (n_sub, id_poly))
    print("  K   | " + " | ".join("%-11s" % n for n in names))
    print("-" * (8 + 14 * len(names)))
    best = None
    for K in identify_shots:
        Str = np.array([_summary_poly(summ(i, K), id_poly) for i in tr])
        Ste = np.array([_summary_poly(summ(i, K), id_poly) for i in te])
        ident = MultiRidge().fit(Str, descs[tr])
        mae = np.abs(ident.predict(Ste) - descs[te]).mean(0)
        best = (K, mae, ident)
        print("  %-3d | " % K + " | ".join(
            "%4.2f(%3.0f%%)" % (mae[j], 100*mae[j]/max(pop_sd[j], 1e-9))
            for j in range(len(names))))
        del Str, Ste
    print("-" * (8 + 14 * len(names)))
    K, mae, ident = best
    print("at K=%d (MAE native units, %% of population spread):" % K)
    for j in range(len(names)):
        pct = 100 * mae[j] / max(pop_sd[j], 1e-9)
        v = "well identified" if pct < 35 else "partial" if pct < 70 else "weak"
        print("  %-12s %6.2f %-3s  pop SD %5.2f  -> %s" % (names[j], mae[j], units[j], pop_sd[j], v))

    cold, warm = [], []
    for i in range(bench_users):
        sc = autosim.Simulator(900000 + i); su = sc.new_subject()
        cold.append(train_subject(sc, su, new_calibrator(cfg))[0])
        sw = autosim.Simulator(900000 + i); su = sw.new_subject()
        warm.append(train_subject(sw, su, WarmCalibrator(prior, cfg))[0])
    print("\n== warm-start with the %d-eye prior (%d held-out users) ==" % (n_sub, bench_users))
    print("  cold median %.0f   warm median %.0f   -> %.1fx fewer answers"
          % (np.median(cold), np.median(warm), np.median(cold)/max(np.median(warm), 1e-9)))

    np.savez(os.path.join(out_dir, "mega_prior.npz"),
             W=prior.W, xm=prior.xm, xs=prior.xs, degree=prior.degree)
    np.savez(os.path.join(out_dir, "mega_identifier.npz"),
             W=ident.W, xm=ident.xm, xs=ident.xs, ym=ident.ym, ys=ident.ys,
             K=K, id_poly=id_poly)
    print("\nsaved prior + identifier to %s   (analysis %.0fs)" % (out_dir, time.time()-t0))
    return prior, ident


def run(n_subjects=100000, m=24, seed=0, out_dir=None):
    cfg = Config()
    out_dir = out_dir or os.path.dirname(cfg.db_path)
    X, YL, YT, descs, offsets = generate(n_subjects, m, seed)
    ckpt = os.path.join(out_dir, "mega_samples.npz")
    np.savez(ckpt, X=X, YL=YL, YT=YT, descs=descs, offsets=offsets)   # crash-safe checkpoint
    print("checkpointed samples -> %s" % ckpt)
    return analyze(X, YL, YT, descs, offsets, out_dir=out_dir)


def analyze_checkpoint(out_dir=None):
    """Re-run analysis from a saved checkpoint (no regeneration)."""
    cfg = Config(); out_dir = out_dir or os.path.dirname(cfg.db_path)
    d = np.load(os.path.join(out_dir, "mega_samples.npz"))
    return analyze(d["X"], d["YL"], d["YT"], d["descs"], d["offsets"], out_dir=out_dir)
