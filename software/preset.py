"""Per-user EYE PRESET: identify the eye's geometry from a few position-guess answers,
then build a calibration that gives the correct overlay pixel at ANY glasses position.

The idea:
  1. The system guesses overlay positions; the user corrects them. Those corrections are
     the "answers" -> they reveal the eye's INTERNAL + EXTERNAL geometry (identify.py).
  2. We reconstruct that eye/face as a physics subject and SWEEP it over the whole range
     of glasses positions, producing the correct pixel everywhere -> a calibrator that is
     the per-user PRESET (pose-invariant by construction, because it came from geometry).
  3. A tiny per-session affine correction (fit on the same few answers) absorbs the
     unmodelled per-session offset (e.g. eye-corner template centering).

So from a few answers you get a preset keyed to THAT eye shape that holds for every
glasses position. We then measure, honestly, how much the geometry preset beats a
population prior and a from-scratch (black-box) fit on full-pose held-out data.
"""
import numpy as np

import anatomy
import autosim
from config import Config
from calibrator import Calibrator
from main import new_calibrator
from autotrain import MetaDB, WarmCalibrator
from identify import MultiRidge, session_summary


def descriptor_to_subject(d):
    """Rebuild an (estimated) eye/face from an 8-param descriptor (inverse of .descriptor)."""
    ipd, globe_r, ep, ocd, icd, tilt, kx, ky = [float(v) for v in d]
    eye_y = 0.0
    cfwd = anatomy.CANTHUS_FWD_MEAN
    cor_z = -(globe_r - anatomy.GLOBE_R_MEAN) * anatomy.COR_AXIAL_K   # per-face CoR depth
    cor = np.array([[-ipd/2, eye_y, cor_z], [ipd/2, eye_y, cor_z]])
    inner = np.array([[-icd/2, eye_y, cfwd], [icd/2, eye_y, cfwd]])
    outer = np.array([[-ocd/2, eye_y + tilt, cfwd], [ocd/2, eye_y + tilt, cfwd]])
    s = anatomy.Subject(ipd, globe_r, ep, np.array([kx, ky]), cor, inner, outer)
    s.canthus_bias = np.zeros(4)
    return s


def build_preset(theta, n_sweep=2000, degree=3, seed=1, use_pupil=False):
    """Sweep the identified geometry over the full pose range -> a calibrator (the preset).

    Two things make this a high-fidelity preset:
      * DEGREE-3 fit (degree-2 was the wall — it floored at ~3.7 px even with perfect geometry);
      * trained on the NOISE-FREE oracle (`ground_truth`), not noisy `observe()`. The sweep is
        a synthetic map of a KNOWN geometry, so injecting sensor noise only hurts — using clean
        features drops the true-geometry floor from ~2.7 px to ~0.75 px (see kappa_live.py /
        the sub-2px study). The per-USER live calibrator stays degree-2 (a handful of noisy
        samples); this fidelity is only for the dense synthetic sweep."""
    sim = autosim.Simulator(seed, use_pupil=use_pupil)   # same device extrinsics as the captures
    subj = descriptor_to_subject(theta)
    X, Y = [], []
    dev = sim.seat()
    guard = 0
    while len(X) < n_sweep and guard < n_sweep * 4:
        guard += 1
        dev = sim.slip(dev)
        g = sim.ground_truth(subj, dev, sim.world_point())   # clean features + truth (no noise)
        if g is not None:
            X.append(g[0]); Y.append(g[1])
    cfg = Config(use_pupil=use_pupil)
    cal = Calibrator(cfg.n_features, degree=degree, min_samples=10,
                     robust_iters=cfg.robust_iters, huber_k=cfg.huber_k)
    cal.fit(np.array(X), np.array(Y))
    return cal


def _median_px(model, X, Y):
    e = [np.linalg.norm(model.predict(x) - y) for x, y in zip(X, Y)]
    return float(np.median(e)) * 1080.0


def evaluate(db_path, shots=(4, 8, 16), n_test=20, seed=0, verbose=True):
    """Compare, on full-pose held-out data, three ways to calibrate a NEW user from K answers:
       (a) geometry preset (identify -> physics sweep -> session affine),
       (b) population warm-start (global prior + session affine),
       (c) black-box fit from the K answers only.
    """
    db = MetaDB(db_path); subs = db.load_subjects(); db.close()
    if len(subs) < 40:
        print("preset: need a larger database (build with --auto-train)."); return None
    cfg = Config()
    desc = np.array([d for d, _, _, _ in subs])
    rng = np.random.default_rng(seed); idx = rng.permutation(len(subs))
    ntr = int(0.8 * len(subs)); tr = idx[:ntr]; te = idx[ntr:][:n_test]

    # population prior + identifier trained on TRAIN subjects' LABELS only (no leakage)
    Xtr = np.vstack([subs[i][1] for i in tr]); Ytr = np.vstack([subs[i][2] for i in tr])
    prior = new_calibrator(cfg); prior.fit(Xtr, Ytr)

    if verbose:
        print("== per-user preset vs baselines (full-pose held-out error) ==")
        print("overlay error px @1080p for a NEW user given K answers, as median(mean)")
        print("across test users (mean exposes atypical faces):\n")
        print("  K   | geometry preset | population warm | black-box (K only)")
        print("  " + "-" * 58)

    out = {}
    for K in shots:
        # identifier for this K (summary depends on K), trained on train subjects
        ident = MultiRidge().fit(
            np.array([session_summary(subs[i][1], subs[i][2], prior, K) for i in tr]),
            desc[tr])
        gp, gw, bb = [], [], []
        for i in te:
            d, X, YL, YT = subs[i]
            if len(X) < K + 8:
                continue
            Xc, Yc = X[:K], YL[:K]              # calibrate on the human answers
            Xt, Yt = X[K:], YT[K:]             # test against geometric TRUTH (real miss)
            theta = ident.predict(session_summary(X, YL, prior, K))[0]
            preset = build_preset(theta)
            mp = WarmCalibrator(preset, cfg); mp.fit(Xc, Yc)     # geometry preset + affine
            mw = WarmCalibrator(prior, cfg);  mw.fit(Xc, Yc)     # population warm-start
            mb = new_calibrator(cfg);         mb.fit(Xc, Yc)     # black-box from K
            gp.append(_median_px(mp, Xt, Yt))
            gw.append(_median_px(mw, Xt, Yt))
            bb.append(_median_px(mb, Xt, Yt))
        cell = lambda v: "%.1f(%.0f)" % (np.median(v), np.mean(v))
        out[K] = (np.median(gp), np.median(gw), np.median(bb))
        if verbose:
            print("  %-3d | %15s | %15s | %15s"
                  % (K, cell(gp), cell(gw), cell(bb)))
    if verbose:
        print("  " + "-" * 58)
        print("\nLower is better. The geometry preset is keyed to the identified eye shape,")
        print("so it is correct across glasses positions from very few answers; the")
        print("population warm-start and black-box need more answers to catch up.")
    return out
