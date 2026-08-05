"""Automated self-play calibration + cross-subject meta-learning (numpy + stdlib).

The loop, with no human:
  1. The simulator generates a subject (eye/face geometry) and seats the glasses.
  2. Each iteration the glasses slip slightly (natural drift), a dot is drawn, and the
     glasses-AI (a calibrator) predicts the overlay pixel from the camera features.
  3. The oracle ("the other AI") reports the TRUE pixel; we store that correction and
     refit. Repeat until the prediction has been accurate for `patience` trials in a
     row -> this subject is calibrated.
  4. Regenerate a new subject and repeat.

Across many subjects we pool every (features -> true pixel) sample and the geometry
descriptors into a SQLite meta-database, then fit a GLOBAL PRIOR. For a brand-new user
the prior already predicts most of the mapping from their eye-corner geometry, so a
warm-started calibrator needs far fewer samples than starting cold, that is the
streamlined-onboarding payoff, and the benchmark measures it.
"""
import os
import json
import sqlite3
import numpy as np

from config import Config
from calibrator import Calibrator
from autosim import Simulator
from main import new_calibrator


# ----------------------------------------------------------------------
#  Warm-start calibrator: a frozen population prior + a fast per-user residual
# ----------------------------------------------------------------------
class WarmCalibrator:
    """predict = prior(features) + small per-user residual correction."""

    def __init__(self, prior: Calibrator, cfg: Config):
        self.prior = prior
        # Residual is a per-session correction fit on very few samples, so it must be
        # strongly regularized: a high lambda floor keeps it ~a gentle offset and stops
        # it overfitting/blowing up when K ≈ the feature count (the K≈8 failure mode).
        self.resid = Calibrator(cfg.n_features, degree=1, min_samples=2,
                                lambdas=np.logspace(-1.0, 2.5, 15), robust_iters=1)
        self.lambda_ = None

    @property
    def is_trained(self):
        return True                       # the prior already predicts from sample 0

    def predict(self, x):
        base = self.prior.predict(x)
        if self.resid.is_trained:
            base = base + self.resid.predict_raw(x)
        return np.clip(base, 0.0, 1.0)

    def fit(self, X, Y, weights=None):
        X = np.atleast_2d(X)
        base = np.array([self.prior.predict(x) for x in X])
        self.resid.fit(X, np.atleast_2d(Y) - base, weights)
        self.lambda_ = self.resid.lambda_


# ----------------------------------------------------------------------
#  Meta-database
# ----------------------------------------------------------------------
class MetaDB:
    def __init__(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            descriptor TEXT NOT NULL,   -- json geometry descriptor (anatomy.DESCRIPTOR_NAMES)
            iters INTEGER NOT NULL      -- samples to converge (cold)
        );
        CREATE TABLE IF NOT EXISTS prior_samples (
            subject_id INTEGER NOT NULL,
            features TEXT NOT NULL,
            px REAL NOT NULL, py REAL NOT NULL,   -- human label (what you train on)
            tx REAL, ty REAL                       -- geometric truth (what you test on)
        );
        """)
        self.conn.commit()

    def add_subject(self, descriptor, iters, X, YL, YT):
        cur = self.conn.execute(
            "INSERT INTO subjects (descriptor, iters) VALUES (?,?)",
            (json.dumps([float(v) for v in descriptor]), int(iters)))
        sid = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO prior_samples (subject_id, features, px, py, tx, ty) "
            "VALUES (?,?,?,?,?,?)",
            [(sid, json.dumps([float(v) for v in f]),
              float(yl[0]), float(yl[1]), float(yt[0]), float(yt[1]))
             for f, yl, yt in zip(X, YL, YT)])
        self.conn.commit()

    def load_prior_samples(self):
        """Pooled (features -> human label), the prior is trained on labels."""
        rows = self.conn.execute("SELECT features, px, py FROM prior_samples").fetchall()
        if not rows:
            return np.empty((0, 8)), np.empty((0, 2))
        X = np.array([json.loads(r[0]) for r in rows])
        Y = np.array([[r[1], r[2]] for r in rows])
        return X, Y

    def load_subjects(self):
        """Per-subject (descriptor, X, label Y, truth Yt) for identifier + preset eval."""
        out = []
        for sid, desc in self.conn.execute("SELECT id, descriptor FROM subjects"):
            rows = self.conn.execute(
                "SELECT features, px, py, tx, ty FROM prior_samples "
                "WHERE subject_id=? ORDER BY rowid", (sid,)).fetchall()
            if not rows:
                continue
            X = np.array([json.loads(r[0]) for r in rows])
            YL = np.array([[r[1], r[2]] for r in rows])
            YT = np.array([[r[3], r[4]] for r in rows])
            out.append((np.array(json.loads(desc)), X, YL, YT))
        return out

    def stats(self):
        n = self.conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        ns = self.conn.execute("SELECT COUNT(*) FROM prior_samples").fetchone()[0]
        return n, ns

    def close(self):
        self.conn.close()


# ----------------------------------------------------------------------
#  One subject: closed-loop calibration to convergence
# ----------------------------------------------------------------------
def train_subject(sim, subject, cal, tol=0.013, max_iters=220, patience=10):
    """Returns (iters, X, label Y, truth Yt). Train on the human label; measure the
    overlay miss against geometric truth. 'converged' = the ROLLING-MEAN miss over the
    last `patience` attempts drops below tol (robust to single-frame noise spikes)."""
    dev = sim.seat()
    X, YL, YT = [], [], []
    window = []
    iters = max_iters
    for it in range(1, max_iters + 1):
        dev = sim.slip(dev)                       # glasses move slightly, naturally
        obs = sim.observe(subject, dev, sim.world_point())
        if obs is None:
            continue
        f, label, truth = obs
        err = float(np.linalg.norm(cal.predict(f) - truth))   # real overlay miss
        X.append(f); YL.append(label); YT.append(truth)
        cal.fit(np.array(X), np.array(YL))        # train on what the human gives
        window.append(err)
        if len(window) > patience:
            window.pop(0)
        if len(window) >= patience and float(np.mean(window)) < tol:
            iters = it
            break
    return iters, np.array(X), np.array(YL), np.array(YT)


# ----------------------------------------------------------------------
#  Build the database + global prior across many subjects
# ----------------------------------------------------------------------
def auto_train(n_subjects=40, seed=0, tol=0.013, max_iters=220,
               db_path=None, verbose=True):
    cfg = Config()
    sim = Simulator(seed)
    db_path = db_path or os.path.join(os.path.dirname(cfg.db_path), "meta.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = MetaDB(db_path)
    cold_iters = []
    if verbose:
        print("== auto-training across %d synthetic subjects ==" % n_subjects)
        print("%8s | %12s | %14s" % ("subject", "IPD/kappa", "iters->converge"))
        print("-" * 44)
    for s in range(n_subjects):
        subj = sim.new_subject()
        it, X, YL, YT = train_subject(sim, subj, new_calibrator(cfg), tol, max_iters)
        cold_iters.append(it)
        db.add_subject(subj.descriptor(), it, X, YL, YT)
        if verbose and (s + 1) % max(1, n_subjects // 10) == 0:
            print("%8d | %5.1f / %4.1f | %14d" %
                  (s + 1, subj.ipd, subj.kappa[0], it))
    X, Y = db.load_prior_samples()
    prior = new_calibrator(cfg)
    prior.fit(X, Y)
    if verbose:
        nsub, nsamp = db.stats()
        print("-" * 44)
        print("database: %d subjects, %d pooled samples" % (nsub, nsamp))
        print("cold-start iters-to-converge: median %.0f  mean %.1f"
              % (np.median(cold_iters), np.mean(cold_iters)))
        print("global prior fitted (lambda %.4f)" % (prior.lambda_ or 0))
    db.close()
    return prior, cold_iters, db_path


# ----------------------------------------------------------------------
#  Benchmark: warm-start (prior) vs cold-start on held-out new users
# ----------------------------------------------------------------------
def benchmark(prior, n_test=24, seed_base=7000, tol=0.013, max_iters=220):
    cfg = Config()
    cold, warm = [], []
    for i in range(n_test):
        # identical subject + observation sequence for both arms (same seed)
        sc = Simulator(seed_base + i); subj = sc.new_subject()
        cold.append(train_subject(sc, subj, new_calibrator(cfg), tol, max_iters)[0])
        sw = Simulator(seed_base + i); subj = sw.new_subject()
        warm.append(train_subject(sw, subj, WarmCalibrator(prior, cfg), tol, max_iters)[0])
    cold, warm = np.array(cold), np.array(warm)
    print("\n== warm-start benchmark on %d brand-new users ==" % n_test)
    print("samples to reach %.0f px accuracy:" % (tol * 1080))
    print("  cold start (from scratch) : median %.0f   mean %.1f"
          % (np.median(cold), cold.mean()))
    print("  warm start (from prior)   : median %.0f   mean %.1f"
          % (np.median(warm), warm.mean()))
    speedup = np.median(cold) / max(1e-9, np.median(warm))
    print("  -> %.1fx fewer calibration samples for a new user" % speedup)
    return cold, warm
