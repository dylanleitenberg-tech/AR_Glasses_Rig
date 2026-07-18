"""Geometry identification by SIGNATURE MATCHING — the database idea, tested honestly.

The approach: build a database of simulated faces whose eye geometry we KNOW
(including the hard-to-image internals: entrance-pupil depth, globe radius, angle kappa).
For each face record its *fingerprint* = the guess INACCURACY pattern across a fixed set of
glasses positions and gaze directions. When a new user arrives, measure their fingerprint
the same way and MATCH it to the nearest database face(s); read that face's geometry off as
the estimate.

This is nearest-neighbour inversion of the geometry -> fingerprint map. It recovers a
parameter only to the extent that parameter leaves a SEPARABLE mark on the fingerprint
(identifiability). This script measures, per parameter:
  * recovery error (MAE in native units and as % of the population spread), and
  * the neighbour SPREAD (how tightly the matched faces agree) — a direct read on
    identifiability: a big spread means many different eyes share the fingerprint.

It also runs the IMU ablation: does adding the glasses tilt sensor ("level") to the
fingerprint improve recovery? Compared feature sets:
    inacc            guess inaccuracy only         (the core idea)
    inacc+eyecam     + eye-corner camera readings   (full realistic observation)
    inacc+eyecam+imu(raw)         + level, head confound included
    inacc+eyecam+imu(comp)        + level, head pose compensated out

Why the IMU might help geometry ID specifically: pupil-depth and globe-radius are PARALLAX
terms — they shift the inaccuracy as the glasses tilt. The eye-corner cameras sense tilt
only indirectly; a level senses it directly, so it can pin the pantoscopic tilt and let the
matcher attribute the rest of the inaccuracy to parallax. Whether that beats the noise floor
is an empirical question — this answers it.

Run:  python3 match.py        (or:  python3 main.py --match)
"""
import numpy as np

import autosim
import anatomy
import rig
import imu
import pupil_sensor
from config import Config
from main import new_calibrator


# fixed probe conditions, applied identically to every face so fingerprints are comparable
def probe_poses():
    """A few glasses positions spanning the slip envelope (deg/mm deviations from rest)."""
    poses = []
    for dty in (-1.0, 2.0, 5.0):                 # mm slid down the nose
        drx = dty * rig.PITCH_PER_MM             # correlated pantoscopic tilt
        for dtx in (-1.5, 1.5):                  # mm lateral
            poses.append(np.array([drx, 0, 0, dtx, dty, 0.0]))
    return poses                                  # 6 positions


def probe_dots():
    """Gaze directions: central dots so they stay on-screen for essentially every face."""
    xs = (-90.0, 0.0, 90.0); ys = (-60.0, 60.0)
    return [np.array([x, y, rig.DOT_Z_MEAN]) for y in ys for x in xs]   # 6 directions


def _avg_observe(sim, subj, dev, P, prior, repeats):
    """Average over `repeats` noisy captures: signed guess inaccuracy + mean eye-corner read.
    Returns (inacc(2,), eyecam(4,)) or None if the dot won't stay on-screen."""
    miss = []; corners = []
    tries = 0
    while len(miss) < repeats and tries < repeats * 6:
        tries += 1
        o = sim.observe(subj, dev, P)
        if o is None:
            continue
        f, _label, truth = o
        miss.append(prior.predict(f) - truth)    # the generic guess's signed error
        corners.append(f[4:8])                    # eye-corner camera readings
    if not miss:
        return None
    return np.mean(miss, axis=0), np.mean(corners, axis=0)


def _avg_pupil(subj, dev, P, repeats, rng, cam):
    """Average NIR pupil-centre + iris-diameter read over repeats (3,) or zeros if unseen."""
    out = []
    for _ in range(repeats):
        r = pupil_sensor.pupil_read(subj, dev, P, rng, cam)
        if r is not None:
            out.append(np.concatenate([r[0], r[1]]))
    return np.mean(out, axis=0) if out else np.zeros(3)


def fingerprint(sim, subj, prior, poses, dots, repeats, rng, cam):
    """Fingerprint blocks for one face: inaccuracy, eye-corner, IMU(raw/comp), pupil sensor."""
    inacc, eyecam, imu_raw, imu_cmp, pupil = [], [], [], [], []
    for dev in poses:
        imu_raw.append(imu.imu_tilt(dev, rng, compensated=False))
        imu_cmp.append(imu.imu_tilt(dev, rng, compensated=True))
        per_pose_corner = []
        for P in dots:
            r = _avg_observe(sim, subj, dev, P, prior, repeats)
            if r is None:
                inacc.append([0.0, 0.0]); per_pose_corner.append([0, 0, 0, 0])
            else:
                inacc.append(r[0]); per_pose_corner.append(r[1])
            pupil.append(_avg_pupil(subj, dev, P, repeats, rng, cam))   # NIR pupil/iris
        eyecam.append(np.mean(per_pose_corner, axis=0))
    return (np.concatenate(inacc), np.concatenate(eyecam),
            np.concatenate(imu_raw), np.concatenate(imu_cmp), np.concatenate(pupil))


def train_prior(seed=7, n_faces=250, per=70):
    """Generic population guesser: features -> registration pixel, fit across many faces."""
    sim = autosim.Simulator(seed)
    X, Y = [], []
    for _ in range(n_faces):
        subj = sim.new_subject(); dev = sim.seat(); got = guard = 0
        while got < per and guard < per * 4:
            guard += 1; dev = sim.slip(dev)
            o = sim.observe(subj, dev, sim.world_point())
            if o is not None:
                X.append(o[0]); Y.append(o[2]); got += 1
    prior = new_calibrator(Config()); prior.fit(np.array(X), np.array(Y))
    return prior


def build_db(prior, n_faces, seed, repeats=4, verbose=True):
    sim = autosim.Simulator(seed)
    rng = np.random.default_rng(seed + 100)
    cam = pupil_sensor.build_camera()
    poses, dots = probe_poses(), probe_dots()
    INA, EYE, IMR, IMC, PUP, DESC = [], [], [], [], [], []
    for i in range(n_faces):
        subj = sim.new_subject()
        ia, ey, ir, ic, pu = fingerprint(sim, subj, prior, poses, dots, repeats, rng, cam)
        INA.append(ia); EYE.append(ey); IMR.append(ir); IMC.append(ic); PUP.append(pu)
        DESC.append(subj.descriptor())
        if verbose and (i + 1) % 500 == 0:
            print("    ... built %d/%d faces" % (i + 1, n_faces))
    return (np.array(INA), np.array(EYE), np.array(IMR), np.array(IMC),
            np.array(PUP), np.array(DESC))


def _knn_recover(train_sig, train_desc, test_sig, k=8):
    """Standardize, Euclidean k-NN; return recovered descriptors + neighbour spread."""
    mu, sd = train_sig.mean(0), train_sig.std(0); sd[sd < 1e-9] = 1.0
    A = (train_sig - mu) / sd; B = (test_sig - mu) / sd
    rec = np.zeros((len(B), train_desc.shape[1]))
    spread = np.zeros_like(rec)
    for i, b in enumerate(B):
        d = np.linalg.norm(A - b, axis=1)
        nn = np.argpartition(d, k)[:k]
        rec[i] = train_desc[nn].mean(0)
        spread[i] = train_desc[nn].std(0)
    return rec, spread


def evaluate(n_db=3000, n_test=300, seed=0, k=8, repeats=4, verbose=True):
    cfg = Config()
    names, units = anatomy.DESCRIPTOR_NAMES, anatomy.DESCRIPTOR_UNITS
    if verbose:
        print("== geometry identification by fingerprint matching ==")
        print("  training the population prior (generic guesser) ...")
    prior = train_prior()
    if verbose:
        print("  building database of %d faces + %d held-out users "
              "(%d positions x %d gaze dirs x %d repeats) ..."
              % (n_db, n_test, len(probe_poses()), len(probe_dots()), repeats))
    ina, eye, imr, imc, pup, desc = build_db(prior, n_db + n_test, seed, repeats, verbose)
    tr = slice(0, n_db); te = slice(n_db, n_db + n_test)
    pop_sd = desc.std(0)

    variants = [
        ("inacc",                [ina]),
        ("inacc+eyecam",         [ina, eye]),
        ("inacc+eyecam+imu_comp",[ina, eye, imc]),
        ("inacc+eyecam+pupil",   [ina, eye, pup]),
        ("all (eyecam+imu+pupil)",[ina, eye, imc, pup]),
    ]
    print("\n  recovery MAE as %% of population SD  (lower = better; 100%% = no information)")
    print("  %-22s | " % "feature set" + " | ".join("%-9s" % n for n in names))
    print("  " + "-" * (24 + 12 * len(names)))
    results = {}
    for label, blocks in variants:
        S = np.concatenate(blocks, axis=1)
        rec, spread = _knn_recover(S[tr], desc[tr], S[te], k)
        mae = np.abs(rec - desc[te]).mean(0)
        pct = 100 * mae / np.maximum(pop_sd, 1e-9)
        results[label] = (mae, pct, spread.mean(0))
        print("  %-22s | " % label + " | ".join("%6.0f%% " % pct[j] for j in range(len(names))))
    print("  " + "-" * (24 + 12 * len(names)))

    # focus on the two hard internals + what each added sensor does for them
    base = results["inacc+eyecam"][1]
    imu_c = results["inacc+eyecam+imu_comp"][1]
    pup = results["inacc+eyecam+pupil"][1]
    print("\n  hard-to-image internals (recovery, %% of pop SD; lower = better):")
    print("    %-9s %10s %10s %10s   %s" % ("", "eyecam", "+imu", "+pupil", "native MAE w/pupil"))
    for key in ("ep_dist", "globe_r"):
        j = names.index(key)
        m = results["inacc+eyecam+pupil"][0][j]
        print("    %-9s %9.0f%% %9.0f%% %9.0f%%   %.2f %s (pop SD %.2f)"
              % (key, base[j], imu_c[j], pup[j], m, units[j], pop_sd[j]))
    dimu = (base - imu_c).mean(); dpup = (base[[names.index('ep_dist'), names.index('globe_r')]]
                                          - pup[[names.index('ep_dist'), names.index('globe_r')]]).mean()
    print("\n  IMU (level) marginal effect, avg all params : %+.1f pp of pop SD  -> %s"
          % (dimu, "helps" if dimu > 3 else "adds little (as predicted)"))
    print("  PUPIL sensor marginal effect on ep_dist+globe_r: %+.1f pp of pop SD  -> %s"
          % (dpup, "breaks the wall" if dpup > 10 else "helps" if dpup > 3 else "little"))
    print("\n  identifiability note: kappa shares its signature with perceptual bias, so its")
    print("  recovered value is the kappa+bias SUM (all you need to place the pixel); the")
    print("  neighbour spread columns show which params are genuinely pinned vs. ambiguous.")
    return results


if __name__ == "__main__":
    evaluate()
