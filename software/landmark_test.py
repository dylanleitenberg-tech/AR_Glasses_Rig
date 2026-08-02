"""Which canthus should the eye-corner cameras track — OUTER or INNER?

The rig was designed around the OUTER (lateral) canthus. Mounting the reprinted carrier
raised the question of whether the INNER (medial) canthus is the better landmark, so this
runs the two head-to-head through the identical pipeline.

Why it could go either way:
  * OUTER sits ~25 mm from the temple-mounted camera and swings through a longer arc as the
    glasses slip, so it has more image motion per mm of slip — more signal.
  * INNER sits ~53 mm away (smaller, further, more foreshortened) BUT is anchored by the
    medial canthal tendon to bone, whereas the lateral canthus rides the soft tissue that
    forms crow's feet. In hardware that should make it steadier under expression.

The simulator has no measured basis for the second claim, so the default A/B holds
soft-tissue motion EQUAL for both landmarks and measures pure geometry. `--soft-sweep` then
shows how much steadier the inner canthus would have to be for it to win on that account.

Both arms are driven from one fixed list of faces (identical geometry, identical poses), so
the comparison is paired rather than two independent samples.

Run:  python3 landmark_test.py --selftest      # fast plumbing checks (in the release gate)
      python3 landmark_test.py                 # the full A/B
      python3 landmark_test.py --soft-sweep    # + soft-tissue sensitivity
"""
import sys

import numpy as np

import anatomy
import autosim
import match
import pixel_map
import rig
from accuracy_map import _identifier
from preset import build_preset

LANDMARKS = ("outer", "inner")


# ----------------------------------------------------------------------
#  Paired subjects: one fixed population both arms see
# ----------------------------------------------------------------------
def make_subjects(n, seed):
    """Faces built from a DEDICATED rng, so neither arm's rng consumption can shift them.

    Mirrors Simulator.new_subject (which draws from the sim's own rng — unusable here,
    because the two arms reject captures at different rates and would desynchronize).
    """
    rng = np.random.default_rng(seed)
    subs = []
    for _ in range(n):
        s = anatomy.sample_subject(rng)
        s.canthus_bias = rng.normal(0, rig.CANTHUS_BIAS_SD, 4)
        s.human_bias = rng.normal(0, rig.HUMAN_BIAS_SD, 2)
        s.cdrift = np.zeros(4)
        subs.append(s)
    return subs


def _fresh(subs):
    """Reset the per-session drift so each arm starts every face from the same state."""
    for s in subs:
        s.cdrift = np.zeros(4)
    return subs


# ----------------------------------------------------------------------
#  Framing: where does the landmark actually land on the sensor?
# ----------------------------------------------------------------------
def framing(landmark, n_faces=200, seed=31):
    """Landmark position in the eye cam, against the REAL 1280x800 sensor crop.

    optics.PinholeCamera normalizes u and v by the same focal length, i.e. it models a
    SQUARE frame; the OV9281 is 16:10, so the true vertical extent is 800/1280 of the
    horizontal one. Anything outside that band is off the real sensor.
    """
    cam = rig.build(landmark)["eye"][1]
    half = 0.5 * (800 / 1280.0)
    lo, hi = 0.5 - half, 0.5 + half
    subs = make_subjects(n_faces, seed)
    uv = []
    for s in subs:
        pts = s.outer_canthus if landmark == "outer" else s.inner_canthus
        for dev in pixel_map.pose_grid():
            R, t = rig.pose_R_t(dev)
            p = cam.project(((R @ pts.T).T + t)[1])
            if p is not None:
                uv.append(p)
    uv = np.array(uv)
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u <= 1) & (v >= lo) & (v <= hi)
    return dict(u=np.median(u), v=np.median(v), v_lo=np.percentile(v, 1),
                inframe=100 * inside.mean(),
                spread=np.hypot(u.std(), v.std()))


# ----------------------------------------------------------------------
#  One arm of the A/B
# ----------------------------------------------------------------------
def run_arm(landmark, subs_train, subs_test, repeats=4, seed=0,
            soft_sd=None, verbose=True):
    """Train prior + identifier on subs_train, then measure per-position overlay error."""
    if verbose:
        print("  [%s] training prior + identifier on %d faces ..."
              % (landmark, len(subs_train)))
    prior = match.train_prior(landmark=landmark)
    poses, dots = match.probe_poses(), match.probe_dots()
    cam = match.pupil_sensor.build_camera()

    sim = autosim.Simulator(seed, landmark=landmark)
    sim.soft_tissue_override = soft_sd
    rng = np.random.default_rng(seed + 100)
    INA, EYE, DESC = [], [], []
    for s in _fresh(subs_train):
        ia, ey, _ir, _ic, _pu = match.fingerprint(sim, s, prior, poses, dots, repeats, rng, cam)
        INA.append(ia); EYE.append(ey); DESC.append(s.descriptor())
    INA, EYE, DESC = map(np.array, (INA, EYE, DESC))
    ident = _identifier([INA, EYE], DESC)

    eval_poses, eval_dots = pixel_map.pose_grid(), pixel_map.dot_grid()
    tsim = autosim.Simulator(seed + 12345, landmark=landmark)
    tsim.soft_tissue_override = soft_sd
    trng = np.random.default_rng(seed + 999)
    if verbose:
        print("  [%s] measuring %d held-out faces x %d positions ..."
              % (landmark, len(subs_test), len(eval_poses)))
    err, desc_err = [], []
    for s in _fresh(subs_test):
        ia, ey, _ir, _ic, _pu = match.fingerprint(tsim, s, prior, poses, dots, repeats, trng, cam)
        theta = ident([ia[None], ey[None]])[0]
        desc_err.append(np.abs(theta - s.descriptor()))
        pre = build_preset(theta, landmark=landmark)
        for dev in eval_poses:
            e = []
            for P in eval_dots:
                g = tsim.ground_truth(s, dev, P)
                if g is None:
                    continue
                f, truth = g
                e.append(np.linalg.norm(pre.predict(f) - truth))
            if e:
                err.append(np.median(e) * 1080.0)
    return np.array(err), np.array(desc_err)


# ----------------------------------------------------------------------
#  The comparison
# ----------------------------------------------------------------------
def compare(n_train=800, n_test=20, seed=0, soft_sweep=False, verbose=True):
    subs_train = make_subjects(n_train, seed + 4242)
    subs_test = make_subjects(n_test, seed + 8888)

    print("== tracked-landmark A/B: OUTER vs INNER canthus ==")
    print("   %d training faces / %d held-out, %d glasses positions x %d dots, paired faces\n"
          % (n_train, n_test, len(pixel_map.pose_grid()), len(pixel_map.dot_grid())))

    print("  where the landmark lands on the REAL 1280x800 sensor:")
    print("    %-7s %8s %8s %10s %12s" % ("", "u", "v", "in-frame", "image spread"))
    for lm in LANDMARKS:
        f = framing(lm)
        print("    %-7s %8.3f %8.3f %9.1f%% %12.4f"
              % (lm, f["u"], f["v"], f["inframe"], f["spread"]))
    print("    (image spread = how much the landmark MOVES across the slip envelope;")
    print("     more motion per mm of slip = more signal for the pose fit)\n")

    res = {}
    for lm in LANDMARKS:
        err, desc_err = run_arm(lm, subs_train, subs_test, seed=seed, verbose=verbose)
        res[lm] = (err, desc_err)

    print("\n  systematic overlay error vs the oracle (px @1080p, every face x position):")
    print("    %-7s %9s %9s %9s" % ("landmark", "median", "95th", "max"))
    for lm in LANDMARKS:
        e = res[lm][0]
        print("    %-7s %9.2f %9.2f %9.2f"
              % (lm, np.median(e), np.percentile(e, 95), e.max()))
    mo, mi = np.median(res["outer"][0]), np.median(res["inner"][0])
    verdict = "INNER better" if mi < mo else "OUTER better"
    print("\n    -> %s by %.2f px median (%.0f%%)"
          % (verdict, abs(mi - mo), 100 * abs(mi - mo) / max(mo, 1e-9)))

    print("\n  geometry recovery (MAE as %% of population spread; lower = better ID):")
    names = anatomy.DESCRIPTOR_NAMES
    pop_sd = np.array([s.descriptor() for s in subs_train]).std(0)
    print("    %-13s %10s %10s" % ("param", "outer", "inner"))
    for j, nm in enumerate(names):
        print("    %-13s %9.0f%% %9.0f%%"
              % (nm, 100 * res["outer"][1][:, j].mean() / max(pop_sd[j], 1e-9),
                 100 * res["inner"][1][:, j].mean() / max(pop_sd[j], 1e-9)))

    if soft_sweep:
        print("\n  soft-tissue sensitivity — how each landmark degrades as its per-capture")
        print("  motion grows (rig.SOFT_TISSUE_SD = %.2f mm is the current placeholder):"
              % rig.SOFT_TISSUE_SD)
        print("    %-10s %12s %12s" % ("motion mm", "outer px", "inner px"))
        for sd in (0.0, 0.15, 0.30, 0.60):
            row = []
            for lm in LANDMARKS:
                e, _ = run_arm(lm, subs_train[:300], subs_test[:10], seed=seed,
                               soft_sd=sd, verbose=False)
                row.append(np.median(e))
            print("    %-10.2f %12.2f %12.2f" % (sd, row[0], row[1]))
        print("    (the inner canthus wins on this axis only if its real motion is enough")
        print("     LOWER than the outer's to overcome any geometry deficit above)")
    return res


# ----------------------------------------------------------------------
#  Selftest — plumbing + regression guard
# ----------------------------------------------------------------------
def selftest():
    ok = True

    # 1. the default path is untouched: build() == build("outer")
    a, b = rig.build(), rig.build("outer")
    same = all(np.allclose(ca.pos, cb.pos) and np.allclose(ca.R, cb.R)
               for ca, cb in zip(a["eye"], b["eye"]))
    print("  default build == build('outer'):            %s" % ("PASS" if same else "FAIL"))
    ok &= same

    # 2. inner build moves the AIM but not the camera POSITION (aim-only experiment)
    c = rig.build("inner")
    pos_same = all(np.allclose(ca.pos, cc.pos) for ca, cc in zip(a["eye"], c["eye"]))
    aim_diff = not any(np.allclose(ca.R, cc.R) for ca, cc in zip(a["eye"], c["eye"]))
    print("  inner: positions identical, aim differs:    %s"
          % ("PASS" if (pos_same and aim_diff) else "FAIL"))
    ok &= pos_same and aim_diff

    # 3. the inner arm really tracks the inner canthus
    subs = make_subjects(1, 3)
    dev = pixel_map.pose_grid()[0]
    so = autosim.Simulator(0, landmark="outer")
    si = autosim.Simulator(0, landmark="inner")
    P = np.array([0.0, 0.0, rig.DOT_Z_MEAN])
    go, gi = so.ground_truth(subs[0], dev, P), si.ground_truth(subs[0], dev, P)
    moved = go is not None and gi is not None and not np.allclose(go[0][4:8], gi[0][4:8])
    print("  inner arm reports a different landmark:     %s" % ("PASS" if moved else "FAIL"))
    ok &= moved

    # 4. the registration TRUTH is landmark-independent (cameras observe; they don't
    #    change where the pixel must go)
    truth_same = go is not None and gi is not None and np.allclose(go[1], gi[1])
    print("  true pixel unchanged by landmark choice:    %s" % ("PASS" if truth_same else "FAIL"))
    ok &= truth_same

    # 5. feature contract intact: still 8 features, same order
    n_ok = go is not None and len(go[0]) == 8 and len(gi[0]) == 8
    print("  feature count still 8 in both arms:         %s" % ("PASS" if n_ok else "FAIL"))
    ok &= n_ok

    # 6. both landmarks stay on the real sensor at the design aim
    fo, fi = framing("outer", n_faces=40), framing("inner", n_faces=40)
    fr = fo["inframe"] > 90 and fi["inframe"] > 90
    print("  both land on the real sensor (>90%%):        %s  (outer %.0f%%, inner %.0f%%)"
          % ("PASS" if fr else "FAIL", fo["inframe"], fi["inframe"]))
    ok &= fr

    print("landmark_test selftest: %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    compare(soft_sweep="--soft-sweep" in sys.argv)
