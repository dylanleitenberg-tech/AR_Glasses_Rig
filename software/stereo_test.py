"""Does a SECOND eye-corner camera (stereo) break under 1 px deployed?

The 1.26 px deployed floor is the canthi-depth ambiguity of a single eye-corner view. A
second view triangulates that depth, so it should (a) improve the cfwd/tilt geometry ID and
(b) better-constrain the physics pose-fit. This compares, on held-out users, the complete-
geometry physics preset with MONO vs STEREO eye-corner cameras (both with the NIR pupil cam).

Run:  python3 stereo_test.py     (or: python3 main.py --stereo-test)
"""
import numpy as np

import autosim
import anatomy
import rig
import match
import pixel_map
import pupil_sensor
import physics_preset as pp
import complete_geometry as cg
from accuracy_map import _identifier


def _fingerprint(sim, subj, prior, poses, dots, repeats, rng, cam):
    """inacc (from the 8-feat prior) + eyecam MONO (4) + eyecam STEREO (8) + pupil-sensor (3)."""
    inacc, eyemono, eyestereo, pup = [], [], [], []
    for dev in poses:
        pm, ps, pp_ = [], [], []
        for P in dots:
            ms = []
            for _ in range(repeats):
                o = sim.observe(subj, dev, P)          # 14-feat (world4, eye4, stereo4, pupil2)
                if o is None:
                    continue
                f, _label, truth = o
                ms.append(prior.predict(f[:8]) - truth)
                pm.append(f[4:8]); ps.append(f[4:12])
            inacc.append(np.mean(ms, axis=0) if ms else np.zeros(2))
            r = pupil_sensor.pupil_read(subj, dev, P, rng, cam)
            pp_.append(np.concatenate([r[0], r[1]]) if r is not None else np.zeros(3))
        eyemono.append(np.mean(pm, axis=0) if pm else np.zeros(4))
        eyestereo.append(np.mean(ps, axis=0) if ps else np.zeros(8))
        pup.append(np.mean(pp_, axis=0))
    return (np.concatenate(inacc), np.concatenate(eyemono),
            np.concatenate(eyestereo), np.concatenate(pup))


def evaluate(n_train=900, n_test=18, seed=0, verbose=True):
    prior = match.train_prior()
    poses, dots = cg.dense_poses(), cg.dense_dots()
    cam = pupil_sensor.build_camera()
    mono = autosim.Simulator(0, use_pupil=True, use_stereo=False)   # shares device w/ stereo
    stereo = autosim.Simulator(0, use_pupil=True, use_stereo=True)
    rng = np.random.default_rng(100); pop = np.random.default_rng(7)
    if verbose:
        print("== stereo eye-corner camera: does it break under 1 px? ==")
        print("  training mono vs stereo complete-geometry identifiers on %d eyes ..." % n_train)
    INA, EM, ES, PU, FULL = [], [], [], [], []
    for _ in range(n_train):
        s = anatomy.sample_subject(pop)
        s.canthus_bias = stereo.rng.normal(0, rig.CANTHUS_BIAS_SD, 4)
        s.human_bias = stereo.rng.normal(0, rig.HUMAN_BIAS_SD, 2)
        ia, em, es, pu = _fingerprint(stereo, s, prior, poses, dots, 4, rng, cam)
        INA.append(ia); EM.append(em); ES.append(es); PU.append(pu); FULL.append(cg.full_descriptor(s))
    INA, EM, ES, PU, FULL = map(np.array, (INA, EM, ES, PU, FULL))
    id_mono = _identifier([INA, EM, PU], FULL)
    id_stereo = _identifier([INA, ES, PU], FULL)

    ep_, ed = pixel_map.pose_grid(), pixel_map.dot_grid()
    err_mono, err_stereo = [], []
    rec_mono, rec_stereo = [], []
    tpop = np.random.default_rng(54321)
    if verbose:
        print("  measuring %d held-out users x %d positions x %d dots ..." % (n_test, len(ep_), len(ed)))
    for _ in range(n_test):
        s = anatomy.sample_subject(tpop)
        s.canthus_bias = stereo.rng.normal(0, rig.CANTHUS_BIAS_SD, 4)
        s.human_bias = stereo.rng.normal(0, rig.HUMAN_BIAS_SD, 2)
        ia, em, es, pu = _fingerprint(stereo, s, prior, poses, dots, 3, rng, cam)
        th_m = id_mono([ia[None], em[None], pu[None]])[0]
        th_s = id_stereo([ia[None], es[None], pu[None]])[0]
        rec_mono.append(np.abs(th_m - cg.full_descriptor(s)))
        rec_stereo.append(np.abs(th_s - cg.full_descriptor(s)))
        sub_m = cg.reconstruct_full(th_m); sub_s = cg.reconstruct_full(th_s)
        for dev in ep_:
            for P in ed:
                gm = mono.ground_truth(s, dev, P)
                gs = stereo.ground_truth(s, dev, P)
                if gm is not None:
                    b = pp.physics_predict(mono, sub_m, gm[0], use_pupil=True, use_stereo=False)
                    if b is not None:
                        err_mono.append(np.linalg.norm(b - gm[1]))
                if gs is not None:
                    b = pp.physics_predict(stereo, sub_s, gs[0], use_pupil=True, use_stereo=True)
                    if b is not None:
                        err_stereo.append(np.linalg.norm(b - gs[1]))
    rm, rs = np.array(rec_mono).mean(0), np.array(rec_stereo).mean(0)
    if verbose:
        names = cg.FULL_NAMES; pop_sd = FULL.std(0)
        print("\n  geometry recovery (%% pop spread):  %-8s %-8s" % ("MONO", "STEREO"))
        for j in (names.index("cfwd"), names.index("canthal_tilt"), names.index("OCD")):
            print("    %-12s %6.0f%%   %6.0f%%" % (names[j], 100 * rm[j] / pop_sd[j], 100 * rs[j] / pop_sd[j]))
        em_, es_ = np.array(err_mono) * 1080, np.array(err_stereo) * 1080
        print("\n  deployed physics-preset systematic (px @1080p):")
        print("    %-18s median %6.3f  95th %6.3f" % ("MONO (1 eye cam)", np.median(em_), np.percentile(em_, 95)))
        tag = "  <-- UNDER 1 px" if np.median(es_) < 1.0 else ""
        print("    %-18s median %6.3f  95th %6.3f%s" % ("STEREO (2 eye cams)", np.median(es_), np.percentile(es_, 95), tag))
    return err_mono, err_stereo


if __name__ == "__main__":
    evaluate()
