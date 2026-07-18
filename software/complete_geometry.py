"""Complete-geometry identification: recover the FULL eye geometry (not the lossy 8-param
descriptor) so the physics preset can reach its ceiling deployed.

The 8-param descriptor drops vertical eye position (eye_y) and canthus forward depth (cfwd),
which alone floored the physics preset at ~1.44 px even with perfect kappa. Here we identify
all 10 (8 + eye_y + cfwd) WITH the pupil sensor, reconstruct the complete eye, and run the
white-box physics preset. This is the systematic-error half of the under-0.75-px campaign.

Run:  python3 complete_geometry.py     (or: python3 main.py --complete-geometry)
"""
import numpy as np

import autosim
import anatomy
import match
import pixel_map
import physics_preset as pp
from accuracy_map import _identifier

FULL_NAMES = anatomy.DESCRIPTOR_NAMES + ["eye_y", "cfwd", "cor_z"]


def full_descriptor(s):
    """11-param complete geometry: 8-param descriptor + vertical eye pos + canthus depth +
    CoR depth (the per-face center-of-rotation depth, which the registration parallax sees)."""
    return np.concatenate([s.descriptor(), [s.cor[0][1], s.inner_canthus[0][2], s.cor[0][2]]])


def reconstruct_full(theta):
    """Inverse of full_descriptor -> a complete anatomy.Subject (matches sample_subject)."""
    ipd, globe_r, ep, ocd, icd, tilt, kx, ky, eye_y, cfwd, cor_z = [float(v) for v in theta]
    cor = np.array([[-ipd / 2, eye_y, cor_z], [ipd / 2, eye_y, cor_z]])
    inner = np.array([[-icd / 2, eye_y, cfwd], [icd / 2, eye_y, cfwd]])
    outer = np.array([[-ocd / 2, eye_y + tilt, cfwd], [ocd / 2, eye_y + tilt, cfwd]])
    s = anatomy.Subject(ipd, globe_r, ep, np.array([kx, ky]), cor, inner, outer)
    s.canthus_bias = np.zeros(4)
    return s


def dense_poses():
    """More glasses positions -> the eye-corner cams get more views of the outer canthi,
    pinning the cfwd/OCD/tilt geometry the physics pose-fit needs."""
    poses = []
    for dty in (-1.0, 1.0, 3.0, 5.0, 7.0):           # vertical slide down the nose (mm)
        drx = dty * 0.25                             # correlated pantoscopic tilt (rig.PITCH_PER_MM)
        for dtx in (-2.0, 0.0, 2.0):                 # lateral shift (mm)
            poses.append(np.array([drx, 0, 0, dtx, dty, 0.0]))
    return poses                                     # 15 positions


def dense_dots():
    xs = (-120.0, -60.0, 0.0, 60.0, 120.0); ys = (-80.0, 0.0, 80.0)
    import rig
    return [np.array([x, y, rig.DOT_Z_MEAN]) for y in ys for x in xs]   # 15 gaze dirs


def evaluate(n_train=900, n_test=18, seed=0, dense=True, verbose=True):
    prior = match.train_prior()
    poses = dense_poses() if dense else match.probe_poses()
    dots = dense_dots() if dense else match.probe_dots()
    cam = match.pupil_sensor.build_camera()
    sim = autosim.Simulator(0); rng = np.random.default_rng(100)
    INA, EYE, PUP, FULL = [], [], [], []
    if verbose:
        print("== complete-geometry (10-param) identification + physics preset ==")
        print("  training identifier on %d eyes (with pupil sensor) ..." % n_train)
    for _ in range(n_train):
        s = sim.new_subject()
        ia, ey, _ir, _ic, pu = match.fingerprint(sim, s, prior, poses, dots, 4, rng, cam)
        INA.append(ia); EYE.append(ey); PUP.append(pu); FULL.append(full_descriptor(s))
    INA, EYE, PUP, FULL = map(np.array, (INA, EYE, PUP, FULL))
    idp = _identifier([INA, EYE, PUP], FULL)

    f8 = autosim.Simulator(54321); ev = autosim.Simulator(54321, use_pupil=True)
    f8.rng = np.random.default_rng(7)
    ep_, ed = pixel_map.pose_grid(), pixel_map.dot_grid()
    true_, dep = [], []
    rec_err, pop_sd = [], FULL.std(0)
    if verbose:
        print("  measuring %d held-out users x %d positions x %d dots ..." % (n_test, len(ep_), len(ed)))
    for _ in range(n_test):
        subj = f8.new_subject()
        ia, ey, _ir, _ic, puf = match.fingerprint(f8, subj, prior, poses, dots, 3, rng, cam)
        th = idp([ia[None], ey[None], puf[None]])[0]
        rec_err.append(np.abs(th - full_descriptor(subj)))
        fsub = reconstruct_full(th)
        for dev in ep_:
            for P in ed:
                g = ev.ground_truth(subj, dev, P)
                if g is None:
                    continue
                for store, sub in ((true_, subj), (dep, fsub)):
                    b = pp.physics_predict(ev, sub, g[0], use_pupil=True)
                    if b is not None:
                        store.append(np.linalg.norm(b - g[1]))
    rec_err = np.array(rec_err)
    if verbose:
        print("\n  complete-geometry recovery (MAE, %% of pop spread):")
        for j, nm in enumerate(FULL_NAMES):
            print("    %-12s %6.2f  (%3.0f%%)" % (nm, rec_err[:, j].mean(),
                                                  100 * rec_err[:, j].mean() / max(pop_sd[j], 1e-9)))
        print("\n  physics-preset systematic error (px @1080p):")
        for nm, a in (("TRUE geometry (ceiling)", true_),
                      ("DEPLOYED (10-param ID + pupil)", dep)):
            e = np.array(a) * 1080.0
            tag = "  <- under 0.75 px" if np.median(e) < 0.75 else ""
            print("    %-32s median %6.3f  95th %6.3f%s" % (nm, np.median(e), np.percentile(e, 95), tag))
    return true_, dep, rec_err


if __name__ == "__main__":
    evaluate()
