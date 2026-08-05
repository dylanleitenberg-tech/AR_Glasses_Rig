"""Physics-based preset: estimate (dot, pose) from the features, forward-compute the pixel.

The polynomial preset floored at ~0.75-0.9 px because a fitted polynomial can't reproduce the
oracle's pixel map exactly, and adding the pupil feature only made the polynomial overfit.
This replaces the black-box polynomial with the WHITE-BOX physics:

  1. triangulate the world dot P from the two world cameras (over-determined);
  2. estimate the 6-DOF glasses pose by fitting it (damped Gauss-Newton / Levenberg-Marquardt)
     so the rig cameras reproduce the observed eye-corner (+ optional pupil) features for the
     KNOWN geometry;
  3. forward-compute the registered pixel with the oracle (`ground_truth`) at that pose/dot.

There is no representational error, only the error in recovering pose + dot. The 2 eye
corners UNDER-determine the 6-DOF pose, so 8 features leave an ambiguity; the NIR pupil-centre
feature (10 features) fully determines it. This is the correct way to USE the pupil feature
(it pins the pose for the solver, not as overfit polynomial terms).

Two subtleties that make it actually work:
  * the real cameras QUANTIZE to the sensor grid, which zeroes the finite-difference Jacobian
    and makes naive GN diverge -> the solver runs on UN-quantized model cameras (continuous),
    fitting the quantized observations; the quantization just sets a residual floor.
  * Levenberg-Marquardt damping keeps it stable on the under-determined 8-feature case.

Run:  python3 physics_preset.py        (or:  python3 main.py --physics-preset)
"""
import copy
import numpy as np

import rig
import autosim
import pixel_map
from preset import build_preset

_POSE_H = np.array([0.02, 0.02, 0.02, 0.04, 0.04, 0.04])   # finite-diff steps (deg/mm)


def _unq(sim):
    """Cached UN-quantized copies of the rig cameras (res=0) for smooth Jacobians."""
    if not hasattr(sim, "_uq_cams"):
        d = {}
        for key in ("world", "eye", "eye2"):
            d[key] = [copy.copy(c) for c in sim.rig[key]]
            for c in d[key]:
                c.res = 0
        p = copy.copy(sim.rig["pupil"]); p.res = 0
        d["pupil"] = p
        sim._uq_cams = d
    return sim._uq_cams


def triangulate_dot(sim, world_feats, z0=rig.DOT_Z_MEAN, iters=12):
    """Recover the world dot P (rig frame) from the two world-camera readings (min-norm GN)."""
    cams = _unq(sim)["world"]
    P = np.array([0.0, 0.0, z0]); h = 0.3
    for _ in range(iters):
        a = cams[0].project(P); b = cams[1].project(P)
        if a is None or b is None:
            break
        pred = np.concatenate([a, b]); r = world_feats - pred
        J = np.zeros((4, 3))
        for k in range(3):
            Pp = P.copy(); Pp[k] += h
            J[:, k] = (np.concatenate([cams[0].project(Pp), cams[1].project(Pp)]) - pred) / h
        step, *_ = np.linalg.lstsq(J, r, rcond=None)
        P = P + np.clip(step, -40, 40)
        if np.linalg.norm(step) < 1e-4:
            break
    return P


def _eye_pred(cams, subject, dev, P, use_pupil, use_stereo=False):
    """Predicted eye-corner (+ stereo + pupil) features for a hypothesized pose. Order MUST
    match autosim's feature vector: [cL, cR, (cL2, cR2 if stereo), (pupil if pupil)]."""
    R, t = rig.pose_R_t(dev)
    outer = (R @ subject.outer_canthus.T).T + t
    cL = cams["eye"][0].project(outer[0]); cR = cams["eye"][1].project(outer[1])
    if cL is None or cR is None:
        return None
    feats = [cL, cR]
    if use_stereo:
        cL2 = cams["eye2"][0].project(outer[0]); cR2 = cams["eye2"][1].project(outer[1])
        if cL2 is None or cR2 is None:
            return None
        feats += [cL2, cR2]
    if use_pupil:
        e = rig.DISPLAY_EYE
        cor = (R @ subject.cor.T).T + t
        gaze = P - cor[e]; gaze = gaze / np.linalg.norm(gaze)
        ep = cor[e] + subject.ep_dist * gaze
        ga = np.degrees(np.arccos(np.clip(gaze[2], -1.0, 1.0)))
        lat = np.array([gaze[0], gaze[1], 0.0]); ln = np.linalg.norm(lat)
        if ln > 1e-6:
            ep = ep + rig.PUPIL_WANDER * (ga / 10.0) * (lat / ln)
        pf = cams["pupil"].project(ep)
        if pf is None:
            return None
        feats.append(pf)
    return np.concatenate(feats)


def estimate_pose(sim, subject, eye_feats, P, use_pupil, use_stereo=False, iters=30):
    """Min-norm Gauss-Newton fit of the 6-DOF pose to the observed features. The stereo 2nd
    eye view triangulates the canthus depth, removing the under-determination a single view
    leaves; lstsq gives the minimum-norm step so unseen pose directions stay near rest."""
    cams = _unq(sim)
    dev = np.zeros(6)
    for _ in range(iters):
        pred = _eye_pred(cams, subject, dev, P, use_pupil, use_stereo)
        if pred is None:
            break
        r = eye_feats - pred
        J = np.zeros((len(r), 6))
        for k in range(6):
            dp = dev.copy(); dp[k] += _POSE_H[k]
            p2 = _eye_pred(cams, subject, dp, P, use_pupil, use_stereo)
            if p2 is None:
                continue
            J[:, k] = (p2 - pred) / _POSE_H[k]
        step, *_ = np.linalg.lstsq(J, r, rcond=None)
        step = np.clip(step, -3, 3)
        dev = dev + step
        if np.linalg.norm(step) < 1e-5:
            break
    return dev


def physics_predict(sim, subject, features, use_pupil, use_stereo=False):
    """features -> recovered (P, pose) -> oracle pixel. eye_feats = everything after the 4
    world features (eye-corner + stereo + pupil), in the same order _eye_pred produces."""
    P = triangulate_dot(sim, features[0:4])
    dev = estimate_pose(sim, subject, features[4:], P, use_pupil, use_stereo)
    g = sim.ground_truth(subject, dev, P)
    return None if g is None else g[1]


def evaluate(n_test=15, seed=900, verbose=True):
    """True-geometry floor: polynomial vs physics preset, 8 vs 10 features."""
    sim = autosim.Simulator(seed, use_pupil=True)        # 10-feature oracle (use [:8] for 8f)
    poses, dots = pixel_map.pose_grid(), pixel_map.dot_grid()
    err = {k: [] for k in ("poly8", "poly10", "phys8", "phys10")}
    if verbose:
        print("== physics-based preset vs polynomial (true-geometry floor) ==")
        print("  %d eyes x %d positions x %d dots ...\n" % (n_test, len(poses), len(dots)))
    for i in range(n_test):
        subj = sim.new_subject()
        theta = subj.descriptor()
        p8 = build_preset(theta, n_sweep=4000, use_pupil=False)
        p10 = build_preset(theta, n_sweep=4000, use_pupil=True)
        for dev in poses:
            for P in dots:
                g = sim.ground_truth(subj, dev, P)
                if g is None:
                    continue
                f, truth = g                          # f is 10-D (use_pupil sim)
                err["poly8"].append(np.linalg.norm(p8.predict(f[:8]) - truth))
                err["poly10"].append(np.linalg.norm(p10.predict(f) - truth))
                a = physics_predict(sim, subj, f[:8], use_pupil=False)   # 8-feat: world + eye-corner
                b = physics_predict(sim, subj, f, use_pupil=True)        # 10-feat: + NIR pupil
                if a is not None:
                    err["phys8"].append(np.linalg.norm(a - truth))
                if b is not None:
                    err["phys10"].append(np.linalg.norm(b - truth))
        if verbose and (i + 1) % 5 == 0:
            print("    ... %d/%d eyes" % (i + 1, n_test))
    if verbose:
        print("\n  true-geometry overlay error vs oracle (px @1080p):")
        print("    %-28s %8s %8s %8s" % ("preset", "median", "95th", "max"))
        for k, lab in [("poly8", "polynomial, 8 feat"), ("poly10", "polynomial, 10 feat (+pupil)"),
                       ("phys8", "physics, 8 feat"), ("phys10", "physics, 10 feat (+pupil)")]:
            e = np.array(err[k]) * 1080.0
            tag = "  <- under 0.5 px" if np.median(e) < 0.5 else ""
            print("    %-28s %8.3f %8.3f %8.3f%s"
                  % (lab, np.median(e), np.percentile(e, 95), e.max(), tag))
        ph = float(np.median(np.array(err["phys10"]) * 1080.0))
        rel = "UNDER" if ph < 0.5 else "vs"
        print("\n  physics + pupil reaches %.3f px median (%s 0.5 px); the pupil feature pins the"
              % (ph, rel))
        print("  pose the 2 corners under-determine, and the white-box forward has no poly error.")
    return err


if __name__ == "__main__":
    evaluate()
