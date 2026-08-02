"""Forward simulator / oracle: anatomy + optics + glasses pose -> (features, truth).

It mirrors the real rig (camera placement comes from `rig.py`, which mirrors the printed
bracket). For a subject, a glasses pose, and a world point it returns:
  * features (8,) = [worldL dot xy, worldR dot xy, eyeL corner xy, eyeR corner xy] — the
    exact readings the four bracket cameras produce, and
  * the TRUE display pixel that registers the overlay for the display (right) eye.

Registration physics
---------------------
The eye fixates the point P (it images on the fovea). A see-through (≈collimated) AR
optic emits a fixed angular direction per pixel, so to co-register the displayed ray must
equal the eye's line of sight: the chief ray ENTRANCE-PUPIL -> P. The entrance pupil sits
ahead of the center of rotation and moves as the eye rotates, so for a finite-distance P
the required direction depends on eye position -> parallax. When the glasses slip, the eye
and canthi move in the rig frame, the cameras see it, and the required pixel changes.
Angle kappa adds a per-subject bias the cameras cannot see -> the irreducible reason each
user still calibrates, and what the cross-subject prior can only partly remove.

Frame: rig/optic frame, origin at the optic center, +x right, +y up, +z forward. mm.
"""
import numpy as np

import rig
import anatomy
import optics


class Simulator:
    def __init__(self, seed=0, use_pupil=False, use_stereo=False, landmark=None):
        # landmark=None -> rig.TRACKED_LANDMARK (the settled project default), so the choice
        # lives in ONE place. Pass it explicitly only to A/B the alternative.
        landmark = rig.TRACKED_LANDMARK if landmark is None else landmark
        self.rng = np.random.default_rng(seed)
        self.use_pupil = use_pupil          # append the NIR pupil-centre feature
        self.use_stereo = use_stereo        # append the 2nd eye-corner (stereo) features
        # which canthus the eye-corner cams aim at AND track: "outer" (the built design) or
        # "inner" (the medial-canthus variant — see landmark_test.py). Feature ORDER and count
        # are unchanged; only which facial point those 4 numbers report.
        self.landmark = landmark
        # Per-capture soft-tissue motion of the tracked landmark. None => rig.SOFT_TISSUE_SD
        # for BOTH landmarks: there is no measured basis in this project for the medial
        # canthus being steadier than the lateral one, so the default A/B compares pure
        # geometry. The sensitivity sweep sets this explicitly instead of assuming a value.
        self.soft_tissue_override = None
        self.rig = rig.build(landmark)
        self._apply_unit_tolerance()       # fixed per-device errors (same every session)

    def _apply_unit_tolerance(self):
        """One printed unit: fixed camera extrinsic + intrinsic errors, display distortion,
        and a display-to-camera misregistration. Constant per device, so it is LEARNABLE
        — it shapes the mapping the calibrator fits, it does not add per-attempt noise."""
        d = np.random.default_rng(rig.DEVICE_SEED)

        def perturb(cam, g):
            cam.pos = cam.pos + g.normal(0, rig.UNIT_TRANS_SD, 3)
            cam.R = optics.rot_xyz(*g.normal(0, rig.UNIT_ROT_SD, 3)) @ cam.R
            cam.fscale = 1.0 + g.normal(0, rig.UNIT_FOCAL_SD)
            cam.cx = g.normal(0, rig.UNIT_PRINCIPAL_SD)
            cam.cy = g.normal(0, rig.UNIT_PRINCIPAL_SD)
            cam.k2 = g.normal(rig.CAM_K2, rig.CAM_K2 * 0.3)
            cam.p1 = g.normal(0, rig.CAM_TANGENTIAL_SD)
            cam.p2 = g.normal(0, rig.CAM_TANGENTIAL_SD)

        for cam in self.rig["world"] + self.rig["eye"] + [self.rig["pupil"]]:
            perturb(cam, d)
        disp = self.rig["display"]
        disp.k2 = rig.DISPLAY_K2
        disp.warp = d.normal(0, rig.DISPLAY_WARP_SD, 4)
        self.R_dc = optics.rot_xyz(*d.normal(0, rig.DISPLAY_CAM_ROT_SD, 3))
        self.t_dc = d.normal(0, rig.DISPLAY_CAM_TRANS_SD, 3)
        # 2nd eye-corner pair gets its OWN rng so it doesn't shift the device above (existing
        # experiments stay bit-identical whether or not stereo is used).
        d2 = np.random.default_rng(rig.DEVICE_SEED + 1)
        for cam in self.rig["eye2"]:
            perturb(cam, d2)

    # ---- tracked landmark -------------------------------------------------
    def _canthus(self, subject):
        """(2,3) face-frame position of the landmark the eye-corner cams track."""
        return subject.outer_canthus if self.landmark == "outer" else subject.inner_canthus

    def soft_tissue_sd(self):
        """Per-capture soft-tissue motion of the tracked landmark (mm)."""
        return rig.SOFT_TISSUE_SD if self.soft_tissue_override is None \
            else self.soft_tissue_override

    # ---- subjects & poses ------------------------------------------------
    def new_subject(self):
        s = anatomy.sample_subject(self.rng)
        s.canthus_bias = self.rng.normal(0, rig.CANTHUS_BIAS_SD, 4)  # template centering
        s.human_bias = self.rng.normal(0, rig.HUMAN_BIAS_SD, 2)      # perceptual offset
        s.cdrift = np.zeros(4)
        return s

    def seat(self):
        """How the user first puts the glasses on (small random offset from rest)."""
        return np.array([self.rng.normal(0, rig.SEAT_ROT_SD) for _ in range(3)] +
                        [self.rng.normal(0, rig.SEAT_TRANS_SD) for _ in range(3)])

    def slip(self, dev):
        """Realistic per-iteration motion: a slow slide down the nose (vertical drift +
        correlated pantoscopic tilt) + jitter, with occasional re-seating."""
        if self.rng.random() < rig.RESEAT_PROB:        # took them off / nudged back on
            return self.seat()
        dev = np.asarray(dev, float).copy()
        slide = abs(self.rng.normal(rig.SLIDE_RATE, rig.SLIDE_RATE * 0.5))
        dev[4] += slide                                # eyes drift up in the rig frame
        dev[0] += slide * rig.PITCH_PER_MM             # pantoscopic tilt grows with it
        dev[:3] = dev[:3] * (1 - rig.SLIP_REVERT) + self.rng.normal(0, rig.SLIP_ROT_SD, 3)
        dev[3:] = dev[3:] * (1 - rig.SLIP_REVERT) + self.rng.normal(0, rig.SLIP_TRANS_SD, 3)
        dev[3:] = np.clip(dev[3:], -rig.TRANS_CAP, rig.TRANS_CAP)
        return dev

    def world_point(self):
        """The drawn dot roams a wide page so the overlay maps display pixels across the
        WHOLE field (different dot positions -> different target pixels)."""
        return np.array([self.rng.uniform(-rig.DOT_X, rig.DOT_X),
                         self.rng.uniform(-rig.DOT_Y, rig.DOT_Y),
                         self.rng.normal(rig.DOT_Z_MEAN, rig.DOT_Z_SD)])

    def _noise(self, sd, n, tp=0.0, ts=1.0):
        """Gaussian noise, occasionally from a heavier tail (prob tp, scale ts)."""
        s = sd * (ts if self.rng.random() < tp else 1.0)
        return self.rng.normal(0, s, n)

    # ---- the oracle ------------------------------------------------------
    def observe(self, subject, dev, P):
        """Return (features (8,), human label (2,), geometric truth (2,)) or None.

        TRAIN on the human label (all reality provides); EVALUATE against the geometric
        truth (what actually registers). Per-session/per-unit effects are constant
        (learnable); per-attempt effects set the accuracy floor.
        """
        if self.rng.random() < rig.BLINK_PROB:        # capture spoiled by a blink
            return None
        dev = np.asarray(dev, float).copy()           # head motion during the capture
        dev[:3] += self.rng.normal(0, rig.HEAD_MOTION_ROT_SD, 3)
        dev[3:] += self.rng.normal(0, rig.HEAD_MOTION_TRANS_SD, 3)
        R, t = rig.pose_R_t(dev)
        cor = (R @ subject.cor.T).T + t
        st = self.rng.normal(0, self.soft_tissue_sd(), (2, 3))   # soft-tissue canthus motion
        outer = (R @ (self._canthus(subject) + st).T).T + t

        wL = self.rig["world"][0].project(P)
        wR = self.rig["world"][1].project(P)
        cL = self.rig["eye"][0].project(outer[0])
        cR = self.rig["eye"][1].project(outer[1])
        if wL is None or wR is None or cL is None or cR is None:
            return None
        # eye-corner tracker: per-session bias + within-session drift + heavy-tail noise
        subject.cdrift = subject.cdrift + self.rng.normal(0, rig.CANTHUS_DRIFT_SD, 4)
        ct = (rig.CANTHUS_PIX_SD, 2, rig.CANTHUS_TAIL_PROB, rig.CANTHUS_TAIL_SCALE)
        cL = cL + subject.canthus_bias[0:2] + subject.cdrift[0:2] + self._noise(*ct)
        cR = cR + subject.canthus_bias[2:4] + subject.cdrift[2:4] + self._noise(*ct)
        if self.rng.random() < rig.CANTHUS_GROSS_PROB:        # gross tracker failure
            (cL if self.rng.random() < 0.5 else cR)[:] += self.rng.uniform(-0.08, 0.08, 2)
        wL = wL + self._noise(rig.DOT_PIX_SD, 2)
        wR = wR + self._noise(rig.DOT_PIX_SD, 2)
        features = np.concatenate([wL, wR, cL, cR])
        if self.use_stereo:                            # 2nd eye-corner view (stereo on each canthus)
            cL2 = self.rig["eye2"][0].project(outer[0])
            cR2 = self.rig["eye2"][1].project(outer[1])
            if cL2 is None or cR2 is None:
                return None
            features = np.concatenate([features, cL2 + self._noise(*ct), cR2 + self._noise(*ct)])

        # ---- geometric registration pixel (display / right eye) ----
        e = self.rig["display_eye"]
        gaze = P - cor[e]; gaze = gaze / np.linalg.norm(gaze)
        ep_dist = subject.ep_dist + self.rng.normal(0, rig.EP_PUPIL_SD)   # pupil size
        ep = cor[e] + ep_dist * gaze
        ga = np.degrees(np.arccos(np.clip(gaze[2], -1.0, 1.0)))           # gaze angle off-axis
        lat = np.array([gaze[0], gaze[1], 0.0]); ln = np.linalg.norm(lat)
        if ln > 1e-6:                                                     # pupil wander
            ep = ep + rig.PUPIL_WANDER * (ga / 10.0) * (lat / ln)
        if self.use_pupil:                          # NIR pupil-centre feature (noisy)
            pf = self.rig["pupil"].project(ep)
            if pf is None:
                return None
            features = np.concatenate([features, pf + self._noise(rig.PUPIL_PIX_SD, 2)])
        epd = self.R_dc @ ep + self.t_dc            # display-to-camera misregistration
        Pd = self.R_dc @ P + self.t_dc
        opticd = self.R_dc @ self.rig["optic"] + self.t_dc
        dd = Pd - epd; dd = dd / np.linalg.norm(dd)
        jx = np.radians(self.rng.normal(0, rig.FIXATION_JITTER_DEG))
        jy = np.radians(self.rng.normal(0, rig.FIXATION_JITTER_DEG))
        az = np.arctan2(dd[0], dd[2]) + (1.0 if e == 1 else -1.0)*np.radians(subject.kappa[0]) + jx
        el = np.arctan2(dd[1], dd[2]) + np.radians(subject.kappa[1]) + jy
        dk = np.array([np.tan(az), np.tan(el), 1.0]); dk = dk / np.linalg.norm(dk)
        true_px = self.rig["display"].pixel_for_target(epd, dk, opticd)
        m = rig.ONSCREEN_MARGIN
        if np.any(true_px < m) or np.any(true_px > 1.0 - m):  # dot would be off the display
            return None
        # ---- human-corrected label (what actually gets stored) ----
        label = true_px + subject.human_bias + self._noise(
            rig.HUMAN_NOISE_SD, 2, rig.HUMAN_TAIL_PROB, rig.HUMAN_TAIL_SCALE)
        if self.rng.random() < rig.HUMAN_GROSS_PROB:          # fat-finger correction
            label = label + self.rng.uniform(-0.1, 0.1, 2)
        return features, label, true_px

    # ---- noise-free oracle (the ground-truth pixel map) ------------------
    def ground_truth(self, subject, dev, P):
        """Deterministic, noise-free version of `observe`: for one eye `subject`, one
        glasses pose `dev`, and one world dot `P`, return the CLEAN camera features and
        the EXACT display pixel that registers the overlay — i.e. *where the pixel should
        be displayed*. Returns (features (8,), true_px (2,)) or None if the dot lands off
        the display. Same physics as `observe`, with every per-attempt random term removed
        (no blink/head-motion/jitter/pupil-size/tracker noise), so it is repeatable."""
        R, t = rig.pose_R_t(dev)
        cor = (R @ subject.cor.T).T + t
        outer = (R @ self._canthus(subject).T).T + t
        wL = self.rig["world"][0].project(P)
        wR = self.rig["world"][1].project(P)
        cL = self.rig["eye"][0].project(outer[0])
        cR = self.rig["eye"][1].project(outer[1])
        if wL is None or wR is None or cL is None or cR is None:
            return None
        features = np.concatenate([wL, wR, cL, cR])
        if self.use_stereo:                            # clean 2nd eye-corner view
            cL2 = self.rig["eye2"][0].project(outer[0])
            cR2 = self.rig["eye2"][1].project(outer[1])
            if cL2 is None or cR2 is None:
                return None
            features = np.concatenate([features, cL2, cR2])

        e = self.rig["display_eye"]
        gaze = P - cor[e]; gaze = gaze / np.linalg.norm(gaze)
        ep = cor[e] + subject.ep_dist * gaze
        ga = np.degrees(np.arccos(np.clip(gaze[2], -1.0, 1.0)))
        lat = np.array([gaze[0], gaze[1], 0.0]); ln = np.linalg.norm(lat)
        if ln > 1e-6:                                          # deterministic pupil shift
            ep = ep + rig.PUPIL_WANDER * (ga / 10.0) * (lat / ln)
        if self.use_pupil:                                     # NIR pupil-centre feature (clean)
            pf = self.rig["pupil"].project(ep)
            if pf is None:
                return None
            features = np.concatenate([features, pf])
        epd = self.R_dc @ ep + self.t_dc
        Pd = self.R_dc @ P + self.t_dc
        opticd = self.R_dc @ self.rig["optic"] + self.t_dc
        dd = Pd - epd; dd = dd / np.linalg.norm(dd)
        az = np.arctan2(dd[0], dd[2]) + (1.0 if e == 1 else -1.0)*np.radians(subject.kappa[0])
        el = np.arctan2(dd[1], dd[2]) + np.radians(subject.kappa[1])
        dk = np.array([np.tan(az), np.tan(el), 1.0]); dk = dk / np.linalg.norm(dk)
        true_px = self.rig["display"].pixel_for_target(epd, dk, opticd)
        m = rig.ONSCREEN_MARGIN
        if np.any(true_px < m) or np.any(true_px > 1.0 - m):
            return None
        return features, true_px

    # ---- BINOCULAR groundwork (2026-07-03): the LEFT eye's clean pixel ----------------
    # The XREAL One Pro is binocular (a display per eye); the validated pipeline registers
    # the RIGHT/display eye. These ADDITIVE methods compute the left eye's registration
    # pixel with the same physics (e=0, optic_l, horizontal kappa sign mirrored — kappa is
    # nasal-ward in both eyes). Features are shared (the cameras see one scene), so per-eye
    # calibrators consume the SAME feature vector with per-eye labels.
    # NOTE: ep_dist/kappa are per-SUBJECT here; per-eye asymmetry is a future anatomy
    # refinement.
    def ground_truth_left(self, subject, dev, P):
        """Noise-free LEFT-eye registration pixel, or None if off that display."""
        R, t = rig.pose_R_t(dev)
        cor = (R @ subject.cor.T).T + t
        e = 0
        gaze = P - cor[e]; gaze = gaze / np.linalg.norm(gaze)
        ep = cor[e] + subject.ep_dist * gaze
        ga = np.degrees(np.arccos(np.clip(gaze[2], -1.0, 1.0)))
        lat = np.array([gaze[0], gaze[1], 0.0]); ln = np.linalg.norm(lat)
        if ln > 1e-6:                                          # deterministic pupil shift
            ep = ep + rig.PUPIL_WANDER * (ga / 10.0) * (lat / ln)
        epd = self.R_dc @ ep + self.t_dc
        Pd = self.R_dc @ P + self.t_dc
        opticd = self.R_dc @ self.rig["optic_l"] + self.t_dc
        dd = Pd - epd; dd = dd / np.linalg.norm(dd)
        az = np.arctan2(dd[0], dd[2]) - np.radians(subject.kappa[0])   # mirrored kappa_x
        el = np.arctan2(dd[1], dd[2]) + np.radians(subject.kappa[1])
        dk = np.array([np.tan(az), np.tan(el), 1.0]); dk = dk / np.linalg.norm(dk)
        true_px = self.rig["display"].pixel_for_target(epd, dk, opticd)
        m = rig.ONSCREEN_MARGIN
        if np.any(true_px < m) or np.any(true_px > 1.0 - m):
            return None
        return true_px

    def ground_truth_both(self, subject, dev, P):
        """(clean features, right px, left px|None) — the binocular oracle."""
        g = self.ground_truth(subject, dev, P)
        if g is None:
            return None
        return g[0], g[1], self.ground_truth_left(subject, dev, P)
