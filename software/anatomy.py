"""Anatomically-grounded parametric eye + face model (numpy only).

This is the "PhD understanding" encoded as geometry. Everything that matters for
where an AR pixel must sit to register on a real-world point depends on ocular and
facial anatomy, so we model it explicitly and sample realistic population variance.

Why each quantity matters for AR registration
----------------------------------------------
* Interpupillary distance (IPD): sets where each eye sits laterally; the optic must
  serve a pupil that can be 52-78 mm apart. Mean ~63 mm, SD ~3.8 mm (Dodgson 2004).
* Eyeball radius / axial length: globe ~24 mm long, radius ~12 mm. The CENTER OF
 ROTATION sits ~13.5 mm behind the corneal apex, the eye rotates about this point
  to fixate, so gaze direction is the line CoR->target.
* Entrance pupil: the eye's real aperture sits ~3 mm behind the cornea, i.e.
  ~10.5 mm IN FRONT of the center of rotation. The chief ray (entrance pupil -> point)
  is the visual line we must match. Its position shifts as the eye rotates -> parallax.
* Angle kappa: the visual axis (fovea->fixation) differs from the pupillary/optical
  axis by ~5 deg horizontally (temporal) and ~1.5 deg vertically, and this varies per
  person. It is essentially UNOBSERVABLE from the outside, so it is the irreducible
  per-user term that forces individual calibration.
* Canthi (eye corners): the inner/outer canthus positions come from palpebral fissure
  length (~28-30 mm), inter-canthal distance (~32 mm), outer-canthal distance (~89 mm)
  and canthal tilt (outer corner ~5 deg higher in youth). The inward cameras measure
  these, so they encode (a) how the glasses sit relative to the eye and (b) the eye's
  individual shape. They are the bridge between "glasses pose" and "which pixel".

Face frame: origin midway between the two eyeball centers of rotation,
  +x = wearer's right, +y = up, +z = forward (toward the world). Millimetres.
"""
from dataclasses import dataclass, field
import numpy as np

FACE_SCALE_SD = 0.045          # shared size factor: bigger faces -> bigger IPD/OCD/ICD

# ---- population statistics (mm / degrees) ----------------------------------
IPD_MEAN, IPD_SD            = 63.0, 3.8     # interpupillary distance
GLOBE_R_MEAN, GLOBE_R_SD    = 12.0, 0.45    # eyeball radius (axial length varies)
EP_AHEAD_COR_MEAN, EP_SD    = 10.5, 0.7     # entrance pupil ahead of center of rotation
ICD_MEAN, ICD_SD            = 32.0, 2.5     # inner inter-canthal distance
OCD_MEAN, OCD_SD            = 89.0, 4.0     # outer-canthal distance
CANTHAL_TILT_MEAN, TILT_SD  = 5.0, 2.2      # outer canthus elevation (deg)
KAPPA_X_MEAN, KAPPA_X_SD    = 5.0, 1.5      # angle kappa horizontal, temporal+ (deg)
KAPPA_Y_MEAN, KAPPA_Y_SD    = 1.5, 1.0      # angle kappa vertical (deg)
CANTHUS_FWD_MEAN, CFWD_SD   = 11.0, 1.0     # canthi sit ~near the front of the globe (mm)
EYE_VERT_SD                 = 1.5           # vertical placement variation of the eyes
# Depth of the center of rotation behind the canthi/face surface DIFFERS PER FACE: a longer
# eyeball (bigger globe_r / axial length) sits its CoR deeper, and eye relief / socket depth
# vary per person. This is what gives globe_r a real registration effect (longer eye -> CoR
# deeper -> different parallax -> different pixel) and makes it identifiable.
COR_AXIAL_K                 = 1.7           # mm extra CoR depth per mm of globe_r above the mean
COR_DEPTH_SD                = 1.4           # mm independent per-face eye-relief / socket-depth


@dataclass
class Subject:
    """One person's fixed ocular + facial geometry, in the face frame."""
    ipd: float
    globe_r: float
    ep_dist: float                # entrance pupil distance ahead of CoR
    kappa: np.ndarray             # (2,) [horizontal, vertical] deg, magnitudes
    cor: np.ndarray               # (2,3) eyeball centers of rotation [L, R]
    inner_canthus: np.ndarray     # (2,3) [L, R]
    outer_canthus: np.ndarray     # (2,3) [L, R]
    # session/sensor nuisances set by the simulator (not anatomy):
    canthus_bias: np.ndarray = field(default_factory=lambda: np.zeros(4))   # corner offset
    human_bias: np.ndarray = field(default_factory=lambda: np.zeros(2))     # perceptual offset
    cdrift: np.ndarray = field(default_factory=lambda: np.zeros(4))         # within-session drift

    def descriptor(self) -> np.ndarray:
        """Full geometry signature we try to identify (units: mm and degrees)."""
        ocd = float(np.linalg.norm(self.outer_canthus[1] - self.outer_canthus[0]))
        icd = float(np.linalg.norm(self.inner_canthus[1] - self.inner_canthus[0]))
        tilt = float(self.outer_canthus[1, 1] - self.inner_canthus[1, 1])
        return np.array([self.ipd, self.globe_r, self.ep_dist, ocd, icd, tilt,
                         self.kappa[0], self.kappa[1]])


DESCRIPTOR_NAMES = ["IPD", "globe_r", "ep_dist", "OCD", "ICD",
                    "canthal_tilt", "kappa_x", "kappa_y"]
DESCRIPTOR_UNITS = ["mm", "mm", "mm", "mm", "mm", "mm", "deg", "deg"]


def sample_subject(rng: np.random.Generator) -> Subject:
    """Draw one anatomically-plausible subject from the population."""
    # a shared face-size factor correlates IPD / inter-canthal / outer-canthal distances,
    # as in real anthropometry; each still gets an independent residual.
    scale = float(rng.normal(1.0, FACE_SCALE_SD))
    ipd = float(IPD_MEAN * scale + rng.normal(0, IPD_SD * 0.45))
    icd = float(ICD_MEAN * scale + rng.normal(0, ICD_SD * 0.5))
    ocd = float(OCD_MEAN * scale + rng.normal(0, OCD_SD * 0.5))
    globe_r = float(rng.normal(GLOBE_R_MEAN, GLOBE_R_SD))
    ep = float(rng.normal(EP_AHEAD_COR_MEAN, EP_SD))
    kappa = np.array([rng.normal(KAPPA_X_MEAN, KAPPA_X_SD),
                      rng.normal(KAPPA_Y_MEAN, KAPPA_Y_SD)])
    tilt = np.radians(rng.normal(CANTHAL_TILT_MEAN, TILT_SD))
    eye_y = float(rng.normal(0.0, EYE_VERT_SD))
    cfwd = float(rng.normal(CANTHUS_FWD_MEAN, CFWD_SD))

    # eyeball centers of rotation: DEPTH VARIES PER FACE: deeper for longer eyeballs
    # (globe_r above the mean) plus an independent eye-relief/socket-depth term. -z = deeper
    # (behind the canthi at +cfwd), so the CoR-to-face-surface distance differs per subject.
    cor_z = -(globe_r - GLOBE_R_MEAN) * COR_AXIAL_K + float(rng.normal(0.0, COR_DEPTH_SD))
    cor = np.array([[-ipd / 2, eye_y, cor_z],
                    [+ipd / 2, eye_y, cor_z]])
    # canthi: inner near the nose, outer out by the temple and raised by canthal tilt
    fissure = (ocd - icd) / 2.0
    rise = fissure * np.tan(tilt)
    inner = np.array([[-icd / 2, eye_y,        cfwd],
                      [+icd / 2, eye_y,        cfwd]])
    outer = np.array([[-ocd / 2, eye_y + rise, cfwd],
                      [+ocd / 2, eye_y + rise, cfwd]])
    return Subject(ipd, globe_r, ep, kappa, cor, inner, outer)
