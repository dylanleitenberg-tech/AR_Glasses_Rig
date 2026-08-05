"""NIR pupil / iris sensor model, the sensor that breaks the identifiability wall.

WHY this exists
---------------
The fingerprint-matching study (`match.py`) found two eye parameters that the eye-corner
cameras + guess inaccuracy CANNOT recover:

 * entrance-pupil depth `ep_dist`, it IS in the physics (the entrance pupil sits
    `ep_dist` ahead of the center of rotation and drives parallax), but its imprint on the
    registration error is tiny, so it stays buried in noise (~80% of population spread).
 * eyeball radius `globe_r`, it is in the anatomy descriptor but has NO effect on any
    current observable, so it is non-identifiable BY CONSTRUCTION (recovery ~= guessing the
    mean). A sensor can only recover it once the model gives it a physical effect.

This module adds the sensor the eye-tracking upgrade already calls for (an NIR camera at the
display eye imaging the pupil + iris, cf. EYE_TRACKING.md / pupil_tracker.py) and the two
observables that make those parameters identifiable:

  1. PUPIL CENTRE across gaze. The entrance pupil is `cor + ep_dist*gaze`; as the eye
     rotates to fixate different dots it sweeps a sphere of radius `ep_dist` about the
     centre of rotation. An eye camera sees that arc, so fitting it across several gaze
     directions recovers `ep_dist` directly (geometry, not buried in registration noise).
  2. IRIS DIAMETER. The visible iris (HVID ~11.7 mm) scales with eyeball size, so the
     imaged iris diameter is a direct, if noisy, read on `globe_r`. This is the physical
     effect `globe_r` was missing.

The camera is modelled separately from rig.build so the validated registration pipeline is
untouched; it only adds NEW observables for identification. Magnitudes are PLACEHOLDERS at
plausible values, replace with the measured OV9281 + IR-LED rig once built.
"""
import numpy as np

import rig
import anatomy
from optics import look_at, PinholeCamera

# iris size model: HVID correlates with axial length / eyeball radius
IRIS_R_MEAN = 5.85                 # mm, mean visible iris radius (HVID/2)
IRIS_R_PER_GLOBE = 0.55            # mm visible-iris-radius per mm of globe_r deviation
NIR_PIX_SD = 0.0016                # pupil-centre localisation noise (norm, ~1 px @640)
IRIS_DIAM_SD = 0.004               # iris-diameter measurement noise (norm)

def build_camera():
    # single source of truth: the same NIR pupil cam the rig/simulator use (rig.py)
    return rig.build_pupil_camera()


def pupil_read(subject, dev, P, rng, cam):
    """Imaged (pupil_centre_xy (2,), iris_diameter (1,)) for this face/pose/gaze, or None.

    pupil_centre encodes ep_dist (+ cor); iris_diameter encodes globe_r.
    """
    R, t = rig.pose_R_t(dev)
    cor = (R @ subject.cor.T).T + t
    e = rig.DISPLAY_EYE
    gaze = P - cor[e]
    gaze = gaze / np.linalg.norm(gaze)
    pupil = cor[e] + subject.ep_dist * gaze          # the entrance pupil (same as the oracle)
    uv = cam.project(pupil)
    if uv is None:
        return None
    uv = uv + rng.normal(0, NIR_PIX_SD, 2)

    iris_r = IRIS_R_MEAN + IRIS_R_PER_GLOBE * (subject.globe_r - anatomy.GLOBE_R_MEAN)
    dist = np.linalg.norm(pupil - cam.pos)
    diam = cam.f * (2.0 * iris_r) / dist             # imaged iris diameter (normalised)
    diam = diam + rng.normal(0, IRIS_DIAM_SD)
    return uv, np.array([diam])
