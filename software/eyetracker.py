"""Full NIR eye-tracker model + what it does for REGISTRATION accuracy.

The pupil/iris sensor (pupil_sensor.py) answered the geometry-ID question. This answers the
product question: if the glasses MEASURE gaze directly (PCCR: pupil centre + corneal glints
from known IR LEDs), how much better does the overlay register, and how much faster does a
new user calibrate?

What the tracker gives: the eye's OPTICAL axis (gaze) as two angles in the rig frame, with
realistic per-attempt accuracy (~0.7 deg; cf. Quest Pro ~1.08 deg). Crucially it does NOT
give the VISUAL axis — the angle-kappa offset between them is unobservable, so the per-user
kappa + perceptual bias must STILL be learned. So the prediction is:
  * +gaze removes the slip/parallax UNCERTAINTY the eye-corner cams can only infer
    -> faster convergence + lower error, BUT
  * it cannot beat the kappa/perceptual-bias floor -> the curves meet a common floor.
This script measures both, paired (same user + same observation seed for both arms).

Run:  python3 eyetracker.py        (or:  python3 main.py --eyetracker)
"""
import numpy as np

import rig
import autosim
import pupil_sensor
from calibrator import Calibrator

# MODELLED per-attempt gaze sigma (deg). Placeholder (cf. Quest Pro ~1.08 deg) — MEASURE on
# hardware and set it here. This is the sensor's accuracy, NOT a tunable gain: shrinking it just
# makes the sim optimistic, and using it raw with no filtering is what makes live gaze jittery.
# Live tracking STABILITY comes from GazeStabilizer (temporal filter), not from this number.
GAZE_NOISE_DEG = 0.7
GAZE_NOISE_MAX = 2.0        # sanity clamp: a configured sigma above this is almost certainly wrong
_PUPIL_CAM = pupil_sensor.build_camera()


def gaze_read(subject, dev, P, rng, noise_deg=None):
    """Measured optical-axis gaze as (az, el) radians in the rig frame, with tracker noise.

    `noise_deg` overrides the module default so the per-attempt sigma is no longer hardcoded
    inside the function — pass your MEASURED tracker accuracy. Clamped to [0, GAZE_NOISE_MAX]
    so a stray large value can't silently destabilise tracking."""
    nd = GAZE_NOISE_DEG if noise_deg is None else float(noise_deg)
    nd = float(np.clip(nd, 0.0, GAZE_NOISE_MAX))
    R, t = rig.pose_R_t(dev)
    cor = (R @ subject.cor.T).T + t
    g = P - cor[rig.DISPLAY_EYE]
    g = g / np.linalg.norm(g)
    az = np.arctan2(g[0], g[2]) + np.radians(rng.normal(0, nd))
    el = np.arctan2(g[1], g[2]) + np.radians(rng.normal(0, nd))
    return np.array([az, el])


class GazeStabilizer:
    """Live gaze temporal filter — the actual remedy for the jitter/instability that the raw
    per-attempt noise (GAZE_NOISE_DEG) would cause if fed straight to the overlay.

    Velocity-adaptive exponential smoother (one-euro style): heavy smoothing while the eye
    fixates (kills jitter), light smoothing during a saccade (keeps up, no lag). Returns the
    filtered (az, el). Drop it in front of the live gaze stream; it does not touch the
    simulator's accuracy model above.
    """

    def __init__(self, min_cut=0.6, beta=0.10):
        self.min_cut = float(min_cut)   # Hz: cutoff at zero velocity (smaller = steadier fixation)
        self.beta = float(beta)         # how fast the cutoff opens up with gaze speed
        self._prev = None

    def reset(self):
        self._prev = None

    def update(self, gaze, dt=1.0 / 120.0):
        gaze = np.asarray(gaze, float)
        if self._prev is None:
            self._prev = gaze.copy()
            return gaze
        speed = float(np.linalg.norm(gaze - self._prev)) / max(dt, 1e-6)
        cut = self.min_cut + self.beta * speed       # raise the cutoff with speed -> less lag
        tau = 1.0 / (2 * np.pi * cut)
        a = dt / (dt + tau)                          # EMA gain in [0,1]
        self._prev = a * gaze + (1 - a) * self._prev
        return self._prev.copy()


def _new_cal(nf):
    return Calibrator(nf, degree=2, min_samples=6,
                      lambdas=np.logspace(-4.0, 1.5, 18), robust_iters=2)


def _extra(mode, subject, dev, P, rng):
    """Extra eye-tracker features for the chosen arm (or empty)."""
    if mode == "gaze":
        return gaze_read(subject, dev, P, rng)
    if mode == "pupil":                          # measured eye POSITION (parallax-sensitive)
        r = pupil_sensor.pupil_read(subject, dev, P, rng, _PUPIL_CAM)
        return r[0] if r is not None else np.zeros(2)
    return np.empty(0)


def train_user(sim, subject, mode, tol=0.010, max_iters=240, patience=10):
    """Cold-start calibration loop. Returns (iters_to_converge, median overlay error px@1080).
    mode: 'none' (8 corner/world feats), 'gaze' (+2 gaze angles), 'pupil' (+2 pupil-centre)."""
    nf = 8 + (0 if mode == "none" else 2)
    cal = _new_cal(nf)
    dev = sim.seat()
    X, YL, YT = [], [], []
    window = []
    iters = max_iters
    for it in range(1, max_iters + 1):
        dev = sim.slip(dev)
        P = sim.world_point()
        obs = sim.observe(subject, dev, P)
        if obs is None:
            continue
        f, label, truth = obs
        ex = _extra(mode, subject, dev, P, sim.rng)
        if ex.size:
            f = np.concatenate([f, ex])
        err = float(np.linalg.norm(cal.predict(f) - truth))
        X.append(f); YL.append(label); YT.append(truth)
        cal.fit(np.array(X), np.array(YL))
        window.append(err)
        if len(window) > patience:
            window.pop(0)
        if len(window) >= patience and float(np.mean(window)) < tol:
            iters = it
            break
    floor = float(np.median(window)) * 1080.0 if window else float("nan")
    return iters, floor


def compare(n_users=40, seed_base=4000, verbose=True):
    """Paired comparison of three feature arms across held-out users (same user+seed/pair)."""
    modes = ["none", "gaze", "pupil"]
    res = {m: [] for m in modes}
    for i in range(n_users):
        for m in modes:
            s = autosim.Simulator(seed_base + i); su = s.new_subject()
            res[m].append(train_user(s, su, m))
    R = {m: np.array(res[m], float) for m in modes}
    if verbose:
        print("== eye-tracker feature arms vs eye-corner cameras only ==")
        print("   paired over %d brand-new users (cold start; same user+seed per arm)\n" % n_users)
        print("   metric                       corner-only   +gaze(angle)  +pupil(position)")
        print("   --------------------------------------------------------------------------")
        print("   samples to converge (median) %9.0f %12.0f %14.0f"
              % (np.median(R['none'][:, 0]), np.median(R['gaze'][:, 0]), np.median(R['pupil'][:, 0])))
        print("   overlay error px (median)    %9.1f %12.1f %14.1f"
              % (np.median(R['none'][:, 1]), np.median(R['gaze'][:, 1]), np.median(R['pupil'][:, 1])))
        print("   overlay error px (90th pct)  %9.1f %12.1f %14.1f"
              % (np.percentile(R['none'][:, 1], 90), np.percentile(R['gaze'][:, 1], 90),
                 np.percentile(R['pupil'][:, 1], 90)))
        print("\n   reading: gaze ANGLE is redundant with the world cameras (they already see the")
        print("   target direction); the lever for registration is eye POSITION (pupil centre),")
        print("   which fights slip parallax. Neither beats the kappa/perceptual-bias floor, so a")
        print("   short per-user calibration is always required — eye tracking buys speed + slip-")
        print("   robustness + (via pupil/iris) the geometry ID that the corner cameras can't get.")
    return R


if __name__ == "__main__":
    compare()
