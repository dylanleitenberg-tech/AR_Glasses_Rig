"""IMU / inclinometer ("level") model for the glasses (numpy only).

A level measures the glasses' tilt relative to GRAVITY, i.e. pitch & roll of the frame in
the world. What calibration needs is the glasses' tilt relative to the EYE (the
pantoscopic-tilt / slip term `dev[0]` and roll `dev[2]` in the rig frame). These differ by
the wearer's HEAD pose:

    tilt_vs_gravity  =  head_pose  +  tilt_on_face

So a raw level reading is the useful slip term BURIED UNDER the head pitch/roll, which
changes constantly as the user looks around. Two readings are modelled:

  * raw          — head confound included (what a bare accelerometer gives in the wild);
  * compensated  — head pose removed (what you get if a world camera or AHRS estimates head
                   pose, or the user holds the head level during a capture). This isolates
                   the pantoscopic-tilt slip term the calibration actually wants.

`tilt_on_face` uses the rig's rest pantoscopic tilt (`rig.PANTO_DEG`) plus the per-pose
deviation, so the IMU is consistent with the same pose `dev` the cameras and oracle see.
All magnitudes are PLACEHOLDERS at plausible values — replace with measured IMU specs.
"""
import numpy as np

import rig

# head pose vs gravity during a capture (the confound); users don't hold perfectly level
HEAD_PITCH_SD = 6.0     # deg, looking up/down + posture
HEAD_ROLL_SD = 2.0      # deg, head tilt
ACCEL_TILT_NOISE = 0.20 # deg, static accelerometer angle noise (good MEMS part)
COMP_RESIDUAL = 0.8     # deg, leftover error after head-pose compensation (imperfect)


def imu_tilt(dev, rng, compensated=False):
    """Glasses tilt vs gravity (pitch, roll) in degrees for glasses pose `dev`.

    dev = [drx,dry,drz, dtx,dty,dtz] deviation from rest (rig.pose_R_t convention).
    Pantoscopic pitch on the face = rig.PANTO_DEG + dev[0]; frame roll = dev[2].
    """
    on_face = np.array([rig.PANTO_DEG + dev[0], dev[2]])      # tilt relative to the face
    if compensated:
        # head pose subtracted; only an imperfect-compensation residual + sensor noise remain
        n = rng.normal(0, COMP_RESIDUAL, 2) + rng.normal(0, ACCEL_TILT_NOISE, 2)
        return on_face + n
    head = np.array([rng.normal(0, HEAD_PITCH_SD), rng.normal(0, HEAD_ROLL_SD)])
    return on_face + head + rng.normal(0, ACCEL_TILT_NOISE, 2)


# ===========================================================================
#  DRIFT-CANCELLING TILT FILTER  (Kalman fuse + active temporal smoothing)
# ---------------------------------------------------------------------------
#  Raw MEMS tilt has two failure modes: the GYRO integrates to a smooth angle but
#  its rate bias makes that angle DRIFT without bound; the ACCELEROMETER angle is
#  drift-free but noisy/spiky. The classic fix is a 2-state Kalman filter
#  (state = [angle, gyro_bias]) that PREDICTS with the gyro rate and CORRECTS with
#  the accel angle, so the fused tilt keeps the gyro's smoothness while the bias
#  state actively estimates and cancels the drift. A light exponential smoother on
#  the output is the "active temporal filtering" that further suppresses residual
#  drift artifacts/spikes. numpy/stdlib only — ports straight to the XIAO firmware.
# ===========================================================================
class TiltKalman:
    """2-state (angle, gyro-bias) Kalman tilt filter for ONE axis (pitch or roll).

    q_angle / q_bias: process noise on the angle and on the slowly-varying gyro bias.
    r_measure:        accelerometer-angle measurement noise (deg^2). Bigger -> trust the
                      gyro more (smoother, slower to correct); smaller -> trust the accel.
    Formulation after Lauszus' standard tilt Kalman; all angles in degrees, dt in seconds.
    """

    def __init__(self, q_angle=0.001, q_bias=0.003, r_measure=ACCEL_TILT_NOISE ** 2):
        self.q_angle = float(q_angle)
        self.q_bias = float(q_bias)
        self.r_measure = float(r_measure)
        self.angle = 0.0          # fused tilt estimate (deg)
        self.bias = 0.0           # estimated gyro rate bias (deg/s)
        self.P = np.zeros((2, 2))  # error covariance
        self._init = False

    def reset(self, angle=0.0):
        self.angle = float(angle); self.bias = 0.0
        self.P = np.zeros((2, 2)); self._init = True

    def update(self, accel_angle, gyro_rate, dt):
        """Fuse one accel angle (deg) + gyro rate (deg/s) over dt (s) -> drift-free tilt."""
        if not self._init:                      # seed on the first accel reading (no startup drift)
            self.reset(accel_angle); return self.angle
        # --- predict (gyro integration, bias-compensated) ---
        rate = gyro_rate - self.bias
        self.angle += dt * rate
        P = self.P
        P[0, 0] += dt * (dt * P[1, 1] - P[0, 1] - P[1, 0] + self.q_angle)
        P[0, 1] -= dt * P[1, 1]
        P[1, 0] -= dt * P[1, 1]
        P[1, 1] += self.q_bias * dt
        # --- correct with the (drift-free) accelerometer angle ---
        S = P[0, 0] + self.r_measure
        K0 = P[0, 0] / S
        K1 = P[1, 0] / S
        y = accel_angle - self.angle
        self.angle += K0 * y
        self.bias += K1 * y
        P00, P01 = P[0, 0], P[0, 1]
        P[0, 0] -= K0 * P00
        P[0, 1] -= K0 * P01
        P[1, 0] -= K1 * P00
        P[1, 1] -= K1 * P01
        return self.angle


class ImuFilter:
    """Two-axis (pitch, roll) drift-cancelling tilt filter for the glasses IMU.

    update() takes the accelerometer tilt (deg, drift-free, noisy) and the gyro rates
    (deg/s, smooth, drifting) and returns the fused, drift-corrected tilt. `smooth` adds an
    extra exponential (active temporal) pass on the Kalman output to flatten residual
    drift artifacts; set 0 to disable. If no gyro is available, pass rates=0 and the filter
    degrades gracefully to a denoising smoother of the accel angle.
    """

    def __init__(self, smooth=0.25, **kf):
        self.kf = (TiltKalman(**kf), TiltKalman(**kf))
        self.smooth = float(smooth)
        self._out = None

    def reset(self):
        for k in self.kf:
            k._init = False
        self._out = None

    def update(self, accel_tilt, gyro_rate=(0.0, 0.0), dt=1.0 / 200.0):
        accel_tilt = np.asarray(accel_tilt, float)
        gyro_rate = np.asarray(gyro_rate, float) if np.ndim(gyro_rate) else \
            np.array([gyro_rate, gyro_rate], float)
        fused = np.array([self.kf[i].update(accel_tilt[i], gyro_rate[i], dt)
                          for i in range(2)])
        if self.smooth > 0:                     # active temporal smoothing of the artifacts
            self._out = fused if self._out is None else (
                self.smooth * fused + (1 - self.smooth) * self._out)
            return self._out.copy()
        return fused


def selftest(n=4000, seed=0, verbose=True):
    """Prove the Kalman filter removes gyro drift: a drifting raw gyro-integration vs the
    bias-cancelling filter, both fed the SAME noisy accel + biased gyro of a known true tilt."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 200.0
    gyro_bias_true = np.array([0.8, -0.5])          # deg/s constant gyro bias -> unbounded drift
    t = np.arange(n) * dt
    true_tilt = np.column_stack([rig.PANTO_DEG + 3.0 * np.sin(0.3 * t),  # slow real head motion
                                 2.0 * np.sin(0.21 * t)])
    filt = ImuFilter(smooth=0.2)
    raw_angle = true_tilt[0].copy()                 # naive gyro-only integration (drifts)
    raw_err, kal_err = [], []
    for i in range(n):
        rate_true = (true_tilt[i] - true_tilt[i - 1]) / dt if i > 0 else np.zeros(2)
        gyro = rate_true + gyro_bias_true + rng.normal(0, 0.05, 2)   # biased, noisy gyro
        accel = true_tilt[i] + rng.normal(0, ACCEL_TILT_NOISE, 2)    # noisy, drift-free accel
        raw_angle = raw_angle + dt * gyro           # gyro-only -> accumulates the bias as drift
        kal = filt.update(accel, gyro, dt)
        if i > n // 4:                              # skip warm-up
            raw_err.append(np.abs(raw_angle - true_tilt[i]))
            kal_err.append(np.abs(kal - true_tilt[i]))
    raw_rms = float(np.sqrt(np.mean(np.square(raw_err))))
    kal_rms = float(np.sqrt(np.mean(np.square(kal_err))))
    accel_rms = ACCEL_TILT_NOISE
    ok = kal_rms < accel_rms and kal_rms < raw_rms / 5.0
    if verbose:
        print("== IMU Kalman drift self-test (200 Hz, %.1fs) ==" % (n * dt))
        print("  raw gyro-integration drift RMS : %7.2f deg  (unbounded -> grows with time)" % raw_rms)
        print("  raw accel-only noise RMS       : %7.2f deg  (drift-free but noisy)" % accel_rms)
        print("  Kalman fused tilt error RMS    : %7.2f deg  (smooth AND drift-free)" % kal_rms)
        print("  =>", "IMU FILTER OK — drift cancelled, accel noise smoothed ✅" if ok
              else "WEAK ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
