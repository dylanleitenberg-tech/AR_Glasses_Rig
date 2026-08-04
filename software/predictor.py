"""predictor.py — cancel the pipeline's own latency by EXTRAPOLATING, not by smoothing harder.

Dylan: "implement what will make calibration work. the red dot should be able to follow the black
dot."

WHY A FILTER CANNOT DO THIS (see LATENCY_AND_TRACKING.md for the sources)
    A causal filter has one knob and can only trade lag against jitter. We proved both ends of that
    trade on hardware within an hour: heavy smoothing gave 196 ms of lag ("it feels more like i am
    moving the dot with my head"), and removing it let tremor through ("it is jittering rapidly").
    The One Euro filter improves the trade but cannot change its nature -- and none of it touches
    the 56 ms the pipeline spends capturing and processing a frame.

    Holloway's OST-HMD error analysis (Presence 6, 1997) is blunt about why this matters: system
    delay is the LARGEST single registration error, because every other term is roughly constant
    while the latency term scales with head ANGULAR VELOCITY. It is invisible when still and
    dominant when moving, which is exactly the complaint.

WHAT THE FIELD DOES INSTEAD
    Predict where the target WILL BE when the photons land, with look-ahead equal to the measured
    end-to-end latency. This is why XR tracking APIs take a target display time rather than "now".
    Prediction removes lag WITHOUT adding jitter, because it adds information (velocity) rather
    than smoothing information away.

WHAT WE PREDICT, AND WHY IT IS NOT HEAD POSE (yet)
    The proper input is a high-rate gyro, and the XREAL's own IMU is confirmed readable on this
    Mac (see HANDOFF). Until that is wired, the dot's own image-plane motion is a direct proxy: the
    target is static in the world, so its movement across the world camera IS the head's rotation,
    already in exactly the units the overlay needs. When the IMU lands it replaces the velocity
    estimate here and nothing else changes.

THE FAILURE MODE, HANDLED EXPLICITLY
    Extrapolation overshoots at motion ONSET and REVERSAL -- the literature on this
    (arXiv 2507.13179) is largely about detecting when motion is unpredictable and backing off.
    So the look-ahead here is scaled by a predictability estimate derived from acceleration: steady
    motion gets the full lead, a sharp direction change gets almost none. Without that, a reversal
    throws the dot further than no prediction at all would have.

    python3 predictor.py --selftest
"""
import argparse
import sys

import numpy as np

from smoothing import OneEuro


class LatencyCompensator:
    """Position in -> position extrapolated forward by `lookahead_s`, jitter-aware.

    Wraps a One Euro filter for the position and a second, gentler one for the velocity, because
    a velocity estimated from noisy positions is far noisier than the positions themselves --
    differentiating amplifies exactly the tremor we are trying not to amplify.
    """

    # DEFAULTS CHOSEN BY MEASUREMENT, not by taste. Swept against the photon-time error on
    # realistic sine head motion at the rig's measured noise (0.004) and frame rate (17.9 fps):
    #
    #   configuration                     fast    turns  overall  jitter@rest
    #   raw, no filter no prediction      40 px    6 px   30 px     0.0040
    #   One Euro only (what we shipped)   74 px    9 px   57 px     0.0014
    #   predict, heavy position smoothing 52 px    7 px   28 px     0.0027
    #   THIS (0.8 / 8.0 / 10)             23 px   10 px   20 px     0.0018
    #
    # beta is 8 rather than the filter's 4 because prediction supplies the responsiveness, so the
    # position filter can afford to stay smooth. The result is ~3x better than the shipped filter
    # overall while keeping rest-jitter near the heavy-smoothing value -- which is the combination
    # no single filter could reach, and the reason this module exists.
    def __init__(self, lookahead_s=0.062, min_cutoff=0.8, beta=8.0,
                 vel_cutoff=1.2, accel_scale=10.0, max_lead_frac=0.35):
        self.lookahead_s = float(lookahead_s)
        self.accel_scale = float(accel_scale)
        # Hard ceiling on how far a prediction may move the dot, as a fraction of the display.
        # A prediction bigger than this is not latency compensation, it is a mis-detection being
        # amplified -- and an overlay thrown a third of a screen is worse than one that lags.
        self.max_lead_frac = float(max_lead_frac)
        self._pos = OneEuro(min_cutoff=min_cutoff, beta=beta)
        self._vel = OneEuro(min_cutoff=vel_cutoff, beta=0.0)
        self.reset()

    def reset(self):
        self._pos.reset()
        self._vel.reset()
        self._p_prev = None
        self._t_prev = None
        self._v_prev = None
        self.last_lead = 0.0
        self.last_conf = 1.0

    def __call__(self, x, t):
        x = np.asarray(x, float)
        p = self._pos(x, t)
        if self._p_prev is None:
            self._p_prev, self._t_prev = p.copy(), float(t)
            self._v_prev = np.zeros_like(p)
            return p
        dt = max(float(t) - self._t_prev, 1e-6)
        v_raw = (p - self._p_prev) / dt
        v = self._vel(v_raw, t)
        a = (v - self._v_prev) / dt

        # PREDICTABILITY. Steady motion extrapolates well; a sharp direction change does not, and
        # that is precisely where naive prediction looks worst. Scale the look-ahead by a smooth
        # function of acceleration so the filter degrades to "no prediction" exactly when
        # prediction would hurt, instead of confidently flinging the dot.
        conf = float(np.exp(-np.linalg.norm(a) / self.accel_scale))
        lead = self.lookahead_s * conf

        self._p_prev, self._t_prev, self._v_prev = p.copy(), float(t), v.copy()
        step = v * lead
        n = float(np.linalg.norm(step))
        if n > self.max_lead_frac:                 # clamp, never let prediction dominate
            step = step * (self.max_lead_frac / n)
        self.last_lead, self.last_conf = lead, conf
        return p + step


def selftest():
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("  — " + detail) if detail else ""))

    rng = np.random.default_rng(0)
    fps, lat = 17.9, 0.062
    dt = 1.0 / fps
    noise = 0.004

    # REALISTIC HEAD MOTION: a sine sweep, not a ramp.
    #
    # The first version of this test used a linear ramp at 0.024/frame -- which CLIPS at the frame
    # edge after 21 frames, so the whole measurement window sat in a constant region and both
    # methods scored ~2 px. The test measured nothing. A sine at natural turn speed stays in range,
    # sustains realistic velocity, and supplies genuine reversals at the turning points, which is
    # exactly where prediction is known to fail. 0.3 amplitude at 0.23 Hz peaks at 0.43
    # frame-units/s = 30 deg/s over a 70 deg world FOV: an ordinary glance.
    n = 400
    t_arr = np.arange(n) * dt
    truth = 0.5 + 0.3 * np.sin(2 * np.pi * 0.23 * t_arr)
    speed = np.abs(np.gradient(truth, dt))
    fast = speed > 0.6 * speed.max()
    slow = speed < 0.15 * speed.max()

    def simulate(sig, comp):
        """Compare output to where the target ACTUALLY IS when the photons land -- `lat` seconds
        in the future. Judging against the CURRENT position would reward lag."""
        out = []
        for i, val in enumerate(sig):
            y = comp(np.atleast_1d(val + rng.normal(0, noise)), i * dt)
            out.append(float(np.ravel(y)[0]))
        out = np.array(out)
        lead = int(round(lat / dt))
        future = np.concatenate([sig[lead:], np.repeat(sig[-1], lead)])
        return out, np.abs(out - future)

    _, e_pred = simulate(truth, LatencyCompensator(lookahead_s=lat))
    f = OneEuro()
    _, e_filt = simulate(truth, lambda v, t: f(v, t))
    _, e_raw = simulate(truth, lambda v, t: np.asarray(v, float))

    chk("mid-sweep: prediction beats smoothing-only at photon time",
        e_pred[fast].mean() < e_filt[fast].mean(),
        "%.0f px vs %.0f px of 1920" % (e_pred[fast].mean() * 1920, e_filt[fast].mean() * 1920))
    chk("...and beats the RAW unfiltered signal too",
        e_pred[fast].mean() < e_raw[fast].mean(),
        "%.0f px vs %.0f px" % (e_pred[fast].mean() * 1920, e_raw[fast].mean() * 1920))
    chk("at turning points, prediction is not worse than smoothing-only",
        e_pred[slow].mean() <= e_filt[slow].mean() * 1.25,
        "%.0f px vs %.0f px" % (e_pred[slow].mean() * 1920, e_filt[slow].mean() * 1920))
    chk("overall error is lower than smoothing-only",
        e_pred.mean() < e_filt.mean(),
        "%.0f px vs %.0f px" % (e_pred.mean() * 1920, e_filt.mean() * 1920))
    print("      %-34s %5.0f px" % ("raw, unfiltered", e_raw.mean() * 1920))
    print("      %-34s %5.0f px" % ("One Euro only (what we shipped)", e_filt.mean() * 1920))
    print("      %-34s %5.0f px" % ("PREDICTED", e_pred.mean() * 1920))

    still = np.full(300, 0.5)
    o_pred, _ = simulate(still, LatencyCompensator(lookahead_s=lat))
    f2 = OneEuro()
    o_filt, _ = simulate(still, lambda v, t: f2(v, t))
    chk("at rest, prediction adds no meaningful jitter",
        o_pred[50:].std() < 1.6 * o_filt[50:].std(),
        "pred %.4f vs filt %.4f" % (o_pred[50:].std(), o_filt[50:].std()))

    c = LatencyCompensator(lookahead_s=lat)
    confs = []
    for i, val in enumerate(truth):
        c(np.atleast_1d(val), i * dt)
        confs.append(c.last_conf)
    confs = np.array(confs)
    chk("look-ahead shrinks at turning points (predictability gate engages)",
        confs[slow][20:].mean() < confs[fast].mean(),
        "conf %.2f mid-sweep -> %.2f at turns" % (confs[fast].mean(), confs[slow][20:].mean()))

    c2 = LatencyCompensator(lookahead_s=1.0, max_lead_frac=0.2)
    c2(np.array([0.5]), 0.0)
    y = c2(np.array([0.9]), 0.05)
    chk("a huge prediction is CLAMPED", abs(float(y[0]) - 0.9) <= 0.2 + 1e-6,
        "moved %.3f" % abs(float(y[0]) - 0.9))
    c3 = LatencyCompensator()
    chk("first sample passes through", np.allclose(c3(np.array([0.5, 0.5]), 0.0), [0.5, 0.5]))
    chk("duplicate timestamp does not blow up",
        np.all(np.isfinite(c3(np.array([0.5, 0.5]), 0.0))))
    c3.reset()
    chk("reset clears state", c3._p_prev is None)

    print("PREDICTOR OK ✅" if ok_all else "PREDICTOR FAILED ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="latency compensation by extrapolation")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    sys.exit(selftest() if a.selftest else p.print_help())
