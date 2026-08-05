"""smoothing.py, a velocity-adaptive filter: still when you are still, fast when you move.

THE TRADE-OFF THIS EXISTS TO BREAK. A fixed EMA has one knob and two jobs it cannot do at once.
Both failures were observed on this rig within an hour of each other:

  * HEAVY (feat_smooth 0.4 + pred_smooth 0.5): 140 ms of added lag on a 56 ms pipeline. Dylan:
    "it feels more like i am moving the dot with my head than the dot is trying to reach the point."
  * NONE (the fix for that): detection noise and body tremor pass straight through. Dylan:
    "it is jittering rapidly. i think it is overestimating how much it needs to move based on my
    body's vibration."

Turning one knob just moves the pain between those. The information a fixed filter throws away is
that JITTER AND MOTION LOOK DIFFERENT: jitter is small and fast and reverses; real head motion is
sustained and directional. Adapting the cutoff to the measured SPEED uses exactly that.

THE ONE EURO FILTER (Casiez, Roussel & Vogel, CHI 2012) is the standard answer for noisy human
motion in interactive systems, and it is about twenty lines:

    cutoff = min_cutoff + beta * |velocity|

At rest the cutoff is low, so the signal is smoothed hard and jitter dies. While moving, the
cutoff rises with speed, so lag collapses. `beta` sets how aggressively it trades one for the
other; `min_cutoff` sets how still it is when you are still.

Dylan's instruction was "reduce the amount it adjusts for small movements", which is precisely
what a low `min_cutoff` does -- and this way it costs no lag on the large movements that matter.

    python3 smoothing.py --selftest
"""
import argparse
import math
import sys

import numpy as np


class OneEuro:
    """Scalar or vector One Euro filter. Call with (value, timestamp_seconds)."""

    # DEFAULTS TUNED TO THIS RIG'S UNITS AND SPEEDS, not copied from the paper.
    #
    # `speed` here is in FRAME-UNITS PER SECOND of the world camera. The world FOV is 70 deg, so a
    # 30 deg/s head turn -- an ordinary glance -- is 30/70 = 0.43 units/s, and 100 deg/s is 1.43.
    # beta must therefore be O(1..10) for the cutoff to open meaningfully at those speeds; the
    # paper's small betas assume pixel units and are ~100x off here. A first attempt at beta=0.02
    # left the cutoff pinned at min_cutoff and lagged WORSE than a heavy EMA -- caught by the
    # selftest below, which is why that comparison is pinned rather than assumed.
    #
    #   at rest      cutoff 0.8 Hz  -> alpha 0.22 at 17.9 fps: heavy smoothing, tremor dies
    #   at 30 deg/s  cutoff 2.5 Hz  -> alpha 0.47: about one frame of lag
    #   at 100 deg/s cutoff 6.5 Hz  -> alpha 0.70: essentially keeps up
    def __init__(self, min_cutoff=0.8, beta=4.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = float(t)
            return x.copy()
        dt = max(float(t) - self._t_prev, 1e-6)
        self._t_prev = float(t)
        # derivative, itself low-passed so noise does not inflate the speed estimate and
        # accidentally open the filter wide at rest
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx_prev = a_d * dx + (1 - a_d) * self._dx_prev
        speed = float(np.linalg.norm(self._dx_prev))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        self._x_prev = a * x + (1 - a) * self._x_prev
        return self._x_prev.copy()


def selftest():
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (", " + detail) if detail else ""))

    rng = np.random.default_rng(0)
    fps = 17.9
    dt = 1.0 / fps
    noise = 0.004

    # INTERMITTENT motion -- what a head actually does, and what a FIXED filter cannot serve.
    # A steady ramp is the easy case: a fixed EMA tracks constant velocity with a constant offset,
    # so it can look fine on both axes at once. The first version of this test used a steady ramp
    # and duly "proved" a fixed EMA was adequate, which is not the situation on the rig. Real use
    # is: hold still (tremor only), turn quickly, hold still again.
    truth = [0.5] * 120
    for _ in range(3):
        for _k in range(25):                    # ~30 deg/s = 0.024 frame-units/frame
            truth.append(truth[-1] + 0.024)
        truth += [truth[-1]] * 80
    truth = np.clip(np.array(truth), 0.0, 1.0)
    v = np.abs(np.diff(truth, prepend=truth[0]))
    still_mask = np.zeros(len(truth), bool); still_mask[40:] = v[40:] < 1e-9
    moving_mask = v > 1e-9
    sig = truth + rng.normal(0, noise, len(truth))
    ts = np.arange(len(truth)) * dt

    def ema(x, a):
        out = np.zeros_like(x); out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = a * x[i] + (1 - a) * out[i - 1]
        return out

    f = OneEuro()
    oe = np.array([f(np.array([val]), t)[0] for val, t in zip(sig, ts)])
    jitter = lambda y: float(np.std(y[still_mask] - truth[still_mask]))
    lag = lambda y: float(np.mean(np.abs(y[moving_mask] - truth[moving_mask])))
    oe_j, oe_l, raw_j = jitter(oe), lag(oe), jitter(sig)

    chk("at rest, One Euro removes most tremor", oe_j < 0.5 * raw_j,
        "%.4f -> %.4f (%.0f%% removed)" % (raw_j, oe_j, 100 * (1 - oe_j / raw_j)))
    chk("while moving, One Euro keeps up", oe_l * 1920 < 60, "%.0f px of 1920" % (oe_l * 1920))

    # THE PREMISE, pinned: no single fixed EMA beats it on BOTH axes at once. If one ever does,
    # this module is not earning its keep and the simpler filter should be used instead.
    beaten = [round(float(a), 2) for a in np.arange(0.1, 1.01, 0.05)
              if jitter(ema(sig, a)) <= oe_j and lag(ema(sig, a)) <= oe_l]
    chk("NO fixed EMA beats it on jitter AND lag together", not beaten,
        "one-euro jitter %.4f / lag %.0f px; beaten by %s" % (oe_j, oe_l * 1920, beaten))

    for a, label in ((0.4, "heavy EMA 0.4 (the laggy one)"),
                     (0.9, "light EMA 0.9 (the jittery one)")):
        y = ema(sig, a)
        print("      %-32s jitter %.4f   lag %3.0f px" % (label, jitter(y), lag(y) * 1920))
    print("      %-32s jitter %.4f   lag %3.0f px" % ("ONE EURO", oe_j, oe_l * 1920))

    f3 = OneEuro()
    chk("first sample passes through unchanged", np.allclose(f3(np.array([0.5, 0.5]), 0.0), [0.5, 0.5]))
    chk("duplicate timestamp does not blow up", np.all(np.isfinite(f3(np.array([0.5, 0.5]), 0.0))))
    f3.reset()
    chk("reset clears state", f3._x_prev is None)

    print("SMOOTHING OK ✅" if ok_all else "SMOOTHING FAILED ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="velocity-adaptive smoothing")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    sys.exit(selftest() if a.selftest else p.print_help())
