"""autoexpose.py, continuous, per-role exposure/gain control so the cameras keep a usable
image as light changes (the "the cams constantly track and adjust the image" requirement).

WHY per-role, not the driver's auto-exposure:
  * The NIR PUPIL cams must stay DARK-PUPIL with crisp, UN-saturated glints (the PCCR corner
    signal). Driver auto-exposure hunts for a mid-gray whole-frame average and blows the glints
    out. We instead target a high PERCENTILE (the glints) just below clipping and keep the field
    dark, the opposite of consumer AE.
  * The EYE-CORNER cams want the canthus mid-toned for stable template matching.
  * The WORLD cams want a mid-gray scene, but exposure CAPPED so motion stays sharp (short
    integration on a global shutter), the world-mesh tracker needs crisp corners, not a bright
    but smeared frame.
  * All cameras must adapt SLOWLY and STABLY (no oscillation) while the pipeline runs every frame.

The controller is a clamped integral loop on a single "brightness" actuator that maps to
exposure first, then gain once exposure is maxed (gain adds noise, so it's the last resort). It
never fights itself: a deadband around the target stops hunting, and the step is proportional to
the error so it converges fast then settles.

Runs headless: a synthetic camera whose measured metric is a monotonic function of the actuator
lets `--selftest` prove convergence, stability, no-windup, and the per-role targeting.
"""
import argparse
import sys

import numpy as np


# Per-role control policy. metric = which image statistic we regulate; target/deadband in [0,1]
# (fraction of 255). expo_cap keeps world integration short for sharpness.
ROLE_POLICY = {
    "worldL": dict(metric="mean", target=0.42, deadband=0.05, expo_cap=0.75),
    "worldR": dict(metric="mean", target=0.42, deadband=0.05, expo_cap=0.75),
    "eyeL":   dict(metric="mean", target=0.50, deadband=0.05, expo_cap=1.0),
    "eyeR":   dict(metric="mean", target=0.50, deadband=0.05, expo_cap=1.0),
    # pupil: regulate the 99th percentile (the glints) to sit just under clipping, field stays dark
    "pupilL": dict(metric="p99",  target=0.85, deadband=0.04, expo_cap=1.0),
    "pupilR": dict(metric="p99",  target=0.85, deadband=0.04, expo_cap=1.0),
    "eye2L":  dict(metric="mean", target=0.50, deadband=0.05, expo_cap=1.0),
    "eye2R":  dict(metric="mean", target=0.50, deadband=0.05, expo_cap=1.0),
}


def frame_metric(frame, metric):
    """Normalized (0..1) brightness statistic used by the controller. Grayscale-agnostic."""
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32) / 255.0
    if metric == "mean":
        return float(a.mean())
    if metric == "p99":
        return float(np.percentile(a, 99))
    if metric == "p95":
        return float(np.percentile(a, 95))
    raise ValueError("unknown metric %r" % metric)


class ExposureController:
    """One camera's closed loop. `actuator` is an abstract brightness command in [0,1] that the
    loop splits into exposure (0..expo_cap) then gain (the remainder), exposure is preferred
    because gain adds sensor noise. Call update(frame) every frame; it returns the new actuator
    and (when a `cap`/backend is attached) pushes it to the device."""

    def __init__(self, role, gain=0.6, start=0.5, apply=None, policy=None):
        self.role = role
        self.p = dict(policy or ROLE_POLICY.get(role, ROLE_POLICY["eyeL"]))
        self.k = gain                    # proportional step (fraction of error per frame)
        self.a = float(np.clip(start, 0.0, 1.0))   # actuator state
        self.apply = apply               # optional callback(role, exposure01, gain01)
        self.settled = False
        self._hist = []

    # actuator -> (exposure, gain), both normalized 0..1
    def _split(self):
        cap = self.p["expo_cap"]
        expo = min(self.a, cap)
        gain = 0.0 if self.a <= cap else (self.a - cap) / max(1e-6, 1.0 - cap)
        return expo, gain

    def update(self, frame):
        m = frame_metric(frame, self.p["metric"])
        err = self.p["target"] - m
        if abs(err) <= self.p["deadband"]:
            self.settled = True                    # inside deadband: hold, don't hunt
        else:
            self.settled = False
            self.a = float(np.clip(self.a + self.k * err, 0.0, 1.0))   # clamped integral step
        expo, gain = self._split()
        if self.apply is not None:
            self.apply(self.role, expo, gain)
        self._hist.append(m)
        return dict(metric=m, error=err, actuator=self.a, exposure=expo, gain=gain,
                    settled=self.settled)

    def state(self):
        expo, gain = self._split()
        return dict(role=self.role, actuator=self.a, exposure=expo, gain=gain,
                    settled=self.settled)


def make_cv2_apply(caps):
    """Build an apply(role, expo01, gain01) that pushes normalized commands to real cv2 cameras.
    caps: {role: cv2.VideoCapture}. Disables the driver's own auto-exposure first (once)."""
    import cv2
    done = set()

    def apply(role, expo01, gain01):
        cap = caps.get(role)
        if cap is None:
            return
        if role not in done:               # take manual control (0.25 = manual on most UVC/AVF)
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            except Exception:
                pass
            done.add(role)
        # UVC exposure is driver-scaled; map 0..1 across the property's usable span. These
        # bounds are conservative and re-tuned per sensor in rig_test.py --tune.
        try:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(-11 + expo01 * 9))   # ~2^-11..2^-2 s (log scale)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_GAIN, float(gain01))
        except Exception:
            pass
    return apply


class AutoExposureBank:
    """One controller per role, driven each frame from a SyncFrame (or any {role: frame})."""

    def __init__(self, roles, apply=None, **kw):
        self.ctl = {r: ExposureController(r, apply=apply, **kw) for r in roles}

    def update(self, frames):
        out = {}
        for role, c in self.ctl.items():
            f = frames.get(role) if hasattr(frames, "get") else frames[role]
            if f is not None:
                out[role] = c.update(f)
        return out

    def all_settled(self):
        return all(c.settled for c in self.ctl.values())

    def states(self):
        return {r: c.state() for r, c in self.ctl.items()}


# ==========================================================================
#  Self-test (no hardware): synthetic sensor whose measured metric is a smooth
#  monotonic function of the actuator + scene light. Proves convergence, no
#  windup, stability, and the per-role (pupil p99 vs world mean) targeting.
# ==========================================================================
class _SynthSensor:
    """measured_metric ≈ saturating(actuator * scene_gain). Distinct response per role +
    a mid-run light change to show the loop keeps adjusting continuously."""

    def __init__(self, role, scene=1.0, rng=None):
        self.role = role
        self.scene = scene
        self.rng = rng or np.random.default_rng(0)
        self.metric = ROLE_POLICY[role]["metric"]

    def frame(self, actuator):
        # exposure*scene, soft-saturated to [0,1], plus small measurement noise
        lit = 1.0 - np.exp(-3.0 * actuator * self.scene)
        lit = float(np.clip(lit + self.rng.normal(0, 0.005), 0, 1))
        w, h = 64, 48
        if self.metric.startswith("p"):
            # dark-pupil model: a dark field + a bright glint cluster (~3% of pixels, so the
            # 99th percentile actually lands ON the glints) that tracks `lit`. Field stays dark
            # so the whole-frame mean is low while p99 is hot.
            img = self.rng.random((h, w)) * 0.04
            nbright = int(0.03 * w * h)
            img.flat[self.rng.integers(0, w * h, nbright)] = lit
            return (img * 255).astype("uint8")
        base = np.full((h, w, 3) if self.role.startswith("world") else (h, w), lit)
        return (np.clip(base + self.rng.normal(0, 0.01, base.shape), 0, 1) * 255).astype("uint8")


def selftest(verbose=True):
    if verbose:
        print("== autoexpose self-test (synthetic sensors, no hardware) ==")
    roles = ("worldL", "eyeR", "pupilR")
    scenes = {"worldL": 0.6, "eyeR": 1.2, "pupilR": 1.1}    # different starting light per role
    rng = np.random.default_rng(1)
    sensors = {r: _SynthSensor(r, scene=scenes[r], rng=rng) for r in roles}
    bank = AutoExposureBank(roles, gain=0.8, start=0.5)
    checks = []

    def run(nframes):
        last = {}
        for _ in range(nframes):
            frames = {r: sensors[r].frame(bank.ctl[r].a) for r in roles}
            last = bank.update(frames)
        return last

    # (1) converges: every role reaches its target within deadband inside 60 frames
    last = run(60)
    conv = all(abs(last[r]["error"]) <= ROLE_POLICY[r]["deadband"] + 1e-6 for r in roles)
    checks.append(("all roles converge to target within deadband (%s)"
                   % ", ".join("%s %.2f" % (r, last[r]["metric"]) for r in roles), conv))

    # (2) pupil regulates the GLINT percentile high while the field stays dark (dark-pupil)
    pf = sensors["pupilR"].frame(bank.ctl["pupilR"].a)
    field_dark = frame_metric(pf, "mean") < 0.25
    glint_hot = frame_metric(pf, "p99") > 0.7
    checks.append(("pupil cam: glints hot (p99>0.7) but field dark (mean<0.25), dark-pupil",
                   field_dark and glint_hot))

    # (3) stability: once settled, the actuator barely moves (no oscillation)
    a_before = {r: bank.ctl[r].a for r in roles}
    run(40)
    drift = max(abs(bank.ctl[r].a - a_before[r]) for r in roles)
    checks.append(("settled loop is stable (max actuator drift %.4f < 0.05)" % drift, drift < 0.05))

    # (4) keeps ADJUSTING when the light changes mid-run (continuous control)
    for r in roles:
        sensors[r].scene *= 0.7             # room got darker (targets still reachable)
    m_dark = {r: sensors[r].frame(bank.ctl[r].a) for r in roles}
    m0 = {r: frame_metric(m_dark[r], ROLE_POLICY[r]["metric"]) for r in roles}
    last = run(60)
    recovered = all(abs(last[r]["error"]) <= ROLE_POLICY[r]["deadband"] + 0.02 for r in roles)
    brightened = all(last[r]["metric"] > m0[r] for r in roles)
    checks.append(("recovers after a light change (re-brightens & re-converges)",
                   recovered and brightened))

    # (5) no windup / clamped: actuator, exposure, gain all stay in [0,1]
    inb = all(0.0 <= bank.ctl[r].a <= 1.0 for r in roles)
    st = bank.states()
    split_ok = all(0 <= st[r]["exposure"] <= 1 and 0 <= st[r]["gain"] <= 1 for r in roles)
    # drive a sensor that can NEVER hit target -> actuator must saturate at 1.0, not overflow
    starve = _SynthSensor("eyeR", scene=0.02, rng=rng)
    cc = ExposureController("eyeR", gain=0.8, start=0.5)
    for _ in range(80):
        cc.update(starve.frame(cc.a))
    checks.append(("clamped, no windup (actuator/expo/gain in [0,1]; starved -> saturates at %.2f)"
                   % cc.a, inb and split_ok and cc.a <= 1.0 + 1e-9 and cc.a > 0.9))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "AUTOEXPOSE OK, per-role targeting, converges, stable, keeps adjusting ✅"
              if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="continuous per-role auto-exposure")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    sys.exit(selftest())
