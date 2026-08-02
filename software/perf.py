"""perf.py — keep the FULL pipeline inside a frame budget: cameras + mesh + overlay.

bank_bringup proved the *capture* layer is cheap (4 cams ≈ 4% of 16 cores). The expensive part
is what runs on top of it: ORB/LK world tracking, mesh maintenance, people tracking, and the
avatar compositor. `live_rig`/`augment_rig` call those every frame in an unthrottled loop, so
when they are all switched on together the loop consumes whatever the machine will give it —
and on an Intel i9 that means fans, then thermal de-rating, then a slower loop anyway.

This module is the answer, in three parts:

  * **Profiler** — per-STAGE timing (`with prof.stage("mesh"): ...`). You cannot budget what you
    cannot attribute; a single "fps" number never says whether to cut features or resolution.
    Reports mean/p95 ms per stage and each stage's share of the frame.

  * **QualityController** — one LEVEL (full..min) mapping to the concrete knobs that actually
    cost time: ORB feature count, mesh update stride, and overlay render scale. It steps DOWN
    when frames overrun the budget or the CPU guard complains, and back UP only after sustained
    headroom, with hysteresis so it settles instead of oscillating. Degrading quality is what
    keeps the loop real-time; the alternative is a "high quality" setting that misses every
    deadline and looks worse.

  * **FrameBudget** — target fps -> ms per frame, and the honest verdict on whether the last
    window met it.

Everything is pure logic over injected timings, so `--selftest` proves the control behaviour
headless: no cameras, no load, deterministic.

    python3 perf.py --selftest
    python3 perf.py --demo        # show the controller reacting to a synthetic load ramp
"""
import argparse
import sys
import time
from collections import deque

import numpy as np

from bank_bringup import CpuGuard          # one source of truth for CPU accounting


# --------------------------------------------------------------------------
#  Quality levels: the knobs that actually cost time
# --------------------------------------------------------------------------
# orb_feat     -> world_mesh.WorldTracker feature count (ORB detect + match is the heaviest stage)
# mesh_stride  -> run the mesh update every Nth frame (pose still carried by IMU/last pose)
# render_scale -> avatar/overlay compositor works at this fraction of full res, upscaled at blit
# detect_stride-> people detector every Nth frame (tracker dead-reckons between detections)
QUALITY_LEVELS = [
    dict(name="full",   orb_feat=400, mesh_stride=1, render_scale=1.00, detect_stride=1),
    dict(name="high",   orb_feat=300, mesh_stride=1, render_scale=1.00, detect_stride=2),
    dict(name="medium", orb_feat=200, mesh_stride=2, render_scale=0.75, detect_stride=3),
    dict(name="low",    orb_feat=120, mesh_stride=3, render_scale=0.60, detect_stride=4),
    dict(name="min",    orb_feat=60,  mesh_stride=4, render_scale=0.50, detect_stride=6),
]


class FrameBudget:
    """Target frame rate expressed as a per-frame millisecond budget."""

    def __init__(self, target_fps=30.0):
        self.target_fps = float(target_fps)

    @property
    def budget_ms(self):
        return 1000.0 / self.target_fps

    def verdict(self, frame_ms):
        """'ok' under budget, 'over' past it, 'headroom' with room to raise quality."""
        if frame_ms > self.budget_ms:
            return "over"
        if frame_ms < self.budget_ms * 0.6:
            return "headroom"
        return "ok"


class Profiler:
    """Per-stage frame timing. Attribution, not just a frame rate.

        prof = Profiler()
        with prof.stage("sync"):   ...
        with prof.stage("mesh"):   ...
        prof.end_frame()
    """

    def __init__(self, window=120, clock=time.perf_counter):
        self.clock = clock
        self.window = window
        self.samples = {}                 # stage -> deque of ms
        self.frames = deque(maxlen=window)
        self._open = {}
        self._frame_accum = 0.0

    class _Stage:
        def __init__(self, prof, name):
            self.prof, self.name = prof, name

        def __enter__(self):
            self.t0 = self.prof.clock()
            return self

        def __exit__(self, *exc):
            dt = (self.prof.clock() - self.t0) * 1e3
            self.prof.add(self.name, dt)
            return False

    def stage(self, name):
        return self._Stage(self, name)

    def add(self, name, ms):
        """Record a stage duration directly (used by stage() and by the selftest)."""
        if name not in self.samples:
            self.samples[name] = deque(maxlen=self.window)
        self.samples[name].append(float(ms))
        self._frame_accum += float(ms)

    def end_frame(self):
        """Close the frame; returns its total measured ms."""
        total = self._frame_accum
        self.frames.append(total)
        self._frame_accum = 0.0
        return total

    def frame_ms(self, pct=50):
        return float(np.percentile(self.frames, pct)) if self.frames else 0.0

    def stage_ms(self, name, pct=50):
        s = self.samples.get(name)
        return float(np.percentile(s, pct)) if s else 0.0

    def report(self):
        """{stage: {mean, p95, share}} — share is the fraction of the median frame."""
        frame = self.frame_ms(50) or 1e-9
        out = {}
        for name, s in self.samples.items():
            a = np.asarray(s)
            out[name] = dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                             share=float(a.mean() / frame))
        return out

    def lines(self):
        rep = self.report()
        order = sorted(rep, key=lambda k: -rep[k]["mean"])
        out = ["  %-14s %8s %8s %7s" % ("stage", "mean ms", "p95 ms", "share")]
        for k in order:
            v = rep[k]
            out.append("  %-14s %8.2f %8.2f %6.0f%%" % (k, v["mean"], v["p95"], v["share"] * 100))
        out.append("  %-14s %8.2f %8.2f" % ("FRAME", self.frame_ms(50), self.frame_ms(95)))
        return out


class QualityController:
    """Holds the frame budget by moving one quality LEVEL up or down.

    Down-steps are fast (a missed deadline is already visible); up-steps need SUSTAINED headroom
    so the loop settles rather than oscillating between two levels. The CPU guard can force a
    down-step even when frame time looks fine — that catches the case where the loop is meeting
    its deadline only because it is burning every core to do it."""

    def __init__(self, budget=None, level=0, down_after=3, up_after=45, levels=None,
                 settle=30, min_gain=0.10):
        self.budget = budget or FrameBudget()
        self.levels = levels or QUALITY_LEVELS
        self.level = int(np.clip(level, 0, len(self.levels) - 1))
        self.down_after = down_after
        self.up_after = up_after
        self.settle = settle          # frames to wait before judging a down-step
        self.min_gain = min_gain      # fractional frame-time improvement that counts as "worked"
        self._over = 0
        self._head = 0
        self.changes = 0
        self._probe = None            # (level_before, frame_ms_before, frames_left)
        self.frozen = False           # True once degrading is proven not to help
        self.frozen_reason = ""

    @property
    def settings(self):
        return dict(self.levels[self.level])

    @property
    def name(self):
        return self.levels[self.level]["name"]

    def update(self, frame_ms, cpu_verdict="ok"):
        """Feed one frame's total ms (+ optional CpuGuard verdict). Returns the new settings.

        EFFECTIVENESS PROBE (added after the 2026-08-01 full-load run): degrading quality only
        helps when the frame time is dominated by the work these knobs control. On this rig it
        was not — 27% of the frame was USB wait and the CPU sat at 9% — so the controller walked
        straight to `min` and held it for 45 s while frame time barely moved (91 -> 77 ms). That
        is a pure quality loss. So after each down-step we WATCH: if the next `settle` frames do
        not improve by `min_gain`, the step is reverted and further degrading is frozen. Real CPU
        pressure still overrides the freeze, because that is the case degrading genuinely fixes.
        """
        cpu_pressed = cpu_verdict in ("throttle", "stop")
        v = "over" if cpu_pressed else self.budget.verdict(frame_ms)

        # judge the outstanding probe before considering another step
        if self._probe is not None:
            lvl_before, ms_before, left = self._probe
            left -= 1
            if left <= 0:
                gain = (ms_before - frame_ms) / ms_before if ms_before > 0 else 0.0
                if gain < self.min_gain and not cpu_pressed:
                    self.level = lvl_before          # it did not help — take the quality back
                    self.frozen = True
                    self.frozen_reason = ("degrading gained only %.0f%% (%.1f -> %.1f ms); "
                                          "frame time is not bound by these knobs"
                                          % (gain * 100, ms_before, frame_ms))
                self._probe = None
            else:
                self._probe = (lvl_before, ms_before, left)
            return self.settings

        if cpu_pressed and self.frozen:
            self.frozen = False                       # real CPU pressure re-enables degrading
            self.frozen_reason = ""

        if v == "over":
            self._head = 0
            self._over += 1
            if (self._over >= self.down_after and self.level < len(self.levels) - 1
                    and not self.frozen):
                self._probe = (self.level, frame_ms, self.settle)
                self.level += 1
                self.changes += 1
                self._over = 0
        elif v == "headroom":
            self._over = 0
            self._head += 1
            if self._head >= self.up_after and self.level > 0:
                self.level -= 1
                self.changes += 1
                self._head = 0
        else:
            self._over = 0
            self._head = 0
        return self.settings

    def line(self):
        s = self.settings
        base = ("quality[%s] orb=%d stride=%d render=%.2f detect=%d (budget %.1f ms)"
                % (s["name"], s["orb_feat"], s["mesh_stride"], s["render_scale"],
                   s["detect_stride"], self.budget.budget_ms))
        if self.frozen:
            base += "\n  degrading FROZEN: " + self.frozen_reason
        return base


class LoadManager:
    """Bundle: profiler + quality controller + CPU guard, driven once per frame.

    This is what live_rig/augment_rig hold. One call per frame keeps the three in step, so the
    quality level reflects BOTH the wall-clock deadline and the CPU cost of meeting it."""

    def __init__(self, target_fps=30.0, cpu_budget=0.50, level=0, guard=None, clock=time.perf_counter):
        self.prof = Profiler(clock=clock)
        self.quality = QualityController(FrameBudget(target_fps), level=level)
        self.guard = guard if guard is not None else CpuGuard(budget=cpu_budget)
        self._since_check = 0

    def stage(self, name):
        return self.prof.stage(name)

    def end_frame(self):
        """Close the frame and re-evaluate quality. Returns (frame_ms, settings)."""
        frame_ms = self.prof.end_frame()
        self._since_check += 1
        verdict = "ok"
        if self._since_check >= 15:            # sample CPU a few times a second, not every frame
            self._since_check = 0
            _, verdict = self.guard.update()
        return frame_ms, self.quality.update(frame_ms, verdict)

    def lines(self):
        return self.prof.lines() + ["  " + self.quality.line(), "  " + self.guard.line()]


# ==========================================================================
#  Self-test (no hardware, no load): deterministic control behaviour.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== perf self-test (injected timings, no hardware) ==")
    checks = []

    # (1) FrameBudget converts fps -> ms and classifies correctly
    b = FrameBudget(30.0)
    checks.append(("FrameBudget: 30 fps -> 33.3 ms; over/ok/headroom classified",
                   abs(b.budget_ms - 33.333) < 0.01 and b.verdict(50) == "over"
                   and b.verdict(30) == "ok" and b.verdict(10) == "headroom"))

    # (2) Profiler attributes time per stage and computes shares that sum to ~100%
    fake_t = {"now": 0.0}
    prof = Profiler(clock=lambda: fake_t["now"])
    for _ in range(20):
        for name, cost in (("sync", 0.002), ("mesh", 0.020), ("render", 0.008)):
            with prof.stage(name):
                fake_t["now"] += cost
        prof.end_frame()
    rep = prof.report()
    shares = sum(v["share"] for v in rep.values())
    checks.append(("Profiler: mesh is the heaviest stage (%.1f ms, %.0f%%), shares sum to 100%%"
                   % (rep["mesh"]["mean"], rep["mesh"]["share"] * 100),
                   abs(rep["mesh"]["mean"] - 20.0) < 0.01
                   and max(rep, key=lambda k: rep[k]["mean"]) == "mesh"
                   and abs(shares - 1.0) < 0.02 and abs(prof.frame_ms(50) - 30.0) < 0.01))

    # (3) QualityController steps DOWN under sustained overrun and clamps at the floor.
    #     The load here RESPONDS to the knobs (frame time scales with the ORB budget), which is
    #     the case degrading is meant for — so every step earns its keep and it walks to `min`.
    #     (A load that does NOT respond is covered by check 5b, where it must freeze instead.)
    q = QualityController(FrameBudget(30.0), down_after=3, settle=5, min_gain=0.10)
    start = q.name
    stepped = None
    for _ in range(300):
        ms = 120.0 * (q.settings["orb_feat"] / 400.0)   # 120 ms at full -> 18 ms at min
        q.update(ms)
        if stepped is None and q.level == 1:
            stepped = q.name
    checks.append(("QualityController: '%s' -> '%s' under responsive overrun, clamps at '%s'"
                   % (start, stepped, q.name),
                   start == "full" and stepped == "high" and q.name == "min"
                   and q.level == len(QUALITY_LEVELS) - 1 and not q.frozen))

    # (4) ...and back UP only after SUSTAINED headroom (hysteresis, no oscillation)
    q2 = QualityController(FrameBudget(30.0), level=2, down_after=3, up_after=10)
    for _ in range(9):
        q2.update(5.0)                       # headroom, but not yet sustained
    not_yet = q2.level == 2
    q2.update(5.0)                           # 10th -> step up
    stepped_up = q2.level == 1
    # a single slow frame must NOT immediately drop it back (that is the oscillation guard)
    q2.update(90.0)
    stable = q2.level == 1
    checks.append(("QualityController: up-step needs sustained headroom; one slow frame ≠ down-step",
                   not_yet and stepped_up and stable))

    # (5) CPU pressure forces a down-step even when the deadline IS being met
    q3 = QualityController(FrameBudget(30.0), down_after=2)
    for _ in range(2):
        q3.update(5.0, cpu_verdict="throttle")   # fast frames, but the CPU is pinned
    checks.append(("CPU 'throttle' outranks a met deadline (level %d)" % q3.level, q3.level == 1))

    # (5b) THE REGRESSION FROM THE REAL RIG (2026-08-01): an I/O-bound loop, 73 ms/frame against
    #      a 33 ms budget, but only 9% CPU — degrading did nothing (91 -> 77 ms over 45 s) yet the
    #      controller pinned itself at `min`. It must now revert and freeze instead.
    q_io = QualityController(FrameBudget(30.0), down_after=3, settle=10, min_gain=0.10)
    for _ in range(3):
        q_io.update(73.0, cpu_verdict="ok")           # over budget -> takes one step down
    stepped = q_io.level == 1
    for _ in range(10):
        q_io.update(71.0, cpu_verdict="ok")           # ...which buys ~3%, not enough
    checks.append(("I/O-bound loop: down-step reverted and degrading frozen (level %d, %s)"
                   % (q_io.level, "frozen" if q_io.frozen else "NOT frozen"),
                   stepped and q_io.level == 0 and q_io.frozen
                   and "not bound by these knobs" in q_io.frozen_reason))

    # (5c) ...but when degrading DOES pay off, it is kept and the controller keeps going.
    q_cpu = QualityController(FrameBudget(30.0), down_after=3, settle=10, min_gain=0.10)
    for _ in range(3):
        q_cpu.update(73.0)
    for _ in range(10):
        q_cpu.update(40.0)                            # a real 45% win -> keep the step
    kept = q_cpu.level == 1 and not q_cpu.frozen
    # and genuine CPU pressure re-enables degrading even after a freeze
    q_io.update(71.0, cpu_verdict="throttle")
    checks.append(("effective down-step is kept (level %d); CPU pressure unfreezes (%s)"
                   % (q_cpu.level, not q_io.frozen), kept and not q_io.frozen))

    # (6) knobs move monotonically the right way as quality drops
    orb = [l["orb_feat"] for l in QUALITY_LEVELS]
    stride = [l["mesh_stride"] for l in QUALITY_LEVELS]
    scale = [l["render_scale"] for l in QUALITY_LEVELS]
    checks.append(("levels monotone: ORB %s down, stride %s up, render %s down"
                   % (orb[0], stride[0], scale[0]),
                   all(a > b for a, b in zip(orb, orb[1:]))
                   and all(a <= b for a, b in zip(stride, stride[1:]))
                   and all(a >= b for a, b in zip(scale, scale[1:]))))

    # (7) LoadManager: a loop that starts over budget ends up inside it
    t = {"now": 0.0}
    lm = LoadManager(target_fps=30.0, clock=lambda: t["now"],
                     guard=CpuGuard(budget=0.5, cores=8,
                                    clock=lambda: t["now"], cpu_fn=lambda: t["now"] * 0.1))
    costs = {"mesh": 0.060}                  # 60 ms mesh: way over a 33 ms budget
    over_at_start = None
    for i in range(400):
        with lm.stage("mesh"):
            # cost scales with the ORB knob — the controller's lever actually does something
            t["now"] += costs["mesh"] * (lm.quality.settings["orb_feat"] / 400.0)
        frame_ms, _ = lm.end_frame()
        if i == 0:
            over_at_start = frame_ms > lm.quality.budget.budget_ms
    ended_ok = lm.prof.frame_ms(50) <= lm.quality.budget.budget_ms
    checks.append(("LoadManager: starts over budget (%s), degrades, ends inside it (%.1f ms <= %.1f)"
                   % (over_at_start, lm.prof.frame_ms(50), lm.quality.budget.budget_ms),
                   over_at_start and ended_ok and lm.quality.level > 0))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "PERF OK — stages attributed, budget held by adaptive quality ✅"
              if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


def demo():
    """Watch the controller react to a load ramp (no hardware)."""
    t = {"now": 0.0}
    lm = LoadManager(target_fps=30.0, clock=lambda: t["now"])
    print("frame  load_ms  frame_ms  level")
    for i in range(300):
        load = 0.010 + 0.055 * min(1.0, i / 100.0)      # ramp 10 ms -> 65 ms of raw work
        with lm.stage("mesh"):
            t["now"] += load * (lm.quality.settings["orb_feat"] / 400.0)
        frame_ms, _ = lm.end_frame()
        if i % 30 == 0:
            print("%5d %8.1f %9.1f  %s" % (i, load * 1e3, frame_ms, lm.quality.name))
    print("\n".join(lm.prof.lines()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="frame-budget profiler + adaptive quality")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        sys.exit(demo())
    sys.exit(selftest())
