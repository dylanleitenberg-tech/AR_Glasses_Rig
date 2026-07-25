"""rig_test.py — one-command hardware BRING-UP test for the 6-camera rig.

Run this right after plugging the rig into the Mac (and after `connect.py --auto/--identify`).
It opens the whole role-mapped bank through the synchronized capture layer and reports, per role
and overall, the numbers that decide whether the rig is fit to calibrate on:

  ENUMERATE   every mapped role actually delivers frames
  FPS         sustained frame rate per role (USB bandwidth / MJPG sanity for 6 streams)
  SYNC JITTER the spread of capture timestamps across the bank (must beat your budget, e.g. 3 ms)
  IR TEST     with the IR LEDs strobed on, the 2 pupil cams brighten (proves NoIR + illumination)
  FOCUS       per-role sharpness (variance-of-Laplacian) so you can set each lens before gluing

Everything is a plain function over frames + timestamps, so `--selftest` exercises the metrics
(fps, jitter, sharpness, IR jump) on synthetic frames with NO hardware. `--run` needs opencv,
the cameras, and a saved role map (connect.py).
"""
import argparse
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
#  Metrics (pure numpy — testable without cameras)
# --------------------------------------------------------------------------
def focus_score(frame):
    """Sharpness = variance of a discrete Laplacian. Higher = crisper focus. Grayscale-agnostic,
    numpy-only (no cv2) so it runs in the selftest and on real frames alike."""
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32)
    # 4-neighbour Laplacian via shifts (interior only)
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def fps_from_stamps(stamps):
    """Frames-per-second from a list of monotonic capture timestamps."""
    s = [t for t in stamps if t is not None]
    if len(s) < 2:
        return 0.0
    dt = np.diff(sorted(s))
    dt = dt[dt > 0]
    return float(1.0 / np.mean(dt)) if len(dt) else 0.0


def ir_jump(dark_brightness, lit_brightness):
    """Brightness increase when the IR LEDs turn on (0..1). Pupil cams should show a clear jump."""
    return float(lit_brightness) - float(dark_brightness)


def brightness(frame):
    return float(np.asarray(frame).mean()) / 255.0


class RoleReport:
    def __init__(self, role):
        self.role = role
        self.frames = 0
        self.fps = 0.0
        self.focus = 0.0
        self.brightness = 0.0
        self.ir_jump = None
        self.ok = False

    def line(self):
        ir = "" if self.ir_jump is None else "  IRΔ %+.2f" % self.ir_jump
        return ("  %-7s frames %4d  fps %5.1f  focus %8.1f  bright %.2f%s  [%s]"
                % (self.role, self.frames, self.fps, self.focus, self.brightness, ir,
                   "OK" if self.ok else "CHECK"))


def evaluate(per_role_stamps, per_role_last_frame, jitter_ms, budget_ms=3.0,
             ir_jumps=None, min_fps=30.0, min_focus=15.0):
    """Turn collected per-role timestamps/frames into a report. `per_role_stamps` {role:[ts..]},
    `per_role_last_frame` {role: frame}, `ir_jumps` {role: jump} for the pupil cams."""
    reports = {}
    for role, stamps in per_role_stamps.items():
        r = RoleReport(role)
        r.frames = len([t for t in stamps if t is not None])
        r.fps = fps_from_stamps(stamps)
        f = per_role_last_frame.get(role)
        if f is not None:
            r.focus = focus_score(f)
            r.brightness = brightness(f)
        if ir_jumps and role in ir_jumps:
            r.ir_jump = ir_jumps[role]
        pupil_ok = (r.ir_jump is None) or (r.ir_jump >= 0.05)
        r.ok = (r.frames > 0 and r.fps >= min_fps and r.focus >= min_focus and pupil_ok)
        reports[role] = r
    overall = (all(r.ok for r in reports.values()) and jitter_ms <= budget_ms)
    return reports, overall


# --------------------------------------------------------------------------
#  Hardware run (opencv + saved role map + sync layer)
# --------------------------------------------------------------------------
def run(seconds=3.0, budget_ms=3.0, strobe=None, fps=100, verbose=True):
    """Open the mapped bank, collect ~`seconds` of synchronized frames, report. `strobe` is an
    optional callable(bool) that turns the IR LEDs on/off for the IR test (via a serial/GPIO
    hook to the XIAO); if None the IR test is skipped."""
    from connect import load_map, validate_map
    from sync_capture import SyncBank
    role_index = load_map()
    if not role_index:
        print("no role map — run:  python3 connect.py --auto   (then --identify)")
        return 1
    probs = validate_map(role_index)
    if probs:
        print("role map invalid:", "; ".join(probs))
        return 1

    bank = SyncBank(role_index, fps=fps).start()
    per_stamps = {r: [] for r in role_index}
    per_last = {r: None for r in role_index}
    jitters = []
    try:
        # IR OFF baseline
        if strobe:
            strobe(False); time.sleep(0.2)
        dark = {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            fs = bank.sync_frame()
            jitters.append(fs.jitter_ms)
            for r in role_index:
                per_stamps[r].append(fs.ts.get(r))
                if fs.get(r) is not None:
                    per_last[r] = fs.get(r)
        for r in role_index:
            if per_last[r] is not None:
                dark[r] = brightness(per_last[r])
        # IR ON for the pupil test
        ir_jumps = None
        if strobe:
            strobe(True); time.sleep(0.3)
            fs = bank.sync_frame()
            ir_jumps = {}
            for r in ("pupilL", "pupilR"):
                if r in role_index and fs.get(r) is not None:
                    ir_jumps[r] = ir_jump(dark.get(r, 0.0), brightness(fs.get(r)))
            strobe(False)
    finally:
        bank.close()

    jitter_ms = float(np.median(jitters)) if jitters else 0.0
    reports, overall = evaluate(per_stamps, per_last, jitter_ms, budget_ms, ir_jumps)
    if verbose:
        print("== rig bring-up test ==")
        for r in reports.values():
            print(r.line())
        print("  sync jitter (median): %.2f ms  (budget %.1f ms)  [%s]"
              % (jitter_ms, budget_ms, "OK" if jitter_ms <= budget_ms else "TOO LOOSE"))
        print("  =>", "RIG READY ✅" if overall else "NOT READY — see CHECK rows ⚠️")
    return 0 if overall else 1


# ==========================================================================
#  Self-test (no hardware): the metrics behave on synthetic frames.
# ==========================================================================
def _checker(n=64, cell=4):
    """A sharply-focused, high-detail frame."""
    y, x = np.mgrid[0:n, 0:n]
    return (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)


def _defocused(n=64, rng=None):
    """A badly out-of-focus / lens-capped frame: near-uniform, no high-frequency detail — what a
    defocused lens actually delivers. Should fall UNDER the focus floor."""
    rng = rng or np.random.default_rng(0)
    return np.clip(128 + rng.normal(0, 0.5, (n, n)), 0, 255).astype(np.uint8)


def selftest(verbose=True):
    if verbose:
        print("== rig_test self-test (synthetic frames + timestamps, no hardware) ==")
    checks = []

    # (1) focus_score: a sharp checkerboard scores far higher than a defocused (near-uniform) frame
    sharp = focus_score(_checker())
    blurry = focus_score(_defocused())
    checks.append(("focus_score ranks sharp >> defocused (%.0f vs %.1f)" % (sharp, blurry),
                   sharp > blurry * 100 and blurry < 15))

    # (2) fps_from_stamps recovers a known rate
    stamps = list(np.arange(0, 1.0, 1 / 90.0))     # 90 fps
    checks.append(("fps_from_stamps recovers 90 fps (%.1f)" % fps_from_stamps(stamps),
                   abs(fps_from_stamps(stamps) - 90) < 1.0))

    # (3) ir_jump positive only for the strobed pupil cam
    checks.append(("ir_jump: pupil brightens (+), eye ~flat",
                   ir_jump(0.1, 0.32) > 0.05 and abs(ir_jump(0.1, 0.11)) < 0.05))

    # (4) evaluate(): a good bank passes; a blurry/low-fps role is flagged; loose jitter fails
    roles = ("worldL", "worldR", "eyeL", "eyeR", "pupilL", "pupilR")
    good_stamps = {r: list(np.arange(0, 1.0, 1 / 90.0)) for r in roles}
    good_last = {r: _checker() for r in roles}
    irj = {"pupilL": 0.2, "pupilR": 0.22}
    reps, overall = evaluate(good_stamps, good_last, jitter_ms=1.2, ir_jumps=irj)
    good_ok = overall and all(reps[r].ok for r in roles)

    bad_last = dict(good_last); bad_last["eyeR"] = _defocused()          # eyeR out of focus
    _, overall_bad = evaluate(good_stamps, bad_last, jitter_ms=1.2, ir_jumps=irj)
    _, overall_jit = evaluate(good_stamps, good_last, jitter_ms=9.0, ir_jumps=irj)   # loose sync
    checks.append(("evaluate: passes a good bank, fails on blur (eyeR) and on loose jitter",
                   good_ok and (not overall_bad) and (not overall_jit)))

    # (5) a dark pupil cam that DOESN'T brighten under IR is flagged (dead LED / IR-cut lens)
    _, overall_noir = evaluate(good_stamps, good_last, jitter_ms=1.2,
                               ir_jumps={"pupilL": 0.2, "pupilR": 0.0})
    checks.append(("evaluate: pupil cam with no IR jump is flagged", not overall_noir))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "RIG_TEST OK — enumerate/fps/jitter/IR/focus metrics validated ✅"
              if ok else "PROBLEM ⚠️")
        print("  on hardware:  python3 rig_test.py --run   (after connect.py maps the cameras)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="6-camera rig bring-up test")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="open real cameras and report")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--budget-ms", type=float, default=3.0, help="max acceptable sync jitter")
    args = ap.parse_args()
    if args.run:
        sys.exit(run(seconds=args.seconds, budget_ms=args.budget_ms))
    sys.exit(selftest())
