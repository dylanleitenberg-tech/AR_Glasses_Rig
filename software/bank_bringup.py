"""bank_bringup.py — bring up WHATEVER cameras are on the hub, as one bank, without cooking the Mac.

The existing bring-up chain (connect -> rig_test -> snapshot -> live_rig) assumes the FINISHED
rig: a complete 6-role map, and IR LEDs available to tell the pupil cams from the eye-corner
cams. Neither holds at the start of assembly — you plug cameras into the powered hub one or two
at a time, with no illumination built yet. This module is the step BEFORE that chain:

    * accepts a PARTIAL bank (1..8 cameras) instead of demanding all six;
    * needs NO IR LEDs (roles are provisional and clearly labelled as such);
    * reports what each camera actually NEGOTIATED (format/resolution/fps), not what was asked
      for — a camera silently handing back 1280x800 when you asked for 640x480 is 4x the decode
      cost, and that is exactly the surprise that melts a laptop;
    * runs the bank under an explicit CPU BUDGET with automatic back-off.

WHY THE CPU GUARD EXISTS
    Every frame of every camera is an MJPEG decode on the CPU. sync_capture's barrier means the
    camera threads only grab when the controller asks, so the LOOP RATE is the master throttle —
    but rig_test/live_rig call sync_frame() in an unthrottled while-loop, i.e. "as fast as the
    machine allows". On an 8-core Intel i9 with six streams that is a fan-pinning, thermally
    throttling load. So here:
      - the loop runs at a TARGET RATE (default 20 Hz) with real sleep between sets;
      - CpuGuard measures this process's actual CPU seconds per wall second (stdlib `resource`,
        no psutil) as a fraction of the whole machine, and backs the rate off when it exceeds
        the budget, then STOPS if backing off to the floor still can't get under it;
      - ThermalProbe watches macOS's own throttle signal (`pmset -g therm` CPU_Speed_Limit);
        anything under 100 means the OS is already de-rating the CPU and we back off too;
      - OpenCV's internal thread pool is capped (cv2.setNumThreads) so decode can't fan out
        across all 16 threads.

    The guard logic, the rate controller, the thermal parse and the role assignment are pure
    functions over injected inputs, so `--selftest` proves all of it headless — including a full
    loop over FakeCapture cameras — with no hardware and no load on the machine.

USAGE
    python3 bank_bringup.py --selftest        # headless; no cameras, no load
    python3 bank_bringup.py --scan            # what is plugged in: index, colour/mono, real size
    python3 bank_bringup.py --run             # bring the whole bank up, throttled + guarded
    python3 bank_bringup.py --run --view      # + a downscaled live mosaic (costs CPU; off by default)
    python3 bank_bringup.py --run --save-map  # write a PROVISIONAL data/rig_cameras.json

The provisional map is honest about what it cannot know: colour/1280 cameras become worldL/R,
mono cameras become eyeL/eyeR then pupilL/pupilR **in index order**. Eye-vs-pupil and L-vs-R are
NOT determined — fix them with `connect.py --identify` once the cams are on the carrier, or with
the IR-strobe test once the LEDs exist.
"""
import argparse
import os
import re
import resource
import subprocess
import sys
import time

import numpy as np

from connect import classify_frame, classify_bank, save_map, CORE_ROLES
from rig_test import focus_score, brightness, fps_from_stamps


# --------------------------------------------------------------------------
#  CPU accounting (stdlib only — no psutil dependency)
# --------------------------------------------------------------------------
def cpu_seconds():
    """Total CPU seconds burned by THIS process (all threads, user + system).

    resource.RUSAGE_SELF covers every thread in the process, so the cv2 decode threads and the
    per-camera grab threads are all included — which is precisely the load we are budgeting."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def machine_cores():
    return os.cpu_count() or 1


class CpuGuard:
    """Measures process CPU as a fraction of the WHOLE machine and rules on it.

    load = ΔCPU-seconds / (Δwall-seconds × cores). 1.0 means every core pinned by this process.

    Verdicts:
        "ok"       — under budget, carry on
        "throttle" — over budget, the caller should reduce the loop rate
        "stop"     — over the HARD ceiling continuously for `grace` seconds; the caller should
                     shut the bank down rather than keep cooking

    `clock` and `cpu_fn` are injected so the selftest can drive it deterministically.
    """

    def __init__(self, budget=0.50, hard=None, cores=None, grace=5.0,
                 clock=time.monotonic, cpu_fn=cpu_seconds):
        self.budget = float(budget)
        # hard ceiling: well over budget but never so high that "stop" is unreachable
        self.hard = float(hard) if hard is not None else min(0.90, self.budget * 1.6)
        self.cores = cores or machine_cores()
        self.grace = float(grace)
        self.clock = clock
        self.cpu_fn = cpu_fn
        self._t = clock()
        self._c = cpu_fn()
        self.load = 0.0
        self.peak = 0.0
        self._over_since = None

    def update(self):
        """Sample once. Returns (load_fraction, verdict). Call at most a few times a second."""
        t, c = self.clock(), self.cpu_fn()
        dt, dc = t - self._t, c - self._c
        if dt <= 0:
            return self.load, "ok"
        self._t, self._c = t, c
        self.load = dc / (dt * self.cores)
        self.peak = max(self.peak, self.load)
        if self.load > self.hard:
            if self._over_since is None:
                self._over_since = t
            elif t - self._over_since >= self.grace:
                return self.load, "stop"
            return self.load, "throttle"
        self._over_since = None
        return self.load, ("throttle" if self.load > self.budget else "ok")

    def line(self):
        return "cpu %4.0f%% of %d cores (peak %.0f%%, budget %.0f%%)" % (
            self.load * 100, self.cores, self.peak * 100, self.budget * 100)


class RateController:
    """Target loop rate with multiplicative back-off and slow recovery.

    Back-off is immediate (halve) because thermal damage is cumulative; recovery is gradual
    (+25%) so the loop doesn't oscillate between hammering and idling."""

    def __init__(self, target=20.0, floor=2.0):
        self.target = float(target)
        self.floor = float(floor)
        self.rate = float(target)

    def back_off(self):
        self.rate = max(self.floor, self.rate * 0.5)
        return self.rate

    def recover(self):
        self.rate = min(self.target, self.rate * 1.25)
        return self.rate

    @property
    def at_floor(self):
        return self.rate <= self.floor + 1e-9

    def sleep_for(self, elapsed):
        """Seconds to sleep after a loop body that took `elapsed` seconds (never negative)."""
        return max(0.0, (1.0 / self.rate) - elapsed)


# --------------------------------------------------------------------------
#  macOS thermal signal
# --------------------------------------------------------------------------
def parse_thermal(text):
    """Pull CPU_Speed_Limit / CPU_Scheduler_Limit out of `pmset -g therm` output.

    100 = no de-rating. Anything lower means macOS is ALREADY throttling the CPU, which is the
    machine telling us it is too hot — a stronger signal than our own load number."""
    out = {}
    for key in ("CPU_Speed_Limit", "CPU_Scheduler_Limit", "CPU_Available_CPUs"):
        m = re.search(key + r"\s*=\s*(\d+)", text)
        if m:
            out[key] = int(m.group(1))
    return out


def thermal_state():
    """Current thermal de-rating, or {} if unavailable (non-Mac, sandbox, etc.)."""
    try:
        txt = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                             timeout=4.0).stdout
    except Exception:
        return {}
    return parse_thermal(txt)


def thermally_throttled(state):
    """True when macOS reports the CPU de-rated below full speed."""
    return state.get("CPU_Speed_Limit", 100) < 100


# --------------------------------------------------------------------------
#  Partial-bank role assignment (no IR LEDs required)
# --------------------------------------------------------------------------
# Which mono roles exist, in fill order, per eye-side layout.
#   "binocular" = the finished 6-cam CORE: an eye-corner + a pupil cam for EACH eye.
#   "one_eye"   = the 4-cam interim build: both inward cams serve the RIGHT eye. Right, not
#                 left, because rig.py DISPLAY_EYE = 1 and build_pupil_camera() models the
#                 right eye as canonical — so a one-eye rig on the right needs no sim changes,
#                 while a left-eye one would have to mirror everything.
MONO_ORDER = {"binocular": ("eyeL", "eyeR", "pupilL", "pupilR"),
              "one_eye": ("eyeR", "pupilR")}

# NATIVE sensor modes (2026-08-01, measured on the real modules). BOTH sensors are 16:10 —
# SyncBank derives height as int(res * 3 // 4), i.e. 4:3, so its world request lands on
# 1280x960, a mode NEITHER sensor offers. Asking for a non-existent mode makes the driver fall
# back to whatever it can do, which is how four streams stalled the bus. These are the real ones.
NATIVE_MODES = {"worldL": (1920, 1200), "worldR": (1920, 1200),      # ELP AR0234
                "eyeL": (1280, 800), "eyeR": (1280, 800),            # OV9281
                "pupilL": (1280, 800), "pupilR": (1280, 800),
                "eye2L": (1280, 800), "eye2R": (1280, 800)}


def assign_roles(descs, skip=(), layout="binocular"):
    """Provisional {role: index} from however many cameras are present, + honest notes.

    Colour/1280 cameras -> worldL, worldR. Mono cameras fill MONO_ORDER[layout] in index order.
    This CANNOT distinguish eye-corner from pupil (that needs the IR strobe) nor left from right
    (that needs a human) — the notes say so, and nothing downstream should treat the map as final
    until connect.py --identify has confirmed it."""
    if layout not in MONO_ORDER:
        raise ValueError("layout must be one of %s, got %r" % (sorted(MONO_ORDER), layout))
    descs = {i: d for i, d in descs.items() if i not in set(skip)}
    groups = classify_bank(descs)
    world = groups["world"] + groups["unknown"]          # a colour cam at odd res is still world-ish
    mono = groups["eye_or_pupil"]
    mono_roles = MONO_ORDER[layout]
    role_index, notes = {}, []

    for role, idx in zip(("worldL", "worldR"), world):
        role_index[role] = idx
    if len(world) > 2:
        notes.append("%d colour cams found but only 2 world roles exist — extras ignored: %s"
                     % (len(world), world[2:]))
    elif len(world) < 2:
        notes.append("%d/2 world (colour) cams present" % len(world))

    for role, idx in zip(mono_roles, mono):
        role_index[role] = idx
    if len(mono) > len(mono_roles):
        notes.append("%d mono cams found but the %s layout has only %d mono roles — extras "
                     "ignored: %s" % (len(mono), layout, len(mono_roles), mono[len(mono_roles):]))
    elif len(mono) < len(mono_roles):
        notes.append("%d/%d mono (OV9281) cams present" % (len(mono), len(mono_roles)))

    if mono:
        notes.append("eye-vs-pupil is PROVISIONAL (index order) — no IR LEDs to run the strobe "
                     "A/B test; confirm with connect.py --identify")
    if layout == "one_eye":
        notes.append("ONE-EYE layout: both inward cams serve the RIGHT eye (rig.py DISPLAY_EYE=1). "
                     "The 8-feature calibration contract needs eyeL too — bring-up, focus, "
                     "framing and sync are all valid now; per-user calibration is not.")
    else:
        missing = [r for r in CORE_ROLES if r not in role_index]
        if missing:
            notes.append("not yet a full rig — missing: %s" % ", ".join(missing))
    return role_index, notes


VALID_ROLES = CORE_ROLES + ("eye2L", "eye2R")


def parse_roles(specs):
    """Parse ['eyeL=4', 'worldR=3'] into {role: index}, the manual override for when
    auto-classification is wrong or ambiguous.

    Colour-vs-mono classification is a heuristic over one frame; the wearer looking at the live
    view is ground truth. This lets that judgement win without editing any file."""
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("role spec %r must look like eyeL=4" % spec)
        role, _, idx = spec.partition("=")
        role, idx = role.strip(), idx.strip()
        if role not in VALID_ROLES:
            raise ValueError("unknown role %r (valid: %s)" % (role, ", ".join(VALID_ROLES)))
        try:
            out[role] = int(idx)
        except ValueError:
            raise ValueError("role %s needs an integer camera index, got %r" % (role, idx))
    dupes = [i for i in set(out.values()) if list(out.values()).count(i) > 1]
    if dupes:
        raise ValueError("camera index %s assigned to more than one role" % dupes)
    return out


def parse_res(spec):
    """Parse a --res value into (width, height), or None.

    Accepts '640' (SyncBank's 4:3 convention, -> 640x480) or an explicit '1280x800'. Explicit
    WxH matters because these sensors are 16:10 and the modes the driver will hand you MJPEG on
    are not predictable from width alone — you have to be able to ask for an exact one."""
    if spec is None:
        return None
    s = str(spec).lower().replace(" ", "")
    if "x" in s:
        w, _, h = s.partition("x")
        try:
            return (int(w), int(h))
        except ValueError:
            raise ValueError("--res %r must look like 1280x800 or 640" % spec)
    try:
        w = int(s)
    except ValueError:
        raise ValueError("--res %r must look like 1280x800 or 640" % spec)
    return (w, int(w * 3 // 4))


def device_names():
    """Best-effort UVC device names in AVFoundation order (which cv2 indices usually follow).

    Purely informational — used to label the scan table so you can tell the built-in FaceTime
    camera from an InnoMaker/ELP module without opening each one."""
    try:
        txt = subprocess.run(["system_profiler", "SPCameraDataType"],
                             capture_output=True, text=True, timeout=15.0).stdout
    except Exception:
        return []
    names = []
    for line in txt.splitlines():
        s = line.strip()
        if s.endswith(":") and not s.startswith("Camera") and "ID" not in s:
            names.append(s[:-1])
    return names


# --------------------------------------------------------------------------
#  Hardware: scan
# --------------------------------------------------------------------------
def _open_probe(cv2, index, width, height):
    """Open one index, request MJPG + a size, return (cap, negotiated dict) or (None, None)."""
    cap = cv2.VideoCapture(index, getattr(cv2, "CAP_AVFOUNDATION", 0))
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None, None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    # RETRY (2026-08-01): a single read() misses cameras that need a moment to start streaming —
    # index 5 enumerated but returned nothing on the first probe. Give each device several
    # attempts with a short settle before declaring it dead.
    ok, frame = False, None
    for attempt in range(8):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.15)
    if not ok or frame is None:
        cap.release()
        return None, None
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # AVFoundation often reports 0 here; keep only printable chars so a null/garbage code shows
    # as "?" instead of printing invisible bytes that look like a blank column.
    tag = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    tag = "".join(c for c in tag if c.isprintable() and not c.isspace())
    got = dict(
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1],
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0],
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        fourcc=tag or "?",
    )
    return cap, (frame, got)


def scan(max_index=8, width=1280, height=800, skip=(), layout="binocular", verbose=True):
    """Probe every index and print what is actually attached. Opens each camera briefly, one at
    a time — near-zero sustained CPU."""
    import cv2
    cv2.setNumThreads(2)
    names = device_names()
    descs, rows = {}, []
    for i in range(max_index):
        if i in set(skip):
            continue
        cap, got = _open_probe(cv2, i, width, height)
        if cap is None:
            continue
        frame, neg = got
        cap.release()
        d = classify_frame(frame)
        descs[i] = d
        rows.append((i, d, neg, focus_score(frame)))

    if verbose:
        print("== camera scan (indices 0..%d) ==" % (max_index - 1))
        if not rows:
            print("  no cameras responded. Check: hub 12 V brick in FIRST, then cams, then hub->Mac.")
        for i, d, neg, foc in rows:
            # chroma is printed because it is the ONLY thing separating colour from mono, and the
            # 0.02 threshold is a guess. On a DARK frame a mono sensor's noise clears it easily,
            # so trust RESOLUTION (AR0234 1920x1200 vs OV9281 1280x800) and your eyes instead.
            print("  [%d] %-5s chroma %.4f  %dx%d @%.0f %s  focus %8.1f  bright %.2f"
                  % (i, "color" if d["is_color"] else "mono", d["chroma"],
                     neg["w"], neg["h"], neg["fps"], neg["fourcc"], foc, d["brightness"]))
        # Names are reported as an UNORDERED set: system_profiler's order is not cv2's index
        # order (observed shuffling between two consecutive scans), so pairing them per-row
        # actively misleads. The USB tree is the reliable inventory.
        if names:
            print("  macOS reports these cameras (order does NOT map to the indices above):")
            print("    " + " | ".join(names))
        role_index, notes = assign_roles(descs, skip, layout)
        print("  provisional roles (%s):" % layout, role_index or "(none)")
        for n in notes:
            print("    note:", n)
        st = thermal_state()
        if st:
            print("  thermal: CPU_Speed_Limit=%d%%" % st.get("CPU_Speed_Limit", 100))
    return descs


# --------------------------------------------------------------------------
#  The guarded loop (backend-agnostic: real SyncBank or FakeCapture bank)
# --------------------------------------------------------------------------
class BankStats:
    """Rolling per-role health across the run."""

    def __init__(self, roles):
        self.roles = list(roles)
        self.stamps = {r: [] for r in roles}
        self.last = {r: None for r in roles}
        self.misses = {r: 0 for r in roles}
        self.jitters = []
        self.sets = 0

    def add(self, fs):
        self.sets += 1
        self.jitters.append(fs.jitter_ms)
        for r in self.roles:
            t = fs.ts.get(r)
            self.stamps[r].append(t)
            f = fs.get(r)
            if f is None:
                self.misses[r] += 1
            else:
                self.last[r] = f

    def report(self):
        out = {}
        for r in self.roles:
            f = self.last[r]
            out[r] = dict(fps=fps_from_stamps(self.stamps[r]),
                          misses=self.misses[r],
                          focus=focus_score(f) if f is not None else 0.0,
                          bright=brightness(f) if f is not None else 0.0,
                          shape=(tuple(f.shape[:2]) if f is not None else None))
        return out

    @property
    def jitter_ms(self):
        return float(np.median(self.jitters)) if self.jitters else 0.0


def run_loop(bank, roles, seconds=10.0, guard=None, rate=None, on_frame=None, warm=True,
             clock=time.monotonic, sleep=time.sleep, verbose=True):
    """Drive an already-started bank under the CPU budget. Backend-agnostic so the selftest can
    run the identical code path over FakeCapture cameras.

    `warm` controls sync_frame's flush grab. warm=True costs TWO grabs per set (one to drain a
    stale driver buffer, one aligned); warm=False costs one, so the set rate can roughly double.
    Since the barrier only lets a camera grab when the controller crosses it, every tick is
    already fresh here — but the flush is kept as the default because it is what the rest of the
    pipeline (snapshot.py, rig_test.py) assumes.

    Returns (BankStats, reason) where reason is "done" | "cpu-stop" | "camera-died"."""
    guard = guard or CpuGuard()
    rate = rate or RateController()
    stats = BankStats(roles)
    t_end = clock() + seconds
    next_check = clock() + 1.0
    next_therm = clock() + 5.0
    reason = "done"
    while clock() < t_end:
        t0 = clock()
        try:
            fs = bank.sync_frame(warm=warm)
        except RuntimeError as e:                     # a camera thread died -> barrier broke
            if verbose:
                print("  camera dropped out: %s" % e)
            reason = "camera-died"
            break
        stats.add(fs)
        if on_frame is not None:
            on_frame(fs)
        now = clock()
        if now >= next_check:
            next_check = now + 1.0
            load, verdict = guard.update()
            if verdict == "stop":
                if verbose:
                    print("  CPU over hard ceiling (%.0f%%) even at the rate floor — stopping."
                          % (load * 100))
                reason = "cpu-stop"
                break
            if verdict == "throttle":
                r = rate.back_off()
                if verbose:
                    print("  over budget (%.0f%%) -> loop rate %.1f Hz" % (load * 100, r))
            else:
                rate.recover()
        if now >= next_therm:
            next_therm = now + 5.0
            if thermally_throttled(thermal_state()):
                r = rate.back_off()
                if verbose:
                    print("  macOS is thermally de-rating the CPU -> loop rate %.1f Hz" % r)
        sleep(rate.sleep_for(clock() - t0))
    return stats, reason


def dead_camera_advice(dead, shapes):
    """What to actually do when a camera delivers nothing. -> list of lines.

    BUS STARVATION AND ONE BAD CABLE LOOK IDENTICAL IN THE RESULT TABLE. Both give a dead camera
    and, because sync_capture's barrier waits for everyone, a whole bank crawling at ~1 fps. This
    module used to blame the bus unconditionally and tell you to cut resolution.

    They are told apart by the frame size that was actually DELIVERED. Starvation is a bandwidth
    problem, so it cannot be the cause when the survivors are already serving small frames — a bus
    short of bandwidth does not carry three streams flawlessly and drop the fourth entirely.

    MEASURED 2026-08-04, which is why this exists: eyeL dead with all four requested at 640x480 and
    the whole bank at 1 fps. Dropping eyeL took the other three to 38 fps, 0 misses, 7.13 ms
    jitter — the bus had capacity to spare — and eyeL then delivered 0/30 frames opened ALONE at
    every mode. It was the cable. The old advice would have had us cut a resolution already cut.
    """
    shapes = [s for s in shapes if s]
    small = [s for s in shapes if max(s[:2]) <= 800]
    if len(dead) == 1 and shapes and len(small) == len(shapes):
        return [
            "ONE camera is dead while the rest already serve small frames, so this is",
            "NOT bandwidth — a bus cannot starve one stream and carry the others.",
            "The barrier makes it look bank-wide: every camera waits for %s." % dead[0],
            "ISOLATE IT: re-run with --skip <its index>. If the rest jump to ~38 fps,",
            "the bus is fine and the fault is that camera's cable, connector or hub",
            "port. Then open it ALONE — an intermittent lead delivers a frame to",
            "--scan and nothing a minute later.",
        ]
    return [
        "%d camera(s) delivered nothing. On this rig that is usually the shared" % len(dead),
        "USB 2.0 bus: it carries THREE native streams, and a 4th starves while",
        "sync_capture's barrier stalls the whole bank.",
        "Re-run with --res 640 (all four serve that over MJPEG), or use --ramp to",
        "find how many streams this bus actually carries.",
    ]


def run(seconds=15.0, max_index=8, target_rate=20.0, budget=0.50, res=None, skip=(),
        layout="binocular", roles=None, view=False, save=False, cv_threads=2, warm=True,
        native=False, verbose=True):
    """Full hardware bring-up of whatever is plugged in.

    `roles` (from --roles) overrides auto-classification entirely — use it once you have
    eyeballed each index and know which camera is which."""
    import cv2
    from sync_capture import SyncBank, open_cv2_capture
    cv2.setNumThreads(cv_threads)                     # keep decode off all 16 threads

    if roles:
        role_index, notes = dict(roles), ["roles set MANUALLY from --roles (no classification)"]
    else:
        descs = scan(max_index, skip=skip, layout=layout, verbose=False)
        role_index, notes = assign_roles(descs, skip, layout)
    if not role_index:
        print("no cameras found. Order: hub 12 V brick to the wall FIRST, then the cameras into")
        print("the hub, then the hub into the Mac (rail up before devices = clean enumeration).")
        return 1

    if verbose:
        print("== bank bring-up: %d camera(s), %s layout ==" % (len(role_index), layout))
        for r, i in sorted(role_index.items()):
            d = None if roles else descs.get(i)
            print("  %-7s index %d%s" % (r, i, "" if d is None else
                                         "  %-5s %dx%d" % ("color" if d["is_color"] else "mono",
                                                           d["w"], d["h"])))
        for n in notes:
            print("  note:", n)

    # The scan above opened and released every device; macOS UVC can refuse an immediate reopen,
    # so let the devices settle before the bank claims them for real.
    time.sleep(0.4)

    # Explicit capture size. --native asks each sensor for its REAL mode (16:10); --res forces one
    # width on everything (4:3); otherwise use SyncBank.ROLE_MODE verbatim.
    #
    # DO NOT re-derive the height from ROLE_RES here. This branch used to compute
    # `h = int(ROLE_RES[role] * 3 // 4)`, which asks the 16:10 world cams for 1280x960 — a mode
    # NEITHER sensor has. The driver falls back to an uncompressed mode, four streams swamp the
    # shared USB 2.0 bus, and sync_capture's barrier stalls the whole bank: measured 2026-08-02 at
    # ~1 fps with 118 ms jitter and TWO of four cameras missing every single grab, while still
    # printing "BANK UP". ROLE_MODE is the single source of truth for capture modes and already
    # encodes the 640x480 that all four cameras actually serve over MJPEG (~40 fps, 6.7 ms jitter).
    def opener(role, idx):
        if native:
            w, h = NATIVE_MODES[role]
        elif res:
            w, h = res if isinstance(res, tuple) else (res, int(res * 3 // 4))
        else:
            w, h = SyncBank.ROLE_MODE[role]
        return open_cv2_capture(idx, w, h, 100)

    bank = SyncBank(role_index, fps=100, opener=opener).start()
    guard = CpuGuard(budget=budget)
    rate = RateController(target=target_rate)
    win = "bank bring-up" if view else None
    if win:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    drawn = {"n": 0}

    def draw(fs):
        drawn["n"] += 1
        if drawn["n"] % 3:                             # draw every 3rd set — the window is not free
            return
        tiles = []
        for r in sorted(fs.frames):
            f = fs.get(r)
            if f is None:
                continue
            g = f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            g = cv2.resize(g, (320, 240))
            cv2.putText(g, r, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            tiles.append(g)
        if not tiles:
            return
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imshow(win, np.vstack(rows))
        cv2.waitKey(1)

    try:
        stats, reason = run_loop(bank, list(role_index), seconds=seconds, guard=guard,
                                 rate=rate, on_frame=(draw if win else None), warm=warm,
                                 verbose=verbose)
    finally:
        bank.close()
        if win:
            cv2.destroyAllWindows()

    rep = stats.report()
    # A role that delivered NOTHING is a failed bring-up, even when the loop ran its full clock.
    # `reason == "done"` only means the timer expired; it says nothing about whether every camera
    # actually produced frames. Reporting "BANK UP" while half the bank sat at 100% misses is how
    # a starved USB bus gets mistaken for a healthy rig (measured 2026-08-02: eyeR and worldL
    # missed 8 of 8 grabs and the run still printed BANK UP).
    dead = sorted(r for r in rep
                  if rep[r]["shape"] is None or (stats.sets and rep[r]["misses"] >= stats.sets))
    if verbose:
        grabs = 2 if warm else 1
        print("\n== result (%d sets, %.1f s, %d grab%s per set) =="
              % (stats.sets, seconds, grabs, "" if grabs == 1 else "s"))
        for r in sorted(rep):
            v = rep[r]
            print("  %-7s fps %5.1f  misses %3d  %s  focus %8.1f  bright %.2f"
                  % (r, v["fps"], v["misses"], v["shape"], v["focus"], v["bright"]))
        print("  sync jitter (median): %.2f ms" % stats.jitter_ms)
        print("  NOTE: 'fps' above is the SET rate. Each set costs %d grab(s) per camera, so the"
              % grabs)
        print("        cameras are actually delivering ~%.0f fps each." % (stats.sets / seconds * grabs))
        print("  " + guard.line() + "   final loop rate %.1f Hz" % rate.rate)
        st = thermal_state()
        if st:
            print("  thermal: CPU_Speed_Limit=%d%%%s" % (
                st.get("CPU_Speed_Limit", 100),
                "  <-- macOS de-rated the CPU during the run" if thermally_throttled(st) else ""))
        if dead:
            print("  => NO FRAMES FROM: %s ❌" % ", ".join(dead))
            for line in dead_camera_advice(dead, [getattr(r, "shape", None)
                                                  for r in rep.values()]):
                print("     " + line)
        else:
            print("  =>", {"done": "BANK UP ✅", "cpu-stop": "STOPPED ON CPU BUDGET ⚠️",
                           "camera-died": "A CAMERA DROPPED OUT ⚠️"}[reason])
        if reason == "cpu-stop":
            print("  levers: --rate lower, --res 640 (force small frames), fewer cameras,")
            print("          --cv-threads 1, and check `--scan` for a camera that negotiated a")
            print("          bigger frame than you asked for.")

    if save:
        path = save_map(role_index, meta={"method": "bank_bringup (PROVISIONAL)",
                                          "notes": notes})
        print("  wrote PROVISIONAL map ->", path)
        print("  confirm eye-vs-pupil + L/R with: python3 connect.py --identify")
    return 0 if (reason == "done" and not dead) else 1


# --------------------------------------------------------------------------
#  Bandwidth ramp: how many of these cameras can this bus actually carry?
# --------------------------------------------------------------------------
def ramp(role_index, seconds=4.0, res=None, opener=None, cv_threads=2, verbose=True):
    """Bring cameras up ONE AT A TIME, cumulatively, measuring the bank at each step.

    WHY: a UVC camera reserves isochronous USB bandwidth when it starts streaming. On a shared
    bus the first few succeed and a later one silently gets nothing — it opens, but never
    delivers a frame. Because sync_capture's barrier waits for every camera, ONE starved camera
    throttles the whole bank, so the aggregate number hides which camera actually broke.

    Stepping 1..N and reporting the WORST per-camera fps at each step shows exactly where the
    bus falls over, which is the number the rig design depends on.

    Returns a list of dicts, one per step."""
    from sync_capture import SyncBank
    _opener = opener
    if _opener is None:
        import cv2
        from sync_capture import open_cv2_capture
        cv2.setNumThreads(cv_threads)

        def _opener(role, idx):
            w = res or SyncBank.ROLE_RES[role]
            return open_cv2_capture(idx, w, int(w * 3 // 4), 100)

    order = list(role_index)
    rows = []
    if verbose:
        print("== bandwidth ramp: adding one camera at a time (%.0f s each) ==" % seconds)
        print("  n  roles                              worst fps  total fps  misses  jitter")
    for n in range(1, len(order) + 1):
        subset = {r: role_index[r] for r in order[:n]}
        try:
            bank = SyncBank(subset, fps=100, opener=_opener).start()
        except Exception as e:
            rows.append(dict(n=n, roles=list(subset), error=str(e)))
            if verbose:
                print("  %d  %-34s OPEN FAILED: %s" % (n, ",".join(subset), e))
            continue
        try:
            stats, reason = run_loop(bank, list(subset), seconds=seconds,
                                     guard=CpuGuard(budget=0.95),
                                     rate=RateController(target=60.0), verbose=False)
        finally:
            bank.close()
        rep = stats.report()
        fps = {r: rep[r]["fps"] for r in subset}
        misses = sum(rep[r]["misses"] for r in subset)
        row = dict(n=n, roles=list(subset), fps=fps, worst=min(fps.values()) if fps else 0.0,
                   total=sum(fps.values()), misses=misses, jitter_ms=stats.jitter_ms,
                   sets=stats.sets, reason=reason)
        rows.append(row)
        if verbose:
            print("  %d  %-34s %8.1f  %9.1f  %6d  %6.1f ms%s"
                  % (n, ",".join(subset), row["worst"], row["total"], misses,
                     row["jitter_ms"], "  <-- BUS LIMIT" if row["worst"] < 1.0 else ""))
        time.sleep(0.4)                     # let the devices release before the next step
    if verbose:
        good = [r for r in rows if r.get("worst", 0) >= 1.0]
        if good:
            best = max(good, key=lambda r: r["n"])
            print("  => this bus carried %d camera(s) with every stream alive "
                  "(worst %.1f fps)" % (best["n"], best["worst"]))
        else:
            print("  => no step kept every stream alive — check power/cabling first")
        print("  if the limit is below the 6 the design needs: force MJPG, drop resolution,")
        print("  or split the cameras across two separate host controllers (opposite-side ports).")
    return rows


# ==========================================================================
#  Self-test (no hardware, no load): guard/rate/thermal/roles + a full loop
#  over FakeCapture cameras through the real run_loop code path.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== bank_bringup self-test (fake cameras + injected clocks, no hardware) ==")
    checks = []

    # (1) CpuGuard: an idle process is "ok"; a pinning one is "throttle"; sustained over the
    #     hard ceiling is "stop". Clock and CPU meter are injected, so this is exact.
    class _Fake:
        def __init__(self, burn):
            self.t = 0.0
            self.c = 0.0
            self.burn = burn        # CPU seconds consumed per wall second, per machine

        def clock(self):
            return self.t

        def cpu(self):
            return self.c

        def tick(self, dt, cores):
            self.t += dt
            self.c += self.burn * dt * cores

    idle = _Fake(0.10)
    g = CpuGuard(budget=0.5, cores=4, clock=idle.clock, cpu_fn=idle.cpu)
    idle.tick(1.0, 4)
    load_ok, v_ok = g.update()

    hot = _Fake(0.70)
    g2 = CpuGuard(budget=0.5, cores=4, grace=5.0, clock=hot.clock, cpu_fn=hot.cpu)
    hot.tick(1.0, 4)
    load_hot, v_hot = g2.update()          # 70% > budget 50%, under hard 80% -> throttle

    burn = _Fake(0.95)
    g3 = CpuGuard(budget=0.5, cores=4, grace=3.0, clock=burn.clock, cpu_fn=burn.cpu)
    verdicts = []
    for _ in range(6):                     # 6 s over the hard ceiling -> must reach "stop"
        burn.tick(1.0, 4)
        verdicts.append(g3.update()[1])
    checks.append(("CpuGuard: idle=ok (%.0f%%), busy=throttle (%.0f%%), pinned->stop"
                   % (load_ok * 100, load_hot * 100),
                   v_ok == "ok" and v_hot == "throttle" and "stop" in verdicts
                   and verdicts[0] == "throttle"))

    # (2) CpuGuard recovers: once load drops back under budget it stops demanding throttle
    cool = _Fake(0.95)
    g4 = CpuGuard(budget=0.5, cores=4, grace=3.0, clock=cool.clock, cpu_fn=cool.cpu)
    cool.tick(1.0, 4); g4.update()
    cool.burn = 0.05
    cool.tick(1.0, 4)
    checks.append(("CpuGuard clears its over-ceiling timer when load drops",
                   g4.update()[1] == "ok"))

    # (3) RateController: halves on back-off, floors, recovers, caps at target; sleep is sane
    rc = RateController(target=20.0, floor=2.5)
    r1 = rc.back_off()                    # 10
    r2 = rc.back_off()                    # 5
    for _ in range(5):
        rc.back_off()                     # floors at 2.5
    floored = rc.at_floor and abs(rc.rate - 2.5) < 1e-9
    for _ in range(20):
        rc.recover()
    capped = abs(rc.rate - 20.0) < 1e-9
    sleep_ok = (abs(RateController(target=10.0).sleep_for(0.02) - 0.08) < 1e-9
                and RateController(target=10.0).sleep_for(5.0) == 0.0)
    checks.append(("RateController: 20->%.0f->%.0f, floors at 2.5, recovers to target, sleep math"
                   % (r1, r2), r1 == 10.0 and r2 == 5.0 and floored and capped and sleep_ok))

    # (4) thermal parse on real `pmset -g therm` text
    sample = ("Note: No thermal warning level has been recorded\n"
              "2026-08-01 11:31:44 -1000 CPU Power notify\n"
              "\tCPU_Scheduler_Limit \t= 100\n\tCPU_Available_CPUs \t= 16\n"
              "\tCPU_Speed_Limit \t= 70\n")
    st = parse_thermal(sample)
    checks.append(("parse_thermal reads CPU_Speed_Limit=70 and flags de-rating",
                   st.get("CPU_Speed_Limit") == 70 and thermally_throttled(st)
                   and not thermally_throttled({"CPU_Speed_Limit": 100})))

    # (5) assign_roles handles a PARTIAL bank (the whole point) and a full one
    rng = np.random.default_rng(3)

    def mono_desc():
        g = rng.integers(0, 255, (800, 1280), dtype=np.uint8)
        return classify_frame(np.stack([g, g, g], axis=2))

    def color_desc():
        f = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
        f[:, :, 0] = np.clip(f[:, :, 0].astype(int) + 40, 0, 255)
        return classify_frame(f)

    two_mono = {1: mono_desc(), 2: mono_desc()}
    ri_partial, notes_partial = assign_roles(two_mono)
    partial_ok = (ri_partial == {"eyeL": 1, "eyeR": 2}
                  and any("2/4 mono" in n for n in notes_partial)
                  and any("PROVISIONAL" in n for n in notes_partial))

    full = {0: color_desc(), 1: color_desc(), 2: mono_desc(), 3: mono_desc(),
            4: mono_desc(), 5: mono_desc()}
    ri_full, _ = assign_roles(full)
    full_ok = set(ri_full) == set(CORE_ROLES) and len(set(ri_full.values())) == 6

    skip_ok = 0 not in assign_roles(full, skip=(0,))[0].values()
    checks.append(("assign_roles: 2 mono -> partial map + honest notes; 6 -> full; --skip honoured",
                   partial_ok and full_ok and skip_ok))

    # (5b) ONE-EYE layout: 2 inward cams serve the RIGHT eye (eyeR + pupilR), not eyeL/eyeR —
    #      and the map must warn that the 8-feature calibration contract is not satisfied.
    ri_one, notes_one = assign_roles({1: color_desc(), 2: color_desc(),
                                      3: mono_desc(), 4: mono_desc()}, layout="one_eye")
    one_eye_ok = (ri_one == {"worldL": 1, "worldR": 2, "eyeR": 3, "pupilR": 4}
                  and any("RIGHT eye" in n for n in notes_one)
                  and any("8-feature" in n for n in notes_one))
    # a lone inward cam fills eyeR first, and an extra one is reported, not silently dropped
    ri_one1, _ = assign_roles({3: mono_desc()}, layout="one_eye")
    _, notes_one3 = assign_roles({3: mono_desc(), 4: mono_desc(), 5: mono_desc()},
                                 layout="one_eye")
    checks.append(("assign_roles(one_eye): 2 mono -> eyeR+pupilR, 1 -> eyeR, extras reported",
                   one_eye_ok and ri_one1 == {"eyeR": 3}
                   and any("extras ignored" in n for n in notes_one3)))

    # (5c) parse_roles: the manual override that beats a wrong classification
    parsed = parse_roles(["eyeL=4", "eyeR=5", "worldL=2", "worldR=3"])
    def _rejects(spec):
        try:
            parse_roles(spec); return False
        except ValueError:
            return True
    checks.append(("parse_roles: reads role=index, rejects bad role / non-int / duplicate index",
                   parsed == {"eyeL": 4, "eyeR": 5, "worldL": 2, "worldR": 3}
                   and _rejects(["nose=1"]) and _rejects(["eyeL=x"])
                   and _rejects(["eyeL=2", "eyeR=2"]) and _rejects(["eyeL"])))

    # (6) THE LOOP: drive the real run_loop over fake cameras with an injected clock, and prove
    #     (a) it runs, (b) it throttles a hot machine, (c) it stops rather than cook it.
    from sync_capture import SyncBank, FakeCapture
    roles = ("worldL", "worldR", "eyeL", "eyeR")
    bank = SyncBank({r: i for i, r in enumerate(roles)}, fps=100,
                    opener=lambda role, idx: FakeCapture(role, res=SyncBank.ROLE_RES[role],
                                                         fps=100)).start()
    slept = {"total": 0.0}

    def fake_sleep(s):
        slept["total"] += s

    stats, reason = run_loop(bank, list(roles), seconds=0.4, guard=CpuGuard(budget=0.95),
                             rate=RateController(target=50.0), sleep=fake_sleep, verbose=False)
    bank.close()
    loop_ok = (stats.sets > 0 and reason == "done" and slept["total"] > 0
               and all(v["fps"] >= 0 for v in stats.report().values()))
    checks.append(("run_loop drives a 4-cam fake bank (%d sets), sleeps to hold its rate"
                   % stats.sets, loop_ok))

    # (5d) parse_res: explicit WxH matters because these sensors are 16:10, not 4:3
    def _res_rejects(spec):
        try:
            parse_res(spec); return False
        except ValueError:
            return True
    checks.append(("parse_res: '640'->640x480, '1280x800' exact, None passes, junk rejected",
                   parse_res("640") == (640, 480) and parse_res("1280x800") == (1280, 800)
                   and parse_res("1920X1200") == (1920, 1200) and parse_res(None) is None
                   and _res_rejects("abc") and _res_rejects("1280xyz")))

    # (6b) ramp: steps 1..N cumulatively and reports a row per step, worst-fps included
    from sync_capture import FakeCapture as _FC
    ramp_rows = ramp({"worldL": 0, "worldR": 1, "eyeL": 2}, seconds=0.15,
                     opener=lambda role, idx: _FC(role, res=SyncBank.ROLE_RES[role], fps=100),
                     verbose=False)
    checks.append(("ramp steps 1..N cumulatively, one row per step with worst/total fps",
                   [r["n"] for r in ramp_rows] == [1, 2, 3]
                   and [len(r["roles"]) for r in ramp_rows] == [1, 2, 3]
                   and all("worst" in r and r["sets"] > 0 for r in ramp_rows)))

    # (7) the same loop, but the machine is pinned: it must throttle and then stop early
    pinned = _Fake(0.99)
    bank2 = SyncBank({r: i for i, r in enumerate(roles)}, fps=100,
                     opener=lambda role, idx: FakeCapture(role, res=SyncBank.ROLE_RES[role],
                                                          fps=100)).start()

    def pinned_clock():
        pinned.tick(0.25, 4)               # every clock read advances 0.25 s of a pinned machine
        return pinned.t

    g_hot = CpuGuard(budget=0.5, cores=4, grace=2.0, clock=pinned.clock, cpu_fn=pinned.cpu)
    stats2, reason2 = run_loop(bank2, list(roles), seconds=30.0, guard=g_hot,
                               rate=RateController(target=50.0), clock=pinned_clock,
                               sleep=lambda s: None, verbose=False)
    bank2.close()
    checks.append(("run_loop STOPS a pinned machine instead of cooking it (reason=%s)" % reason2,
                   reason2 == "cpu-stop"))

    # DEAD-CAMERA ADVICE. The two causes are indistinguishable in the table and the wrong advice
    # sends you to cut a resolution that is already cut, so pin which one is named. Written from
    # the 2026-08-04 fault: one dead camera, everyone else at 640x480.
    SMALL = [(480, 640)] * 3
    NATIVE = [(1200, 1920)] * 3
    one_small = " ".join(dead_camera_advice(["eyeL"], SMALL + [None]))
    checks.append(("dead-cam advice: 1 dead + small frames -> ISOLATE, not bandwidth",
                   "NOT bandwidth" in one_small and "--skip" in one_small
                   and "--res 640" not in one_small))
    # NEGATIVE CONTROL: the same single dead camera with the survivors at NATIVE resolution is the
    # bus story, and must still be told as one -- otherwise this just renames the old advice.
    one_native = " ".join(dead_camera_advice(["eyeL"], NATIVE + [None]))
    checks.append(("dead-cam advice: 1 dead + NATIVE frames -> still the bus",
                   "--res 640" in one_native and "NOT bandwidth" not in one_native))
    two_small = " ".join(dead_camera_advice(["eyeL", "eyeR"], SMALL))
    checks.append(("dead-cam advice: 2 dead -> the bus, whatever the frame size",
                   "--res 640" in two_small and "NOT bandwidth" not in two_small))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "BANK BRINGUP OK — partial bank maps, CPU budget enforced ✅"
              if ok else "PROBLEM ⚠️")
        print("  on hardware:  python3 bank_bringup.py --scan   then   --run")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="bring up a partial camera bank under a CPU budget")
    ap.add_argument("--selftest", action="store_true", help="headless; no cameras, no load")
    ap.add_argument("--scan", action="store_true", help="list what is plugged in")
    ap.add_argument("--run", action="store_true", help="open the bank and run it, guarded")
    ap.add_argument("--ramp", action="store_true",
                    help="add cameras one at a time and report where the USB bus falls over")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--rate", type=float, default=20.0, help="target loop Hz (the CPU master lever)")
    ap.add_argument("--budget", type=float, default=0.50,
                    help="max fraction of the WHOLE machine's CPU this process may use")
    ap.add_argument("--res", type=str, default=None, metavar="WxH",
                    help="force capture size for every camera: '1280x800' or '640' (=640x480)")
    ap.add_argument("--cv-threads", type=int, default=2, help="cap OpenCV's decode thread pool")
    ap.add_argument("--skip", type=int, nargs="*", default=(),
                    help="indices to ignore (e.g. 0 = the built-in FaceTime camera)")
    ap.add_argument("--one-eye", action="store_true",
                    help="interim build: both inward cams serve the RIGHT eye (eyeR + pupilR)")
    ap.add_argument("--max-index", type=int, default=8)
    ap.add_argument("--view", action="store_true", help="live mosaic window (costs CPU)")
    ap.add_argument("--save-map", action="store_true", help="write a PROVISIONAL role map")
    ap.add_argument("--native", action="store_true",
                    help="request each sensor's real 16:10 mode (world 1920x1200, eye 1280x800)")
    ap.add_argument("--no-warm", action="store_true",
                    help="one grab per set instead of two (roughly doubles the set rate)")
    ap.add_argument("--roles", nargs="*", default=None, metavar="ROLE=INDEX",
                    help="override classification, e.g. --roles eyeL=4 eyeR=5 worldL=2 worldR=3")
    args = ap.parse_args()
    layout = "one_eye" if args.one_eye else "binocular"
    try:
        roles = parse_roles(args.roles) if args.roles else None
    except ValueError as e:
        sys.exit("bad --roles: %s" % e)
    try:
        res = parse_res(args.res)
    except ValueError as e:
        sys.exit(str(e))
    if args.scan:
        scan(args.max_index, skip=tuple(args.skip), layout=layout)
        sys.exit(0)
    if args.ramp:
        if not roles:
            sys.exit("--ramp needs --roles (it adds them in the order you list them)")
        ramp(roles, seconds=args.seconds if args.seconds < 10 else 4.0, res=res,
             cv_threads=args.cv_threads)
        sys.exit(0)
    if args.run:
        sys.exit(run(seconds=args.seconds, max_index=args.max_index, target_rate=args.rate,
                     budget=args.budget, res=res, skip=tuple(args.skip), layout=layout,
                     roles=roles, view=args.view, save=args.save_map, cv_threads=args.cv_threads,
                     warm=not args.no_warm, native=args.native))
    sys.exit(selftest())
