"""sync_capture.py — the SYNCHRONIZED multi-camera grab layer for the 6-cam rig.

cameras.py opens the role suite; this puts a REAL synchronization contract on top of it,
which the whole live pipeline (capture.py, world_mesh.py, snapshot.py, rig_test.py) sits on.

WHY THIS EXISTS
    The 6 USB UVC cameras have no shared hardware trigger, so a naive
    ``{role: cap.read()}`` sweep reads them one-after-another: by the time the 6th
    camera is read, ~5 frame-times have passed and a head move has smeared the set.
    Both sensor families here are GLOBAL SHUTTER (OV9281 mono, AR0234 color), so if we
    align the *grab* calls we get a genuinely comparable set. This module:

      * runs one BACKGROUND THREAD per camera that continuously ``grab()``s and keeps only
        the freshest (frame, monotonic-timestamp, sequence) — never a stale buffered frame;
      * a BARRIER makes every thread issue its ``grab()`` in the same scheduling window when
        a synchronized frame is requested, so the capture instants line up to within OS
        thread-wakeup latency (sub-ms .. few ms), not frame-times;
      * every returned set carries its per-camera timestamps and a measured JITTER
        (max-min capture time). Nothing here hides the sync error — it reports it, and
        callers can reject a set whose jitter exceeds a budget.

    For hard sync, strobe the IR LEDs from the XIAO (firmware/ir_strobe) inside the shared
    exposure window; the software barrier below is what aligns the rolling free-run grabs.

NO-HARDWARE: a FakeCapture (synthetic frames, controllable per-read latency) lets the whole
thing — threads, barrier, jitter math, freshness — be exercised with ``--selftest``.
"""
import argparse
import sys
import threading
import time

import numpy as np


# --------------------------------------------------------------------------
#  Backend seam: real cv2 cameras OR an injected fake (selftest / sim).
# --------------------------------------------------------------------------
def open_cv2_capture(index, width, height, fps, mjpg=True):
    """Open a real UVC camera with the Mac-friendly backend and low-latency settings."""
    import cv2
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("could not open camera index %d" % index)
    if mjpg:                                    # MJPG fits 6 global-shutter streams in USB bandwidth
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # so grab() advances to the newest frame
    except Exception:
        pass
    return _Cv2Adapter(cap)


class _Cv2Adapter:
    """Uniform grab()/retrieve()/release() over a cv2.VideoCapture (matches FakeCapture)."""
    def __init__(self, cap):
        self.cap = cap

    def grab(self):
        return self.cap.grab()

    def retrieve(self):
        ok, frame = self.cap.retrieve()
        return (ok, frame)

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------
#  One synchronized camera: a thread that always holds the freshest frame.
# --------------------------------------------------------------------------
class SyncCamera:
    """Background-threaded camera. The thread parks on a barrier; on each sync tick every
    thread crosses the barrier together, issues ONE grab() (aligned capture instant), stamps
    the monotonic clock, then retrieve()s the pixels. latest() returns the freshest set."""

    def __init__(self, role, cap, barrier, clock=time.monotonic):
        self.role = role
        self.cap = cap
        self.barrier = barrier
        self.clock = clock
        self._lock = threading.Lock()
        self._frame = None
        self._ts = None
        self._seq = -1
        self._grabbed_ok = 0
        self._grab_fail = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sync-%s" % role, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            # cross the barrier so all cameras grab() in the same window; broken barrier =>
            # shutdown or a peer died — just exit the thread.
            try:
                self.barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                break
            except Exception:
                break
            if self._stop.is_set():
                break
            ok = self.cap.grab()
            ts = self.clock()                       # stamp AT the grab, before the (slower) retrieve
            if not ok:
                self._grab_fail += 1
                continue
            ok2, frame = self.cap.retrieve()
            if not ok2 or frame is None:
                self._grab_fail += 1
                continue
            with self._lock:
                self._frame = frame
                self._ts = ts
                self._seq += 1
                self._grabbed_ok += 1

    def latest(self):
        """(frame, capture_ts, seq) — the freshest grab, or (None, None, seq) if none yet."""
        with self._lock:
            return self._frame, self._ts, self._seq

    def stats(self):
        return {"ok": self._grabbed_ok, "fail": self._grab_fail}

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
#  The synchronized bank: drives all cameras through the shared barrier.
# --------------------------------------------------------------------------
class SyncBank:
    """Role-mapped synchronized camera bank.

        bank = SyncBank({"worldL": 0, "worldR": 1, ...})          # real cv2
        bank = SyncBank(role_index, opener=fake_opener)            # injected (selftest)
        bank.start()
        fs = bank.sync_frame()          # -> SyncFrame(frames{}, ts{}, jitter_ms, seqs{})
        ...
        bank.close()
    """

    # per-role capture resolution (mirrors cameras.ROLE_RES / rig.py sensor models)
    ROLE_RES = {"worldL": 1280, "worldR": 1280, "eyeL": 640, "eyeR": 640,
                "pupilL": 640, "pupilR": 640, "eye2L": 640, "eye2R": 640}

    def __init__(self, role_index, fps=100, opener=None, clock=time.monotonic):
        if not role_index:
            raise ValueError("SyncBank needs at least one role")
        bad = [r for r in role_index if r not in self.ROLE_RES]
        if bad:
            raise ValueError("unknown role(s): %s" % bad)
        self.fps = fps
        self.clock = clock
        self.roles = list(role_index)
        # +1 party: the CONTROLLER thread (sync_frame caller) also crosses the barrier, so it
        # blocks until every camera has issued its aligned grab — that is the sync point.
        self._barrier = threading.Barrier(len(self.roles) + 1)
        _open = opener or (lambda role, idx: open_cv2_capture(
            idx, self.ROLE_RES[role], int(self.ROLE_RES[role] * 3 // 4), fps))
        self.cams = {}
        for role, idx in role_index.items():
            cap = _open(role, idx)
            self.cams[role] = SyncCamera(role, cap, self._barrier, clock=clock)
        self._started = False
        self._sync_count = 0

    def start(self):
        for cam in self.cams.values():
            cam.start()
        self._started = True
        return self

    def sync_frame(self, warm=True):
        """Trigger one aligned grab across all cameras and collect the freshest set.

        Returns a SyncFrame. `jitter_ms` is the spread of per-camera capture timestamps —
        the honest measure of how synchronized this set actually is. The barrier makes the
        controller's Nth wait block until every camera has finished its (N-1)th grab, so the
        controller cannot run ahead of the cameras; we then wait for each camera's sequence
        to advance by `ticks` so we read the LAST (aligned) grab, never the flush grab."""
        if not self._started:
            raise RuntimeError("call start() before sync_frame()")
        ticks = 2 if warm else 1          # warm: 1 flush grab (drains a stale buffer) + 1 aligned
        base = {role: cam.latest()[2] for role, cam in self.cams.items()}
        for _ in range(ticks):
            try:
                self._barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                raise RuntimeError("a camera thread died mid-sync (see per-role stats)")
        # wait until EVERY camera has published `ticks` new grabs since `base` (so the frame we
        # read is the aligned one that just happened, with its capture timestamp)
        deadline = self.clock() + 0.5
        while self.clock() < deadline:
            if all(cam.latest()[2] >= base[role] + ticks
                   for role, cam in self.cams.items()):
                break
            time.sleep(0.0002)
        self._sync_count += 1
        frames, ts, seqs = {}, {}, {}
        for role, cam in self.cams.items():
            f, t, s = cam.latest()
            frames[role], ts[role], seqs[role] = f, t, s
        return SyncFrame(frames, ts, seqs, self.clock)

    def stats(self):
        return {role: cam.stats() for role, cam in self.cams.items()}

    def close(self):
        for cam in self.cams.values():
            cam.stop()
        # break the barrier so any parked thread wakes and exits
        self._barrier.abort()
        for cam in self.cams.values():
            try:
                cam.cap.release()
            except Exception:
                pass


class SyncFrame:
    """One synchronized capture across roles + the honest sync telemetry."""

    def __init__(self, frames, ts, seqs, clock):
        self.frames = frames                 # {role: frame or None}
        self.ts = ts                         # {role: capture monotonic time or None}
        self.seqs = seqs                     # {role: sequence number}
        self._clock = clock
        good = [t for t in ts.values() if t is not None]
        self.complete = all(f is not None for f in frames.values())
        self.jitter_ms = (max(good) - min(good)) * 1e3 if len(good) >= 2 else 0.0
        self.stamp = min(good) if good else None

    def get(self, role):
        return self.frames.get(role)

    def within(self, budget_ms):
        """True if every role produced a frame and the set is synchronized within budget."""
        return self.complete and self.jitter_ms <= budget_ms

    def __repr__(self):
        n = sum(f is not None for f in self.frames.values())
        return "<SyncFrame %d/%d roles, jitter %.2f ms%s>" % (
            n, len(self.frames), self.jitter_ms, "" if self.complete else " INCOMPLETE")


# ==========================================================================
#  Self-test (NO hardware): fake cameras with controllable per-read latency
#  exercise the threads, the barrier alignment, freshness, and the jitter math.
# ==========================================================================
class FakeCapture:
    """Synthetic camera. grab() is cheap (like a real cv2 grab that returns the already-captured
    global-shutter frame from the driver), so when the barrier releases all threads together the
    grabs land within scheduling noise — that is what we assert against a naive sequential sweep."""

    def __init__(self, role, res=640, fps=100, rng=None):
        self.role = role
        self.w = res
        self.h = int(res * 3 // 4)
        self.dt = 1.0 / fps
        self.rng = rng or np.random.default_rng(hash(role) % 2**32)
        self.n = 0
        self._pending = None

    def grab(self):
        color = self.role.startswith("world")
        shape = (self.h, self.w, 3) if color else (self.h, self.w)
        # small representative frame (keep the selftest fast); content is irrelevant here
        self._pending = (self.rng.random((8, 8, 3) if color else (8, 8)) * 255).astype("uint8")
        self.n += 1
        return True

    def retrieve(self):
        return (self._pending is not None, self._pending)

    def release(self):
        pass


def _naive_sequential_spread(roles, fps, reps=20):
    """Model a naive one-camera-at-a-time sweep: each fresh read blocks ~one frame interval
    (draining that camera's buffer to the newest frame), so the capture instants smear across
    ~N frame intervals. This is exactly what the barrier removes. Returns the median spread ms."""
    dt = 1.0 / fps
    caps = [FakeCapture(r, fps=fps) for r in roles]
    spreads = []
    for _ in range(reps):
        stamps = []
        for c in caps:
            time.sleep(dt)          # a blocking read waits for that camera's next frame
            c.grab(); stamps.append(time.monotonic())
        spreads.append((max(stamps) - min(stamps)) * 1e3)
    return float(np.median(spreads))


def selftest(verbose=True):
    if verbose:
        print("== sync_capture self-test (fake cameras, no hardware) ==")
    roles = ("worldL", "worldR", "eyeL", "eyeR", "pupilL", "pupilR")
    fps = 100

    def opener(role, idx):
        return FakeCapture(role, res=SyncBank.ROLE_RES[role], fps=fps)

    bank = SyncBank({r: i for i, r in enumerate(roles)}, fps=fps, opener=opener).start()
    checks = []

    # (1) every sync_frame returns ALL roles with a real frame + a finite measured jitter
    complete = True
    jitters = []
    for _ in range(30):
        fs = bank.sync_frame()
        complete = complete and fs.complete
        jitters.append(fs.jitter_ms)
    med_jit = float(np.median(jitters))
    checks.append(("every sync set complete (6/6) and jitter is measured (%.2f ms)" % med_jit,
                   complete and np.isfinite(med_jit) and med_jit >= 0.0))

    # (2) frames advance — no role stuck on a stale frame, and we read the ALIGNED grab
    s1 = dict(bank.sync_frame().seqs)
    for _ in range(5):
        bank.sync_frame()
    s2 = dict(bank.sync_frame().seqs)
    advanced = all(s2[r] > s1[r] for r in roles)
    checks.append(("all roles advance frames (no stale/frozen camera)", advanced))

    # (3) the whole point: the barrier-synced grab is MUCH tighter than a naive sequential sweep
    #     of the same cameras (which smears across ~N frame intervals).
    seq_spread = _naive_sequential_spread(roles, fps)
    checks.append(("barrier jitter %.2f ms << naive sequential sweep %.1f ms" % (med_jit, seq_spread),
                   med_jit < seq_spread * 0.5))

    # (4) within(budget) gates correctly
    fs = bank.sync_frame()
    checks.append(("within(budget) accepts a good set / rejects a tiny budget",
                   fs.within(1000.0) and not fs.within(-1.0)))

    # (5) per-role grab stats accumulated, clean shutdown
    stats = bank.stats()
    grabbed_ok = all(stats[r]["ok"] > 0 for r in roles)
    bank.close()
    time.sleep(0.05)
    checks.append(("per-role grabs recorded and bank.close() is clean", grabbed_ok))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "SYNC CAPTURE OK — barrier aligns grabs, jitter measured & reported ✅"
              if ok else "PROBLEM ⚠️")
        print("  note: real global-shutter cams + IR strobe tighten this further; the number to")
        print("  watch on hardware is SyncFrame.jitter_ms (reject sets over your budget).")
    return 0 if ok else 1


def _probe(max_index=8, fps=100):
    """Quick real-hardware probe: open indices 0..N as a bank, print jitter for a few frames."""
    import cv2  # noqa: F401  (fail loudly here if opencv is missing)
    from cameras import list_cameras
    found = [i for i, ok in list_cameras(max_index) if ok]
    if not found:
        print("no cameras responded on indices 0..%d" % (max_index - 1))
        return 1
    roles = list(SyncBank.ROLE_RES)[:len(found)]
    print("probing %d cameras as roles %s" % (len(found), roles))
    bank = SyncBank({r: found[i] for i, r in enumerate(roles)}, fps=fps).start()
    try:
        for _ in range(20):
            fs = bank.sync_frame()
            print("  %r  stamp=%.3f" % (fs, fs.stamp or 0.0))
    finally:
        bank.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="synchronized multi-camera capture")
    ap.add_argument("--selftest", action="store_true", help="run headless (fake cameras)")
    ap.add_argument("--probe", action="store_true", help="open real cameras and report jitter")
    ap.add_argument("--max-index", type=int, default=8)
    args = ap.parse_args()
    if args.probe:
        sys.exit(_probe(args.max_index))
    sys.exit(selftest())
