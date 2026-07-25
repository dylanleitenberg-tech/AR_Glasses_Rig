"""imu_serial.py — the XIAO -> USB serial IMU feed (Phase 4), testable with no hardware.

PROTOCOL (the XIAO firmware side is mechanical to write from this):
  115200 baud, one CSV line per sample at ~100-200 Hz:
      millis,ax,ay,az,gx,gy,gz\\n
  accel in g (board X->+x right, Y->+y forward, Z->+z up — the imu_mount tower orientation),
  gyro in deg/s. Lines starting with '#' are ignored (boot banners). Malformed lines are
  dropped and counted, never raised — USB serial glitches.

WHAT IT FEEDS: imu.ImuFilter (accel tilt + gyro rate -> filtered pantoscopic/roll tilt) for
slip/bump detection + gravity-down reference — ROBUSTNESS only, not geometry ID (validated
in sim: IMU marginal ~0 for ID/registration).

USAGE:
  ImuSerial("/dev/tty.usbmodem*", on_sample=...)   # real XIAO (needs pyserial)
  ImuSerial(simulate=True, ...)                    # synthetic gravity+noise+slip stream
  python3 imu_serial.py --selftest                 # parser + filter over a synthetic stream
"""
import glob
import math
import time

import numpy as np

import imu


def parse_line(line):
    """CSV line -> (t_ms, accel_g[3], gyro_dps[3]) or None. Never raises."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        v = [float(x) for x in parts]
    except ValueError:
        return None
    return v[0], np.array(v[1:4]), np.array(v[4:7])


def accel_to_tilt(accel_g):
    """Gravity direction -> (pantoscopic, roll) tilt in degrees (imu.py's convention:
    board flat = 0/0; pitch about +x maps to dev[0], roll about +y to dev[2])."""
    ax, ay, az = accel_g
    pitch = math.degrees(math.atan2(ay, az)) if (ay or az) else 0.0
    roll = math.degrees(math.atan2(-ax, az)) if (ax or az) else 0.0
    return pitch, roll


class SimStream:
    """Synthetic XIAO: gravity for a slowly swaying head + occasional slip/bump events."""
    def __init__(self, rate_hz=150, seed=0):
        self.dt = 1.0 / rate_hz
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.pitch = -7.0        # resting pantoscopic tilt
        self.roll = 0.0
        self.next_event = self.rng.uniform(3, 8)

    def readline(self):
        time_ms = int(self.t * 1000)
        self.t += self.dt
        self.pitch += 0.4 * self.dt * math.sin(0.3 * self.t)      # head sway
        if self.t > self.next_event:                               # slip/bump
            self.pitch += self.rng.normal(0, 1.2)
            self.roll += self.rng.normal(0, 0.6)
            self.next_event = self.t + self.rng.uniform(3, 8)
        p, r = math.radians(self.pitch), math.radians(self.roll)
        g = np.array([-math.sin(r), math.sin(p) * math.cos(r), math.cos(p) * math.cos(r)])
        acc = g + self.rng.normal(0, 0.006, 3)                     # accel noise (g)
        gyr = self.rng.normal(0, 0.4, 3)                           # deg/s noise
        return "%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f\n" % (time_ms, *acc, *gyr)


class ImuSerial:
    """Line-pump: real serial port or SimStream -> parse -> ImuFilter -> on_sample callback
    with (t_ms, filtered_pitch, filtered_roll, bump). `bump` = gyro magnitude spike."""
    BUMP_DPS = 25.0

    def __init__(self, port=None, simulate=False, on_sample=None, baud=115200, seed=0,
                 on_raw=None):
        self.on_sample = on_sample
        self.on_raw = on_raw            # optional callback(t_ms, accel_g[3], gyro_dps[3]) — raw,
        self.filter = imu.ImuFilter()   #  used by GyroIntegrator to integrate a rotation increment
        self.bad_lines = 0
        self.n = 0
        if simulate:
            self.src = SimStream(seed=seed)
        else:
            import serial                      # pyserial, only needed for real hardware
            if port is None or "*" in (port or ""):
                hits = glob.glob(port or "/dev/tty.usbmodem*")
                if not hits:
                    raise RuntimeError("no XIAO serial port found — is it plugged in?")
                port = hits[0]
            self.src = serial.Serial(port, baud, timeout=1)

    def pump(self, n_samples=None, duration_s=None):
        """Read+process; returns list of (t_ms, pitch, roll, bump)."""
        out = []
        t_end = time.time() + duration_s if duration_s else None
        last_ms = None
        while True:
            if n_samples is not None and self.n >= n_samples:
                break
            if t_end is not None and time.time() > t_end:
                break
            raw = self.src.readline()
            if isinstance(raw, bytes):
                raw = raw.decode("ascii", "replace")
            rec = parse_line(raw)
            if rec is None:
                self.bad_lines += 1
                continue
            t_ms, acc, gyr = rec
            if self.on_raw:
                self.on_raw(t_ms, acc, gyr)
            dt = 1e-3 * (t_ms - last_ms) if last_ms is not None else 1.0 / 150
            last_ms = t_ms
            ap, ar = accel_to_tilt(acc)
            p, r = self.filter.update((ap, ar), gyro_rate=(gyr[0], gyr[1]),
                                      dt=max(1e-4, dt))
            bump = bool(np.linalg.norm(gyr) > self.BUMP_DPS)
            self.n += 1
            sample = (t_ms, p, r, bump)
            out.append(sample)
            if self.on_sample:
                self.on_sample(*sample)
        return out


def _rodrigues(rvec):
    """Axis-angle rotation vector (radians) -> 3x3 rotation matrix (exact, small-angle safe)."""
    rvec = np.asarray(rvec, float)
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


class GyroIntegrator:
    """Integrates the 3-axis gyro into a rotation INCREMENT the world-mesh tracker can consume.

    The mesh needs, each camera frame, an estimate of how the head (and thus the camera) rotated
    since the last frame — used as a prior and to carry the pose through a visual dropout. The
    gyro gives body angular velocity (deg/s); this composes each sample's incremental rotation
    (Rodrigues of omega*dt) into an accumulator, and consume_rotation() hands back the net
    rotation since the previous call and resets. Runs the ImuSerial pump on a background thread so
    the vision loop just calls consume_rotation() at its own rate.

    NOTE: the body->camera axis mapping (imu_mount orientation) is a fixed one-time calibration;
    the mesh uses this as a PRIOR/gate (imu_gate_deg), so an approximate mapping is fine. Set
    `axes` to permute/flip gyro axes onto the camera frame once measured on hardware."""

    def __init__(self, port=None, simulate=False, seed=0, axes=(0, 1, 2), signs=(1, 1, 1)):
        import threading
        self._lock = threading.Lock()
        self._accum = np.eye(3)
        self._last_ms = None
        self.samples = 0
        self.axes = axes
        self.signs = np.asarray(signs, float)
        self.imu = ImuSerial(port=port, simulate=simulate, seed=seed, on_raw=self._ingest)
        self._thread = None
        self._threading = threading

    def _ingest(self, t_ms, acc, gyr):
        dt = 1e-3 * (t_ms - self._last_ms) if self._last_ms is not None else 1.0 / 150
        self._last_ms = t_ms
        g = np.asarray(gyr, float)[list(self.axes)] * self.signs        # map onto camera axes
        rvec = np.radians(g) * max(1e-4, dt)                            # deg/s -> rad increment
        R_step = _rodrigues(rvec)
        with self._lock:
            self._accum = R_step @ self._accum
            self.samples += 1

    def consume_rotation(self):
        """Net world->cam rotation increment since the last call; resets the accumulator. Returns
        identity if no samples have arrived yet."""
        with self._lock:
            R = self._accum
            self._accum = np.eye(3)
            return R

    def start(self):
        """Run the pump on a daemon thread (real hardware / continuous sim)."""
        self._thread = self._threading.Thread(target=self.imu.pump, daemon=True)
        self._thread.start()
        return self

    def pump_n(self, n):
        """Synchronously pump n samples (used by the sim selftest)."""
        self.imu.pump(n_samples=n)


def gyro_selftest(verbose=True):
    """Integrator math: a known constant angular velocity integrates to the expected rotation,
    and a quiet (noise-only) sim stream stays near identity over a short window."""
    checks = []
    # (1) constant 30 deg/s about +z for 2 s -> ~60 deg net rotation about z
    gi = GyroIntegrator(simulate=True)          # src unused; we feed _ingest directly
    for k in range(200):                        # 200 samples * 0.01 s = 2 s
        gi._ingest(k * 10, [0, 0, 1.0], [0.0, 0.0, 30.0])
    R = gi.consume_rotation()
    ang = math.degrees(math.acos((np.trace(R) - 1) / 2))
    axis_z = abs(R[1, 0] - R[0, 1]) > 1e-6      # rotation is about z (sanity)
    checks.append(("constant 30 deg/s x 2 s integrates to %.1f deg about z" % ang,
                   abs(ang - 60.0) < 0.5 and axis_z))
    # (2) consume RESETS the accumulator
    R2 = gi.consume_rotation()
    checks.append(("consume_rotation resets to identity",
                   np.allclose(R2, np.eye(3), atol=1e-9)))
    # (3) a quiet noise-only sim stream stays near identity over ~1 s
    gq = GyroIntegrator(simulate=True, seed=2)
    gq.pump_n(150)
    Rq = gq.consume_rotation()
    drift = math.degrees(math.acos(np.clip((np.trace(Rq) - 1) / 2, -1, 1)))
    checks.append(("quiet sim stream drift over ~1 s is small (%.2f deg)" % drift, drift < 5.0))
    ok = all(v for _, v in checks)
    if verbose:
        print("== gyro integrator selftest ==")
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  => %s" % ("GYRO INTEGRATOR OK ✅" if ok else "PROBLEM ⚠️"))
    return ok


def selftest(verbose=True):
    """Parser robustness + end-to-end sim stream through the filter."""
    ok = True
    ok &= parse_line("12,0,0,1,0,0,0") is not None
    ok &= parse_line("# boot banner") is None
    ok &= parse_line("garbage,here") is None
    ok &= parse_line("1,2,3") is None
    ok &= parse_line("") is None
    p, r = accel_to_tilt(np.array([0.0, 0.0, 1.0]))
    ok &= abs(p) < 1e-9 and abs(r) < 1e-9
    src = ImuSerial(simulate=True, seed=1)
    rows = src.pump(n_samples=1200)                       # ~8 s of sim
    pitches = np.array([x[1] for x in rows[300:]])
    resting = np.median(pitches)
    ok &= len(rows) == 1200 and src.bad_lines == 0
    ok &= -12 < resting < -2                              # tracks the ~-7 deg resting tilt
    jitter = np.std(np.diff(pitches))
    ok &= jitter < 0.5                                    # filtered, not raw-noise level
    if verbose:
        print("== imu_serial selftest ==")
        print("  parser: banners/garbage dropped, CSV parsed        [%s]" % ("PASS" if ok else "FAIL"))
        print("  sim stream: %d samples, resting tilt %.1f deg (true -7), step jitter %.3f deg"
              % (len(rows), resting, jitter))
        print("  => %s" % ("PASS ✅" if ok else "FAIL ❌"))
    ok = bool(ok) and gyro_selftest(verbose)          # include the rotation-integrator checks
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    src = ImuSerial(simulate="--simulate" in sys.argv,
                    port=None if "--simulate" in sys.argv else "/dev/tty.usbmodem*",
                    on_sample=lambda t, p, r, b: print("%8d ms  pitch %+6.2f  roll %+6.2f%s"
                                                       % (t, p, r, "  BUMP" if b else "")))
    src.pump(duration_s=5)
