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

    def __init__(self, port=None, simulate=False, on_sample=None, baud=115200, seed=0):
        self.on_sample = on_sample
        self.filter = imu.ImuFilter()
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
