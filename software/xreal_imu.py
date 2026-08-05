"""xreal_imu.py, read the XREAL One Pro's OWN IMU. No soldering, no extra hardware.

WHY THIS AND NOT A SEPARATE IMU
    The CAD has a mount for a XIAO-based IMU (see imu_serial.py), and that is real hardware work:
    solder, mount, route a cable on a carrier that is FINAL with no printer access. The glasses
    already contain an IMU and already expose it over the USB-C cable that is plugged in.

    It is also the more CORRECT sensor. The XREAL's IMU is rigid to the GLASSES, which is where the
    DISPLAY is, and late-stage reprojection needs display-relative rotation. A carrier-mounted IMU
    would be rigid to the CAMERAS -- the right reference for VIO, the wrong one for reprojection.
    Our carrier is zip-tied and moves relative to the glasses, so the distinction is not academic.

WHAT IT IS FOR
    1. LATENCY. The pipeline runs 55.9 ms/frame (17.9 fps). An IMU runs 100-1000 Hz. That gap is
       the whole point of the fast/slow split every XR stack uses: gyro for high-rate rotation,
       cameras for absolute correction. predictor.py currently estimates velocity from the dot's
       image motion; when this lands it simply supplies that velocity instead, and nothing else
       in the chain changes.
    2. WorldTracker's 0 map points, which HANDOFF records as a geometry problem needing an IMU
       (stereo init degrades under rotation-dominant motion). Same component, two problems.

PROTOCOL, REVERSED AND CONFIRMED ON THIS DEVICE 2026-08-03. It is NOT HID.
    The glasses present as a USB NETWORK device. On this Mac they came up as `en8` with the host
    at 169.254.2.10 and the glasses at **169.254.2.1**, serving a binary stream on **TCP 52998**.
    (The HID interfaces exist -- VendorID 0x3318, usage pages 0x41 and 0x0c -- and can be opened,
    but produce no reports. That is a dead end; do not spend time on it.)

    Stream: fixed **134-byte records**, magic `28 36 00 00`, arriving at **~1400 Hz**.

        offset 0x22 (34)   3x float32 LE   GYRO   rad/s
        offset 0x2e (46)   3x float32 LE   ACCEL  m/s^2

    CONFIRMED BY PHYSICS, NOT BY EYE: stationary, the accel triple has magnitude **9.77 m/s^2**
    against gravity's 9.81 -- 0.4% off, which is mounting tilt and bias, not a mis-parse. The gyro
    triple sits at ~0.001 rad/s with ~0.0013 std, which is what a still gyro looks like. Those two
    facts together are what make this a decode rather than a plausible-looking guess; this project
    has shipped enough of the latter to insist on the difference.

    python3 xreal_imu.py --live       live gyro/accel, verifies |a| ~ g
    python3 xreal_imu.py --selftest   decode + integrator logic, no hardware needed
    python3 xreal_imu.py --check      HID diagnostic (dead end, kept for the record)
"""
import argparse
import struct
import sys
import time

import numpy as np

XREAL_VID = 0x3318          # 13080, confirmed on this rig via ioreg
IMU_USAGE_PAGE = 0x41       # vendor-defined page the community drivers read IMU from


def enumerate_xreal():
    """All HID interfaces belonging to the glasses. Empty list means hidden or absent."""
    try:
        import hid
    except ImportError:
        return None
    return [d for d in hid.enumerate() if d.get("vendor_id") == XREAL_VID]


def ioreg_sees_it():
    """Does the OS know the device exists, even if HID access is denied?

    This is the whole diagnostic: if IOKit lists it and hidapi does not, the device is fine and
    the problem is PERMISSION -- which is a very different fix from a cable or a driver.
    """
    import subprocess
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDDevice", "-w0", "-l"],
                             capture_output=True, text=True, timeout=20).stdout
        return out.count('"VendorID" = %d' % XREAL_VID)
    except Exception:
        return 0


def check():
    """Report exactly which of the three possible states we are in."""
    n_ioreg = ioreg_sees_it()
    devs = enumerate_xreal()
    print("IOKit HID entries with VendorID 0x%04x : %d" % (XREAL_VID, n_ioreg))
    if devs is None:
        print("hidapi                                  : NOT INSTALLED")
        print("\n  fix:  pip install hidapi")
        return 1
    print("hidapi devices visible                  : %d" % len(devs))
    for d in devs:
        print("   pid 0x%04x  usage_page 0x%02x usage 0x%02x  iface %s  %r"
              % (d["product_id"], d.get("usage_page", 0), d.get("usage", 0),
                 d.get("interface_number"), d.get("product_string")))
    if n_ioreg and not devs:
        print("\n  >>> THE DEVICE IS PRESENT BUT macOS IS HIDING IT FROM THIS PROCESS.")
        print("      IOKit lists %d HID interfaces for the glasses; hidapi sees none. That gap is" % n_ioreg)
        print("      the privacy gate, not a hardware fault.")
        print("\n      FIX: System Settings -> Privacy & Security -> Input Monitoring,")
        print("           then ADD AND ENABLE the app running this (Terminal, iTerm, or VS Code).")
        print("           You may need to quit and reopen it afterwards.")
        print("\n      Sanity check that it is really permission: a device you HAVE granted")
        print("      access to will appear in `hid.enumerate()`; Apple's own devices always do,")
        print("      which is why the list is not simply empty.")
        return 2
    if not n_ioreg:
        print("\n  >>> THE GLASSES ARE NOT ENUMERATING AT ALL. Check the USB-C cable and hub;")
        print("      this is the same failure mode as the cameras dropping off the bus.")
        return 3
    print("\n  ACCESS OK, %d interface(s) readable. Use --dump to capture raw reports." % len(devs))
    return 0


def open_imu():
    """Open the interface most likely to carry IMU reports. Returns (device, info) or (None, why)."""
    devs = enumerate_xreal()
    if devs is None:
        return None, "hidapi not installed (pip install hidapi)"
    if not devs:
        return None, ("no XREAL HID interface visible, run --check; on macOS this is almost "
                      "always the Input Monitoring permission")
    import hid
    # Prefer the vendor-defined usage page, which is where the community drivers find the IMU.
    ranked = sorted(devs, key=lambda d: 0 if d.get("usage_page") == IMU_USAGE_PAGE else 1)
    for d in ranked:
        try:
            h = hid.device()
            h.open_path(d["path"])
            h.set_nonblocking(True)
            return h, d
        except Exception:
            continue
    return None, "found %d interface(s) but none could be opened" % len(devs)


IMU_HOST = "169.254.2.1"
IMU_PORT = 52998
REC_LEN = 134
REC_MAGIC = b"\x28\x36\x00\x00"
OFF_GYRO = 0x22          # 3x float32 LE, rad/s
OFF_ACCEL = 0x2e         # 3x float32 LE, m/s^2
GRAVITY = 9.81


def decode(rec):
    """134-byte record -> (gyro rad/s, accel m/s^2). Returns None if the record is not ours."""
    if len(rec) < REC_LEN or rec[:4] != REC_MAGIC:
        return None
    g = np.array(struct.unpack_from("<3f", rec, OFF_GYRO), float)
    a = np.array(struct.unpack_from("<3f", rec, OFF_ACCEL), float)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(a))):
        return None
    return g, a


class XrealIMU:
    """Streaming reader. `read()` drains the socket and returns the newest (gyro, accel, t).

    Non-blocking by design: the caller is a render loop at 17.9 fps while this arrives at 1400 Hz,
    so the useful operation is "give me the latest", not "give me every sample". Anything that
    needs every sample (integration) should call `drain()`.
    """

    def __init__(self, host=IMU_HOST, port=IMU_PORT, timeout=2.0):
        self.host, self.port, self.timeout = host, int(port), float(timeout)
        self.sock = None
        self._buf = b""
        self.n_records = 0

    def open(self):
        import socket
        s = socket.socket()
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        s.setblocking(False)
        self.sock = s
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *a):
        self.close()

    def drain(self):
        """All complete records since the last call, oldest first: [(gyro, accel, t), ...]."""
        if self.sock is None:
            return []
        t = time.time()
        while True:
            try:
                d = self.sock.recv(65536)
            except Exception:
                break
            if not d:
                break
            self._buf += d
        out = []
        # Resync on the magic rather than assuming alignment -- a partial read at startup would
        # otherwise offset every subsequent record by a few bytes and silently produce garbage
        # that still decodes to finite floats.
        i = self._buf.find(REC_MAGIC)
        if i < 0:
            if len(self._buf) > 4 * REC_LEN:
                self._buf = self._buf[-REC_LEN:]
            return out
        while i + REC_LEN <= len(self._buf):
            r = decode(self._buf[i:i + REC_LEN])
            if r is None:
                nxt = self._buf.find(REC_MAGIC, i + 1)
                if nxt < 0:
                    break
                i = nxt
                continue
            out.append((r[0], r[1], t))
            self.n_records += 1
            i += REC_LEN
        self._buf = self._buf[i:]
        return out

    def read(self):
        """Newest sample only, or None. What a render loop wants."""
        rs = self.drain()
        return rs[-1] if rs else None


def dump(seconds=10.0, max_reports=40):
    """Print raw HID reports so the packet format can be reversed against the prior art.

    DELIBERATELY NOT A DECODER. Guessing a layout and shipping it is how a plausible-looking but
    wrong number ends up in a calibration -- this project has been bitten by exactly that class of
    error repeatedly. Capture first, decode against evidence.
    """
    h, info = open_imu()
    if h is None:
        print("cannot open: %s" % info)
        return 1
    print("opened %r (usage_page 0x%02x). Move your head; reports follow.\n"
          % (info.get("product_string"), info.get("usage_page", 0)))
    t0, n, lens = time.time(), 0, {}
    while time.time() - t0 < seconds and n < max_reports:
        try:
            data = h.read(64)
        except Exception as e:
            print("read failed: %s" % e)
            break
        if data:
            n += 1
            lens[len(data)] = lens.get(len(data), 0) + 1
            if n <= max_reports:
                print("  [%2d] len=%2d  %s" % (n, len(data),
                                               " ".join("%02x" % b for b in data[:32])))
        else:
            time.sleep(0.002)
    h.close()
    print("\n%d reports in %.1fs; length histogram: %s" % (n, time.time() - t0, lens))
    if n == 0:
        print("NO REPORTS. The interface opened but produced nothing, likely the wrong one of the "
              "several HID interfaces, or the device needs an enable command first (the nrealAir "
              "drivers send one). Compare against the prior art listed in this module's docstring.")
    return 0


class GyroIntegrator:
    """Integrate angular rate into a rotation estimate, with the drift caveat made explicit.

    A GYRO integrates cleanly over SHORT intervals, which is all reprojection needs -- tens of
    milliseconds between camera corrections. It does NOT integrate cleanly over long ones, and
    nothing here pretends otherwise: this is the fast half of a fast/slow pair, and the cameras
    supply the absolute reference. Used open-loop it will drift without bound.
    """

    def __init__(self, drift_warn_deg=5.0):
        self.drift_warn = float(drift_warn_deg)
        self.reset()

    def reset(self):
        self.angle = np.zeros(3)        # radians, x=pitch y=yaw z=roll
        self._t = None
        self.since_correction = 0.0

    def update(self, gyro_rad_s, t):
        g = np.asarray(gyro_rad_s, float)
        if self._t is None:
            self._t = float(t)
            return self.angle.copy()
        dt = max(float(t) - self._t, 0.0)
        self._t = float(t)
        self.angle = self.angle + g * dt
        self.since_correction += dt
        return self.angle.copy()

    def correct(self, angle_rad=None):
        """Absolute correction from the cameras, the slow half closing the loop."""
        if angle_rad is not None:
            self.angle = np.asarray(angle_rad, float).copy()
        self.since_correction = 0.0

    @property
    def drifting(self):
        """True once open-loop long enough that the estimate should not be trusted alone."""
        return self.since_correction > 1.0


def predict_rotation(gyro_rad_s, lookahead_s):
    """Angle the head will have rotated `lookahead_s` from now, at the current rate.

    This is the input predictor.LatencyCompensator wants. It replaces the image-motion velocity
    estimate with a direct measurement at 100-1000 Hz instead of 17.9 fps.
    """
    return np.asarray(gyro_rad_s, float) * float(lookahead_s)


def rad_to_display_px(angle_rad, display_fov_deg=50.0, display_px=1920):
    """Angle -> display pixels, tangent-correct (see geometry.py on why not a linear ratio)."""
    th = np.tan(np.radians(display_fov_deg) / 2.0)
    return np.tan(np.asarray(angle_rad, float)) / (2.0 * th) * display_px


# ---------------------------------------------------------------------------
#  Gyro -> image-plane velocity: the mapping MUST be measured, never assumed
# ---------------------------------------------------------------------------
MAP_PATH = "../data/imu_map.npz"


def measure_axis_map(world_left=3, world_right=2, seconds=20.0, verbose=True):
    """Regress world-dot image velocity against gyro to find the axis/sign mapping.

    WHY MEASURED. The gyro reports rate about the glasses' own axes; the predictor needs velocity
    in normalised IMAGE units. Which gyro axis drives u, which drives v, and with what SIGN, depends
    on how the IMU is mounted inside the glasses and how the world cameras are oriented -- and this
    rig has already been bitten by a flipped axis (rig.py +y UP vs image +y DOWN cost a 381 px
    bias) and by a reversed stereo pair. A guessed sign here would predict the overlay BACKWARDS,
    which looks exactly like the lag it is meant to cure.

    So: move your head for `seconds` while a static target is in view, and solve the 2x3 least
    squares  [du/dt, dv/dt] = M @ [wx, wy, wz]. The R^2 reported per row is the evidence that the
    mapping is real -- a low R^2 means the fit is noise and must NOT be used.
    """
    import cv2
    from cameras import Camera, ROLE_MODE
    from dot_detector import DotDetector
    det = DotDetector()
    L = Camera(world_left, *ROLE_MODE["worldL"], name="worldL")
    R = Camera(world_right, *ROLE_MODE["worldR"], name="worldR")
    imu = XrealIMU().open()

    # ON-SCREEN INSTRUCTIONS, because terminal output does not reach someone wearing the glasses.
    # This exact coordination failure wasted several hardware runs in one session: the operator
    # cannot read a console while the rig is on their face, so a run that needs them to DO
    # something must say so on the display. It also shows live gyro rate and dot lock, so a bad
    # run is obvious while it is happening rather than 20 s later.
    win = "IMU axis map, follow the instruction.  q aborts"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _hud(phase, remain, dps, locked, n):
        img = np.zeros((300, 900, 3), np.uint8)
        cv2.putText(img, phase, (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        cv2.putText(img, "%4.1f s left   (small moves - KEEP THE TARGET IN VIEW)" % max(remain, 0), (24, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        col = (0, 255, 0) if dps > 15 else (0, 165, 255) if dps > 5 else (0, 0, 255)
        cv2.putText(img, "head %5.1f deg/s %s" % (dps, "" if dps > 15 else "<- MOVE MORE"),
                    (24, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
        cv2.putText(img, "flow %s   samples %d" % ("OK" if locked else "no texture", n),
                    (24, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if locked else (0, 0, 255), 2)
        cv2.imshow(win, img)
        return cv2.waitKey(1) & 0xFF

    for c in range(3, 0, -1):                      # let them settle and read the first phase
        _hud("GET READY, look at the target", c, 0.0, False, 0)
        time.sleep(1.0)
    # SAMPLE ONLY ON A GENUINELY NEW FRAME.
    # Camera.read() happily returns the same buffered frame again, and dividing a ~zero position
    # change by a ~zero dt manufactures enormous fake velocities that swamp the regression. The
    # first attempt did exactly that and produced R^2 = 0.004 with a coefficient of 86 -- noise
    # wearing the shape of a fit. Require both a minimum interval AND a changed frame.
    MIN_DT = 0.035                        # ~ one frame at 17.9 fps, minus slack
    samples = []
    t0 = time.time()
    last_t = 0.0
    last_sig = None
    gyro_seen = []
    prev_gray = None
    prev_pts = None
    flow_pos = (0.0, 0.0)
    try:
        while time.time() - t0 < seconds:
            now = time.time()
            fL, fR = L.read(), R.read()
            batch = imu.drain()
            if batch:
                gyro_seen.extend(b[0] for b in batch)
            if fL is None or fR is None or not batch:
                continue
            sig = float(np.asarray(fL[::16, ::16], np.float32).sum())   # cheap frame fingerprint
            if sig == last_sig or now - last_t < MIN_DT:
                continue
            # GLOBAL OPTICAL FLOW, not the single dot.
            #
            # The dot is the fragile part of this measurement: it depends on one small target
            # winning against a whole room, and when it mis-detects it jumps to an unrelated object
            # whose motion has nothing to do with the head. Measured consequence -- across three
            # runs the fitted axis assignment FLIPPED between yaw and pitch with correlations of
            # 0.13, which is a fit converging on nothing.
            #
            # But head rotation moves the ENTIRE SCENE, so every pixel carries the signal. Median
            # sparse flow across a few hundred corners is enormously more robust than one blob, is
            # unaffected by the target leaving frame, and needs no detector at all.
            gcur = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY) if fL.ndim == 3 else fL
            gcur = cv2.resize(gcur, (320, 200))
            pl = None
            if prev_gray is not None and prev_pts is not None and len(prev_pts) > 20:
                nxt, st, _e = cv2.calcOpticalFlowPyrLK(prev_gray, gcur, prev_pts, None,
                                                       winSize=(21, 21), maxLevel=3)
                if nxt is not None and st is not None and int(st.sum()) > 20:
                    good_new = nxt[st.ravel() == 1].reshape(-1, 2)
                    good_old = prev_pts[st.ravel() == 1].reshape(-1, 2)
                    fl = np.median(good_new - good_old, axis=0)     # median = outlier-proof
                    # pixels -> normalised frame units of the DOWNSCALED image
                    pl = (float(fl[0]) / 320.0, float(fl[1]) / 200.0)
            prev_gray = gcur
            prev_pts = cv2.goodFeaturesToTrack(gcur, maxCorners=300, qualityLevel=0.01,
                                               minDistance=7)
            # REFRESH THE HUD EVERY FRAME, INCLUDING FAILURES. The first version only drew after a
            # SUCCESSFUL sample, so the instant the target left the frame the display froze on its
            # last good state -- the operator kept turning, saw nothing wrong, and the run returned
            # zero samples. A status display that stops updating exactly when things go wrong is
            # worse than none.
            el = now - t0
            phase = ("TURN LEFT <-> RIGHT" if el < seconds * 0.5 else "NOD UP <-> DOWN")
            dps = float(np.degrees(np.linalg.norm(np.mean([b[0] for b in batch], axis=0))))
            key = _hud(phase, seconds - el, dps, pl is not None, len(samples))
            if key in (ord("q"), 27):
                break
            if pl is None:
                continue
            g = np.mean([b[0] for b in batch], axis=0)     # mean rate over this frame interval
            flow_pos = (flow_pos[0] + pl[0], flow_pos[1] + pl[1])   # integrate flow into a track
            samples.append((now, flow_pos[0], flow_pos[1], g[0], g[1], g[2]))
            last_t, last_sig = now, sig
    finally:
        L.release(); R.release(); imu.close()
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
    # DID THE HEAD ACTUALLY MOVE? Without this, "you stood still" and "the fit is wrong" are
    # indistinguishable, and the previous run could not tell them apart.
    gs = np.array(gyro_seen) if gyro_seen else np.zeros((1, 3))
    peak_dps = float(np.degrees(np.linalg.norm(gs, axis=1).max()))
    med_dps = float(np.degrees(np.median(np.linalg.norm(gs, axis=1))))
    if verbose:
        print("\nhead motion seen: median %.1f deg/s, peak %.1f deg/s" % (med_dps, peak_dps))
        if peak_dps < 15:
            print("   !! BARELY ANY MOTION. The glasses were still (or not on a head). Nothing "
                  "\n      can be fitted from this, move noticeably, ~30 deg/s, both axes.")
    if len(samples) < 30:
        if verbose:
            print("only %d usable samples, need the target in view throughout" % len(samples))
        return None
    # REGRESS DISPLACEMENT AGAINST INTEGRATED GYRO, NOT VELOCITY AGAINST RATE.
    #
    # Differentiating the dot position is the obvious approach and it is the wrong one: at 17.9 fps
    # with ~0.004 detection noise, the velocity noise is 0.004/0.056 = 0.07 units/s, against a
    # typical signal of ~0.12 -- an SNR near 1.7, which is exactly the R^2 = 0.03-0.07 the first
    # attempts produced. The structure was already visibly correct (u driven by yaw, v by pitch);
    # it was drowning, not absent.
    #
    # So integrate instead. The gyro integrates cleanly over short spans, and comparing DISPLACEMENT
    # over a window against the INTEGRATED rate over the same window averages the position noise
    # down by ~sqrt(N) while the signal grows linearly with the window. Same physics, far better
    # conditioning -- and it never differentiates a noisy quantity.
    a = np.array(samples)
    t_s, uv_s, w_s = a[:, 0], a[:, 1:3], a[:, 3:6]
    WIN = 0.30                                             # s; long enough to average, short
    duv, w = [], []                                        # enough that gyro drift is irrelevant
    j = 0
    for i in range(len(t_s)):
        while j < len(t_s) and t_s[j] - t_s[i] < WIN:
            j += 1
        if j >= len(t_s):
            break
        span = t_s[j] - t_s[i]
        if span < WIN * 0.7:
            continue
        # trapezoidal integral of the gyro across the window = angle turned
        seg_t, seg_w = t_s[i:j + 1], w_s[i:j + 1]
        ang = np.trapz(seg_w, seg_t, axis=0)
        duv.append(uv_s[j] - uv_s[i])
        w.append(ang)
    if len(duv) < 30:
        if verbose:
            print("only %d windows, run longer or keep the dot in view" % len(duv))
        return None
    duv = np.array(duv)
    w = np.array(w)
    # Drop physically impossible image velocities before fitting. At a 70 deg FOV even a fast
    # 200 deg/s head turn moves the dot ~2.9 frame-units/s; anything far beyond that is a
    # mis-detection jumping between objects, and one such outlier dominates a least-squares fit.
    keep = np.linalg.norm(duv, axis=1) < 2.0        # displacement, in frame units, over one window
    duv, w = duv[keep], w[keep]
    if len(duv) < 30:
        if verbose:
            print("only %d samples survived outlier rejection" % len(duv))
        return None
    # CONSTRAINED FIT: the physics has FOUR unknowns, not six.
    #
    # A free 2x3 least squares has six free parameters and natural head motion couples the axes --
    # you cannot yaw without a little roll -- so it happily attributes u to roll, which is
    # physically wrong. Measured: it did exactly that, reporting a coefficient of -2.76 on az where
    # the whole plausible range is |1/FOV| = 0.82.
    #
    # What is actually unknown is only: WHICH gyro axis drives u, WHICH drives v, and the SIGN of
    # each. The magnitude follows from the world camera's field of view -- one radian of head
    # rotation moves the image by 1/FOV_rad frame-units, by definition of the projection. So pick
    # the axes by correlation, take the signs from them, and keep the theoretical scale. Four
    # parameters, all well determined, and the fitted-vs-theoretical scale ratio becomes a CHECK
    # rather than a free parameter: it should land near 1.0, and if it does not the mapping is
    # wrong in a way a free fit would have silently absorbed.
    import rig as _rig
    fov_rad = np.radians(_rig.WORLD_FOV)
    # PER-AXIS scale: the sensor is 16:10, so the VERTICAL field of view is not the horizontal
    # one. Using the horizontal figure for both predicted a v-scale ratio of 0.68 where 1.0 was
    # "correct"; the measured 0.73 matched the aspect-corrected prediction, not the naive one --
    # which is a second, independent confirmation that the mapping is real rather than fitted.
    fov_v_rad = 2.0 * np.arctan(np.tan(fov_rad / 2.0) * 10.0 / 16.0)
    theo_uv = (1.0 / fov_rad, 1.0 / fov_v_rad)
    theo = theo_uv[0]                        # reported headline value (horizontal)
    axis_names = ("wx(pitch)", "wy(yaw)", "wz(roll)")
    M = np.zeros((2, 3))
    picked, scales, r2 = [], [], []
    for row in (0, 1):
        cors = [abs(np.corrcoef(w[:, k], duv[:, row])[0, 1]) if w[:, k].std() > 1e-9 else 0.0
                for k in range(3)]
        k = int(np.argmax(cors))
        sgn = np.sign(np.corrcoef(w[:, k], duv[:, row])[0, 1]) or 1.0
        fitted = float(np.dot(w[:, k], duv[:, row]) / max(np.dot(w[:, k], w[:, k]), 1e-12))
        M[row, k] = sgn * theo_uv[row]
        pred = w[:, k] * M[row, k]
        ss = np.sum((duv[:, row] - pred) ** 2)
        st = max(np.sum((duv[:, row] - duv[:, row].mean()) ** 2), 1e-12)
        picked.append((k, cors[k]))
        scales.append(fitted / (sgn * theo_uv[row]))
        r2.append(1.0 - ss / st)
    r2 = np.array(r2)
    if verbose:
        print("\n%d windows (%.2f s each). Constrained fit, axis + sign measured, scale from FOV:"
              % (len(duv), WIN))
        print("   theoretical scale: u %.3f, v %.3f frame-units/rad (FOV %.0f deg h, %.1f deg v)"
              % (theo_uv[0], theo_uv[1], _rig.WORLD_FOV, np.degrees(fov_v_rad)))
        for row, lbl in ((0, "u"), (1, "v")):
            k, c = picked[row]
            print("   d%s <- %-10s  sign %+d   |corr| %.2f   fitted/theoretical %.2f   R^2 %.3f"
                  % (lbl, axis_names[k], int(np.sign(M[row, k])), c, scales[row], r2[row]))
        if min(r2) < 0.3:
            print("\n   !! R^2 IS LOW, do NOT use this. Either the head barely moved, or the dot"
                  "\n      was lost for much of the run.")
        elif not (0.4 < min(scales) and max(scales) < 2.5):
            print("\n   !! the fitted scale is far from the FOV prediction (%.2f, %.2f). The axis"
                  "\n      choice may be right but something else is off, check WORLD_FOV."
                  % tuple(scales))
    return {"M": M, "r2": np.array(r2), "n": len(duv)}


def save_map(res, path=MAP_PATH):
    np.savez(path, M=res["M"], r2=res["r2"], n=res["n"])
    return path


def load_map(path=MAP_PATH):
    """(M, r2) or (None, None). Refuses a low-R^2 map rather than letting it predict backwards."""
    try:
        d = np.load(path)
        M, r2 = d["M"], d["r2"]
        if float(np.min(r2)) < 0.3:
            return None, r2
        return M, r2
    except Exception:
        return None, None


class ImuVelocity:
    """Image-plane velocity from the gyro, at IMU rate rather than camera rate.

    This is what predictor.LatencyCompensator wants. Differentiating the dot's position gives the
    same quantity at 17.9 Hz and one frame late; the gyro gives it at ~1100 Hz and current.
    """

    def __init__(self, M=None, imu=None):
        self.M = M if M is not None else load_map()[0]
        self.imu = imu
        self.last = np.zeros(2)

    def available(self):
        return self.M is not None and self.imu is not None

    def velocity(self):
        """Latest image-plane [du/dt, dv/dt], or the last known value."""
        if not self.available():
            return self.last
        batch = self.imu.drain()
        if batch:
            g = batch[-1][0]
            self.last = self.M @ np.asarray(g, float)
        return self.last


def selftest():
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (", " + detail) if detail else ""))

    # --- integrator ---
    gi = GyroIntegrator()
    rate = np.array([0.0, np.radians(30.0), 0.0])       # 30 deg/s yaw
    t = 0.0
    gi.update(rate, t)
    for _ in range(100):                                 # 1.0 s at 100 Hz
        t += 0.01
        gi.update(rate, t)
    chk("integrating 30 deg/s for 1 s gives ~30 deg of yaw",
        abs(np.degrees(gi.angle[1]) - 30.0) < 0.5, "%.2f deg" % np.degrees(gi.angle[1]))
    chk("open-loop for >1 s is flagged as drifting", gi.drifting)
    gi.correct(np.zeros(3))
    chk("a camera correction clears both the angle and the drift flag",
        not gi.drifting and np.allclose(gi.angle, 0))

    # --- prediction, and the number that motivates the whole thing ---
    for deg_s in (30.0, 100.0):
        ang = predict_rotation([0, np.radians(deg_s), 0], 0.062)[1]
        px = rad_to_display_px(ang)
        expect = deg_s * 0.062
        chk("at %3d deg/s, 62 ms of lag = %.1f deg = %.0f px" % (deg_s, np.degrees(ang), px),
            abs(np.degrees(ang) - expect) < 0.01,
            "matches the measured lag budget" if deg_s == 30 else "")

    chk("zero rate predicts zero motion", abs(predict_rotation([0, 0, 0], 0.062)[1]) < 1e-12)
    chk("angle->px is tangent-correct, not linear",
        abs(rad_to_display_px(np.radians(25.0)) - 1920 / 2.0) < 1.0,
        "half-FOV maps to the frame edge: %.0f px" % rad_to_display_px(np.radians(25.0)))

    # --- DECODER, against synthetic records so it runs with no hardware ---
    rec = bytearray(REC_LEN)
    rec[0:4] = REC_MAGIC
    struct.pack_into("<3f", rec, OFF_GYRO, 0.1, -0.2, 0.3)
    struct.pack_into("<3f", rec, OFF_ACCEL, 0.0, -8.3, 5.2)
    d = decode(bytes(rec))
    chk("decodes gyro at 0x22", d is not None and np.allclose(d[0], [0.1, -0.2, 0.3]))
    chk("decodes accel at 0x2e", d is not None and np.allclose(d[1], [0.0, -8.3, 5.2]))
    chk("that accel is ~1 g, the check that validated the offsets",
        abs(np.linalg.norm(d[1]) - GRAVITY) / GRAVITY < 0.02,
        "%.3f vs %.2f m/s^2" % (np.linalg.norm(d[1]), GRAVITY))
    bad = bytearray(rec); bad[0:4] = b"\x00\x00\x00\x00"
    chk("a record without the magic is REFUSED, not parsed anyway", decode(bytes(bad)) is None)
    chk("a short record is refused", decode(bytes(rec[:40])) is None)

    # --- resync: a stream that starts mid-record must not silently decode garbage ---
    class _FakeSock:
        def __init__(self, data):
            self.data = data
        def recv(self, n):
            if not self.data:
                raise BlockingIOError()
            out, self.data = self.data[:n], self.data[n:]
            return out
        def close(self):
            pass
    imu = XrealIMU()
    imu.sock = _FakeSock(b"\xaa\xbb\xcc" + bytes(rec) * 3)     # 3 junk bytes first
    got = imu.drain()
    chk("resyncs on the magic after a partial/offset start", len(got) == 3,
        "recovered %d of 3 records" % len(got))
    chk("...and the resynced values are correct",
        len(got) == 3 and np.allclose(got[0][1], [0.0, -8.3, 5.2]))

    # --- the availability check must not crash when the device is absent/hidden ---
    devs = enumerate_xreal()
    chk("HID enumeration never raises (that path is a dead end but must not crash)",
        devs is None or isinstance(devs, list),
        "hidapi missing" if devs is None else "%d XREAL HID interface(s) visible" % len(devs))

    print("XREAL IMU OK ✅" if ok_all else "XREAL IMU FAILED ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XREAL One Pro internal IMU over USB HID")
    p.add_argument("--check", action="store_true", help="can this process see the device?")
    p.add_argument("--dump", action="store_true", help="raw HID reports (dead end, kept)")
    p.add_argument("--live", action="store_true", help="live gyro/accel over TCP")
    p.add_argument("--map", action="store_true",
                   help="measure the gyro->image axis mapping (move your head, target in view)")
    p.add_argument("--world-left", type=int, default=3)
    p.add_argument("--world-right", type=int, default=2)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.check:
        sys.exit(check())
    if a.map:
        res = measure_axis_map(a.world_left, a.world_right, a.seconds)
        if res and float(np.min(res["r2"])) >= 0.3:
            print("\nsaved -> %s" % save_map(res))
            sys.exit(0)
        print("\nNOT SAVED, the fit did not clear R^2 >= 0.3.")
        sys.exit(1)
    if a.live:
        with XrealIMU() as _imu:
            time.sleep(0.3)
            g, ac, n, t0 = [], [], 0, time.time()
            while time.time() - t0 < a.seconds:
                for gy, acc, _t in _imu.drain():
                    g.append(gy); ac.append(acc); n += 1
                time.sleep(0.005)
            g = np.array(g); ac = np.array(ac)
            el = time.time() - t0
            mag = np.linalg.norm(ac, axis=1)
            print("%d records in %.1fs -> %.0f Hz" % (n, el, n / el))
            print("accel |a| %.3f m/s^2 (gravity %.2f, %.2f%% off)"
                  % (mag.mean(), GRAVITY, 100 * abs(mag.mean() - GRAVITY) / GRAVITY))
            print("gyro  |w| %.4f rad/s = %.2f deg/s" %
                  (np.linalg.norm(g, axis=1).mean(),
                   np.degrees(np.linalg.norm(g, axis=1).mean())))
        sys.exit(0)
    if a.dump:
        sys.exit(dump(a.seconds))
    p.print_help()
