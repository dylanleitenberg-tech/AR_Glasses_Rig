"""Worn-rig eye-camera diagnostic: is the eye cam usable, and does the model lock?

WHY THIS EXISTS. "The canthus isn't being picked up" has at least four causes that look identical
from the outside -- the rig is off the face, the sensor is blown out by daylight, the lens is out
of focus, or the model's plausibility band is refusing a good landmark. Each has a different fix
and three of them are physical. This prints the one number that separates each, so a session stops
guessing.

READ THE CAVEATS, THEY ARE THE POINT:

  * MEASURE ONLY WHILE THE RIG IS WORN. On a desk the cams stare at the room and every number here
    is meaningless. Settled measurement 2026-08-02: the same two cams read 40.9%/20.2% of pixels
    saturated on a table and 5.3%/3.3% once worn. That ratio is the single fastest way to tell
    whether the glasses are actually on a face, which is why it is the first thing printed.

  * WHOLE-FRAME LAPLACIAN IS NOT A FOCUS METRIC HERE. An eye socket is mostly smooth skin; a
    living room is wall-to-wall edges. A perfectly focused eye cam reads ~15 while a world cam
    reads ~800. It is printed for continuity with bank_bringup, NOT as a verdict -- judge focus by
    whether lashes and iris texture resolve in the saved PNG.

  * THERE IS NO SOFTWARE EXPOSURE CONTROL ON MACOS. AVFoundation rejects CAP_PROP_EXPOSURE,
    AUTO_EXPOSURE, GAIN and BRIGHTNESS on these UVC devices -- every set() returns False. If this
    reports blowout the only levers are physical: kill daylight (NoIR sensors soak up IR, so
    sunlight hits them far harder than LED light).

Usage:  python3 eye_check.py --eyeL 0 --eyeR 1 [--seconds 10] [--delay 5]
"""
import argparse
import sys
import time

import numpy as np

# Settled reference signatures, measured 2026-08-02 on this rig (see HANDOFF.md).
SAT_WORN = {"eyeL": 5.3, "eyeR": 3.3}
SAT_DESK = {"eyeL": 40.9, "eyeR": 20.2}
SAT_BLOWN = 15.0        # above this, the image is too clipped for the model to read texture


def _verdict(role, sat, states, in_band, n):
    """Turn the raw numbers into the one sentence that names the next action."""
    out = []
    worn, desk = SAT_WORN[role], SAT_DESK[role]
    if sat >= SAT_BLOWN:
        near = "desk/blown-out (%.1f%%)" % desk if abs(sat - desk) < abs(sat - worn) else "blown out"
        out.append("BLOWN OUT — %.1f%% saturated, matches the %s signature, not the worn one "
                   "(%.1f%%). Either the rig is not on your face or daylight is hitting the "
                   "sensor. NO software fix exists (AVFoundation rejects exposure control): "
                   "kill daylight, close blinds, work under LED." % (sat, near, worn))
    elif sat > worn * 3:
        out.append("BRIGHT — %.1f%% saturated vs %.1f%% expected worn. Usable but hot; "
                   "reduce ambient light before trusting a calibration." % (sat, worn))
    else:
        out.append("EXPOSURE OK — %.1f%% saturated, consistent with the worn signature (%.1f%%)."
                   % (sat, worn))

    ok = states.get("ok", 0)
    if n and ok == 0:
        if in_band == 0:
            out.append("MODEL: never in band. The landmark the model reports is outside rig.py's "
                       "plausibility band for this camera every frame — this is the band or the "
                       "framing, not a flaky detection.")
        else:
            out.append("MODEL: in band %d%% of frames but never reached state 'ok' — the JUMP gate "
                       "is holding it. Hold still, or the landmark is oscillating." % (100 * in_band // n))
    elif n:
        out.append("MODEL: locked on %d%% of frames (in band %d%%)."
                   % (100 * ok // n, 100 * in_band // n))
    return out


def run(idx_l, idx_r, seconds=10.0, delay=0.0, save_dir="."):
    import cv2
    from canthus_net import CanthusTracker, CanthusNet
    try:
        from canthus_auto import build_prior
        prior = build_prior()
    except Exception:
        prior = None

    if delay > 0:
        print("Settle the rig on your face — starting in %d s..." % delay)
        for k in range(int(delay), 0, -1):
            print("  %d" % k, end="\r", flush=True)
            time.sleep(1)
        print("   ")

    net = CanthusNet()
    pad = 0.08
    lo = hi = None
    if prior is not None:
        lo, hi = prior["u_lo"] - pad, prior["u_hi"] + pad
        print("plausibility gate (mirrored-u): [%.3f, %.3f]\n" % (lo, hi))

    results = {}
    for idx, role, mir in ((idx_l, "eyeL", True), (idx_r, "eyeR", False)):
        cap = cv2.VideoCapture(idx, getattr(cv2, "CAP_AVFOUNDATION", 0))
        # 640x480 matches every other tool in the repo. The net resizes to a fixed 160x100, so the
        # 16:10-vs-4:3 stretch cancels in normalised coordinates -- measured, du/dv both 0.000.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("%s: COULD NOT OPEN index %d — is another tool (rig_view, preflight) still "
                  "running? On macOS only ONE process may hold a camera." % (role, idx))
            continue
        trk = CanthusTracker(mirrored=mir)
        sats, states, uus, last = [], {}, [], None
        t_end = time.time() + seconds
        while time.time() < t_end:
            ok, f = cap.read()
            if not ok or f is None:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            last = g
            sats.append(float((g > 250).mean() * 100))
            u, v, cp, sharp = net.predict(g)
            uu = (1.0 - u) if mir else u
            uus.append(uu)
            uv, score = trk.track(f)
            st = "lost" if uv is None else ("ok" if score >= 1.0 else "held")
            states[st] = states.get(st, 0) + 1
        cap.release()
        if not sats:
            print("%s: no frames captured." % role)
            continue

        n = len(sats)
        sat = float(np.median(sats))
        uus = np.array(uus)
        in_band = int(((uus >= lo) & (uus <= hi)).sum()) if lo is not None else 0
        lap = float(cv2.Laplacian(last, cv2.CV_64F).var())
        path = "%s/%s_worn.png" % (save_dir.rstrip("/"), role)
        cv2.imwrite(path, last)

        print("== %s (cam %d, %d frames) ==" % (role, idx, n))
        print("  saturated   %.1f%%   (worn %.1f%% | desk %.1f%%)"
              % (sat, SAT_WORN[role], SAT_DESK[role]))
        print("  mean level  %.0f" % last.mean())
        print("  laplacian   %.0f   (NOT a focus verdict for an eye cam — look at the PNG)" % lap)
        print("  model u     median %.3f as the gate sees it   (band [%s, %s])"
              % (np.median(uus), "%.3f" % lo if lo else "?", "%.3f" % hi if hi else "?"))
        print("  tracker     %s" % (states or "{}"))
        print("  frame saved %s" % path)
        for line in _verdict(role, sat, states, in_band, n):
            print("  -> %s" % line)
        print()
        results[role] = dict(sat=sat, states=states, in_band=in_band, n=n,
                             u_med=float(np.median(uus)))
    return results


def selftest():
    """Pin the verdict logic against the two signatures it exists to tell apart."""
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    # Known-bad input first: the desk signature must NOT read as usable.
    v = " ".join(_verdict("eyeL", 40.9, {"lost": 30}, 0, 30))
    chk("desk signature (40.9%) is called BLOWN OUT", "BLOWN OUT" in v)
    chk("desk signature does not claim exposure is OK", "EXPOSURE OK" not in v)
    v = " ".join(_verdict("eyeL", 5.3, {"ok": 30}, 30, 30))
    chk("worn signature (5.3%) is called OK", "EXPOSURE OK" in v)
    chk("worn+locked does not warn", "BLOWN OUT" not in v)
    # A locked model on a blown-out frame must still flag the exposure, not be excused by the lock.
    v = " ".join(_verdict("eyeR", 20.2, {"ok": 30}, 30, 30))
    chk("blown-out but locked STILL reports blowout", "BLOWN OUT" in v)
    # Never-in-band vs jump-gate must be distinguishable, they have different fixes.
    v = " ".join(_verdict("eyeL", 5.3, {"lost": 30}, 0, 30))
    chk("never-in-band names the band, not the jump gate", "never in band" in v)
    v = " ".join(_verdict("eyeL", 5.3, {"held": 30}, 30, 30))
    chk("in-band-but-held names the JUMP gate", "JUMP gate" in v)
    print("EYE CHECK OK ✅" if ok else "EYE CHECK FAILED ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="worn-rig eye-camera diagnostic")
    p.add_argument("--eyeL", type=int, default=0)
    p.add_argument("--eyeR", type=int, default=1)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--save-dir", default=".")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    sys.exit(selftest() if a.selftest else (run(a.eyeL, a.eyeR, a.seconds, a.delay,
                                                a.save_dir) and 0))
