"""Worn-rig eye-camera diagnostic: is the eye cam usable, and does the model lock?

WHY THIS EXISTS. "The canthus isn't being picked up" has at least four causes that look identical
from the outside -- the rig is off the face, the sensor is blown out by daylight, the lens is out
of focus, or the model's plausibility band is refusing a good landmark. Each has a different fix
and three of them are physical. This prints the one number that separates each, so a session stops
guessing.

READ THE CAVEATS, THEY ARE THE POINT:

  * MEASURE ONLY WHILE THE RIG IS WORN. On a desk the cams stare at the room and every number here
    is meaningless. Worn-vs-not is decided by smooth_frac() -- the share of the frame that is flat
    skin -- NOT by the saturated-pixel ratio. Saturation was the original test and it was WRONG:
    it called a dim room "consistent with the worn signature" at 0.2% saturated. Read
    smooth_frac's docstring before trusting anything here. When the rig is not on a face this
    tool now refuses to report exposure or model verdicts at all, because numbers about the
    furniture are worse than no numbers.

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


SMOOTH_WORN_MIN = 84.0   # % of frame in flat regions; see smooth_frac() for why this and not sat


def smooth_frac(gray, cv2):
    """% of pixels sitting in a locally-flat region. THE worn-vs-not discriminator.

    SATURATION DOES NOT WORK FOR THIS AND I SHIPPED IT ANYWAY -- read this before trusting any
    number above. The first version of this file asserted that the saturated-pixel ratio was "the
    single fastest way to tell whether the glasses are actually on a face". It is not. On
    2026-08-03 it reported "EXPOSURE OK, consistent with the worn signature" about a camera
    pointed at a wall with a picture frame on it: the room happened to be dim, so it read 0.2%
    saturated, well inside the worn band. A metric that says "on a face" about a photograph of a
    wall is the ninth-instance pattern all over again.

    What actually separates the two is scene STRUCTURE, and it follows from the optics: the M12
    lenses are focused for a canthus at ~3 cm, so when worn the frame is dominated by large flat
    expanses of skin, while a room 3 m away is edges and objects everywhere. Measured over six
    labelled frames from this rig:

        WORN  92.1  90.7        ROOM  83.2  81.5        DESK  77.2  77.3

    Separated by every threshold in 84..90; 87 is the midpoint of the gap. CALIBRATED ON SIX
    FRAMES FROM ONE RIG -- treat it as a screen that catches the gross case, not proof. The
    confirmation is still looking at the saved PNG, which is why one is always written.
    """
    g = gray.astype(np.float32)
    local = cv2.blur((g - cv2.blur(g, (9, 9))) ** 2, (9, 9)) ** 0.5
    return float((local < 3.0).mean() * 100)


def _verdict(role, sat, states, in_band, n, smooth=None):
    """Turn the raw numbers into the one sentence that names the next action."""
    out = []
    worn, desk = SAT_WORN[role], SAT_DESK[role]
    # NOT-WORN SHORT-CIRCUITS EVERYTHING. Reporting exposure and model verdicts on a picture of a
    # room is worse than reporting nothing: it sends the session off adjusting hardware against
    # numbers that describe furniture.
    if smooth is not None and smooth < SMOOTH_WORN_MIN:
        return ["NOT ON A FACE — only %.0f%% of the frame is flat skin (worn reads >%.0f%%, a room "
                "reads ~77-83%%). Every other number below is about the room, not your eye. Put "
                "the rig on and re-run; look at the saved PNG to confirm."
                % (smooth, SMOOTH_WORN_MIN)]
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
        # 640x400, NOT 640x480. MEASURED 2026-08-03: asking these OV9281 cams for 640x480 returns
        # a CROP of the 1280x800 sensor at scale 1.00 -- eyeL's 640x480 covers only x[288..928],
        # y[50..530], about 30% of the sensor area and a materially narrower field of view.
        # 640x400 is a true 2x DOWNSCALE that keeps the FULL frame (full-frame correlation 0.981
        # vs a best crop match of 0.725). It is also the sensor's native 16:10 and needs LESS
        # bandwidth than 640x480. Training frames are full-FOV, so inference must be too.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
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
        smooth = smooth_frac(last, cv2)
        path = "%s/%s_worn.png" % (save_dir.rstrip("/"), role)
        cv2.imwrite(path, last)

        print("== %s (cam %d, %d frames) ==" % (role, idx, n))
        print("  saturated   %.1f%%   (worn %.1f%% | desk %.1f%%)"
              % (sat, SAT_WORN[role], SAT_DESK[role]))
        print("  mean level  %.0f" % last.mean())
        print("  laplacian   %.0f   (NOT a focus verdict for an eye cam — look at the PNG)" % lap)
        print("  model u     median %.3f as the gate sees it   (band [%s, %s])"
              % (np.median(uus), "%.3f" % lo if lo else "?", "%.3f" % hi if hi else "?"))
        print("  flat-skin   %.0f%%   (worn >%.0f%% | room ~77-83%%)" % (smooth, SMOOTH_WORN_MIN))
        print("  tracker     %s" % (states or "{}"))
        print("  frame saved %s" % path)
        for line in _verdict(role, sat, states, in_band, n, smooth):
            print("  -> %s" % line)
        print()
        results[role] = dict(sat=sat, states=states, in_band=in_band, n=n,
                             u_med=float(np.median(uus)))
    return results


def live(idx_l, idx_r):
    """Live seating aid: adjust the brow clamp while WATCHING the lock, instead of iterating blind.

    WHY LIVE AND NOT A SIGN IN A DOC. The direction the carrier must move to bring the eye up in
    frame depends on a coordinate convention this project has got backwards before -- the reversed
    world pair and the mirrored-u gate were both sign errors that looked identical to a real
    effect from the outside. Rather than assert a sign, this shows the number moving as you move
    the clamp: loosen the M3 thumbscrews, shift the carrier a hair, watch LOCK%. If it drops, go
    the other way. Two runs settles what a paragraph of geometry might get wrong.

    Draws, per eye: the model's landmark (RED), the allowed band (GREEN), and a rolling lock rate.
    Target is the whole band, but aim for the MIDDLE of it -- the edge is where eyeR already sits
    and why it drops frames intermittently.
    """
    import cv2
    from canthus_net import CanthusTracker, CanthusNet
    from canthus_auto import build_prior
    net, prior, pad = CanthusNet(), build_prior(), 0.08
    lo, hi = prior["u_lo"] - pad, prior["u_hi"] + pad
    caps, trks, hist = {}, {}, {}
    for idx, role, mir in ((idx_l, "eyeL", True), (idx_r, "eyeR", False)):
        c = cv2.VideoCapture(idx, getattr(cv2, "CAP_AVFOUNDATION", 0))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not c.isOpened():
            print("%s: could not open index %d — is rig_view or preflight still running? "
                  "On macOS only ONE process may hold a camera." % (role, idx))
            return 1
        caps[role], trks[role], hist[role] = c, CanthusTracker(mirrored=mir), []
    win = "seat the rig — adjust the brow clamp until both read LOCK.  q quits"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Adjust the brow clamp and watch LOCK%. q quits.")
    while True:
        tiles = []
        for role, mir in (("eyeL", True), ("eyeR", False)):
            ok, f = caps[role].read()
            if not ok or f is None:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            u, v, cp, sh = net.predict(g)
            uu = (1.0 - u) if mir else u
            uv, score = trks[role].track(f)
            st = "LOST" if uv is None else ("LOCK" if score >= 1.0 else "held")
            h = hist[role]
            h.append(st == "LOCK")
            if len(h) > 60:
                h.pop(0)
            rate = 100.0 * sum(h) / len(h)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            H, W = vis.shape[:2]
            bl, bh = ((1 - hi), (1 - lo)) if mir else (lo, hi)
            cv2.rectangle(vis, (int(bl * W), 0), (int(bh * W), H), (0, 220, 0), 3)
            col = (0, 255, 0) if st == "LOCK" else ((0, 200, 255) if st == "held" else (0, 0, 255))
            cv2.drawMarker(vis, (int(u * W), int(v * H)), col, cv2.MARKER_CROSS, 60, 3)
            cv2.circle(vis, (int(u * W), int(v * H)), 30, col, 2)
            cv2.putText(vis, "%s  %s  lock %3.0f%%" % (role, st, rate), (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            cv2.putText(vis, "u %.3f (gate %.3f)  v %.3f" % (u, uu, v), (10, H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            tiles.append(cv2.resize(vis, (640, 480)))
        if tiles:
            cv2.imshow(win, np.hstack(tiles) if len(tiles) == 2 else tiles[0])
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break
    for c in caps.values():
        c.release()
    cv2.destroyAllWindows()
    return 0


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
    # THE FRAME THAT FOOLED THE OLD CHECK: a dim room read 0.2% saturated -- inside the worn
    # band -- and was reported as "EXPOSURE OK". Flat-skin must veto it regardless of exposure.
    v = " ".join(_verdict("eyeR", 0.2, {"ok": 30}, 30, 30, smooth=81.5))
    chk("dim ROOM (0.2% sat, looks worn by exposure) is caught as NOT ON A FACE",
        "NOT ON A FACE" in v)
    chk("not-on-a-face suppresses the exposure verdict", "EXPOSURE OK" not in v)
    chk("not-on-a-face suppresses the model verdict", "MODEL" not in v)
    chk("genuinely worn (92.1%) is NOT vetoed",
        "NOT ON A FACE" not in " ".join(_verdict("eyeL", 1.6, {"ok": 30}, 30, 30, smooth=92.1)))
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
    p.add_argument("--live", action="store_true",
                   help="live seating aid: adjust the brow clamp while watching the lock rate")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.live:
        sys.exit(live(a.eyeL, a.eyeR))
    run(a.eyeL, a.eyeR, a.seconds, a.delay, a.save_dir)
    sys.exit(0)
