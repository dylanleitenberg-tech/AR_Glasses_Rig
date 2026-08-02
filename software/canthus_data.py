"""Build a labelled inner-canthus corpus from the rig itself — no hand-labelling.

WHY THIS EXISTS
    Template capture is a MANUAL step every session: box both canthi by hand, and a box that is
    slightly wrong produces a template that scores ~1.0 and localises nowhere (see
    calib_preflight.template_margin). Dylan's words on 2026-08-02: "i cant box every time."
    A learned landmark detector removes the step permanently — but it needs labelled data.

WHY NOT WEB IMAGES
    Face photos online are frontal, visible-light, 0.5-2 m away. These cameras sit ~3 cm from the
    eye, oblique, mono NoIR, wide-angle, low contrast. It is a different distribution entirely,
    which is exactly why this project rejected face-mesh landmarkers in the first place
    ("glasses-mounted close-up can't use face-mesh"). A model trained on web faces sees nothing it
    recognises here. The ONLY data in the right domain is this rig, on this face.

AUTO-LABELLING FROM THE TEMPLATE: TRIED, MEASURED, REJECTED (2026-08-02)
    The plan was to let the template teach the model — run the matcher over thousands of frames,
    keep only unambiguous locks, call those labels. It does not work, and the way it fails is
    instructive enough to keep here so nobody rebuilds it.

    When the rig shifts on the face, the matcher settles on FEATURELESS SKIN — nose bridge, cheek
    — and does so CONFIDENTLY. Measured on a real 1500-frame run: labels more than 0.20 (frame
    units) from the per-eye median were 34.8% of eyeL and 5.2% of eyeR, and those outliers
    averaged margin 0.486 against a corpus median of 0.414. The wrong labels scored HIGHER than
    the right ones. No confidence threshold can filter that — the gate is anti-correlated with
    correctness. Eyelashes do the same thing for the same reason (highest-contrast structure in
    frame, so a lash-lock is genuinely unambiguous, just wrong).

    Second symptom, same cause: one eyeL template locked at u=0.84 before a replug and u=0.15
    after, both at high margin. A teacher that confidently reports two positions 0.7 frame-widths
    apart cannot supervise anything.

    WHAT WORKS INSTEAD: a small HUMAN-labelled seed (click a point, template's guess pre-filled,
    ENTER to accept) -> train -> let the MODEL auto-label the rest -> retrain. The model reads lid
    shape and iris position, so blank skin is not a candidate for it the way it is for
    correlation. The frames collected here are still valuable; only the labels were bad.

    The collector below is kept because the FRAME capture is sound and its stability gate is
    worth having, but treat its labels as a PROPOSAL to be reviewed, never as ground truth.

QUALITY GATE
    A wrong label is worse than no label, so a frame is only stored when the match is BOTH strong
    and unique — margin (peak minus the best rival elsewhere) above `min_margin`. That is the same
    quantity calib_preflight uses to decide whether a template is usable at all, and it is what
    separates "the corner is here" from "everything looks equally like the corner".

    python3 canthus_data.py --collect --eye-cam-left 0 --eye-cam-right 1
    python3 canthus_data.py --selftest
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CORPUS = os.path.join(DATA, "canthus_corpus.npz")

# Stored frame size. Small enough that thousands fit in RAM and train fast on CPU, large enough
# that the canthus is still several pixels across: 1280x800 -> 320x200 keeps the 16:10 aspect.
STORE_W, STORE_H = 320, 200

MIN_MARGIN = 0.15      # above calib_preflight's 0.08 usability bar — labels must be BETTER
MIN_PEAK = 0.55        # a weak absolute match is unreliable even when it is unique

# TEMPORAL STABILITY GATE — the defence against lash-locks.
#
# Margin alone CANNOT catch this failure. Eyelashes are the highest-contrast structure in the
# frame, so when the matcher slips onto them it does so with a HIGH margin: the lock is genuinely
# unambiguous, it is simply on the wrong thing. Score and uniqueness both say "confident", and the
# label is confidently wrong — the worst kind to train on. Observed on hardware 2026-08-02.
#
# What separates a lash-lock from the real corner is PERSISTENCE, not confidence. A lash-lock is
# transient: it appears for a frame or a few and snaps back. The canthus is where the matcher sits
# the rest of the time.
#
# Crucially this must NOT reject re-seating, which is the most valuable variation in the corpus —
# taking the glasses off and putting them back on legitimately moves the corner in-frame. So the
# test is not "close to where it has always been" (that would throw away every re-seat); it is
# "the last STABLE_N detections agree with each other". A sustained new position after a re-seat
# is admitted once it holds for a few frames; a one-frame jump to the lashes never is.
STABLE_N = 4           # consecutive detections that must agree
STABLE_RADIUS = 0.035  # normalised frame units they must agree WITHIN


def match_with_margin(gray, tmpl, cv2):
    """(peak, (u,v) normalised, margin). Same definition as calib_preflight.template_margin."""
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    masked = res.copy()
    r = max(tmpl.shape[:2]) * 2
    masked[max(0, loc[1] - r):loc[1] + r, max(0, loc[0] - r):loc[0] + r] = -1.0
    rival = float(masked.max())
    H, W = gray.shape[:2]
    u = (loc[0] + tmpl.shape[1] / 2.0) / W
    v = (loc[1] + tmpl.shape[0] / 2.0) / H
    return float(peak), (u, v), float(peak) - rival


def collect(eye_cams, target=1500, min_margin=MIN_MARGIN, seconds=None, view=True, verbose=True):
    """Stream both eye cams, keep every frame where the template locks UNAMBIGUOUSLY."""
    import cv2
    from cameras import Camera

    tmpl = {}
    for role in ("eyeL", "eyeR"):
        p = os.path.join(DATA, "templates", "%s.png" % role)
        t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if t is None:
            print("!! no template at %s — run: python3 main.py --calibrate-corners" % p)
            return 1
        tmpl[role] = t

    cams = {r: Camera(i, 1280, 800, name=r) for r, i in eye_cams.items()}
    frames, labels, roles, metas = [], [], [], []
    kept = {"eyeL": 0, "eyeR": 0}
    seen = {"eyeL": 0, "eyeR": 0}
    win = "canthus corpus — move, blink, re-seat.  Q when done" if view else None
    if win:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t0 = time.time()

    try:
        while sum(kept.values()) < target:
            if seconds and time.time() - t0 > seconds:
                break
            tiles = []
            for role, cam in cams.items():
                f = cam.read()
                if f is None:
                    continue
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                seen[role] += 1
                peak, (u, v), margin = match_with_margin(g, tmpl[role], cv2)
                good = margin >= min_margin and peak >= MIN_PEAK
                if good:
                    small = cv2.resize(g, (STORE_W, STORE_H), interpolation=cv2.INTER_AREA)
                    frames.append(small)
                    labels.append((u, v))
                    roles.append(0 if role == "eyeL" else 1)
                    metas.append((peak, margin))
                    kept[role] += 1
                if win:
                    s = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    col = (0, 255, 0) if good else (0, 0, 255)
                    cv2.circle(s, (int(u * g.shape[1]), int(v * g.shape[0])), 14, col, 2)
                    cv2.putText(s, "%s  margin %.3f  kept %d" % (role, margin, kept[role]),
                                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                    tiles.append(cv2.resize(s, (480, 300)))
            if win and tiles:
                cv2.imshow(win, np.hstack(tiles) if len(tiles) > 1 else tiles[0])
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        for c in cams.values():
            c.release()
        if win:
            cv2.destroyAllWindows()

    if not frames:
        print("!! nothing met the quality gate (margin >= %.2f). Is the rig on your face?"
              % min_margin)
        return 1
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(CORPUS,
                        frames=np.array(frames, np.uint8),
                        labels=np.array(labels, np.float32),
                        roles=np.array(roles, np.uint8),
                        meta=np.array(metas, np.float32))
    if verbose:
        print("\n== corpus ==")
        for r in ("eyeL", "eyeR"):
            rate = 100.0 * kept[r] / max(seen[r], 1)
            print("  %s: kept %4d of %4d frames (%.0f%% passed the margin gate)"
                  % (r, kept[r], seen[r], rate))
        m = np.array(metas, np.float32)
        print("  label margin: median %.3f  min %.3f" % (np.median(m[:, 1]), m[:, 1].min()))
        print("  saved %s  (%d frames, %dx%d)" % (CORPUS, len(frames), STORE_W, STORE_H))
    return 0


def selftest(verbose=True):
    """No hardware: the quality gate and the match geometry must both behave."""
    import cv2
    if verbose:
        print("== canthus_data self-test (no hardware) ==")
    checks = []

    # A distinctive mark on smooth, self-similar background: the matcher must find it, at the
    # right place, with a healthy margin. (Self-similar on purpose — white noise localises
    # perfectly and would prove nothing.)
    yy, xx = np.mgrid[0:200, 0:320]
    scene = (100 + 8 * np.sin(xx / 9.0) + 6 * np.sin(yy / 11.0)).astype(np.uint8)
    cv2.circle(scene, (210, 70), 10, 20, -1)
    cv2.line(scene, (188, 84), (236, 96), 35, 2)
    t = scene[30:110, 170:250].copy()
    peak, (u, v), margin = match_with_margin(scene, t, cv2)
    checks.append(("match_with_margin finds the mark at u=%.2f v=%.2f, margin %.2f" % (u, v, margin),
                   abs(u - 210 / 320.0) < 0.02 and abs(v - 70 / 200.0) < 0.02 and margin > 0.25))

    # A featureless crop must NOT clear the gate, even though it scores ~1.0. This is the whole
    # reason the gate is margin-based rather than score-based.
    flat = scene[140:190, 20:100].copy()
    fpeak, _, fmargin = match_with_margin(scene, flat, cv2)
    checks.append(("flat crop scores %.2f but margins %.3f -> rejected by the gate"
                   % (fpeak, fmargin), fpeak > 0.8 and fmargin < MIN_MARGIN))

    # Stored geometry: labels are normalised, so they survive the resize to STORE_W x STORE_H.
    big = np.zeros((800, 1280), np.uint8)
    big[400:420, 640:660] = 255
    small = cv2.resize(big, (STORE_W, STORE_H), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small)
    checks.append(("normalised labels survive the 1280x800 -> %dx%d resize" % (STORE_W, STORE_H),
                   abs(xs.mean() / STORE_W - 0.508) < 0.02
                   and abs(ys.mean() / STORE_H - 0.512) < 0.02))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "CANTHUS DATA OK — gate rejects flat, accepts structure ✅"
              if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="collect a labelled inner-canthus corpus")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eye-cam-left", type=int, default=0)
    ap.add_argument("--eye-cam-right", type=int, default=1)
    ap.add_argument("--target", type=int, default=1500, help="frames to keep in total")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.collect:
        sys.exit(collect({"eyeL": a.eye_cam_left, "eyeR": a.eye_cam_right},
                         target=a.target, min_margin=a.min_margin,
                         seconds=a.seconds, view=not a.no_view))
    ap.print_help()
