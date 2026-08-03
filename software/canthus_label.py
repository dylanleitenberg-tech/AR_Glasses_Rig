"""Hand-label inner canthi on the collected corpus — the SEED set the model learns from.

WHY BY HAND AT ALL
    Template auto-labelling was tried and failed (see canthus_data): when the rig shifts, the
    matcher locks on featureless skin CONFIDENTLY, and those wrong labels scored HIGHER margin
    than the right ones, so no threshold filters them. A learned model needs ground truth that is
    actually true, and for a few hundred frames the cheapest reliable source is a human.

    This is a ONE-TIME cost, not the per-session boxing Dylan objected to. Label once, train once,
    and the model does every session afterwards.

MADE FAST ON PURPOSE
    Most of the template's proposals are right; it is the minority that are catastrophically
    wrong. So the tool PRE-FILLS the proposal and asks only for a verdict:

        ENTER / SPACE   accept the proposal as-is        (one key, the common case)
        click           place the true point yourself    (only when the proposal is wrong)
        N               skip this frame (blink, eye out of view, unusable)
        U               undo the previous frame
        Q               save and quit

    Frames are shown in a deterministic shuffled order so the seed set spans the whole session —
    consecutive frames are near-duplicates and labelling 300 in a row would teach almost nothing.

    python3 canthus_label.py --label --n 300
    python3 canthus_label.py --selftest
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CORPUS = os.path.join(DATA, "canthus_corpus.npz")
SEED = os.path.join(DATA, "canthus_seed.npz")

ZOOM = 3           # display scale: 320x200 -> 960x600, enough to click a canthus precisely


def sample_order(n_total, n_want, seed=20260802):
    """Deterministic spread across the corpus. Consecutive frames are near-duplicates, so a
    contiguous block would label the same instant 300 times; a fixed-seed shuffle spans the whole
    session (every seating, blink and lighting) while staying reproducible run to run."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return idx[:min(n_want, n_total)]


def seed_pass(n_want=30, corpus=CORPUS, out=SEED, zoom=None, verbose=True):
    """Place the TEAR DUCT on ~30 frames. This is the ground truth the whole system lacks.

    WHY THIS EXISTS AND WHY IT IS SHORT
        The template's centre sits on the eyelid, not the tear duct, and the template is the only
        thing in the pipeline that encodes what a canthus IS. So every estimator faithfully
        reproduces that error, and two independent estimators simply agree on the lid (measured:
        A/B separation 0.029, both on the lid margin). No amount of estimator independence repairs
        a wrong definition — something has to show the system the actual landmark once.

        Thirty is enough to define it. The model generalises from there and the existing machinery
        (mount band, anatomical prior, A/B agreement, closed-eye skip) filters its pseudo-labels
        across the remaining ~1470 frames.

    FULL FRAME, NEVER CROPPED — Dylan's requirement. A crop hides the surrounding anatomy that
    tells you which corner is which, and the whole failure here was mistaking one part of the eye
    for another. You see the entire frame, scaled up; the marker goes exactly where you click.

        CLICK   place / move the marker      ENTER  accept, eye OPEN
        C       accept, eye CLOSED                   U      undo the previous
        N       skip this frame              Q      save and quit

    CLOSED FRAMES STILL GET A POSITION. The inner canthus is where the lids MEET, so it stays
    visible through a blink — closed-ness is a separate property of the frame, not a reason to
    discard the landmark. That is why the model has two heads rather than a filter, and it is why
    the seed must contain BOTH states: a head cannot learn a class it has never seen.
    """
    import cv2
    if not os.path.exists(corpus):
        print("!! no corpus at %s" % corpus)
        return 1
    d = np.load(corpus)
    F, R = d["frames"], d["roles"]
    # Target ~2560 px wide, not 1280.
    #
    # Click precision depends on how big the landmark is ON SCREEN, not on the stored resolution.
    # Showing a native 1280x800 frame at 1:1 makes the eye HALF the on-screen size of the old
    # 640x400 corpus displayed at 2x -- and the labels measured it: archived 640 clicks had
    # u std 0.007, native-at-1:1 clicks 0.022, a 3x regression from the "higher resolution" corpus.
    # Dylan's screen is 3072x1920, so 2x of native (2560x1600) fits with room to spare and gives
    # both the full uncropped frame and a large target.
    if zoom is None:
        zoom = max(1, int(round(2560.0 / F.shape[2])))

    # NO openness pre-filter, deliberately.
    #
    # eye_openness (pupil ellipse area) does NOT reliably separate open from closed on real
    # frames: a selection ranked "most open" came back about half closed by Dylan's eye. Ranking
    # the seed by a broken metric would skew which frames the model ever sees AND teach the
    # eye-state head from a label source we know is wrong. So take an unbiased deterministic
    # spread across the corpus and let the human's ENTER/C calls define both classes — which is
    # the whole point of a seed set.
    #
    # SELECT FOR GAZE DIVERSITY, not at random.
    #
    # Dylan on hardware: "when i look off screen, guess goes to my eye ball." That is the model
    # having learned a SHORTCUT -- with few training frames the iris sat in a near-constant
    # position relative to the canthus, so "offset from the big dark circle" fitted the data as
    # well as the real landmark did. At extreme gaze the shortcut breaks and the prediction
    # follows the eyeball. It is the same pupil-anchoring trap that broke the classical pipeline.
    #
    # The cure is data where iris position and canthus position DECORRELATE: frames spanning the
    # whole gaze range. So pick by farthest-point sampling in PUPIL space, which spreads the seed
    # across gaze directions instead of over-sampling whatever the eye did most often. The pupil
    # detector is reliable here (96-100% on real frames) and is used only to CHOOSE frames -- it
    # never touches the label, which stays entirely the human's.
    picks = []
    try:
        from pupil_tracker import PupilTracker
        trk = PupilTracker()
        for rid in (0, 1):
            rows = np.where(R == rid)[0]
            if len(rows) == 0:
                continue
            cand, pts = [], []
            for i in rows[::max(1, len(rows) // 260)]:
                r = trk.detect(F[i])
                if getattr(r, "ok", False):
                    cand.append(int(i)); pts.append(r.pupil)
            want = n_want // 2
            if len(cand) <= want:
                picks += cand
                continue
            pts = np.array(pts)
            chosen = [0]                                    # farthest-point sampling
            dist = np.linalg.norm(pts - pts[0], axis=1)
            while len(chosen) < want:
                k = int(np.argmax(dist))
                chosen.append(k)
                dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
            picks += [cand[k] for k in chosen]
    except Exception:
        picks = []
    if not picks:
        for rid in (0, 1):
            rows = np.where(R == rid)[0]
            if len(rows):
                picks += [int(rows[j]) for j in sample_order(len(rows), n_want // 2 + 1)]

    prev = {}
    if os.path.exists(out):
        s = np.load(out)
        # A SEED LABEL IS ONLY MEANINGFUL AGAINST THE CORPUS IT WAS PLACED ON.
        #
        # Labels are stored as (corpus_index, u, v) -- the index is a POSITION IN THE ARRAY, not
        # an identity. Re-capture the corpus and every stored index silently resolves to a
        # different photograph. On 2026-08-03 this happened for real: 98 labels from the previous
        # session were "resumed" onto a freshly captured 3000-frame corpus, so 98 of 125 labels
        # (78%) pointed at unrelated images, and nothing in the output looked wrong -- it printed
        # "resuming - 98 already placed" and carried on. Training on that would have been strictly
        # worse than training on the 27 good labels alone, which is this project's oldest lesson:
        # a wrong label beats a missing one only in the sense that it does more damage.
        #
        # The fix is to make the seed file remember WHICH corpus it belongs to. n_frames is a
        # weak fingerprint but it catches the case that actually occurs (re-capture changes the
        # count); the mtime check catches a same-size re-capture.
        stamp = (int(F.shape[0]), int(os.path.getmtime(CORPUS)))
        old_stamp = tuple(int(x) for x in s["corpus"]) if "corpus" in s.files else None
        if old_stamp is not None and old_stamp != stamp:
            print("!! REFUSING TO RESUME — %s holds %d labels placed against a DIFFERENT corpus\n"
                  "   (was %d frames / mtime %d, now %d / %d). Those indices point at other\n"
                  "   images now. Move it aside and start a fresh seed set:\n"
                  "     mv %s %s.stale"
                  % (out, len(s["index"]), old_stamp[0], old_stamp[1], stamp[0], stamp[1],
                     out, out))
            return 1
        if old_stamp is None and len(s["index"]):
            print("!! %s has %d labels but NO corpus stamp (written before this check existed).\n"
                  "   Cannot prove they match the current corpus. Verify or move aside." % (out, len(s["index"])))
            return 1
        prev = {int(i): (float(u), float(v)) for i, (u, v) in zip(s["index"], s["labels"])}
        if verbose:
            print("resuming — %d already placed (corpus stamp matches)" % len(prev))
    todo = [i for i in picks if i not in prev][:n_want]
    if not todo:
        print("nothing left (%d done)" % len(prev))
        return 0

    results = dict(prev)
    closed = {}
    if os.path.exists(out):
        _s = np.load(out)
        if "closed" in _s.files:
            closed = {int(i): int(c) for i, c in zip(_s["index"], _s["closed"])}
    win = "CLICK the tear duct  |  ENTER=open  ·  C=closed  ·  N skip  ·  U undo  ·  Q save"
    # AUTOSIZE, not NORMAL. A NORMAL window scales the image to fit whatever size it opens at, so
    # a native 1280x800 frame gets resampled and looks soft — which is exactly what it did, and
    # the resulting scatter showed up as 3x worse label consistency (u std 0.024 vs 0.007).
    # AUTOSIZE pins the window to the image so every pixel shown is a pixel the sensor measured.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    click = {"xy": None}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (x / float(zoom * F.shape[2]), y / float(zoom * F.shape[1]))
    cv2.setMouseCallback(win, on_mouse)

    hist, k = [], 0
    while k < len(todo):
        fi = todo[k]
        # DISPLAY-ONLY enhancement. Stored frames and label coordinates are untouched; the model
        # trains on raw data.
        #
        # A hard 1-99 percentile stretch was WRONG here and Dylan spotted it immediately: these
        # frames are dark (mean ~53-66, the price of eliminating 19% clipping), so aggressively
        # rescaling them multiplies the sensor grain along with the signal. The result looked
        # NOISIER than cam_view's raw feed and read as "not max res" when the resolution was
        # in fact native. Detail was never the problem; amplified noise was.
        #
        # So: a gentle lift, then an edge-preserving denoise. Bilateral keeps the lid margin and
        # lash edges — the things being clicked — while flattening the grain between them.
        g = F[fi]
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        lift = np.clip((g.astype(np.float32) - lo) * 235.0 / max(hi - lo, 1e-6) + 10, 0, 255)
        lift = lift.astype(np.uint8)
        lift = cv2.bilateralFilter(lift, 5, 40, 5)
        base = cv2.cvtColor(lift, cv2.COLOR_GRAY2BGR)
        if zoom != 1:
            # NEAREST, not LINEAR: at 2x this keeps real pixel boundaries visible instead of
            # smearing them into an interpolated blur that only looks higher-resolution.
            base = cv2.resize(base, (F.shape[2] * zoom, F.shape[1] * zoom),
                              interpolation=cv2.INTER_NEAREST)
        H, W = base.shape[:2]
        click["xy"] = None
        role = "eyeL" if R[fi] == 0 else "eyeR"
        while True:
            shown = base.copy()
            if click["xy"]:
                p = (int(click["xy"][0] * W), int(click["xy"][1] * H))
                cv2.drawMarker(shown, p, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.circle(shown, p, 20, (0, 0, 255), 1)
            cv2.putText(shown, "%s   %d/%d placed   %s" %
                        (role, len(results), n_want,
                         "ENTER=open   C=closed" if click["xy"]
                         else "click the tear duct   (or C alone = closed, no point)"),
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, shown)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and click["xy"]:
                results[fi] = click["xy"]; closed[fi] = 0
                hist.append(fi); k += 1; break
            if key in (ord("c"), ord("C")):
                # C works WITH OR WITHOUT a click. Requiring a point first meant that on a blink
                # where the tear duct is hard to place, C silently did nothing -- which is exactly
                # when it is most needed. If no point was placed, the frame is still recorded as a
                # closed EXAMPLE with an invalid position (-1,-1); the eye-state head learns from
                # it and the landmark head masks it out.
                results[fi] = click["xy"] if click["xy"] else (-1.0, -1.0)
                closed[fi] = 1
                hist.append(fi); k += 1; break
            if key in (ord("n"), ord("N")):
                hist.append(None); k += 1; break
            if key in (ord("u"), ord("U")):
                if hist:
                    last = hist.pop()
                    if last is not None:
                        results.pop(last, None)
                    k = max(0, k - 1)
                break
            if key in (ord("q"), 27):
                k = len(todo); break
    cv2.destroyAllWindows()

    if not results:
        print("nothing placed")
        return 1
    idx = np.array(sorted(results), np.int32)
    xy = np.array([results[int(i)] for i in idx], np.float32)
    fl = np.array([closed.get(int(i), 0) for i in idx], np.uint8)
    # Stamp the corpus this seed set belongs to, so a later --seed cannot silently resume these
    # labels onto a re-captured corpus. See the refusal in the resume block above.
    np.savez_compressed(out, index=idx, labels=xy, closed=fl,
                        corpus=np.array([int(F.shape[0]), int(os.path.getmtime(CORPUS))], np.int64))
    if verbose:
        print("\n== seed ==")
        print("  placed %d points -> %s" % (len(idx), out))
        print("  eye state: %d open / %d closed" % (int((fl == 0).sum()), int(fl.sum())))
        if fl.sum() < 5:
            print("  NOTE: fewer than 5 closed examples — the closed head will be weak.")
        valid = xy[:, 0] >= 0
        if valid.any():
            agree = np.linalg.norm(xy[valid] - d["labels"][idx][valid], axis=1)
            print("  median distance from the template's answer: %.3f frame units"
                  % np.median(agree))
        print("  positioned %d | closed-without-point %d (eye-state only)"
              % (int(valid.sum()), int((~valid).sum())))
        print("  (large is EXPECTED and is the point — the template was on the lid)")
    return 0


def closed_pass(corpus=CORPUS, seed=SEED, verbose=True):
    """Second pass: mark each labelled frame EYE OPEN or EYE CLOSED. One key per frame.

    WHY IT IS WORTH A SEPARATE HEAD
        rig.py already models blink dropouts, but nothing in the live loop DETECTS one. A sample
        recorded mid-blink is bad twice over: the gaze is not fixated, and the lid deforms the
        soft tissue around the canthus — the exact tissue a 200 px template patch is full of. So
        the model should say "eye is closed, do not trust this frame", and the calibration loop
        should skip it.

        Note the landmark itself does NOT vanish during a blink: the inner canthus is where the
        lids MEET, so it stays visible and stays labellable. Closed-ness is a separate property of
        the frame, not a reason to drop the position label — which is why this is a second head
        rather than a filter.

    Kept separate from the position pass on purpose: binary judgements go at a completely
    different speed from precise clicking, and mixing them slows both.

        O / ENTER  eye open      C  eye closed      U  undo      Q  save+quit
    """
    import cv2
    if not os.path.exists(seed):
        print("!! label positions first: python3 canthus_label.py --label")
        return 1
    d = np.load(corpus)
    F = d["frames"]
    s = np.load(seed)
    idx, xy = s["index"], s["labels"]
    closed = {int(i): int(c) for i, c in zip(s["index"], s["closed"])} \
        if "closed" in s.files else {}

    todo = [int(i) for i in idx if int(i) not in closed]
    if not todo:
        print("all %d frames already marked open/closed" % len(idx))
        return 0
    win = "eye state  |  O open  ·  C closed  ·  U undo  ·  Q save+quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    hist, i = [], 0
    while i < len(todo):
        fi = todo[i]
        img = cv2.cvtColor(F[fi], cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (F.shape[2] * ZOOM, F.shape[1] * ZOOM),
                         interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]
        k_at = np.where(idx == fi)[0][0]
        p = (int(xy[k_at][0] * W), int(xy[k_at][1] * H))
        cv2.drawMarker(img, p, (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, "OPEN or CLOSED?   %d / %d" % (len(closed), len(idx)),
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("o"), ord("O"), 13, 32):
            closed[fi] = 0; hist.append(fi); i += 1
        elif k in (ord("c"), ord("C")):
            closed[fi] = 1; hist.append(fi); i += 1
        elif k in (ord("u"), ord("U")) and hist:
            closed.pop(hist.pop(), None); i = max(0, i - 1)
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    flags = np.array([closed.get(int(i), 0) for i in idx], np.uint8)
    np.savez_compressed(seed, index=idx, labels=xy, closed=flags)
    if verbose:
        print("\n== eye state ==")
        print("  marked %d of %d | closed %d (%.0f%%)"
              % (len(closed), len(idx), int(flags.sum()), 100.0 * flags.mean()))
        if flags.sum() < 15:
            print("  NOTE: few closed examples — the closed-eye head will be weak. Blink more")
            print("        during the next collection run if you want it reliable.")
    return 0


def label(n_want=300, corpus=CORPUS, out=SEED, verbose=True):
    import cv2
    if not os.path.exists(corpus):
        print("!! no corpus at %s — run: python3 canthus_data.py --collect" % corpus)
        return 1
    d = np.load(corpus)
    F, L, R = d["frames"], d["labels"], d["roles"]

    prev = {}
    if os.path.exists(out):                      # resume: keep what is already labelled
        s = np.load(out)
        prev = {int(i): (float(u), float(v)) for i, (u, v) in zip(s["index"], s["labels"])}
        if verbose:
            print("resuming — %d frames already labelled" % len(prev))

    order = [i for i in sample_order(len(F), n_want) if int(i) not in prev]
    if not order:
        print("nothing left to label (%d done)" % len(prev))
        return 0

    results = dict(prev)
    win = "canthus  |  ENTER accept  ·  CLICK to correct  ·  N skip  ·  U undo  ·  Q save+quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    click = {"xy": None}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (x / float(ZOOM * F.shape[2]), y / float(ZOOM * F.shape[1]))
    cv2.setMouseCallback(win, on_mouse)

    k_done, history = 0, []
    i = 0
    while i < len(order):
        fi = int(order[i])
        img = cv2.cvtColor(F[fi], cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (F.shape[2] * ZOOM, F.shape[1] * ZOOM),
                         interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]
        click["xy"] = None
        proposal = (float(L[fi][0]), float(L[fi][1]))
        role = "eyeL" if R[fi] == 0 else "eyeR"

        while True:
            shown = img.copy()
            pt = click["xy"] or proposal
            p = (int(pt[0] * W), int(pt[1] * H))
            col = (0, 255, 0) if click["xy"] else (0, 165, 255)
            cv2.drawMarker(shown, p, col, cv2.MARKER_CROSS, 26, 2)
            cv2.circle(shown, p, 16, col, 1)
            cv2.putText(shown, "%s   %d/%d labelled   %s" %
                        (role, len(results), n_want,
                         "YOUR POINT" if click["xy"] else "template proposal"),
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            cv2.imshow(win, shown)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 32):                       # ENTER / SPACE -> accept what is shown
                results[fi] = pt
                history.append(fi); k_done += 1; i += 1; break
            if k in (ord("n"), ord("N")):           # skip
                history.append(None); i += 1; break
            if k in (ord("u"), ord("U")):           # undo
                if history:
                    last = history.pop()
                    if last is not None:
                        results.pop(last, None)
                    i = max(0, i - 1)
                break
            if k in (ord("q"), 27):
                i = len(order); break
    cv2.destroyAllWindows()

    if not results:
        print("nothing labelled")
        return 1
    idx = np.array(sorted(results), np.int32)
    xy = np.array([results[int(i)] for i in idx], np.float32)
    np.savez_compressed(out, index=idx, labels=xy)
    if verbose:
        agree = np.linalg.norm(xy - L[idx], axis=1)
        print("\n== seed set ==")
        print("  labelled %d frames -> %s" % (len(idx), out))
        print("  vs the template proposal: median disagreement %.3f, %.0f%% moved >0.05"
              % (np.median(agree), 100.0 * (agree > 0.05).mean()))
        print("  (that second number IS the template's error rate — it is why we are doing this)")
    return 0


def selftest(verbose=True):
    if verbose:
        print("== canthus_label self-test (no hardware) ==")
    checks = []

    # Sampling must SPREAD, not take a contiguous block: consecutive frames are near-duplicates.
    o = sample_order(1500, 300)
    checks.append(("sample_order spreads across the corpus (not a contiguous block)",
                   len(o) == 300 and len(set(o.tolist())) == 300
                   and o.max() > 1200 and o.min() < 300 and np.std(np.diff(np.sort(o))) > 0))

    # Deterministic: resuming a part-finished session must revisit the same frames.
    checks.append(("sample_order is deterministic across runs",
                   np.array_equal(sample_order(1500, 300), sample_order(1500, 300))))

    # Asking for more than exists must clamp, not crash.
    checks.append(("sample_order clamps when n_want exceeds the corpus",
                   len(sample_order(50, 300)) == 50))

    # Click -> normalised coordinate maths (the mapping the mouse callback performs).
    W, H, Z = 320, 200, ZOOM
    u, v = (480 / float(Z * W), 300 / float(Z * H))
    checks.append(("click at display centre maps to u=0.5 v=0.5 (zoom %dx)" % Z,
                   abs(u - 0.5) < 1e-6 and abs(v - 0.5) < 1e-6))

    ok = all(x for _, x in checks)
    if verbose:
        for name, x in checks:
            print("  [%s] %s" % ("PASS" if x else "FAIL", name))
        print("  =>", "CANTHUS LABEL OK ✅" if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="hand-label the canthus seed set")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--seed", action="store_true",
                    help="place the TEAR DUCT on ~30 full frames (never cropped)")
    ap.add_argument("--closed-pass", action="store_true",
                    help="second pass: mark each labelled frame eye-open or eye-closed")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=30, help="how many frames to label")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.label:
        sys.exit(label(n_want=a.n))
    if a.seed:
        sys.exit(seed_pass(n_want=a.n))
    if a.closed_pass:
        sys.exit(closed_pass())
    ap.print_help()
