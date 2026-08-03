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
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=300, help="how many frames to label")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.label:
        sys.exit(label(n_want=a.n))
    ap.print_help()
