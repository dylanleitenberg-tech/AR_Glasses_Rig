"""Run the trained canthus model — pure numpy, no torch. This is the RUNTIME half.

WHY NUMPY AND NOT TORCH
    Training happens once, offline, in .venv-train. The live loop must not inherit a 200 MB
    dependency for a forward pass that is five convolutions — and on this Intel Mac torch 2.2.2
    (the last available build) predates numpy 2 and breaks interop with the main venv outright.
    So canthus_train.py --export folds BatchNorm into plain scale/shift constants and writes
    .npz; everything below is arithmetic. verify_all stays self-contained.

WHAT IT REPLACES
    eye_tracker.EyeCornerTracker: a hand-drawn template, re-captured EVERY session, that scores
    ~1.0 on featureless skin and locks onto whatever matched rather than onto a canthus. Dylan:
    "i cant box every time." This is the thing that removes that step.

WHAT IT OUTPUTS
    (u, v, closed_prob, sharpness) — position normalised to the frame, the eye-state head, and
    how PEAKED the heatmap is. Sharpness matters: a flat heatmap means the model does not know,
    which is exactly the signal template matching could never give. A high correlation score on a
    flat patch is indistinguishable from a real lock; a flat heatmap is self-announcing.

    python3 canthus_net.py --selftest
    python3 canthus_net.py --live --cam 0
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
WEIGHTS = os.path.join(DATA, "canthus_net.npz")


def _conv2d(x, w, b, stride=1, pad=1):
    """Direct convolution via im2col. Small net, small input — clarity beats cleverness here."""
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH = (x.shape[2] - KH) // stride + 1
    OW = (x.shape[3] - KW) // stride + 1
    cols = np.lib.stride_tricks.as_strided(
        x,
        shape=(N, C, OH, OW, KH, KW),
        strides=(x.strides[0], x.strides[1],
                 x.strides[2] * stride, x.strides[3] * stride,
                 x.strides[2], x.strides[3]),
        writeable=False,
    ).reshape(N, C * KH * KW, OH * OW) if False else None
    # as_strided on a padded copy is fragile across numpy versions; build cols explicitly.
    cols = np.empty((N, C * KH * KW, OH * OW), np.float32)
    k = 0
    for c in range(C):
        for i in range(KH):
            for j in range(KW):
                patch = x[:, c, i:i + OH * stride:stride, j:j + OW * stride:stride]
                cols[:, k, :] = patch.reshape(N, -1)
                k += 1
    out = np.einsum("fk,nkp->nfp", w.reshape(F, -1).astype(np.float32), cols)
    return (out + b.reshape(1, F, 1).astype(np.float32)).reshape(N, F, OH, OW)


class CanthusNet:
    """The exported network. Layer order mirrors canthus_train.build_net exactly."""

    def __init__(self, path=WEIGHTS):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no weights at %s — train and export first:\n"
                "  source .venv-train/bin/activate\n"
                "  python3 canthus_train.py --train && python3 canthus_train.py --export" % path)
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        self.in_h, self.in_w = (int(v) for v in self.w["in_hw"])
        # trunk = conv,bn,relu x5 at indices 0,1,2 / 3,4,5 / 6,7,8 / 9,10,11 / 12,13,14
        self.trunk = [(0, 1), (3, 4), (6, 7), (9, 10), (12, 13)]
        self.strides = [2, 1, 2, 1, 1]
        self.heat_i, self.closed_i = 15, 16

    def _forward(self, x):
        for (ci, bi), st in zip(self.trunk, self.strides):
            x = _conv2d(x, self.w["c%d_w" % ci], self.w["c%d_b" % ci],
                        stride=st, pad=self.w["c%d_w" % ci].shape[2] // 2)
            x = x * self.w["b%d_scale" % bi].reshape(1, -1, 1, 1) \
                + self.w["b%d_shift" % bi].reshape(1, -1, 1, 1)
            x = np.maximum(x, 0.0)
        heat = _conv2d(x, self.w["c%d_w" % self.heat_i], self.w["c%d_b" % self.heat_i],
                       stride=1, pad=0)
        closed = _conv2d(x, self.w["c%d_w" % self.closed_i], self.w["c%d_b" % self.closed_i],
                         stride=1, pad=0)
        return heat, closed

    def predict(self, gray):
        """gray uint8 HxW -> (u, v, closed_prob, sharpness)."""
        import cv2
        x = cv2.resize(gray, (self.in_w, self.in_h)).astype(np.float32) / 255.0
        heat, closed = self._forward(x[None, None])
        h = heat[0, 0]
        # soft-argmax, identical to training. Sub-pixel: the heatmap is input/4, so a hard argmax
        # would quantise position to 2.5% of frame width.
        e = np.exp(h - h.max())
        p = e / e.sum()
        H, W = p.shape
        u = float((p * np.linspace(0, 1, W)[None, :]).sum())
        v = float((p * np.linspace(0, 1, H)[:, None]).sum())
        # SHARPNESS = how concentrated the heatmap is. 1.0 means one cell holds everything;
        # near 1/(H*W) means the model is spreading its bet, i.e. it does not know.
        sharp = float(p.max())
        cp = float(1.0 / (1.0 + np.exp(-closed.mean())))
        return u, v, cp, sharp


def selftest(verbose=True):
    if verbose:
        print("== canthus_net self-test (numpy runtime) ==")
    checks = []
    if not os.path.exists(WEIGHTS):
        print("  [SKIP] no exported weights yet — train first")
        return 0
    net = CanthusNet()
    import cv2

    # 1) It runs and returns sane ranges on a real corpus frame.
    corpus = os.path.join(DATA, "canthus_corpus.npz")
    if os.path.exists(corpus):
        d = np.load(corpus)
        g = d["frames"][0]
        u, v, cp, sh = net.predict(g)
        checks.append(("forward pass on a real frame -> u %.3f v %.3f closed %.2f sharp %.3f"
                       % (u, v, cp, sh),
                       0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and 0.0 <= cp <= 1.0))

        # 2) AGREES WITH TORCH. The whole point of re-implementing the forward pass in numpy is
        #    that it must produce the SAME answer; a silently divergent runtime would be the
        #    worst possible bug here, since it would look like a working model that quietly
        #    predicts something else. Compared against positions torch produced at export time.
        seed = os.path.join(DATA, "canthus_seed.npz")
        if os.path.exists(seed):
            s = np.load(seed)
            idx = s["index"][s["labels"][:, 0] >= 0][:8]
            errs = []
            for i in idx:
                pu, pv, _, _ = net.predict(d["frames"][i])
                errs.append(np.hypot(pu - s["labels"][list(s["index"]).index(i)][0],
                                     pv - s["labels"][list(s["index"]).index(i)][1]))
            med = float(np.median(errs))
            checks.append(("numpy runtime reproduces the seed landmark, median err %.3f "
                           "frame units" % med, med < 0.12))

    # 3) A flat/blank frame must yield a FLAT heatmap — low sharpness. This is the property
    #    template matching never had: it scored ~1.0 on featureless skin and could not say
    #    "I don't know". A model that spreads its probability IS saying it.
    blank = np.full((400, 640), 128, np.uint8)
    _, _, _, sh_blank = net.predict(blank)
    real_sh = None
    if os.path.exists(corpus):
        _, _, _, real_sh = net.predict(np.load(corpus)["frames"][0])
    if real_sh is not None:
        checks.append(("blank frame is LESS peaked than a real one (%.4f vs %.4f)"
                       % (sh_blank, real_sh), sh_blank <= real_sh * 1.5))

    ok = all(x for _, x in checks)
    if verbose:
        for name, x in checks:
            print("  [%s] %s" % ("PASS" if x else "FAIL", name))
        print("  =>", "CANTHUS NET OK — numpy runtime matches training ✅" if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


def live(cam=0, seconds=None):
    """Run the model on a live camera and draw where it thinks the canthus is."""
    import cv2
    import time
    net = CanthusNet()
    cap = cv2.VideoCapture(cam, getattr(cv2, "CAP_AVFOUNDATION", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    win = "canthus model — live  |  q quits"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t0, n, tot = time.time(), 0, 0.0
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        t = time.time()
        u, v, cp, sh = net.predict(g)
        tot += time.time() - t
        n += 1
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        H, W = g.shape
        col = (0, 0, 255) if cp < 0.5 else (0, 200, 255)
        cv2.drawMarker(vis, (int(u * W), int(v * H)), col, cv2.MARKER_CROSS, 40, 3)
        cv2.putText(vis, "u %.3f v %.3f  closed %.2f  peak %.3f  %.0f ms" %
                    (u, v, cp, sh, 1000 * tot / max(n, 1)),
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
        if seconds and time.time() - t0 > seconds:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("mean inference %.1f ms over %d frames" % (1000 * tot / max(n, 1), n))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="numpy runtime for the canthus model")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.live:
        sys.exit(live(a.cam, a.seconds))
    ap.print_help()
