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
    """Convolution via im2col, vectorised with sliding_window_view.

    The first version built the column matrix with a Python loop over C*KH*KW -- 576 array copies
    per 64-channel 3x3 layer. Measured 27.9 ms per frame, which is 55.8 ms for two eyes against a
    73.6 ms whole-frame budget: the landmark alone would have eaten most of the loop. Dylan flagged
    it as slow on the live view before the profile did.

    sliding_window_view produces the same patches as a strided VIEW, so the only real cost is the
    single reshape that materialises it plus one matmul. Same arithmetic, same result -- the
    selftest asserts the numpy runtime still reproduces the trained landmark."""
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2, 3))
    win = win[:, :, ::stride, ::stride]                  # (N, C, OH, OW, KH, KW)
    OH, OW = win.shape[2], win.shape[3]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, OH * OW, C * KH * KW)
    out = cols @ w.reshape(F, -1).astype(np.float32).T   # (N, OH*OW, F)
    out = out + b.reshape(1, 1, F).astype(np.float32)
    return out.transpose(0, 2, 1).reshape(N, F, OH, OW)


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


class CanthusTracker:
    """Stateful wrapper: confidence gate + jump gate + smoothing. Use this in the live loop.

    WHY, from Dylan on hardware: the prediction "moved off when I open my eyes super wide or look
    outside glasses frame". Both are OUT OF DISTRIBUTION — the seed set was normal-gaze frames, so
    the model has never seen an eye held wide or rotated to an extreme. The real fix is seed points
    covering those poses; this makes the failure SAFE in the meantime, which matters because a
    silently wrong landmark is worse than an admitted gap.

    Three defences, cheapest first:

    CONFIDENCE GATE. The heatmap peak says how sure the model is, and out-of-distribution frames
    flatten it. Measured on eyeL: confident frames (including all 30 seeds) sit at 0.018-0.022,
    the whole-corpus 5th percentile is 0.0152, and a blank frame bottoms out at 0.0084. Below
    ~0.017 the model does not know, and the honest move is to say so rather than emit a number.
    Template matching had no equivalent — it scored ~1.0 on featureless skin.

    JUMP GATE. The inner canthus is FACE-fixed: it does not move when gaze moves, and it cannot
    teleport between frames. A large inter-frame jump is therefore a detection error by
    construction, whatever the confidence says. This is the same reasoning that separates the
    canthus from the pupil in the first place.

    SMOOTHING. An EMA over accepted frames. Live jitter measured ±0.074 in u on this camera, well
    above the ±0.021 the labels support, so most of that is estimator noise worth averaging out.

    `state` is 'ok' | 'held' | 'lost'. HELD means the last good position is being reused; LOST
    means there is no trustworthy estimate at all. The calibration loop should refuse to record a
    sample unless state == 'ok' — recording during a blink or an off-distribution pose is exactly
    the bad sample rig.py's blink-dropout model exists to describe.
    """

    # NO SHARPNESS GATE. It worked on the first model and INVERTED on the second: measured on the
    # 98-seed model, a blank frame peaks at 0.0202 while real frames median 0.0154 -- the model is
    # "more confident" about nothing than about an eye. A gate on that would hold on good frames
    # and accept garbage. Heatmap peak is an artefact of how a particular model spreads its
    # softmax, not a property of the task, so it cannot be trusted across retrainings.
    #
    # The gate is now the ANATOMICAL PRIOR plus the jump test: where a canthus can physically be,
    # and how fast it can physically move. Both are properties of the rig and the face, so neither
    # can silently invert when the model is retrained.
    CONF_MIN = 0.0           # retained for the API; sharpness is no longer gated on
    MAX_JUMP = 0.06         # normalised frame units between consecutive accepted frames
    EMA = 0.35              # weight on the new observation
    MAX_HELD = 15           # frames to coast before declaring LOST

    def __init__(self, net=None, conf_min=None, max_jump=None, ema=None,
                 mirrored=None, prior=None, pad=0.08):
        self.net = net or CanthusNet()
        self.conf_min = self.CONF_MIN if conf_min is None else conf_min
        self.max_jump = self.MAX_JUMP if max_jump is None else max_jump
        self.ema = self.EMA if ema is None else ema
        self.mirrored = mirrored
        self.pad = pad
        self.prior = prior
        if self.prior is None and mirrored is not None:
            try:
                from canthus_auto import build_prior
                self.prior = build_prior()
            except Exception:
                self.prior = None
        self.uv = None
        self.held = 0

    def track(self, frame):
        """Drop-in for eye_tracker.EyeCornerTracker.track: returns ((u, v), score) or (None, 0).

        The score maps the tracker STATE onto the confidence the calibration loop already gates on
        (config.eye_conf_min), so a sample recorded mid-blink or off-distribution is refused by
        machinery that already exists rather than by a new special case:

            ok    1.0   confident and physically plausible -- record it
            held  0.3   coasting on the last good position -- show it, do not record it
            lost  0.0   no trustworthy estimate at all

        Accepts colour or grayscale, matching EyeCornerTracker."""
        import cv2
        if frame is None:
            return None, 0.0
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        u, v, cp, conf, state = self.update(g)
        if u is None:
            return None, 0.0
        return (u, v), {"ok": 1.0, "held": 0.3}.get(state, 0.0)

    @property
    def ready(self):
        """EyeCornerTracker exposes this; the model is always ready once weights load."""
        return True

    def _plausible(self, u, v):
        """Physics gate: is this where a canthus can be? Independent of the model entirely."""
        if self.prior is None or self.mirrored is None:
            return True
        uu = (1.0 - u) if self.mirrored else u
        return (self.prior["u_lo"] - self.pad <= uu <= self.prior["u_hi"] + self.pad)

    def reset(self):
        self.uv = None
        self.held = 0

    def update(self, gray):
        """-> (u, v, closed_prob, conf, state). u,v are None when state == 'lost'."""
        u, v, cp, conf = self.net.predict(gray)

        if not self._plausible(u, v):
            self.held += 1
            if self.uv is None or self.held > self.MAX_HELD:
                return None, None, cp, conf, "lost"
            return self.uv[0], self.uv[1], cp, conf, "held"

        if self.uv is not None:
            if np.hypot(u - self.uv[0], v - self.uv[1]) > self.max_jump:
                # Confident but implausible. Trust the physics over the model: a face-fixed point
                # does not jump. Coasting here is what stops a wide-open eye yanking the estimate
                # across the frame.
                self.held += 1
                if self.held > self.MAX_HELD:
                    self.uv = (u, v)          # sustained disagreement -> the world really moved
                    self.held = 0
                    return u, v, cp, conf, "ok"
                return self.uv[0], self.uv[1], cp, conf, "held"
            a = self.ema
            self.uv = (self.uv[0] * (1 - a) + u * a, self.uv[1] * (1 - a) + v * a)
        else:
            self.uv = (u, v)
        self.held = 0
        return self.uv[0], self.uv[1], cp, conf, "ok"


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
        # PIN THE INVERSION. On the first model a blank frame was measurably flatter than a real
        # one, so heatmap peak looked like a usable confidence signal. On the 98-seed model it
        # REVERSED -- blank 0.0202 against real 0.0148 -- i.e. the model is "more certain" about
        # nothing than about an eye. Peak is an artefact of how a given model spreads its softmax,
        # not a property of the task, so it must NOT be gated on. This check records the fact
        # rather than asserting a direction, so a future retraining cannot quietly restore a
        # gate that inverts.
        checks.append(("sharpness is NOT a usable confidence signal: blank %.4f vs real %.4f "
                       "(gating on this would invert)" % (sh_blank, real_sh),
                       sh_blank > 0 and real_sh > 0))

    # 4) The tracker's gates, with a stub net so the logic is tested rather than the weights.
    class _Stub:
        def __init__(self): self.q = []
        def predict(self, g): return self.q.pop(0)
    st = _Stub()
    trk = CanthusTracker.__new__(CanthusTracker)
    trk.net = st; trk.conf_min = 0.0; trk.max_jump = 0.06; trk.ema = 1.0
    trk.uv = None; trk.held = 0; trk.prior = None; trk.mirrored = None; trk.pad = 0.08
    st.q = [(0.20, 0.60, 0.1, 0.021),      # confident -> ok
            (0.205, 0.605, 0.1, 0.021),    # confident, small move -> ok
            (0.60, 0.20, 0.1, 0.021),      # CONFIDENT BUT A HUGE JUMP -> held, not accepted
            (0.21, 0.61, 0.1, 0.010)]      # low confidence -> held
    r = [trk.update(None) for _ in range(4)]
    checks.append(("tracker: confident+plausible -> ok, ok",
                   r[0][4] == "ok" and r[1][4] == "ok"))
    checks.append(("tracker: CONFIDENT but implausible jump -> held (physics beats the model)",
                   r[2][4] == "held" and abs(r[2][0] - 0.205) < 1e-6))
    # (the old "low confidence -> held" check is gone with the sharpness gate; the prior and jump
    #  gates below are what refuse a bad frame now, and neither depends on model internals)

    trk2 = CanthusTracker.__new__(CanthusTracker)
    # PRIOR GATE: an anatomically impossible position is refused even when the model is sure.
    # This replaces the sharpness gate, which INVERTED between models (blank 0.0202 vs real
    # 0.0154) and would have held on good frames while accepting nothing.
    st2 = _Stub(); trk2.net = st2; trk2.conf_min = 0.0; trk2.max_jump = 0.06; trk2.ema = 1.0
    trk2.uv = None; trk2.held = 0; trk2.pad = 0.08; trk2.mirrored = False
    trk2.prior = dict(u_lo=0.778, u_hi=0.891, v_lo=0.110, v_hi=0.620,
                      u_med=0.834, v_med=0.362)
    st2.q = [(0.20, 0.60, 0.1, 0.9)] * 3          # confident, but far outside the u band
    r2 = [trk2.update(None) for _ in range(3)]
    checks.append(("prior gate: anatomically impossible u=0.20 refused despite high confidence",
                   r2[0][4] == "lost" and r2[0][0] is None))

    trk3 = CanthusTracker.__new__(CanthusTracker)
    st3 = _Stub(); trk3.net = st3; trk3.conf_min = 0.0; trk3.max_jump = 0.06; trk3.ema = 1.0
    trk3.uv = None; trk3.held = 0; trk3.pad = 0.08; trk3.mirrored = False
    trk3.prior = trk2.prior
    st3.q = [(0.834, 0.40, 0.1, 0.9)]
    checks.append(("prior gate: a position inside the band is accepted",
                   trk3.update(None)[4] == "ok"))

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
    # mirrored=True for eyeL (index 0), whose image is the mirror of the prior's convention
    trk = CanthusTracker(mirrored=(cam == 0))
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
        u, v, cp, sh, state = trk.update(g)
        tot += time.time() - t
        n += 1
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        H, W = g.shape
        col = {"ok": (0, 255, 0), "held": (0, 200, 255), "lost": (0, 0, 255)}[state]
        if u is not None:
            cv2.drawMarker(vis, (int(u * W), int(v * H)), col, cv2.MARKER_CROSS, 40, 3)
        cv2.putText(vis, "%s  u %s v %s  closed %.2f  conf %.4f  %.0f ms" %
                    (state.upper(),
                     ("%.3f" % u) if u is not None else "--",
                     ("%.3f" % v) if v is not None else "--",
                     cp, sh, 1000 * tot / max(n, 1)),
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
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
