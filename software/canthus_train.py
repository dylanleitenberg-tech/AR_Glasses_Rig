"""Train the inner-canthus landmark model. RUNS IN .venv-train (torch), NOT the main venv.

    source .venv-train/bin/activate
    python3 canthus_train.py --train          # seed -> two independent models
    python3 canthus_train.py --pseudo         # ensemble-agreed labels for the unlabelled rest
    python3 canthus_train.py --train --with-pseudo   # retrain on seed + agreed pseudo-labels
    python3 canthus_train.py --export         # -> data/canthus_net.npz for the numpy runtime

WHY TWO MODELS
    Dylan's call, and it is the right one. The reason template auto-labelling failed is that its
    errors were CONFIDENT: wrong labels scored higher margin than right ones, so no threshold
    could filter them. Confidence from a single estimator is not evidence.

    Two models trained independently — different init, different augmentation stream, different
    data order — fail in uncorrelated ways. Where they AGREE on a held-out frame, the label is
    almost certainly right; where they disagree, it is exactly the case a human should see. That
    turns "confidence" into something that actually carries information, and it is what lets a
    few hundred hand labels bootstrap into thousands.

WHY A HEATMAP, NOT COORDINATE REGRESSION
    Regressing (u,v) directly forces the net to collapse spatial evidence into two numbers through
    fully-connected layers, which generalises poorly and gives no way to see uncertainty. A
    heatmap keeps the prediction spatial: soft-argmax gives sub-pixel position, and the peak's
    SHARPNESS is a usable confidence — a flat heatmap means the model does not know, which is
    precisely the signal the template could never produce.

RUNTIME STAYS NUMPY
    Training happens here, once, in an isolated venv. --export writes plain .npz weights and
    canthus_net.py runs the forward pass in numpy, so the live loop gains no dependency and
    verify_all stays self-contained. This Intel Mac tops out at torch 2.2.2, which predates
    numpy 2 — see .gitignore for the full reasoning.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CORPUS = os.path.join(DATA, "canthus_corpus.npz")
SEED = os.path.join(DATA, "canthus_seed.npz")
PSEUDO = os.path.join(DATA, "canthus_pseudo.npz")
WEIGHTS = os.path.join(DATA, "canthus_net.npz")

IN_H, IN_W = 100, 160          # network input; corpus frames are 200x320, halved for speed
HEAT_H, HEAT_W = 25, 40        # heatmap is input/4
AGREE_PX = 0.02                # ensemble agreement threshold, normalised frame units


# ----------------------------------------------------------------------------------
#  Model — deliberately small. This is a single-landmark task on a fixed camera geometry,
#  not ImageNet; a big net would overfit a few hundred frames and cost frame time later.
# ----------------------------------------------------------------------------------
class CanthusNet:
    """Shared trunk, two heads: a landmark heatmap and an eye-open/closed logit.

    The heads share the trunk deliberately. Whether the eye is closed and where the corner sits
    are read from the SAME evidence — lid margins, lash line, whether the iris is visible — so one
    set of features serves both, and the closed head costs a single extra 1x1 conv rather than a
    second network. It also regularises: the trunk cannot cheat on position by ignoring lid shape
    if it must also report lid state.

    The landmark is NOT dropped when the eye closes. The inner canthus is where the lids MEET, so
    it stays visible through a blink and stays labellable; closed-ness is a separate property of
    the frame. That matters because rig.py models blink dropouts but nothing in the live loop
    detects one, and a sample recorded mid-blink is bad twice over — gaze is not fixated, and the
    lid deforms exactly the soft tissue the landmark sits in.
    """


def build_net(torch, nn, seed):
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),   # /2
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),  # /4
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.heat = nn.Conv2d(64, 1, 1)          # landmark heatmap
            self.closed = nn.Conv2d(64, 1, 1)        # eye-state logit (global-pooled)

        def forward(self, x):
            f = self.trunk(x)
            return self.heat(f), self.closed(f).mean((1, 2, 3))

    return Net()


def soft_argmax(torch, heat):
    """Spatial soft-argmax -> normalised (u,v), differentiable and sub-pixel.

    Sub-pixel matters: the heatmap is input/4 = 40x25 cells, so hard argmax would quantise the
    landmark to 2.5% of frame width. The softmax-weighted centroid recovers position far finer
    than the cell grid."""
    B, _, H, W = heat.shape
    p = torch.softmax(heat.reshape(B, -1), dim=1).reshape(B, 1, H, W)
    xs = torch.linspace(0, 1, W, device=heat.device).reshape(1, 1, 1, W)
    ys = torch.linspace(0, 1, H, device=heat.device).reshape(1, 1, H, 1)
    return torch.cat([(p * xs).sum((1, 2, 3), keepdim=False).reshape(B, 1),
                      (p * ys).sum((1, 2, 3), keepdim=False).reshape(B, 1)], 1)


def augment(rng, img, uv):
    """Photometric + small geometric jitter, applied per-sample so the two models see DIFFERENT
    streams. Brightness/contrast/noise cover the lighting swings these ambient-lit NoIR cameras
    suffer; small shifts cover seating drift. Deliberately NO horizontal flip — that would turn a
    left eye into a right eye and teach the wrong landmark."""
    import cv2
    out = img.astype(np.float32)

    # ILLUMINATION GRADIENT — the augmentation that was missing, and the one that matters most.
    # Dylan's room is lit from one side, so a brightness RAMP crosses the face; the old
    # augmentation only ever changed the LEVEL. Measured consequence: on a synthetic side-lit
    # frame the model went from 26 px to 258 px, and NO preprocessing rescued it -- illumination
    # flattening fixed the gradient but destroyed clean performance (26 -> 243 px) because the
    # model had never seen a flattened image. A gradient the model has never seen cannot be
    # normalised away at inference; it has to be trained through.
    if rng.random() < 0.7:
        H0, W0 = out.shape
        strength = rng.uniform(0.15, 0.75)
        if rng.random() < 0.5:                          # left-right ramp
            ramp = np.linspace(1.0, 1.0 - strength, W0)[None, :]
        else:                                           # top-bottom ramp
            ramp = np.linspace(1.0, 1.0 - strength, H0)[:, None]
        if rng.random() < 0.5:
            ramp = ramp[:, ::-1] if ramp.shape[1] > 1 else ramp[::-1, :]
        out = out * ramp

    # GAMMA — a nonlinear response change, which a linear gain cannot imitate. Range follows the
    # published pupil-detection recipe (ETRA 2019): gamma in [0.6, 1.4].
    if rng.random() < 0.6:
        g = rng.uniform(0.6, 1.4)
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, g)

    # WIDE MULTIPLICATIVE GAIN. The old range was contrast 0.7-1.3 with a +-40 offset -- about
    # 110-190 on a corpus sitting at ~150. The rig's real frames measured 39-94, entirely outside
    # it. 0.3-1.5 covers what the hardware actually produces.
    a = rng.uniform(0.30, 1.50)
    b = rng.uniform(-40, 40)
    out = np.clip(out * a + b, 0, 255)
    if rng.random() < 0.5:
        out += rng.normal(0, rng.uniform(2, 8), out.shape)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    du, dv = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
    H, W = out.shape
    M = np.float32([[1, 0, du * W], [0, 1, dv * H]])
    out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    # MATCH INFERENCE. canthus_net.predict median-normalises every frame, so training must see the
    # same transform or the two disagree about what an image looks like. Normalising here also
    # means the gain/gamma augmentation above is not wasted: it is the GRADIENT and the noise that
    # survive normalisation, which is exactly the part the model must learn to tolerate.
    out = np.clip(out, 0, 255)
    m = float(np.median(out))
    if m > 8.0:
        out = np.clip(out * (161.0 / m), 0, 255)
    return out, (uv[0] + du, uv[1] + dv)


def _load_pairs(with_pseudo):
    d = np.load(CORPUS)
    F, R = d["frames"], d["roles"]
    s = np.load(SEED)
    idx, xy = s["index"], s["labels"]
    cl = s["closed"].astype(np.float32) if "closed" in s.files else np.zeros(len(idx), np.float32)
    # Frames marked CLOSED with no point carry (-1,-1). They are valid EYE-STATE examples but have
    # no landmark, so they must be masked out of the position loss -- training on (-1,-1) as a
    # coordinate would drag the heatmap toward a corner that means nothing.
    pos_valid = (xy[:, 0] >= 0).astype(np.float32)
    if with_pseudo and os.path.exists(PSEUDO):
        p = np.load(PSEUDO)
        idx = np.concatenate([idx, p["index"]])
        xy = np.concatenate([xy, p["labels"]])
        # Pseudo-labels carry no eye-state truth; mark them -1 so the closed head IGNORES them
        # rather than learning "everything the ensemble agreed on was open".
        cl = np.concatenate([cl, -np.ones(len(p["index"]), np.float32)])
        pos_valid = np.concatenate([pos_valid, np.ones(len(p["index"]), np.float32)])
        print("  training on %d seed + %d ensemble-agreed pseudo-labels"
              % (len(s["index"]), len(p["index"])))
    return F, R, idx, xy, cl, pos_valid


def train(with_pseudo=False, epochs=60, verbose=True):
    import torch
    import torch.nn as nn
    import cv2
    F, R, idx, xy, cl, pv = _load_pairs(with_pseudo)

    # Held-out split BY FRAME INDEX so near-duplicate neighbours cannot straddle the split and
    # flatter the validation number.
    order = np.argsort(idx)
    idx, xy, cl, pv = idx[order], xy[order], cl[order], pv[order]
    n_val = max(20, len(idx) // 5)
    rng0 = np.random.default_rng(7)
    perm = rng0.permutation(len(idx))
    val_sel, tr_sel = perm[:n_val], perm[n_val:]

    models = []
    for mi, seed in enumerate((11, 29)):               # two INDEPENDENT models
        net = build_net(torch, nn, seed)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        rng = np.random.default_rng(1000 + seed)       # different augmentation stream
        best, best_state = 1e9, None
        for ep in range(epochs):
            net.train()
            sel = rng.permutation(tr_sel)
            tot = 0.0
            for k0 in range(0, len(sel), 16):
                batch = sel[k0:k0 + 16]
                ims, tgt = [], []
                for j in batch:
                    im, uv = augment(rng, F[idx[j]], xy[j])
                    ims.append(cv2.resize(im, (IN_W, IN_H)) / 255.0)
                    tgt.append(uv)
                x = torch.tensor(np.array(ims), dtype=torch.float32).unsqueeze(1)
                y = torch.tensor(np.array(tgt), dtype=torch.float32)
                heat, closed_logit = net(x)
                pvb = torch.tensor(pv[batch], dtype=torch.float32)
                pred = soft_argmax(torch, heat)
                if float(pvb.sum()) > 0:
                    loss = nn.functional.smooth_l1_loss(pred[pvb > 0], y[pvb > 0], beta=0.02)
                else:
                    loss = pred.sum() * 0.0
                # Eye-state head: supervise ONLY on frames a human actually judged (cl >= 0).
                # Pseudo-labelled frames carry cl = -1 and are masked out, so the head never
                # learns "the ensemble agreed, therefore open".
                cy = torch.tensor(cl[batch], dtype=torch.float32)
                m = cy >= 0
                if bool(m.any()):
                    loss = loss + 0.2 * nn.functional.binary_cross_entropy_with_logits(
                        closed_logit[m], cy[m])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(batch)
            sched.step()
            net.eval()
            with torch.no_grad():
                ims = [cv2.resize(F[idx[j]].astype(np.float32), (IN_W, IN_H)) / 255.0
                       for j in val_sel]
                x = torch.tensor(np.array(ims), dtype=torch.float32).unsqueeze(1)
                heat, _ = net(x)
                pred_v = soft_argmax(torch, heat).numpy()
                vmask = pv[val_sel] > 0
                err = np.linalg.norm(pred_v[vmask] - xy[val_sel][vmask], axis=1)
                med = float(np.median(err)) if len(err) else float("nan")
            if med < best:
                best, best_state = med, {k: v.clone() for k, v in net.state_dict().items()}
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print("  model %d  ep %3d  train %.4f  val median %.4f (best %.4f)"
                      % (mi, ep, tot / max(len(tr_sel), 1), med, best))
        net.load_state_dict(best_state)
        models.append((net, best))
        if verbose:
            print("  model %d done — held-out median error %.4f frame units (%.1f px of 1280)"
                  % (mi, best, best * 1280))
    torch.save({"m0": models[0][0].state_dict(), "m1": models[1][0].state_dict()},
               os.path.join(DATA, "canthus_models.pt"))
    if verbose:
        print("\n  saved data/canthus_models.pt")
        print("  next: --pseudo  (label the rest where BOTH models agree)")
    return 0


def pseudo(verbose=True):
    """Label the unlabelled remainder — keeping ONLY frames where both models agree.

    This is the step that makes a few hundred hand labels into thousands. Agreement between two
    independently-trained models is evidence in a way a single model's confidence is not: the
    template's failure mode was being confidently wrong, and one net can be too. Two nets being
    wrong in the SAME place is far less likely."""
    import torch
    import torch.nn as nn
    import cv2
    d = np.load(CORPUS)
    F = d["frames"]
    s = np.load(SEED)
    done = set(int(i) for i in s["index"])
    todo = np.array([i for i in range(len(F)) if i not in done], np.int32)
    ck = torch.load(os.path.join(DATA, "canthus_models.pt"))
    nets = []
    for key, seed in (("m0", 11), ("m1", 29)):
        n = build_net(torch, nn, seed); n.load_state_dict(ck[key]); n.eval(); nets.append(n)

    preds = []
    with torch.no_grad():
        for k0 in range(0, len(todo), 64):
            batch = todo[k0:k0 + 64]
            ims = [cv2.resize(F[i].astype(np.float32), (IN_W, IN_H)) / 255.0 for i in batch]
            x = torch.tensor(np.array(ims), dtype=torch.float32).unsqueeze(1)
            outs = []
            for n in nets:
                heat, _ = n(x)
                outs.append(soft_argmax(torch, heat).numpy())
            preds.append(np.stack(outs, 0))
    P = np.concatenate(preds, 1)                     # (2, N, 2)
    disagree = np.linalg.norm(P[0] - P[1], axis=1)
    keep = disagree < AGREE_PX
    xy = P[:, keep].mean(0)
    np.savez_compressed(PSEUDO, index=todo[keep], labels=xy.astype(np.float32))
    if verbose:
        print("== ensemble pseudo-labelling ==")
        print("  candidates      %d" % len(todo))
        print("  models agree    %d (%.0f%%) within %.3f frame units"
              % (keep.sum(), 100.0 * keep.mean(), AGREE_PX))
        print("  disagreement    median %.4f | p90 %.4f" %
              (np.median(disagree), np.percentile(disagree, 90)))
        print("  saved %s" % PSEUDO)
        print("  next: --train --with-pseudo, then --export")
    return 0


def export(verbose=True):
    """Freeze model 0 to plain .npz so the RUNTIME needs numpy only, never torch."""
    import torch
    import torch.nn as nn
    ck = torch.load(os.path.join(DATA, "canthus_models.pt"))
    net = build_net(torch, nn, 11)
    net.load_state_dict(ck["m0"]); net.eval()
    out = {}
    layers = list(net.trunk) + [net.heat, net.closed]
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            out["c%d_w" % i] = layer.weight.detach().numpy()
            out["c%d_b" % i] = layer.bias.detach().numpy()
        elif isinstance(layer, nn.BatchNorm2d):
            # Fold BN into constants: numpy runtime does scale/shift, no BN layer needed.
            g, b = layer.weight.detach().numpy(), layer.bias.detach().numpy()
            m, v = layer.running_mean.numpy(), layer.running_var.numpy()
            out["b%d_scale" % i] = g / np.sqrt(v + layer.eps)
            out["b%d_shift" % i] = b - g * m / np.sqrt(v + layer.eps)
    out["in_hw"] = np.array([IN_H, IN_W], np.int32)
    np.savez_compressed(WEIGHTS, **out)
    if verbose:
        print("exported %d arrays -> %s" % (len(out), WEIGHTS))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="train the canthus landmark model (.venv-train)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--pseudo", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--with-pseudo", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    if a.train:
        sys.exit(train(with_pseudo=a.with_pseudo, epochs=a.epochs))
    if a.pseudo:
        sys.exit(pseudo())
    if a.export:
        sys.exit(export())
    ap.print_help()
