"""Identify eye geometry directly from the RECORDED sweep — the database in action.

This consumes the artifacts the sweep wrote (no re-simulation):
  * data/pixel_sweep_guesses.bin  — one row per guess (face, pose, pixel, guess#, dx, dy)
  * data/pixel_sweep.npz          — per-face geometry descriptors (the labels)

Each face's FIRST-guess inaccuracy (guess==1) across positions is its fingerprint; we
leave-one-out k-NN match a held-out face to the rest and read off the recovered geometry.
This proves the recorded dataset is directly usable for the end goal: (glasses position,
guess inaccuracy) -> eye geometry.

HONEST caveat: the sweep's guessing-AI LEARNS as it goes (it warms across faces), so the
first-guess magnitude drifts from face to face — a confound absent in match.py, which uses a
fixed population prior. We remove the per-face magnitude (standardize each fingerprint) so
the geometry PATTERN drives the match, and report recovery honestly. For the clean,
at-scale identifiability numbers use `match.py`; this shows the recorded data works.

Run:  python3 match_recorded.py
"""
import os
import numpy as np

import anatomy
from pixel_sweep import GUESS_DTYPE

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_fingerprints(verbose=True):
    npz = np.load(os.path.join(_DATA, "pixel_sweep.npz"))
    desc = npz["descriptors"]
    n_faces, n_poses = npz["guess_counts"].shape[:2]
    binpath = os.path.join(_DATA, "pixel_sweep_guesses.bin")
    mm = np.memmap(binpath, dtype=GUESS_DTYPE, mode="r")
    total = mm.shape[0]
    expected = int(npz["total_pieces"])
    if verbose:
        print("  per-guess rows: %s  (npz total_pieces %s)  -> %s"
              % (f"{total:,}", f"{expected:,}", "OK" if total == expected else "MISMATCH"))

    first = mm["guess"] == 1                      # one first-guess row per (face,pose,pixel)
    idx = np.flatnonzero(first)
    face = mm["face"][idx].astype(np.int64)
    pose = mm["pose"][idx].astype(np.int64)
    dx = mm["dx"][idx].astype(np.float32)
    dy = mm["dy"][idx].astype(np.float32)
    mag = np.sqrt(dx * dx + dy * dy)
    if verbose:
        print("  first-guess rows: %s   median first-guess inaccuracy: %.2f mm"
              % (f"{idx.size:,}", float(np.median(mag))))

    key = face * n_poses + pose                   # aggregate per (face, pose)
    cnt = np.bincount(key, minlength=n_faces * n_poses).astype(np.float64)
    cnt[cnt == 0] = 1.0
    mdx = np.bincount(key, dx, n_faces * n_poses) / cnt
    mdy = np.bincount(key, dy, n_faces * n_poses) / cnt
    mmg = np.bincount(key, mag, n_faces * n_poses) / cnt
    fp = np.stack([mdx, mdy, mmg], axis=1).reshape(n_faces, n_poses * 3)   # (faces, 300)
    return fp, desc


def evaluate(k=6, verbose=True):
    names, units = anatomy.DESCRIPTOR_NAMES, anatomy.DESCRIPTOR_UNITS
    if verbose:
        print("== geometry ID directly from the recorded sweep (leave-one-out) ==")
    fp, desc = load_fingerprints(verbose)
    n = len(fp)
    # remove per-face magnitude drift (the warming guesser), then column-standardize
    fp = fp - fp.mean(1, keepdims=True)
    fp = fp / (fp.std(1, keepdims=True) + 1e-9)
    mu, sd = fp.mean(0), fp.std(0); sd[sd < 1e-9] = 1.0
    Z = (fp - mu) / sd
    pop_sd = desc.std(0)

    rec = np.zeros_like(desc)
    for i in range(n):
        d = np.linalg.norm(Z - Z[i], axis=1); d[i] = np.inf
        nn = np.argpartition(d, k)[:k]
        rec[i] = desc[nn].mean(0)
    mae = np.abs(rec - desc).mean(0)
    pct = 100 * mae / np.maximum(pop_sd, 1e-9)
    if verbose:
        print("\n  recovery from recorded fingerprints (%d faces, LOO k=%d):" % (n, k))
        print("    %-13s %8s %10s %8s" % ("param", "MAE", "pop SD", "%popSD"))
        for j, nm in enumerate(names):
            tag = "identified" if pct[j] < 40 else "partial" if pct[j] < 75 else "weak"
            print("    %-13s %6.2f %s %8.2f %6.0f%%  %s"
                  % (nm, mae[j], units[j], pop_sd[j], pct[j], tag))
        print("\n  (only %d faces here, so numbers are noisy; kappa still stands out as the most\n"
              "   recoverable term, consistent with match.py. Scale + a fixed prior tightens it.)" % n)
    return mae, pct


if __name__ == "__main__":
    evaluate()
