"""Pixel-placement map — *where must the display pixel be?*

For EVERY eye (sampled across the anatomical population), EACH glasses position (a grid
spanning the realistic slip envelope), and 100 dot positions (a 10x10 grid over the page),
this computes the EXACT display pixel that registers the overlay — the physics oracle's
ground truth — and then checks the learned camera-features->pixel models reproduce it.

Two questions it answers:
  1. Do we KNOW the right pixel?  ->  yes: the noise-free oracle (`Simulator.ground_truth`)
     gives it for any eye / position / dot.
  2. Can we REPRODUCE it without the oracle (on the real rig, from camera readings)?
     ->  measured here for the population prior AND the per-eye geometry preset.

Run:  python3 pixel_map.py        (or:  python3 main.py --pixel-map)
Saves the full ground-truth tensor to data/pixel_map.npz.
"""
import os
import numpy as np

import rig
import autosim
import anatomy
from config import Config
from main import new_calibrator
from preset import build_preset

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def dot_grid(nx=10, ny=10):
    """100 dot positions: a regular grid across the page at the mean viewing depth."""
    xs = np.linspace(-rig.DOT_X, rig.DOT_X, nx)
    ys = np.linspace(-rig.DOT_Y, rig.DOT_Y, ny)
    z = rig.DOT_Z_MEAN
    return np.array([[x, y, z] for y in ys for x in xs])        # (nx*ny, 3)


def pose_grid(n_vert=6, lats=(-2.0, 0.0, 2.0)):
    """Glasses positions spanning the realistic slip envelope: the dominant vertical slide
    down the nose (with its correlated pantoscopic pitch) crossed with a few lateral shifts.
    A pose is a deviation-from-rest dev = [drx, dry, drz, dtx, dty, dtz] (deg, deg, deg, mm)."""
    poses = []
    for dty in np.linspace(-2.0, 8.0, n_vert):                 # mm slid down the nose
        drx = dty * rig.PITCH_PER_MM                           # correlated pantoscopic tilt (deg)
        for dtx in lats:                                       # mm lateral shift
            poses.append([drx, 0.0, 0.0, dtx, dty, 0.0])
    return np.array(poses)


def train_prior(cfg, n_subjects=300, samples_per=80, seed=7):
    """Population prior: the universal features->pixel map, fit on many eyes' data exactly
    like the real pipeline. The per-attempt noise in each sample is averaged out by the fit."""
    sim = autosim.Simulator(seed)              # same fixed device extrinsics (DEVICE_SEED)
    X, Y = [], []
    for _ in range(n_subjects):
        subj = sim.new_subject()
        dev = sim.seat()
        got = guard = 0
        while got < samples_per and guard < samples_per * 4:
            guard += 1
            dev = sim.slip(dev)
            o = sim.observe(subj, dev, sim.world_point())
            if o is not None:
                X.append(o[0]); Y.append(o[2]); got += 1
    prior = new_calibrator(cfg)
    prior.fit(np.array(X), np.array(Y))
    return prior


def _errors(model, feats, truepx, onscreen):
    """px-@1080p reproduction error of `model` over every on-screen (eye,pose,dot)."""
    e = []
    idx = np.argwhere(onscreen)
    for i, j, k in idx:
        e.append(np.linalg.norm(model.predict(feats[i, j, k]) - truepx[i, j, k]))
    return np.array(e) * 1080.0


def build_map(n_eyes=100, n_preset=20, seed=0, verbose=True):
    cfg = Config()
    sim = autosim.Simulator(seed)                  # one device -> fixed extrinsics
    dots = dot_grid()                              # (100, 3)
    poses = pose_grid()                            # (P, 6)
    nP, nD = len(poses), len(dots)
    if verbose:
        print("== Pixel-placement map: where must the display pixel be? ==")
        print("  dots:              %d  (10x10 grid over the page)" % nD)
        print("  glasses positions: %d  (vertical slide x lateral, spanning the slip envelope)" % nP)
        print("  eyes:              %d  sampled from the population (oracle gives it for ANY eye)" % n_eyes)
        print("  training the population prior (universal features->pixel map) ...")
    prior = train_prior(cfg)

    desc     = np.zeros((n_eyes, 8))
    truepx   = np.full((n_eyes, nP, nD, 2), np.nan)
    feats    = np.full((n_eyes, nP, nD, 8), np.nan)
    onscreen = np.zeros((n_eyes, nP, nD), bool)

    pop = np.random.default_rng(seed + 1)          # a population of eyes (not the device rng)
    subs = []
    for i in range(n_eyes):
        s = anatomy.sample_subject(pop)
        subs.append(s); desc[i] = s.descriptor()
        for j, dev in enumerate(poses):
            for k, P in enumerate(dots):
                g = sim.ground_truth(s, dev, P)
                if g is not None:
                    feats[i, j, k], truepx[i, j, k] = g
                    onscreen[i, j, k] = True
        if verbose and (i + 1) % 25 == 0:
            print("    ... mapped %d/%d eyes" % (i + 1, n_eyes))

    cov = onscreen.mean() * 100.0

    pe = _errors(prior, feats, truepx, onscreen)                       # population prior

    rng2 = np.random.default_rng(seed + 2)                             # per-eye geometry preset
    sel = rng2.choice(n_eyes, size=min(n_preset, n_eyes), replace=False)
    se = []
    for n, i in enumerate(sel):
        cal = build_preset(subs[i].descriptor())   # identify->physics-sweep keyed to this eye
        m = onscreen[i]
        for j, k in np.argwhere(m):
            se.append(np.linalg.norm(cal.predict(feats[i, j, k]) - truepx[i, j, k]))
        if verbose and (n + 1) % 5 == 0:
            print("    ... presets built %d/%d" % (n + 1, len(sel)))
    se = np.array(se) * 1080.0

    if verbose:
        print("\n  on-screen coverage: %.1f%% of (eye x position x dot) land on the 50 deg display\n" % cov)
        print("  reproduction error vs the oracle ground truth (px @1080p):")
        print("                              median    95th pct      max")
        print("    population prior         %7.1f  %9.1f  %8.1f"
              % (np.median(pe), np.percentile(pe, 95), pe.max()))
        if len(se):
            print("    per-eye geometry preset  %7.1f  %9.1f  %8.1f"
                  % (np.median(se), np.percentile(se, 95), se.max()))
        print("\n  the oracle KNOWS the exact pixel for every eye/position/dot; the prior")
        print("  reproduces it across the whole population from camera features alone, and the")
        print("  per-eye preset (keyed to the identified eye shape) tightens it further.")

    out = os.path.join(_DATA, "pixel_map.npz")
    np.savez_compressed(out, dots=dots, poses=poses, descriptors=desc,
                        true_px=truepx, onscreen=onscreen,
                        descriptor_names=np.array(anatomy.DESCRIPTOR_NAMES))
    if verbose:
        print("\n  saved full ground-truth map -> %s" % out)
        print("    true_px[eye, position, dot] = the normalized display pixel to light up")
        print("    (NaN where that dot is off this display for that eye + position).")
    return {"coverage": cov, "prior_px": pe, "preset_px": se,
            "dots": dots, "poses": poses, "true_px": truepx, "onscreen": onscreen}


if __name__ == "__main__":
    build_map()
