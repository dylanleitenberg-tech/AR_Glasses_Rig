"""live_features.py — the ONE place the live feature vector is assembled.

THE CONTRACT: the vector produced from the real cameras MUST equal
`Config.feature_names` in order AND normalization, or the validated calibrator /
preset / identifier won't transfer. So every live frame and every `--simulate`
frame goes through `assemble_features()`, and the parity self-test below proves it
reproduces `autosim`'s feature ordering bit-for-bit, for all flag combinations.

Per-camera detections are each a normalized (x, y) in [0, 1] of that camera's
sensor — exactly what `optics.PinholeCamera.project` returns and what the CV
detectors (dot_detector / eye_tracker / pupil_tracker) output (origin top-left,
y increasing downward).

Feature order (mirrors autosim.observe / ground_truth and config.feature_names):
    base 8 : [worldL x,y, worldR x,y, eyeL x,y, eyeR x,y]
    +stereo: [eyeL2 x,y, eyeR2 x,y]      iff cfg.use_stereo   (inserted before pupil)
    +pupil : [pupilR x,y]                iff cfg.use_pupil     (always last)
"""
import sys

import numpy as np

from config import Config


def _xy(v, name):
    a = np.asarray(v, float).reshape(-1)
    if a.shape[0] != 2:
        raise ValueError("%s must be a normalized (x, y); got shape %s" % (name, a.shape))
    return a


def assemble_features(worldL, worldR, eyeL, eyeR,
                      eyeL2=None, eyeR2=None, pupilR=None, cfg=None):
    """Per-camera normalized (x,y) -> the model feature vector, in Config order.

    Raises if a feature required by the active flags is missing or the assembled
    length disagrees with `cfg.n_features` (the contract tripwire).
    """
    cfg = cfg or Config()
    blocks = [_xy(worldL, "worldL"), _xy(worldR, "worldR"),
              _xy(eyeL, "eyeL"), _xy(eyeR, "eyeR")]
    if cfg.use_stereo:
        if eyeL2 is None or eyeR2 is None:
            raise ValueError("use_stereo=True needs eyeL2 and eyeR2")
        blocks += [_xy(eyeL2, "eyeL2"), _xy(eyeR2, "eyeR2")]
    if cfg.use_pupil:
        if pupilR is None:
            raise ValueError("use_pupil=True needs pupilR")
        blocks += [_xy(pupilR, "pupilR")]
    feat = np.concatenate(blocks)
    if feat.shape[0] != cfg.n_features:
        raise AssertionError(
            "assembled %d features but Config expects %d (%s)"
            % (feat.shape[0], cfg.n_features, cfg.feature_names))
    return feat


def split_features(feat, cfg=None):
    """Inverse of `assemble_features`: feature vector -> named per-camera (x,y) blocks.

    Used by the `--simulate` live path (the real-sensor mirror of autosim.observe
    feeds split blocks back through assemble_features) and by the parity test.
    """
    cfg = cfg or Config()
    feat = np.asarray(feat, float).reshape(-1)
    if feat.shape[0] != cfg.n_features:
        raise AssertionError("got %d features, Config expects %d"
                             % (feat.shape[0], cfg.n_features))
    out = {"worldL": feat[0:2], "worldR": feat[2:4],
           "eyeL": feat[4:6], "eyeR": feat[6:8]}
    i = 8
    if cfg.use_stereo:
        out["eyeL2"] = feat[i:i + 2]; out["eyeR2"] = feat[i + 2:i + 4]; i += 4
    if cfg.use_pupil:
        out["pupilR"] = feat[i:i + 2]; i += 2
    return out


# ----------------------------------------------------------------------
#  Parity self-test: prove assemble_features reproduces autosim's ordering
# ----------------------------------------------------------------------
def _expected_blocks(sim, subject, dev, P):
    """Recompute each camera's projection EXACTLY as autosim.ground_truth does,
    so we can confirm assemble_features drops each into the right slot."""
    import rig
    R, t = rig.pose_R_t(dev)
    outer = (R @ subject.outer_canthus.T).T + t
    blocks = {
        "worldL": sim.rig["world"][0].project(P),
        "worldR": sim.rig["world"][1].project(P),
        "eyeL": sim.rig["eye"][0].project(outer[0]),
        "eyeR": sim.rig["eye"][1].project(outer[1]),
    }
    if sim.use_stereo:
        blocks["eyeL2"] = sim.rig["eye2"][0].project(outer[0])
        blocks["eyeR2"] = sim.rig["eye2"][1].project(outer[1])
    if sim.use_pupil:
        e = sim.rig["display_eye"]
        cor = (R @ subject.cor.T).T + t
        gaze = P - cor[e]; gaze = gaze / np.linalg.norm(gaze)
        ep = cor[e] + subject.ep_dist * gaze
        ga = np.degrees(np.arccos(np.clip(gaze[2], -1.0, 1.0)))
        lat = np.array([gaze[0], gaze[1], 0.0]); ln = np.linalg.norm(lat)
        if ln > 1e-6:
            ep = ep + rig.PUPIL_WANDER * (ga / 10.0) * (lat / ln)
        blocks["pupilR"] = sim.rig["pupil"].project(ep)
    return blocks


def selftest(verbose=True):
    """For every (use_pupil, use_stereo) combo: the vector assembled from independently
    recomputed per-camera projections must equal autosim.ground_truth's vector, and
    split->assemble must round-trip."""
    import autosim
    if verbose:
        print("== live_features contract self-test ==")
    ok = True
    for use_pupil, use_stereo in [(False, False), (True, False),
                                  (False, True), (True, True)]:
        sim = autosim.Simulator(0, use_pupil=use_pupil, use_stereo=use_stereo)
        cfg = Config(use_pupil=use_pupil, use_stereo=use_stereo)
        rng = np.random.default_rng(7)
        n, maxd = 0, 0.0
        while n < 300:
            subj = sim.new_subject(); dev = sim.seat(); P = sim.world_point()
            g = sim.ground_truth(subj, dev, P)
            if g is None:
                continue
            n += 1
            blocks = _expected_blocks(sim, subj, dev, P)
            feat = assemble_features(cfg=cfg, **blocks)
            maxd = max(maxd, float(np.abs(feat - g[0]).max()))
            # split(assemble(...)) round-trips back to the same blocks
            rt = split_features(feat, cfg)
            for k, v in rt.items():
                maxd = max(maxd, float(np.abs(v - blocks[k]).max()))
        passed = (maxd < 1e-15) and (cfg.n_features == len(g[0]))
        ok = ok and passed
        if verbose:
            print("  use_pupil=%-5s use_stereo=%-5s  n_features=%2d  max|assembled-oracle|=%.1e  [%s]"
                  % (use_pupil, use_stereo, cfg.n_features, maxd,
                     "PASS" if passed else "FAIL"))
    if verbose:
        print("  =>", "CONTRACT OK — live vector matches autosim ordering ✅" if ok
              else "MISMATCH ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest())
