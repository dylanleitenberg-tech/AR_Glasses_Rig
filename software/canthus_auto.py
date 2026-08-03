"""Automated canthus labelling: a PROPOSER guesses, a CORRECTOR corrects, loop until confirmed.

THE PATTERN IS THIS PROJECT'S OWN
    pixel_sweep.py already does exactly this — a GuessingAI proposes, a UserAI corrects, and a
    pixel is only accepted once it comes back right TWICE IN A ROW. main.py's live loop is the
    human version: predict, nudge, approve. This module applies the same idea to labelling, so a
    corpus can be labelled with no hand-clicking.

        propose -> correct -> propose -> ... -> two consecutive confirmations -> accept
                                             -> too many rounds              -> reject

WHY A THIRD JUDGE THAT IS NOT LEARNED
    Agreement between two estimators is evidence only if their errors are uncorrelated. Template
    matching taught us the failure mode the hard way: it was CONFIDENTLY wrong, and wrong labels
    scored HIGHER margin than right ones (34.8% of eyeL beyond 0.20 frame units, mean margin 0.486
    against a corpus median of 0.414). Two models trained on the same corpus can inherit exactly
    that bias and confirm each other into it.

    So every proposal is also checked against the ANATOMICAL PRIOR — where rig.py's optics say an
    inner canthus can physically land, given the population face model swept over the whole glasses
    pose grid. That judge is physics, not data; it cannot be argued into a consensus. Measured
    prior (300 faces x 18 poses), converted to REAL SENSOR coordinates:

        u  median 0.834   1-99% [0.778, 0.891]
        v  median 0.362   1-99% [0.110, 0.620]

    It is a tight box, and it is decisive: the blank-skin locks that ruined the first corpus sat
    at u~0.15 against a prior floor of 0.778. No confidence threshold caught those. The prior
    rejects them outright.

    MIND THE UNITS — this cost two false alarms. optics.PinholeCamera normalises u AND v by the
    same focal, i.e. a SQUARE frame, while a measured label is normalised over the real 800-px
    height. Raw prior v is [0.256, 0.575]; converted it is [0.110, 0.620]. Comparing the raw
    figure against measurements made the rig look like it disagreed with the simulator when
    nothing was wrong with either. build_prior() does the conversion; do not undo it.

CLOSED EYES SKIP, AND THAT ALSO NEEDS CONFIRMING
    A frame judged closing/closed is skipped rather than labelled — but a single opinion is not
    enough to discard data, so the closed verdict must ALSO be confirmed twice before it counts.
    Same standard as an accept: one estimator saying so is a proposal, not a verdict.

BOOTSTRAP
    On the first pass there are no trained models, so proposer and corrector are classical:
    template matching restricted to the prior box, corrected by a pupil-anchored geometric
    estimate (pupil_tracker fits a dark-pupil ellipse and is far more reliable than any corner
    detector — the iris is big, dark and unambiguous). Once models exist they slot into the same
    interfaces and the loop is unchanged.

    python3 canthus_auto.py --run          # label the corpus automatically
    python3 canthus_auto.py --selftest
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CORPUS = os.path.join(DATA, "canthus_corpus.npz")
AUTO = os.path.join(DATA, "canthus_auto.npz")
PRIOR = os.path.join(DATA, "canthus_prior.npz")

CONFIRM_N = 2          # consecutive confirmations required, as pixel_sweep does
MAX_ROUNDS = 8         # give up rather than loop forever on an ambiguous frame
CONFIRM_TOL = 0.012    # a correction smaller than this counts as "the corrector agrees"
CLOSED_AREA_FRAC = 0.35  # pupil area below this fraction of the session's OPEN median -> closing/closed


# ----------------------------------------------------------------------------------
#  The anatomical prior — computed from rig.py optics, cached, never learned
# ----------------------------------------------------------------------------------
def build_prior(n_faces=300, seed=31, cache=True, verbose=False):
    """Where can an inner canthus physically land in the eye cam? Swept over the population face
    model x the whole glasses pose grid, projected through the real rig optics onto the 16:10
    sensor. This is the judge that cannot be fooled by a confident wrong answer."""
    if cache and os.path.exists(PRIOR):
        d = np.load(PRIOR)
        return {k: float(d[k]) for k in d.files}
    import rig
    import pixel_map
    from landmark_test import make_subjects
    cam = rig.build("outer")["eye"][1]          # AIM is outer (as printed); landmark is inner
    subs = make_subjects(n_faces, seed)
    uv = []
    for s in subs:
        pts = s.inner_canthus if rig.TRACKED_LANDMARK == "inner" else s.outer_canthus
        for dev in pixel_map.pose_grid():
            R, t = rig.pose_R_t(dev)
            p = cam.project(((R @ pts.T).T + t)[1])
            if p is not None:
                uv.append(p)
    uv = np.array(uv)

    # SQUARE-FRAME -> REAL SENSOR coordinates. optics.PinholeCamera normalises u AND v by the
    # same focal ("half-size 0.5 spans half-FOV"), i.e. it models a SQUARE image. The OV9281 is
    # 1280x800, so the real sensor covers the full width but only the middle 800/1280 = 62.5%
    # vertically: v_square in [0.1875, 0.8125]. A measured label, by contrast, is normalised over
    # the ACTUAL 800-px image height, so it spans [0,1].
    #
    # Comparing the two directly is meaningless, and doing so is what produced the phantom
    # "vertical discrepancy" I chased twice: raw prior v [0.256,0.575] vs measured v ~0.61-0.75
    # looked like the rig disagreeing with the simulator, when it was a units error in this
    # function. u was never affected because the square's width IS the sensor width — which is
    # exactly why u agreed (0.809 vs 0.834) while v did not. Same conversion landmark_test.framing
    # already applies.
    lo = 0.5 - 0.5 * (800 / 1280.0)          # 0.1875
    hi = 0.5 + 0.5 * (800 / 1280.0)          # 0.8125
    on_sensor = (uv[:, 1] >= lo) & (uv[:, 1] <= hi) & (uv[:, 0] >= 0) & (uv[:, 0] <= 1)
    uv = uv[on_sensor]                        # a canthus off the sensor cannot be detected anyway
    uv[:, 1] = (uv[:, 1] - lo) / (hi - lo)    # -> [0,1] over the real 800-px height

    out = dict(u_lo=float(np.percentile(uv[:, 0], 1)), u_hi=float(np.percentile(uv[:, 0], 99)),
               v_lo=float(np.percentile(uv[:, 1], 1)), v_hi=float(np.percentile(uv[:, 1], 99)),
               u_med=float(np.median(uv[:, 0])), v_med=float(np.median(uv[:, 1])),
               on_sensor=float(on_sensor.mean()))
    if cache:
        os.makedirs(DATA, exist_ok=True)
        np.savez(PRIOR, **out)
    if verbose:
        print("prior u [%.3f,%.3f] v [%.3f,%.3f]" % (out["u_lo"], out["u_hi"],
                                                     out["v_lo"], out["v_hi"]))
    return out


def in_prior(uv, prior, mirrored, pad=0.06):
    """Is this position anatomically possible? `mirrored` handles the left camera, whose image is
    the mirror of the geometry the prior was computed in. `pad` allows for the prior being a
    POPULATION sweep — one real face sits somewhere inside it, not at its centre."""
    u = (1.0 - uv[0]) if mirrored else uv[0]
    return (prior["u_lo"] - pad <= u <= prior["u_hi"] + pad
            and prior["v_lo"] - pad <= uv[1] <= prior["v_hi"] + pad)


# ----------------------------------------------------------------------------------
#  Eye state — closing/closed frames are skipped, but the verdict must be confirmed too
# ----------------------------------------------------------------------------------
def dark_mass(gray):
    """Fraction of the frame darker than 0.55x its own median. Kept for diagnostics only.

    TWO REJECTED VERSIONS, recorded so neither gets rebuilt:

      1. A PERCENTILE threshold (`gray <= percentile(gray, 12)`) selects ~12% of pixels by
         definition, whatever the image holds — identical for an open eye and a shut one. It
         measured nothing. The selftest caught it.
      2. Even median-scaled, whole-frame darkness does NOT detect a closed eye here. Measured
         across the real corpus it spans only 0.097-0.16, p95/p5 = 1.3-1.5x, with ZERO frames
         below any sane threshold — because frame darkness is dominated by the glasses frame and
         shadows, not by the iris. Use eye_openness() instead.
    """
    med = float(np.median(gray))
    thr = 0.55 * med
    return float((gray <= thr).mean()), thr


def eye_openness(gray, tracker=None):
    """Pupil ellipse AREA — the signal that actually separates open from closed.

    The iris/pupil is the one large, unambiguous dark structure in these frames, and it is the
    thing a closing lid removes. Measured across the real corpus it separates cleanly where
    whole-frame darkness does not: median area 0.020 (eyeL) / 0.033 (eyeR) against a 5th
    percentile of EXACTLY ZERO — i.e. a real population of frames with no detectable pupil at all.
    That is the closed set.

    Area, not merely `ok`: a half-closed lid still yields a fit, just a much smaller one, so area
    catches CLOSING as well as closed — which is what Dylan asked for, since a mid-blink frame is
    as unusable as a shut one."""
    from pupil_tracker import PupilTracker
    trk = tracker or PupilTracker()
    r = trk.detect(gray)
    if not getattr(r, "ok", False):
        return 0.0
    return float(r.axes[0] * r.axes[1])


def looks_closed(gray, ref_open, tracker=None):
    """Closing/closed relative to this session's OPEN baseline.

    Relative, never absolute: exposure and seating swing hard between sessions on these
    ambient-lit cameras, so a fixed area threshold would mean something different every run."""
    a = eye_openness(gray, tracker)
    return a < CLOSED_AREA_FRAC * ref_open, a


# ----------------------------------------------------------------------------------
#  Bootstrap proposer / corrector (classical). Trained models slot into the same interface.
# ----------------------------------------------------------------------------------
class TemplateProposer:
    """Proposes by template match, but ONLY inside the prior box. Restricting the search is what
    makes correlation usable: unconstrained, it wanders onto blank skin with high confidence."""

    def __init__(self, tmpl, prior, mirrored):
        self.t, self.prior, self.mirrored = tmpl, prior, mirrored

    def __call__(self, gray, cv2, hint=None):
        H, W = gray.shape[:2]
        th, tw = self.t.shape[:2]
        if self.mirrored:
            u0, u1 = 1.0 - self.prior["u_hi"] - 0.06, 1.0 - self.prior["u_lo"] + 0.06
        else:
            u0, u1 = self.prior["u_lo"] - 0.06, self.prior["u_hi"] + 0.06
        v0, v1 = self.prior["v_lo"] - 0.06, self.prior["v_hi"] + 0.06
        x0, x1 = max(0, int(u0 * W) - tw // 2), min(W, int(u1 * W) + tw // 2)
        y0, y1 = max(0, int(v0 * H) - th // 2), min(H, int(v1 * H) + th // 2)
        if x1 - x0 < tw + 2 or y1 - y0 < th + 2:
            return None, 0.0
        roi = gray[y0:y1, x0:x1]
        res = cv2.matchTemplate(roi, self.t, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        return ((x0 + loc[0] + tw / 2.0) / W, (y0 + loc[1] + th / 2.0) / H), float(peak)


class PupilCorrector:
    """Corrects a proposal using the pupil as an anchor.

    The pupil is the one thing in these frames that is genuinely easy to find — a large dark
    ellipse — and pupil_tracker already fits it and self-tests to ~1px. The canthus sits at a
    roughly fixed direction from it for a given camera, so the pupil gives an INDEPENDENT estimate
    that does not depend on corner texture at all. That independence is the point: it fails in
    different ways from template matching, so agreement between them means something.
    """

    def __init__(self, prior, mirrored, offset=None):
        self.prior, self.mirrored = prior, mirrored
        self.offset = offset           # learned online: median (canthus - pupil) once confirmed

    def __call__(self, gray, cv2, proposal):
        from pupil_tracker import PupilTracker
        res = PupilTracker().detect(gray)
        if not getattr(res, "ok", False):
            return None
        if self.offset is None:
            # NO FALLBACK CONSTANT. An earlier version returned the prior's median here, which
            # made the corrector a CONSTANT FUNCTION — the loop then averaged every proposal onto
            # that constant and "confirmed" 89% of frames with a standard deviation of 0.002.
            # Convergence was guaranteed and carried zero information about the image. A corrector
            # that cannot see the frame must ABSTAIN, not answer.
            return None
        pu, pv = res.pupil
        return (pu + self.offset[0], pv + self.offset[1])


# ----------------------------------------------------------------------------------
#  The loop
# ----------------------------------------------------------------------------------
def converge(gray, cv2, proposer, corrector, prior, mirrored,
             confirm_n=CONFIRM_N, max_rounds=MAX_ROUNDS, tol=CONFIRM_TOL):
    """propose -> correct -> ... until the corrector confirms `confirm_n` times running.

    Returns (uv, rounds, status) where status is 'confirmed' | 'no-consensus' | 'off-prior' |
    'no-proposal'. A frame that never converges is REJECTED, not accepted with low confidence —
    the whole failure this module exists to prevent was accepting confident garbage.
    """
    uv, peak = proposer(gray, cv2)
    if uv is None:
        return None, 0, "no-proposal"
    hits = 0
    for r in range(1, max_rounds + 1):
        c = corrector(gray, cv2, uv)
        if c is None:
            return None, r, "no-consensus"
        d = float(np.hypot(c[0] - uv[0], c[1] - uv[1]))
        if d <= tol:
            hits += 1
            if hits >= confirm_n:
                if not in_prior(uv, prior, mirrored):
                    return None, r, "off-prior"
                return uv, r, "confirmed"
        else:
            hits = 0
            uv = ((uv[0] + c[0]) / 2.0, (uv[1] + c[1]) / 2.0)   # move halfway, damped
    return None, max_rounds, "no-consensus"


def run(corpus=CORPUS, out=AUTO, verbose=True):
    import cv2
    if not os.path.exists(corpus):
        print("!! no corpus at %s — run canthus_data.py --collect" % corpus)
        return 1
    prior = build_prior()
    d = np.load(corpus)
    F, R = d["frames"], d["roles"]

    tmpl = {}
    for role in ("eyeL", "eyeR"):
        p = os.path.join(DATA, "templates", "%s.png" % role)
        t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if t is None:
            print("!! missing %s" % p)
            return 1
        # corpus frames are stored downscaled, so the template must be scaled to match
        sc = F.shape[2] / 1280.0
        tmpl[role] = cv2.resize(t, (max(8, int(t.shape[1] * sc)), max(8, int(t.shape[0] * sc))))

    # Session OPEN baseline = median pupil area for that eye. Computed once, over a sample.
    from pupil_tracker import PupilTracker
    _trk = PupilTracker()
    ref_open = {}
    for rid in (0, 1):
        m = R == rid
        if m.any():
            sample = np.where(m)[0][:250]
            ref_open[rid] = float(np.median([eye_openness(F[i], _trk) for i in sample]))
    if verbose:
        print("== eye-open baseline (median pupil area) ==")
        for rid, name in ((0, "eyeL"), (1, "eyeR")):
            if rid in ref_open:
                print("  %s: %.5f  -> closing/closed below %.5f"
                      % (name, ref_open[rid], CLOSED_AREA_FRAC * ref_open[rid]))
        print()

    # ---- PASS 1: learn the canthus-to-pupil offset, per role ----------------------------
    # The corrector has to be a function OF THE IMAGE. It reads the pupil (reliable: found in
    # 96-100% of real frames) and adds a fixed offset to reach the canthus. That offset is not
    # known a priori, so it is estimated here from the frames where the template proposal is
    # ANATOMICALLY PLAUSIBLE — the prior does the filtering that confidence could not.
    from pupil_tracker import PupilTracker
    trk = PupilTracker()
    offs = {0: [], 1: []}
    for i in range(len(F)):
        rid = int(R[i])
        mirrored = (rid == 0)
        prop = TemplateProposer(tmpl["eyeL" if rid == 0 else "eyeR"], prior, mirrored)
        uv, _ = prop(F[i], cv2)
        if uv is None or not in_prior(uv, prior, mirrored):
            continue
        res = trk.detect(F[i])
        if not getattr(res, "ok", False):
            continue
        offs[rid].append((uv[0] - res.pupil[0], uv[1] - res.pupil[1]))
    offset = {}
    for rid in (0, 1):
        if len(offs[rid]) >= 20:
            offset[rid] = tuple(np.median(np.array(offs[rid]), axis=0))
    if verbose:
        print("== pass 1: canthus-to-pupil offset ==")
        for rid, name in ((0, "eyeL"), (1, "eyeR")):
            if rid in offset:
                print("  %s: offset (%.3f, %.3f) from %d in-prior frames"
                      % (name, offset[rid][0], offset[rid][1], len(offs[rid])))
            else:
                print("  %s: NOT ESTABLISHED — only %d in-prior frames (need 20)"
                      % (name, len(offs[rid])))
        print()

    # ---- PASS 2: propose / correct / confirm --------------------------------------------
    idx, labels, rounds, status = [], [], [], []
    n_closed = 0
    for i in range(len(F)):
        g = F[i]
        rid = int(R[i])
        mirrored = (rid == 0)
        closed, _ = looks_closed(g, ref_open.get(rid, 0.02), _trk)
        if closed:
            # A closed verdict is a PROPOSAL too: confirm it against neighbours before discarding.
            near = [j for j in (i - 1, i + 1) if 0 <= j < len(F) and int(R[j]) == rid]
            votes = sum(1 for j in near
                        if looks_closed(F[j], ref_open.get(rid, 0.02), _trk)[0])
            if votes >= 1:
                n_closed += 1
                status.append("closed")
                continue
        prop = TemplateProposer(tmpl["eyeL" if rid == 0 else "eyeR"], prior, mirrored)
        corr = PupilCorrector(prior, mirrored, offset=offset.get(rid))
        uv, r, st = converge(g, cv2, prop, corr, prior, mirrored)
        status.append(st)
        if st == "confirmed":
            idx.append(i); labels.append(uv); rounds.append(r)

    idx = np.array(idx, np.int32)
    if len(idx):
        np.savez_compressed(out, index=idx, labels=np.array(labels, np.float32),
                            rounds=np.array(rounds, np.int32))
    if verbose:
        from collections import Counter
        c = Counter(status)
        print("== automated labelling ==")
        print("  frames            %d" % len(F))
        print("  confirmed         %d (%.0f%%)  median %d rounds"
              % (c["confirmed"], 100.0 * c["confirmed"] / len(F),
                 int(np.median(rounds)) if rounds else 0))
        print("  skipped: closed   %d" % c["closed"])
        print("  rejected: off-prior    %d" % c["off-prior"])
        print("  rejected: no-consensus %d" % c["no-consensus"])
        print("  rejected: no-proposal  %d" % c["no-proposal"])
        if len(idx):
            print("  saved %s" % out)
        else:
            print("  NOTHING confirmed — do not proceed; the inputs are wrong, not the loop")
    return 0


def selftest(verbose=True):
    if verbose:
        print("== canthus_auto self-test (no hardware) ==")
    checks = []

    prior = dict(u_lo=0.778, u_hi=0.891, v_lo=0.256, v_hi=0.575, u_med=0.834, v_med=0.414)

    # The prior must ACCEPT a plausible canthus and REJECT the blank-skin failure that ruined the
    # first corpus (u~0.15 against a floor of 0.761). This is the check that has teeth.
    checks.append(("prior accepts u=0.834 v=0.414, rejects the u=0.15 blank-skin lock",
                   in_prior((0.834, 0.414), prior, False)
                   and not in_prior((0.150, 0.414), prior, False)))

    # Mirroring: the left camera sees the mirror image, so 1-u must be tested.
    checks.append(("mirrored camera: u=0.166 accepted as mirrored, rejected as un-mirrored",
                   in_prior((0.166, 0.414), prior, True)
                   and not in_prior((0.166, 0.414), prior, False)))

    # The loop must CONVERGE when proposer and corrector agree...
    class P:
        def __call__(self, g, cv2, hint=None): return (0.834, 0.414), 0.9

    class C:
        def __call__(self, g, cv2, uv): return (0.836, 0.416)
    uv, r, st = converge(None, None, P(), C(), prior, False)
    checks.append(("converges to 'confirmed' when the two agree (%d rounds)" % r,
                   st == "confirmed" and uv is not None))

    # ...and must REJECT rather than accept when they never settle. A frame that will not
    # converge is thrown away; accepting it with low confidence is the exact bug this prevents.
    class C2:
        def __init__(self): self.k = 0

        def __call__(self, g, cv2, uv):
            self.k += 1
            return (0.30, 0.80) if self.k % 2 else (0.95, 0.20)
    uv2, r2, st2 = converge(None, None, P(), C2(), prior, False)
    checks.append(("rejects with 'no-consensus' when they never settle (not a low-conf accept)",
                   st2 == "no-consensus" and uv2 is None))

    # Agreement is NOT enough on its own: two estimators confirming an impossible position must
    # still be thrown out by the prior. This is the template failure, reproduced.
    class P3:
        def __call__(self, g, cv2, hint=None): return (0.150, 0.414), 0.99

    class C3:
        def __call__(self, g, cv2, uv): return (0.151, 0.414)
    uv3, _, st3 = converge(None, None, P3(), C3(), prior, False)
    checks.append(("two estimators AGREEING on an impossible spot -> 'off-prior', not accepted",
                   st3 == "off-prior" and uv3 is None))

    # Closed detection, on synthetic eyes from pupil_tracker's own generator. Openness is PUPIL
    # AREA, not frame darkness: measured on the real corpus, whole-frame darkness spans only
    # 0.097-0.16 (p95/p5 = 1.3-1.5x) and flags nothing, because frame darkness is dominated by the
    # glasses frame and shadows. Pupil area separates cleanly — median 0.020-0.033 against a 5th
    # percentile of exactly zero.
    from pupil_tracker import synth_eye
    open_eye = synth_eye(W=320, H=200, pupil=(0.5, 0.5), pupil_r=0.11)
    if open_eye.ndim == 3:
        open_eye = open_eye[:, :, 0]
    shut = np.full((200, 320), 120, np.uint8)           # lid closed: no dark pupil at all
    ref = eye_openness(open_eye)
    a_shut = eye_openness(shut)
    checks.append(("openness is pupil AREA: open %.4f vs shut %.4f, shut flagged closed"
                   % (ref, a_shut),
                   ref > 0.0 and a_shut < CLOSED_AREA_FRAC * ref
                   and not looks_closed(open_eye, ref)[0] and looks_closed(shut, ref)[0]))

    # The dead detector must stay dead: whole-frame darkness must NOT be used as the signal.
    # Recorded as a check because it looked reasonable and measured nothing.
    dm_open, _ = dark_mass(open_eye)
    dm_shut, _ = dark_mass(shut)
    checks.append(("whole-frame darkness does NOT separate open from shut (%.3f vs %.3f)"
                   % (dm_open, dm_shut), abs(dm_open - dm_shut) < 0.30))

    ok = all(x for _, x in checks)
    if verbose:
        for name, x in checks:
            print("  [%s] %s" % ("PASS" if x else "FAIL", name))
        print("  =>", "CANTHUS AUTO OK — agreement alone never accepts ✅" if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="automated propose/correct/confirm canthus labelling")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prior", action="store_true", help="print the anatomical prior and exit")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.prior:
        print(build_prior(cache=False, verbose=True)); sys.exit(0)
    if a.run:
        sys.exit(run())
    ap.print_help()
