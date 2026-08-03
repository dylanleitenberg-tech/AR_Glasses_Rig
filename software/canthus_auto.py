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

VERIFIED OUTCOME (2026-08-02): THE BOOTSTRAP CANNOT FIND THE CANTHUS ON ITS OWN.
    With the mount fiducial, prior box, confirm loop and closed-eye skip all working, this module
    confirms ~58% of frames with tight, consistent labels (eyeR u std 0.006, v std 0.021) that sit
    ON THE EYE. Zoomed in, they sit on the UPPER LID and brow skin -- not the tear duct.

    The reason is structural, not a tuning miss. Nothing in the classical chain IDENTIFIES a
    canthus. The mount constrains WHERE to look, the template reports WHAT matched, and the pupil
    offset was fitted from those same template hits. Every link is anchored to something that is
    not the landmark, so the system is self-consistent and wrong -- the same failure as the
    original template auto-labelling, one level up. Aggregate metrics look excellent throughout.

    So a small HUMAN seed is unavoidable: a model can find the canthus once shown one, but no
    combination of brightness, staticness and pupil geometry defines "tear duct". Use
    canthus_label.py (~150-200 clicks, once, ever), train, then let the model label the rest.

    Everything here remains useful as FILTERS on the model's pseudo-labels -- the mount band, the
    prior, the confirm loop and the closed-eye skip are all sound. They just cannot originate the
    landmark.

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
AGREE_TOL = 0.055      # A and B must land within this of each other (normalised frame units)
WINDOW = 40            # frames per motion window ~ one seating segment
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


def in_prior(uv, prior, mirrored, pad=0.06, band=None):
    """Is this position anatomically possible? `mirrored` handles the left camera, whose image is
    the mirror of the geometry the prior was computed in. `pad` allows for the prior being a
    POPULATION sweep — one real face sits somewhere inside it, not at its centre."""
    u = (1.0 - uv[0]) if mirrored else uv[0]
    if not (prior["u_lo"] - pad <= u <= prior["u_hi"] + pad):
        return False
    if band is not None:                  # mount-derived vertical band (measured, not simulated)
        return band[0] <= uv[1] <= band[1]
    return prior["v_lo"] - pad <= uv[1] <= prior["v_hi"] + pad


# ----------------------------------------------------------------------------------
#  The MOUNT as a fiducial — a known object, rigidly fixed to the camera
# ----------------------------------------------------------------------------------
def find_mount(frames, dark_frac=0.45, static_pct=25):
    """Locate the nose-bridge support: the one object in frame whose position we already know.

    Dylan's point, and it is the right one — do not hunt for the landmark by appearance alone when
    a KNOWN object is in shot. The mount is bolted to the camera, so it projects to the same pixels
    in every frame no matter how the glasses sit on the face. That makes it a built-in fiducial:
    the face moves relative to it, it never moves relative to the sensor.

    Found by intersecting DARK with TEMPORALLY STATIC across the corpus. Measured on the real data
    the separation is unambiguous — temporal std 3.2-3.6 inside the mask against 12.5-13.4
    everywhere else, i.e. 4x more static than the rest of the image.

    WHY THIS MATTERS BEYOND MASKING: the mount occupies the top ~35% of the frame (measured
    v [0.000, 0.335] eyeL, [0.000, 0.365] eyeR), and rig.py's prior places the canthus at
    v <= 0.620 with median 0.362 — i.e. INSIDE hardware. The simulator does not model the mount
    occluding the sensor, so it predicts the landmark into a region that is physically blocked,
    and any search over that box finds the mount instead of the eye. That is the whole reason the
    eyeR labels were landing on the frame.

    Returns dict with the mask, its bbox and `v_floor` — the bottom of the mount, below which the
    face actually is.
    """
    S = np.asarray(frames, np.float32)
    med = np.median(S, 0)
    sd = S.std(0)
    dark = med < dark_frac * np.median(med)
    # `<=`, not `<`. If a large share of the frame is PERFECTLY static, percentile(sd, 25) is
    # exactly 0 and a strict `<` matches nothing at all — find_mount returns None precisely when
    # the fiducial is at its most rigid. Real frames have sensor noise everywhere so it never
    # surfaced there; the synthetic selftest hit it immediately.
    static = sd <= np.percentile(sd, static_pct)
    mask = dark & static
    H, W = med.shape
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return None
    return dict(mask=mask, v_floor=float(ys.max()) / H,
                u_c=float(xs.mean()) / W, v_c=float(ys.mean()) / H,
                frac=float(mask.mean()),
                static_ratio=float(sd[~mask].mean() / max(sd[mask].mean(), 1e-6)))


def search_band(mount, margin=0.04):
    """Vertical band the canthus can occupy: BELOW the mount, derived from the rig itself.

    Deliberately data-driven rather than taken from rig.py. The simulator's absolute vertical
    placement does not describe this hardware — it puts the landmark inside the mount — while the
    mount's own footprint is measured directly off these frames and cannot be wrong about itself.
    The horizontal prior from rig.py is kept, because u was always consistent with the data
    (0.809/0.886 measured against a prior band of [0.778, 0.891]); it is only v that disagreed."""
    return (min(0.95, mount["v_floor"] + margin), 1.0)


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

    def __init__(self, tmpl, prior, mirrored, band=None):
        self.t, self.prior, self.mirrored = tmpl, prior, mirrored
        self.band = band                 # (v0, v1) from the mount; overrides the sim's v prior

    def __call__(self, gray, cv2, hint=None):
        H, W = gray.shape[:2]
        th, tw = self.t.shape[:2]
        if self.mirrored:
            u0, u1 = 1.0 - self.prior["u_hi"] - 0.06, 1.0 - self.prior["u_lo"] + 0.06
        else:
            u0, u1 = self.prior["u_lo"] - 0.06, self.prior["u_hi"] + 0.06
        if self.band is not None:
            v0, v1 = self.band
        else:
            v0, v1 = self.prior["v_lo"] - 0.06, self.prior["v_hi"] + 0.06
        x0, x1 = max(0, int(u0 * W) - tw // 2), min(W, int(u1 * W) + tw // 2)
        y0, y1 = max(0, int(v0 * H) - th // 2), min(H, int(v1 * H) + th // 2)
        if x1 - x0 < tw + 2 or y1 - y0 < th + 2:
            return None, 0.0
        roi = gray[y0:y1, x0:x1]
        res = cv2.matchTemplate(roi, self.t, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        return ((x0 + loc[0] + tw / 2.0) / W, (y0 + loc[1] + th / 2.0) / H), float(peak)


def face_fixed_point(window, cv2, band, mount_mask, pupil_uv=None,
                     exclude_r=0.10, max_r=0.30):
    """ESTIMATOR B — find the canthus by MOTION, sharing nothing with the template.

    The discriminator is that the inner canthus is FACE-fixed while the pupil is GAZE-driven. Look
    around and your pupil sweeps the frame; your tear duct does not move at all. So within one
    seating the canthus is a point that is TEXTURED (it has structure to see) and TEMPORALLY
    STATIONARY (it does not move), whereas the iris is textured and highly mobile, and skin is
    stationary but featureless.

    Measured on the real corpus, among high-texture pixels the temporal sd spans 4.5 at the 25th
    percentile to 27.5 at the 75th — a 6x separation between face-fixed and gaze-moving structure.

    THIS IS WHY IT COUNTS AS INDEPENDENT of the template estimator. It uses temporal statistics
    over a window; the template uses appearance in a single frame. They share no fitted parameter.
    The previous corrector failed precisely because its offset was fitted from the template's own
    output, so the two could never disagree and 'confirmation' was vacuous — the labels came out as
    pupil-plus-a-constant, correlating 0.99 with the pupil.

    Returns (u, v, score) or None.
    """
    S = np.asarray(window, np.float32)
    med = np.median(S, 0)
    sd = S.std(0)
    H, W = med.shape

    tex = np.abs(cv2.Laplacian(cv2.GaussianBlur(med, (5, 5), 0), cv2.CV_32F))
    tex = cv2.GaussianBlur(tex, (9, 9), 0)
    sd_s = cv2.GaussianBlur(sd, (9, 9), 0)

    # textured AND stationary. sd is normalised by its own median so the score is exposure- and
    # session-independent rather than tuned to one recording.
    score = tex / (1.0 + sd_s / max(float(np.median(sd_s)), 1e-6))

    keep = np.zeros_like(score, bool)
    v0, v1 = band
    keep[int(v0 * H):int(v1 * H), :] = True
    if mount_mask is not None:
        keep &= ~mount_mask                       # never the rig's own hardware
    if pupil_uv is not None:
        # ANNULUS around the eye centre, not merely "not the pupil".
        #
        # "Textured and stationary" describes many face features — nose-bridge edge, cheek
        # highlight, brow — so an unbounded argmax picks the STRONGEST of them, not the canthus.
        # Measured: it landed at u=0.825 on a bright nose-bridge edge while the eye sat at u~0.16.
        # The constraint that actually singles out a canthus is that it lies ON THE BOUNDARY OF
        # THE EYE APERTURE: close to the eye but never on the pupil.
        #
        # The pupil supplies LOCALITY only — where the eye is — never the position itself. That
        # distinction is what went wrong before: anchoring the position to the pupil made the
        # label track gaze (correlation 0.99). Here the estimate is still chosen by texture and
        # stationarity within the region; the pupil only says which region.
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx / W - pupil_uv[0], yy / H - pupil_uv[1])
        keep &= (d > exclude_r) & (d < max_r)
    if not keep.any():
        return None
    sc = np.where(keep, score, -1.0)
    y, x = np.unravel_index(int(np.argmax(sc)), sc.shape)
    return (x / float(W), y / float(H), float(sc[y, x]))


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
             confirm_n=CONFIRM_N, max_rounds=MAX_ROUNDS, tol=CONFIRM_TOL, band=None):
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
                if not in_prior(uv, prior, mirrored, band=band):
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

    # ---- PASS 0: locate the MOUNT and derive the vertical band from it -------------------
    mount, band = {}, {}
    for rid in (0, 1):
        m = R == rid
        if not m.any():
            continue
        mo = find_mount(F[m][:400])
        if mo is None:
            continue
        mount[rid] = mo
        band[rid] = search_band(mo)
    if verbose:
        print("== pass 0: mount fiducial (known object, rigid to the camera) ==")
        for rid, name in ((0, "eyeL"), (1, "eyeR")):
            if rid in mount:
                mo = mount[rid]
                print("  %s: mount covers %.1f%% of frame, floor v=%.3f, %.1fx more static "
                      "than the rest -> canthus band v [%.3f, %.3f]"
                      % (name, 100 * mo["frac"], mo["v_floor"], mo["static_ratio"],
                         band[rid][0], band[rid][1]))
            else:
                print("  %s: mount NOT found — falling back to the simulated v prior" % name)
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
        prop = TemplateProposer(tmpl["eyeL" if rid == 0 else "eyeR"], prior, mirrored,
                                band=band.get(rid))
        uv, _ = prop(F[i], cv2)
        if uv is None or not in_prior(uv, prior, mirrored, band=band.get(rid)):
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

    # ---- PASS 2: TWO INDEPENDENT ESTIMATORS + a proximity gate + a geometric validator ----
    # Dylan's construction: "estimate separately and they have to overlap by some proximity, then
    # a third checks that they are on point." The previous version failed because the two
    # estimators shared a fitted parameter and so could never disagree. These share nothing:
    #   A  appearance  -- template match inside the mount band, single frame
    #   B  motion      -- textured AND temporally stationary over a window (face-fixed, not gaze)
    #   C  geometry    -- mount band, rig.py's u prior, and NOT on the pupil
    from pupil_tracker import PupilTracker
    idx, labels, rounds, status = [], [], [], []
    for rid in (0, 1):
        rows = np.where(R == rid)[0]
        if len(rows) == 0 or rid not in band:
            continue
        mirrored = (rid == 0)
        mmask = mount[rid]["mask"] if rid in mount else None
        prop = TemplateProposer(tmpl["eyeL" if rid == 0 else "eyeR"], prior, mirrored,
                                band=band[rid])
        for w0 in range(0, len(rows), WINDOW):
            win_rows = rows[w0:w0 + WINDOW]
            if len(win_rows) < 8:
                continue
            # B: one estimate per window — the canthus is stationary within a seating, so a
            # per-window answer is the natural granularity for a motion-based estimator.
            # Eye centre = MEDIAN pupil across the window. Averaging over gaze gives the centre
            # of the eye rather than wherever the pupil happened to be in one frame.
            pts = []
            for j in win_rows[::3]:
                rj = _trk.detect(F[j])
                if getattr(rj, "ok", False):
                    pts.append(rj.pupil)
            pu = tuple(np.median(np.array(pts), axis=0)) if len(pts) >= 3 else None
            b = face_fixed_point(F[win_rows], cv2, band[rid], mmask, pu)
            if b is None:
                for i in win_rows:
                    status.append("no-motion-estimate")
                continue
            pB = (b[0], b[1])
            for i in win_rows:
                g = F[i]
                closed, _ = looks_closed(g, ref_open.get(rid, 0.02), _trk)
                if closed:
                    near = [j for j in (i - 1, i + 1) if 0 <= j < len(F) and int(R[j]) == rid]
                    if sum(1 for j in near
                           if looks_closed(F[j], ref_open.get(rid, 0.02), _trk)[0]) >= 1:
                        status.append("closed")
                        continue
                pA, peak = prop(g, cv2)                      # A, independent of B
                if pA is None:
                    status.append("no-proposal")
                    continue
                sep = float(np.hypot(pA[0] - pB[0], pA[1] - pB[1]))
                if sep > AGREE_TOL:                          # the proximity gate
                    status.append("disagree")
                    continue
                uv = ((pA[0] + pB[0]) / 2.0, (pA[1] + pB[1]) / 2.0)
                # C: the third check — geometry, which neither estimator can talk into agreeing
                if not in_prior(uv, prior, mirrored, band=band[rid]):
                    status.append("off-prior")
                    continue
                rp = _trk.detect(g)
                if getattr(rp, "ok", False):
                    if np.hypot(uv[0] - rp.pupil[0], uv[1] - rp.pupil[1]) < 0.05:
                        status.append("on-pupil")            # a canthus is never ON the pupil
                        continue
                idx.append(i); labels.append(uv); rounds.append(int(sep * 1000))
                status.append("confirmed")

    idx = np.array(idx, np.int32)
    if len(idx):
        np.savez_compressed(out, index=idx, labels=np.array(labels, np.float32),
                            rounds=np.array(rounds, np.int32))
    if verbose:
        from collections import Counter
        c = Counter(status)
        print("== automated labelling ==")
        print("  frames            %d" % len(F))
        print("  confirmed         %d (%.0f%%)  median A-B separation %.3f"
              % (c["confirmed"], 100.0 * c["confirmed"] / len(F),
                 (np.median(rounds) / 1000.0) if rounds else 0))
        print("  skipped: closed   %d" % c["closed"])
        print("  rejected: A/B disagree %d" % c["disagree"])
        print("  rejected: off-prior    %d" % c["off-prior"])
        print("  rejected: on-pupil     %d" % c["on-pupil"])
        print("  rejected: no-proposal  %d" % c["no-proposal"])
        print("  rejected: no-motion    %d" % c["no-motion-estimate"])
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

    # The mount fiducial: a dark STATIC bar at the top plus a moving face below it. find_mount
    # must recover the bar and nothing else, and the derived band must sit strictly below it --
    # this is the constraint that stops the proposer matching the rig's own hardware.
    rngm = np.random.default_rng(11)
    stack = []
    for k in range(40):
        fr = rngm.integers(110, 150, (200, 320)).astype(np.float32)
        fr[0:60, :] = 20                                     # mount: dark, never moves
        y = 120 + int(10 * np.sin(k))                        # face: moves frame to frame
        fr[y:y + 40, 100:220] = 60
        stack.append(fr)
    mo = find_mount(np.array(stack))
    bd = search_band(mo) if mo else None
    checks.append(("find_mount recovers the static bar (floor v=%.3f) and NOT the moving face"
                   % (mo["v_floor"] if mo else -1),
                   mo is not None and 0.25 <= mo["v_floor"] <= 0.35
                   and mo["static_ratio"] > 2.0))
    checks.append(("derived band sits strictly BELOW the mount (%s)"
                   % (("[%.3f, %.3f]" % bd) if bd else "none"),
                   bd is not None and bd[0] > mo["v_floor"] and bd[1] <= 1.0))
    checks.append(("a proposal inside the mount is rejected by the band",
                   not in_prior((0.834, 0.10), prior, False, band=bd)
                   and in_prior((0.834, 0.80), prior, False, band=bd)))

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
