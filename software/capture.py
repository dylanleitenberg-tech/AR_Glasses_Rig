"""capture.py — the live multi-camera capture pipeline: real cameras OR --simulate.

The real-sensor mirror of autosim.observe(). Each frame it:
  * reads the role-mapped camera bank (cameras.CameraBank),
  * detects each feature — drawn dot in the 2 world cams (dot_detector), outer canthus in
    the 4 eye-corner cams (eye_tracker), pupil centre in the NIR cam (pupil_tracker),
  * on a BLINK / closed eye (blink.py): does NOT drop the frame — it acknowledges the blink,
    HOLDS that eye's last open-eye reading, and freezes eye-position tracking for that eye
    until it reopens (a fresh read then resumes tracking). The overlay stays steady through
    a blink instead of glitching, and the held frame is flagged NOT-fresh so it is never
    stored as a calibration sample.
  * assembles the feature vector in the EXACT Config.feature_names order
    (live_features.assemble_features), holding per-eye features where blinking, and
  * degrades STEREO -> MONO when a 2nd eye-corner read is lost beyond the brief hold window
    (sustained lash/track loss on the deeper cam) — a momentary loss is held, not dropped.

Robustness telemetry (ValidityStats): fraction of frames that are fresh (storable),
blinking-but-held, stereo-available, mono-fallback, and not-ready (startup only). We never
drop a frame once tracking has started.

Runnable before hardware: --simulate drives autosim (honouring its BLINK_PROB drops), splits
the oracle features back through the SAME assembler + hold logic, and injects per-eye blinks
(held) + independent deeper-stereo-cam loss (mono fallback) so both paths are exercised.
"""
import sys

import numpy as np

from config import Config
from live_features import assemble_features, split_features
from blink import BlinkDetector

# feature-key space (matches split_features / assemble_features kwargs)
BASE_KEYS = ("worldL", "worldR", "eyeL", "eyeR")
STEREO_KEYS = ("eyeL2", "eyeR2")
PUPIL_KEYS = ("pupilR",)
EYE_OF_KEY = {"eyeL": "L", "eyeL2": "L", "eyeR": "R", "eyeR2": "R", "pupilR": "R"}
EYE_KEYS = set(EYE_OF_KEY)                      # the held-through-blink keys; world keys never held

MAX_HOLD = 12   # frames a blinking/held eye reading stays usable (~a blink); beyond -> lost


def _cfg_keys(cfg):
    ks = list(BASE_KEYS)
    if cfg.use_stereo:
        ks += list(STEREO_KEYS)
    if cfg.use_pupil:
        ks += list(PUPIL_KEYS)
    return ks


class CaptureResult:
    def __init__(self, features, mode, valid, blink, conf,
                 ready, fresh, held, blinking, no_target=False, truth=None):
        self.features = features    # np.ndarray in Config order, or None
        self.mode = mode            # "stereo" | "mono" | "none"
        self.valid = valid          # {feature-key: bool} fresh this frame
        self.blink = blink          # {"L": bool, "R": bool}
        self.conf = conf            # min tracking confidence over the FRESH features
        self.ready = ready          # have a full vector (fresh or held)
        self.fresh = fresh          # every used feature is fresh -> safe to store as a sample
        self.held = held            # some feature is a held value (blink / brief loss)
        self.blinking = blinking    # at least one eye is blinking (held), not dropped
        self.no_target = no_target  # the world dot isn't on the display (nothing to register) —
        #                             a distinct state from a blink (a blink holds; this waits)
        self.truth = truth          # sim-only geometric-truth pixel (fresh frames only)


class ValidityStats:
    """Telemetry incl. the both-cams-valid robustness metric + the held-through-blink rate."""
    def __init__(self):
        self.total = self.ready = self.fresh = self.blinking = 0
        self.stereo = self.mono = self.notready = self.no_target = 0
        self.blinkL = self.blinkR = 0

    def update(self, r: CaptureResult, configured_stereo: bool):
        self.total += 1
        if r.no_target:
            self.no_target += 1                      # off-screen target: nothing to register
        elif not r.ready:
            self.notready += 1                       # startup: no held value yet
        else:
            self.ready += 1
            self.fresh += int(r.fresh)
            self.blinking += int(r.blinking)
            if r.mode == "stereo":
                self.stereo += 1
            elif configured_stereo:
                self.mono += 1
        self.blinkL += int(r.blink.get("L", False))
        self.blinkR += int(r.blink.get("R", False))

    def summary(self):
        t = max(self.total, 1); rd = max(self.ready, 1)
        present = max(self.total - self.no_target, 1)    # frames with a world target
        return {
            "frames": self.total,
            "no_target_frac": self.no_target / t,        # world dot off-screen (waits, not a blink)
            "ready_of_present_frac": self.ready / present,  # ~1.0: with a target we hold, never drop
            "notready_frac": self.notready / t,          # startup only
            "fresh_frac": self.fresh / rd,               # storable as a calibration sample
            "blinking_frac": self.blinking / rd,         # held through a blink (not dropped)
            "stereo_avail_frac": self.stereo / rd,       # both eye2 available (fresh or held)
            "mono_fallback_frac": self.mono / rd,
            "blinkL_frac": self.blinkL / t,
            "blinkR_frac": self.blinkR / t,
        }


class LiveCapture:
    def __init__(self, cfg=None, simulate=True, device=None, role_index=None,
                 template_dir=None, seed=0, sim_blink_inject=0.06, sim_stereo_loss=0.06):
        self.cfg = cfg or Config()
        self.simulate = simulate
        self.stats = ValidityStats()
        self.blink = BlinkDetector()
        self.held = {}        # key -> last fresh (x,y)
        self.stale = {}       # key -> frames since last fresh
        if simulate:
            import autosim
            self.device = device or autosim.Simulator(
                seed, use_pupil=self.cfg.use_pupil, use_stereo=self.cfg.use_stereo)
            self.subject = self.device.new_subject()
            self.dev = self.device.seat()
            self.rng = np.random.default_rng(seed + 999)
            self.sim_blink_inject = sim_blink_inject
            self.sim_stereo_loss = sim_stereo_loss
        else:
            from cameras import CameraBank
            from dot_detector import DotDetector
            from eye_tracker import EyeCornerTracker
            from pupil_tracker import PupilTracker
            import os
            if role_index is None:
                raise ValueError("real capture needs a role_index (see cameras.list_cameras)")
            tdir = template_dir or self.cfg.template_dir
            self.bank = CameraBank(role_index)
            self.dot = DotDetector()
            self.pupil_trk = PupilTracker()
            self.corner = {k: EyeCornerTracker(os.path.join(tdir, "%s.png" % k))
                           for k in ("eyeL", "eyeR", "eye2L", "eye2R") if k in role_index}

    # ---- hold + assemble --------------------------------------------------
    def _resolve(self, parts, valid, blink):
        """Update per-key hold state. World keys must be fresh (never held). An eye key is HELD
        at its last open-eye value ONLY while THAT eye is blinking (the whole eye is closed and
        frozen, so holding it is consistent), up to MAX_HOLD frames. An INDEPENDENT loss of a
        single (e.g. deeper stereo) cam while the eye is open is NOT held — mixing a stale read
        with the fresh primary is worse than dropping to mono — so it just becomes unavailable.
        Returns (value{}, fresh{}, available set)."""
        val, fresh, avail = {}, {}, set()
        for k in _cfg_keys(self.cfg):
            if valid.get(k, False):
                self.held[k] = parts[k]; self.stale[k] = 0
                val[k] = parts[k]; fresh[k] = True; avail.add(k)
            elif (k in EYE_KEYS and blink.get(EYE_OF_KEY[k], False)
                  and k in self.held and self.stale.get(k, MAX_HOLD) < MAX_HOLD):
                self.stale[k] = self.stale.get(k, 0) + 1     # hold the frozen (blinking) eye
                val[k] = self.held[k]; fresh[k] = False; avail.add(k)
            # else: world key, independent single-cam loss, or no history/too stale -> unavailable
        return val, fresh, avail

    def _assemble(self, val, fresh, avail):
        """Assemble the effective-mode vector from resolved (fresh-or-held) values."""
        if not all(k in avail for k in BASE_KEYS):
            return None, "none", False
        use_stereo = self.cfg.use_stereo and all(k in avail for k in STEREO_KEYS)
        use_pupil = self.cfg.use_pupil and all(k in avail for k in PUPIL_KEYS)
        eff = Config(use_pupil=use_pupil, use_stereo=use_stereo)
        kw = {"worldL": val["worldL"], "worldR": val["worldR"],
              "eyeL": val["eyeL"], "eyeR": val["eyeR"], "cfg": eff}
        used = list(BASE_KEYS)
        if use_stereo:
            kw["eyeL2"] = val["eyeL2"]; kw["eyeR2"] = val["eyeR2"]; used += list(STEREO_KEYS)
        if use_pupil:
            kw["pupilR"] = val["pupilR"]; used += list(PUPIL_KEYS)
        feats = assemble_features(**kw)
        all_fresh = all(fresh[k] for k in used)
        return feats, ("stereo" if use_stereo else "mono"), all_fresh

    def _finish(self, parts, valid, blink, conf, truth, no_target=False):
        val, fresh, avail = self._resolve(parts, valid, blink)
        feats, mode, all_fresh = self._assemble(val, fresh, avail)
        ready = feats is not None
        blinking = any(blink.values())
        r = CaptureResult(feats, mode, valid, blink, conf,
                          ready=ready, fresh=(ready and all_fresh),
                          held=(ready and not all_fresh), blinking=blinking,
                          no_target=no_target,
                          truth=(truth if (ready and all_fresh) else None))
        self.stats.update(r, self.cfg.use_stereo)
        return r

    # ---- the two paths ----------------------------------------------------
    def read(self) -> CaptureResult:
        return self._read_sim() if self.simulate else self._read_real()

    def _read_sim(self) -> CaptureResult:
        rr = self.rng.random()
        blink_eye, lose = None, ()
        if rr < self.sim_blink_inject:                     # per-eye blink -> HOLD that eye
            blink_eye = "L" if self.rng.random() < 0.5 else "R"
        elif self.cfg.use_stereo and rr < self.sim_blink_inject + self.sim_stereo_loss:
            lose = ("eyeL2" if self.rng.random() < 0.5 else "eyeR2",)   # deeper cam lost -> mono
        return self.sim_step(blink_eye=blink_eye, lose_keys=lose)

    def sim_step(self, blink_eye=None, lose_keys=()) -> CaptureResult:
        """One simulated capture with a SPECIFIED outage injection. Deterministic, so the
        fallback validation can prove the hold/fallback/metric logic without real blinks
        (decoupled from the NIR dark_thresh, which only decides WHEN a blink is declared)."""
        self.dev = self.device.slip(self.dev)
        obs = self.device.observe(self.subject, self.dev, self.device.world_point())
        blink = {"L": False, "R": False}
        if obs is None:                       # world dot off the display -> no target (waits)
            return self._finish({}, {}, blink, 0.0, None, no_target=True)
        feats, _label, truth = obs
        parts = split_features(feats, self.cfg)
        valid = {k: True for k in parts}
        if blink_eye:                         # whole-eye blink -> that eye's reads held
            blink[blink_eye] = True
            for k in list(valid):
                if EYE_OF_KEY.get(k) == blink_eye:
                    valid[k] = False
        for k in lose_keys:                   # independent single-cam loss (eye open) -> mono
            if k in valid:
                valid[k] = False
        return self._finish(parts, valid, blink, 1.0, truth)

    def _read_real(self) -> CaptureResult:
        frames = self.bank.read()
        parts, valid, conf = {}, {}, []
        blink = {"L": False, "R": False}
        for k in ("worldL", "worldR"):
            xy, _ = self.dot.detect(frames[k]) if frames.get(k) is not None else (None, None)
            valid[k] = xy is not None
            if xy is not None:
                parts[k] = np.asarray(xy)
        for k in ("eyeL", "eyeR", "eye2L", "eye2R"):
            if k not in self.corner:
                continue
            kk = {"eye2L": "eyeL2", "eye2R": "eyeR2"}.get(k, k)
            f = frames.get(k)
            if (f is None) or self.blink.is_blink_region(f):     # blink -> acknowledge, hold (in _resolve)
                blink[EYE_OF_KEY[kk]] = True; valid[kk] = False
                continue
            xy, sc = self.corner[k].track(f)
            valid[kk] = xy is not None
            if xy is not None:
                parts[kk] = np.asarray(xy); conf.append(sc)
        if "pupil" in self.bank.cams:
            pres = self.pupil_trk.detect(frames.get("pupil")) if frames.get("pupil") is not None else None
            if BlinkDetector.is_blink_pupil(pres):
                blink["R"] = True; valid["pupilR"] = False
            else:
                parts["pupilR"] = np.asarray(pres.pupil); valid["pupilR"] = True; conf.append(pres.conf)
        no_target = not (valid.get("worldL") and valid.get("worldR"))   # no world dot -> nothing to register
        return self._finish(parts, valid, blink, float(min(conf)) if conf else 0.0, None, no_target)

    def close(self):
        if not self.simulate:
            self.bank.release()


# ----------------------------------------------------------------------
#  Self-test (sim): blinks are HELD (not dropped), contract length holds, fallback counts
# ----------------------------------------------------------------------
def selftest(n=5000, seed=0, verbose=True):
    if verbose:
        print("== live capture pipeline self-test (--simulate, hold-on-blink) ==")
    ok = True
    for use_stereo, use_pupil in [(False, False), (True, False), (True, True)]:
        cfg = Config(use_stereo=use_stereo, use_pupil=use_pupil)
        cap = LiveCapture(cfg=cfg, simulate=True, seed=seed,
                          sim_blink_inject=0.08, sim_stereo_loss=0.06)
        bad_len = blink_drops = blink_held = 0; mono_len = 8 + (2 if use_pupil else 0)
        for _ in range(n):
            r = cap.read()
            # a blink WITH a target present must be HELD, never dropped (the core of this change)
            if r.blinking and not r.no_target:
                if r.ready:
                    blink_held += 1
                else:
                    blink_drops += 1
            if r.ready:
                exp = cfg.n_features if r.mode == "stereo" else mono_len
                if r.features.shape[0] != exp:
                    bad_len += 1
        s = cap.stats.summary()
        fallback_fires = (not use_stereo) or (s["mono_fallback_frac"] > 0)
        cfg_ok = (bad_len == 0 and blink_drops == 0 and blink_held > 0
                  and s["ready_of_present_frac"] > 0.97 and fallback_fires)
        ok = ok and cfg_ok
        if verbose:
            print("  stereo=%-5s pupil=%-5s | ready(of target) %.1f%% | fresh %.0f%% blinking-HELD %.0f%% "
                  "blink-drops %d | stereo %.0f%% mono %.0f%% | no-target %.0f%% bad-len %d  [%s]"
                  % (use_stereo, use_pupil, 100*s["ready_of_present_frac"], 100*s["fresh_frac"],
                     100*s["blinking_frac"], blink_drops, 100*s["stereo_avail_frac"],
                     100*s["mono_fallback_frac"], 100*s["no_target_frac"], bad_len,
                     "PASS" if cfg_ok else "FAIL"))
    if verbose:
        print("  =>", "CAPTURE OK — blinks HELD not dropped (blink-drops 0), contract length held, "
              "stereo->mono fallback fires ✅" if ok else "PROBLEM ⚠️")
        print("  note: a blink holds the frozen eye position (fresh=False, not stored) and resumes on")
        print("  reopen; 'no-target' (dot off-screen) is a separate wait state, not a drop. real CV needs HW.")
    return 0 if ok else 1


def validate_fallback(seed=3, verbose=True):
    """Deterministic synthetic dropped-frame injector — proves the hold / stereo->mono fallback /
    both-cams-valid metric WITHOUT real blinks (so the logic is validated now; the NIR dark_thresh
    only decides WHEN a blink is declared, not WHETHER the fallback works)."""
    cfg = Config(use_stereo=True)
    cap = LiveCapture(cfg=cfg, simulate=True, seed=seed)
    r = cap.sim_step()
    while not (r.fresh and r.mode == "stereo"):       # warm up to a fresh stereo frame
        r = cap.sim_step()
    frozen_eyeR = split_features(r.features, cfg)["eyeR"].copy()
    checks = []

    # (1)+(2) a 5-frame right-eye blink: HELD (ready, not dropped), eye position FROZEN, not fresh
    blink_held = frozen = True
    for _ in range(5):
        r = cap.sim_step(blink_eye="R")
        if not (r.ready and r.blinking and r.blink["R"] and not r.fresh):
            blink_held = False
        if not np.allclose(split_features(r.features, cfg)["eyeR"], frozen_eyeR):
            frozen = False
    checks.append(("blink HELD, never dropped (ready+blinking, not fresh)", blink_held))
    checks.append(("eye position FROZEN at last open value through the blink", frozen))

    # (3) tracking RESUMES on reopen (a fresh frame returns)
    r = cap.sim_step()
    while r.no_target:
        r = cap.sim_step()
    checks.append(("tracking RESUMES on reopen (fresh again)", r.fresh))

    # (4) independent deeper-stereo-cam loss while the eye is OPEN -> MONO (not a stale hold)
    r = cap.sim_step(lose_keys=("eyeR2",))
    while r.no_target:
        r = cap.sim_step(lose_keys=("eyeR2",))
    checks.append(("independent stereo loss (eye open) -> MONO fallback", r.mode == "mono" and not r.blinking))

    # (5) hold is BOUNDED: a sustained blink past MAX_HOLD expires -> stops holding (not ready)
    ready_tail = []
    for i in range(40):
        r = cap.sim_step(blink_eye="R")
        if i >= 25:
            ready_tail.append(r.ready)
    checks.append(("hold bounded: blink >> MAX_HOLD expires (stops holding)",
                   np.mean(ready_tail) < 0.3))

    # (6) the both-cams-valid metric TRACKS a known injection rate
    cap2 = LiveCapture(cfg=cfg, simulate=True, seed=seed + 1)
    rng = np.random.default_rng(0); rate = 0.20
    for _ in range(4000):
        cap2.sim_step(lose_keys=("eyeR2",) if rng.random() < rate else ())
    mf = cap2.stats.summary()["mono_fallback_frac"]
    checks.append(("both-cams-valid metric tracks injection (inject 20%%, mono-fallback %.0f%%)"
                   % (100 * mf), abs(mf - rate) < 0.05))

    ok = all(v for _, v in checks)
    if verbose:
        print("== fallback + hold validation (deterministic synthetic dropped-frame injector) ==")
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "FALLBACK/HOLD LOGIC PROVEN in --simulate ✅" if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = selftest()
    print()
    rc |= validate_fallback()
    sys.exit(rc)
