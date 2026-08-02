"""calib_preflight.py — is this rig actually ready to run a calibration session?

`main.py --world-cam-left ... --fullscreen` needs SIX things true at once. When one is missing
the loop still starts, shows a window, and silently never stores a sample — which looks like a
broken algorithm and is really a missing template or a camera pointed at the ceiling. This
checks all six and names the one that is wrong.

  1. CAMERAS      all four base roles (worldL/R, eyeL/R) open and deliver frames
  2. WORLD DOT    DotDetector finds the target dot in BOTH world cams — capture.py sets
                  `no_target` if either misses, and a no-target frame can never become a sample
  3. TEMPLATES    an eye-corner template exists per eye (created by --calibrate-corners)
  4. CORNER LOCK  EyeCornerTracker actually matches those templates in the live eye frames
  5. DISPLAY      a second display is attached for --fullscreen (the AR glasses)
  6. SAMPLE DB    the sample DB path is writable

The pure parts (template/DB/display checks) run headless in `--selftest`; the camera parts need
the hardware and opencv.

    python3 calib_preflight.py --selftest
    python3 calib_preflight.py --run
    python3 calib_preflight.py --run --roles worldL=2 worldR=3 eyeL=0 eyeR=1
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

from config import Config
from bank_bringup import parse_roles

BASE_ROLES = ("worldL", "worldR", "eyeL", "eyeR")


class Check:
    def __init__(self, name, ok, detail="", fix=""):
        self.name, self.ok, self.detail, self.fix = name, ok, detail, fix

    def line(self):
        return "  [%s] %-13s %s" % ("PASS" if self.ok else "CHECK", self.name, self.detail)


# --------------------------------------------------------------------------
#  Pure checks (filesystem / OS — no cameras)
# --------------------------------------------------------------------------
def check_templates(template_dir, roles=("eyeL", "eyeR")):
    """An eye-corner template PNG must exist per eye, or template matching has nothing to match."""
    missing = [r for r in roles
               if not os.path.exists(os.path.join(template_dir, "%s.png" % r))]
    return Check("templates", not missing,
                 "all present in %s" % template_dir if not missing
                 else "missing: %s" % ", ".join(missing),
                 fix="python3 main.py --calibrate-corners")


def check_db_writable(db_path):
    """The sample DB must be creatable/writable or approvals vanish at the end of the session."""
    d = os.path.dirname(db_path) or "."
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".preflight_write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return Check("sample db", True, "writable: %s" % db_path)
    except Exception as e:
        return Check("sample db", False, "NOT writable (%s): %s" % (e, db_path),
                     fix="fix permissions on %s" % d)


def parse_displays(text):
    """Count attached displays in `system_profiler SPDisplaysDataType` output.

    Each display appears as a 'Resolution:' line; the AR glasses show up as an ordinary
    external monitor because the One Pro is a USB-C DisplayPort device."""
    return re.findall(r"Resolution:\s*(.+)", text)


def check_display():
    """--fullscreen needs somewhere to put the overlay: a second display."""
    try:
        txt = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return Check("display", False, "could not query displays (%s)" % e)
    res = parse_displays(txt)
    ok = len(res) >= 2
    return Check("display", ok,
                 "%d display(s): %s" % (len(res), "; ".join(r.strip() for r in res)),
                 fix="connect the XREAL One Pro over USB-C DisplayPort (it mounts as a monitor)")


# --------------------------------------------------------------------------
#  Live checks (cameras + opencv)
# --------------------------------------------------------------------------
def check_cameras(role_index, frames):
    """Every base role must have produced a frame."""
    dead = [r for r in BASE_ROLES if frames.get(r) is None]
    return Check("cameras", not dead,
                 "all %d roles delivering" % len(BASE_ROLES) if not dead
                 else "no frames from: %s" % ", ".join(dead),
                 fix="python3 bank_bringup.py --scan   (check indices/ports)")


def check_world_dot(frames, detector=None):
    """BOTH world cams must see the target dot — capture.py's `no_target` needs both."""
    from dot_detector import DotDetector
    det = detector or DotDetector()
    seen, where = [], {}
    for r in ("worldL", "worldR"):
        f = frames.get(r)
        if f is None:
            continue
        xy, _ = det.detect(f)
        if xy is not None:
            seen.append(r)
            where[r] = xy
    ok = len(seen) == 2
    # DotDetector returns NORMALISED (0..1) coords — print 3 decimals, not %.0f, or every
    # position rounds to 0/1 and a real detection is indistinguishable from a corner artefact.
    detail = "found in both (%s)" % ", ".join("%s@%.3f,%.3f" % (r, *where[r]) for r in seen)
    if ok:
        depth_mm, dv, warns = dot_geometry(where["worldL"], where["worldR"])
        detail += "  depth %s" % ("∞" if depth_mm is None else "%.0f mm" % depth_mm)
        for w in warns:
            detail += "\n                     ⚠ " + w
        ok = not warns                       # a geometrically impossible "dot" is NOT a pass
    return Check("world dot", ok, detail if seen
                 else "seen by %s only" % (", ".join(seen) or "neither"),
                 fix="put a dark round dot on white paper where BOTH world cams see it, "
                     "roughly centred in each frame")


def dot_geometry(pL, pR, width_px=640, edge_margin=0.05,
                 min_depth_mm=300.0, max_depth_mm=20000.0, max_dv=0.02):
    """Sanity-check that two dot detections are plausibly the SAME physical point.

    Both cameras finding *a* dark blob is not evidence they found the SAME one — loose cameras
    each lock onto their own screw head or shadow, the check goes green, and every stored sample
    is nonsense. Two geometric tests catch that:

      * EPIPOLAR — the world cams are a horizontal pair, so one point must land on nearly the
        same ROW in both. A large vertical offset means two different objects.
      * DEPTH — disparity converts to a distance via the rig's own focal length and baseline.
        A target at 18 cm (or 200 m) is not the thing you taped to the wall.
      * DISPARITY SIGN — a real point in front of the rig lands FURTHER RIGHT in the LEFT cam,
        so uL - uR must be POSITIVE. Negative means the pair is reversed: either the two
        --world-cam-* indices are swapped, or both sensors are mounted 180 deg.

    The sign test is not cosmetic. world_mesh.WorldTracker filters correspondences on
    `(a[0] - b[0]) > 0.5`, so a reversed pair has EVERY stereo match silently discarded and the
    map stays empty. Taking abs() here (as this function used to) hides that completely: a
    reversed pair yields the identical depth number and reads green, which is exactly how a
    swapped world pair survived a full preflight on 2026-08-02. USB indices shift between runs,
    so this is a per-session hazard, not a one-time wiring mistake.

    Returns (depth_mm or None for infinity, dv, [warnings])."""
    from world_mesh import DEFAULT_F, DEFAULT_B
    warns = []
    dv = abs(float(pL[1]) - float(pR[1]))
    if dv > max_dv:
        warns.append("vertical offset %.3f between the two cams (epipolar violation) — they are "
                     "probably locked onto DIFFERENT objects, not one shared dot" % dv)
    f = DEFAULT_F * (width_px / 1280.0)          # focal scales with capture width
    signed_px = (float(pL[0]) - float(pR[0])) * width_px
    if signed_px < 0:
        warns.append("disparity is NEGATIVE (%.1f px): the dot sits further right in worldR than "
                     "in worldL. A point in front of the rig must be further right in the LEFT "
                     "cam, so the world pair is REVERSED — swap --world-cam-left/--world-cam-right "
                     "(or the sensors are mounted 180 deg). WorldTracker discards every match "
                     "with negative disparity, so the mesh will stay empty until this is fixed"
                     % signed_px)
    d_px = abs(signed_px)
    depth = (f * DEFAULT_B / d_px) if d_px > 1e-6 else None
    if depth is not None and depth < min_depth_mm:
        warns.append("implied depth %.0f mm — far too close to be the target" % depth)
    elif depth is not None and depth > max_depth_mm:
        warns.append("implied depth %.0f mm — effectively at infinity, likely a false match"
                     % depth)
    elif depth is None:
        warns.append("zero disparity — identical position in both cams, not a real stereo pair")
    for name, p in (("worldL", pL), ("worldR", pR)):
        if min(p[0], p[1], 1 - p[0], 1 - p[1]) < edge_margin:
            warns.append("%s hit is hard against a frame edge (%.3f,%.3f) — likely an artefact"
                         % (name, p[0], p[1]))
    return depth, dv, warns


def check_corner_lock(frames, template_dir):
    """The templates must actually match in the live frames, not merely exist on disk."""
    from eye_tracker import EyeCornerTracker
    scores, missing = {}, []
    for r in ("eyeL", "eyeR"):
        trk = EyeCornerTracker(os.path.join(template_dir, "%s.png" % r))
        if not trk.ready:
            missing.append(r)
            continue
        f = frames.get(r)
        if f is None:
            missing.append(r)
            continue
        xy, sc = trk.track(f)
        scores[r] = (xy, sc)
    ok = len(scores) == 2 and all(xy is not None and sc > 0.5 for xy, sc in scores.values())
    detail = ("locked: " + ", ".join("%s score %.2f" % (r, scores[r][1]) for r in scores)) \
        if scores else "no template/frame for: %s" % ", ".join(missing)
    if scores and not ok:
        detail += "  (weak — aim the cam at the eye corner / re-grab the template)"
    return Check("corner lock", ok, detail,
                 fix="python3 main.py --calibrate-corners  (with the cam aimed at your canthus)")


def run(roles=None, seconds=3.0, verbose=True):
    """Open the bank, grab a synchronized set, and run every check against it."""
    from sync_capture import SyncBank
    cfg = Config()
    if roles is None:
        from connect import load_map
        roles = load_map() or {}
        roles = {r: i for r, i in roles.items() if r in BASE_ROLES}
    missing_roles = [r for r in BASE_ROLES if r not in roles]
    if missing_roles:
        print("no camera indices for: %s" % ", ".join(missing_roles))
        print("pass them explicitly, e.g.:")
        print("  python3 calib_preflight.py --run --roles worldL=2 worldR=3 eyeL=0 eyeR=1")
        return 1

    checks = []
    bank = SyncBank(roles).start()
    try:
        frames = {}
        import time
        t_end = time.monotonic() + seconds          # let exposure settle before judging
        while time.monotonic() < t_end:
            fs = bank.sync_frame()
            for r in roles:
                if fs.get(r) is not None:
                    frames[r] = fs.get(r)
        checks.append(check_cameras(roles, frames))
        checks.append(check_world_dot(frames))
        checks.append(check_templates(cfg.template_dir))
        checks.append(check_corner_lock(frames, cfg.template_dir))
    finally:
        bank.close()
    checks.append(check_display())
    checks.append(check_db_writable(cfg.db_path))

    ok = all(c.ok for c in checks)
    if verbose:
        print("== calibration preflight ==")
        for c in checks:
            print(c.line())
        print("  =>", "READY TO CALIBRATE ✅" if ok else "NOT READY — fix the CHECK rows ⚠️")
        if not ok:
            print("\n  next steps:")
            for c in checks:
                if not c.ok and c.fix:
                    print("    %-13s %s" % (c.name + ":", c.fix))
        else:
            idx = " ".join("--%s %d" % ({"worldL": "world-cam-left", "worldR": "world-cam-right",
                                         "eyeL": "eye-cam-left", "eyeR": "eye-cam-right"}[r],
                                        roles[r]) for r in BASE_ROLES)
            print("\n  run:  python3 main.py %s --fullscreen" % idx)
    return 0 if ok else 1


# ==========================================================================
#  Self-test (no hardware): the pure checks behave.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== calib_preflight self-test (no hardware) ==")
    import tempfile
    checks = []

    # (1) templates: absent -> CHECK (and names the missing eye), present -> PASS
    with tempfile.TemporaryDirectory() as td:
        c_missing = check_templates(td)
        open(os.path.join(td, "eyeL.png"), "wb").close()
        c_half = check_templates(td)
        open(os.path.join(td, "eyeR.png"), "wb").close()
        c_full = check_templates(td)
        checks.append(("templates: none->CHECK, one->CHECK naming eyeR, both->PASS",
                       (not c_missing.ok) and (not c_half.ok) and "eyeR" in c_half.detail
                       and c_full.ok))

    # (2) db writable -> PASS; an unwritable path -> CHECK (not an exception)
    with tempfile.TemporaryDirectory() as td:
        good = check_db_writable(os.path.join(td, "samples.db"))
        bad = check_db_writable("/System/nope/definitely/not/writable/samples.db")
        checks.append(("sample db: writable->PASS, unwritable->CHECK (no crash)",
                       good.ok and not bad.ok))

    # (3) display parse counts monitors from real system_profiler text
    two = """Graphics/Displays:
      Chipset Model: AMD Radeon Pro
          Displays:
            Color LCD:
              Resolution: 3072 x 1920 Retina
            XREAL One Pro:
              Resolution: 1920 x 1080 (1080p FHD)"""
    one = """Graphics/Displays:
          Displays:
            Color LCD:
              Resolution: 3072 x 1920 Retina"""
    checks.append(("display parse: 2 monitors vs 1 (fullscreen needs 2)",
                   len(parse_displays(two)) == 2 and len(parse_displays(one)) == 1))

    # (4) cameras check flags exactly the dead roles
    frames = {"worldL": np.zeros((8, 8, 3), np.uint8), "worldR": None,
              "eyeL": np.zeros((8, 8, 3), np.uint8), "eyeR": None}
    c = check_cameras({r: i for i, r in enumerate(BASE_ROLES)}, frames)
    checks.append(("cameras: names the dead roles (worldR, eyeR)",
                   (not c.ok) and "worldR" in c.detail and "eyeR" in c.detail))

    # (5) world dot: a synthetic dark dot on white is found in BOTH cams -> PASS;
    #     found in only one -> CHECK (this is capture.py's `no_target` condition)
    import cv2
    def page_with_dot():
        img = np.full((240, 320, 3), 255, np.uint8)
        cv2.circle(img, (160, 120), 9, (10, 10, 10), -1)
        return img
    blank = np.full((240, 320, 3), 255, np.uint8)
    one_only = check_world_dot({"worldL": page_with_dot(), "worldR": blank})
    checks.append(("world dot: one cam only -> CHECK (matches capture.py no_target)",
                   not one_only.ok))

    # (6) dot_geometry: the tests that separate ONE shared target from two unrelated blobs.
    #     A plausible pair (same row, ~1.5 m) must be clean; the real failure we hit on
    #     hardware (18 cm + epipolar offset) must be caught.
    from world_mesh import DEFAULT_F, DEFAULT_B
    W = 640
    f = DEFAULT_F * (W / 1280.0)
    d_good = (f * DEFAULT_B / 1500.0) / W                    # disparity for a 1.5 m target
    good_depth, good_dv, good_warns = dot_geometry((0.50, 0.50), (0.50 - d_good, 0.50))
    # the REAL measured pair from this rig (2026-08-01): caught by DEPTH (181 mm), not by the
    # epipolar test — its dv is 0.017, inside the 0.02 tolerance that loose, uncalibrated
    # cameras need. Depth is the discriminator here; epipolar only bites once they are mounted.
    bad_depth, bad_dv, bad_warns = dot_geometry((0.387, 0.054), (0.652, 0.071))
    epi_depth, _, epi_warns = dot_geometry((0.50, 0.50), (0.50 - d_good, 0.58))  # same depth, off-row
    far_depth, _, far_warns = dot_geometry((0.50, 0.50), (0.4999, 0.50))         # ~zero disparity
    checks.append(("dot_geometry: 1.5 m pair clean; 181 mm pair caught by DEPTH (dv %.3f is "
                   "within tolerance); off-row pair caught by EPIPOLAR; ~0 disparity caught"
                   % bad_dv,
                   not good_warns and abs(good_depth - 1500.0) < 25.0
                   and any("too close" in w for w in bad_warns)
                   and not any("epipolar" in w for w in bad_warns)
                   and any("epipolar" in w for w in epi_warns)
                   and any("infinity" in w for w in far_warns)))

    # (7) disparity SIGN: a REVERSED world pair is invisible to every other test here — abs()
    #     hands back the identical depth, the rows still line up, neither hit is near an edge, so
    #     the row reads green. That is exactly what happened on 2026-08-02: the preflight passed
    #     at 2054 mm with the pair swapped, while WorldTracker's `(uL - uR) > 0.5` filter threw
    #     away every correspondence and the map stayed empty. The assert below pins the reason
    #     depth cannot catch it: both orderings give the SAME distance.
    fwd_depth, _, fwd_warns = dot_geometry((0.50, 0.50), (0.50 - d_good, 0.50))
    rev_depth, _, rev_warns = dot_geometry((0.50 - d_good, 0.50), (0.50, 0.50))
    checks.append(("dot_geometry: reversed pair caught by SIGN — depth alone cannot "
                   "(both orderings give %.0f mm)" % rev_depth,
                   not any("REVERSED" in w for w in fwd_warns)
                   and any("REVERSED" in w for w in rev_warns)
                   and fwd_depth is not None and rev_depth is not None
                   and abs(fwd_depth - rev_depth) < 1e-6))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "CALIB PREFLIGHT OK — every precondition is checkable ✅"
              if ok else "PROBLEM ⚠️")
        print("  on hardware:  python3 calib_preflight.py --run")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="check every precondition for a calibration session")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="run the live checks (needs cameras)")
    ap.add_argument("--roles", nargs="*", default=None, metavar="ROLE=INDEX",
                    help="camera indices, e.g. --roles worldL=2 worldR=3 eyeL=0 eyeR=1")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    if args.run:
        try:
            roles = parse_roles(args.roles) if args.roles else None
        except ValueError as e:
            sys.exit("bad --roles: %s" % e)
        sys.exit(run(roles=roles, seconds=args.seconds))
    sys.exit(selftest())
