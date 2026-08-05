"""depth.py, the world stereo pair used as a METRIC DEPTH SENSOR, with honest uncertainty.

WHY THIS EXISTS, AND WHY IT IS NOT JUST triangulate_stereo
    `world_mesh.triangulate_stereo` already converts a matched pair into a 3D point, and this
    module calls it rather than re-deriving the geometry. What was missing is everything AROUND
    that number:

      * HOW WRONG IS IT. Depth error grows as Z^2, so a disparity that is half a pixel off costs
        1 mm at 0.5 m and 1.6 m at 20 m. A depth with no uncertainty attached invites exactly the
        mistake this project keeps making -- trusting a number because it exists.
      * IS IT GOOD ENOUGH FOR WHAT WE WANT IT FOR. The consumer of depth here is parallax
        correction for the overlay, and the accuracy that needs also grows as Z^2. Comparing the
        two is the only way to answer "can the rig do this at that distance", and the answer is
        much better than raw depth error suggests.
      * WHEN TO REFUSE. Behind the camera, disparity below the noise floor, or a pair that
        violates the epipolar constraint are all "no depth", not "a large depth".

THE RESULT THAT SHAPES THE DESIGN (derived in usable_range, verified in the selftest)
    Stereo depth error and the depth accuracy REQUIRED for sub-pixel parallax BOTH scale as Z^2,
    so their ratio is CONSTANT with range. The 67 mm baseline is therefore matched to the task at
    every distance, not merely up close -- stereo "dying at range" is only a problem if you forget
    that distant objects are correspondingly forgiving of depth error.

    That is a statement about the RATIO. The absolute numbers still degrade, and the margin is
    thin: at the optimistic 0.25 px disparity precision the ratio is ~0.9 (10% headroom), and at a
    realistic 0.5 px it is ~1.8, i.e. about 2 display px of parallax error. Fine to look at, not
    fine to call sub-pixel. Both are reported rather than one being hidden.

WHAT THE GLASSES CANNOT DO, for contrast (see HANDOFF "RENDERING ARCHITECTURE")
    An IMU integrates rotation cleanly but translation needs double-integrated acceleration, which
    drifts quadratically -- metres within seconds. So a 3DoF stabiliser cancels rotation exactly
    and translation not at all. Depth is the thing THIS RIG has and the glasses do not, which is
    why parallax correction has to happen here.

    python3 depth.py --selftest
"""
import argparse
import sys

import numpy as np

import rig
from world_mesh import DEFAULT_B, DEFAULT_F, triangulate_stereo

# Disparity precision of the matcher, in pixels. Two values on purpose.
#
# 0.25 px is what a good sub-pixel refinement achieves on a clean, high-contrast target; 0.5 px is
# what to expect on real imagery. Quoting only the first is how a spec sheet lies, so every
# accuracy answer here is reported at BOTH and the caller decides which to believe.
DISP_PRECISION_BEST = 0.25
DISP_PRECISION_REAL = 0.50

# Display scale for turning an angle into "how many pixels wrong", from the XREAL panel.
DISPLAY_PX = 1920
DISPLAY_FOV_DEG = 50.0


def display_px_per_deg():
    return DISPLAY_PX / DISPLAY_FOV_DEG


def depth_sigma_mm(Z_mm, disp_px=DISP_PRECISION_REAL, f=DEFAULT_F, B=DEFAULT_B):
    """1-sigma depth uncertainty at range Z, from disparity precision.

    Z = f*B/d  =>  dZ/dd = -f*B/d^2 = -Z^2/(f*B).  The Z^2 is the whole story of stereo: doubling
    the range quadruples the error. It is also why the baseline is the only real lever -- f and B
    enter only as a product.
    """
    Z = np.asarray(Z_mm, float)
    return Z * Z * float(disp_px) / (f * B)


def parallax_px(Z_mm, translation_mm=100.0):
    """Display pixels of overlay error from an UNCORRECTED head translation at depth Z.

    This is the quantity a 3DoF stabiliser cannot fix, because the shift depends on the depth it
    does not know. Measured on this rig's numbers: 434 px at 0.5 m, 110 px at 2 m, 11 px at 20 m.
    """
    Z = np.asarray(Z_mm, float)
    ang = np.degrees(np.arctan2(float(translation_mm), np.maximum(Z, 1e-6)))
    return ang * display_px_per_deg()


def depth_needed_mm(Z_mm, tol_px=1.0, translation_mm=100.0):
    """Depth accuracy required so that parallax correction is within `tol_px` display pixels.

    Differentiating the parallax angle t/Z w.r.t. Z gives -t/Z^2, so the tolerable depth error is
    tol_angle * Z^2 / t -- the SAME Z^2 that governs stereo error. That coincidence is the reason
    the rig stays matched to the task across its whole range.
    """
    Z = np.asarray(Z_mm, float)
    tol_rad = np.radians(float(tol_px) / display_px_per_deg())
    return tol_rad * Z * Z / float(translation_mm)


def capability_ratio(disp_px=DISP_PRECISION_REAL, tol_px=1.0, translation_mm=100.0,
                     f=DEFAULT_F, B=DEFAULT_B):
    """(depth error we get) / (depth error we need). CONSTANT with range, that is the point.

    <= 1 means parallax correction is within tol_px at every distance the pair can match at all.
    Returned as a scalar precisely because it does not depend on Z.
    """
    tol_rad = np.radians(float(tol_px) / display_px_per_deg())
    return (float(disp_px) / (f * B)) / (tol_rad / float(translation_mm))


class StereoDepth:
    """Metric depth from the world pair, with uncertainty and explicit refusal.

    Coordinates are NORMALISED (0..1) to match dot_detector / the rest of the pipeline, and are
    converted to pixels internally. Refusing is a first-class result: `ok=False` with a reason
    beats a confident number, which is this project's most expensive recurring failure.
    """

    # A pair further apart in rows than this is not one physical point on a rectified pair.
    max_row_offset_norm = 0.030
    # Disparity below this is indistinguishable from the matcher's own noise.
    min_disp_px = 1.0

    def __init__(self, res=None, f=None, B=None, disp_px=DISP_PRECISION_REAL):
        self.res = float(res if res is not None else rig.WORLD_RES)
        self.f = float(f if f is not None else DEFAULT_F)
        self.B = float(B if B is not None else DEFAULT_B)
        self.disp_px = float(disp_px)

    def __call__(self, uvL, uvR):
        return self.measure(uvL, uvR)

    def measure(self, uvL, uvR):
        """Normalised (u,v) in each world cam -> dict with Z_mm, sigma_mm, ok, why.

        SIGN IS NOT SILENTLY ABSORBED. A negative disparity means the pair is REVERSED (or the
        sensors are mounted 180 deg), and this reports that as a refusal naming the cause rather
        than taking abs() -- which is precisely the bug that let a swapped pair pass a full
        preflight at 2054 mm on 2026-08-02.
        """
        uL, vL = float(uvL[0]), float(uvL[1])
        uR, vR = float(uvR[0]), float(uvR[1])
        row = abs(vL - vR)
        if row > self.max_row_offset_norm:
            return self._bad("epipolar violation: rows differ by %.3f (max %.3f), the two cams "
                             "are almost certainly on DIFFERENT objects"
                             % (row, self.max_row_offset_norm))
        disp_px = (uL - uR) * self.res
        if disp_px < 0:
            return self._bad("disparity is NEGATIVE (%.1f px), the world pair is REVERSED "
                             "(swap worldL/worldR), or the sensors are mounted 180°" % disp_px)
        if disp_px < self.min_disp_px:
            return self._bad("disparity %.2f px is below the noise floor (%.1f px), the point is "
                             "too far for this %.0f mm baseline to range"
                             % (disp_px, self.min_disp_px, self.B))
        pts, good = triangulate_stereo([[uL * self.res, vL * self.res]],
                                       [[uR * self.res, vR * self.res]],
                                       f=self.f, B=self.B, min_disp=self.min_disp_px)
        if not bool(good[0]):
            return self._bad("triangulation rejected the pair (disparity %.2f px)" % disp_px)
        Z = float(pts[0][2])
        sig = float(depth_sigma_mm(Z, self.disp_px, self.f, self.B))
        return {"ok": True, "why": "", "Z_mm": Z, "sigma_mm": sig,
                "disp_px": disp_px, "row_offset": row,
                "rel_sigma": sig / Z if Z else float("inf"),
                "xyz_mm": tuple(float(x) for x in pts[0]),
                "parallax_px_100mm": float(parallax_px(Z)),
                "needed_mm_1px": float(depth_needed_mm(Z)),
                "sufficient": sig <= float(depth_needed_mm(Z))}

    @staticmethod
    def _bad(why):
        return {"ok": False, "why": why, "Z_mm": None, "sigma_mm": None}


def usable_range(disp_px=DISP_PRECISION_REAL, tol_px=1.0):
    """Table of what the pair can actually do, for the doc and for sanity."""
    rows = []
    for Z in (300, 500, 1000, 2000, 5000, 20000, 100000):
        d = DEFAULT_B * DEFAULT_F / Z
        sig = float(depth_sigma_mm(Z, disp_px))
        need = float(depth_needed_mm(Z, tol_px))
        rows.append((Z, d, sig, need, float(parallax_px(Z)), sig <= need))
    return rows


def selftest():
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (", " + detail) if detail else ""))

    sd = StereoDepth()

    # --- KNOWN-BAD INPUTS FIRST. Each must be REFUSED with the cause named. ---
    r = sd.measure((0.60, 0.50), (0.50, 0.30))
    chk("epipolar violation is refused and named", (not r["ok"]) and "epipolar" in r["why"], r["why"][:60])
    r = sd.measure((0.40, 0.50), (0.60, 0.50))          # right-of-left => negative disparity
    chk("REVERSED pair is refused, not abs()'d", (not r["ok"]) and "REVERSED" in r["why"], r["why"][:60])
    r = sd.measure((0.50000, 0.50), (0.49999, 0.50))
    chk("sub-noise disparity is refused", (not r["ok"]) and "noise floor" in r["why"], r["why"][:60])

    # --- a real pair round-trips to the right distance ---
    for Z in (500.0, 2000.0, 20000.0):
        disp_norm = (DEFAULT_B * DEFAULT_F / Z) / rig.WORLD_RES
        r = sd.measure((0.50 + disp_norm, 0.50), (0.50, 0.50))
        chk("Z=%.0f mm round-trips" % Z, r["ok"] and abs(r["Z_mm"] - Z) < max(1.0, 0.01 * Z),
            "got %.0f mm" % (r["Z_mm"] if r["ok"] else -1))

    # --- the Z^2 laws, stated as a property rather than a fitted constant ---
    s1, s2 = depth_sigma_mm(1000.0), depth_sigma_mm(2000.0)
    chk("depth sigma scales as Z^2 (2x range -> 4x error)", abs(s2 / s1 - 4.0) < 1e-6,
        "%.1f -> %.1f mm" % (s1, s2))
    n1, n2 = depth_needed_mm(1000.0), depth_needed_mm(2000.0)
    chk("required depth accuracy ALSO scales as Z^2", abs(n2 / n1 - 4.0) < 1e-6,
        "%.1f -> %.1f mm" % (n1, n2))

    # THE DESIGN CLAIM: because both scale as Z^2, the ratio is range-INDEPENDENT. Pinned across
    # three decades of distance so a future change that breaks the property fails loudly here.
    ratios = [depth_sigma_mm(Z) / depth_needed_mm(Z) for Z in (300.0, 3000.0, 30000.0)]
    chk("capability ratio is CONSTANT with range (the whole design argument)",
        max(ratios) - min(ratios) < 1e-9, "%.3f / %.3f / %.3f" % tuple(ratios))
    chk("capability_ratio() agrees with the measured ratio",
        abs(capability_ratio() - ratios[0]) < 1e-9,
        "%.3f vs %.3f" % (capability_ratio(), ratios[0]))

    # --- honesty: the optimistic and realistic precisions must NOT report the same verdict ---
    best, real = capability_ratio(DISP_PRECISION_BEST), capability_ratio(DISP_PRECISION_REAL)
    chk("optimistic 0.25 px is ~sub-pixel capable (ratio <= 1)", best <= 1.0, "%.2f" % best)
    chk("realistic 0.50 px is NOT, and says so (ratio > 1)", real > 1.0, "%.2f" % real)
    chk("the two differ by exactly the precision ratio", abs(real / best - 2.0) < 1e-9)

    # --- parallax sanity against the figures in HANDOFF ---
    chk("parallax at 0.5 m ~ 434 px", abs(parallax_px(500.0) - 434) < 6, "%.0f" % parallax_px(500.0))
    chk("parallax at 20 m ~ 11 px", abs(parallax_px(20000.0) - 11) < 2, "%.0f" % parallax_px(20000.0))

    print("DEPTH OK ✅" if ok_all else "DEPTH FAILED ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="world-stereo depth sensor")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--table", action="store_true", help="print the usable-range table")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.table:
        print("baseline %.0f mm   focal %.0f px   disparity precision %.2f px\n"
              % (DEFAULT_B, DEFAULT_F, DISP_PRECISION_REAL))
        print("%-10s %-11s %-13s %-13s %-12s %s"
              % ("distance", "disparity", "depth sigma", "needed (1px)", "parallax", "verdict"))
        for Z, d, sig, need, par, good in usable_range():
            print("%-10s %-11s %-13s %-13s %-12s %s"
                  % ("%.1f m" % (Z / 1000.0), "%.1f px" % d, "%.0f mm" % sig,
                     "%.0f mm" % need, "%.0f px" % par, "OK" if good else "insufficient"))
        print("\ncapability ratio (constant with range): %.2f at %.2f px, %.2f at %.2f px"
              % (capability_ratio(DISP_PRECISION_REAL), DISP_PRECISION_REAL,
                 capability_ratio(DISP_PRECISION_BEST), DISP_PRECISION_BEST))
        sys.exit(0)
    p.print_help()
