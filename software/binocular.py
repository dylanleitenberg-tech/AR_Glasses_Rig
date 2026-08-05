"""binocular.py, validate the LEFT-eye oracle physics + the fusion-comfort numbers.

The One Pro is BINOCULAR; the plan (per-eye-correct => binocular-correct) hangs on two
things this checks in sim, before any hardware:

 1. COVERAGE, how often a target visible to the right display is also on the left
     (the binocular working volume).
 2. VERTICAL DISPARITY between the two eyes' TRUE overlay pixels, human fusion tolerates
     ~6 arcmin (Panum) / 15-20 arcmin comfort; if the TRUE pixels' vertical disparity is
     already small, two per-eye-correct overlays fuse automatically, and the dichoptic
     vernier (vernier.py --sbs) only has to null the CALIBRATION error between eyes.
 3. VERGENCE SANITY, horizontal disparity must grow as the target gets closer
     (1/d law); a sign/magnitude error here means the left oracle is wrong.

Run:  python3 main.py --binocular-test
"""
import numpy as np

import autosim
import pixel_map
import rig


def arcmin_per_px(display_fov_deg=50.0, width_px=1920):
    return display_fov_deg * 60.0 / width_px


def evaluate(n_subjects=25, seed=4200, verbose=True):
    apx = arcmin_per_px()
    poses = pixel_map.pose_grid()[::2]
    dots = pixel_map.dot_grid()[::4]
    both, r_only, vdisp = 0, 0, []
    for i in range(n_subjects):
        s = autosim.Simulator(seed + i)
        subj = s.new_subject()
        for dev in poses:
            for P in dots:
                g = s.ground_truth_both(subj, dev, P)
                if g is None:
                    continue
                _, pr, pl = g
                if pl is None:
                    r_only += 1
                    continue
                both += 1
                vdisp.append(abs(pl[1] - pr[1]) * 1080.0)
    vdisp = np.asarray(vdisp)
    cover = both / max(1, both + r_only)
    vmed, v95 = np.median(vdisp), np.percentile(vdisp, 95)
    vmed_arc = vmed * apx
    # MIRROR test = the exact correctness gate for the left oracle. A symmetric subject at
    # the centred pose, viewing a MIDLINE point, must give px_L = mirror(px_R): with kappa
    # too, since az_L = -az_R and kappa mirrors nasal-ward. Any sign/optic/CoR error in
    # ground_truth_left breaks this at machine precision. (The sim display has cx=cy=warp=0,
    # so the panel mapping itself is mirror-symmetric.)
    s = autosim.Simulator(seed)
    # IDEAL device for the exactness test: the sim deliberately jitters the display warp and
    # the display-to-camera extrinsic per device (real units are asymmetric); zero them so
    # any remaining asymmetry can only come from the left-oracle code itself.
    s.R_dc = np.eye(3)
    s.t_dc = np.zeros(3)
    s.rig["display"].warp = np.zeros(4)
    s.rig["display"].cx = s.rig["display"].cy = 0.0
    subj = s.new_subject()
    subj.cor = np.array([[-subj.ipd / 2, subj.cor[1][1], subj.cor[1][2]], subj.cor[1]])
    dev = np.zeros(6)
    dtest = np.array([350.0, 500.0, 800.0, 1500.0])
    mirr, hsigned = [], []
    for dz in dtest:
        for oy in (-30.0, 0.0, 40.0):
            g = s.ground_truth_both(subj, dev, np.array([0.0, oy, dz]))
            if g is None or g[2] is None:
                continue
            _, pr, pl = g
            mirr.append(max(abs(pl[0] - (1.0 - pr[0])), abs(pl[1] - pr[1])))
            if oy == 0.0:
                hsigned.append((pl[0] - pr[0]) * 1920.0)
    mirr_max = max(mirr) if mirr else np.inf
    hsigned = np.asarray(hsigned)
    ok_cover = cover > 0.85
    ok_fusion = vmed_arc < 6.0
    ok_mirror = mirr_max < 1e-9
    ok_verg = len(hsigned) >= 3 and np.all(np.diff(hsigned) < 0)   # signed disparity falls with depth
    if verbose:
        print("== binocular oracle validation (%d subjects, %d samples) ==" % (n_subjects, both))
        print("  both-displays coverage      : %5.1f%%  (right-only %d)  [%s]"
              % (100 * cover, r_only, "PASS" if ok_cover else "CHECK"))
        print("  TRUE vertical disparity     : median %.2f px (~%.1f arcmin), 95th %.2f px  [%s vs ~6' Panum]"
              % (vmed, vmed_arc, v95, "PASS" if ok_fusion else "OVER"))
        print("  L/R mirror symmetry         : max err %.2e (midline pts, symmetric eye)  [%s]"
              % (mirr_max, "PASS" if ok_mirror else "FAIL"))
        print("  signed h-disp vs depth %s mm : %s px, monotonic falling  [%s]"
              % (dtest.astype(int).tolist(), np.round(hsigned, 1).tolist(),
                 "PASS" if ok_verg else "FAIL"))
    ok_verg = ok_verg and ok_mirror
    if verbose:
        print("  => %s, per-eye-correct overlays fuse; the dichoptic vernier only nulls "
              "the CALIBRATION residual" % ("BINOCULAR PHYSICS OK ✅"
                                            if (ok_cover and ok_fusion and ok_verg) else "REVIEW ❌"))
    return ok_cover and ok_fusion and ok_verg


if __name__ == "__main__":
    import sys
    sys.exit(0 if evaluate() else 1)
