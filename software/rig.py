"""Single source of truth for the physical rig geometry (numpy only).

These camera positions/orientations and the rest pose are what make the simulator a
faithful mirror of the printed bracket (`cad/xreal_one_mount.scad`). If you change the
hardware, change it HERE and the simulator follows.

Two coordinate frames are involved; keep them straight:
  * CAD frame (xreal_one_mount.scad): x = right, y = FORWARD, z = UP, origin at the
    rail/brow on top of the glasses.
  * Rig/sim/anatomy frame (here): x = right, y = UP, z = FORWARD (toward world), origin
    at the OPTIC CENTER (between the eyes, at the lens plane, ~eye level).
  Mapping: sim = (cad_x, cad_z, cad_y) with the origin shifted down from the brow to the
  optic center. The values below are the bracket's physical camera placements expressed
  directly in the rig/sim frame, so no conversion is needed at run time.

Cameras are fixed on the bracket at the NOMINAL IPD (the print doesn't adapt per user);
each wearer's real eyes sit at their own IPD, so the cameras are generally NOT perfectly
over the pupil — that offset is part of what calibration learns.
"""
import numpy as np

from optics import rot_xyz, look_at, PinholeCamera, DisplayOptics
import anatomy

# ---- how the glasses rest on the face (face -> rig transform) -------------
# MEASURED on the actual XREAL One Pro (Size L) — see MEASUREMENT_CHECKLIST.md.
NOMINAL_IPD = 67.0     # MEASURED user pupil IPD; bracket is printed for this (subjects vary around it)
DISPLAY_IPD = 68.13    # MEASURED display optic center-to-center (slightly wider than the pupils;
#                        calibration learns the ~1 mm pupil-vs-optic offset). Drives OPTIC_R/L below.
EYE_BEHIND = 28.5      # CoR behind the optic = MEASURED vertex 15 mm + ~13.5 mm cornea->center-of-rotation
PANTO_DEG = -3.5       # pantoscopic tilt: the One Pro adjusts in 3 stages up to ±3.5° — the old
#                        -7.0 was NOT ACHIEVABLE on this hardware (spec audit 2026-07-06). Set the
#                        glasses to the max nose-down stage and record it (MEASUREMENT_CHECKLIST B2).
#                        Changing this moves nominal_outer_canthus() -> update CANTH_SIM in the CAD.
R0 = rot_xyz(PANTO_DEG, 0, 0)
T0 = np.array([0.0, 0.0, -EYE_BEHIND])

# ---- camera placements on the bracket (rig frame, mm) ---------------------
# NO camera may sit inside either eye's see-through cone (apex at the CoR [+-IPD/2,0,-EYE_BEHIND],
# half-angle = the display half-FOV), or it would block the wearer's view of the world. This is
# enforced at import by assert_no_occlusion() at the bottom of the file. See occlusion_report().
#
# World cameras: RAISED well ABOVE the see-through cone, looking forward (+z) over the TOP of the
# lens. (Were at [+-32, 8, 7] = inside the cone; the main occluder.)
WC_X = NOMINAL_IPD / 2     # lateral: over each nominal pupil (keeps the 64 mm stereo baseline)
WC_UP = 49.0               # raised 44->49 (2026-07-23): the world board's LOWER mounting screws
#                            could not be threaded (rail/boom structure behind them); +5 lifts
#                            them clear. Small world-cam move; re-run the accuracy sim before the
#                            FINAL build to refresh the deployed number (fast gate parity holds).
# was 30->44 (2026-07-02): at 30 the 38 mm board's LOWER STANDOFFS +
#                            PCB edge sat inside the GLASSES BROW band (brow top = optic +22.7 up;
#                            board bottom = WC_UP - 19 was 11 mm below it) — found by the printed-
#                            solid check (cad_overlap.py); no earlier check compared parts against
#                            the glasses body. At 44 the whole module clears the brow top ~2.3 mm
#                            and the see-through-cone margin grows to ~+20 mm.
WC_FWD = 22.0              # forward so (a) the holder stack clears the forehead (5->12,
#                            2026-07-02) and (b) the BOARD PLANE sits wholly IN FRONT of the boom
#                            riser at the rail (12->22, 2026-07-04): at 12 the riser passed through
#                            the 38 mm board's back-component zone (847 mm^3, caught by the split
#                            boom-vs-pcbzone check in cad_overlap.py). Cone margin stays > +12.
WORLD_FOV = 70.0
WORLD_K1 = -0.06           # mild barrel distortion of the world lenses
WORLD_RES = 1280           # sensor resolution (quantizes the readings)
# Eye-corner cameras: out by the temples (peripheral, clear of the cone) and BEHIND the lens plane,
# in the lens->eye gap, aimed back/in/down at the outer canthus. (Were at [+-54, 5, 5] = barely
# outside the cone AND forward of the lens; eye-imaging cams belong in the gap, not the world side.)
# MOVED to the temple to CLEAR the world-cam board (the 38 mm world + 36 mm eye boards were
# physically overlapping by 15 mm at the old [+24,10,-2] — caught by the inter-camera clearance
# check, cad_fit.camera_clearance). Out + down + back: now +3.8 mm board clearance, and looking UP
# at the canthus (lower-rim placement = less lash occlusion).
EC_X = NOMINAL_IPD / 2 + 36.0
EC_UP = -5.0           # lowered -2 -> -5 (2026-07-16: longer side booms); the aim at
#                        the canthus re-tilts automatically (look_at), no other change needed
EC_FWD = -6.0
# EYE-CORNER cam AIM BIAS (2026-07-26): the mounted eye cam framed the outer corner low + inboard
# (bottom-left of frame), risking losing it on a glasses slip. Re-aim DOWN + OUTWARD so the canthus
# sits nearer frame centre. Mirrored per side. AIM ONLY — EC_X/UP/FWD (position) and
# nominal_outer_canthus() (the tracked landmark, hence CANTH_SIM parity) are UNCHANGED.
# RE-AIM REMOVED 2026-07-27 (both were 6.0 / 3.0): the bias was inferred from a mounted photo
# showing the corner edge-riding, but that framing is explained by the lens being ~45 deg rather
# than the modelled 90 — a magnification effect, not an aim error. Two independent checks agree
# the un-biased aim is the better one: the framing sweep puts the canthus at v=0.455 (frame
# centre) un-biased vs 0.350 biased, with edge margin 0.442 vs 0.350 and 0.0% vs 0.7% falling off
# the real 16:10 top edge; and accuracy_map is neutral either way (1.88 vs 1.90 px median).
EC_AIM_DOWN = 6.0   # mm the aim point drops (rig +y is up) -> ~11 deg downward tilt
EC_AIM_OUT  = 3.0   # mm the aim point moves outward (toward the temple), mirrored per side
# MEASURED ON HARDWARE (2026-08-01): the eye-cam modules wear a ~45 deg M12 lens, NOT the 90 deg
# one this model assumed. Confirmed by direct FOV measurement and corroborated by the mounted
# images (the eye fills far more of the frame than a 90 deg lens would give). Consequence, with
# the AS-PRINTED aim (outer canthus + EC_AIM_DOWN/OUT) against the real 16:10 sensor crop:
#     outer canthus  41.9% in-frame  -> NOT TRACKABLE
#     inner canthus 100.0% in-frame  (median u 0.838, v 0.414)
# So the tracked landmark is the INNER canthus (autosim.Simulator(landmark="inner")), while the
# camera AIM stays "outer" because that is what is physically printed and cannot be changed —
# aim and tracked landmark are deliberately independent. The carrier is final (no printer
# access), so remaining framing margin comes from the brow-clamp slop, not from geometry.
EYE_FOV = 45.0

# THE TRACKED LANDMARK — settled 2026-08-01, do not re-litigate per session.
# The camera AIM is "outer" because that is what is physically printed (and the carrier is final:
# ASA/PETG done, no printer access). The TRACKED point is "inner" because at the real 45 deg lens
# the outer canthus is only 41.9% in-frame while the inner is 100%. Aim and tracked landmark are
# independent by design. Defaulted here so every consumer (autosim.Simulator, preset.build_preset,
# match.train_prior) inherits it from ONE place instead of each carrying its own default.
TRACKED_LANDMARK = "inner"
EYE_K1 = -0.10            # wide eye-corner lenses distort more
EYE_RES = 640
# ---- eye2: STEREO eye-corner pair = the 8-cam "FULL" FUTURE UPGRADE (NOT part of the 6-cam CORE) --
# A 2nd eye-corner camera per eye at a clearly DIFFERENT position gives STEREO on each outer canthus,
# triangulating its depth (cfwd) — the term a single view can't see. This is the higher-accuracy
# evolution to build AFTER the 6-cam CORE (use_stereo / build_stereo / --stereo-test); it is kept
# isolated here and is off by default so the CORE stays a clean 6-camera model.
# Stereo cam: WEARABLE temple position, paired BELOW the primary eye cam (same x, ~at the lens
# plane) so its board extends out to the temple, not deep into the cheek. Parallax on the canthus
# ~56 deg (vs the un-wearable deep-gap 80 deg) — still triangulates canthus depth.
EC2_X = NOMINAL_IPD / 2 + 36.0   # moved with the primary eye cam to the temple (stereo pair)
EC2_UP = -16.0                   # below + behind the primary so the two stereo boards also clear
EC2_FWD = -10.0
EYE2_FOV = 90.0
EYE2_K1 = -0.10
EYE2_RES = 640

# ---- see-through occlusion guard ------------------------------------------
# Each eye's viewing cone has its apex at the eye's center of rotation (CoR), opens forward (+z),
# and its half-angle equals the display half-FOV. Any opaque camera inside that cone blocks the
# wearer's see-through view. assert_no_occlusion() (run at import) forbids it for BOTH eyes.
CONE_HALF_DEG = 25.0       # display half-FOV (DisplayOptics fov_deg=50 -> +-25 deg horizontal)
OCCLUSION_MARGIN = 8.0     # mm: camera half-size + clearance baked into the keep-out radius

# ---- display optic --------------------------------------------------------
DISPLAY_EYE = 1            # which eye the legacy single-eye sim registers (right). The XREAL One Pro
#                            is actually BINOCULAR (a display per eye); OPTIC_L is the left display.
OPTIC_R = np.array([DISPLAY_IPD / 2, 0.0, 0.0])   # right display optic center (MEASURED spacing)
OPTIC_L = np.array([-DISPLAY_IPD / 2, 0.0, 0.0])  # left display optic center (binocular registration)
VIRTUAL_DIST = 2000.0      # virtual image distance (mm); finite -> real vergence parallax

# ---- NIR pupil camera (display eye) ---------------------------------------
# Images the entrance pupil; its reading pins the 6-DOF glasses pose that the 2 eye-corner
# cams under-determine (and feeds geometry ID). Lower-nasal of the display eye, BEHIND the lens
# plane (in the gap), aimed up/out at the pupil. (Was [26, -14, 6] = forward of the lens, inside
# the cone.)
PUPIL_POS = np.array([NOMINAL_IPD / 2 - 8.0, -40.0, 6.0])    # right-eye NIR cam. LOWERED -33 -> -40 (2026-07-16:
#                        longer front mast); aim at the eye centre re-tilts automatically (below-forward
#                                                              so the board clears the cone after EYE_BEHIND=28.5)
PUPIL_POS_L = np.array([-(NOMINAL_IPD / 2 - 8.0), -33.0, 6.0])  # BINOCULAR: left-eye NIR cam (mirror)
PUPIL_FOV = 55.0
PUPIL_K1 = -0.05
PUPIL_RES = 640
PUPIL_PIX_SD = 0.0016      # NIR pupil-centre localization (~1 px on a 640 sensor)

# ---- pose dynamics + sensor noise (used by the simulator) -----------------
SLIP_ROT_SD = 0.6      # deg, per-iteration jitter
SLIP_TRANS_SD = 0.9    # mm, per-iteration jitter
SLIP_REVERT = 0.05     # weak mean reversion (so drift can accumulate)
SLIDE_RATE = 0.35      # mm/step the glasses creep DOWN the nose (eyes drift +y in rig)
PITCH_PER_MM = 0.25    # pantoscopic tilt change correlated with the slide (deg/mm)
RESEAT_PROB = 0.03     # chance per step the user re-seats the glasses (a fresh pose)
TRANS_CAP = 9.0        # clamp accumulated translation (mm) to plausible range
SEAT_ROT_SD = 1.2      # initial seating spread
SEAT_TRANS_SD = 1.8
# ===========================================================================
#  REAL-WORLD EFFECT MODELS.  All magnitudes below are PLACEHOLDERS at plausible
#  engineering values — replace each with the MEASURED value once hardware is in
#  hand (display checkerboard, camera calibration, eye-corner tracker stats, and a
#  few instrumented human sessions). They are faithful models, not penalties.
# ===========================================================================
# ---- per-attempt RANDOM noise (sets the accuracy floor; averaged out by the fit) ----
CANTHUS_PIX_SD = 0.0018     # eye-corner localization (~1.2 px on a 640 sensor)
CANTHUS_TAIL_PROB = 0.10    # fraction of reads drawn from a heavier tail
CANTHUS_TAIL_SCALE = 3.0
CANTHUS_GROSS_PROB = 0.015  # gross tracker failure (large outlier) per read
CANTHUS_DRIFT_SD = 0.0012   # within-session random walk of the corner offset
SOFT_TISSUE_SD = 0.15       # mm canthus motion per capture (user holding still to calibrate)
DOT_PIX_SD = 0.0015         # world-cam dot localization (~1 px)
FIXATION_JITTER_DEG = 0.12  # microsaccade / imperfect fixation on the dot
EP_PUPIL_SD = 0.25          # entrance-pupil shift with pupil size (mm)
PUPIL_WANDER = 0.25         # mm apparent entrance-pupil shift per 10 deg of gaze
HEAD_MOTION_TRANS_SD = 0.3  # mm pose change during a single capture
HEAD_MOTION_ROT_SD = 0.2    # deg
BLINK_PROB = 0.04           # capture dropped by a blink

# ---- per-SESSION systematic (fixed within a session -> learned/cancels) ----
CANTHUS_BIAS_SD = 0.008     # eye-corner template-centering offset
HUMAN_BIAS_SD = 0.006       # perceptual-alignment bias (vergence conflict ~ confounds kappa)
HUMAN_NOISE_SD = 0.004      # per-correction human alignment imprecision
HUMAN_TAIL_PROB = 0.10
HUMAN_TAIL_SCALE = 3.0
HUMAN_GROSS_PROB = 0.02     # fat-finger correction

# ---- per-UNIT manufacturing tolerance (fixed for one printed device) ----
DEVICE_SEED = 12345         # one physical unit -> one fixed perturbation
UNIT_TRANS_SD = 0.4         # mm camera mounting position tolerance
UNIT_ROT_SD = 0.4           # deg camera mounting orientation tolerance
UNIT_FOCAL_SD = 0.006       # fractional focal-length error per camera
UNIT_PRINCIPAL_SD = 0.003   # principal-point error (normalized) per camera
CAM_K2 = 0.02               # 2nd-order radial distortion of the camera lenses
CAM_TANGENTIAL_SD = 0.0008  # tangential distortion per lens
DISPLAY_K2 = 0.02           # 2nd-order display distortion
DISPLAY_WARP_SD = 0.004     # spatially-varying display residual
DISPLAY_CAM_TRANS_SD = 0.5  # mm display-to-camera extrinsic miscalibration
DISPLAY_CAM_ROT_SD = 0.25   # deg display-to-camera extrinsic miscalibration

# ---- overlay coverage: how far the drawn dot roams (drives display-pixel spread) ----
DOT_X = 185.0               # mm half-range of the dot on the page
DOT_Y = 135.0
DOT_Z_MEAN, DOT_Z_SD = 480.0, 55.0
ONSCREEN_MARGIN = 0.03      # reject targets outside this [0,1] margin (off the display)


def pose_R_t(dev):
    """Deviation [drx,dry,drz,dtx,dty,dtz] -> (R,t) mapping face -> rig frame."""
    R = rot_xyz(dev[0], dev[1], dev[2]) @ R0
    t = T0 + np.array(dev[3:6])
    return R, t


def nominal_outer_canthus():
    """Population-mean outer canthus in the rig frame (used to aim the eye cams)."""
    rise = ((anatomy.OCD_MEAN - anatomy.ICD_MEAN) / 2) * np.tan(
        np.radians(anatomy.CANTHAL_TILT_MEAN))
    face = np.array([[-anatomy.OCD_MEAN/2, rise, anatomy.CANTHUS_FWD_MEAN],
                     [+anatomy.OCD_MEAN/2, rise, anatomy.CANTHUS_FWD_MEAN]])
    return (R0 @ face.T).T + T0


def nominal_inner_canthus():
    """Population-mean INNER (medial) canthus in the rig frame.

    Alternative tracked landmark (see `landmark=` on build()/Simulator). Canthal tilt is
    defined as the OUTER canthus rising relative to the inner one, so the inner canthus
    carries no rise — it sits at eye level, ICD apart, at the same canthus depth.
    """
    face = np.array([[-anatomy.ICD_MEAN/2, 0.0, anatomy.CANTHUS_FWD_MEAN],
                     [+anatomy.ICD_MEAN/2, 0.0, anatomy.CANTHUS_FWD_MEAN]])
    return (R0 @ face.T).T + T0


def nominal_canthus(landmark="outer"):
    """The tracked landmark's population-mean position: 'outer' (default) or 'inner'."""
    if landmark == "outer":
        return nominal_outer_canthus()
    if landmark == "inner":
        return nominal_inner_canthus()
    raise ValueError("landmark must be 'outer' or 'inner', got %r" % (landmark,))


def build_pupil_camera():
    """NIR camera imaging the right eye's entrance pupil (rig frame)."""
    eye_rest = OPTIC_R + T0                      # nominal CoR of the right eye
    return PinholeCamera(PUPIL_POS, look_at(PUPIL_POS, eye_rest),
                         PUPIL_FOV, PUPIL_K1, 0, PUPIL_RES)


def build_pupil_camera_left():
    """BINOCULAR: NIR camera imaging the LEFT eye's entrance pupil (mirror of the right)."""
    eye_rest = OPTIC_L + T0                      # nominal CoR of the left eye
    return PinholeCamera(PUPIL_POS_L, look_at(PUPIL_POS_L, eye_rest),
                         PUPIL_FOV, PUPIL_K1, 0, PUPIL_RES)


def build_eye2_cameras(landmark="outer"):
    """Second eye-corner camera pair (stereo with rig['eye']) for canthus-depth triangulation."""
    canth = nominal_canthus(landmark)
    return [
        PinholeCamera([-EC2_X, EC2_UP, EC2_FWD],
                      look_at([-EC2_X, EC2_UP, EC2_FWD], canth[0]), EYE2_FOV, EYE2_K1, 0, EYE2_RES),
        PinholeCamera([+EC2_X, EC2_UP, EC2_FWD],
                      look_at([+EC2_X, EC2_UP, EC2_FWD], canth[1]), EYE2_FOV, EYE2_K1, 0, EYE2_RES),
    ]


def build(landmark="outer"):
    """Build the rig (6-cam BINOCULAR CORE): display optic + 2 world + 2 eye-corner + 2 NIR pupil
    (pupil = right/display eye, pupil_l = left eye mirror). 'eye2' is the 8-cam FULL upgrade pair
    (stereo eye-corner), returned for use_stereo builds but not part of the 6-cam CORE.

    `landmark` selects which canthus the eye-corner cams AIM at (and, paired with
    Simulator(landmark=), which one they track): 'outer' is the built design and the only
    one CAD parity covers; 'inner' is the medial-canthus variant under evaluation
    (landmark_test.py). Camera POSITIONS are identical either way — only the aim changes,
    so this is an aim-only experiment, exactly like EC_AIM_DOWN/OUT."""
    canth = nominal_canthus(landmark)
    world = [
        PinholeCamera([-WC_X, WC_UP, WC_FWD], np.eye(3), WORLD_FOV, WORLD_K1, 0, WORLD_RES),
        PinholeCamera([+WC_X, WC_UP, WC_FWD], np.eye(3), WORLD_FOV, WORLD_K1, 0, WORLD_RES),
    ]
    aimL = canth[0] + np.array([-EC_AIM_OUT, -EC_AIM_DOWN, 0.0])   # DOWN + OUTWARD (left)
    aimR = canth[1] + np.array([+EC_AIM_OUT, -EC_AIM_DOWN, 0.0])   # DOWN + OUTWARD (right, mirror)
    eye = [
        PinholeCamera([-EC_X, EC_UP, EC_FWD],
                      look_at([-EC_X, EC_UP, EC_FWD], aimL), EYE_FOV, EYE_K1, 0, EYE_RES),
        PinholeCamera([+EC_X, EC_UP, EC_FWD],
                      look_at([+EC_X, EC_UP, EC_FWD], aimR), EYE_FOV, EYE_K1, 0, EYE_RES),
    ]
    # fov_deg is the HORIZONTAL FOV. XREAL One Pro is specced at 57 deg DIAGONAL, which on a
    # 16:9 1080p panel is ~50.6 deg horizontal -> 50 here matches the One Pro (NOT a mismatch).
    display = DisplayOptics(fov_deg=50.0, k1=0.06, virtual_dist=VIRTUAL_DIST)
    return {"display": display, "world": world, "eye": eye,
            "eye2": build_eye2_cameras(landmark),
            "pupil": build_pupil_camera(), "pupil_l": build_pupil_camera_left(),
            "display_eye": DISPLAY_EYE, "optic": OPTIC_R, "optic_l": OPTIC_L}


# ---- occlusion guard: no camera inside either eye's see-through cone -------
def _camera_centers():
    """Every camera CENTER in the rig frame (mm) with a label. Mirrored cams give L/R.
    Covers the full 6-cam BINOCULAR CORE (2 world + 2 eye-corner + 2 NIR pupil) PLUS the
    eye2 stereo pair (the 8-cam FULL future upgrade) so every position is occlusion-checked."""
    cams = []
    for s in (-1, +1):
        side = "R" if s > 0 else "L"
        cams.append(("world" + side, np.array([s * WC_X, WC_UP, WC_FWD])))
        cams.append(("eye" + side, np.array([s * EC_X, EC_UP, EC_FWD])))
        cams.append(("eye2" + side, np.array([s * EC2_X, EC2_UP, EC2_FWD])))  # FULL upgrade
    cams.append(("pupilR", PUPIL_POS.copy()))    # 6-cam CORE: NIR pupil cam, right eye
    cams.append(("pupilL", PUPIL_POS_L.copy()))  # 6-cam CORE: NIR pupil cam, left eye (binocular)
    return cams


def occlusion_report():
    """Clearance (mm) of every camera vs each eye's see-through cone. For a camera at (cx,cy,cz)
    and an eye CoR at (sign*IPD/2, 0, -EYE_BEHIND): d = cz + EYE_BEHIND is the forward distance;
    a camera behind the eye (d<=0) cannot occlude; otherwise it must lie OUTSIDE the cone radius
    tan(half)*d + MARGIN at that depth. clearance = off - keepout; >=0 means clear. Returns rows
    (name, center, worst_clearance, worst_eye_sign)."""
    t = np.tan(np.radians(CONE_HALF_DEG))
    rows = []
    for name, c in _camera_centers():
        worst, worst_eye = np.inf, None
        for sign in (-1, +1):
            d = c[2] + EYE_BEHIND
            if d <= 0:
                clear = np.inf                       # behind the eye -> cannot block the cone
            else:
                off = np.hypot(c[0] - sign * NOMINAL_IPD / 2, c[1])
                clear = off - (t * d + OCCLUSION_MARGIN)
            if clear < worst:
                worst, worst_eye = clear, sign
        rows.append((name, c, worst, worst_eye))
    return rows


def assert_no_occlusion():
    """Raise if any camera sits inside either eye's see-through cone. Runs at import so a
    position edit can never silently re-occlude the wearer's view of the world."""
    bad = [(n, c, m) for (n, c, m, _e) in occlusion_report() if m < 0]
    if bad:
        detail = "; ".join("%s at %s inside the cone by %.2f mm"
                           % (n, np.round(c, 2).tolist(), -m) for n, c, m in bad)
        raise AssertionError("rig see-through occlusion — camera(s) block the view: " + detail)


assert_no_occlusion()       # validate the fixed rig geometry at import time
