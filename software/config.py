"""Central configuration for the AR eye-corner calibration rig.

All values are defaults; main.py exposes the important ones as CLI flags.
Coordinates inside the learning pipeline are normalized to [0, 1] so the model
is independent of any specific camera/display resolution.
"""
import os
from dataclasses import dataclass, field
from typing import List

# Anchor data paths to the repo root (parent of software/) so they're the same
# no matter what directory you launch from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DATA = os.path.join(_ROOT, "data")


@dataclass
class Config:
    # ---- cameras (device indices; overridden by CLI) ----
    # CANONICAL HARDWARE = 6-camera BINOCULAR build (XREAL One Pro has a display PER EYE):
    #   2 outward "world" cams (shared) + 4 eye-facing cams (2 eye-corner + 2 NIR pupil,
    #   one of each per eye). See WIRING.md / ORDER_LIST.md / cad/xreal_one_mount.scad.
    # The four indices below are the WORLD + EYE-CORNER base that the live calibration loop
    # reads (8 features); the 2 NIR pupil cams are added with use_pupil (live_features /
    # --use-pupil). The 8-cam stereo "FULL" upgrade (use_stereo) is a SEPARATE future tier.
    # The calibration registers the display/right eye as the representative; the left is the
    # mirror (binocular), so a single registration model still describes the device.
    world_cam_left: int = 0
    world_cam_right: int = 1
    eye_cam_left: int = 2
    eye_cam_right: int = 3
    cam_width: int = 640
    cam_height: int = 480

    # ---- display / overlay ----
    display_w: int = 1920          # render target (your AR monitor in Path A)
    display_h: int = 1080
    overlay_window: str = "AR-overlay"
    dot_radius_px: int = 8
    fullscreen: bool = False       # set True to push onto the AR monitor

    # ---- joystick / nudge ----
    nudge_gain: float = 0.006      # normalized display units per key/stick step (faster traversal)
    deadzone: float = 0.15
    approve_button: int = 0        # gamepad "A"
    quit_button: int = 1           # gamepad "B"

    # ---- eye-corner tracking ----
    # template-match search; templates captured via --calibrate-corners
    template_dir: str = os.path.join(_DATA, "templates")
    search_margin: float = 0.35    # fraction of frame to search around last hit

    # ---- learning ----
    db_path: str = os.path.join(_DATA, "samples.db")
    # RE-SEAT THE GLASSES EVERY N APPROVED SAMPLES (0 = never, the old behaviour).
    #
    # The eye-corner cameras exist to measure where the glasses sit on the face, and that term can
    # only be validated against data where the seat actually CHANGED. The 2026-08-03 run had
    # eye-feature sd 0.032 / 0.0085 -- the glasses did not move once across 17 samples -- so every
    # value of geometry.EYE_SHIFT_GAIN lost to leaving it off, which says nothing about the
    # correction and everything about the data. See reseat.py.
    reseat_every: int = 0
    # WHICH XREAL DISPLAY MODE A RUN WAS CAPTURED IN, stored with every sample.
    # "follow" (0DoF) is the correct one: a display pixel then corresponds to a FIXED direction
    # relative to the glasses, which is the only condition under which the learned map exists.
    # "locked" (anchor/3DoF) has the glasses' own IMU shift the image WITHIN the optics by an
    # amount depending on head pose, so the same features require different pixels at different
    # head poses. On 2026-08-04 this was recorded WRONGLY for a whole session and reasoned from as
    # settled for a day, which is why it is now a stored per-sample field rather than a memory.
    display_mode: str = "follow"
    poly_degree: int = 2
    min_samples_for_model: int = 6 # below this, fall back to weighted-mean pixel
    retrain_every: int = 1         # retrain after every N new samples
    # regularization is auto-selected by GCV across this log-spaced range:
    lambda_lo: float = 1e-4
    lambda_hi: float = 31.6
    lambda_steps: int = 18
    robust_iters: int = 2          # Huber IRLS passes that suppress bad samples
    huber_k: float = 1.5           # outlier threshold, in robust sigmas

    # ---- capture quality / smoothing ----
    eye_conf_min: float = 0.45     # reject an approve if eye-corner match < this
    # SMOOTH THE SLOW SIGNAL, NOT THE FAST ONE. Measured 2026-08-03: the pipeline runs 55.9 ms/frame
    # (17.9 fps) and these two EMAs added 84 + 56 = 140 ms on top -- 71% of a 196 ms total, which is
    # 225 px of lag at a gentle 30 deg/s head turn. Dylan: "it feels more like i am moving the dot
    # with my head than the dot is trying to reach the point."
    #
    # The two feature groups have completely different dynamics and must not share a filter:
    #   WORLD DOT (features 0-3) is the DIRECTION term. It moves as fast as the head does, it is
    #     already protected against mis-detections by main._gate_dot, and smoothing it is precisely
    #     what converts head motion into lag. Runs UNSMOOTHED.
    #   EYE CORNERS (features 4-7) encode how the glasses SIT on the face. That changes slowly --
    #     a re-seat, a slip -- so smoothing costs nothing and genuinely steadies the estimate.
    feat_smooth: float = 0.4       # EMA on the EYE features only (0..1, higher=snappier)
    feat_smooth_world: float = 1.0 # world dot: NO smoothing (1.0 = pass through)
    # pred_smooth existed to hide POLYNOMIAL jitter. Geometry is now the backbone (see geometry.py)
    # and is stable frame-to-frame, so heavy smoothing here buys nothing and costs a frame of lag.
    pred_smooth: float = 0.9       # (legacy EMA, superseded by the One Euro filter below)
    # VELOCITY-ADAPTIVE prediction smoothing. See smoothing.py for the measured comparison against
    # fixed EMAs; the short version is that a fixed filter must choose between lag and jitter and
    # this one does not. min_cutoff sets how still the dot is when you are still (lower = stiller);
    # beta sets how fast it opens up when you move (higher = less lag, more jitter during motion).
    one_euro_min_cutoff: float = 0.8
    one_euro_beta: float = 8.0     # prediction supplies responsiveness, so the filter stays smooth
    # PIPELINE LATENCY, measured 2026-08-03: 55.9 ms/frame capture+process plus filter delay.
    # This is the look-ahead the predictor extrapolates by, so it must track the real pipeline --
    # RE-MEASURE IT if the camera modes, resolution or per-frame work change.
    latency_s: float = 0.062
    # How fast the look-ahead backs off when motion becomes unpredictable (a turn reversal).
    # Lower = more cautious. Swept: 2 / 10 / 50 gave overall 24 / 20 / 20 px, so 10 is the knee.
    predict_accel_scale: float = 10.0

    # NIR pupil-centre feature (8 -> 10): the 2 NIR pupil cams of the 6-camera binocular CORE.
    # Off by default so the 8-feature WORLD+EYE-CORNER base pipeline (selftest/pixel_sweep/
    # match/the live --calibrate-corners loop) stays unchanged; turn on (--use-pupil) to add
    # the NIR pupil signal that pins the 6-DOF pose the eye-corner cams under-determine and
    # drives the preset floor toward sub-pixel. This is the canonical 6-cam CORE result.
    use_pupil: bool = False
    # 8-cam "FULL" FUTURE UPGRADE (separate tier, not the 6-cam model): the 2nd eye-corner
    # (stereo) cams, +4 features, for <1 px. Off by default.
    use_stereo: bool = False

    # feature order fed to the model (documented in one place). All four cameras:
    feature_names: List[str] = field(default_factory=lambda: [
        "worldL_dot_x", "worldL_dot_y",   # dot in the LEFT world cam (norm)
        "worldR_dot_x", "worldR_dot_y",   # dot in the RIGHT world cam (norm) -> stereo
        "eyeL_corner_x", "eyeL_corner_y", # left eye corner in left inward cam (norm)
        "eyeR_corner_x", "eyeR_corner_y", # right eye corner in right inward cam (norm)
    ])

    def __post_init__(self):
        # order MUST match autosim observe/ground_truth: base 8, then stereo (+4), then pupil (+2)
        if self.use_stereo and "eyeL2_x" not in self.feature_names:
            self.feature_names = self.feature_names + ["eyeL2_x", "eyeL2_y", "eyeR2_x", "eyeR2_y"]
        if self.use_pupil and "pupilR_x" not in self.feature_names:
            self.feature_names = self.feature_names + ["pupilR_x", "pupilR_y"]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)
