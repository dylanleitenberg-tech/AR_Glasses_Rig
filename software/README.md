# Software

Real-time human-in-the-loop calibrator that learns to place an AR pixel on a
real-world dot, using the corners of your eyes to sense how the glasses are seated.

## Install

```bash
cd ~/ar-eye-calibration/software
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The learning core needs only `numpy`, so the **self-test runs without installing
opencv/pygame**:

```bash
python3 main.py --selftest        # prints a learning curve, should end in PASS
```

## Modules

| File | Role |
|------|------|
| `config.py` | all defaults + the canonical feature order |
| `calibrator.py` | polynomial-ridge regression `f(dot, eye-corners) -> pixel` (numpy only) |
| `dataset.py` | SQLite store of approved samples (numpy + stdlib) |
| `synth.py` | synthetic ground-truth physics for sim/self-test (numpy only) |
| `cameras.py` | OpenCV capture wrapper; `snapshot()` = the high-speed approval grab |
| `dot_detector.py` | finds the drawn dot in the world frame -> normalized (x,y) |
| `eye_tracker.py` | template-matches your eye corner -> normalized (x,y) |
| `pupil_tracker.py` | **eye-tracking upgrade**: IR pupil + glint detection -> gaze features (`--selftest`; see `../EYE_TRACKING.md`) |
| `overlay.py` | black canvas + red dot (+ green truth target in sim) on the AR display |
| `input_ctl.py` | keyboard nudge/approve/quit (gamepad optional via pygame) |
| `main.py` | the integrated loop + all the modes |

### Hardware bring-up layer (connect · sync · adjust · mesh)

The host-side stack that gets the 6 cameras running on the Mac — see **[../HARDWARE_BRINGUP.md](../HARDWARE_BRINGUP.md)**. Every module has a headless `--selftest` (in the release gate):

| File | Role |
|------|------|
| `connect.py` | auto-classify the 6 USB cams (color/mono, res), assign roles, persist `data/rig_cameras.json` |
| `sync_capture.py` | `SyncBank`: barrier-aligned **synchronized** grab across all cameras + measured jitter |
| `autoexpose.py` | per-role **continuous** exposure/gain control (world sharp, pupil glints hot / field dark) |
| `world_mesh.py` | **real-world mesh tracking**: stereo triangulate + robust Kabsch VO + IMU-fused pose + Delaunay mesh |
| `imu_serial.py` | `GyroIntegrator`: XIAO gyro → per-frame rotation increment for the mesh |
| `rig_test.py` | one-command bring-up test: enumerate · fps · sync jitter · IR test · focus |
| `snapshot.py` | save one synchronized frame-set (6 PNGs + trustworthy `meta.json`) |
| `live_rig.py` | the integrated driver: sync grab → auto-expose → mesh track → feature capture, live telemetry |

```bash
python3 connect.py --auto && python3 connect.py --identify   # map cameras -> roles (once)
python3 rig_test.py --run        # is the rig fit to calibrate on?
python3 live_rig.py --run --imu  # everything running together
```

## Running

```bash
# 1) See the model learn, no hardware:
python3 main.py --simulate            # align RED onto GREEN, ENTER approves

# 2) On hardware, one-time: teach it your eye corners
python3 main.py --list-cams
python3 main.py --calibrate-corners --eye-cam-left 1 --eye-cam-right 2

# 3) Run for real
python3 main.py --world-cam 0 --eye-cam-left 1 --eye-cam-right 2 --fullscreen
```

### Controls
- `w a s d`: fine (pixel-level) nudge · `W A S D`: coarse jump
- arrow keys, nudge (shift = fine) · ENTER/SPACE, approve · Q, quit (vernier.py UI; gamepad optional)
- `ENTER`: approve & snapshot · `R`: recalibrate corners · `Q`/`ESC`: quit

## How a sample is captured (the core loop)

1. `read_real_features()` → dot in both world cams (x,y ×2) + both eye corners (x,y ×2)
   = 8 numbers (the two world cams give stereo parallax on the dot).
2. `calibrator.predict()` → where it *thinks* the pixel should go.
3. You nudge the red dot onto the real dot.
4. `ENTER` → `snapshot_features()` does the **high-speed eye snapshot** (flushes
   the camera buffer, grabs the freshest frame so a small head move doesn't blur the
   canthus), pairs those features with the pixel you approved, writes it to SQLite,
   and refits the model. Nudge resets; draw/move the dot and go again.

As samples accumulate across different dot positions *and* different ways the glasses
sit on your face, the prediction in step 2 lands closer and closer to the real dot,
until you barely nudge at all. Inspect progress any time:

```bash
sqlite3 ../data/samples.db "SELECT COUNT(*), AVG(pixel_x), AVG(pixel_y) FROM samples;"
```

## The mapping model (calibrator.py)

Degree-2 polynomial regression from the 8 features to the display pixel, hardened so
it stays accurate on a small, human-collected, noisy dataset:

- **Standardized** features + polynomial terms → well-conditioned fit; the
  regularization means the same thing everywhere on screen.
- **Auto-regularization (GCV)**: the ridge penalty `lambda` is chosen automatically
  each refit via one SVD (Generalized Cross-Validation). No hand-tuning; it
  self-adjusts from 6 samples to thousands. The chosen `lambda` shows in the HUD.
- **Robust (Huber IRLS)**: a single mis-tracked eye corner or fat-fingered approve is
  down-weighted instead of warping the whole map. In the self-test, with 12% gross
  outliers, naive least-squares lands ~32 px off vs ~10 px for the robust fit.
- **Confidence-weighted**: each sample carries the eye-tracker's match score
  (`weight` column); crisp captures count more. Approvals below `eye_conf_min` are
  rejected outright so junk never enters the dataset.
- **Smoothing**: live eye features (`feat_smooth`) and the shown prediction
  (`pred_smooth`) are EMA-filtered to kill jitter.

Check the held-out accuracy any time with `holdout_error()` (k-fold; returns RMS +
median in normalized units), or watch it live in `--selftest`.

## Tuning notes
- Finer nudges: lower `nudge_gain` in `config.py`, or use the stick.
- Add features (pupil position, IMU tilt): extend `config.feature_names` and the
  readers, the calibrator adapts to any feature count automatically.
- Below `min_samples_for_model` (6) it predicts the confidence-weighted mean pixel;
  keep approving and it switches to the learned model.
- Robustness/where regularization can roam: `robust_iters`, `huber_k`,
  `lambda_lo/hi/steps` in `config.py`.

```bash
sqlite3 ../data/samples.db "SELECT COUNT(*), AVG(weight) FROM samples;"   # dataset health
```
