# Real-world overlay on moving people ("turn people into monkeys")

A world-locked AR effect: detect and track people through the world cameras, place a monkey on
each one, and keep it glued to them as **they and your head move** — rendered through the
**calibrated** glasses.

```bash
cd ~/ar-eye-calibration/software && source .venv/bin/activate
python3 augment_rig.py --selftest     # whole pipeline, headless (no cameras)
python3 augment_rig.py --run --imu    # live on the glasses (needs calibration + cameras)
python3 avatar.py --save-texture ~/Desktop/monkey.png   # see the placeholder sprite
```

## Precondition (important)

**The glasses must already be calibrated.** This runtime *consumes* the calibration — it does not
create it. Before it works you need:

1. `connect.py --auto && --identify` — cameras mapped to roles.
2. The human-in-the-loop calibration loop run (`main.py`) so the pixel map (`calibrator`) is
   trained — this is what turns "a person at 3D point X" into "the right display pixel."
3. A rigid, bring-up-tested rig (`rig_test.py`).

## The pipeline (per frame)

| Stage | Module | What it does |
|-------|--------|--------------|
| synchronized grab | `sync_capture` | all cameras captured together, with jitter |
| adjust image | `autoexpose` | keep the world frames well-exposed + sharp |
| head pose in world | `world_mesh` | where the glasses are in the world (VO + IMU) |
| find + track people | `people_track` | detect in both world cams → stereo 3D → **stable IDs**, dead-reckon through dropouts |
| world-lock | `anchor` | 3D world point → display pixel via mesh pose + the **calibrated map** (parallax-correct, uses stereo depth) |
| monkey | `avatar` | pose/scale/orient a monkey billboard to each person, depth-sorted for occlusion |
| display | `augment_rig` | composite (black = transparent) → AR display |

Because every corner of each monkey is a **world point re-projected each frame**, the monkeys stay
locked to the people. A fixed display pixel would smear off as the head turns (proven in
`anchor.py --selftest`: 252 px drift over one head move, 0.00 px with re-projection).

## Honest scope — what's real, what's a placeholder

- **Registration + tracking are complete and proven in sim** (stereo 3D recovery, stable IDs,
  occlusion coasting, parallax-correct world-lock — all in the release gate). This is the hard part.
- **Detector**: the default is OpenCV's built-in HOG pedestrian + Haar face (no downloads) — it
  *works* but is modest (misses odd poses, ~real-time). Drop a YOLO/MediaPipe model into the
  `detect(frame) -> [Detection]` seam for robust, full-pose detection. The geometry is
  detector-agnostic.
- **"Realistic" monkey**: the asset is a **procedural billboard sprite** (a placeholder — see
  `monkey_sprite.png`). Realistic means swapping in (a) a real textured PNG via
  `MonkeyAvatar(texture=...)`, or (b) a **rigged 3D monkey** articulated to body-pose keypoints
  (feed keypoints as extra anchor points). A billboard already tracks position/scale/orientation;
  articulation (matching limb motion) needs a body-pose model.
- **Display FOV**: the AR display is ~50° — a person closer than ~3.3 m won't fit head-to-toe, so
  the monkey is **clipped** to the display window (their visible part still shows).
- **Nothing has run on hardware yet.** All sim-validated.

## Making it look good later

1. Swap the sprite for a real monkey texture / 3D model.
2. Add a body-pose model (MediaPipe Pose / YOLO-pose) → articulate the monkey to the person's limbs.
3. Add face-landmark alignment so the monkey face maps onto the real face (head turns, expressions).
4. Temporal smoothing on the monkey pose (already have velocity prediction) to kill jitter.

The registration backbone doesn't change for any of these — they're all asset/detector swaps at
the seams this pipeline already exposes.
