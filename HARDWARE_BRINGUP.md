# Hardware bring-up: plugging the rig into the Mac

Everything below runs from `software/` in the venv:

```bash
cd ~/ar-eye-calibration/software
source .venv/bin/activate      # numpy + opencv (already installed)
```

The whole stack is **simulation-first**: every module has a `--selftest` that runs headless with
no cameras, so the logic is proven before hardware. `python3 verify_all.py --fast` runs them all
as part of the release gate.

## The 6 cameras → roles

The pipeline addresses cameras by **role**, not by USB index:

| Role | Sensor | What it sees |
|------|--------|--------------|
| `worldL`, `worldR` | AR0234 color, 1280 | the real world (stereo), the mesh + overlay register to this |
| `eyeL`, `eyeR` | OV9281 mono NoIR, 640 | outer eye corner (canthus), how the glasses sit on the face |
| `pupilL`, `pupilR` | OV9281 mono NoIR, 640 | the IR-lit pupil + glints (dark-pupil / PCCR gaze) |

### 1. Map the cameras (once)

```bash
python3 connect.py --auto        # probe + classify: splits 2 color/1280 world vs 4 mono/640
python3 connect.py --identify    # live view; press 1-6 to confirm each role, s to save
python3 connect.py --show        # print the saved map (data/rig_cameras.json)
```

`--auto` classifies by color-vs-mono and resolution automatically. It can't tell **L from R** or
**eye from pupil** from a static frame, so confirm those with `--identify` (or the IR-strobe test
in `rig_test.py`, where only the pupil cams brighten when the LEDs pulse).

### 2. Test the rig is fit to use

```bash
python3 rig_test.py --run        # enumerate · per-role fps · SYNC JITTER · IR test · FOCUS
```

Watch for: every role **OK**, `fps ≥ 30`, **sync jitter under your budget** (default 3 ms), the
2 pupil cams showing a positive **IRΔ** when strobed, and each `focus` high (set the lenses now , 
they're glued after). A `CHECK` row tells you which camera to fix.

### 3. Grab synchronized snapshots

```bash
python3 snapshot.py              # one synchronized set -> data/snapshots/<stamp>/
python3 snapshot.py --burst 20   # 20 sets (e.g. a calibration sweep)
```

Each set is 6 PNGs + `meta.json` carrying the timestamps, measured `jitter_ms`, and a `synced`
flag. A smeared (over-budget) set is saved but flagged `synced:false` so it never becomes a
calibration sample by mistake.

## Synchronous capture

`sync_capture.SyncBank` runs one background thread per camera parked on a shared **barrier**; a
`sync_frame()` releases them together so all six `grab()` calls fire in the same window
(global-shutter OV9281 + AR0234 ⇒ a genuinely comparable set). Every set reports `jitter_ms` , 
the honest spread of capture instants. In the selftest the barrier holds ~0.2 ms vs ~60 ms for a
naive one-at-a-time sweep. For tighter-than-software sync, strobe the IR LEDs
(`firmware/ir_strobe`) inside the shared exposure window.

```bash
python3 sync_capture.py --probe  # open real cameras, print live jitter
```

## The image constantly adjusts itself

`autoexpose.AutoExposureBank` closes an exposure/gain loop **per role, every frame**:

- **world** → mid-gray, exposure capped so motion stays sharp for the mesh tracker;
- **eye-corner** → mid-toned canthus for stable template matching;
- **pupil** → the *glints* (99th percentile) sit just under clipping while the field stays dark , 
  the opposite of consumer auto-exposure, which would blow the glints out.

It keeps adjusting as the light changes and settles without hunting.

## Real-world mesh tracking

`world_mesh.py` turns the two world cameras into a live model of the real world:

1. **triangulate** ORB features matched across the stereo pair → 3D points (left-cam frame);
2. **track** them frame-to-frame (optical flow) and solve the rigid **camera motion** with a
   robust Kabsch fit (RANSAC + Tukey-IRLS rejects mismatched features);
3. **fuse the IMU**, `imu_serial.GyroIntegrator` integrates the gyro into a per-frame rotation
   increment, used as a prior and to carry the pose through a visual dropout;
4. **maintain the mesh**, a growing cloud of world map points + a Delaunay surface over the
   visible ones. This is the frame the AR overlay locks onto as the head moves.

The geometry is proven in numpy (`--selftest` recovers a known trajectory + map, survives 25%
feature outliers, and shows the IMU carrying a dropout). On hardware `WorldTracker` feeds it
ORB/LK features from the world cams.

## Run everything together

```bash
python3 live_rig.py --run          # sync grab -> auto-expose -> mesh track -> feature capture
python3 live_rig.py --run --imu    # also fuse the XIAO gyro
```

One loop: `SyncBank` → `AutoExposureBank` → `WorldTracker`/`WorldMesh` → `capture.LiveCapture`
features, printing live telemetry (fps, jitter, exposure-settled, mesh size, camera pose,
tracking health). This is the "cameras capture synchronously and constantly track + adjust the
image" driver.

## USB / power notes

Six UVC streams need a **powered USB hub** and MJPG (already set) to fit the bandwidth; if a
camera drops out under load, lower `fps` in `SyncBank` or the world resolution. Running 6 cameras
+ the IMU is I/O-bound, not CPU-bound, it will not fry the Mac (the trackers are light; the mesh
front-end is the heaviest and still runs comfortably at capture rate).

## One command to verify the whole software stack

```bash
python3 verify_all.py --fast       # includes every hardware-layer selftest above
```
