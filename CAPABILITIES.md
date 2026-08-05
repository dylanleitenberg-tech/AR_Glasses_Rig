# RIG + XREAL CAPABILITY INVENTORY

**Why this file exists.** Dylan, 2026-08-03: *"don't underutilize our resources."* Capability was
scattered across 78 modules, `rig.py`, three hardware docs and several sessions' worth of
measurements, so it was genuinely hard to know what the rig could already do. Every number here is
either measured on this hardware or read from `rig.py` (the single source of truth), where
something is **assumed or unverified it says so explicitly**, because an inventory that mixes the
two is worse than none.

Regenerate the depth figures with `python3 software/depth.py --table`.

---

## 1. WHAT THE HARDWARE PHYSICALLY IS

| | spec | source |
|---|---|---|
| Glasses | XREAL One Pro, birdbath optical see-through | |
| Display to host | second monitor, **1920×1080** | measured |
| Display FOV | ~50° (used for px↔deg) | assumed, **not measured** |
| World cams | 2× ELP **AR0234**, 1920×1200 native, global shutter | |
| World FOV / res used | **70°**, 1280 px → focal **914 px** | `rig.py` |
| World baseline | **67.0 mm** (= `NOMINAL_IPD`) | `rig.py` |
| World cam position | x ±33.5, up 49.0, fwd 22.0 mm | `rig.py` |
| Eye cams | 2× **OV9281**, 1280×800 native, mono **NoIR**, global shutter | |
| Eye FOV / res | **45°**, 640 | `rig.py` (measured, was wrongly 90) |
| Eye cam position | x ±69.5, up −5.0, fwd −6.0 mm; aim down 6.0, out 3.0 | `rig.py`, as printed |
| Tracked landmark | **inner canthus** (`rig.TRACKED_LANDMARK`) | settled |
| Pantoscopic tilt | −3.5° | `rig.py` |
| Carrier | ASA/PETG, **FINAL, no printer access** | settled |
| Mount | **ZIP TIES**, not the M3 thumbscrews `ASSEMBLY.md` describes | measured 2026-08-03 |
| IMU | mount exists in CAD, **NOT wired** | |
| IR LEDs | **none wired** | settled |
| Bus | one SABRENT hub, **USB 2.0**, all four cams share it | measured |

### Hard limits, all measured
- **Four streams do not fit at native.** Three at native ≈ 37 fps is the ceiling; a fourth starves
  and stalls the bank to ~1 fps via `sync_capture`'s barrier.
- **640×400 for four streams**, *not* 640×480, see §5, this is a correctness issue, not just bandwidth.
- **No software exposure control on macOS.** AVFoundation rejects `CAP_PROP_EXPOSURE`,
  `AUTO_EXPOSURE`, `GAIN`, `BRIGHTNESS`; every `set()` returns False. Exposure is a physical lever only.
- **Full-load profile:** track 45.7 ms / sync 19.6 ms / overlay 8.8 ms = **73.6 ms/frame**, CPU 9%
 of 16 cores. The loop is **I/O-bound, not CPU-bound**, throwing quality away does not buy speed.

---

## 2. WHAT THE RIG CAN DO THAT THE GLASSES CANNOT

This is the section that matters for architecture.

| capability | rig | XREAL alone |
|---|---|---|
| Cancel head **rotation** | yes (IMU once wired, or world cams) | **yes**, near-zero latency, in the display pipeline |
| Cancel head **translation / parallax** | **yes, stereo depth** | **no**, needs a depth it cannot obtain |
| Measure **metric depth** | **yes**, 67 mm baseline | **no** built-in world cams (believed; *verify*) |
| Know **where a real object is** | yes | no, anchor holds where it was *placed* |
| Know **how the glasses sit on the face** | **yes, eye cams**, unique to this rig | no |
| Sub-50 ms motion-to-photon | no (73.6 ms loop) | yes |

**Why the glasses cannot get depth:** an IMU integrates rotation cleanly, but translation requires
double-integrating acceleration, which drifts quadratically, metres within seconds. Physics, not
firmware. A 3DoF stabiliser therefore cancels rotation *exactly* and translation *not at all*.

**Consequence (decided 2026-08-03):** keep the XREAL in **head-following (0DoF)**, never
anchor/locked, and do low-latency rotation compensation with **our own IMU** as late as possible
before drawing. Anchor mode inserts a transform we can neither query nor calibrate, which breaks
the calibration *and* the rendering. Full reasoning in `HANDOFF.md` → RENDERING ARCHITECTURE.

---

## 3. DEPTH SENSING: what the stereo pair is actually good for

`software/depth.py`. Regenerate with `--table`.

| distance | disparity | depth σ (0.5 px) | needed for 1 px parallax | uncorrected parallax per 10 cm |
|---|---|---|---|---|
| 0.3 m | 204 px | 1 mm | 0.4 mm | **708 px** |
| 0.5 m | 122 px | 2 mm | 1 mm | **434 px** |
| 1 m | 61 px | 8 mm | 5 mm | 219 px |
| 2 m | 31 px | 33 mm | 18 mm | 110 px |
| 5 m | 12 px | 204 mm | 114 mm | 44 px |
| 20 m | 3.1 px | 3.3 m | 1.8 m | **11 px** |
| 100 m | 0.6 px | 82 m | 45 m | 2 px |

**The design result:** stereo depth error and the depth accuracy *required* for sub-pixel parallax
**both scale as Z²**, so their ratio is **constant with range**, 1.80 at a realistic 0.5 px
disparity precision, 0.90 at an optimistic 0.25 px. The 67 mm baseline is matched to the task at
*every* distance, not merely up close. "Stereo dies at range" is only a problem if you forget that
distant objects are correspondingly forgiving of depth error.

**Honest reading:** at 0.5 px the ratio is 1.8, i.e. **~2 display px** of parallax error, fine to
look at, not "sub-pixel". Only the optimistic 0.25 px reaches sub-pixel, with 10% headroom. Both
are reported; neither is hidden.

**Refusals are first-class.** `StereoDepth.measure` returns `ok=False` with a named cause for an
epipolar violation, a **negative disparity** (reversed pair, never `abs()`'d, that bug let a
swapped pair pass a full preflight at 2054 mm), or disparity below the noise floor.

---

## 4. SOFTWARE CAPABILITY INVENTORY (78 modules)

**Perception**
- `depth.py`, metric stereo depth + uncertainty + refusal *(new 2026-08-03)*
- `world_mesh.py`, stereo VO/mesh tracking, `triangulate_stereo`, Kabsch. **0 map points on real
 frames**, a geometry problem (rotation-dominant motion, far landmarks vs 67 mm baseline), not a bug
- `world_memory.py`, rolling TSDF, forgets old geometry
- `people_track.py`, detect + track people in 3D
- `dot_detector.py`, calibration target; `detect_pair()` picks the stereo pair **jointly**
- `canthus_net.py`, learned inner-canthus landmark, pure numpy 13 ms/frame, + mount fiducial
- `pupil_tracker.py` / `pupil_sensor.py`, IR pupil + glint (**no IR LEDs wired**)
- `blink.py`, `eye_tracker.py`, closure detection, classical corner tracker
- `imu.py`, `imu_serial.py`, IMU filter + XIAO serial feed (**not wired**)

**Calibration**
- `calibrator.py`, `main.py`, the learning core and live loop
- `calib_preflight.py`, six preconditions, names the failing one
- `eye_check.py`, worn-rig eye diagnostic *(new)*
- `canthus_data/label/train/auto`, corpus → labels → model
- `intrinsics.py`, checkerboard per-camera calibration
- `display_calib.py`, measure real XREAL FOV + distortion
- `kappa*.py`, `vernier.py`, angle-kappa study, hyperacuity alignment
- `preset.py`, `identify.py`, `match.py`, `complete_geometry.py`, per-user eye geometry ID

**Rendering**
- `overlay.py`, the AR surface (black = transparent)
- `anchor.py`, world-locked projection
- `content_anchor.py`, surface placement + occlusion + persistence
- `avatar.py`, `augment_rig.py`, the people→monkeys runtime

**Infrastructure**
- `cameras.py`, `sync_capture.py`, `snapshot.py`, `bank_bringup.py`, `connect.py`, `rig_view.py`
- `perf.py`, frame budget, adaptive quality with an **effectiveness probe** (reverts a down-step
 that does not buy frame time, this rig is I/O-bound)
- `autoexpose.py`, per-role control (**inert on macOS**, see §1)
- `verify_all.py`, the release gate, **40 checks**

**Simulation**, `autosim.py`, `optics.py`, `anatomy.py`, `rig.py`, `megarun.py`, `accuracy_map.py`,
`pixel_sweep.py`, `autotrain.py`, `realtime.py`, `stereo_test.py`, `binocular.py`

---

## 5. UNDER-USED RESOURCES: the actual point of this file

1. **The IMU is unwired and solves THREE problems at once:** late-stage reprojection latency
   (73.6 ms loop = **283 px** of lag at a normal 100 °/s glance), `WorldTracker`'s 0 map points,
   and the rotation-dominant motion the SLAM literature says breaks stereo init. Mount is in the
   CAD. **Highest-value unbuilt work in the project.**
2. **The world cams were never used as a depth sensor** until `depth.py`, only for the dot and a
   mesh that returns 0 points. Depth is the rig's unique advantage over the glasses.
3. **640×480 wastes 70% of the eye sensor.** It is a **CROP** at scale 1.00, not a downscale , 
   eyeL's 640×480 covers only x[288..928], y[50..530] of 1280×800. `640×400` is a true 2× downscale
   keeping the **full FOV**, at **less** bandwidth. Fixed in `cameras.py`, `rig_view`, `eye_check`.
4. **Native resolution is available for 2-camera work.** The bus limit applies to four streams; the
   corpus is captured at full 1280×800.
5. **A pupil head on the canthus net is free**, auto-labelled by `pupil_tracker` (96–100% reliable).
   Literature: presence/absence 92.8% → 99.6%, which is exactly the weak closed-eye head. Do **not**
   expect registration gains (`eyetracker.py`: 9.9 px corner-only vs 10.1 px +pupil).
6. **The mount is a free fiducial.** Bolted to the camera, so it lands on the same pixels regardless
 of the face, `canthus_net.MountAnchor` uses it for drift. With a **zip-tie mount that is not
   repeatable between sessions**, making the plausibility band **mount-relative** rather than
   sim-derived would cancel seating entirely.
7. **Global shutter on all four cameras**, unexploited. Enables motion-robust capture and, with
   the IMU, proper rolling-shutter-free reprojection.
8. **`perf.py`'s effectiveness probe** already knows the loop is I/O-bound; the real speed lever is
   fewer/smaller reads, not lower quality.

---

## 6. KNOWN-UNVERIFIED: do not treat as fact
- **Display FOV ~50°** is assumed. `display_calib.py` exists to measure it and has not been run.
- **XREAL has no built-in world cameras** (XREAL Eye believed to be a clip-on), reasoned, not verified.
- **Whether follow mode is rigid or damped.** If damped, the image lags during fast head turns,
  which would fit "big head shifts make it not work". Unmeasured.
- **The 0.09 eye asymmetry is real geometry** (two independent labelling passes, gap 0.102 and
 0.084), but *why* the rig is asymmetric is unknown, and `rig.py`'s symmetric prior does not
  describe it.
- **The accuracy corpus is STALE**, `data/mega_prior*`, `calibration_db.npz`, the 1.88 px
  `accuracy_map` figure were all generated at `EYE_FOV=90` tracking the **outer** canthus.
