# HOW EVERYONE ELSE SOLVES THIS — and what it means for our rig

Written 2026-08-03 after a session spent fighting overlay lag and jitter by tuning filters. The
short version: **filtering is the wrong class of tool for latency, and the field has known this
since the 1990s.** A filter can only trade lag against jitter. The standard answer is to
**predict forward, then reproject late** — which removes lag *without* adding jitter, because it
adds information rather than smoothing it away.

Our measured numbers are in §6. Everything before that is what other people do.

---

## 1. THE FRAMING: latency is not one error among many, it is THE error

Holloway's *Registration Error Analysis for Augmented Reality* (Presence 6, 1997) did the
end-to-end error budget for an optical see-through HMD and found **system delay is the largest
single contributor to registration error** — larger than tracker noise, larger than calibration
error, larger than optical distortion. The reason is simple and applies exactly to us: every other
error is roughly constant, while the latency error is **proportional to head angular velocity**.
Stand still and a laggy system looks perfect. Turn your head and it is the only error you can see.

That reframes our whole session. We measured 196 ms and treated it as a tuning problem. In this
literature 50 ms was considered typical for a mid-range 1990s system, and modern XR targets
**under 20 ms motion-to-photon**. We were 10x off the modern target and calling it smoothing.

**Consequence for us:** effort spent on calibration accuracy is wasted while latency dominates.
Dylan's instinct — *"this must run perfectly before anything else can be done"* — matches the
literature exactly.

---

## 2. WHAT EVERYONE ACTUALLY DOES: predict, then reproject late

Two distinct mechanisms, usually stacked. Neither is a filter.

### 2a. POSE PREDICTION (removes lag before rendering)
Estimate where the head **will be** when the photons actually land, and render *that* pose. The
look-ahead equals the measured end-to-end latency. This is why every XR runtime's tracking API
takes a **target display time** rather than "now" — OpenXR's `xrLocateViews` is explicitly
predictive, and the runtime is expected to extrapolate.

The reference open implementation to read is **[`fraunhoferhhi/pred6dof`](https://github.com/fraunhoferhhi/pred6dof)**
(Gül et al., ACM MM 2020), which is a Kalman-filter 6DoF head-motion predictor evaluated at
**20/40/60/80/100 ms** look-ahead against an autoregressive baseline and a no-prediction baseline,
over 14 real HoloLens motion traces. Reported: the Kalman predictor beats autoregression by
**~0.5° at 60 ms look-ahead**. Structure is small and readable — `runners.py` holds the predictors,
`evaluator.py` the MAE/RMSE metrics, and the traces are resampled to 200 Hz.

The critical property, and the reason this beats filtering: **prediction removes lag without
adding jitter.** A filter must smooth to reduce noise, and smoothing *is* lag. A predictor
extrapolates, so it can be simultaneously smooth and current. It fails differently — badly at
turn onsets and reversals, where extrapolation overshoots — which is why the recent work
([Predictability-Aware Motion Prediction, arXiv 2507.13179](https://arxiv.org/html/2507.13179))
is about knowing *when* the motion is predictable and backing off when it is not.

### 2b. TIMEWARP / LATE-STAGE REPROJECTION (removes the lag that remains)
Just before the display scans out, take the frame that was already rendered and **warp it using
the freshest tracker data**. Rotation-only warp is a cheap 2D homography, which is why it is
affordable at field rate. This is Asynchronous Timewarp (ATW) on the consumer headsets, and the
literature describes early+late two-stage variants that additionally fix pose error in the
received frame and cut colour separation.

The key insight for us: **the warp runs at display rate and is decoupled from the render loop.**
Our render loop is 17.9 fps; a reprojection stage does not care, because it only needs the latest
rotation and the last completed frame. That is precisely the structure that makes a slow camera
loop acceptable.

Open-source runtime to read: **[Monado](https://monado.freedesktop.org/)**, the FOSS OpenXR
runtime — `src/xrt/compositor` is where distortion and reprojection live, `src/xrt/auxiliary` has
the tracking/filtering utilities. It is the most directly readable production implementation of
this whole stack.

---

## 3. THE SENSOR ARCHITECTURE: fast IMU + slow camera, fused

Nobody tracks head motion at camera rate. The universal pattern is:

- **IMU at 100–1000 Hz** → propagation. Gyro integrates cleanly over short intervals; this is what
  gives low-latency rotation.
- **Camera at 10–60 Hz** → correction. Fixes the drift the IMU accumulates, and supplies absolute
  position/scale.

This is Visual-Inertial Odometry. The three open implementations worth reading, all with published
comparisons ([arXiv 2108.01654](https://ar5iv.labs.arxiv.org/html/2108.01654)):

| system | approach | note |
|---|---|---|
| **[OpenVINS](https://docs.openvins.com/)** | MSCKF (filter): IMU propagates, camera updates | cleanest reference for the filter formulation; best-in-class monocular |
| **[VINS-Mono / VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)** | sliding-window optimisation | IMU pre-integration with bias correction, online extrinsic calibration, loop closure |
| **Basalt** | graph-based | fastest of the three; best stereo accuracy |

**This is the same conclusion the project already reached from the other direction.** `HANDOFF.md`
records that `WorldTracker` builds 0 map points because stereo initialisation degrades under
rotation-dominant motion with far landmarks — and that an IMU is the standard fix. The latency work
and the SLAM work want *the same component*.

---

## 4. FILTERING: what it is legitimately for

Filtering is not useless — it is for **sensor noise on an interaction signal**, not for latency.

The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) is the standard here and is what we now
use: `cutoff = min_cutoff + beta * |velocity|`, smoothing hard when still and opening up when
moving. It is ~20 lines and is in `software/smoothing.py`.

But note what it does **not** do: it cannot remove the 56 ms our pipeline takes to capture and
process a frame. No causal filter can. That is what §2 is for.

---

## 5. THE CALIBRATION SIDE, for completeness

- **SPAAM** (Tuceryan & Navab) — align a marker to a single 3D point from many viewpoints, solve
  the projection. This is what our calibration loop is. Known weakness: user alignment noise, and
  the need for varied viewpoint *and distance*.
- **INDICA** (Itoh & Klinker, 3DUI 2014) — split calibration into an **offline display part** and
  an **online eye-position part**, so a re-seat does not require recalibration. Directly relevant
  to our zip-tied, non-repeatable mount.
- **Corneal-imaging calibration** (Itoh & Klinker, TVCG 2015) — uses the display's reflection on
  the cornea, removing landmark tracking entirely.
- **Parallax-free OST-HMD work** ([PMC7806030](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7806030/))
  — conditions under which short-focal-distance OST displays mitigate parallax registration error.

---

## 6. WHAT THIS MEANS FOR *THIS* RIG, with our measured numbers

Measured on hardware 2026-08-03:

| quantity | value |
|---|---|
| full pipeline (4 cams + dot pair + 2 canthus) | **55.9 ms/frame = 17.9 fps** |
| overlay latency after the smoothing fix | **62 ms** |
| overlay latency before it | 196 ms |
| lag at a gentle 30 °/s head turn | 72 px (was 225 px) |
| modern XR motion-to-photon target | **< 20 ms** |

**We are ~3x over the target, and no amount of filter tuning will close it.** The remaining 56 ms
is capture-and-process; a causal filter cannot remove it, and the One Euro filter is already at the
point where reducing jitter costs lag back.

### The three things the field says to do, in order of value to us

1. **PREDICT FORWARD BY THE MEASURED LATENCY.** We know the number (62 ms). The literature says
   head motion is predictable at that horizon and that a Kalman predictor at 60 ms look-ahead is
   accurate to well under a degree. This is a software change with no new hardware, and
   `pred6dof` is a readable reference. **Highest value per unit of work.**
2. **USE THE XREAL'S IMU FOR LATE-STAGE ROTATION CORRECTION.** Already established as readable:
   the glasses enumerate as USB VendorID 13080 with HID interfaces, and there is working prior art
   (`SamiMitwalli/One-Pro-IMU-Retriever-Demo`, `adidoes/xrealair-sdk-macos`). An IMU at 100–1000 Hz
   against our 17.9 fps loop is exactly the fast/slow split in §3. **This is the piece that makes
   the slow camera loop stop mattering.**
3. **THEN the camera loop speed is nearly irrelevant** — which is the whole point of the
   architecture. Do not optimise the vision pipeline for latency before doing 1 and 2.

### What we should stop doing
- **Tuning filters to fix lag.** We now have the measurement to know it cannot work.
- **Treating calibration accuracy as the bottleneck** while 62 ms of latency dominates the
  registration error (§1). Holloway's analysis says the latency term swamps the rest under motion.

### Honest caveat
Prediction has a characteristic failure: it overshoots at motion onset and reversal, which will
look like a *different* kind of instability from what we have now. The recent literature is
explicitly about detecting when motion is unpredictable and reducing the look-ahead then. Expect
to need that, and expect the first naive implementation to feel worse at turn reversals before it
feels better overall.
