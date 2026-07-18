# AR Eye Calibration Rig

AR glasses drift on your face all day, and every millimeter of slip moves where a display
pixel lands in the real world. This project is a camera rig and learning system that
measures how the glasses sit on your face in real time and corrects the overlay, so that
a pixel meant to land on a real object actually lands on it.

I designed and built this from scratch: the physics simulation, the calibration math, the
3D printed camera mount, the electronics and safety design, and the verification tooling
that checks all of it.

## Build overview and results

The rig is a 6-camera clip-on carrier for the XREAL One Pro AR glasses:

- 2 forward color cameras (global shutter) that triangulate a target drawn on paper
- 2 cameras watching the corners of your eyes, which is how the rig knows the glasses
  pose on your face at every instant
- 2 near-infrared cameras imaging each pupil, with 940 nm illumination behind hardware
  safety interlocks
- an IMU for slip and bump detection, all on one printed part that clamps to the glasses
  brow with padded jaws shaped to the measured curve of the frame

Results so far, from a physics simulation built to be pessimistic about human error:

- With a naive calibration interface, deployed accuracy is about 13 pixels. Most of that
  error is the human in the loop, not the optics.
- With the interface I built instead (a vernier alignment task plus a per-user offset
  stage), simulated deployed accuracy is about 4.3 pixels as the user perceives it, on a
  1080p display per eye.
- The same pipeline with ideal inputs reaches 0.89 pixels, so the hardware and math
  support sub-pixel registration. Closing that gap requires a multi-distance calibration
  protocol that only works on real hardware. That is the next phase.
- Just as important are the negative results. I tested and rejected several appealing
  shortcuts (bias-invariant fingerprints, richer per-user correction models, direct
  geometry refinement on user labels) because the simulation showed each one quietly
  overfits. The failures shaped the design as much as the successes.

Current status: simulation and CAD are complete and pass a full automated release gate.
The carrier has been printed and dry-fitted on the actual glasses through several
revisions. Cameras are ordered and arriving, and hardware bring-up is the current phase.

## Why this problem

An AR display pixel corresponds to a ray leaving your eye through the optic into the
world. Where that ray lands depends on three things: the pixel you light up, the fixed
optics of the display, and where the glasses currently sit relative to your eyeball
(eye relief, slip down your nose, a few degrees of tilt, your IPD). That third item is
the killer. Shift the glasses 2 mm and the same pixel lines up with a different point in
the world. This is why one-time AR calibration never holds.

The key idea, which came out of an early design discussion with our robotics coach Kim,
is to measure the glasses-on-face pose continuously by watching the corners of your
eyes. The outer canthus is a stable facial landmark, and its position in a
glasses-mounted camera tells you exactly how the glasses are seated right now. The
system then learns a function from (target direction, eye corner geometry) to the
display pixel that lands on the target, using samples the user labels:

1. The forward camera pair finds a dot drawn on paper and triangulates it.
2. The model predicts a pixel and renders a red dot there.
3. You nudge with the arrow keys until the red dot sits on the real dot.
4. On approve, the rig snapshots the eye corner cameras at that instant and stores the
   sample.
5. Retrain, move the dot, re-seat the glasses, repeat.

After enough samples the function generalizes and places pixels correctly on its own,
even as the glasses move. Precision comes from the interface: the alignment task is a
vernier judgment (lining up segments, which people do several times more precisely than
covering a dot), because the simulation showed calibration accuracy is gated by the
quality of human corrections more than by any sensor.

## Why the XREAL One Pro

I originally wanted to modify Meta glasses, since they already carry cameras. They turn
out to be closed systems: Ray-Ban Meta has no display, and Meta Ray-Ban Display exposes
no framebuffer or camera access. The XREAL One Pro presents to the host as a normal
USB-C DisplayPort monitor, so the overlay renders to it with zero reverse engineering.
It is see-through with a display per eye, and it has a rigid front frame the printed
carrier clamps onto. The software is display-agnostic, and the CAD supports the Rokid
Max as a drop-in alternative.

## What is in the repo

- `software/` is the simulation and application code. The physics simulator models eye
  anatomy across a population, glasses slip, camera noise, lens distortion, blinks, and
  human error in the calibration loop. The application runs the live calibration with
  real cameras or fully simulated ones. Everything has self-tests.
- `cad/` is the parametric OpenSCAD model of the carrier. Camera positions are locked to
  the simulator's geometry file, and a parity check fails the build if the CAD and the
  simulation ever disagree about where a camera sits or points.
- `software/verify_all.py` is the release gate: one command that compiles everything,
  runs every self-test, checks CAD-to-simulation parity, verifies the printed solids
  against 35 pairwise clearance checks (parts against each other, the glasses body, the
  eyeballs, and the see-through view cones), exports the STL, and confirms it is a
  single connected solid.
- `ORDER_LIST.md`, `WIRING.md`, `ASSEMBLY.md`, `SAFETY.md`, and `EYE_TRACKING.md` cover
  parts, wiring with the power and interlock architecture, build steps, the hazard
  review, and the IR eye-safety design.
- `RESEARCH.md` maps the project onto the published state of the art (corneal-imaging
  calibration, PCCR gaze tracking, slippage-robust methods, synthetic training data,
  Quest Pro and Vision Pro benchmarks).

## Try it without hardware

The whole loop runs in simulation. It fakes the cameras and synthesizes ground truth, so
you can watch the model learn:

```bash
cd software
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --simulate     # arrow keys nudge, ENTER approves, Q quits
```

`python3 verify_all.py --fast` runs the release gate's quick tier. See
`software/README.md` for the full module breakdown.

## Building the hardware

1. Print the carrier from `cad/xreal_one_mount.scad` and the IR rings as a second small
   print. The clamp jaws are shaped to caliper measurements of the actual glasses brow,
   taken from a hand-drawn dimensioned sketch and fitted with a spline.
2. Buy parts per `ORDER_LIST.md`. Cameras: 4 global-shutter mono modules for the eye
   side, 2 global-shutter color modules for the world side. Color matters there because
   the world cameras are the color reference for generated overlay content.
3. Clamp the carrier on, wire everything through a powered USB hub (one USB-C cable to
   the computer), and read `SAFETY.md` and the IR interlock section of `WIRING.md`
   before powering any LED.
4. Run the corner calibration once, then run the loop.

The safety design assumes failure: the IR LED current limits are set so that even a
stuck-on illuminator stays under conservative exposure limits, with the strobe and
watchdog as extra margin rather than the safety mechanism.

## Large-scale simulation studies

The repo includes the tooling I used to answer design questions with data instead of
guesses: population studies across thousands of simulated eyes, a pixel-by-pixel sweep
that walked 100 eye geometries times 100 glasses positions across the display (about
198 million labeled corrections), and ablations that measured what each sensor actually
contributes. Several cameras earned their place in the build this way, and one proposed
sensor (an IMU for pose estimation) was demoted to bump detection because the data said
it added nothing to registration.

## Acknowledgments

The eye-corner measurement idea came from our robotics coach, Kim. Much of the early
speculation and problem-solving came out of design discussions with him.

## License

Personal research project. Do what you like with it.
