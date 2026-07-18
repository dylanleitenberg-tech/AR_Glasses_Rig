# Assembly, full build, step by step

Builds the **6-camera binocular** rig from the printed **carrier** (`cad/xreal_one_mount.scad`),
mounted with **removable padded brow CLAMPS + M3 thumbscrews** (no adhesive, bonding via
`bond_pads` is the later rigid-phase upgrade). The carrier holds all 6 cameras **and** the IMU
(midline tower); the **IR rings are 2 separate prints** (`part="ir_ring"`). The XREAL One
Pro's **own padded nose pads** carry the face, no separate printed nose bridge is required.
Wire it to your host through the powered hub. Read `EYE_TRACKING.md` first, especially **§3 IR
eye safety**, and `WIRING.md` for the power + safety interlocks, and `ORDER_LIST.md` for parts.
Allow an afternoon; aiming/focus is the fiddly step.

Cameras (6-cam CORE): **2 world** (outward, on the brow), **2 eye-corner** + **2 NIR pupil**
(eye-facing, one of each per eye). The 8-cam FULL upgrade (+2 stereo) comes later, build the
CORE first.

Legend:  🔧 do  ·  ✅ check  ·  ⚠️ safety

---

## 0. Pre-flight, measure and tune (before printing)

1. 🔧 With **calipers**, measure on your glasses: the **brow thickness**, the **bridge bar
   thickness**, and roughly where each **pupil** sits. Measure your **IPD**.
2. 🔧 In `cad/xreal_one_mount.scad` set `target`, `xr_size`, `ipd`, `xr_brow_z`, `clip_x`,
   `OPTIC_DROP` for your unit (see `cad/README.md`). Leave `build_stereo = false` (6-cam CORE).
3. ✅ Open it in OpenSCAD, **F5 preview**: the eye-facing lenses point at the pupils/canthi,
   no camera sits in the see-through cone (the red cone is the preview guide; `software/cad_fit.py`
   enforces it), the IR rings (baffled) sit around both eye apertures, and the IMU pocket is on
   the rail.

---

## 1. Print (stiff material, the frame must not flex)

1. 🔧 Print in **SLA resin or PC / nylon-CF** (NOT plain PETG/PLA), stiffness keeps
   camera-to-frame motion < 0.2 mm. 3 perimeters / high infill, brim on thin booms.
   ```
   openscad -D 'part="carrier"' -o carrier.stl  cad/xreal_one_mount.scad
   openscad -D 'part="ir_ring"' -o ir_ring.stl  cad/xreal_one_mount.scad   # 2 rings in one file
   ```
   (The IMU tower is part of the carrier; the IR rings print separately and mount at the lens
   rims. `part="bond_pads"` exists for the LATER rigid/bonded phase only.)
   **Print a cheap PLA DRAFT first** (fast settings, supports under the horizontal boom legs),
   dry-fit it on the glasses + your face, and only then print the stiff version.
   ⚠️ **Brow heat (measured on this unit):** the One Pro's brow gets noticeably warm in use , 
   enough to soften PLA over a long session. Keep PLA-draft wear sessions SHORT (fit checks
   only, glasses powered off when possible); the PETG/resin final is unaffected.
2. 🔧 Glue the **1.5 mm carbon-fibre rods** into the boom channels (`cf_rod_d`) for stiffness.
3. ✅ Test-fit the carrier's clamps on the brow **before** populating electronics: jaws over
   the brow taper, silicone pads in, thumbscrews snug. Re-print with adjusted `brow_grip_t` /
   `clamp_clear` if tight/loose.

---

## 2. Prep the cameras (6)

1. ✅ Confirm each **eye-facing** camera (2 eye-corner + 2 NIR pupil) is **mono + global
   shutter + NO IR-cut** (NIR-sensitive). World cams are ordinary visible wide-FOV.
2. 🔧 Fit an **IR-pass filter** over each **NIR pupil** cam lens (passes 940 nm glints, blocks
   visible). The eye-corner cams stay NoIR (no filter).
3. 🔧 Set focus close-range (~2-4 cm to the eye); fine-tune on a live view in step 6.
4. 🔧 **LOCK each M12 lens after final focus** (step 6): tighten the lens locking ring
   (usually included with the module, check on arrival), or a dab of removable
   threadlocker / PTFE tape on the threads. Unlocked M12 lenses unthread with vibration
   and thermal cycling, silently ruining focus calibration.

---

## 3. Populate the carrier (cameras + IR rings)

1. 🔧 Screw each camera board onto its holder's **4 M2 standoffs** with **SELF-TAPPING M2
   screws** (the 1.8 mm bores are sized for these, NOT heat-set inserts), cable out the notch,
   zip-tied to the strain-relief post. Six holders: worldL/R (brow), eyeL/R (temple),
   pupilL/R (under-eye). There is **no printed lens shroud**: after focusing, baffle stray
   light on the NIR cams with a slip-on collar or matte tape around the M12 barrel.
2. 🔧 Press the **940 nm IR LEDs** into the ring bosses around **both** eye apertures, seated in
   their **baffles** (`ir_baffle`: keeps each LED recessed and ≥ 10 mm from the eye). Bend leads
   back along the rail.
3. ⚠️ **Wire each LED with its current-limiting resistor (330-470 Ω, ~5-15 mA).** Never connect
   an IR LED straight to 5 V. Spread, low current, see `EYE_TRACKING.md §3`.
4. 🔧 Common the LED leads into the IR bundle (each LED with its own resistor). This bundle is
   the **strobed IR branch**: keep it as a twisted pair, separate from the data cables.

---

## 4. Mount the IMU (robustness sensor, slip/motion + Kalman drift filter)

1. 🔧 Seat the **ICM-20948** breakout flat in the carrier's IMU pocket (`imu_mount`, on the
   rail), **board X → +x (right), Y → +y (forward)**, M2 screws, so its tilt maps to pose
   `dev[0]`/`dev[2]` as in `software/imu.py`.
2. 🔧 Run the IMU's **I2C** leads (SDA/SCL/3V3/GND) back with the bundle to the strobe MCU.
3. ✅ It must be **rigidly** coupled (no wobble), any flex reads as false tilt that even the
   Kalman drift filter (`imu.py`) shouldn't have to fight.

---

## 5. Wire + power it up (strobed + interlocked, see WIRING.md)

Power tree: **12 V brick → industrial USB 3.0 hub → 5 V/3 A main rail**, split to (a) the 6
bus-powered cameras + hub, and (b) **a 300 mA PTC polyfuse → the IR branch**.

1. 🔧 Cameras (all 6) → the **industrial powered USB 3.0 hub**. Run MJPEG.
2. 🔧 Decouple the 5 V rail at the carrier: **470 µF 10 V electrolytic + 0.1 µF ceramic**.
3. 🔧 IR branch: 5 V rail → **300 mA polyfuse** → LED strings; switch the **low side with a
   2N7002 logic-level MOSFET** (NOT a 2N2222) gated by the **MCU**, with a **gate pull-down**
   (fail-safe OFF). Add a **5 V TVS** clamp across the rail.
4. 🔧 Trunk wire = **24/26 AWG silicone** (carries 3 A); component branches = **30 AWG** only.
5. 🔧 Route the **IR twisted pair separate from the USB data cables**; give the IR branch its
   **own return to the rail's star-ground point** (EM + data-integrity).
6. 🔧 **IMU I2C → the strobe MCU** (Mac path: it bridges I2C→USB and runs the safety watchdog).
7. ✅ With the host running, all **6** cameras enumerate (+ the MCU as a serial device):
   ```
   cd software && source .venv/bin/activate && python3 main.py --list-cams
   ```

---

## 6. Mount to the glasses + pad + aim/focus (the important step)

1. 🔧 **Clamp** the carrier onto the brow: silicone pads lining both jaws, drop the clamps
   over the brow, tighten the **M3 thumbscrews** until snug (pads take up the brow taper, do
   not crank). Fully removable, no marks. (**Bonding** on the `bond_pads` hard-points is the
   later rigid-phase upgrade once the geometry is proven.)
2. 🔧 **Padding:** the XREAL One Pro's **own padded nose pads** carry the face. The carrier mounts
   to the glasses brow, not skin, but **pad any printed surface that does touch the face** (the
   `brow_pad` recess) with a soft silicone/foam pad. No bare hard plastic on skin.
3. 🔧 Put the glasses on. Open a live view of each eye-facing camera. For each eye:
   - ✅ The **pupil + at least 2 glints** (pupil cam) and the **outer canthus** (corner cam) are
     in frame across your normal gaze range.
   - 🔧 Re-aim the boom (or re-print with tweaked positions) until they stay framed.
   - 🔧 Adjust focus until the **iris texture / canthus is sharp**.
5. 🔧 **Route cables per WIRING.md:** thin leads (IR pair, IMU I2C) in the rail's wire
   groove under the tabs; the 6 fat camera USB cables on the rail's rear top edge, lashed by
   zip ties through the rail's through-slots; bundle exits one rail end, along that temple.

---

## 7. Exposure + IR setup (and the safety pass)

1. 🔧 Set a **short exposure** (global shutter) + gain: pupil reads **dark**, glints read
   **bright** and crisp.
2. 🔧 Flash `firmware/ir_strobe/` to the MCU. ✅ **Verify the fail-safe FIRST:** with the host
   app NOT running, the IR gate is **low (off)** and stays off; unplugging USB kills IR.
3. ⚠️ Now enable strobing and back the **IR current DOWN** to the lowest level that still gives a
   clean pupil + glints. Confirm 940 nm (invisible). Strobe sync ON (pulse only during exposure).
   If you have a meter, verify corneal irradiance well under ~1 mW/cm².
4. ✅ Confirm the interlocks: **blink cutoff** (IR drops when an eye closes), **voltage
   safeguard** (IR disables if the 5 V rail leaves ~4.5-5.5 V), **polyfuse** (a deliberate brief
   short on the bench trips and self-resets). See `WIRING.md §ELECTRICAL SAFETY INTERLOCKS`.

---

## 8. Software bring-up

1. ✅ **IMU drift filter** self-test (no hardware): `python3 main.py --imu-test` → the Kalman
   filter should cancel gyro drift (raw drift ≫ filtered).
2. ✅ **Pupil tracker** self-test: `python3 pupil_tracker.py` → detects the synthetic pupil ~1 px.
3. 🔧 Point it at a real pupil camera and confirm a live lock (`python3 pupil_tracker.py --cam <i>`).
4. 🔧 Capture eye-corner templates, then run the calibration loop:
   ```
   python3 main.py --world-cam-left 0 --world-cam-right 1 --eye-cam-left 2 --eye-cam-right 3 --fullscreen
   ```
   In the loop: **ENTER** approve, **U** undo the last (mis-stored) sample, **Z** cancel the
   current nudge, **Q** quit. `--reset-db` wipes all samples if the data got corrupted.
5. ✅ After ~20-40 corrections across the field, measure overlay error against a known target.

---

## 9. Validate

- ✅ **Tracking stability:** eyes still → steady pupil center (low jitter); during saccades it
  keeps up (global shutter earning its place). The live `GazeStabilizer` smooths fixation jitter.
- ✅ **Calibration accuracy:** overlay error vs a known target is your real result, compare
  honestly to the simulation, not the other way around.
- 🔧 Poor tracking? More even IR (move/add ring LEDs), better focus, IR-pass filter, shorter
  exposure, re-aim. Illumination usually matters more than the camera.

---

## ⚠️ Safety recap
- 940 nm only, low current (~5-15 mA), spread, **strobed in sync with the shutter**, recessed
  behind baffles **≥ 10 mm from the eye**, powered only while tracking.
- **Fail-safe IR-OFF:** USB/host drop, over/under-voltage, polyfuse trip, blink, or MCU reset all
  cut IR immediately (2N7002 gate pull-down). Verify this before trusting the rig.
- The retina can't feel IR, when unsure, use **less** light. Never use security-camera IR
  illuminators near the eye.
