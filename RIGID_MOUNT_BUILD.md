# Rigid‑Mount Build, design + complete order list

Why rigid (not a clip): the camera‑to‑display geometry must be a **fixed, one‑time‑calibrated
constant**, not a per‑use unknown. If it flexes it becomes a second unknown confounded with the
eye geometry (under‑determined, can't solve). The sim's whole premise is that this geometry is
constant and factory‑calibratable. Target: **camera‑to‑frame motion < 0.2 mm / 0.2°** (1 mm/1° =
+1.3 px, more than the entire accuracy budget, from the flex sweep in simulation; the
`data/` outputs are generated locally and not committed).

The glasses are the rigid reference: **XREAL One Pro** front frame (flat X‑prism optic, ~11 mm
front, 87 g). We bond a single stiff camera carrier to it.

---

## 1. Mount design

### Architecture
One **monolithic carrier** holds every sensor at the exact positions in `software/rig.py`.
The **6‑camera binocular CORE** = world ±[33.5,44,5] (2) + eye‑corner ±[69.5,−2,−6] (2) + NIR
pupil ±[25.5,−33,6] (2, one per eye); the **8‑cam FULL** future upgrade adds the stereo
eye‑corner pair ±[69.5,−16,−10] (optic‑centre origin, MEASURED IPD 67). World cams RAISED 30→44
(2026‑07‑02): at 30 the 38 mm board's lower standoffs + PCB edge sat inside the glasses BROW , 
found by `software/cad_overlap.py`, the printed‑solid vs glasses/cone/eyeball intersection check. These positions are **WEARABLE**:
the real 36-38 mm camera boards
clear the see‑through cone, the eyeball, **and the facial surface** (so the glasses still rest
normally on the nose + ears), verified by `software/cad_fit.py` (cone + eyeball) and
`software/wearable.py` (face). The eye cams sit at the temple with their boards extending **out**
to the temple (not deep into the gap, where a 36 mm board would jam into the cheek). The carrier
couples to the front frame two ways, pick one:

- **A. Bonded (recommended, most rigid).** Epoxy the carrier's 3 hard‑points (brow centre + both
  temple corners, `bond_pads()` in `xreal_one_mount.scad`) to the front frame. Semi‑permanent;
  makes cameras + display effectively one body. Best dimensional stability.
- **B. Kinematic (rigid *and* removable).** Bond 3 steel balls to the frame; the carrier has 3
  mating vee/cone seats + magnet preload → seats to microns, repeatably → no re‑calibration on
  re‑mount. Choose this only if the glasses must come off intact.

### Camera holders (they actually hold the modules)
Each camera (6 CORE / 8 FULL) mounts as its **real board**, not an abstract pocket: the carrier gives every
camera a **backing plate + 4 standoffs at the module's mount pitch** (≈28 mm for the OV9281;
**self-tapping M2 screws** into 1.8 mm bores + zip-tie backup slots, the PCB screws on), the **M12 lens looks out through open air**, and a
**cable‑exit notch** clears the connector. The optical centre (front of the M12 lens) sits at the
`rig.py` position; the PCB sits ~10 mm behind it, so the board stays out of the see‑through cone
(`board_cam` / `camera_holder` in `xreal_one_mount.scad`). Module footprints the holders are built
for: **ELP AR0234 USB 38×38 mm** (world), **Arducam OV9281 USB 36×36 mm** (eye/stereo/pupil). Run
`python3 software/cad_fit.py` after any position change, it re‑checks that the real boards still fit.

### Stiffening (stacks with A or B), the frame must be STIFF
- **Carbon‑fibre rod** (1.5 mm) glued through each cantilevered eye‑corner/pupil boom, `cf_rod_d`
  in the CAD bores the channel; CF beam ≈ 10× stiffer than PETG.
- **Print the carrier in a stiff material:** SLA resin (most rigid/precise) or PC / nylon‑CF FDM.
  Avoid plain PETG/PLA for the carrier. Short, thick, ribbed booms; a spread **truss** (two
  struts + root tie per boom, in `cam_at`) so no cantilever can sag or resonate.
- **IMU** in a rigid pocket on the carrier, board axes X→+x, Y→+y (`imu_mount`, matches `imu.py`).
  Rigid coupling matters: any flex reads as false tilt before the Kalman drift filter sees it.

### Face-contact padding + IR keep-out (comfort + safety)
- **Pad every printed surface that touches the face.** Each face-contact pad (nose pads, any
  brow/cheek rest) is a recess that takes a **soft silicone / PORON foam pad**: no bare hard
  plastic on skin. Round all edges (the CAD does).
- **IR blocking / standoff.** Each IR LED sits in a printed **baffle** that keeps it recessed
  and **≥ 10 mm from the eye** and blocks any close-range direct path (`ir_baffle` in
  `xreal_one_mount.scad`): the mechanical block backs up the electrical current/strobe limits.

### One‑time rig calibration (this is what "rigid" buys you)
Before/after bonding, calibrate the camera→display geometry **once** with a checkerboard imaged
through the optic from an **eyebox camera** (reuse a spare OV9281 or even a phone camera placed at
the eye position, no new part needed), store it as the fixed device extrinsics, and never touch it. A printed **bonding jig** holds the carrier at the correct pose relative to the frame while
the epoxy cures, accuracy of the bond depends on it.

### Build order
print carrier → press in cameras/IMU + glue CF rods → wire + strain‑relieve → mount on bonding
jig at correct pose → factory‑calibrate rig geometry → epoxy (or kinematic‑seat) to frame → cure
→ verify rigidity (wiggle test < 0.2 mm) → per‑user eye calibration.

---

## 2. Order list

Prices are rough CAD, verify at checkout. **CORE** = first build (honest sim accuracy ~4.3 px perceived with the vernier
calibration UI). **STEREO/▲** = upgrades for margin (~0.5 px), add after core works.

### A. Display + compute
| ✔ | tier | part | qty | ~CAD | notes / search |
|---|---|---|---|---|---|
| ☐ | core | **XREAL One Pro** AR glasses | 1 | 600-700 | the display + rigid frame; USB‑C DP. "XREAL One Pro" |
| ☐ | core | **Mac** (host) | have |, | USB‑C w/ DisplayPort‑alt‑mode (drives the glasses) |
| ☐ | ▲ | **NVIDIA Jetson Orin Nano** dev kit | 1 | 400-500 | **needed for the 8‑camera FULL stereo build** (USB bandwidth); the 6‑cam CORE runs on the Mac |

### B. Cameras + lenses
| ✔ | tier | part | qty | ~CAD ea | notes |
|---|---|---|---|---|---|
| ☐ | core | **NIR global‑shutter mono cam** (Arducam **OV9281** USB, **no IR‑cut**, M12), eye‑corner ×2 + NIR pupil ×2 (one per eye, binocular) | 4 | 75-95 | global shutter (freezes saccades), NIR‑sensitive, mono |
| ☐ | ▲ | same OV9281, **stereo** 2nd eye‑corner pair (8‑cam FULL upgrade) | +2 | 75-95 | second view per eye → triangulates canthus/CoR depth (1.5→1.15 px) |
| ☐ | core | **World camera** (ELP 2 MP wide‑angle USB, ~70-100°) | 2 | 40-55 | forward, at the pupils; sees the target/scene |
| ☐ | core | **M12 lens assortment** (90° eye, 55° pupil, 70° world) | 1 kit | 15-25 | if modules' stock lenses don't match the FOVs in `rig.py` |

### C. NIR illumination (PCCR for the pupil cam + lighting the eye‑corner cams)
| ✔ | tier | part | qty | ~CAD | notes |
|---|---|---|---|---|---|
| ☐ | core | **940 nm IR LEDs**, 3 mm (NOT 850 nm) | 1 pack (20-50) | 10-15 | spread around both eyes for even light + glints |
| ☐ | core | **IR‑pass filter** 850-1000 nm (one per NIR cam) | 3 (▲5) | 10-18 ea | passes 940 nm, blocks visible |
| ☐ | core | **Resistor kit** (need 330-470 Ω, ~10 mA/LED) | 1 | 10-14 | one current‑limit resistor per LED |
| ☐ | core | **Seeed XIAO** (SAMD21/RP2040) or Arduino Nano | 1 | 10-20 | strobes the LEDs, bridges IMU I²C→USB, runs the safety watchdog (`firmware/ir_strobe/`) |
| ☐ | core | **2N7002 logic‑level MOSFET** (low‑side IR strobe; NOT a 2N2222 BJT) | 1 pack | 8-12 | switches from 3V3, gate pull‑down = fail‑safe OFF |
| ☐ | core | **300 mA PTC polyfuse** + **470 µF 10 V caps** + **0.1 µF** + **5 V TVS** | 1 ea | 10-16 | IR‑branch overcurrent + 5 V rail decoupling + transient/voltage safeguard |
| ☐ | core | **Perfboard** | 1 | 9-13 | tidy LED + resistor + MOSFET board |

### D. IMU (robustness, slip/bump/motion, not geometry ID)
| ✔ | tier | part | qty | ~CAD | notes |
|---|---|---|---|---|---|
| ☐ | core | **6/9‑axis IMU** (ICM‑20948 or BNO055) breakout | 1 | 15-28 | rigid on the carrier; 3V3 logic |

### E. Mounting / structural, the rigid mount
| ✔ | tier | part | qty | ~CAD | notes |
|---|---|---|---|---|---|
| ☐ | core | **Rigid print material:** SLA resin **or** PC / nylon‑CF filament (1 kg) | 1 | 40-70 | the carrier MUST be stiff; not plain PETG/PLA. No printer → use a print service |
| ☐ | core | **Carbon‑fibre rod, 1.5 mm** (pack) | 1 | 10-15 | reinforce the eye‑corner booms (`cf_rod_d`) |
| ☐ | core | **Structural epoxy** (2‑part, plastic‑bonding e.g. JB Weld PlasticWeld / 3M DP) | 1 | 10-15 | bonds carrier to the frame; **+ isopropyl + abrasive** for surface prep |
| ☐ | core | **Self-tapping M2 screw assortment** (bores are 1.8 mm; inserts don't fit) | 1 | 10-18 | secure cameras + IMU in the carrier |
| ☐ | ▲ | **Kinematic kit:** 3× 6 mm precision steel balls + 3× neodymium magnets | 1 | 10-20 | only if you want it removable (Option B) |
| ☐ | core | **Bonding/alignment jig** | printed |, | holds the carrier at the correct pose while epoxy cures (print from the CAD) |

### F. Wiring / connectivity
| ✔ | tier | part | qty | ~CAD | notes |
|---|---|---|---|---|---|
| ☐ | core | **Industrial powered USB 3.0 hub w/ own 12 V PSU** | 1 (▲+1) | 40-70 | regulates 12 V → the **5 V/3 A (15 W) main rail**; one hub carries the 6‑cam CORE, ▲ a 2nd hub/Jetson for the 8‑cam FULL |
| ☐ | core | **USB‑C dock / multiport adapter** | 1 | 25-45 | the glasses take one Mac USB‑C (DP‑alt) and the hub another; a dock frees ports |
| ☐ | core | **Short USB cables / micro‑USB leads** for each camera | ~6 (▲8) | 3-6 ea | keep short; strain‑relieved |
| ☐ | core | **24/26 AWG silicone** (5 V trunk, 3 A) **+ 30 AWG silicone** (component branches) + heat‑shrink | 1 | 16-24 | trunk vs branch gauges; keep the IR pair separate from data (star ground) |
| ☐ | core | **JST‑SH connectors / Kapton tape** | 1 | 10-15 | tidy detachable joints + stray‑light baffling |

### G. Calibration + tooling
| ✔ | tier | part | qty | ~CAD | notes |
|---|---|---|---|---|---|
| ☐ | core | **Digital calipers** | 1 | 20-30 | measure the frame + your IPD to tune the CAD |
| ☐ | core | **Checkerboard / calibration target** (rigid, printed on flat board) | 1 | 15-30 | for the one‑time camera→display rig calibration |
| ☐ | core | **Soldering iron kit** | if needed | 30-55 | LED/wire work |
| ☐ | core | **Gamepad** (8BitDo / Xbox wired) | 1 | 30-60 | nudge / vernier alignment input |
| ☐ | ▲ | **IR / optical power meter** | 1 | 45-75 | verify corneal IR is eye‑safe (< ~1 mW/cm²), borrow if possible |

### H. Comfort / consumables
| ✔ | tier | part | qty | ~CAD | notes |
|---|---|---|---|---|---|
| ☐ | core | **Adhesive silicone nose pads** | 1 | 8-12 | comfort on the bridge module |
| ☐ | core | **Soft silicone / PORON foam pad stock** | 1 | 6-10 | pad **every** printed face‑contact surface (no bare hard plastic on skin) |
| ☐ | core | **Foam / Kapton tape** | 1 | 8-12 | baffle stray light around the eye cams |

---

## 3. Rough totals
- **CORE (6 cameras, Mac, bonded):** XREAL ~650 + 4 OV9281 ~340 + 2 world ~95 + IMU ~22 +
  IR/driver + protection (2N7002/polyfuse/caps/TVS) ~70 + filters ~45 + mount/epoxy/CF/inserts
  ~120 + industrial 12 V hub ~60 + wiring ~45 + calipers/target ~60 ≈
  **CA$1,550-1,850** (assumes you own a printer + soldering gear).
- **FULL/▲ upgrade (8 cameras):** +2 OV9281 ~190 + Jetson ~450 + 2nd hub ~40 + extra filters ~36 ≈
  **+CA$700-800**.

---

## 4. Don't‑miss / verification checklist
1. **Cameras (6 core / 8 full):** 2 world + 2 eye‑corner + 2 NIR pupil (one per eye), +2 stereo ▲
  , matches `rig.py` roles. Eye/pupil = global‑shutter mono NIR; world = wide‑angle. ✔
2. **Illumination:** 940 nm LEDs (both eyes) + per‑NIR‑cam IR‑pass filter + resistors + strobe
   MCU + **2N7002 MOSFET**; eye‑safety meter. (850 nm is wrong, visible + higher hazard.) ✔
3. **Power + safety:** 12 V → industrial hub → **5 V/3 A rail**; split to cameras+hub and (via a
   **300 mA polyfuse**) to IR; **470 µF/0.1 µF** decoupling, **TVS** clamp; strobe‑sync, USB‑drop
   kill, voltage safeguard, blink cutoff, fail‑safe‑OFF (`WIRING.md`/`firmware/ir_strobe/`). ✔
4. **Rigidity + padding:** stiff carrier material + CF rods + truss booms + structural epoxy
   (+ surface prep); bonding jig; **soft pads on all face‑contact surfaces**;
   **IR baffle ≥10 mm standoff**. ✔
5. **Compute/bandwidth:** the **6‑cam CORE** runs on Mac + one industrial hub; the **8‑cam FULL**
   needs a 2nd hub or a Jetson, UVC streams saturate USB. MCU also bridges the IMU's I²C. ✔
6. **Calibration/fit/tools:** checkerboard + eyebox view for the one‑time rig calibration; calipers
   to set IPD/`OPTIC_DROP`/vertex in the CAD; keyboard arrow keys for alignment (vernier.py); nose pads + baffling. ✔

Open hardware unknowns to settle by measurement (not simulation): the XREAL front‑frame material
(epoxy compatibility), the true vertex distance (~4.5 mm sim/CAD gap), and real human alignment
precision (sets how close the kappa calibration gets to the sub‑1 px result).
