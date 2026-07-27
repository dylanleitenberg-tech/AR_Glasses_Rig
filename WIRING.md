# Wiring, power & safety, the 6-camera binocular build (XREAL One Pro)

How every camera, LED, and the IMU connects, how the rig is powered, and the electrical
safety interlocks that protect the eye. The authoritative camera geometry is
`software/rig.py`; the carrier that holds them is `cad/xreal_one_mount.scad`; the IR-safety
rationale is `EYE_TRACKING.md §3`; the strobe/interlock logic is `firmware/ir_strobe/`.

---

## The model: 6 cameras, BINOCULAR

The XREAL One Pro has a **display per eye**, so both eyes are tracked and registered.

**CORE = 6 cameras = 2 outward + 4 eye-facing:**

| group | role | cams | sensor | purpose |
|---|---|---|---|---|
| **outward (2)** | `worldL`, `worldR` | 2 | ELP AR0234, visible, wide FOV | the AR dot / forward scene (shared by both eyes) |
| **eye-facing (4)** | `eyeL`, `eyeR` | 2 | OV9281 mono NoIR, global shutter | eye-corner (canthus), glasses-on-face pose |
| | `pupilL`, `pupilR` | 2 | OV9281 mono NoIR + IR-pass | NIR pupil + corneal glints (PCCR), one per eye |

\+ **940 nm IR LED brackets — 2 LEDs per eye = 4 total** (strobed; on the separate printed
`led_bracket` parts under each eye, aimed at the pupil) and a **6-axis IMU (MPU-6050;
ICM-20948 also works)** on the carrier's midline TOWER (flat shelf on the rail top,
board X→+x right / Y→+y forward, the `imu.py` axis contract — only accel + gyro are used).
Honest accuracy: ~4.3 px PERCEIVED deployed (vernier UI + per-user offset, simulated with realistic user error), 0.89 px pipeline bound; the <1 px pathway (stereo + multi-vergence kappa) awaits hardware validation.

> **FUTURE UPGRADE (separate tier, not the 6-cam model): 8-camera FULL.** Adds `eye2L`,
> `eye2R` (a stereo eye-corner cam per eye), the accuracy upgrade. Build and validate the 6-cam CORE
> first; the stereo pair is the higher-accuracy evolution that comes after. In software it is
> the `use_stereo` / `build_stereo` path and is off by default.

```
  carrier (bonded to the frame)                         host
   worldL,R ───USB──┐
   eyeL,eyeR ──USB──┤      ┌─ industrial powered USB 3.0 hub ─┐
   pupilL,R(NIR)USB─┤ 6×USB┤   (own 12 V PSU -> 5 V/3 A rail) ├─USB─► Mac
   IR bracketsL,R ─► MCU(XIAO): strobe + IMU I2C→USB ───────USB──┘
   IMU ──I2C──► MCU
```

The 6 UVC cameras run **MJPEG** so they fit one good industrial USB 3.0 hub's bandwidth.
(See the bandwidth note at the bottom; the 8-cam FULL upgrade is what pushes you to a Jetson
/ a second host controller.)

---

## Power architecture

One AC source feeds everything: a **single industrial USB 3.0 hub with its own 12 V power
supply**, which regulates down to a **5 V / 3 A (15 W) main rail**. Nothing is drawn from the
Mac's port for power.

> ### ⚠️ Three wiring details that protect parts (get these right)
> 1. **IMU on 3.3 V, never 5 V.** Power the MPU-6050 from the XIAO **3V3** pin. Many GY-521
>    boards pull SDA/SCL up to VCC, so 5 V would drive the I²C lines to 5 V and **over-volt the
>    XIAO's 3.3 V GPIO (abs-max ~3.6 V).**
> 2. **Rail-monitor ADC needs a 2:1 divider.** The XIAO ADC maxes at 3.3 V — feed the 5 V rail
>    through **two equal ~10 kΩ resistors** (5 V → 2.5 V), never straight in. (Resistors are in
>    the kit.)
> 3. **IR 5 V + the divider tap the RAIL, not the XIAO's 5 V pin** — keeps the pulsed strobe
>    current off the MCU's power path.
>
> All three are buildable with parts already in the BOM — no extra components.

```
  AC mains
   └─ 12 V DC brick (the industrial hub's own PSU)
        └─ INDUSTRIAL USB 3.0 HUB  (12 V in → regulates to 5 V)
             └───────────────  5 V / 3 A (15 W) MAIN RAIL  ───────────────┐
                  │   decoupling: 10 V 470 µF electrolytic + 0.1 µF        │
                  │   ceramic across the rail at the carrier (stabilises   │
                  │   the 5 V against the pulsed IR load + camera inrush)  │
                  │                                                        │
       ┌──────────┴───────────┐                          ┌────────────────┴───────────────┐
       │ CAMERAS + hub logic  │                          │ 300 mA PTC POLYFUSE             │
       │ (6× UVC, bus-powered │                          │   └─► IR branch (on carrier):   │
       │  5 V on their own    │                          │     IR LED brackets 2×/eye 940nm│
       │  USB cables)         │                          │        switched by a 2N7000     │
       └──────────────────────┘                          │        MOSFET (MCU strobe)      │
                                                          └────────────────────────────────┘
```

- **12 V → hub → 5 V/3 A rail.** The hub's internal regulator is the 5 V/3 A (15 W) source.
  At MJPEG / modest resolution (~0.25-0.35 A per UVC camera) the 6 cameras draw ~1.5-2.1 A,
  leaving headroom for the low-current IR (**~0.05 A: 4 LEDs × ~12 mA**) + IMU branch. ⚠️
  **Verify your cameras' actual draw**, at full resolution 6 cams can approach the 3 A limit;
  if so, raise the rail budget (a higher-current industrial hub) or run lower resolution.
  (The SABRENT 60 W / 12 V-5 A hub supplies ~5 A at 5 V — comfortable margin over 3 A.)
- **The rail SPLITS two ways:**
  1. **Cameras + hub controller**: the six cameras are USB **bus-powered**, taking clean 5 V
     from the hub's own ports. No separate camera PSU.
  2. **300 mA PTC polyfuse → IR-bracket branch**: the IR branch taps the 5 V rail
     through a **resettable 300 mA polyfuse** so any IR-side short or overcurrent trips the
     branch open (and self-resets after power-down) without taking down the cameras.
     ⚠️ **Tap this 5 V from the rail (a hub-port 5 V), NOT through the XIAO's 5 V/VBUS pin** —
     so the pulsed strobe current stays off the MCU's power path. The XIAO only supplies the
     *gate signal* and reads the rail (below).
- **Decoupling caps:** a **10 V 470 µF electrolytic** (bulk) **+ a 0.1 µF ceramic** (HF) sit
  across the 5 V rail at the carrier. The bulk cap absorbs the IR strobe's pulsed current and
  camera inrush so the rail doesn't sag/ring; the ceramic kills high-frequency switching
  noise. Place one set near the IR driver, one near the camera hub feed.

### Wire gauge
- **Main 5 V trunk** (hub → carrier): **24 or 26 AWG silicone wire**: low resistance so the
  rail holds 5 V under the full 3 A with minimal drop and warmth. Use a separate matched
  return of the same gauge.
- **Individual component branches** (a single LED string, the IMU 3V3, one cap lead):
  **30 AWG silicone** only, thin and flexible so the bundle stays light on the face.
- Never run the 3 A trunk on 30 AWG.

### IR power vs data, EM separation + star ground
The IR LEDs are **pulsed** (strobed in sync with the shutter), so their supply current is a
square wave that will couple into nearby high-speed USB data if you let it. Two rules:

1. **Separate the pulsing IR power lines from the data lines.** Run the IR V+/return as their
   own **twisted pair**, physically routed away from the USB camera cables (cross at 90°, never
   bundle them parallel for any length). This kills the magnetic-loop coupling.
2. **Isolated return / STAR GROUND.** The IR branch gets its **own return conductor back to a
   single star-ground point** at the 5 V rail, it does **not** share the camera/data ground
   along its length. That keeps the pulsed IR current out of the data ground, so it can't
   create ground-bounce on the USB signals. One star point; IR ground and data ground meet
   only there.

---

## ⚠️ Electrical SAFETY INTERLOCKS (all default to IR-OFF, fail-safe)

The retina cannot feel IR (no blink reflex to warn of overexposure), so the IR drive is built
**fail-safe**: it is OFF unless the MCU is **actively** driving it during an exposure window,
with the host connected and the rail in spec. Logic lives in `firmware/ir_strobe/`.

1. **Strobe in sync with the shutter.** The MCU pulses the IR only during each camera's
   exposure window (a few hundred µs), not continuously. Average IR power = peak × duty cycle
   (≪ 1), so dose and heat are a fraction of continuous-on. The 2N7000 gate is driven only
   inside that window.
2. **USB drop instantly kills IR.** The MCU requires a periodic host heartbeat over USB; if
   the USB/serial link drops (unplug, host sleep, app crash), a **watchdog forces the 2N7000
   gate low within one frame** → IR off. (And because the LEDs are powered from the hub's 5 V
   rail, pulling USB/host power also de-energises them outright.)
3. **Voltage safeguard.** The MCU watches the 5 V rail on an ADC; if it leaves a safe window
   (≈ 4.5-5.5 V, brownout, regulator fault, wrong PSU) it disables the strobe. ⚠️ **The XIAO
   ADC maxes at 3.3 V, so the 5 V rail MUST feed it through a 2:1 resistor divider** (two equal
   ~10 kΩ from the resistor kit; 5 V → 2.5 V, so firmware scales the reading ×2). **Never feed
   5 V straight into the ADC pin — it will damage it.** A **TVS diode / clamp** (SMAJ5.0A)
   across the rail handles fast transients; the polyfuse handles sustained overcurrent.
4. **Polyfuse overcurrent protection.** The **300 mA PTC** in series with the IR branch trips
   open on any IR-side short/overcurrent and self-resets after power-down, overcurrent can
   never drive the LEDs (or the wiring) to a dangerous level.
5. **IR cutoff on blinks.** The host runs `software/blink.py`; on a detected blink it commands
   the MCU to drop that eye's IR for the closure. No point illuminating a closed lid, it cuts
   heat build-up and average dose with zero tracking cost.
6. **Any power anomaly → IR off immediately.** Brown-out, over/under-voltage, polyfuse trip,
   watchdog timeout, or MCU reset all converge on the same state: **gate low, IR off.** A
   gate **pull-down resistor** guarantees OFF during power-up and any MCU reset.
7. **Mechanical IR keep-out (see CAD).** A printed **baffle/standoff** around each LED keeps
   the emitter recessed and ≥ 10 mm from the eye and blocks any close-range direct path
   (`ir_baffle` in `cad/xreal_one_mount.scad`): the electrical limits and the mechanical
   block back each other up. Run each LED at ≈ 5-15 mA through its resistor regardless.

---

## Connection table

| Item | Connector | To | Power | Notes |
|---|---|---|---|---|
| World cam L / R | USB | hub | bus 5 V | visible, wide FOV, shared by both eyes |
| Eye-corner cam L / R | USB | hub | bus 5 V | OV9281 mono global shutter, NoIR |
| NIR pupil cam L / R | USB | hub | bus 5 V | OV9281 mono + IR-pass filter (one per eye) |
| IMU 6-axis (MPU-6050; ICM-20948 OK) | I2C (SDA/SCL/**3V3**/GND) | strobe MCU (Mac path) | ~5 mA, **3.3 V — NOT 5 V** | ⚠️ power from the XIAO **3V3** pin: many GY-521 boards pull SDA/SCL up to VCC, so 5 V would push the I²C lines to 5 V and **over-volt the XIAO's 3.3 V GPIO**. Rigid on the midline tower; accel+gyro for slip + Kalman drift (`imu.py`) |
| IR LED brackets (2 LEDs/eye = 4) | twisted pair | 5 V rail **via 300 mA polyfuse**, low-side **2N7000** | ~8-12 mA/LED (330-470 Ω each), ~48 mA total | 940 nm; strobed; per-LED resistor; 5 V tapped from the **rail, not the XIAO**; **isolated return / star ground** |
| Strobe MCU (Seeed XIAO) | USB | Mac | bus | strobes IR (2N7000), bridges IMU I2C→USB, runs the safety watchdog |
| **10 kΩ gate pull-down** | XIAO strobe GPIO → GND | at the MOSFET gate |, | GPIOs FLOAT during MCU boot/reset; the pull-down holds the gate OFF until firmware takes over (the LEDs would only reach the resistor-limited safe current anyway, Rule 7, but a floating gate must not flash them at all) |
| Industrial USB 3.0 hub | USB-C/A + 12 V brick | Mac + AC | 12 V → 5 V/3 A | the 5 V/3 A (15 W) main rail; self-powered |
| Decoupling |, | across 5 V rail |, | 10 V 470 µF electrolytic + 0.1 µF ceramic |

---

## Pin map, Seeed XIAO RP2040 (exact connections)

| XIAO pin | Connects to | Wire | Function |
|---|---|---|---|
| **D0** (GPIO26) | 100 Ω series → 2N7000 **gate**; **10 kΩ pull-down** gate→GND at the FET | 30 AWG | IR strobe control (LOW = off; floats-at-boot held off by the pull-down) |
| **D4** (GPIO6, SDA) | MPU-6050 **SDA** | 30 AWG (twisted with SCL) | IMU I²C data |
| **D5** (GPIO7, SCL) | MPU-6050 **SCL** | 30 AWG | IMU I²C clock |
| **3V3** | MPU-6050 **VDD** (⚠️ **3.3 V, NOT the 5 V pin** — keeps its I²C at 3.3 V) | 30 AWG | IMU power (~5 mA) |
| **A1** (GPIO27, ADC) | **2:1 divider midpoint** (5 V rail → 10 kΩ → A1 → 10 kΩ → GND) | 30 AWG | rail monitor; reads ~2.5 V at 5 V, firmware ×2. ⚠️ never wire 5 V straight to A1 |
| **GND** | MPU-6050 GND + 2N7000 **source** + IR-branch return + divider bottom (star point) | 30 AWG | common ground, ONE star point at the perfboard |
| **USB-C** | Mac (direct or dock, NOT through the camera hub) | stock | power + serial (IMU stream out, strobe/watchdog control in) |

2N7000: **drain** → IR LED cathode rail; LED anodes → per-LED 330-470 Ω → 5 V **via the
300 mA polyfuse** off the hub rail (~48 mA total, 4 LEDs). 470 µF + 0.1 µF + SMAJ5.0A TVS
across 5 V/GND at the perfboard. **The IR 5 V and the ADC divider both tap the RAIL, not the
XIAO's 5 V pin.** Firmware contract: gate LOW at boot, RP2040 **hardware watchdog** armed
before the first strobe, any host silence > 500 ms ⇒ gate LOW + watchdog reset.

## Cable routing on the glasses (comfort + strain relief + EM)

1. **Thin leads** (IR twisted pair, IMU I2C) lie IN the carrier rail's wire groove (top-rear,
   3×2.5 mm, under the retaining tabs). **Fat stock USB cables** (all 6 cameras) lie ON the
   rail's rear top edge, lashed by zip ties through the rail's 4 through-slots (x ±8 / ±25).
2. The combined bundle exits at ONE rail end and runs along that **temple arm** to the tip
   (centred loads only in the groove, so nothing tugs one nose pad).
3. **Keep the IR twisted pair separate from the USB data cables** inside the bundle, a thin
   spiral wrap or its own channel; cross other cables at 90°, never parallel-bundle for length.
4. **Strain-relieve at two points**: a clip at the bridge and one at the temple tip, so a
   yank is taken by the frame, not the nose or a solder joint.
5. **Main 5 V trunk = 24/26 AWG silicone; component branches = 30 AWG.**

---

## Bring-up sequence (electrical)

1. Plug the hub's **12 V brick into the wall first**, then the cameras into the hub, then the
   hub to the Mac. (Rail up before devices = clean enumeration.)
2. Confirm the **5 V rail reads 4.9-5.1 V** under load before energising IR.
3. `python3 software/main.py --list-cams` → you should see **6** camera indices. Note which
   is which (worldL/R, eyeL/R, pupilL/R).
4. Flash `firmware/ir_strobe/` to the XIAO. Verify the **fail-safe**: with the host app NOT
   running, the IR gate is **low (off)** and stays off.
5. Only then enable IR strobing **at low current**, with the host running. ⚠️ 940 nm, ~5-15 mA,
   spread out, strobed, see `EYE_TRACKING.md §3`.
6. `python3 software/pupil_tracker.py --cam <pupil-index>` → confirm a live pupil lock.

---

## Bandwidth note

Six UVC MJPEG streams fit a single quality **industrial** USB 3.0 hub on one Mac controller;
that is why the build standardises on one hub (and one 5 V/3 A rail). The **8-camera FULL
upgrade** (+2 stereo) is the case that exceeds one controller's headroom → split across a
second powered hub on a separate host port, or move to a **Jetson Orin Nano** (the cameras
stay UVC, so the capture code ports with little change). Run MJPEG either way.
