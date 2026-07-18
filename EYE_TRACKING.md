# Eye-tracking upgrade, design, placement, comfort, and safety

This **adds** real **eye tracking**: near-infrared (NIR) cameras + IR LEDs that image each
eye's **pupil and corneal glints**: **alongside** the eye-**corner** (canthus) cameras. In
the canonical **6-camera binocular** build (XREAL One Pro, a display per eye) the four
eye-facing cameras are **2 eye-corner + 2 NIR pupil** (one of each per eye); the two outward
world cameras see the AR dot. The pupil/glint cameras are an upgrade, not a detour, because
they give exactly what the AR registration needs, and more, while the corner cameras keep
pinning the glasses-on-face pose.

---

## 0. Why this is the right move (not just a cooler sensor)

The whole project's goal is: place an AR pixel so it lands on a real point, surviving the
glasses shifting on your face. That needs two things about the eye:

1. **Where the eye is** relative to the optic (eye relief, how the glasses sit).
2. **Where the eye is looking** (so the overlay can be foveated / registered to gaze).

Eye-corner cameras gave a weak proxy for (1) and nothing for (2). **Pupil + corneal
glints give both, directly:**

- **Glint geometry** (reflections of known IR LEDs off the cornea) pins the eyeball's 3D
  position relative to the glasses, far better than canthi.
- **Pupil-center → glint vector** gives gaze direction.

So this single sensor change makes the calibration *more* solvable, and unlocks foveated
rendering, auto-IPD, slip detection, and iris biometrics as free byproducts.

### Honest scope of "see inside the eye"
- **Iris** (the colored ring) and **pupil/limbus**: **yes**: NIR iris imaging is standard
  (it's how iris biometrics work). With a close camera + IR light you get a usable iris.
- **Retina / fundus** (back of the eye): **no.** That requires ophthalmoscope optics, a
  camera whose axis passes *through* the pupil with coaxial illumination, using the eye's
  own lens to focus on the retina. It cannot be done with a tiny off-axis camera on
  glasses. Anyone claiming glasses-mounted retinal imaging is overselling. We image the
  **anterior** eye (iris, pupil, sclera, glints). That is plenty.

---

## 1. Camera placement, the real analysis

The eye is a ~12 mm sphere with the cornea bulging at the front; the pupil sits at the
front center and the eyeball rotates (typically ±30°, up to ±50° at extremes) to fixate.
A tracking camera must keep the pupil + glints in view across that range. The constraints:

- **Occlusion:** the **upper lid + lashes** are the worst offenders (they cover the eye
  from above, especially when looking down). The lower lid moves less. The **nose** and
  medial canthus occlude from the nasal side.
- **Viewing angle:** the closer to on-axis (looking along the gaze), the rounder the pupil
  and the easier/more accurate the detection. Oblique angles make the pupil an ellipse
  (still trackable by ellipse-fitting) but at *extreme* angles the limbus edge clips it.
- **Comfort + optics intrusion:** cameras must not sit in your forward view, must not add
  pressure points, and must be light and balanced.

### The three candidate placements

| Placement | Pupil view | Occlusion | Comfort | Build difficulty | Notes |
|---|---|---|---|---|---|
| **Lower rim, looking up** (industry standard: Quest Pro, Pupil Labs, Varjo) | good, mild ellipse | **best**: lashes point away from a bottom-up view | good (weight on rim, not nose) | easy | the proven default |
| **Nose bridge / nasal**, looking out at the eye | ok, but oblique nasal view; worse when looking nasally; caruncle/nose can clip | medium | weight on the nose (already the contact point), must be light | easy, compact, central, hidden, one module feeds both eyes | the compact/central option |
| **Temple + hot mirror** (IR-reflective fold on the lens) | **best**: on-axis | low | good | **hard**: needs an IR mirror coating on the lens you don't own | best optics, worst DIY |

### Decision (SUPERSEDED): originally a nose-bridge module with camera arms, the CURRENT
### design mounts the NIR pupil cams UNDER-EYE on the one-piece carrier (see
### `cad/xreal_one_mount.scad` + RIGID_MOUNT_BUILD.md); the optical reasoning below still applies

The original nose-bridge design put the module at the bridge while borrowing the
lower-rim advantage:

- **Mount on the nose bridge** (central, hidden, one small module, short symmetric wiring,
  sits where the glasses already contact).
- **But the camera sits at the lower-nasal corner of each eye and looks up-and-out**, so it
  inherits the bottom-up placement's low lash occlusion and a workable pupil angle , 
  instead of a pure nasal side view (which struggles when you look toward your nose).
- Each eye gets its **own** camera (one bridge cam trying to cover both eyes is too
  wide-angle and too oblique for either). The bridge module carries both.

Net: central/comfortable mounting + near-lower-rim tracking quality. The NIR pupil cameras
sit under each eye looking up-and-out at the pupil, realised on the bonded carrier
(`cad/xreal_one_mount.scad`, `pupil_cam`, aimed with a look-at at the eye's centre of
rotation). The carrier is the single printed part, it also carries the eye-corner and world
cams, the IR rings, and the IMU.

### IR LED placement
Tracking quality depends more on **illumination** than on the camera. Place **3-4 IR LEDs
per eye**, spread around the eye (some on the bridge module, some on the lens rim) so you
get **multiple glints** (better eye-pose solve) and **even, shadow-free** pupil
illumination. Spread > clustered. Aim them at the eye, diffuse them slightly.

---

## 2. Comfort

The nose bridge is the comfort-critical contact, so:

- **Keep it light.** The two micro-cameras (~1 g each) + LEDs + thin wiring is only a few
  grams. Print the module thin-walled. Don't over-build the arms.
- **Soft, broad nose pads.** The module carries **stick-on silicone nose pads** (or
  reuses the XREAL One pads) on a slightly widened footprint to spread pressure, narrow
  hard plastic on the nose is what hurts.
- **Balance.** Mount cameras symmetrically; route the cable bundle straight back over the
  bridge so it doesn't tug one side.
- **No skin contact from hot/sharp parts.** Round all edges (the CAD does), keep LEDs and
  camera bodies off the skin, strain-relieve the cable so a snag doesn't yank the nose.
- **Removable.** It clips to the existing bridge, take it off without tools.

---

## 3. ⚠️ IR EYE SAFETY, read before powering any IR LED

You are putting infrared light next to the eye. **The retina cannot feel IR**, so there is
no pain or blink reflex to warn you of overexposure, this is exactly why IR sources can be
dangerous and why you must be conservative.

Rules this design follows:

1. **Use 940 nm LEDs**, not 850 nm. 940 nm is invisible (no red glow) and more of it is
   absorbed by the eye's water before reaching the retina. (850 nm has better camera SNR
   but a faint visible glow and slightly higher retinal hazard.)
2. **Low power.** Drive each LED at a **low current (≈5-15 mA)**, not its max. Commercial
   eye trackers sit far below safety limits; so do we. The BOM sizes resistors for this.
3. **Diffuse and spread** the light (multiple low-power LEDs, frosted) rather than one
   bright point, lower peak irradiance, better images.
4. **Strobe, don't stare.** Ideally pulse the LEDs only during the camera's exposure
   (sync), so average power is a fraction of continuous. At minimum, don't leave them on
   when not tracking.
5. **Stay well under the limit.** The relevant standard is **IEC 62471 / ICNIRP** infrared
   ocular exposure. As a conservative DIY rule of thumb, keep corneal irradiance **well
   under ~1 mW/cm²** average. If you can, measure it with an IR power meter; if you can't,
   stay at the low end of the current range and keep LEDs ≥10 mm from the eye.
6. **No high-power IR emitters.** Do **not** substitute IR illuminator modules meant for
   security cameras (hundreds of mW, meters of range), those are unsafe near the eye.
7. **Size the hardware for STUCK-ON.** The per-LED resistor must set a current that is safe
   for CONTINUOUS exposure at the mounted distance, never rely on the strobe duty cycle or
   the software watchdog for safety. Failure model: MOSFET fails short / MCU crashes with the
   gate asserted → the LEDs run continuous at the resistor-set current, and that state must
   already be under the limit. Strobing then only buys SNR and headroom, not safety.

If in doubt, use fewer/dimmer LEDs. The cameras are sensitive; you need less light than you
think. When you build it, verify the eye is comfortably illuminated in the camera at the
*lowest* current that works.

### 3a. The interlocks that ENFORCE the rules above (built fail-safe, default IR OFF)

The rules are only as good as the hardware that enforces them. The IR drive is OFF unless the
MCU is *actively* driving it during an exposure window, with the host connected and the rail
in spec. Full wiring + topology in `WIRING.md`; logic in `firmware/ir_strobe/`.

1. **Strobe in sync with the camera shutter.** IR pulses only during each exposure (a few
   hundred µs), never continuous → average power (and heat) = peak × a tiny duty cycle.
2. **A USB / host drop instantly kills IR.** The MCU needs a periodic host heartbeat; lose it
   and a watchdog forces the strobe MOSFET gate low within one frame. (Losing hub/USB power
   de-energises the LEDs outright too.)
3. **Voltage safeguard.** The MCU monitors the 5 V rail; outside ≈ 4.5-5.5 V it disables IR. A
   TVS clamp handles transients.
4. **Polyfuse overcurrent.** A **300 mA PTC** in series with the IR branch trips on any short/
   overcurrent and self-resets after power-down, overcurrent can't over-drive the LEDs.
5. **IR cutoff on blinks.** `software/blink.py` tells the MCU to drop IR for a detected closure
  , no dose into a closed lid, less heat.
6. **Any power anomaly → IR off immediately**, and a gate pull-down guarantees OFF through
   power-up / MCU reset. Use a **2N7002 logic-level MOSFET** (not a 2N2222 BJT) for the
   low-side strobe switch.
7. **Mechanical IR blocking / keep-out.** A printed **baffle + standoff** (`ir_baffle` in
   `cad/xreal_one_mount.scad`) keeps each LED recessed and **≥ 10 mm from the eye** and blocks
   any close-range direct path, the mechanical block and the electrical limits back each
   other up.

---

## 4. Sensor architecture (what to build)

| Block | Choice | Why |
|---|---|---|
| **NIR pupil cameras ×2** | OV9281 global-shutter **mono** + IR-pass, ~120 fps+ (one per eye) | global shutter freezes saccades (eyes hit 300-900°/s); mono = full-res; NIR + IR-pass = clean pupil/glint image |
| **Eye-corner cameras ×2** | OV9281 global-shutter **mono** NoIR (one per eye) | canthus tracking pins the glasses-on-face pose the pupil cams under-determine |
| **IR illumination** | 940 nm LEDs, ~3-6 per eye, low current, **strobed** | glints + even pupil light, invisible, eye-safe at low power; rings around BOTH eyes |
| **World cameras ×2** | ELP AR0234, wide FOV, **shared** by both eyes | forward dot/scene; binocular vergence also supplies target depth |
| **Host** | Mac + USB (6-cam CORE) **or** Jetson Orin Nano (8-cam FULL) | USB to start; Jetson for the stereo upgrade's bandwidth |

**Camera count: 6 total (BINOCULAR CORE)**: 2 outward world + 4 eye-facing (2 eye-corner +
2 NIR pupil), one of each eye-facing pair per eye. The **8-camera FULL** upgrade adds 2 stereo
eye-corner cams (the FULL accuracy upgrade; build the CORE first). Full wiring + power + safety in
`WIRING.md`.

**Speed matters:** during a saccade the pupil moves fast; rolling-shutter or low-fps
cameras smear it. Global shutter + ≥120 fps (OV9281 does this, more at reduced resolution)
is why the camera choice is what it is. Full part numbers + sourcing in
`ORDER_LIST.md`.

---

## 5. How it plugs into the calibration software

- `software/pupil_tracker.py` turns each eye frame into features: **pupil center, pupil
  ellipse, and glint positions** → a compact per-eye gaze/position vector.
- These **add to** (do not replace) the canthus features. Matching `software/config.py`
  feature order, the vector is the 8-feature base `[worldL_dot, worldR_dot, eyeL_corner,
  eyeR_corner]` **plus** the NIR pupil-centre features when `use_pupil` is on (the 6-cam CORE),
  and the **same** calibrator / prior / preset pipeline learns the mapping, now from a much
  stronger signal. The 8-cam FULL upgrade appends the stereo eye-corner features (`use_stereo`).
- The simulator (`autosim`) can be extended to emit pupil/glint features (the eye is
  already modeled as a sphere with a center of rotation, gaze, and entrance pupil; glints
  are corneal reflections of the modeled IR LED positions). That extension is the natural
  follow-on so the prior/identifier retrain on the new features.

See `ASSEMBLY.md` for the physical build and `ORDER_LIST.md` for parts.
