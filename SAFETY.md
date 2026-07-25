# Safety review, 6-cam binocular clip-on (2026-07-03 design)

Hazard-by-hazard review of the CURRENT design (removable clamp carrier, L-booms, IMU tower,
separate IR rings). IR specifics live in `EYE_TRACKING.md §3/3a`; electrical interlocks in
`WIRING.md`. This file is the consolidated checklist, walk it before every wear session
during bring-up, then before any change.

## 1. IR / eye (the governing hazard) , engineered, verify at build
- 940 nm only, ≥10 mm rim standoff (enforced by an `assert` in the CAD), recessed baffles,
  per-LED resistor 330-470 Ω (≈5-15 mA), 300 mA polyfuse branch, low-side MOSFET with
  default-OFF gate + watchdog. **NEW RULE 7 (EYE_TRACKING §3): resistor current must be
  safe for CONTINUOUS stuck-on**, never lean on the strobe/watchdog for safety.
- The interlock **logic** is verified in simulation: `firmware/ir_strobe/safety_model.py`
  self-tests every fault condition (host drop, under/over-voltage, blink, dose/duty caps,
  disable/reset) and asserts IR is forced OFF on each, all checks pass (2026-07-13). That
  proves the logic only; the same conditions must be bench-tested on real hardware at bring-up.
- ✅ At build: verify fail-safe FIRST (host app off ⇒ IR off; USB unplug ⇒ IR off), then
  **measure corneal-plane irradiance with the IR power meter** (ORDER_LIST), aggregate of
  ALL 2 LEDs per eye (on the under-eye illuminator bracket), target ≪ 1 mW/cm² average, and check the stuck-on (continuous) case.
- ⚠️ The retina cannot feel IR, never "check by feel"; never substitute security-camera
  IR illuminators.

## 2. Mechanical, near the face
- **Impact:** the carrier is rigid and worn on the face. Mitigations: desk use only , 
  do NOT walk around, drive, or do anything athletic wearing the rig; the padded clamps are
  a friction mount that can shear off upward in an impact (do not convert to the bonded
  mount until the desk-use phase is done); all printed ends are rounded (spheres/rbox).
- **Pointy bits audit:** M2 screw tips end inside the standoff bores (point away from skin);
  the clamp thumbscrews point FORWARD (away from the face); CF rod ends, trim flush and
  bury in the channel with adhesive, never leave a proud rod end (splinter hazard).
- **Nose-bridge drop legs** (pupil booms at x=±6): they hang 14.6 mm FORWARD of the lens
  plane, clear of the nose, but they are near the midline. ✅ Dry-fit check: put the draft
  on, look down/converge (reading posture) and confirm no contact at the nose in any pose.
- **Resin prints:** if SLA, fully wash + post-cure, uncured resin is a skin sensitizer.
  Pad every skin-contact surface (the jaws get silicone pads; the One Pro's own nose pads
  carry the face).

## 3. Electrical
- All logic/camera power is USB bus 5 V through the self-powered hub; no LiPo, no mains on
  the head. The 12 V brick stays on the desk.
- IR branch behind its own 300 mA polyfuse; star ground; twisted pair for the IR run
  (see WIRING). Insulate every joint near skin (heat-shrink, not tape).
- ✅ Before first wear: continuity + short check of the IR branch with LEDs disconnected;
  power the rig OFF-face for 10 min and touch-check every board.

## 4. Thermal
- 6 cameras ≈ 1-2 W each worst case; boards sit millimetres from skin at the temple and
  under-eye. ✅ During the first ON-face session, break every 10 min and touch-check the
  eye-corner + pupil boards; if any board is more than warm (>40 °C-ish), add standoff foam
  or reduce resolution/framerate. PLA softens ~55 °C, another reason drafts are PLA but
  the wear version is PETG/resin.

## 5. Weight + ergonomics
- The loaded carrier adds roughly 100-150 g to 87 g glasses, cantilevered forward. Expect
  nose-pad pressure and neck fatigue: limit early sessions to ~15-20 min, stop on pressure
  marks or headache. ✅ Weigh the loaded rig at build and record it (affects slip modelling
  too, heavier rig slips more).
- Cable drag: route per WIRING (groove + tie slots, one temple), leave a service loop at
  the temple tip, and make the hub-side connector the breakaway point (a snagged cable
  must unplug at the desk, not yank the head).

## 6. Vision
- The see-through cone is verified clear of every solid (cad_overlap.py), but only within
  the modelled 50°/29° display FOV, booms and boards DO obstruct far peripheral vision.
  Treat the rig as tunnel-vision equipment: seated desk use, no locomotion.

## 7. Hygiene
- NIR cams + IR rings sit close to the eyes: wipe the pads and any near-eye surface with
  isopropyl between users; don't share during illness; lash-contact with any lens = stop
  and re-fit (also ruins the images).

## Session checklist (run before every wear session at bring-up)
1. Verify the IR fail-safe (app off ⇒ dark on the NIR view).
2. Check thumbscrews snug, pads present, nothing loose (shake test off-face).
3. Run the 10-min thermal touch-check for this configuration.
4. Confirm seated use, cables strain-relieved, breakaway at the hub.
5. Set a timer (≤20 min early sessions).
