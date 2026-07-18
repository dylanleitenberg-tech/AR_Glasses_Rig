# XREAL One Pro, measurement checklist (do this once, carefully)

Each row maps a measurement → the parameter it sets → the file → the nominal value to sanity-check
against. XREAL publishes no mechanical CAD, so these come off **your** unit with calipers. After
measuring: edit the files, run `python3 software/cad_fit.py`, then re-run the sims.

**Already known (no measuring):**
- IPD = **67 mm** (measured) → `NOMINAL_IPD` (`rig.py`) + camera-bracket `ipd` (`xreal_one_mount.scad`)
- Size = **L** → `xr_size = "L"` (display optic centers ≈ 69 mm)
- `target = "xreal_one_pro"`

Tools: digital calipers, a small ruler, and the glasses both **on the table** and **on your face**.

---

## A. Frame measurements (glasses on the table)

| # | Measure | Sets | File | Nominal | How |
|---|---------|------|------|---------|-----|
| A1 | **Brow rail thickness** (front-to-back depth of the top edge the carrier clips over) | `xr_brow_z` | `xreal_one_mount.scad` | ~**11 mm** | calipers across the top edge, front face to back face |
| A2 | **Flat clamp span on the brow** (width of a flat, grip-able section, clear of sensors/dimming module) | `clip_x` + clip width | `xreal_one_mount.scad` | ~**38 mm** | find a flat run on the brow; note its width + where it is |
| A3 | **Optic drop**: vertical distance from the brow rail (where the carrier sits) DOWN to the **lens optical center** | **`OPTIC_DROP`** ⭐ | `xreal_one_mount.scad` | ~**17 mm** | ruler from the top brow edge straight down to the center of the see-through window |
| A4 | **Optic-center horizontal spacing** (center-to-center of the two see-through windows) | display IPD / `OPTIC_L/R` | `rig.py` | ~**69 mm** (L) | calipers between the two lens centers |
| A5 | **Front frame width** (outer temple hinge to hinge, front) | glasses stand-in, `clip_x` bound | `xreal_one_mount.scad` | ~**151.6 mm** | calipers across the front |
| A6 | **Front frame height** (top brow to bottom of the optic housing) | stand-in | `xreal_one_mount.scad` | ~**50 mm** | calipers vertically |
| A7 | **See-through window size** (the clear aperture per eye) | cone keep-out check | `cad_fit.py` |, | W × H of the transparent area (cameras must clear it) |
| A8 | **Brow-frame material** (note plastic type / finish) | epoxy choice for the bond pads |, |, | check for a resin code; note glossy/matte (affects surface prep) |
| A9 | **Obstructions on the top edge** (sensors, the electrochromic dimming module, ribbon cables) | where the clip may grip | `xreal_one_mount.scad` |, | note their positions so the clip avoids them |
| A10 | *(optional)* **Bridge bar thickness** (front-to-back), only if you also want the clip-on nose support | `xr_bridge_t` | `eye_tracking_bridge.scad` | ~**3.6 mm** | calipers across the nose-bridge bar |

## B. As-worn measurements (glasses on your face)

| # | Measure | Sets | File | Nominal | How |
|---|---------|------|------|---------|-----|
| B1 | **Vertex distance**: back of the lens/optic to the front of your **eye (cornea)** when worn | `EYE_BEHIND` (= vertex + ~12 mm globe) | `rig.py` | vertex ~**12 mm** → `EYE_BEHIND` ~**24 mm** | side view in a mirror; ruler from lens back to your eye surface (or have a friend judge) |
| B2 | **Pantoscopic tilt**: SET your preferred tilt stage first, then RECORD it; measure the lens-plane angle from vertical if you can | `PANTO_DEG` | `rig.py` | **−3.5°** = the max nose-down stage (One Pro adjusts ±3.5° in 3 stages; the sim now uses the achievable −3.5, fixed 2026-07-06) | note the stage; optionally photograph from the side and measure the angle |
| B3 | **Pupil vs optic center**: where your pupil sits relative to the lens center, vertically (horizontal you already have via IPD) | seating check |, | should be near center | look straight ahead in a mirror; note if your pupil sits high/low/off-center in the window |

---

## After measuring, the update loop
1. Edit `xreal_one_mount.scad`: `xr_size="L"`, `ipd=67`, `xr_brow_z`, `clip_x`, `OPTIC_DROP` (+ obstruction notes).
2. Edit `rig.py`: `NOMINAL_IPD=67`, `EYE_BEHIND`, `PANTO_DEG`.
3. `python3 software/cad_fit.py`: confirms every camera board still clears the cone + eyeball + face.
4. Open `xreal_one_mount.scad` in OpenSCAD (F5), eyeball that the clip matches your brow and no camera enters the see-through cone.
5. Re-run the sim suite (`--complete-geometry`, `--physics-preset`, etc.) for build-real numbers.
6. Print a **cheap PLA draft** of the carrier and dry-fit on the glasses BEFORE the rigid print.

## The three that matter most (if you do nothing else, nail these)
- **A3 `OPTIC_DROP`**: places every camera vertically. Get this wrong and everything is off.
- **A1 `xr_brow_z`**: whether the clip actually grips your brow.
- **B1 vertex / `EYE_BEHIND`**: the eye-relief the whole registration is built on (currently an open ~4.5 mm unknown).
