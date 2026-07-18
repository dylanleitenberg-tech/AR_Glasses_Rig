# CAD, the camera carrier

The build needs `xreal_one_mount.scad`: the **removable clamp-on carrier** (padded brow clamps
+ M3 thumbscrews; bonding is the later rigid-phase upgrade) that holds all 6 cameras and the IMU.
Print TWO parts from it: `part="carrier"` and `part="ir_ring"` (the rings mount at the lens rims).

| File | Status | What it is |
|------|--------|-----------|
| **`xreal_one_mount.scad`** ⭐ | **THE part, print this** | Removable clamp-on **6-camera BINOCULAR** carrier for the **XREAL One Pro**: 2 world (outward) + 2 eye-corner + 2 NIR pupil (one of each per eye) + the IMU tower; separate `part="ir_ring"` print for the 940 nm rings (baffled, ≥10 mm standoff). `build_stereo=true` adds the 2 stereo cams → 8-cam FULL upgrade. |
| `eye_tracking_bridge.scad` | optional / legacy | A clip-on nose + IMU support. **Not required**: the IMU and face padding are on the carrier now; the XREAL's own nose pads carry the face. |
| `glasses_frame.scad` | legacy / Path B | A full from-scratch Meta-style frame for the DIY-combiner fallback (no XREAL display). |

> **Default display = XREAL One Pro** (57° diag FOV, 87 g, flat X-prism optic, front frame
> ~11 mm, IPD size M 57-66 / L 66-75 mm). Set `target` (`"xreal_one_pro"` / `"xreal_one"` /
> `"rokid_max"`) and `xr_size`. XREAL publishes no mechanical CAD, **confirm gripped edges with
> calipers** before printing.

The authoritative camera geometry is `software/rig.py`; `xreal_one_mount.scad` places each
camera's optical centre at `sim2cad(rig.py position)`, and `software/cad_fit.py` verifies every
real board clears the see-through cone, the eyeball, and the face.

## Install OpenSCAD

```bash
brew install --cask openscad        # macOS  (or download from https://openscad.org)
```

## Preview

Open `xreal_one_mount.scad` and press F5 (preview) / F6 (render). `part = "all"` shows the
assembly with translucent glasses + eye stand-ins, colored by subsystem:

- grey = rail + brow clips      · green boards = the camera modules you buy
- orange = camera holders/booms · red = IR LED rings (baffled)
- seagreen = IMU pocket         · gold = epoxy bond pads

The red cone is each eye's see-through keep-out (no camera may sit inside it).

## Export STLs for printing

One print covers everything (cameras + IR baffles + IMU pocket are all on the carrier):

```bash
cd ~/ar-eye-calibration/cad
openscad -D 'part="carrier"'   -o carrier.stl   xreal_one_mount.scad   # the whole carrier
openscad -D 'part="bond_pads"' -o bond_pads.stl xreal_one_mount.scad   # (optional, separate pads)
# build_stereo=true on the carrier export adds the 2 stereo cams (8-cam FULL upgrade)
```

`part` options: `all` | `carrier` | `board_cam` | `bond_pads` | `imu_mount`.

## Tune before you print (important)

Measure your unit and edit the parameters at the top of `xreal_one_mount.scad`:

| Parameter | What to measure | Typical |
|-----------|-----------------|---------|
| `target`, `xr_size` | your XREAL model + size | `"xreal_one_pro"`, `"M"` |
| `ipd` | your interpupillary distance | 64 (M) / 69 (L) mm |
| `xr_brow_z` | front-to-back thickness of the brow edge the clip grips | ~11 mm |
| `clip_x` | a flat span on the brow to clamp | ~38 mm |
| `OPTIC_DROP` | how far the optic centre sits below the brow rail | measure |
| `cf_rod_d` | carbon-fibre stiffening rod diameter (0 = none) | 1.5 mm |
| `ir_ring_r`, `ir_min_standoff` | IR ring radius + min LED-to-eye clearance (safety) | 13 / ≥10 mm |

Print one cheap draft, dry-fit on the brow, confirm in OpenSCAD that the eye-facing lenses
frame the pupils/canthi and no camera enters the cone (run `python3 software/cad_fit.py` after
any position change). The pupil/canthus aim is the part most worth iterating.

## Material & print notes

- **Stiff material, SLA resin or PC / nylon-CF filament, NOT plain PETG/PLA** (the carrier must
  hold camera-to-frame motion < 0.2 mm). 0.2 mm layers, 3 perimeters / high infill.
- Glue **1.5 mm carbon-fibre rods** into the boom channels (`cf_rod_d`) for stiffness; the booms
  are trussed so they can't sag/resonate.
- **Pad any printed surface that touches the face** (the `brow_pad` recess) with soft
  silicone/foam, the XREAL's own pads carry the nose. No bare hard plastic on skin.
- **IR safety is mechanical too:** each LED sits in a baffle (`ir_baffle`) that recesses it and
  keeps the ring ≥ `ir_min_standoff` mm from the eye, see `EYE_TRACKING.md §3`.
