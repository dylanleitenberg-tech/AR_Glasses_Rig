# Media

CAD renders of the 6-camera binocular carrier, exported from
`cad/xreal_one_mount.scad` with the OpenSCAD CLI (2026-07-14):

- **`assembly_front34.png`**, `part="preview"`: the carrier on the XREAL One Pro
  stand-in (dark frame + translucent optic), with eyeball stand-ins. Orange = printed
  holders/booms, grey = rail + brow clamps, green = IMU pocket.
- **`assembly_top.png`**, the same preview from above (boom routing over the brow).
- **`carrier_only.png`**, `part="carrier"`: the single printable part, as sliced.

Regenerate: `openscad -o out.png --imgsize=1920,1440 --autocenter --viewall
-D 'part="preview"' cad/xreal_one_mount.scad`

## session-2026-08-04

Rig photos after the August software work: camera boards being measured with calipers, the
carrier clamped to the glasses, the eye corner camera on its boom, and the complete wired rig.
See `session-2026-08-04/README.md` for what each one shows.

## session-2026-08-03

Frames captured straight off the eye cameras during the August 3 session, including the
worn and not worn comparisons that the brightness normalisation work came out of. These are
raw sensor output, not photographs.
