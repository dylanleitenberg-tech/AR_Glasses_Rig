// =====================================================================
//  AR Calibration Rig — REMOVABLE 6-camera BINOCULAR clip-on carrier for XREAL One Pro
//  (6-cam CORE; +2 stereo = 8-cam FULL future upgrade via build_stereo)
//  Fits XREAL One Pro (default), XREAL One, and Rokid Max / Max 2 (set `target`)
// =====================================================================
//  Your glasses ARE the display + frame (Path A). You print this carrier and CLAMP it onto the top
//  brow with the removable PADDED brow clamps (no glue, no marks — pops off clean). It holds the cameras as REAL board
//  modules (not abstract stand-ins). BINOCULAR: the XREAL One Pro has a display PER EYE, so both
//  eyes are registered. Build/test the 6-cam CORE first (build_stereo=false):
//      world  x2 : ELP AR0234 USB, 38x38 mm board + M12 lens   (brow, looks forward; SHARED)
//      eye    x2 : InnoMaker OV9281 USB, 32x32 mm board + M12   (temple; one per eye)
//      pupil  x2 : InnoMaker OV9281 NoIR USB, 32x32 mm + IR ring (under-eye; ONE PER EYE)
//  Accuracy (honest, sim-validated with realistic user corrections, 2026-07-03): ~4.3 px
//  PERCEIVED deployed (vernier UI + per-user offset); pipeline bound 0.89 px with ideal
//  corrections; the <1 px pathway (stereo + multi-vergence kappa separation) awaits hardware
//  validation. (2 world + 2 eye-corner register BOTH eyes; the 2 NIR cams add per-eye
//  geometry-ID + dichoptic alignment.) 8-cam FULL (build_stereo=true) adds:
//      eye2   x2 : Arducam OV9281 USB, 36x36 mm  (stereo eye-corner; one per eye).
//  All positions are WEARABLE (boards clear the cone, the eyeball, AND the face).
//  Every holder is a backing PLATE + 4 standoffs at the module's mount pitch (SELF-TAPPING
//  M2 screws into 1.8 mm bores + zip-tie backup slots) so the actual PCB screws on, with the
//  M12 lens looking out through open air and a cable-exit notch. Module sizes are from the vendor pages (see ORDER_LIST.md) and the fit of
//  the real boards (cone clearance + eyeball clearance) is verified by software/cad_fit.py.
//
//  COORDINATES (Z-up so the preview looks like worn glasses):
//      +x = right   +y = FORWARD (world)   +z = UP   origin at the brow rail centre.
//  The simulator's authoritative geometry is software/rig.py (x=right, y=UP, z=FORWARD, origin
//  at the optic centre). Mapping:  sim = (cad_x, cad_z + OPTIC_DROP, cad_y), i.e.
//  cad = sim2cad(sim) = [sim_x, sim_z, sim_y - OPTIC_DROP]. Each camera's OPTICAL CENTRE is
//  placed at sim2cad(rig position) so the printed lens lands on the validated geometry exactly
//  (parity checked by software/cad_fit.py / the CAD<->rig check).
//
//  ⚠️ MEASURE your glasses + modules with calipers (brow thickness, the PCB mount pitch) and
//     print a cheap PLA DRAFT to dry-fit before the rigid print. Export a printable part with `part`:
//       openscad -D 'part="carrier"' -o carrier.stl xreal_one_mount.scad   # the clip-on
//       openscad -D 'part="ir_ring"' -o ir_ring.stl xreal_one_mount.scad   # the IR ring
// =====================================================================

// PRINTABLE by default: `part="carrier"` renders ONLY the clip-on (rail + padded brow clamps +
// camera holders/booms + IMU pocket) as one connected solid — no glasses, no eye, no camera-module
// stand-ins. The IR ring is a SEPARATE print (`part="ir_ring"`) since it mounts at the lens rim.
// Set part="preview" to see everything (glasses + eye + bought modules) for checking the fit.
part = "carrier";        // "carrier"(print) | "ir_ring"(print) | "imu_mount" | "board_cam" | "preview" | "bond_pads"
show_glasses = false;    // translucent glasses stand-in — OFF for a clean printable view
show_eye     = false;    // translucent eyeball + canthus + cone — OFF for print
show_cams    = false;    // the camera MODULES (PCB+lens) you BUY — OFF (you don't print those)
build_stereo = false;    // false = 6-cam BINOCULAR CORE (2 world + 2 eye-corner + 2 NIR pupil,
                         // ~1.4px/eye); build + test this FIRST. true = add the 2 stereo
                         // eye-corner cams -> 8-cam FULL future upgrade (<1px).
$fn = 40;

// ---------- which glasses are you clipping onto? ----------------------
//  XREAL One Pro (measured specs): front frame ~151.6 mm wide x ~50.5 mm tall, 87 g, 57 deg
//  diagonal FOV (~50.6 deg horizontal on 16:9), pantoscopic tilt ADJUSTABLE +-3.5 deg in 3 stages.
//  IPD is HARDWARE-SIZED, not per-mm: Size M = 63 mm centre (fits 57-66), Size L = 69 mm (66-75).
//  Set xr_size to YOUR purchased size; it sets the display-optic IPD the calibration registers to.
target  = "xreal_one_pro";
xr_size = "L";           // MEASURED: Size L (this unit). "M" (63 mm, IPD 57-66) | "L" (69, IPD 66-75)
// MEASURED brow is a TAPER (front-to-back): ~8.9 mm at the hinge, 12.14 at center, ~19.9 at the brow
// peak. The removable C-clamp (below) grips top-brow + bottom-housing with padded jaws, so it doesn't
// depend on one brow thickness; xr_brow_z is the nominal used for the preview/stand-in only.
xr_brow_z = 19.4;   // nominal front-to-back depth (see the measured step profile in glasses_dummy)
clip_x    = (target == "rokid_max") ? 40 : 38;
// MEASURED user pupil IPD = 67 (cameras print over the pupils). The display optics are 68.13 mm
// center-to-center (rig.py DISPLAY_IPD); calibration learns the ~1 mm pupil-vs-optic offset.
ipd       = 67;
// ---- REAL-WORLD SIZE LEDGER (2026-07-17 audit) -------------------------------
//  MEASURED, in the model: OPTIC_DROP 25.3 | user IPD 67 | display IPD 68.13 |
//    vertex/EYE_BEHIND 28.5 | panto -3.5 (max hardware stage) | front width 151.6 |
//    brow depth 19.4 (max 19.75) + smooth stepped-top profile (IMG_1233) in BOTH the
//    clamp underside and the glasses stand-in.
//  FROM VENDOR DRAWING (accepted for print): ELP AR0234 world boards 38x38, 28 pitch.
//  MEASURED 2026-07-19 (InnoMaker OV9281 in hand, IMG_1446-1450): board 32x32 (OV_BOARD),
//    28 mm mount pitch confirmed to line up (OV_PITCH), JST port on the back at CONN_POS with
//    a pass-through slot cut in the plate. ELP world board still from vendor drawing until
//    that unit is in hand.
// -------------------------------------------------------------------------------

// ===========================================================================
//  SIM <-> CAD.  software/rig.py is the source of truth. sim2cad maps a rig position to CAD.
//  OPTIC_DROP = how far the optic centre sits below the brow rail (MEASURE on your unit).
//  rig.py positions (sim mm, IPD 67):  world ±[33.5,44,22]  eye-corner ±[69.5,-5,-6]
//                                      stereo eye2 ±[69.5,-16,-10]   NIR pupil [25.5,-33,6]
//  (positions are WEARABLE: the real 36-38 mm boards clear the cone, the eyeball, AND the face,
//   so the glasses still rest normally — verified by software/wearable.py + cad_fit.py.)
// ===========================================================================
OPTIC_DROP = 25.3;       // MEASURED: brow rail top -> lens/optic center (was 17 nominal; deeper on this unit)
function sim2cad(s) = [s[0], s[2], s[1] - OPTIC_DROP];
function unit(v) = v / norm(v);

WORLD_SIM = [ipd/2,    49, 22];      // brow, looks forward. RAISED 44->49 (2026-07-23) so the lower
                                     // mounting screws clear the rail/boom behind them. Earlier
                                     // 30->44 (2026-07-02): at 30 the
                                     // 38 mm board's lower standoffs + PCB edge sat INSIDE the
                                     // glasses brow band (z<-2.6) — caught by cad_overlap.py; at 44
                                     // the whole module clears ABOVE the brow top (+2.3 mm) and the
                                     // cone margin grows to ~+20. Matches rig.py WC_UP. FWD 5->12
                                     // (2026-07-02): the corrected holder stack (PCB@-10.8 + 4mm
                                     // standoffs + plate) reached the forehead line at fwd=5.
                                     // FWD 12->22 (2026-07-04): the boom riser passed THROUGH the
                                     // board's back-component zone at 12 (split-check, 847mm^3);
                                     // at 22 the whole board plane sits in front of the riser.
EYE_SIM   = [ipd/2+36, -5, -6];      // TEMPLE; lowered -2 -> -5 (2026-07-16) on a longer side boom
EYE2_SIM  = [ipd/2+36, -16, -10];    // stereo pair, below+behind the primary so the boards also clear
PUPIL_SIM = [ipd/2-8,  -40,  6];     // under-eye; lowered -33 -> -40 (2026-07-16) on a longer centre mast
// AIM TARGETS must equal rig.py's exactly so the printed holder points where the sim camera
// points (recompute if EYE_BEHIND / PANTO_DEG / anatomy means change):
//   eye cams  -> rig.nominal_outer_canthus()[1] (anatomy means through R0/T0). The old value
//                [44.5,3.82,-13.39] predated the measured EYE_BEHIND 24->28.5 = the printed
//                aim was 8.8 DEG OFF the sim camera (caught 2026-07-02).
//   pupil cams-> OPTIC +T0 = the DISPLAY-IPD eye centre (68.13/2), not the user-pupil x.
CANTH_SIM = [44.5, 3.160, -17.673];  // = rig.nominal_outer_canthus()[1], EYE_BEHIND 28.5, panto -3.5
                                     //   (panto fixed -7 -> -3.5 in the 2026-07-06 spec audit: the
                                     //    One Pro's 3-stage tilt tops out at ±3.5°)
disp_ipd  = 68.13;                   // rig.py DISPLAY_IPD (pupil cams aim at OPTIC + T0)
COR_SIM   = [disp_ipd/2, 0, -28.5];  // right eye nominal CoR (pupil-cam aim; = OPTIC_R + T0)

// ---------- real camera MODULES (vendor dims; verify with calipers) ----
WORLD_BOARD = 38;  WORLD_PITCH = 28;   // ELP AR0234 USB: 38x38 board, ~28 mm mount pattern
OV_BOARD    = 32;  OV_PITCH    = 28;   // InnoMaker OV9281 USB, MEASURED 2026-07-19 (IMG_1446-1450):
                                       //  32x32 board; the 28 mm standoff pattern lines up with the
                                       //  real mount holes (confirmed on the received unit).
// ---- JST connector on the OV9281 BACK face (the face toward the plate) ----------------------
//  The board's USB cable plugs into a JST port that STANDS ~5 mm proud of the PCB back, taller
//  than the plate-to-PCB gap, so it hits the plate. Fix: a SLOT through the plate under the port
//  lets the connector + cable pass straight out the back (chosen over taller standoffs, which
//  would lengthen the posts and lose rigidity). Board-local frame: origin at board centre, +x =
//  the side that appears LEFT when the lens points AWAY from you (the face Dylan measured), +y up.
//  MEASURED: port far/inner corner 14.2 mm from the left edge, 30.35 mm from the bottom, board 32.
//  ⚠️ VERIFY IN F5 (show_cams=true): the ivory connector block must sit over your real port; if it
//  is mirrored to the wrong side, flip the sign of CONN_POS[0] (one edit).
CONN_POS  = [5.95, 12.75];             // board-local centre of the JST port
CONN_SIZE = [8.3, 3.2];                // the ACTUAL port (drawn in preview to verify the fit)
CONN_SLOT = [9.5, 6.5];                // the plate CUT, oversized; thin side 4.5 -> 5.5 -> 6.5
                                       //  (2026-07-23) for more cable/connector room
CONN_SLOT_CTR = [5.5, 12.75];          // slot nudged 0.45 mm -x of the port so its far edge keeps
                                       //  a ~1.25 mm wall to the standoff at (14,14)
// M12 board-lens optical model (matches software/cad_fit.py): the projection centre is BACK mm
// in FRONT of the sensor; the lens holder (radius RLENS) runs from the sensor to FRONT past it.
BACK = 10;  FRONT = 7;  RLENS = 9;

// ---------- holder / print params ----
plate_t = 2.6;  standoff_h = 4;  m2_d = 2.5;  m2_boss = 5.5;  boom_w = 9;   // bulked 8->9 (booms carry cams)
// m2_d WIDENED 1.8 -> 2.3 -> 2.5 (2026-07-23): at 1.8 the printed holes came out nearly CLOSED (a small
// vertical hole shrinks/bridges shut in FDM). 2.3 prints open ~2.0 so an M2 self-tapping screw
// threads straight in and bites the walls; boss widened 5 -> 5.5 to keep a ~1.5 mm wall.
cf_rod_d = 1.5;        // glued-in carbon rod through each boom leg (0 = none) — stiffness
// L-ROUTED booms (2026-07-02): the old straight diagonals cut through the brow
// clamps, the glasses brow, and each other. Each boom now rises from the rail top, runs OUT
// horizontally at BOOM_ELEV (above the brow, clamps, hinge bump, IMU and wire channel), then
// drops STRAIGHT DOWN to the camera, at a forward standoff >= Y_CLEAR when the drop goes below
// the brow top (so the vertical leg falls in open air in front of/next to the glasses).
BOOM_ELEV = 11.5;      // z of the horizontal transit legs (leg bottom = +7: clears the taller
                       //  rail top at +5.5 by 1.5 and the hinge bump at +3.6 by 3.4)
Y_CLEAR   = 7;         // min forward y for any drop leg that goes below the brow top (z -2.6)

// ---------- carrier rail + brow clip ----------------------------------
//  The rail sits at the brow, co-located with the glasses' own opaque brow frame (above the lens
//  aperture), so it only grazes the very top edge of the geometric cone. Kept thin.
clip_w = 20;  wall = 2.6;  screw_d = 2.8;   // clamp width 20; M3 self-threads into the 2.8 hole
// BROW RAIL widened + RAISED (2026-07-02): the old thin rail (t=4, z -4..0) sat 1.4 mm INSIDE the
// glasses brow band (brow top is z=-2.6). The rail is now a 10 mm-deep platform ON TOP of the brow
// (z -0.5..3.5, overlapping the clamp ceilings 0.5 mm so they fuse), with the wire channel in it.
// Rail sized TO THE BROW (2026-07-02): depth = the measured brow-centre thickness
// (12.14 -> 12), and it runs the brow's full span, stopping just short of the hinge bumps at
// x ~±64.8 (front frame is 151.6 wide; the hinge tops poke ABOVE the brow line so the bar
// cannot pass over them).
rail_half = 64;  rail_t = 12;  rail_h = 6;   // h 4->6 (2026-07-04: structural integrity)
// ---- WIRE CHANNEL: open groove along the rail top's REAR half + retaining bridge tabs ----
//  REALITY CHECK (2026-07-02): stock USB module cables are ~3.5-4 mm thick — SIX of them cannot
//  hide in any groove this rail can carry. Routing plan: THIN leads (IMU, IR strobe, re-terminated
//  30 AWG) live IN the groove under the tabs; the FAT stock USB bundle LIES ON TOP of the rail's
//  rear edge, lashed down by zip ties through the through-slots below (thin 2.5 mm ties pass
//  under the rail — there's a 2.1 mm gap above the brow top). Boom legs cross 2.5 mm above the
//  groove so nothing pinches at the crossings.
wire_ch_y = -3;   wire_ch_w = 3;   wire_ch_d = 2.5;    // groove: y -4.5..-1.5, z 1.0..3.5
wire_slot_xs = [-25, -8, 8, 25];   // zip-tie through-slots (clear of risers ±13/±43.5, clamps ±28..48,
                                   // tabs ±18/±35, and the IMU pedestal's front-half base)
wire_tab_t = 1.2; wire_tab_w = 4;  wire_tab_xs = [-35, -18, 18, 35];  // bridge tabs retaining the
                                   // wires (none at x=0 — the IMU tower pedestal roots there)
// ---- REMOVABLE PADDED brow clamp — rebuilt to the MEASURED side profile (IMG_1233, 2026-07-06) ----
//  Cross-section at the grip spot: front-to-back 19.4 mm (LARGEST ~19.75 — the slip-on must clear
//  this); the TOP STEPS DOWN toward the face: 8.9 tall over the front ~5 mm -> 8.2 -> 6.6 -> 3.6
//  at the back edge (heights above the brow base line). The old jaws assumed a 15 mm brow — too
//  narrow to seat. Grip = padded FRONT jaw on the tall 8.9 face + padded SHALLOW rear jaw on the
//  3.6 back ledge + the flat ceiling bearing on the tall front section (the top falls away
//  rearward, so a flat ceiling naturally bears only there). Extra front clearance for easy
//  slip-on; the M3 screw (self-threading, one per clamp) closes the gap.
brow_depth_max = 19.75;   // largest measured front-to-back depth
brow_front_y   = 6;       // brow front-face plane (CAD y, unchanged reference)
brow_front_h   = 8.9;     // tall front step (the front jaw grips this face)
brow_back_h    = 3.6;     // short back ledge (the rear jaw grips this face)
// FIT TIGHTENED (2026-07-19): the resting fit should already be near-aligned so the screw
// only closes the last fraction. Pad allowance cut 1.5 -> 0.5 (line the jaws with THIN
// PTFE/felt tape, not thick silicone: squish was slop), clearances sized for ~2.5 mm total
// bare-plastic play (easy slip-on, inside the 1-3 mm safety band), ~1.5 mm with tape on.
clamp_pad_t = 0.5;        // thin low-friction tape lining each jaw (non-marking)
slip_front  = -0.2;       // TIGHTENED 1.5 mm total (2026-07-23, was 1.2): the draft clamp was
slip_rear   =  0.0;        //  loose. Front now slightly interferes (screw + pad take it up), rear
                          //  face rides right on the brow back. Total slot 1.5 mm smaller.

// ---------- NIR illumination (940 nm rings around BOTH eyes — binocular) ----
ir_n = 6;  ir_ring_r = 13;  ir_led_d = 3.2;  ir_led_h = 3.0;   // 940 nm LEDs (EYE_TRACKING.md §3)
// ---- IR BLOCKING / KEEP-OUT (safety): each LED sits recessed in a baffle so the EMITTER never
//      gets close to the eye, and a hard standoff rim keeps the assembly >= ir_min_standoff mm
//      from the cornea (the retina can't feel IR, so this mechanical block backs up the
//      electrical current/strobe limits — see EYE_TRACKING.md §3 / WIRING.md).
ir_baffle_wall = 1.2;   // shroud wall around each LED
ir_recess      = 1.2;   // LED tip set this far BEHIND the baffle's eye-facing rim (no exposed near emitter)
ir_min_standoff = 10;   // mm minimum LED-rim-to-eye clearance the geometry must preserve
// cornea (CAD): sim2cad of the cornea apex ~[ipd/2, 0, -12]; the IR ring plane sits at cad y=0,
// so the rim-to-cornea clearance is ~12 mm (>= ir_min_standoff). The baffle does NOT extend
// toward the eye, so it cannot reduce that clearance.
ir_cornea_clear = 12;
assert(ir_cornea_clear >= ir_min_standoff,
       "IR ring too close to the eye — increase standoff (see EYE_TRACKING.md §3)");

// ---------- IMU (ICM-20948 9-DoF) — rigid pocket ON THE CARRIER ----
//  The carrier clamps onto the frame, so an IMU pocket here is rigidly coupled to the glasses
//  (this is why the whole build needs only THIS one printed part — the IMU rides the carrier, not a
//  separate bridge). Board FLAT in the x-y plane, X -> +x (right), Y -> +y (forward), so its tilt
//  maps to pose dev[0] (pantoscopic) / dev[2] (roll) exactly as software/imu.py models — the input
//  the Kalman drift filter consumes.
imu_w = 21;  imu_h = 18;  imu_t = 2.0;  imu_wall = 1.6;  imu_screw_d = 2.2;

// ---------- face-contact padding provision ----
//  The carrier mounts to the GLASSES brow (the XREAL One Pro's OWN nose pads carry the face), so no
//  printed surface normally touches skin. brow_pad_t recesses any brow-rest contact for a soft
//  silicone/foam pad anyway — "pad all printed face-contact surfaces" (EYE_TRACKING.md §2).
brow_pad_t = 1.6;

// ---------- glasses stand-in (visual only) ----
g_front_w = 151.6;  g_visor_h = 34;  g_visor_th = 4;  g_wrap_arc = 60;   // One Pro front ~151.6 mm wide
g_brow_h = 8;  g_nose_w = 32;  g_nose_h = 20;  g_temple_len = 150;

// =====================================================================
//  PRIMITIVES
// =====================================================================
module rbox(w, d, h, r) {
    hull() for (sx=[-1,1]) for (sy=[-1,1]) for (sz=[-1,1])
        translate([sx*(w/2-r), sy*(d/2-r), sz*(h/2-r)]) sphere(r, $fn = 16);  // light rounding, fast
}
module rrect(w, h, r) { offset(r) offset(-r) square([w, h], center = true); }
module curved_visor(R, arc, th, h, r) {
    translate([0, -R, 0]) rotate([0, 0, 90 - arc/2])
        rotate_extrude(angle = arc, $fn = 160) translate([R, 0]) rrect(th, h, r);
}
module capsule(p0, p1, d) { hull() { translate(p0) sphere(d/2); translate(p1) sphere(d/2); } }
module aim_z_at(from, to) {                 // aim children's +z from `from` toward `to`
    v = [to[0]-from[0], to[1]-from[1], to[2]-from[2]]; L = norm(v);
    rotate([-asin(v[1]/L), atan2(v[0], v[2]), 0]) children();
}

// =====================================================================
//  REAL BOARD-CAMERA HOLDER
//  Local frame: +z = optical axis (lens looks toward the target). The PROJECTION CENTRE is at
//  the local origin (so it lands on the rig.py position). The PCB sits BACK mm behind it; the
//  M12 lens runs sensor->FRONT. The printed holder is a backing plate + 4 standoffs at `pitch`
//  with M2 bores (heat-set inserts) the PCB screws onto, a relief/cable cut, and a cable notch.
// =====================================================================
module camera_module(board, ov_conn = false) {   // the part you BUY (visual stand-in only)
    color([0.10, 0.45, 0.20]) translate([0, 0, -BACK]) rbox(board, board, 1.6, 1.5);   // PCB
    color([0.16, 0.16, 0.16]) translate([0, 0, -BACK + 0.8])
        cylinder(d = 2*RLENS, h = BACK + FRONT - 0.8);                                  // M12 lens + holder
    if (ov_conn)                                                                        // JST port on the back
        color("ivory") translate([CONN_POS[0], CONN_POS[1], -BACK - 3.3])
            cube([CONN_SIZE[0], CONN_SIZE[1], 5], center = true);
}
// MINIMAL camera holder (Iter 1): no board-sized rim/tray. Just a thin backing PLATE sized to the
// standoff pattern (the PCB cantilevers slightly past it, which is fine), 4 M2 standoffs the PCB
// screws onto, and a LENS SHROUD tube from behind the PCB to a front collar — the shroud connects
// the plate to the collar AND baffles stray light. The slim front means adjacent holders no longer
// collide. Local +z = optical axis; projection centre at the origin.
// NO lens shroud (2026-07-02): the old shroud tube started 3.2 mm in FRONT of the backing plate
// (the PCB seats between them) so it was a FLOATING solid connected to NOTHING — every camera
// holder printed a loose tube (caught by the STL shell count; it imported into OnShape as
// separate bodies). It was only a stray-light baffle: baffle instead with a slip-on collar or
// matte tape around the M12 barrel after assembly (matters for the NIR eye cams).
// MOUNT GEOMETRY (fixed 2026-07-02): the PCB's BACK face sits at z = -(BACK+0.8) (sensor ~BACK mm
// behind the projection centre at the local origin — that's what lands the lens on the rig.py
// position). The old standoffs topped at -6 = 4.8 mm IN FRONT of that plane, so the boards could
// not screw on at the right depth (the lens would sit ~4 mm forward of the validated geometry).
// Standoffs now run plate -> exactly the PCB-back plane, standoff_h tall for back-side components.
// HOLDING THE CAM (2026-07-03): primary = 4 self-tapping M2 screws into the
// standoffs; backup/positive retention = 2 ZIP TIES over the PCB through the 4 through-slots
// (works even if the real board's mount pitch differs from the assumed 28 mm — the ties clamp
// the board onto the standoff tops regardless). Slots are pure material removal, so every
// verified cone/face/glasses clearance is untouched. The BOSS on the plate's back face is the
// boom's attachment pad — the boom's end sphere seats inside it and never crosses the plate's
// front face (the old attachment bulged 2.8 mm into the PCB back-component zone).
module camera_holder(board, pitch, boss_off = [0, 0], ov_conn = false, wcable = false, boss_depth = 0) {
    pcb_back = -BACK - 0.8;                      // where the PCB's back face must land
    zt = pcb_back - standoff_h - 1.3;            // backing-plate CENTRE (2.6 thick, top at pcb_back - standoff_h)
    bp = pitch + m2_boss + 4;                    // plate just covers the standoff pattern (~37 mm)
    difference() {
        union() {
            // rounding MUST be < half the thickness: rbox hulls corner spheres, so r=2 on a
            // 2.6 plate silently made it 5.4 thick (bulging 1.4 into the PCB clearance AND
            // 1.4 backward — found via the eye wing's vanished gap, 2026-07-04)
            translate([0, 0, zt]) rbox(bp, bp, 2.6, 1.2);                     // thin backing plate
            for (sx=[-1,1]) for (sy=[-1,1])                                    // 4 standoffs: plate -> PCB back
                translate([sx*pitch/2, sy*pitch/2, zt]) cylinder(d = m2_boss, h = standoff_h + 1.3);
            translate([boss_off[0], boss_off[1], zt - 1.3 - 2.5 - boss_depth]) // boom-attachment boss;
                cylinder(d = 11, h = 2.7 + boss_depth);                        //  boss_depth extends it
                                                                               //  BACKWARD so the boom can
                                                                               //  attach fully behind the
                                                                               //  plate (never pokes front)
        }
        for (sx=[-1,1]) for (sy=[-1,1])                                       // M2 bores (self-tapping;
            translate([sx*pitch/2, sy*pitch/2, zt - 2]) cylinder(d = m2_d, h = standoff_h + 5);  // start below the plate
        csx = (CONN_POS[0] >= 0) ? 1 : -1;                                    // connector's x-side
        for (sx=[-1,1]) for (sy=[-1,1])                                       // zip-tie through-slots
            if (!(ov_conn && sy > 0 && sx == csx))                            // skip the one the JST slot crowds
                translate([sx*(pitch/2 - 1), sy*(bp/2 - 3.2), zt])            //  (standoff blocks moving it; 3
                    cube([3.2, 2, 6], center = true);                         //   ties still back up the 4 screws)
        if (ov_conn)                                                         // JST port pass-through slot
            translate([CONN_SLOT_CTR[0], CONN_SLOT_CTR[1], zt])              //  (connector + cable exit the back)
                cube([CONN_SLOT[0], CONN_SLOT[1], 8], center = true);
        else if (wcable)                                                     // WORLD cable: CENTERED back slot
            translate([0, -9, zt])                                           //  (just below the central boss,
                cube([CONN_SLOT[0], CONN_SLOT[1], 8], center = true);        //   clear of all 4 screw holes)
        else
            translate([0, -bp/2, zt]) cube([7, 6, 4], center = true);        // edge cable notch (fallback)
    }
    // (strain-relief posts REMOVED 2026-07-23 per Dylan: the small holeless pegs printed as fragile
    //  loose-looking pins with no function; zip-tie the cable to a rail tab instead.)
}
module board_cam(board, pitch, boss_off = [0, 0], ov_conn = false, wcable = false, boss_depth = 0) {
    if (show_cams) color([0.30, 0.30, 0.32]) camera_module(board, ov_conn);
    color("orange") camera_holder(board, pitch, boss_off, ov_conn, wcable, boss_depth);
}
// PCB KEEP-OUT solid (checks only, never printed): the physical board + its back-side
// component zone — plate front face (-BACK-0.8-standoff_h) to PCB front face (-BACK+0.8).
// `u_keep` trims the modeled zone in local x: [u0, u1]. The pupil cams exclude the board's
// NASAL 5 mm edge strip (outside the 28 mm mount holes = the screw/washer margin, kept bare
// by board-design practice) so the nose-bridge drop leg may pass it. ⚠️ VERIFY on the real
// OV9281 that this edge strip is component-free before the rigid print.
module pcb_zone(board, u_keep = [-99, 99]) {
    zlo = -BACK - 0.8 - standoff_h;
    zhi = -BACK + 0.8;
    u0 = max(-board/2, u_keep[0]);  u1 = min(board/2, u_keep[1]);
    translate([(u0 + u1)/2, 0, (zlo + zhi)/2]) cube([u1 - u0, board, zhi - zlo], center = true);
}
// Place a board cam: optical centre at C, lens aimed at `target`, held by an L-ROUTED boom
// (2026-07-02: booms extend directly out, then drop straight down). Route:
//   riser: straight UP from the rail top (anchor, front half — clear of the wire groove)
//   OUT leg: horizontal at zt = max(BOOM_ELEV, case-back z) to above/below the case back —
//            passes OVER the brow, clamps, hinge bump, IMU shelf and wire channel
//   DOWN leg: straight vertical drop to case-back height, held at y >= Y_CLEAR when the drop
//             goes below the brow top so it falls in open air, not through the glasses
//   in-jog: short horizontal run onto the holder's back
// Straight legs each carry a CF-rod channel; elbows get reinforcing spheres. Sub-pixel tracking
// still needs camera-to-frame motion < 0.2 mm (data/flex.log, sim) — the L-boom is less stiff
// than a straight diagonal, so keep legs thick + rods glued in; the bonded carrier stays the
// final answer for rigidity.
// `drop_x` (optional) overrides the drop leg's lateral position when a straight drop above the
// case-back would fall inside the see-through cone (the pupil cams need this — see pupil_cam).
// BOOM END RETREATED (2026-07-03 — the ball ends interfered with the cam holders): the
// old attachment point sat MID-plate, so the end sphere (r=boom_w/2) bulged
// 2.8 mm through the plate's FRONT face into the PCB back-component zone. The end now seats
// inside the boss on the plate's BACK face (att = 2.5 mm behind the plate's back surface;
// the r4 end sphere stays >=0.9 mm short of the front face). Elbow-reinforcement spheres
// deleted (the p2 sphere was the other protruding "ball"); the shared capsule end-spheres
// carry the joint. `render` splits the module for the overlap harness:
//   "all" (print) | "boom" | "holder" | "pcbzone" (the physical board keep-out, checks only)
// `att_off` = [u, v] boss offset in the HOLDER's plate plane (same frame aim_z_at builds):
// lets the boom meet the plate at the edge nearest its drop corridor instead of dead centre,
// so the jog never crosses the physical board's keep-out zone.
// `elev` overrides the horizontal transit height (default BOOM_ELEV) so a boom's arms can be
// lowered without moving the camera. `elbow_fill` rounds the boom's corners (the turn-downs).
module cam_at(anchor, C, target, board, pitch, drop_x, render = "all", att_off = [0, 0],
              zone_u_keep = [-99, 99], foot_wing_len = 0, ov_conn = false, boss_depth = 0,
              elev = undef, elbow_fill = false, wcable = false, elbow_top = true) {
    ax = unit([target[0]-C[0], target[1]-C[1], target[2]-C[2]]);
    ra = -asin(ax[1]);  rb = atan2(ax[0], ax[2]);    // aim_z_at's rotation angles
    e1 = [cos(rb), 0, -sin(rb)];                     // holder-local +x in world
    e2 = [sin(rb)*sin(ra), cos(ra), cos(rb)*sin(ra)];// holder-local +y in world
    back = -(BACK + 0.8 + standoff_h + 2.6) - 2.5 - boss_depth;  // behind the plate back, in the boss;
                                                                //  boss_depth pushes the boom fully behind
    att = [C[0] + back*ax[0] + att_off[0]*e1[0] + att_off[1]*e2[0],
           C[1] + back*ax[1] + att_off[0]*e1[1] + att_off[1]*e2[1],
           C[2] + back*ax[2] + att_off[0]*e1[2] + att_off[1]*e2[2]];      // boom attachment point
    elv = (elev == undef) ? BOOM_ELEV : elev;
    zt = max(elv, att[2] + 2);                                            // transit height (lowerable via elev)
    dy = (att[2] < -2.6) ? max(att[1], Y_CLEAR) : att[1];                 // forward standoff for low drops
    dx = (drop_x == undef) ? att[0] : drop_x;                            // lateral drop position
    top = [anchor[0], anchor[1], zt];
    p1  = [dx, dy, zt];                                                   // out-leg end / drop start
    p2  = [dx, dy, att[2]];                                               // drop-leg end
    if (render == "all" || render == "boom") color("orange") {
        boom_foot(anchor[0]);                                             // flat foot inside the rail
        if (foot_wing_len > 0) foot_wing(anchor[0], foot_wing_len);       // inboard root extension
        capsule(anchor, top, boom_w);                                     // riser (rooted in the foot)
        capsule(top, p1, boom_w);                                         // OUT leg
        capsule(p1, p2, boom_w);                                          // DOWN leg
        capsule(p2, att, boom_w);                                         // in-jog into the boss
        if (elbow_fill) {                                                 // ROUND the turn-down corners
            if (elbow_top) translate(top) sphere(boom_w/2 + 1);           //  (rail-side elbow — SKIP for
            translate(p1)  sphere(boom_w/2 + 1);                          //   world: its riser is right
            translate(p2)  sphere(boom_w/2 + 1);                          //   behind the plate) + turn-down
        }                                                                //   elbow + base of the down-leg
        // The boom-to-boss gap is closed by the EXTENDED BOSS (boss_depth in camera_holder)
        // reaching back to att, so there is NO forward-reaching fill ball to poke the plate front.
        if (cf_rod_d > 0) {                                               // CF rods: the two long legs
            capsule(top, p1, cf_rod_d);
            capsule(p1, p2, cf_rod_d);
        }
    }
    if (render == "all" || render == "holder")
        translate(C) aim_z_at(C, target) board_cam(board, pitch, att_off, ov_conn, wcable, boss_depth);
    if (render == "pcbzone")
        translate(C) aim_z_at(C, target) pcb_zone(board, zone_u_keep);
}

// =====================================================================
//  THE CAMERAS — 6-cam CORE (2 world + 2 eye-corner + 2 NIR pupil) + 2 stereo (8-cam FULL)
//  (optical centres at sim2cad(rig.py); see software/cad_fit.py for the fit)
// =====================================================================
// BOOM FEET (2026-07-04 — ball ends removed): the risers' capsule end-spheres
// used to poke below the rail bottom and out its front face — anything proud of the rail's
// surfaces offsets how the carrier seats on the glasses. Each riser now stands on a FLAT
// rectangular FOOT contained entirely inside the rail envelope (y 2.25 +-3.5: clear of the
// wire groove at -1.5 and flush inside the front face at +6; bottom at -0.3, above the rail
// bottom), and the round boom starts at z=8 with its end-sphere buried in the foot.
ANCHOR_Y = 2.5;  ANCHOR_Z = 8;
FOOT_Y = 2.25;  FOOT_W_EXTRA = 2;   // foot width = boom_w + 2
module boom_foot(x) {
    translate([x, FOOT_Y, 4.35]) rbox(boom_w + FOOT_W_EXTRA, 7, 9.3, 1.5);
}
// FOOT WING (2026-07-04): the SIDE (eye) booms are the longest cantilevers — extend
// their connection surface INWARD along the rail top for a much larger fused root. The wing's
// section is slimmer than the foot so it threads the mid-rail obstacles: y <= 4.1 clears the
// world cam's plate (y >= 4.6), z >= 0.5 clears the clamp tops (z <= 0), rear edge -1.25
// clears the wire groove (-1.5).
module foot_wing(x, len) {
    sgn = (x >= 0) ? 1 : -1;
    translate([x - sgn*len/2, 1.425, 4.75]) rbox(len + boom_w + FOOT_W_EXTRA, 5.35, 8.5, 1.5);
}
module world_cam(side, render = "all") {
    C = sim2cad([side*WORLD_SIM[0], WORLD_SIM[1], WORLD_SIM[2]]);
    // STRAIGHT-UP riser (2026-07-23, per Dylan): anchor directly under the plate centre (x = C_x)
    // so the boom rises straight up to the middle instead of up-then-sideways. This puts the foot
    // inside the brow-clamp span (28..48) so it FUSES with the clamp top (extra rigidity; the
    // clamp's jaw slot is below, unaffected). boss_depth=3.5 pushes the boom attachment fully
    // behind the plate; wcable = centered back cable slot.
    cam_at([side*33.5, ANCHOR_Y, ANCHOR_Z], C, [C[0], C[1]+100, C[2]], WORLD_BOARD, WORLD_PITCH,
           render = render, wcable = true, boss_depth = 3.5); // looks +y; no elbow fill (2026-07-23)
    // BROW-BAR extends FORWARD + up to the front boom's holder (2026-07-23, per Dylan): the world
    // plate's lower edge hangs to ~z5.2 y4.6-7.2, overhanging the rail's front-top. This local
    // buttress (only at the world x) grows the rail forward to meet and support it.
    if (render == "all" || render == "holder")
        color("dimgray") translate([side*33.5, 0, 0]) hull() {
            translate([0, 2.5, rail_h/2 - 0.5]) cube([boom_w + 6, 7, rail_h], center = true);  // rail root
            translate([0, 6.0, 5]) cube([boom_w + 6, 3, 7], center = true);                    // fwd+up to plate
        }
}
module eye_cam(side, render = "all") {
    C = sim2cad([side*EYE_SIM[0], EYE_SIM[1], EYE_SIM[2]]);
    // anchor at x=57: outboard of the world riser (43.5) AND of the world holder's deeper
    // standoffs (they reach x=50 since the 2026-07-02 holder fix; at x=52 the riser hit one)
    // boss offset [0,10]: toward the plate's top edge, where the drop leg arrives — the
    // centred boss forced the jog through the board's keep-out (4.2 mm^3, split-check)
    cam_at([side*58, ANCHOR_Y, ANCHOR_Z], C,
           sim2cad([side*CANTH_SIM[0], CANTH_SIM[1], CANTH_SIM[2]]), OV_BOARD, OV_PITCH,
           render = render, att_off = [0, -10],    // 58: foot clears the world plate (52) and the
           foot_wing_len = 12, ov_conn = true,     // wing SHORTENED 24->12 (2026-07-23) so it clears
           elev = BOOM_ELEV - 2);                  // the world foot; side boom LOWERED 2 mm, no elbow fill
                                                   // (down to x~28.5, 1.5 clear of the world foot)
}
module eye_cam2(side) {
    // FUTURE 8-cam FULL upgrade only (build_stereo). NOTE (2026-07-02): its drop leg lands ~2 mm
    // from the primary eye cam's drop leg AND cad_fit shows the stereo/primary boards overlapping
    // -18.8 mm — the stereo layout needs its own deconfliction pass before building FULL.
    C = sim2cad([side*EYE2_SIM[0], EYE2_SIM[1], EYE2_SIM[2]]);
    cam_at([side*(ipd/2+13), ANCHOR_Y, ANCHOR_Z], C,
           sim2cad([side*CANTH_SIM[0], CANTH_SIM[1], CANTH_SIM[2]]), OV_BOARD, OV_PITCH);
}
module pupil_cam(side, render = "all") {    // NIR eye-tracking cam — ONE PER EYE (binocular)
    C = sim2cad([side*PUPIL_SIM[0], PUPIL_SIM[1], PUPIL_SIM[2]]);
    // DROP AT THE NOSE BRIDGE (drop_x=±6, 2026-07-02): a straight drop above the case-back
    // (x≈23.5) falls INSIDE the see-through cone — a visible post in the wearer's forward view
    // (603 mm^3 vs the elliptical cone, caught by cad_overlap.py; the old diagonal boom had the
    // same flaw unchecked). At the drop plane (y=14.6, 43 mm in front of the eye's CoR) the
    // right-eye cone starts at |x|=13.4, and the LEFT eye's cone at x=-13.4 — so the only clear
    // vertical corridor is the nose-bridge gap between them. The leg drops there and jogs out to
    // the holder back below the cone. Left/right legs sit 12 mm apart at the midline.
    // riser at x=±13 (not over the holder): at ±25.5 the riser + elbow passed through the WORLD
    // cam's lens shroud (56.7 mm^3) and grazed its lower standoff — x<=±18 clears both
    // boss offset toward the NASAL plate edge (local +u ~ world +x on both sides, so the
    // offset flips with side): the centred boss put the 16 mm jog straight through the
    // board's keep-out zone (119 mm^3, split-check)
    // UNITED CENTRAL MAST (2026-07-04): both pupil booms share ONE drop column at
    // x=0 (drop_x=0 both sides -> coincident legs union into a single column, dead centre of
    // the nose-bridge corridor, 13.4 mm from either eye's FOV cone and clear of both boards'
    // nasal edges — the strict FULL-board keep-out passes again, no edge-strip exemption).
    // Risers at x=±7 fuse INTO the IMU tower pedestal: pedestal + risers + column = one mast.
    cam_at([side*7, ANCHOR_Y, ANCHOR_Z], C, sim2cad([side*COR_SIM[0], COR_SIM[1], COR_SIM[2]]),
           OV_BOARD, OV_PITCH, drop_x = 0, render = render, att_off = [-13*side, 0], ov_conn = true,
           elev = BOOM_ELEV - 1, elbow_fill = true);  // bottom boom LOWERED 1 mm + corners filled (2026-07-23)
}
// One IR LED inside a SAFETY BAFFLE: the shroud + LED extend AWAY from the eye (+y), the LED tip
// is RECESSED behind the eye-facing cap, and the cap has only a small beam aperture. So no emitter
// sits close to the eye, close-range/off-axis direct paths are blocked, and the eye-facing rim
// (at the ring plane) preserves the >= ir_min_standoff clearance (the bare LED used to poke toward
// the eye and cut into it). 940 nm, low current, strobed — the mechanical block backs the electrical limits.
module ir_led_baffle(pos) {
    translate(pos) rotate([-90, 0, 0]) {                 // local +z now points +y, AWAY from the eye
        difference() {
            cylinder(h = ir_led_h + ir_recess + ir_baffle_wall, d = ir_led_d + 2*ir_baffle_wall); // shroud
            translate([0, 0, ir_recess]) cylinder(h = ir_led_h + 1, d = ir_led_d);                // recessed LED pocket
            translate([0, 0, -0.1]) cylinder(h = ir_recess + 0.2, d = ir_led_d - 0.8);            // beam aperture (blocks stray)
        }
    }
}
module ir_leds(side) {                       // 940 nm ring around one eye's aperture (baffled)
    cen = sim2cad([side*ipd/2, 0, 0]);
    for (i = [0:ir_n-1]) {
        a = i * 360 / ir_n;
        ir_led_baffle([cen[0] + ir_ring_r*cos(a), cen[1], cen[2] + ir_ring_r*sin(a)]);
    }
}

// =====================================================================
//  RAIL + BROW HOOKS + BOND PADS
// =====================================================================
// Rail platform ON TOP of the brow (z -0.5..3.5), with the open wire groove cut into its top-rear
// and thin bridge tabs over the groove so the cables can't lift out. Booms rise from its front half.
module rail() {
    difference() {
        translate([0, 0, rail_h/2 - 0.5]) rbox(2*rail_half, rail_t, rail_h, 1.5);
        translate([0, wire_ch_y, rail_h - 0.5]) // groove: full length minus the ends, opens up
            cube([2*rail_half - 8, wire_ch_w, 2*wire_ch_d], center = true);
        for (x = wire_slot_xs)                  // zip-tie through-slots (for the fat USB bundle)
            translate([x, wire_ch_y, rail_h/2 - 0.5]) cube([4, 2, rail_h + 2], center = true);
    }
    for (x = wire_tab_xs)                       // retaining tabs bridging the groove — DIP INTO the
        translate([x, wire_ch_y, rail_h - 0.5])  // rail (2026-07-23): centred on the rail top so the
            cube([wire_tab_w, wire_ch_w + 3, wire_tab_t + 1.4], center = true);  // tab OVERLAPS the rail
                                                //  at its y-ends (was a coincident face that floated off)
}
// Removable PADDED brow clamp: a C that drops onto the brow rail; jaw opening = brow + silicone
// pads + clearance; a thumbscrew through the front jaw tightens it. No adhesive, no marks.
// Jaw depths: FRONT stops at -11.5 (full grip of the 8.9 face, still under the vertical-FOV
// line ~-12.8, cone-checked); REAR at -11.0 (the 3.6 back ledge's face spans -7.9..-11.5 —
// a deeper rear jaw would hang past the ledge toward the forehead for nothing).
front_jaw_z = 11.5;  rear_jaw_z = 16.5;   // rear DOWN 5 mm (2026-07-23, 11.5 -> 16.5): the back
                                          // clamp piece reaches further down behind the brow for a
                                          // more secure hook. Was 11.0 -> 11.5 (2026-07-19): covers the
                                          // back ledge's FULL face (-7.9..-11.5) for aligned grip
// Catmull-Rom spline sampler (vector form; endpoints doubled so the curve hits them)
function _cr(p0, p1, p2, p3, t) = 0.5 * ((2*p1) + (-p0 + p2)*t
    + (2*p0 - 5*p1 + 4*p2 - p3)*t*t + (-p0 + 3*p1 - 3*p2 + p3)*t*t*t);
function brow_curve(c, n) = let(p = concat([c[0]], c, [c[len(c)-1]]))
    concat([for (i = [1 : len(p) - 3], j = [0 : n - 1])
        _cr(p[i-1], p[i], p[i+1], p[i+2], j / n)], [c[len(c)-1]]);
// CONTOURED ceiling (2026-07-16, from the IMG_1233 brow measurements): the underside now FOLLOWS the measured
// stepped brow top (8.9 -> 8.2 -> 6.6 -> 3.6 above the base) so the clamp seats SNUGLY along the
// full 19.4 mm depth instead of bridging on the tall front step. Steps in CAD z (brow front top
// = -2.6 datum): -2.6 / -3.3 / -4.9 / -7.9 at the measured y stations (+6..+1 / +1..-4.6 /
// -4.6..-9 / -9..-13.4). Built as ONE extruded cross-section (ceiling + both jaws).
module brow_hook() {
    fs = brow_front_y + clamp_pad_t + slip_front;                    // slot front plane (+8.5)
    rs = brow_front_y - brow_depth_max - clamp_pad_t - slip_rear;    // slot rear plane (-15.55)
    // SMOOTH curve (2026-07-17): the measured steps were sampling stations of the
    // brow's smooth top — the underside is now a Catmull-Rom spline through the station
    // midpoints. The tall-front section stays FLAT at the -2.6 datum (seated height / the
    // OPTIC_DROP reference is preserved) and a progressive 0..0.3 mm relief grows toward the
    // back so the clamp registers at the front and cannot rock; pads + the screw make it snug.
    // anchored at each measured step's REAR-TOP corner = the conservative upper envelope
    // (midpoint anchoring dipped into the step corners, 40.9 mm^3 vs the measured stand-in)
    ctrl = [[fs, -2.6], [1.0, -2.6], [-4.6, -3.15], [-9.0, -4.65],
            [-13.4, -7.6], [rs, -8.0]];
    curve = brow_curve(ctrl, 8);                                     // fs -> rs, smooth
    pts = concat(
        [[fs + wall, -front_jaw_z], [fs + wall, 0],                  // front jaw outer face -> top
         [rs - wall, 0], [rs - wall, -rear_jaw_z],                   // top -> rear jaw outer face
         [rs, -rear_jaw_z]],                                         // rear jaw inner face
        [for (i = [len(curve) - 1 : -1 : 0]) curve[i]],              // underside curve rs -> fs
        [[fs, -front_jaw_z]]                                         // front jaw inner face
    );
    difference() {
        translate([-clip_w/2, 0, 0]) rotate([90, 0, 90])
            linear_extrude(height = clip_w) polygon(pts);
        translate([0, fs + wall + 1, -7]) rotate([90, 0, 0])         // M3 self-thread hole,
            cylinder(h = wall + 2, d = screw_d);                     //  mid of the 8.9 face
    }
}
module clips_and_rail() { rail(); for (s=[-1,1]) translate([s*clip_x, 0, 0]) brow_hook(); }

bp_w = 10; bp_h = 6; bp_t = 2.5; bp_recess = 0.8;
module bond_pads() {                        // 3 epoxy hard-points to bond the carrier to the frame
    for (x = [0, rail_half - clip_w, -(rail_half - clip_w)])
        translate([x, -bp_t/2, -rail_h/2]) difference() {
            rbox(bp_w, bp_t, bp_h, 1);
            translate([0, -bp_t/2, 0]) rbox(bp_w-2, bp_recess*2, bp_h-2, 0.6);
        }
}

// Rigid IMU pocket — a TOWER on the rail-top midline (2026-07-02 redesign). The old forward
// shelf hung 25 mm over the brow at the midline: its gusset dipped into the glasses brow AND the
// shelf blocked the nose-bridge corridor the pupil booms now drop through. The board now sits
// FLAT (X->+x, Y->+y unchanged — the axis contract software/imu.py assumes) on a short gusseted
// tower at z=16..21: above the transit legs (z<=15 incl. elbow spheres), clear of the drop
// corridors (they start y>=9.6; shelf reaches ~10.6 with z-separation), and far above the FOV
// cone. The lead drops straight to the wire groove. 2 M2 corner bores; pocket opens UP.
module imu_mount() {
    ow = imu_w + 2*imu_wall;  od = imu_h + 2*imu_wall;
    tower_top = 16;
    // pedestal: rooted in the rail top's FRONT half (the rear half carries the wire groove),
    // tapering up to the shelf — a stiff short column
    hull() {
        translate([0, 1.5, rail_h - 0.7]) cube([16, 5, 0.2], center = true);
        translate([0, 0, tower_top]) cube([10, 5, 0.2], center = true);
    }
    translate([0, 0, tower_top + (imu_t + 2*imu_wall)/2]) {
        difference() {
            rbox(ow, od, imu_t + 2*imu_wall, 1.5);                       // shelf block
            translate([0, 0, imu_wall]) cube([imu_w, imu_h, imu_t + 2], center = true);  // PCB pocket (opens +z)
            for (sx = [-1, 1]) for (sy = [-1, 1])                        // M2 corner bores
                translate([sx*(imu_w/2 - 2.2), sy*(imu_h/2 - 2.2), -imu_t - 1])
                    cylinder(h = imu_t + 2*imu_wall + 2, d = imu_screw_d);
        }
    }
}

// =====================================================================
//  EYE / FACE STAND-IN (preview only — positions straight from rig.py via sim2cad)
// =====================================================================
module eye_standin() {
    for (s = [-1, 1]) {
        cor = sim2cad([s*ipd/2, 0, -28.5]);
        color([0.85,0.9,1.0,0.30]) translate(cor) sphere(12);                       // eyeball
        color([0.1,0.1,0.1,0.5])   translate(sim2cad([s*ipd/2, 0, -16.0])) sphere(5.8); // cornea
        color("red")               translate(sim2cad([s*CANTH_SIM[0], CANTH_SIM[1], CANTH_SIM[2]])) sphere(1.6); // outer canthus
        color([1.0,0.3,0.3,0.10]) translate(cor)                                     // SEE-THROUGH CONE
            rotate([-90,0,0]) cylinder(h = 44, r1 = 2, r2 = 44*tan(25), $fn = 64);
    }
    color([0.2,0.4,0.8,0.12]) translate([0, 0, -OPTIC_DROP]) cube([ipd+70, 0.6, 46], center=true); // lens plane
}

// =====================================================================
//  GLASSES STAND-IN (visual only)
// =====================================================================
module glasses_dummy() {
    vz = -g_brow_h - g_visor_h/2 + 1;  Rv = (g_front_w/2 - 4) / sin(g_wrap_arc/2);
    color([0.06,0.07,0.10,0.5]) difference() {
        translate([0,0,vz]) curved_visor(Rv, g_wrap_arc, g_visor_th, g_visor_h, 2.5);
        translate([0,0,vz-g_visor_h/2]) scale([g_nose_w/(2*g_nose_h),1,1])
            rotate([-90,0,0]) cylinder(h = g_visor_th+14, d = 2*g_nose_h, center=true);
    }
    color([0.13,0.13,0.16,0.92]) {
        // brow: MEASURED profile (IMG_1233) as the REAL SMOOTH CURVE — the steps in the
        // sketch were sampling stations; the top is a Catmull-Rom through their corners
        // (front face +6, back edge -13.4, base -8.9). The clamp's underside is the same
        // curve with its 0..0.3 mm progressive relief, so they mate as in reality.
        translate([-g_front_w/2, 0, 0]) rotate([90, 0, 90])
            linear_extrude(height = g_front_w)
                polygon(concat([[6.0, -8.9]],
                               brow_curve([[6.0, 0], [1.0, 0], [-4.6, -0.7],
                                           [-9.0, -2.3], [-13.4, -5.3]], 8),
                               [[-13.4, -8.9]]));
        for (s=[-1,1]) {
            hull() {
                translate([s*(g_front_w/2-7), -3, -g_brow_h/2]) rbox(8, 11, g_visor_h*0.45, 2);
                translate([s*(g_front_w/2-9), -g_temple_len*0.45, -g_brow_h-2]) rbox(7, 11, 9, 2);
            }
            hull() {
                translate([s*(g_front_w/2-9), -g_temple_len*0.45, -g_brow_h-2]) rbox(7, 11, 9, 2);
                translate([s*(g_front_w/2-13), -g_temple_len, -g_brow_h-12]) rbox(6, 10, 8, 2);
            }
        }
    }
}

// =====================================================================
//  ASSEMBLY
// =====================================================================
// THE PRINTABLE CLIP-ON: rail + padded brow clamps + camera holders/booms + IMU pocket, all joined
// into ONE connected solid. The IR rings are NOT here (they sit at the lens rim, structurally
// separate) -> print them with part="ir_ring". No bought camera modules, no epoxy pads.
module carrier() {
    color("dimgray") clips_and_rail();
    world_cam(1);  world_cam(-1);                       // 2 world (shared by both eyes)
    eye_cam(1);    eye_cam(-1);                          // 2 eye-corner (one per eye)
    pupil_cam(1);  pupil_cam(-1);                        // 2 NIR eye-track (one per eye) = 6-cam BINOCULAR CORE
    if (build_stereo) { eye_cam2(1); eye_cam2(-1); }    // +2 stereo -> 8-cam FULL (the <1px upgrade)
    color("seagreen") imu_mount();                      // 9-DoF IMU rides the carrier (no separate bridge)
    // NO bond_pads / NO epoxy: fully REMOVABLE — held by the padded brow clamps. IR ring is part="ir_ring".
}

// IR ring as its own PRINTABLE part: the baffles joined by a thin arc into one loop per eye, so it
// prints as a solid piece that clips/glues to the lens rim (it doesn't hang off the carrier).
module ir_ring_part(side) {
    cen = sim2cad([side*ipd/2, 0, 0]);
    pts = [for (i = [0:ir_n-1]) [cen[0] + ir_ring_r*cos(i*360/ir_n), cen[1],
                                 cen[2] + ir_ring_r*sin(i*360/ir_n)]];
    color("red") {
        for (i = [0:ir_n-1]) ir_led_baffle(pts[i]);                 // the baffles
        for (i = [0:ir_n-1]) capsule(pts[i], pts[(i+1) % ir_n], 2.4); // thin connecting arc -> one loop
    }
}
// Fit-check preview ONLY (not printable): the clip-on + translucent eye + translucent glasses.
// (Set show_cams=true at the top to also see the bought camera modules in the holders.)
module preview() {
    carrier();
    eye_standin();
    translate([0, 0, -wall]) glasses_dummy();
}

if      (part == "carrier")   carrier();                              // <- PRINT THIS (the clip-on)
else if (part == "ir_ring")   { ir_ring_part(1); ir_ring_part(-1); }  // <- PRINT (IR ring, clips to lens rim)
else if (part == "imu_mount") imu_mount();                            // standalone IMU shelf (already in carrier)
else if (part == "board_cam") board_cam(OV_BOARD, OV_PITCH);          // a single holder, for tuning
else if (part == "bond_pads") bond_pads();                            // optional permanent-bond hard-points
else if (part == "preview")   preview();                             // glasses + eye + clip-on (NOT printable)
else if (part == "none")      ;                                       // render nothing (overlap_check.scad harness)
else                          carrier();                              // default = the printable clip-on
