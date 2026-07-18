// =====================================================================
//  Nose-bridge NOSE / IMU SUPPORT  (clips to XREAL One Pro / One / Rokid bridge)
// =====================================================================
//  ⚠️ NOT REQUIRED FOR THE BUILD. The whole rig needs ONLY cad/xreal_one_mount.scad:
//  the bonded carrier now holds all 6 cameras, the IR rings, AND the IMU
//  (module imu_mount). The XREAL One Pro's OWN padded nose pads carry the face. This
//  file is kept as an OPTIONAL accessory (a separate clip-on nose/IMU support if you'd
//  rather not put the IMU on the carrier) — it is legacy and not part of the canonical
//  single-part build.
// =====================================================================
//  ⚠️ NOT A CAMERA MODULE. The 6-camera BINOCULAR build keeps ALL cameras on the single
//  bonded carrier in cad/xreal_one_mount.scad — including the TWO NIR pupil cameras
//  (pupil_cam, one per eye) and the IR LED rings around BOTH eyes (ir_leds), at the
//  software/rig.py positions (the ONE source of truth for camera geometry). The old
//  camera modules in THIS file have been removed from the assembly to avoid a stray cam.
//
//  This file is the NON-camera nose support: the bridge body, soft (PADDED) nose pads, the
//  bridge clip, and the IMU shelf (tilt/slip + the Kalman drift filter — see imu.py /
//  WIRING.md). cam_holder()/ir_unit()/cam_arm() are kept only as reference and are NOT
//  rendered. Same Z-up frame as xreal_one_mount.scad:
//    +x = right, +y = FORWARD (world), +z = UP, origin at the optic center. mm.
//
//  ⚠️ MEASURE + tune the bridge-clip thickness and nose-pad fit. Print in a STIFF material
//    (SLA resin or PC / nylon-CF, NOT plain PETG/PLA), 0.2 mm. The nose pads carry a soft
//    silicone/foam pad in the recess — no bare hard plastic on the nose.
//    openscad -D 'part="all"' -o bridge_support.stl eye_tracking_bridge.scad
//    openscad -D 'part="all"' -o bridge_support.stl eye_tracking_bridge.scad
// =====================================================================

part = "all";   // "all" | "support" | "imu_mount"  (camera parts removed; see xreal_one_mount.scad)
$fn = 56;

// ---------- which glasses? (sets the bridge-bar thickness preset) ------
target = "xreal_one_pro";   // "xreal_one_pro" | "xreal_one" | "rokid_max"

// ---------- eye / face fit (MEASURE) ----------------------------------
ipd       = 64;     // interpupillary distance (One Pro: M 57–66 / L 66–75 mm)
eye_back  = 18;     // pupil sits this far BEHIND the optic plane (-y)
eye_drop  = 2;      // pupil this far below optic center (-z)

// ---------- eye camera (OV9281-class tiny module) ---------------------
ov_w = 9;  ov_h = 9;  ov_d = 6;   // module footprint + depth (tune to your cam)
lens_d = 3.6;                     // lens clear-aperture hole
wall   = 1.6;                     // thin walls = light on the nose

// (REFERENCE ONLY — not rendered.) These old bridge-camera offsets are STALE; the live NIR
// pupil-camera geometry is software/rig.py PUPIL_POS (now [24, -30, 6]) realised on the bonded
// carrier in xreal_one_mount.scad. Do not treat the numbers below as current.
cam_nasal_in = 6;    // |x| inset toward the nose from the pupil (so x = 26 at ipd 64)
cam_fwd      = 6;    // in front of the optic plane  (sim z = 6)
cam_below    = 14;   // below the optic center        (sim y = -14)
arm_w        = 6;    // camera-arm thickness (thicker = stiffer; rigidity matters, see flex.log)

// ---------- IR LEDs (940 nm, low power — see EYE_TRACKING.md safety) ---
led_d     = 3.2;     // 3 mm LED + clearance (use SMD pockets if smaller)
led_wall  = 1.0;
led_depth = 4.5;

// ---------- bridge body + nose pads -----------------------------------
bw = 30;  bd = 12;  bh = 9;       // central body block
body_y = -4;  body_z = -13;       // body center (below + behind the optic)
arm_root_x = 8;                   // where each arm leaves the body
pad_x = 7;  pad_splay = 18;       // nose-pad lateral offset + inward cant (deg)
pad_w = 9;  pad_h = 13;  pad_t = 3;

// ---------- IMU (6/9-axis MEMS breakout: tilt/slip detection, NOT geometry ID) ----
//  Rigidly mounted to the body (which clips to the frame), so it reads the GLASSES'
//  pitch/roll vs gravity. Its job is robustness — bump/re-seat detection + inter-frame
//  motion + a gravity "down" reference — NOT identifying eye geometry (the sim showed a
//  level adds ~0 there). Mount FLAT with the board's X along +x and Y along +y so the
//  reading maps directly to pose dev[0] (pantoscopic pitch) / dev[2] (roll); see imu.py.
imu_w = 21;  imu_h = 18;  imu_t = 2.0;   // typical 9-DoF breakout PCB (e.g. ICM-20948/BNO055)
imu_wall = 1.6;  imu_screw_d = 2.2;      // M2 mount holes
imu_drop = 5;                            // sits this far below the body bottom, on a shelf

// ---------- clip onto the glasses bridge bar (MEASURE xr_bridge_t) -----
xr_bridge_t = (target == "rokid_max") ? 4.5 : (target == "xreal_one") ? 4.0 : 3.6; // bridge-bar thickness (One Pro flat optic ≈ 3.6; MEASURE)
clip_w = 10;  clip_reach = 11;  clip_wall = 2.2;  clip_clear = 0.4;
clip_up = 9;         // how far up from the body the clip reaches to the bridge

// =====================================================================
//  PRIMITIVES (shared style with the main bracket)
// =====================================================================
module rbox(w, d, h, r) {
    hull() for (sx=[-1,1]) for (sy=[-1,1]) for (sz=[-1,1])
        translate([sx*(w/2-r), sy*(d/2-r), sz*(h/2-r)]) sphere(r);
}
module capsule(p0, p1, dia) {
    hull() { translate(p0) sphere(dia/2); translate(p1) sphere(dia/2); }
}
// orient children's +z from `from` toward `to`
module aim_z_at(from, to) {
    v = [to[0]-from[0], to[1]-from[1], to[2]-from[2]];
    L = norm(v);
    rotate([-asin(v[1]/L), atan2(v[0], v[2]), 0]) children();
}

function pupil(side) = [side*ipd/2, -eye_back, -eye_drop];
function campos(side) = [side*(ipd/2 - cam_nasal_in), cam_fwd, -cam_below];

// =====================================================================
//  CAMERA HOLDER  (local +z = optical axis; lens faces +z toward the eye)
// =====================================================================
module cam_holder() {
    ow = ov_w + 2*wall;  oh = ov_h + 2*wall;  od = ov_d + wall;
    difference() {
        translate([-ow/2, -oh/2, -od]) cube([ow, oh, od]);        // body, front at z=0
        translate([-ov_w/2, -ov_h/2, -od-0.5]) cube([ov_w, ov_h, ov_d+0.5]); // board pocket (back)
        translate([0, 0, -wall-0.5]) cylinder(h = wall+1, d = lens_d);       // lens hole (front)
    }
}

// one IR-LED boss aimed at the eye, on a short stalk from the camera
module ir_unit(side, pos) {
    capsule(campos(side), pos, 3);
    translate(pos) aim_z_at(pos, pupil(side))
        difference() {
            cylinder(h = led_depth, d = led_d + 2*led_wall);
            translate([0, 0, -0.5]) cylinder(h = led_depth + 1, d = led_d);
        }
}

// =====================================================================
//  CAMERA ARM  (strut from the body + aimed camera + two IR LEDs)
// =====================================================================
module cam_arm(side) {
    root = [side*arm_root_x, body_y, body_z];
    color("steelblue") {
        capsule(root, campos(side), arm_w);
        translate(campos(side)) aim_z_at(campos(side), pupil(side)) cam_holder();
    }
    color("orange") {
        ir_unit(side, campos(side) + [side*7, 0,  3]);    // lateral LED
        ir_unit(side, campos(side) + [-side*3, 0, -6]);   // lower LED
    }
}

// =====================================================================
//  BODY + NOSE PADS + BRIDGE CLIP
// =====================================================================
module bridge_body() {
    translate([0, body_y, body_z]) rbox(bw, bd, bh, 2);
}

// Soft, broad nose pads. The face-contacting side is a RECESS sized to seat a stick-on
// SILICONE / foam pad (pad_soft_t thick) so the nose never touches bare hard plastic — required
// padding for every printed face-contact surface (EYE_TRACKING.md §2 comfort). Broad + canted to
// spread pressure; rounded edges.
pad_soft_t = 1.6;   // thickness of the soft pad that drops into the recess
module nose_pads() {
    for (s = [-1, 1])
        translate([s*pad_x, body_y, body_z - bh/2])
            rotate([0, 0, s*pad_splay])
                difference() {
                    rbox(pad_w, pad_h, pad_t, 1);
                    // full-face recess for the soft pad (covers the whole contact area, not a thin ring)
                    translate([0, 0, pad_t/2 - pad_soft_t/2 + 0.01])
                        rbox(pad_w-1.6, pad_h-1.6, pad_soft_t, 0.8);
                }
}

// IMU shelf on the BACK (-y) face of the body: a flat recessed pocket + 2 M2 bosses.
// Flat & frame-rigid so the accelerometer's tilt = the glasses' tilt (axes per comment above).
module imu_mount() {
    ow = imu_w + 2*imu_wall;  oh = imu_h + 2*imu_wall;
    // shelf hangs off the back of the body, just below body center
    translate([0, body_y - bd/2 - imu_t/2 - imu_wall, body_z - imu_drop]) {
        difference() {
            rbox(ow, imu_t + 2*imu_wall, oh, 1.5);              // shelf block
            translate([0, imu_wall, 0])
                cube([imu_w, imu_t + 0.4, imu_h], center = true);   // PCB pocket (opens -y)
            for (sx = [-1, 1]) for (sz = [-1, 1])               // M2 mount holes
                translate([sx*(imu_w/2 - 2.2), -imu_t, sz*(imu_h/2 - 2.2)])
                    rotate([90, 0, 0]) cylinder(h = imu_wall + imu_t + 1, d = imu_screw_d);
        }
    }
    // small gusset tying the shelf to the body so it can't flex (rigid coupling)
    translate([0, body_y - bd/2, body_z - imu_drop/2])
        rotate([0, 0, 0]) capsule([0, 0, imu_drop/2], [0, -imu_t - imu_wall, -imu_drop/2], 3);
}

// C-clip that hooks the horizontal glasses bridge bar above the body
module bridge_clip() {
    sy = xr_bridge_t/2 + clip_clear;
    oy = sy + clip_wall;
    z0 = body_z + bh/2;                       // top of the body
    translate([0, body_y, z0]) {
        // riser up to the bridge
        translate([0, 0, clip_up/2]) rbox(clip_w, 2*oy, clip_up, 1.5);
        // hook: capped on top, open at the bottom (slides down onto the bar)
        translate([0, 0, clip_up]) difference() {
            translate([-clip_w/2, -oy, 0]) cube([clip_w, 2*oy, clip_reach]);
            translate([-clip_w/2-1, -sy, -1]) cube([clip_w+2, 2*sy, clip_reach - clip_wall + 1]);
        }
    }
}

// =====================================================================
//  ASSEMBLY
// =====================================================================
// NON-camera nose/IMU support only. The pupil cam + IR LEDs moved to xreal_one_mount.scad;
// cam_arm()/cam_holder()/ir_unit() above are kept for reference but deliberately NOT rendered
// (rendering them would re-introduce the stray 2nd cam at the old, occluding position).
module assembly() {
    color("dimgray") { bridge_body(); nose_pads(); bridge_clip(); }
    color("seagreen") imu_mount();
}

if      (part == "all")        assembly();
else if (part == "support")    assembly();
else if (part == "imu_mount")  imu_mount();
