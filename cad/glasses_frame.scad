// =====================================================================
// AR Eye-Corner Calibration Rig, Meta-Ray-Ban-style frame, modified
// =====================================================================
//  NOTE: This is the LEGACY Path-B (DIY combiner) FALLBACK frame, NOT the canonical
// build. The canonical model is the 6-camera BINOCULAR rig on the XREAL One Pro , 
//  print `xreal_one_mount.scad` (bonded carrier: 2 world + 2 eye-corner + 2 NIR pupil)
//  + `eye_tracking_bridge.scad` (nose/IMU). Use THIS file only if you have no XREAL
//  display and are building a from-scratch Path-B combiner. The world cameras here are
//  placed at the PUPILS (±ipd/2), matching the XREAL mount, so the camera sees what the
//  eye sees.
//  ⚠️ This frame sits directly on the face: pad EVERY face-contact surface (nose pads,
// brow, temple ends) with soft silicone/foam, no bare hard plastic on skin.
//
//  Parametric, 3D-printable. A "Wayfarer"-ish frame with mounts added for:
//    - 2x forward "world" cameras (outer top corners, like Ray-Ban Meta)
//    - 2x inward "eye-corner" tube cameras (booms aimed at the canthi)
//    - 1x display combiner (Path B beamsplitter + microdisplay shelf)
//
//  This is a *starting* parametric model. Measure your own face / your real
//  Ray-Ban Meta frames and tune the parameters at the top before printing.
//  Print in PETG or ABS (lives on your face, near skin heat). Supports off,
// 0.2 mm layers. The eye-cam booms are thin, print slow or add a brim.
//
//  Render a single part for printing by setting `part` below, or "all" to
//  preview the assembly. Then: openscad -o frame.stl glasses_frame.scad
// =====================================================================

part = "all";   // "all" | "frame" | "world_pod" | "eye_boom" | "combiner"

$fn = 56;

// ---------- face / frame fit (MEASURE THESE on your face) -------------
lens_w        = 50;   // lens opening width
lens_h        = 40;   // lens opening height
lens_r        = 9;    // lens corner radius
bridge_gap    = 18;   // gap between the two lenses (over the nose)
ipd = 64; // YOUR interpupillary distance, mm, world cams sit here
rim_t         = 3.5;  // rim border thickness around each lens
front_depth   = 6;    // front-to-back thickness of the frame front
brow_h        = 5;    // height of the top brow bar
panto_tilt    = 8;    // pantoscopic tilt of temples, degrees

// ---------- temples (arms) --------------------------------------------
temple_len    = 132;
temple_h      = 9;
temple_t      = 4;
ear_bend_len  = 22;   // the part that curls behind the ear
ear_bend_deg  = 28;

// ---------- world camera pod (forward) --------------------------------
// Sized for a ~10x10 mm board camera (e.g. small USB module). Tune to yours.
wc_w = 11;  wc_h = 11;  wc_d = 9;  wc_wall = 2;
wc_lens_d = 5;          // hole for the lens barrel to see through

// ---------- eye-corner tube camera (inward) ---------------------------
// Sized for a 7 mm endoscope/borescope tube camera. Tune ec_tube_d to yours.
ec_tube_d   = 7.4;      // tube dia + clearance
ec_ring_t   = 2.4;      // clamp ring wall
ec_boom_len = 24;       // reach from frame edge toward the eye corner
ec_boom_w   = 5;
ec_boom_t   = 3;
ec_aim_in   = 34;       // yaw: point inward toward the outer canthus, deg
ec_aim_down = 12;       // pitch: a touch downward, deg

// ---------- display combiner (Path B) ---------------------------------
// 45-deg beamsplitter in front of the RIGHT eye + shelf for a microdisplay.
cb_glass_w  = 24;       // beamsplitter width
cb_glass_h  = 20;       // beamsplitter height
cb_slot_t   = 1.6;      // glass thickness + clearance
cb_disp_w   = 28;       // microdisplay board footprint
cb_disp_d   = 22;
cb_arm_t    = 3;
which_eye   = "right";  // which lens carries the combiner

// derived lens centers (x); +x is wearer's left in this model
lens_cx = bridge_gap/2 + lens_w/2;

// =====================================================================
//  PRIMITIVES
// =====================================================================
module rrect(w, h, r) {              // rounded rectangle, centered, 2D
    offset(r) offset(-r) square([w, h], center = true);
}

module lens_rim(cx) {                // one lens border, extruded
    translate([cx, 0, 0])
    linear_extrude(front_depth)
    difference() {
        rrect(lens_w + 2*rim_t, lens_h + 2*rim_t, lens_r + rim_t);
        rrect(lens_w, lens_h, lens_r);
    }
}

// =====================================================================
//  FRAME FRONT  (two rims + bridge + brow bar)
// =====================================================================
module bridge_bar() {
    bw = bridge_gap + 2*rim_t + 6;
    translate([-bw/2, lens_h/2 - brow_h, 0])
        cube([bw, brow_h, front_depth]);
}

module brow_bar() {
    bw = 2*lens_cx + lens_w + 2*rim_t;   // span outer-to-outer
    translate([-bw/2, lens_h/2 + rim_t - brow_h, 0])
        cube([bw, brow_h, front_depth]);
}

module nose_pads() {
    for (s = [-1, 1])
        translate([s * (bridge_gap/2 - 1), -lens_h/2 + 8, front_depth/2])
            rotate([0, s*12, 0])
                scale([1, 1.6, 0.6]) sphere(4);
}

// =====================================================================
//  TEMPLES (arms)
// =====================================================================
module temple(side) {                // side = -1 left, +1 right
    sx = side * (lens_cx + lens_w/2 + rim_t);
    translate([sx, lens_h/2 - temple_h, front_depth])
    rotate([panto_tilt, 0, 0])
    rotate([0, side*90, 0]) {        // run the arm backward (+z -> along x)
        // straight part
        cube([temple_len - ear_bend_len, temple_h, temple_t]);
        // ear bend
        translate([temple_len - ear_bend_len, 0, 0])
            rotate([0, 0, side*ear_bend_deg])
                cube([ear_bend_len, temple_h, temple_t]);
    }
}

// =====================================================================
//  WORLD CAMERA POD  (outer top corner, facing forward)
// =====================================================================
module world_pod_solid() {
    cube([wc_w + 2*wc_wall, wc_h + 2*wc_wall, wc_d + wc_wall]);
}
module world_pod_cavity() {
    translate([wc_wall, wc_wall, wc_wall])
        cube([wc_w, wc_h, wc_d + 1]);
    // lens see-through hole out the front (-y face here mapped below)
    translate([(wc_w + 2*wc_wall)/2, -1, (wc_d + wc_wall)/2])
        rotate([-90, 0, 0]) cylinder(h = wc_wall + 2, d = wc_lens_d);
}
module world_pod() {                 // standalone, origin at corner
    difference() { world_pod_solid(); world_pod_cavity(); }
}
module world_pod_on_frame(side) {
    // Place the world cam at the PUPIL (±ipd/2), just above the lens aperture,
    // so its viewpoint matches the eye's (minimal parallax). Pointing forward.
    px = side * ipd / 2;
    translate([px - (wc_w + 2*wc_wall)/2,
               lens_h/2 + rim_t - (wc_h + 2*wc_wall),
               front_depth - (wc_d + wc_wall)])
        world_pod();
}

// =====================================================================
//  EYE-CORNER CAMERA BOOM  (tube clamp on an angled boom)
// =====================================================================
module tube_ring() {
    difference() {
        cylinder(h = ec_boom_w + 2, d = ec_tube_d + 2*ec_ring_t);
        translate([0, 0, -1]) cylinder(h = ec_boom_w + 4, d = ec_tube_d);
        // slit so it clamps onto the tube
        translate([-0.6, 0, -1]) cube([1.2, ec_tube_d, ec_boom_w + 4]);
    }
}
module eye_boom() {                  // standalone, boom from origin
    // boom arm
    cube([ec_boom_len, ec_boom_w, ec_boom_t]);
    // ring at the far end, axis along boom so the tube looks back at the eye
    translate([ec_boom_len, ec_boom_w/2, 0])
        rotate([0, 0, 0]) tube_ring();
}
module eye_boom_on_frame(side) {     // side = -1 left, +1 right
    sx = side * (lens_cx + lens_w/2 + rim_t);
    translate([sx, lens_h/2 - temple_h, front_depth - ec_boom_t])
    rotate([ -ec_aim_down, 0, -side*(90 - ec_aim_in) ])
        eye_boom();
}

// =====================================================================
//  DISPLAY COMBINER  (Path B: beamsplitter + microdisplay shelf)
// =====================================================================
module combiner() {
    s = (which_eye == "right") ? 1 : -1;
    cx = s * lens_cx;
    translate([cx, 0, front_depth]) {
        // 45-deg slot to hold the beamsplitter in front of the eye
        translate([0, 0, 6])
        rotate([45, 0, 0])
        difference() {
            translate([-cb_glass_w/2 - cb_arm_t, -cb_arm_t, -cb_arm_t])
                cube([cb_glass_w + 2*cb_arm_t, cb_glass_h + 2*cb_arm_t, cb_arm_t]);
            translate([-cb_glass_w/2, 0, -cb_slot_t])
                cube([cb_glass_w, cb_glass_h, cb_slot_t]);
        }
        // shelf above the eye to hold the microdisplay facing down into the glass
        translate([-cb_disp_w/2, -cb_disp_d/2, 6 + cb_glass_h*0.7])
            cube([cb_disp_w, cb_disp_d, cb_arm_t]);
        // two support posts from the brow up to the shelf
        for (px = [-cb_disp_w/2 + 2, cb_disp_w/2 - 2 - cb_arm_t])
            translate([px, cb_disp_d/2 - cb_arm_t, 0])
                cube([cb_arm_t, cb_arm_t, 6 + cb_glass_h*0.7]);
    }
}

// =====================================================================
//  ASSEMBLY
// =====================================================================
module assembly() {
    color("dimgray") {
        lens_rim( lens_cx);
        lens_rim(-lens_cx);
        bridge_bar();
        brow_bar();
        nose_pads();
        temple( 1);
        temple(-1);
    }
    color("steelblue") { world_pod_on_frame( 1); world_pod_on_frame(-1); }
    color("orange")    { eye_boom_on_frame( 1);  eye_boom_on_frame(-1); }
    color("mediumseagreen") combiner();
}

// ---------- part selector ----------
if      (part == "all")       assembly();
else if (part == "frame")     { lens_rim(lens_cx); lens_rim(-lens_cx); bridge_bar(); brow_bar(); nose_pads(); temple(1); temple(-1); }
else if (part == "world_pod") world_pod();
else if (part == "eye_boom")  eye_boom();
else if (part == "combiner")  combiner();
