// =====================================================================
//  overlap_check.scad — pairwise solid-intersection harness for the clip-on carrier
//  Driven by software/cad_overlap.py:
//      openscad -D 'part="none"' -D 'compA="worldR"' -D 'compB="glasses"' \
//               --export-format binstl -o pair.stl overlap_check.scad
//  Renders intersection(compA, compB); an empty STL (or ~0 volume) = no overlap.
//  Component names mirror the carrier's real submodules; "glasses"/"cone*"/"eyeball*"
//  are the physical keep-outs (the glasses body stand-in, see-through cones, eyes).
// =====================================================================
compA = "rail";
compB = "glasses";

include <xreal_one_mount.scad>

module comp(name) {
    if      (name == "rail")     rail();
    else if (name == "clampR")   translate([ clip_x, 0, 0]) brow_hook();
    else if (name == "clampL")   translate([-clip_x, 0, 0]) brow_hook();
    else if (name == "imu")      imu_mount();
    else if (name == "worldR")   world_cam(1);
    else if (name == "worldL")   world_cam(-1);
    else if (name == "eyeR")     eye_cam(1);
    else if (name == "eyeL")     eye_cam(-1);
    else if (name == "pupilR")   pupil_cam(1);
    else if (name == "pupilL")   pupil_cam(-1);
    // split components: each camera's BOOM vs its own HOLDER / physical-PCB keep-out
    // (inside one module these were never checkable — the ball-end-through-plate bug hid here)
    else if (name == "worldR_boom") world_cam(1, render = "boom");
    else if (name == "worldR_hold") world_cam(1, render = "holder");
    else if (name == "worldR_pcb")  world_cam(1, render = "pcbzone");
    else if (name == "eyeR_boom")   eye_cam(1, render = "boom");
    else if (name == "eyeR_hold")   eye_cam(1, render = "holder");
    else if (name == "eyeR_pcb")    eye_cam(1, render = "pcbzone");
    else if (name == "pupilR_boom") pupil_cam(1, render = "boom");
    else if (name == "pupilR_hold") pupil_cam(1, render = "holder");
    else if (name == "pupilR_pcb")  pupil_cam(1, render = "pcbzone");
    else if (name == "pupilR_led")  pupil_led(1);                     // mounted LED bracket (right)
    else if (name == "glasses")  translate([0, 0, -wall]) glasses_dummy();  // preview() offset
    else if (name == "eyeballR") translate(sim2cad([ ipd/2, 0, -28.5])) sphere(12);
    else if (name == "eyeballL") translate(sim2cad([-ipd/2, 0, -28.5])) sphere(12);
    // See-through cone, ELLIPTICAL: the display is 16:9, so the vertical half-FOV (~14.6 deg)
    // is much smaller than the horizontal (25 deg). rig.py's import assert keeps the circular
    // 25-deg cone (conservative, for camera CENTERS); for printed SOLIDS near the brow the
    // circular cone over-flags vertically, so the harness scales z by tan(14.6)/tan(25).
    else if (name == "coneR")    translate(sim2cad([ ipd/2, 0, -28.5])) scale([1, 1, tan(14.6)/tan(25)])
                                     rotate([-90,0,0]) cylinder(h = 44, r1 = 2, r2 = 44*tan(25), $fn = 64);
    else if (name == "coneL")    translate(sim2cad([-ipd/2, 0, -28.5])) scale([1, 1, tan(14.6)/tan(25)])
                                     rotate([-90,0,0]) cylinder(h = 44, r1 = 2, r2 = 44*tan(25), $fn = 64);
    else assert(false, str("unknown component: ", name));
}

intersection() { comp(compA); comp(compB); }
