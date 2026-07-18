# Angle Kappa, the comprehensive study, and how to get around it

The recurring wall in this project is that the pixel must land on the **visual axis** (the
line to the fovea), but everything we can cheaply sense references the **pupil / optical
axis**. The offset between them is **angle kappa**, and when we infer it from a subjective
overlay-alignment task it is mathematically tangled with the user's **perceptual bias**. This
document is the deep dive: what kappa actually is, the population data, every objective and
subjective way it is measured in the real world, *why* the confound arises, and a concrete
catalog of ways to break it, several of which are buildable and one of which is demonstrated
in `software/kappa.py`.

The thesis up front: **the confound is not fundamental. It is an artifact of using a single
subjective alignment at a single viewing distance.** Change the modality (image the eye),
change the variable (sample across vergence / gaze / pupil size), or change the task (let the
eye mark its own fovea), and kappa separates out.

---

## 1. The axes of the eye (people conflate these, precision matters)

| Axis | Definition | What senses it |
|---|---|---|
| **Optical axis** | best-fit line through the centers of curvature of cornea + lens | corneal topography, Purkinje images |
| **Pupillary axis** | normal to the cornea through the entrance-pupil center | pupil + corneal reflex (PCCR) |
| **Visual axis** | fixation point → nodal points → **fovea** | requires knowing the fovea |
| **Line of sight** | fixation point → entrance-pupil center → fovea (the **chief ray**) | this is what AR registration actually needs |
| **Achromatic / fixation axis** | center of rotation → fixation point | eye-tracker gaze (rotation) |

The fovea is **not** on the optical axis, it sits ~5° **temporal** and ~1-2° **inferior** to
the posterior pole. That displacement is the root of all these angles.

- **Angle alpha (α)**: optical axis vs visual axis, at the nodal point. The most "anatomical"
  and stable. ~5° (range 4-8°) horizontal.
- **Angle kappa (κ)**: **pupillary axis vs visual axis.** The clinically used one. ~3-6°
  horizontal, ~1-3° vertical. **Positive kappa** (the normal case): the visual axis is nasal
  to the pupillary axis (because the fovea is temporal), so a normal eye looks slightly
  divergent, "positive-kappa pseudo-exotropia."
- **Angle lambda (λ)**: pupillary axis vs line of sight. What the Hirschberg corneal-reflex
  test *actually* measures; clinically used interchangeably with kappa.

For our rig the quantity we need is the **line of sight / chief ray** per eye, i.e. kappa/lambda
in the glasses frame, at the current gaze and the current glasses pose.

---

## 2. Population statistics (what a database must span)

- Horizontal kappa mean ≈ **5°**, SD ≈ **1.5°**; vertical mean ≈ **1.5°**, SD ≈ **1°**
  (these are the values in `anatomy.py`: `KAPPA_X 5±1.5`, `KAPPA_Y 1.5±1.0`).
- **Correlates with refractive error / axial length**: larger (more positive) in hyperopes /
  short eyes, smaller or negative in myopes / long eyes. So kappa is *not independent* of
  globe size, a real cross-link a geometry database can exploit.
- Fairly **stable in adulthood**, roughly mirror-symmetric between the two eyes but not
  identical (inter-eye difference ~1°).
- Larger and more variable in children; shifts a little with pupil size and accommodation
  (because the entrance pupil moves), which is itself a *separating handle* (§5).

Implication for the database idea: kappa is a **2-D per-eye offset with ~1.5°/1° spread**.
At our 50°/1080px display that's **~32 px of population spread** in where the pixel must go , 
which is exactly why getting it wrong dominates the error, and why pinning it is the whole game.

---

## 3. Why kappa limits AR registration, precisely

A see-through pixel emits a fixed direction. To register, that direction must equal the eye's
**line of sight** to the target. Our cameras give the **pupil/optical axis** (pupil center,
corneal glints, eye-corner pose). The fixed angular gap between them, kappa, is unobservable
from the outside *by geometry alone*: two eyes with identical pupils/corneas but different
foveal positions need the pixel in different places, and nothing on the front of the eye shows
where the fovea is.

So historically kappa is recovered **subjectively**: show a target, have the user drive the
overlay onto it, and the correction reveals the line of sight. That works, but see §4.

---

## 4. The confound, stated exactly

A subjective alignment measures **where the user perceives alignment**, which is:

```
observed_offset  =  kappa            (anatomical, constant in the eye frame)
                 +  vergence-accommodation bias        (varies with target DISTANCE)
                 +  field/perceptual bias              (varies with gaze / screen position)
                 +  prism-adaptation / cognitive bias  (slow drift)
                 +  fixation noise (microsaccades)      (zero-mean, ~0.12° per glance)
```

In our current model two of these (`kappa` and a single constant `human_bias`) are **both
constant**, so they are perfectly confounded, you can only recover their **sum**. For *placing
the pixel right now* the sum is all you need (and the user's own calibration captures it). The
confound only bites when you want kappa **per se** (to label a geometry database) or want a
calibration at one condition to **transfer** to another.

**The escape is to exploit that the terms above have *different signatures*.** kappa is
constant in the eye frame; the vergence term varies with distance and **vanishes at the virtual
image plane**; the field term varies with gaze; fixation noise is zero-mean and averages as
1/√N. Sample across those variables and the constant (kappa) separates from the rest.

---

## 5. The catalog of ways around kappa

### A. Measure it objectively, don't infer it from alignment
The cleanest break: get kappa from a modality that doesn't use the overlay task.

1. **Purkinje / corneal-reflex (PCCR).** The 1st Purkinje image (corneal reflection of a
   coaxial IR LED) relative to the pupil center gives the pupillary-axis-to-line-of-sight angle
   directly when the user simply **looks at the light** ("look here" has far fewer free
   parameters than "align this floating overlay"). This is the Hirschberg/Krimsky test, and it
   is exactly what our **NIR eye tracker** does. Estimating kappa from PCCR while fixating known
   real points removes the vergence-accommodation conflict of overlay alignment.
2. **Corneal topography / optical biometry** (Orbscan, Pentacam, Galilei, IOLMaster 700) report
   angle kappa from the corneal vertex normal vs the pupillary axis vs fixation. A **one-time
   clinic/factory measurement** stored as a per-user biometric (recalled by iris ID) gives
   ground-truth kappa with zero ongoing calibration.
3. **Dual-Purkinje-image (DPI) tracking** (P1 + P4) yields the optical axis and eye rotation
   very precisely; with a fixation it pins the optical-visual offset.
4. **OCT / SLO fundus imaging** locates the fovea relative to the disc and pupil → anatomical
   alpha/kappa, fully objective (no subjective task at all).
5. **Retinal birefringence scanning (RBS).** The Henle fiber layer around the fovea is
   birefringent; scanning polarized light detects the **fovea itself** objectively (the basis of
   the Pediatric Vision Scanner). If miniaturizable into a glasses optic, it gives the visual
   axis with no user input.

### B. Let the eye mark its own fovea (subjective but bias-free)
6. **Haidinger's brushes.** Under polarized (or rotating-polarized) light, macular-pigment
   birefringence makes the user see a faint "propeller" **centered on their own fovea**. The
   user can place a target on the brushes, marking the fovea **directly**, bypassing the
   external-target alignment that introduces vergence/prism bias. Used clinically to diagnose
   eccentric fixation. A polarized stimulus in the display could give a near-objective foveal
   landmark.

### C. Statistical separation, same cheap sensors, smarter sampling
These need **no new hardware**, only a calibration that varies a controlled variable, and are
the most immediately buildable on our rig:

7. **Multi-vergence (multi-distance).** kappa is constant with target distance; the
   vergence-accommodation bias scales with `(1/d − 1/d_virtual)` and is **zero at the virtual
   image plane**. Calibrate at 2-3 distances (or at the virtual plane) and regress the offset
   vs vergence demand: the intercept is the vergence-clean alignment, the slope is the VAC bias.
   **This is demonstrated in `software/kappa.py`.**
8. **Multi-gaze.** kappa is constant in the eye frame; field-dependent display distortion and
   crowding/attention biases vary with screen position. Sampling the whole field separates the
   constant from the field-varying part (we already sweep the field).
9. **Binocular constraint.** Both eyes' visual axes must intersect the single real target;
   with two eye trackers this over-determines the geometry and constrains each eye's kappa
   beyond what a monocular subjective task can.
10. **Pupil-size modulation.** The entrance pupil (hence the chief ray) shifts with pupil size
    (Stiles-Crawford / pupil-center shift); kappa and some biases respond differently, adding a
    separating axis (drive pupil size with display luminance).

### D. Active / closed-loop foveal confirmation
11. Use the eye tracker to **confirm** the overlay lands on the visual axis: flash the overlay
    and detect the micro-response (no microsaccade correction / steady pursuit = on-fovea), or
    drive a closed loop that nulls the foveal error, turning a subjective judgement into an
    objective measurement.

### E. The product reframe (don't fight the part you don't need)
For *placing the pixel as the user perceives it*, you need the **sum** at the current
condition, which the user's own corrections capture, and the residual is then **fixation
noise**, which averages down (§ demonstrated). Separating kappa matters specifically for
**cross-condition transfer** and **database labeling**; for those, A + C are the levers.

---

## 6. Recommended strategy for this rig (stacked)

1. **PCCR eye tracker** (already on the BOM) → objective optical/pupillary axis + gaze.
2. **Brief "look at N real dots at 2-3 distances"** calibration → fit kappa as the constant,
   the VAC bias as the distance slope (method 7) → a *vergence-clean* per-eye kappa.
3. **Store it as a biometric** keyed to the iris → returning users skip calibration (method 2
   philosophy without a clinic).
4. **Average the residual** over multiple fixations (1/√N) to beat the microsaccade floor.
5. Optionally, a **polarized foveal-landmark** step (Haidinger, method 6) as a bias-free check.

This does not "delete" kappa, it **measures** it (objectively or by separation) instead of
guessing it from one biased task, which is what breaks the confound and lets the per-position
accuracy approach the fixation-noise floor rather than the perceptual-bias floor.

---

## 7. What the simulation shows (`software/kappa.py`, `--kappa`)

Two simulation results back the argument (plotted in `results/kappa_separation.png`, raw
output in `results/kappa_run.log`):

- **Convergence / perceived-vs-geometric.** Overlay error vs the number of the user's own
  corrections, reported *both* against geometric truth (floored by the perceptual bias) and
  against the user's perceived alignment (floored only by fixation noise, so it crosses 3 px as
  1/√N). Against what the user *perceives*, sub-3 px is reachable; the geometric-truth number
  is inflated by a bias the user does not experience as error.
- **Multi-vergence separation (method 7).** Calibrating at one distance fails to transfer to
  other distances (the VAC bias); calibrating across distances (or at the virtual plane)
  recovers a vergence-clean alignment and restores accuracy across distances, kappa separated
  from the vergence bias with no new hardware.

See the script header and `--kappa` output for the numbers.
