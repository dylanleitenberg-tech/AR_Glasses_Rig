# DOMAIN REFERENCE — eye anatomy, eye tracking, optics, and overlay mechanics

Written 2026-08-03 at Dylan's request for depth across the fields this rig sits in. Organised
around **what changes a decision here**, not around textbook completeness. Every section ends with
what it implies for our hardware.

Three findings up front, because they matter most:

1. **The canthus approach is the literature's answer to slippage, not a workaround.** 3D
   model-based trackers compensate for headset slip; 2D video-based ones have *no known solution*.
   We built the right thing for the right reason.
2. **The correct rendering viewpoint depends on what you are optimising**, and our design has
   already implicitly chosen. That choice explains a measurement we took but never accounted for.
3. **Display FOV is not a constant.** Pupil swim makes it vary with eye position in the eyebox —
   which means `geometry.DISPLAY_FOV_DEG = 50.0` is wrong in a way no single number can fix.

---

## 1. THE EYE — the anatomy that actually enters the maths

### 1.1 Two axes, not one
- **Optical axis** — the line through the eyeball centre, corneal centre, iris centre and pupil
  centre. It is what a camera can *see*.
- **Visual axis** — the line from the **fovea** through the corneal centre to the fixation point.
  It is where the person is actually looking.
- They are **not the same line.** The offset is **angle kappa** (its components alpha and beta),
  typically **~5° horizontal and ~1.5° vertical**, and it **varies between individuals**.

**Why it matters:** any tracker that measures the pupil measures the *optical* axis. Converting to
gaze requires kappa, which is a per-person constant needing at least single-point calibration.
This is exactly what `kappa.py` and `KAPPA.md` are about, and why the project measured
cross-distance error of 13 px → 2.4 px with multi-distance calibration.

### 1.2 The eye rotates around a point ~10 mm behind the pupil
The centre of rotation sits roughly **10 mm behind the pupil**. So as gaze changes, the **pupil
sweeps an arc of ~10 mm radius** while the rotation centre stays put.

This is the single most consequential fact in §3.

### 1.3 The canthus
The inner canthus (medial, tear-duct side) is a **skin/soft-tissue landmark fixed to the skull**,
not to the globe. It moves with the face and not with gaze. `rig.TRACKED_LANDMARK = "inner"` and
the measured pupil-correlation drop from **+0.99 (classical tracker) to +0.19 (learned model)** is
precisely the property being exploited: a face-fixed reference is only useful if it is genuinely
gaze-independent, and that number is the proof it is.

Limitation to keep honest: it is **soft tissue**. `rig.SOFT_TISSUE_SD = 0.15 mm` is a
"holding still" placeholder, and squinting, talking or smiling exceed it. Hence the standing
instruction to hold a neutral face while calibrating.

---

## 2. EYE TRACKING — the methods, and where each breaks

### 2.1 The dominant method: PCCR (pupil–corneal reflection)
IR LEDs make **glints** on the cornea; the vector between pupil centre and glint gives gaze.
**Dark pupil** (off-axis illumination, pupil appears dark) vs **bright pupil** (on-axis, retroreflection
makes it glow) — many systems alternate to segment the pupil robustly by differencing.

Universal in commercial trackers, and IR is used for a reason that is **not** about the pupil: an
IR bandpass filter makes the eye image **independent of ambient light**. Our rig has no IR LEDs, no
filter, and **no software exposure control on macOS** — which is why exposure is a physical lever
here and why blowout has repeatedly wrecked measurements.

### 2.2 2D vs 3D, and the finding that matters most
- **2D / appearance-based** — regress gaze directly from image features. Simple, but the mapping is
  only valid for the head-to-camera geometry present at calibration.
- **3D / model-based** — reconstruct an eye model (corneal centre, optical axis) in 3D, then apply
  kappa.

**The slippage result:** eye trackers using **3D models of the eye can compensate for slippage**,
and there is **no known solution for slippage compensation in 2D video-based eye tracking.**
Measured slippage penalties in the literature run **0.8–3.1° of added gaze error**, and slippage
happens constantly — pushing glasses up the nose, brushing hair, talking.

**This is the strongest external validation of our architecture.** We do not track gaze at all; we
track a **face-fixed landmark to recover the glasses' pose on the head**, which is the slippage
term itself. With a **zip-tied, non-repeatable mount**, that is not a nice-to-have — it is the only
thing standing between us and a calibration that expires the moment the rig shifts.

### 2.3 What a canthus tracker cannot do
It gives **glasses-relative head geometry**, not gaze. So it cannot supply the entrance-pupil
position for a given fixation (§3), cannot apply kappa, and cannot know where the user is looking.
Dylan's own `eyetracker.py` study measured the consequence honestly: corner-only **9.9 px** vs
+pupil **10.1 px** for registration — i.e. **no registration gain from adding the pupil.** §3
explains *why* that result is not surprising.

---

## 3. THE VIEWPOINT QUESTION — where the virtual camera actually goes

This is the deepest item here and it has a real consequence for us.

To render an overlay you need a **projection centre**: the point the virtual camera sits at. There
are three candidates and they are not interchangeable:

| candidate | property | cost |
|---|---|---|
| **Entrance pupil** | **highest angular accuracy** | **moves with gaze** (~10 mm arc, §1.2) |
| **Centre of rotation** | **gaze-independent**, stable | slightly worse angular accuracy |
| Optics' nodal point | matches the lens design | not where the eye is; mismatch causes distortion |

The literature states it directly: depending on whether **angular** or **positional** accuracy
matters, either the centre of rotation or the centre of the entrance pupil is chosen — and **the
entrance pupil yields higher angular accuracy.** Distortion, separately, results from the pupil
sitting away from the **nodal points of the optics**.

### What this means for our rig
**We render from a gaze-independent point, because that is the only thing a canthus can give us.**
That is a real choice with real consequences:

- ✅ It is stable, needs no gaze tracking, and does not require kappa.
- ✅ **It explains the 9.9 vs 10.1 px result.** Adding a pupil feature did not help registration
  because our whole formulation is built around a gaze-independent viewpoint — the pupil's
  information has nowhere to go. That measurement was never a fluke; it is structural.
- ❌ It gives up angular accuracy at **large gaze angles**, where the entrance pupil has swung
  furthest from the rotation centre. Worst case: a 10 mm displacement. At **0.5 m** that subtends
  **~1.1°** (44 px); at **20 m**, **~0.003°** (negligible).

**And that maps onto the stated goal.** The end goal is people and cars **far away**, where the
entrance-pupil-vs-rotation-centre distinction is worth ~0 px. **The design choice is well matched
to the application** — near-field content is where it would cost, and near-field is not the target.

---

## 4. THE DISPLAY — birdbath optics, and why FOV is not a number

The XREAL One Pro is a **birdbath**: a display panel, a beamsplitter, and a spherical
mirror/combiner. It is the same family as XREAL Air, Rokid Air, Huawei Vision Glass.

Key parameters, all of which we currently treat as constants and none of which are:
- **Eye relief / exit pupil distance** — the design distance from eyepiece to eye.
- **Eyebox** — the region the eye may occupy and still see the full FOV. Move outside it and the
  image vignettes or vanishes.
- **Pupil swim** — **aberrations across the pupil mean the image distorts as the eye moves within
  the eyebox.** The literature is explicit that this **exists in all near-to-eye display systems**.

### The uncomfortable implication for us
`geometry.DISPLAY_FOV_DEG = 50.0` is (a) **assumed, never measured**, and (b) **not even a
constant** — the effective mapping from angle to pixel varies with where the eye sits in the
eyebox. A single scalar cannot represent it.

**But our architecture can absorb this, and here is the part worth noticing:** the learned residual
takes the **eye-corner features as inputs**. Those features encode *where the glasses sit relative
to the eye* — which is exactly the variable pupil swim depends on. So a residual model on top of
geometry can in principle learn a **position-dependent** correction that no fixed FOV constant
could. That is a genuine architectural argument for geometry-plus-residual over either alone, and
it was not the reason we adopted it.

**Action:** `display_calib.py` exists to measure real FOV and distortion (K1/K2) and **has never
been run.** It is the highest-value unrun tool in the repo.

---

## 5. REGISTRATION ERROR — the budget, and what dominates

From Holloway's end-to-end analysis (already in `LATENCY_AND_TRACKING.md`, restated because it
frames everything above):

1. **System latency — the largest single term.** Unlike the rest, it scales with **head angular
   velocity**, so it is invisible standing still and dominant in motion. We measure 62 ms against a
   modern target of <20 ms.
2. **Eye position / viewpoint error** — §3. Bounded by ~10 mm, and negligible in the far field.
3. **Optical distortion** — §4. Currently unmodelled and unmeasured.
4. **Tracker noise** — measured ~0.003–0.004 frame-units on our dot detection.
5. **Calibration/parameter error** — what the sample-driven fit addresses.

**Priority follows directly:** latency first (in progress), then measure the display, then worry
about the rest. Chasing calibration accuracy while (1) and (3) are unaddressed is optimising the
smallest terms.

---

## 6. CONSOLIDATED IMPLICATIONS FOR THIS RIG

**Validated by the literature — keep doing:**
- Tracking a **face-fixed landmark for slippage compensation**. This is the 3D-model approach, and
  2D methods have no answer to slippage. With a zip-tied mount it is essential.
- **Gaze-independent viewpoint.** Well matched to a far-field application; costs ~nothing at 20 m.
- **Geometry as backbone + learned residual.** §4 gives an additional, stronger reason: the
  residual has access to eye-position features and can therefore learn pupil-swim-like effects.

**Corrected or newly explained:**
- The **9.9 vs 10.1 px pupil result is structural**, not a null finding — a gaze-independent
  formulation has no way to use gaze information.
- **Display FOV is not a constant**, so no single `DISPLAY_FOV_DEG` is right; the residual must
  carry the position-dependent part.

**Now clearly worth doing, in order:**
1. **Finish latency work** — the dominant term (§5).
2. **Run `display_calib.py`** — measure real FOV and distortion. It is written, unrun, and feeds
   the second-largest unmodelled term.
3. **Wire the XREAL IMU** — serves latency *and* `WorldTracker`.
4. **Pupil head on the canthus net** — worth it for **blink/presence detection and geometry ID**,
   explicitly *not* for registration (§3 says why it cannot help there).

**Known-unverified, do not treat as fact:**
- Display FOV ~50° (assumed), eyebox size and eye relief (unmeasured), our actual position within
  the eyebox, and whether the One Pro's follow mode is rigid or damped.
