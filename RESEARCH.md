# State of the art, how others approach this, and what we should take

A survey of the relevant literature and products, mapped to *our* project (auto-calibrating
an optical see-through display using eye tracking). The short version: **what we're building
is a recognized research problem with a 20-year literature, the design choices here mostly
match it, and there are specific, proven techniques we should adopt.** Sources are linked inline.

---

## 1. The problem, in the field's own words

Optical see-through (OST) head-mounted displays "frequently require recalibration due to
movement, and if these calibrations involve user interactions, they are time-consuming,
distract users, inject user-dependent errors, and reduce user acceptance." That is *exactly*
our thesis, the glasses shift, manual calibration drifts, and the fix is to let an eye
tracker do it. ([OST calibration survey, Grubert/Itoh/Moser/Swan, arXiv 1709.04299](https://arxiv.org/pdf/1709.04299))

The registration also depends on **where the eye is** relative to the display ("eye-position
dependent"), which is why a fixed calibration fails when the headset moves, the core
motivation for our eye-corner/eye-tracking features.

**Validation at the highest level:** Apple Vision Pro "is using eye tracking to **correct the
optics** in addition to foveated rendering… the user will be unaware of all the dynamic
corrections being applied." ([VR.org](https://vr.org/articles/apple-vision-pro-wheelchair-control-eye-tracking-accessibility-2026)) That is our goal shipping in a product.

---

## 2. OST display calibration methods (this is our core)

- **SPAAM** (Single-Point Active Alignment Method, Tuceryan & Navab 2002): the user aligns a
  displayed point with a world point many times; solves a pinhole projection. The manual
  baseline. Variants: **DSPAAM** (display-relative), Stereo/Easy/Recycled-SPAAM.
  ([SPAAM](https://www.researchgate.net/publication/245379276_Single-Point_Active_Alignment_Method_SPAAM_for_Optical_See-Through_HMD_Calibration_for_Augmented_Reality))
  → **Our nudge-the-dot-and-approve loop *is* SPAAM**, with a learned regression instead of a
  single projection matrix.
- **INDICA** (Interaction-free Display Calibration, Itoh & Klinker 2014): does SPAAM **once**
  offline, then uses an **eye tracker to measure eye position online** and regenerate the
  projection automatically, no per-session user alignment. ([INDICA](https://ieeexplore.ieee.org/document/6798846/))
  → **This is precisely our end goal** (eye tracking removes the manual step). We should frame
  the project as "learned INDICA."
- **Corneal-Imaging Calibration** (Plopski et al. 2015): displays a pattern, captures its
  **reflection on the cornea**, and computes the eye→display geometry from the reflected rays
 , fully automatic, glint/reflection-based rather than iris-based. ([survey](https://arxiv.org/pdf/1709.04299))
  → A second proven auto-calibration route; our glints carry the same information.
- **Recycled-INDICA / comparisons** (Moser et al. 2015): showed automatic eye-position
  methods can rival manual SPAAM. ([Moser comparison](https://www.researchgate.net/publication/319700292_A_Survey_of_Calibration_Methods_for_Optical_See-Through_Head-Mounted_Displays))

**Takeaway:** our approach is squarely in this lineage. The principled upgrade is to make the
eye-position estimate **3D and slippage-robust** (next sections), which is what INDICA-class
methods rely on.

---

## 3. Gaze estimation: PCCR, 2D regression vs 3D model

The gold standard is **PCCR** (Pupil-Center Corneal-Reflection): IR illuminates the eye, the
**pupil center** and **glints** are found, and gaze is derived. Two schools
([Tobii](https://www.tobii.com/resource-center/learn-articles/how-do-eye-trackers-work)):

- **2D regression:** map the **pupil→glint vector** to screen coordinates with a polynomial.
  Simple, but "vulnerable to head movements… accuracy depends on people keeping their heads
  in the same position."
  → **This is exactly what our `pupil_tracker.features()` + the calibrator do.** Good news:
  it works and is standard. Caveat the field flags: it's slippage-sensitive, which is *the*
  thing our project is about, so we must lean on slippage-robust features/3D models.
- **3D model-based:** fit a geometric eyeball (cornea center, optical & visual axes via
  ≥2 glints), intersect the visual axis with the scene. Achieves **~1°** but needs more
  setup. ([review, arXiv 1708.01817](https://arxiv.org/pdf/1708.01817); General Theory, Guestrin & Eizenman)

**Takeaway:** start with 2D-regression PCCR (we have it), and add a **3D eye-model option**
for slippage robustness and to reach the ~1° regime.

---

## 4. Slippage robustness, the most important lesson for us

The glasses moving on the face is precisely "slippage," and there's a dedicated literature:

- **Świrski & Dodgson (2013):** fit a **3D eye model** from a single near-eye camera that
  **updates per frame**, so it compensates as the headset shifts. ([robust off-axis pupil](https://www.researchgate.net/publication/229084818_Robust_real-time_pupil_tracking_in_highly_off-axis_images))
- **pye3d / Dierkes (Pupil Labs):** adds **corneal refraction correction**: the pupil you
  see is a *refracted* image, so raw measurements are bent; pye3d runs them through a
  refraction-correction function, and the 3D model "compensates for movements of the headset
  on the face (slippage)." ([pye3d docs](https://docs.pupil-labs.com/core/developer/pye3d/))
- **Near-eye-display slippage papers (2022, 2024):** "Slippage-robust Gaze Tracking for
  Near-eye Display" ([arXiv 2210.11637](https://arxiv.org/pdf/2210.11637)) and "Slippage-robust
  linear features for eye tracking" ([2024](https://www.researchgate.net/publication/386081865_Slippage-robust_linear_features_for_eye_tracking)), directly our use case.

**Two concrete things to adopt:**
1. **Model corneal refraction** in our simulator (we lump it into `ep_dist`; pye3d shows it's
   a real, correctable effect). A refraction term improves sim fidelity *and* the eventual
   3D-model fit.
2. **Prefer slippage-robust features:** the pupil→glint vector and multi-glint geometry are
   already more slippage-robust than raw pupil position, keep ≥2 glints and use their
   geometry, not just the pupil center.

---

## 5. Robust pupil/iris detection (our classic detector's upgrade path)

Real off-axis eyes break naive thresholding (eyelids, lashes, makeup, reflections). The
field moved to deep segmentation:

- **EllSeg** (2020): segment the pupil/iris as **ellipses directly**, robust to occlusion by
  "eyelid shape, camera position, or eyelashes," validated on NVGaze and RIT-Eyes. ([EllSeg](https://arxiv.org/pdf/2007.09600))
- **RITnet** (U-Net+DenseNet): **95.3%** on the OpenEDS 2019 challenge, **>300 Hz** on a 1080 Ti.
  ([survey of DL eye tracking](https://arxiv.org/html/2403.19768))
- **DeepVOG**: open-source FCN pupil segmentation + gaze. ([DeepVOG](https://github.com/pydsgz/DeepVOG))

**Takeaway:** our threshold+ellipse tracker is fine for clean, well-lit IR images and a first
build, but **real under-eye (off-axis) NIR pupil images will need EllSeg/RITnet-class segmentation.**
That requires data + a small GPU (Jetson), a defined upgrade. *Before then*, we harden the
classic detector (glint-in-pupil handling, robust ellipse rejection) and test it on **off-axis
+ occluded** synthetic eyes, implemented in `software/pupil_tracker.py`.

---

## 6. Synthetic data & sim-to-real, validates *and* critiques our approach

This is the part that most directly judges our whole simulation strategy:

- **NVGaze** (NVIDIA 2019): a **2-million-image anatomically-informed *synthetic*** dataset
  (varying face shape, gaze, pupil/iris, skin tone, lighting); a net trained on it hit
  **2.06° on real subjects** held out from training. ([NVGaze](https://research.nvidia.com/publication/2019-05_nvgaze-anatomically-informed-dataset-low-latency-near-eye-gaze-estimation))
  → **Strong evidence our anatomically-grounded synthetic approach can transfer**: *if* it's
  anatomically informed and domain-randomized, which `autosim`/`anatomy.py` already are.
- **RIT-Eyes** (2020): a renderer for photorealistic near-eye images for training. ([RIT-Eyes](https://arxiv.org/pdf/2006.03642))
- **OpenEDS** (Meta): 550k+ gaze-labeled real images, but **on-axis**; "off-axis data is
  essential for realistic VR… absent from on-axis datasets." ([OpenEDS](https://arxiv.org/pdf/1905.03702))
  → Our under-eye NIR pupil cameras are **off-axis**, the harder/realistic case, we must test there.
- **SimGAN** (Apple 2017): refine synthetic eyes with adversarial training to close the
  sim-to-real gap. ([SimGAN](https://arxiv.org/pdf/1612.07828))
  → The path to make our synthetic eye images real enough to train an image-level net later.

**Takeaways:** (1) our synthetic-first strategy is **endorsed by NVGaze**; (2) the field works
at the **image** level (train CNNs), while we work at the **feature** level, fine for the
calibration regression, but to train a robust *pupil detector* we'd want rendered images
(RIT-Eyes-style) + domain randomization + SimGAN refinement; (3) **always validate off-axis.**

---

## 7. The high-speed frontier, event cameras

For very-high-speed tracking, the genuine SOTA is **event (neuromorphic) cameras**:
asynchronous per-pixel brightness-change sensors with microsecond latency.

- **10,000 Hz** update rates, micro-saccade capture; **Retina** runs end-to-end at **2.89-4.8 mW,
  5.6-8 ms latency** on a neuromorphic chip. ([Retina, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Bonazzi_Retina__Low-Power_Eye_Tracking_with_Event_Camera_and_Spiking_CVPRW_2024_paper.pdf); [EyeTrAES](https://arxiv.org/html/2409.18813))
- **Commercial:** 7invensun's **aSee-EVS** uses Prophesee's **GenX320** event sensor for
  ultra-low-latency wearable gaze. ([Prophesee](https://www.prophesee.ai/event-based-vision-eye-tracking/))

**Takeaway:** a global-shutter OV9281 at 120-200 fps is the right *buildable* choice now;
**event cameras are the aspirational upgrade** if you ever need true sub-ms / micro-saccade
tracking. Note them in the roadmap, don't block the first build on them.

---

## 8. Commercial benchmarks (our targets)

- **Meta Quest Pro:** ~**1.08° accuracy at 90 Hz**, video-oculography + ML, drives foveated
  rendering + biometrics. ([Quest Pro eval, arXiv 2403.07210](https://arxiv.org/pdf/2403.07210))
- **Apple Vision Pro:** best-shipping consumer eye tracking, **per-eye calibration on first
  don**, ms-latency, **optics correction via gaze.** ([AVP usability study](https://www.researchgate.net/publication/381121940_Measuring_eye-tracking_accuracy_and_its_impact_on_usability_in_Apple_Vision_Pro))

**Target:** ~1° gaze accuracy is the bar; AVP's "calibrate once on don, then auto-correct" is
the UX to emulate (which is exactly the warm-start-prior → quick per-user calibration we
prototyped).

---

## 9. What this means for OUR project

**We already do right (confirmed by the literature):**
- Auto-calibration of an OST display via eye sensing (INDICA / CIC lineage).
- Nudge-and-approve = SPAAM; learned regression generalizes it.
- Pupil→glint (PCCR 2D-regression) features.
- Anatomically-grounded synthetic data (NVGaze-validated) with domain variation.
- Warm-start prior → fast per-user calibration (mirrors AVP's "calibrate on don").

**Adopt now (pre-hardware, doable in software):**
1. **Model corneal refraction** in the simulator (pye3d lesson), small fidelity add.
2. **Harden the pupil detector** (glint-in-pupil handling, robust-ellipse rejection) and
   **test off-axis + occluded** (NVGaze/EllSeg lesson). *(done in `pupil_tracker.py`)*
3. **Keep multi-glint geometry** for slippage robustness, not just the pupil center.
4. **Frame the math as INDICA/SPAAM** so it's legible to anyone in the field.

**Adopt when hardware/compute arrives:**
5. **Deep ellipse segmentation** (EllSeg/RITnet) for real off-axis eyes, needs data + Jetson.
6. **3D model-based gaze** (Świrski/pye3d) for ~1° accuracy + slippage robustness.
7. **RIT-Eyes-style rendered images + SimGAN** if we train an image-level detector.

**Aspirational:**
8. **Event-camera eye tracking** for true high-speed / micro-saccades.

**Honest gaps the literature confirms we can't shortcut:**
- Off-axis real-eye robustness (only real data settles it).
- The kappa / perceptual-bias confound (intrinsic; commercial trackers calibrate per user
  too, they don't *solve* it, they measure around it).
- Sim-to-real for the *pupil detector* (NVGaze still needed real held-out subjects to report
  2.06°).

---

## Sources
OST calibration survey [1](https://arxiv.org/pdf/1709.04299) · SPAAM [2](https://www.researchgate.net/publication/245379276_Single-Point_Active_Alignment_Method_SPAAM_for_Optical_See-Through_HMD_Calibration_for_Augmented_Reality) · INDICA [3](https://ieeexplore.ieee.org/document/6798846/) · PCCR/Tobii [4](https://www.tobii.com/resource-center/learn-articles/how-do-eye-trackers-work) · gaze review [5](https://arxiv.org/pdf/1708.01817) · pye3d/refraction [6](https://docs.pupil-labs.com/core/developer/pye3d/) · off-axis pupil [7](https://www.researchgate.net/publication/229084818_Robust_real-time_pupil_tracking_in_highly_off-axis_images) · slippage near-eye [8](https://arxiv.org/pdf/2210.11637) · EllSeg [9](https://arxiv.org/pdf/2007.09600) · DL eye-tracking survey [10](https://arxiv.org/html/2403.19768) · DeepVOG [11](https://github.com/pydsgz/DeepVOG) · NVGaze [12](https://research.nvidia.com/publication/2019-05_nvgaze-anatomically-informed-dataset-low-latency-near-eye-gaze-estimation) · RIT-Eyes [13](https://arxiv.org/pdf/2006.03642) · OpenEDS [14](https://arxiv.org/pdf/1905.03702) · SimGAN [15](https://arxiv.org/pdf/1612.07828) · Retina event [16](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Bonazzi_Retina__Low-Power_Eye_Tracking_with_Event_Camera_and_Spiking_CVPRW_2024_paper.pdf) · EyeTrAES [17](https://arxiv.org/html/2409.18813) · Prophesee [18](https://www.prophesee.ai/event-based-vision-eye-tracking/) · Quest Pro eval [19](https://arxiv.org/pdf/2403.07210) · AVP study [20](https://www.researchgate.net/publication/381121940_Measuring_eye-tracking_accuracy_and_its_impact_on_usability_in_Apple_Vision_Pro)
