# Automated calibration via an anatomy-grounded simulator

This replaces the human-in-the-loop with a physics simulator and a self-correcting
oracle, runs self-play across thousands of synthetic eye/face geometries, and learns a
**prior** that streamlines calibration for brand-new users. numpy + stdlib only.

```bash
python3 main.py --auto-train 200            # train across 200 subjects + benchmark
```

---

## 1. The anatomy it's built on (the part that matters)

Where an AR pixel must sit to land on a real point is dictated by ocular and facial
anatomy. The simulator (`anatomy.py`) models the quantities that actually drive
registration, sampled from real population statistics:

| Quantity | Value (mean ± SD) | Why it matters |
|---|---|---|
| Interpupillary distance | 63 ± 3.8 mm | lateral eye position; which part of the eyebox is used |
| Globe radius / axial length | 12 ± 0.45 mm | sets the **center of rotation**, ~13.5 mm behind the cornea, the eye rotates about it to fixate |
| Entrance pupil | 10.5 ± 0.7 mm ahead of CoR | the real aperture; the **chief ray** (pupil→point) is the line of sight we must match; it shifts as the eye rotates → **parallax** |
| Angle kappa | 5° ± 1.5° horiz, 1.5° ± 1° vert | visual axis ≠ pupillary axis, varies per person, and is **unobservable from outside** → the irreducible per-user term |
| Inner inter-canthal dist | 32 ± 2.5 mm | inner eye-corner position |
| Outer-canthal dist | 89 ± 4 mm | outer eye-corner position |
| Canthal tilt | 5° ± 2.2° | outer corner sits higher than inner |
| Palpebral fissure | ≈ (OCD−ICD)/2 ≈ 28 mm | eye-opening geometry the inward cameras see |

The canthi (eye corners) are the linchpin: the inward cameras measure them, and their
positions encode **both** how the glasses currently sit relative to the eye **and** the
person's individual eye shape. They are the observable bridge between "glasses pose" and
"which pixel registers."

## 2. The registration physics (`autosim.py`)

The eye fixates point **P**, so P images on the fovea. A see-through AR optic is ~**collimated**:
a display pixel emits a fixed angular direction. To co-register, that direction must equal
the eye's line of sight, the chief ray from the **entrance pupil** to **P**:

```
required_display_direction = unit( P − entrance_pupil )
entrance_pupil = center_of_rotation + ep_dist · gaze,   gaze = unit(P − CoR)
required_pixel = display_optics( angles(required_direction) + angle_kappa )
```

- **P is at finite distance** → the required direction depends on where the eye is → parallax.
- **Glasses slip on the face** → the eye (and canthi) move in the glasses frame → the
  required pixel changes, and the eye-corner cameras see the change. This is exactly why
  one-time calibration drifts and why the canthi are predictive.
- **Angle kappa** is added as a per-subject angular bias the cameras can't see → the part
  no amount of eye-corner data can predict → why each user still needs *some* calibration.

The simulator returns the same base-8 features the real rig measures (2 world + 2 eye-corner cams; the 6-cam binocular build adds pupil/stereo features behind use_pupil/use_stereo flags)
(`[worldL dot xy, worldR dot xy, left-corner xy, right-corner xy]`) plus the oracle's
true pixel. Camera placement comes from `rig.py`, which mirrors the printed bracket, so
the database transfers to hardware.

## 3. Self-play to convergence (`autotrain.py: train_subject`)

For each subject: seat the glasses, then each iteration **slip them slightly** (a
mean-reverting random walk, natural drift), draw a dot, let the calibrator predict the
overlay pixel, have the **oracle report the true pixel**, store that correction, and
refit. Stop when the prediction has been within tolerance for `patience` iterations in a
row → that subject is calibrated. Then regenerate a new subject.

## 4. Cross-subject prior → streamlined onboarding

Every `(features → true pixel)` sample and each subject's geometry descriptor go into a
SQLite **meta-database**. A **global prior** is fit on the pooled data: because the
features include the canthi, the prior already predicts most of the mapping for a new
person from their eye geometry. A new user is then **warm-started**: predict with the
prior + fit a fast affine residual on a handful of their own samples (to absorb their
kappa/individual bias). The benchmark compares samples-to-accuracy, cold vs warm.

```
WarmCalibrator.predict = prior.predict(features) + per_user_affine_residual(features)
```

## 5. Fidelity, what's modeled (to mirror real hardware)

Beyond the anatomy, the simulator includes the real-world effects that make calibration
necessary and that the database must capture (all in `rig.py` / `optics.py` / `autosim.py`):

- **Camera lens distortion + finite sensor resolution** on all cameras (readings are
  barrel-distorted and quantized to the pixel grid).
- **Finite display virtual-image distance** (~2 m, not collimated to infinity), so the
  registered pixel depends on eye position via vergence, like a real birdbath/waveguide.
- **Realistic glasses motion**: a slow slide DOWN the nose (vertical drift + correlated
  pantoscopic tilt) plus occasional **re-seating**, not just white-noise jitter.
- **Correlated anatomy**: a shared face-size factor links IPD / inter-canthal / outer-
  canthal distances, as in real anthropometry.
- **Per-session tracker bias**: the eye-corner template isn't perfectly centered, adding a
  constant per-user offset the calibration must absorb.

It is a faithful geometric/optical mirror, not a literal copy, it does not model skin
deformation, full ray-traced lenses, or per-device measured distortion maps (those would
be the next fidelity steps, ideally fit to a real unit).

## 6. Results (`python3 main.py --auto-train 400`)

- Database: **400 subjects, ~28,700 pooled samples** spanning the real population
  (IPD 54-74 mm, OCD 75-102 mm, etc.); `data/meta.db` ≈ 5.4 MB.
- Every subject converges (cold median ~71 samples to 13 px; range 32-109).
- **Warm-start needs ~3.5× fewer samples** for a brand-new user (median ~18 vs ~66).
  The speedup grows with database size (≈1.9× at 20 subjects → 2.8× at ~25 → 3.5× at 400),
  so a larger run keeps improving onboarding. That is the end goal: the database lets a new
  user skip most of calibration.

## 7. Identifying the eye, and the per-user preset (`identify.py`, `preset.py`)

From a few position-guess answers we infer the eye's internal + external geometry
(`identify.py`), then rebuild that eye as a physics subject, sweep it over the full pose
range, and fit a calibrator, a **preset keyed to that eye shape that is correct at every
glasses position** (`preset.py`), finished with a small per-session affine alignment.

What the 1000-subject run actually shows (honest):

- **Identification (K≈4 answers, held-out users):** angle kappa MAE ~0.1° (~8% of its
  spread) and outer-canthal distance ~1.3 mm (~30%) are *well* recovered; IPD/ICD come
  only via face-size correlation (partial); ep-depth, globe radius, canthal tilt are
  *weak*, and that's physics: precise IPD/pupil depth needs **pupil/iris imaging** and
  inner-canthus geometry needs a **nose-bridge camera**. This sensor-coverage map is the
  substantive finding.
- **Calibration accuracy:** the geometry preset reaches the ~7 px noise floor in ~8
  answers and is correct across the whole pose range, but it **ties** the population
  warm-start (both ≈7-8 px by K=8). With a good population prior + a quick per-session
  affine, per-user geometry identification does **not** add meaningful pixel accuracy.
  Its value is the recovered *parameters themselves* (IPD/kappa for stereo rendering,
  lens/comfort fitting, biometrics), not extra registration accuracy.

  `python3 main.py --identify`   (per-parameter identifiability)
  `python3 main.py --preset`     (preset vs population vs black-box, full-pose error)

## Files
`anatomy.py` (subjects) · `optics.py` (display + distorting cameras) · `rig.py` (shared
bracket geometry, mirrors the CAD) · `autosim.py` (oracle) · `autotrain.py` (self-play +
meta-DB + prior + benchmark) · `identify.py` (few-shot geometry identification) ·
`preset.py` (per-user eye preset + evaluation). The meta-DB lands at `../data/meta.db`
and the prior/identifier are rebuilt from its pooled samples on demand.
