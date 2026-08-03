# New-session handoff prompt

Paste the block below into a fresh session to pick up this project. Everything it references
is in this repo (`~/ar-eye-calibration`) or the persistent memory file.

---

> You're continuing the **AR eye-calibration rig** (`~/ar-eye-calibration`, a git repo): a
> modified **XREAL One Pro** that registers AR-overlay pixels to the real world using eye-corner
> cameras (how the glasses sit on the face) plus two forward world cameras. The hardware is
> **built and working**; this session is **TESTING**, and the goal is a **first real calibration
> run on hardware**.
>
> **END GOAL, stated by Dylan 2026-08-03:** overlay content on **people and cars that are far
> away**. That matters for design decisions — see the far-field note below, because it changes
> which errors are worth chasing.
>
> ## WHAT HAPPENED 2026-08-02/03 — READ THIS FIRST
> The hand-drawn eye-corner template was replaced with a **learned canthus model**. Dylan's
> reason, verbatim: *"i cant box every time."* Template capture was a manual per-session step, and
> a slightly-wrong box produced a template that scored ~1.0 on flat skin and localised nowhere.
>
> **The model works.** Trained on 98 human-clicked seed points; held-out error **14.1 px of 1280**;
> correlation with pupil position **+0.19**, down from **+0.99** for the classical tracker — i.e.
> the landmark is now gaze-independent, which a face-fixed point must be. Runs in **pure numpy**
> (13 ms/frame), so the live loop needs no torch. Use it with `main.py --use-model`.
>
> **The calibration preflight reached `READY TO CALIBRATE ✅` (6/6).** A run launched and the loop
> executed, but **no usable samples were stored yet** — see OPEN below.
>
> New modules, all selftested and in `verify_all`: `canthus_data` (collect corpus),
> `canthus_label` (hand-label seed + `--closed-pass`), `canthus_train` (two-model ensemble, torch,
> runs in `.venv-train`), `canthus_auto` (automated propose/correct/confirm), `canthus_net`
> (numpy runtime + tracker + mount anchor), `rig_view` (live 4-up camera view).
>
> ## THE PATTERN THAT COST THE MOST TIME — internalise this
> Roughly **eight times** across those two days, a check went green for reasons unrelated to what
> it claimed to measure. Every single one was caught by looking at the underlying distribution or
> the actual image, never by the summary number:
> - `bank_bringup` printed "BANK UP ✅" with two cameras at 100% misses
> - `dot_geometry` took `abs()` of disparity, hiding a fully reversed stereo pair
> - `corner lock` scored a featureless template at 0.99 (rivals 0.005 away)
> - `dark_mass()` used a percentile threshold, so it returned ~12% for every image
> - a "corrector" was a constant function and "confirmed" 89% of frames with label std 0.002
> - two independent estimators agreed at 0.029 — on the eyelid, not the tear duct
> - heatmap sharpness worked as a confidence signal on one model and **inverted** on the next
> - the seed-frame selector ranked by an openness metric that did not separate open from closed
>
> **So: when adding a check, write the failing case first. Validate by spread/distribution and by
> LOOKING at images, never by a pass rate.** Several selftests now deliberately pin a known-bad
> input so the property cannot silently regress.
>
> ## RULE ZERO — SAVE EVERYTHING TO THE DOC, AS YOU GO
> **Dylan's standing instruction, verbatim: "from now on, save everything to the doc. if work is
> done without being saved to the doc, it needs to be done again in the next session."**
>
> This is the completion bar for this project. A finding is **not done when you have demonstrated
> it** — it is done when it is **written down and committed**. Nothing survives in a transcript.
>
> - **Save each thing as it lands, not in a batch at the end.** A session that ends early, hits a
>   context limit, or gets interrupted loses everything not yet written. This has already cost
>   real sessions: the 640×480 USB-bus limit was proven once, left in a transcript, and had to be
>   rediscovered — Dylan's words were "we did this last time."
> - **Where it goes:** this file (`HANDOFF.md`) for anything the next session must know up front;
>   the memory file's settled-facts block for "do not re-derive" facts; **and a commit**, always.
> - **Persist the METHOD, not just the answer, whenever the answer is per-session.** Camera
>   indices shift between runs, so "worldL=3" is worthless next time — what matters is *how* to
>   determine it. The strongest form is a **test or a check that fails loudly**, because a doc
>   line can be skipped and a failing check cannot. Prefer that over prose every time.
> - If you are unsure whether something is worth saving: save it. The cost of a redundant
>   paragraph is nothing next to the cost of re-deriving a measurement on hardware.
>
> **Read first, in this order:**
> 1. The memory file `project_ar_eye_calibration.md` — especially the **`SETTLED HARDWARE FACTS —
>    DO NOT RE-DERIVE`** blocks at the end (there are two; read both). Read them before proposing
>    anything physical.
> 2. `HARDWARE_BRINGUP.md`, then `ASSEMBLY.md` §6-8, `WIRING.md`, `SAFETY.md`.
> 3. `git log --oneline | head -20` — the last three commits are this rig's hardware truth.
>
> ## SETTLED — do not re-derive, do not re-litigate
> These each cost a full session to establish more than once. They are encoded in `rig.py` so
> code and memory agree; if you believe one is wrong, say so explicitly and change **both**.
> - **Eye-cam lens is 45°** (measured). `rig.EYE_FOV = 45.0`. Not 90.
> - **Tracked landmark is the INNER canthus.** `rig.TRACKED_LANDMARK = "inner"` is the single
>   source of truth; `autosim.Simulator`, `preset.build_preset`, `match.train_prior` all default
>   to it via `landmark=None`. At 45° the outer canthus is only **41.9%** in-frame vs the inner
>   at **100%** — that is why.
> - **The camera AIM stays "outer"** (`rig.build()` default) because that is what is *physically
>   printed* (outer canthus + `EC_AIM_DOWN=6`/`EC_AIM_OUT=3`). **Aim and tracked landmark are
>   independent by design.** Never "fix" CAD parity by changing the aim.
> - **The carrier is FINAL.** ASA/PETG printed, **no printer access**. Never propose a reprint or
>   a CAD geometry change as a fix. The only physical adjustment left is the **brow-clamp slop**
>   (a mm or two of carrier placement). Inner canthus sits at median u≈0.838, so clamp slop is
>   the lever if more edge margin is wanted.
> - **NoIR/IR-remote test on the eye cams: already passed.** Do not ask for it again.
> - **No IR LEDs are wired, and none are needed for calibration.** The pupil/PCCR path is
>   `use_pupil=False` by default and was measured not to improve registration.
> - **CUT THE RESOLUTION. 640x480, always.** Four streams only run on this one USB 2.0 bus at
>   640x480. This has now cost more than one session — it is not a thing to rediscover. As of
>   2026-08-02 `bank_bringup.py` reads `SyncBank.ROLE_MODE` so the default is finally correct;
>   `--res 640` is the manual override if anything else ever asks for a bigger mode.
> - **L/R is NOT settled by the scan and must be re-checked EVERY session.** `--scan` assigns
>   L/R by ascending index, which is a coin flip. On 2026-08-02 it was right for the eye pair and
>   WRONG for the world pair. Verify both, every time — the method is below.
>
> ## Hardware state + gotchas
> - 4 cameras: 2× ELP AR0234 world (1920×1200) + 2× OV9281 eye (1280×800), one SABRENT hub.
>   XREAL One Pro connects as a second display (1920×1080).
> - **USB indices SHIFT between runs.** Always `python3 bank_bringup.py --scan` first; never
>   reuse indices from a previous session or from this document.
> - **The cameras are USB 2.0 devices sharing one bus.** Three streams at native resolution
>   (~37 fps) is the limit; a 4th starves and — via `sync_capture`'s barrier — stalls the whole
>   bank to ~1 fps. All four run fine at **640×480 (~54 fps)**, which is the default
>   (`SyncBank.ROLE_MODE`). `WIRING.md`'s "one hub carries six" is **wrong** and not yet fixed.
> - **What bus starvation LOOKS like** (measured 2026-08-02, so you can recognise it instantly):
>   two roles at 100% misses with `shape None`, the survivors at ~1 fps, sync jitter ~118 ms.
>   Same rig at 640×480: all four at ~40 fps, zero misses, **6.7 ms** jitter, CPU 4%. If two
>   cameras look "dead", suspect the bus before the hardware — nothing was wrong with either
>   camera or the hub.
> - Full-load profile measured 2026-08-01: track 45.7 ms / sync 19.6 ms / overlay 8.8 ms,
>   73.6 ms per frame, **CPU only 9% of 16 cores**. The loop is **I/O-bound, not CPU-bound**.
>
> ## >>> THIS SESSION: GET A REAL CALIBRATION RUN <<<
> ```bash
> cd ~/ar-eye-calibration/software && source .venv/bin/activate
> python3 bank_bringup.py --scan                       # indices WILL have moved
> python3 main.py --calibrate-corners --eye-cam-left <L> --eye-cam-right <R>
> python3 calib_preflight.py --run --roles worldL=<..> worldR=<..> eyeL=<..> eyeR=<..>
> python3 main.py --world-cam-left <..> --world-cam-right <..> \
>                 --eye-cam-left <..> --eye-cam-right <..> --fullscreen
> ```
> - **Verify L/R before anything else** (see the settled note above — `--scan` guesses it wrong):
>   - **Eye cams — CROSS-MATCH THE TEMPLATES, it is fully automatic.** Once `eyeL.png`/`eyeR.png`
>     exist they are eye-SPECIFIC, so match each template against BOTH mono cameras and take the
>     pairing with the higher total margin (`canthus_data.match_with_margin`). Measured after a
>     replug on 2026-08-02, rig worn: eyeL→cam0 **0.509** vs eyeL→cam1 0.090, eyeR→cam1 **0.358**
>     vs eyeR→cam0 0.165 — correct pairing won by **3.4x**. No cover test, no human judgement, no
>     guessing from mono close-ups (which I tried, and it is genuinely ambiguous by eye).
>     Requires the rig ON A FACE — a template of your eye will not match your living room.
>   - **Eye cams, first time only** (no templates yet): `--calibrate-corners` opens
>     `select eyeL corner` first. If that window is not your LEFT eye, swap the `--eye-cam-*`
>     indices and re-run.
>   - **World cams:** `calib_preflight.py --run` now fails the `world dot` row with an explicit
>     "disparity is NEGATIVE ... pair is REVERSED" message. Swap `--world-cam-left/right` and
>     the row goes green. Do NOT reason about it from images — sensor orientation and a swapped
>     pair produce the *identical* symptom, and a still photo cannot separate them.
> - **Measure the eye cams only while the glasses are WORN.** On the desk they stare at the room
>   and every number is meaningless: 2026-08-02 the same two cams read 40.9%/20.2% of pixels
>   saturated sitting on a table and **5.3%/3.3% once worn**. Same for focus — the M12 lenses are
>   set for a canthus at ~3 cm, so a room 3 m away is *supposed* to look defocused.
> - **Whole-frame Laplacian is not a focus metric for the eye cams.** An eye socket is mostly
>   smooth skin; a living room is wall-to-wall edges. `bank_bringup` will show eye ~15 vs world
>   ~800 on a perfectly focused rig. Judge eye focus by whether lashes/iris texture resolve.
> - **There is NO software exposure control on macOS.** `cameras.py` sets width/height/fps and
>   nothing else, and that is not an oversight: AVFoundation *rejects* `CAP_PROP_EXPOSURE`,
>   `AUTO_EXPOSURE`, `GAIN` and `BRIGHTNESS` on these UVC devices — every `set()` returns False
>   and reads back 0.000 (probed 2026-08-02). Exposure is a physical lever only: wear the rig,
>   kill daylight (NoIR sensors soak up IR, so sunlight hits them far harder than LED light).
> - In `--calibrate-corners`, drag the ROI box around the **INNER** canthus (tear-duct side).
>   Note it captures ONE frozen frame per eye — whatever it happened to grab is what you draw on.
> - **The template must contain STRUCTURE, not just be in the right place.** Box the lid margin,
>   the lash roots and the caruncle — the highest-contrast thing you can see. A box on smooth
>   skin, cheek or shadow produces a template that correlates ~1.0 with *any* other smooth patch:
>   it scores beautifully and localises nowhere. First attempt on 2026-08-02 produced Laplacian
>   variance ~5 on both eyes, scored 0.99/0.97, and had rivals only 0.017/0.005 away — meaning a
>   blink would move the lock somewhere else entirely. `corner lock` now fails this (min margin
>   0.08) instead of passing it, and prints the margin so you can see how much headroom you have.
>   Sanity numbers: a good template margins > 0.3; anything under ~0.1 is not a template.
> - `calib_preflight.py --run` checks all six preconditions and names the failing one. The
>   historically-red row is **`world dot`**: it needs a **dark round dot on white paper**, ~2-3 cm,
>   ~0.5-2 m away, **roughly centred in BOTH world cams**, away from a bright window. Its
>   `dot_geometry()` check rejects a "dot" at an impossible depth or off the epipolar line —
>   two loose cameras each finding their own dark blob otherwise reads green and poisons samples.
> - In the calibration loop: **ENTER** approve · **U** undo last · **Z** cancel nudge · **Q** quit.
>   Expect ~20-40 corrections across the field before the error means anything.
>
> ## THE MOUNT OCCLUDES THE TOP THIRD OF THE EYE CAM — and the sim does not know
> Measured 2026-08-02: the nose-bridge support covers the top **34-36%** of every eye-cam frame
> (`eyeL` floor v=0.340, `eyeR` v=0.365). `rig.py`'s prior places the inner canthus at v ≤ 0.620,
> median 0.362 — **inside that hardware**. `optics.py` has no occlusion model for the carrier, so
> every framing prediction, including the settled *"inner canthus 100% in-frame"*, overstates the
> usable frame. Do not trust a framing number until that is modelled.
> - **Consequence:** any appearance-based search over the simulated box locks onto the mount
>   rather than the eye. That is exactly what the first auto-labelling runs did.
> - **The technique that fixes it, and reuse it elsewhere:** the mount is bolted to the camera, so
>   it lands on the same pixels in every frame whatever the face does — a built-in fiducial. Find
>   it by intersecting **dark** with **temporally static** across frames (measured 4.4x/3.6x more
>   static than the rest of the image), then derive the search band *below* its floor. See
>   `canthus_auto.find_mount` / `search_band`. Keep `rig.py`'s HORIZONTAL prior — u was always
>   consistent with hardware (0.809/0.886 measured vs prior [0.778, 0.891]); only v disagreed.
>
> ## >>> OPEN — what to do next, in order
> 1. **No calibration samples yet.** The loop runs but the **WASD nudge keys were not
>    registering**. The HUD now prints the raw `cv2.waitKey` code: **blank = focus problem**
>    (click the overlay window), **`119 (w)` = the key arrives and the mapping is at fault**. That
>    one line settles it; nobody has read it yet.
> 2. **`eyeR` is the weak eye.** Dylan's `eyeR` seed clicks sit ~0.06 BELOW `rig.py`'s prior band
>    (he labels u≈0.714; band is [0.778, 0.891]), consistently (±0.014). The model faithfully
>    learned that, so the plausibility gate now rejects its own model's output and `eyeR` reports
>    "no plausible landmark". Either his `eyeR` clicks are on a different point than `eyeL`'s, or
>    the sim's geometry for that camera is wrong — as it already was about mount occlusion.
>    Resolve by re-labelling ~20 `eyeR` seeds, or by widening that camera's band.
> 3. **Model is label-limited, not architecture-limited.** Its error (0.030) sits at Dylan's own
>    click scatter (±0.021). More/《better seeds beat any training change. Weakest coverage:
>    extreme gaze and eyes-held-wide (it drifts there), and the closed head (22 examples).
> 4. **Pupil head — free win, no labelling.** Add a third head to the canthus net for pupil
>    centre/ellipse, auto-labelled by `pupil_tracker` (96-100% reliable on real frames, unlike the
>    template). Literature says the big gain is presence/absence detection: 92.8% → 99.6%
>    (EyeNet), which is exactly the weak closed-eye head. Dylan's own `eyetracker.py` study says
>    do NOT expect better registration from it (corner-only 9.9 px vs +pupil 10.1 px) — the payoff
>    is geometry ID (`ep_dist` 81→54%, `globe_r` 87→50%, IPD 57→16%) and blink rejection.
>
> ## FAR-FIELD: the arithmetic that shapes priorities
> Stereo depth dies with distance on a 67 mm baseline — disparity 30.6 px at 2 m, 3.1 px at 20 m,
> 0.6 px at 100 m (unusable). **But you barely need depth out there:** the world cam sits ~34 mm
> from the eye, which at 20 m subtends **0.096°, about 2 display pixels**. At 0.5 m the same
> offset subtends ~3.9°, i.e. 80+ px. **Near is the hard case; far is direction-dominated and
> comparatively easy.** The real far-field risk is calibrating at ONE distance: `kappa.py`
> measured ~13 px of cross-distance error, dropping to 2.4 px with multi-distance calibration. So
> vary target distance during the run.
>
> ## Open bugs (neither blocks calibration)
> - **`WorldTracker` built 0 map points** across a 45 s hardware run — no usable stereo
>   correspondences on real frames. Calibration does **not** touch the mesh (it uses
>   `dot_detector` + `eye_tracker`), but the world-locked overlay depends on it entirely.
>   **2026-08-02, partially explained — do not re-derive this half:** the world pair was
>   REVERSED, and `world_mesh.py:334` filters on `(uL - uR) > 0.5`, so ~99% of matches were
>   being discarded (measured: 1093 epipolar-consistent ORB matches, 99% negative disparity).
>   Fixing the order takes peak inliers **0 → 6** and track ids **1 → 39** — but map points
>   **stay 0**. So the swap was *a* cause, not *the* cause. Next place to look is
>   `WorldMesh.ingest`, not the correspondence filter, which is now known-good.
> - **`WorldTracker` 0 map points is a GEOMETRY problem, not a bug** (researched 2026-08-03).
>   The SLAM literature names this exact configuration: stereo VI-SLAM initialisation degrades
>   badly under **pure/intense rotation** (head motion in glasses is rotation-dominant, almost no
>   translation), and scale observability collapses when **landmarks are far relative to the
>   baseline** — 67 mm against a room, let alone a street. Reversing the world pair took peak
>   inliers 0→6 and track ids 1→39 but map points stayed 0. An IMU is the standard fix for both,
>   and the CAD already has the mount. Stop hunting for a coding bug here.
> - **`eyeL` framing is marginal** (2026-08-02, worn): the frame is mostly nose and cheek with
>   the eye jammed into the bottom-left corner. Both eyes sit near **v≈0.75-0.85**, far below the
>   settled prediction of **v≈0.414**. `EC_AIM_DOWN=6` is printed and cannot change, so the brow
>   clamp is the only lever. Unresolved — a template cut against a frame edge is what makes the
>   tracker lose lock mid-session, so fix framing BEFORE capturing templates.
> - **The accuracy corpus is STALE.** `data/mega_prior*`, `calibration_db.npz`, `meta.db` and the
>   1.88 px `accuracy_map` figure were all generated at `EYE_FOV=90` tracking the **outer**
>   canthus. They no longer describe this rig — regenerate before quoting any accuracy number.
>
> ## Tools added for hardware work
> - `bank_bringup.py` — partial-bank bring-up under a CPU budget: `--scan`, `--run`, `--ramp`
>   (finds how many streams the bus carries), `--roles ROLE=INDEX` override, `--native`,
>   `--no-warm`, `--view`.
> - `calib_preflight.py` — the six calibration preconditions, with the fix printed per failing row.
>   `dot_geometry()` now also checks the disparity **SIGN**. It used to take `abs()`, which made a
>   reversed world pair completely invisible: identical depth, rows still aligned, row reads green.
>   That is how a swapped pair passed a full preflight at 2054 mm on 2026-08-02. Covered by a
>   selftest that asserts both orderings yield the same distance — i.e. why depth cannot catch it.
> - `perf.py` — per-stage `Profiler` + `QualityController` + `LoadManager`. The controller runs an
>   **effectiveness probe**: it reverts and freezes a down-step that doesn't actually buy frame
>   time (this rig is I/O-bound, so degrading quality mostly costs without helping).
> - `cam_view.py` — live single-camera view with a **FOCUS meter + peak-hold** for setting M12
>   lenses (`r` resets the peak).
>
> ## Guardrails (do NOT break)
> - `software/rig.py` is the single source of truth for camera geometry; `cad/xreal_one_mount.scad`
>   mirrors it (parity checked to 0.0003 mm). Don't reorder `config.feature_names`.
> - Every new module needs a `--selftest` hooked into `verify_all.py`. **`python3 verify_all.py
>   --fast` must end "ALL CHECKS PASS"** (currently 33 checks). Run it after touching shared
>   capture/geometry code — and note it takes ~10 min, so run it in the background.
> - **Persist settled decisions** — see **RULE ZERO** at the top of this file, which is the one
>   guardrail that outranks the rest: code (a named constant others default to) + the memory
>   file's settled-facts block + a commit, written as each thing lands. Never leave a decision
>   only in the transcript. If you finish a session with findings unsaved, the session did not
>   happen.
> - Be honest about sim vs hardware. Almost everything is sim-validated; what has actually run on
>   real cameras is listed above, and no calibration has ever completed on hardware.

---

See `media/session-2026-07/` for images from the build/overlay work referenced above.
