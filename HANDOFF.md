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
> - **(9th, 2026-08-03)** three brand-new mount-anchor selftests passed against a tracker that
>   **could not be constructed at all** — they built it with `__new__` and injected the very
>   attribute `__init__` was missing. **A test that constructs its subject differently from
>   production is not testing production.** Inject the *stub collaborator* you need to control;
>   never hand-assemble the object under test.
>
> **So: when adding a check, write the failing case first. Validate by spread/distribution and by
> LOOKING at images, never by a pass rate.** Several selftests now deliberately pin a known-bad
> input so the property cannot silently regress.
>
> **And add the corollary the 9th cost a session to learn: every module whose object runs on the
> rig needs one dumb PRODUCTION-PATH check** that builds it the way the rig does, with real
> weights and a real frame, and asserts only that it does not raise. `canthus_net` has one now.
> It is the cheapest check in the repo and it is the one that would have caught this.
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
> - **XREAL DISPLAY MODE MUST BE HEAD-FOLLOWING (0DoF), NEVER ANCHOR/LOCKED (3DoF).** Asked by
>   Dylan 2026-08-03 while calibrating; he had follow on, which is correct. The calibration learns
>   a fixed map from (eye features + world dot) -> DISPLAY PIXEL, and that map only exists if a
>   pixel corresponds to a FIXED direction relative to the glasses. Follow mode gives exactly that.
>   Anchor mode has the glasses' own IMU shift the image *within* the optics to hold it world-fixed,
>   so the pixel you draw is moved by an amount depending on head pose before it reaches the eye —
>   you would be fitting your rig's geometry AND XREAL's internal stabiliser simultaneously, with
>   the second changing every frame. It is a second hidden calibration fighting yours.
> - **HYPOTHESIS, NOT MEASURED:** if follow mode uses *smoothed* following rather than rigid
>   attachment, the image lags during and just after a fast head turn. That would fit Dylan's
>   report that "slight adjustments work well, big head shifts make it not work". Unverified — the
>   firmware's damping behaviour is unknown. Prefer an instant/no-smoothing follow if one exists.
> - **MIRRORING IS FINE *IF* THE XREAL IS THE MASTER — I GOT THIS BACKWARDS FIRST TIME.**
>   I claimed mirroring scales the 3072x1920 desktop into the glasses and corrupts every sample,
>   and told Dylan to switch to Extended. **Wrong direction.** `system_profiler` on this rig shows
>   the **XREAL is `Main Display: Yes` and `Mirror Status: Master Mirror`**, with the built-in
>   Retina as the `Hardware Mirror` follower, and `UI Looks like: 1920 x 1080`. So the desktop
>   framebuffer IS the glasses' native 1920x1080 and the overlay lands **1:1**; the LAPTOP is the
>   one showing a scaled copy, and nothing reads pixels from there.
>   **What actually matters is not mirror-vs-extend, it is: is the framebuffer the XREAL's native
>   1920x1080, and is the XREAL the master?** Check with:
>   `system_profiler SPDisplaysDataType | grep -E "Resolution|Mirror|Main Display|UI Looks"`
>   If the built-in were master instead, the glasses WOULD get a rescaled image and the samples
>   would encode it. Extended is then a cosmetic preference (it stops the overlay covering the
>   laptop screen), not a correctness requirement.
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
> - **>>> CAPTURE MODES ARE PER-SENSOR. THE TWO SENSORS DISAGREE. `cameras.ROLE_MODE` is the table.**
>   Measured 2026-08-03 by asking each camera for each mode and comparing what came back:
>   | sensor | use | why |
>   |---|---|---|
>   | **OV9281 eye** (1280x800) | **640x400** | true downscale, FULL FOV. **640x480 is a CROP** (~30% of sensor) |
>   | **AR0234 world** (1920x1200) | **640x480** | true downscale, FULL FOV. **640x400 and 1280x800 are IGNORED -> native 1920x1200** |
>   **AN UNSUPPORTED MODE DOES NOT FAIL — UVC SILENTLY RETURNS NATIVE.** Four natives on one
>   USB 2.0 bus starves half the bank, and the only symptom Dylan saw was *"only 2 cams are
>   running"*. Nothing said the request had been ignored. `Camera.__init__` now compares requested
>   vs actual and PRINTS the discrepancy with the pixel-count multiplier.
>   **Do NOT derive the height from an aspect rule.** Doing so broke this twice in one session:
>   `3//4` gave the eye cams a crop, `10//16` made the world cams run native and starve the bank.
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
> ## >>> STATE AS OF 2026-08-03 SESSION 2, END — READ THIS BEFORE THE OLDER BLOCKS <<<
> The rig ran on a face this session. Two software blockers were fixed, three suspects were killed
> by measurement, and **a fresh 3000-frame worn corpus is captured and waiting to be labelled.**
> The next action is labelling, not hardware.
>
> **THE CURRENT DIAGNOSIS, which supersedes the older "framing" story below.** Dylan pushed back
> with *"the canthus is fully visible, why doesn't it work?"* — and he was right. Drawing the
> model's prediction on worn frames settled it:
> - **The canthus IS fully in frame on both eyes.** The framing story was overstated.
> - **The model is not landing on it.** On `eyeR` it sits on the upper lid crease, ~0.12 of frame
>   width up-and-inboard of the real corner; on `eyeL` it sits on skin beside the corner.
> - **So the gate is doing its job** — it is correctly refusing a wrong landmark. That kills the
>   "widen the band" idea outright: widening it would accept a point on an eyelid. The previous
>   handoff advised exactly that, and it was wrong.
> - **The size of the miss is the tell:** held-out error on the seed frames was 14.1 px of 1280
>   (~0.011), but the live miss is **~0.12-0.13 of frame width, ten times larger**. That is not
>   model capacity, it is **distribution shift** — trained at one seating, run at another.
> - Whether the band is ALSO wrong cannot be separated from this until the model hits the corner.
>   Do not touch the band until after a retrain.
>
> **What is measured and settled this session (do not re-derive):**
> - **Exposure is solved**: 0.7%/0.7% saturated worn. Not a variable any more.
> - **Focus is fine**: lashes and iris texture resolve. (Laplacian says 14/13 — it is useless here.)
> - **`eyeR` framing is GOOD** — leave that boom alone. It reached 48% lock, up from 6%.
> - **`eyeL` is the mis-seated one** (0.815 labelled -> 0.675 -> 0.607 across the session). One
>   carrier adjustment improved `eyeR` and worsened `eyeL`, so **the booms must move
>   independently** — treating it as one carrier tweak is what traded one eye for the other.
> - **Seating is NOT repeatable**: the mount is held by **ZIP TIES**, not the ASSEMBLY.md M3
>   thumbscrews. Adjustment is cut/reposition/re-tie — coarse and different every time. That is an
>   argument for making the band **mount-relative** (the mount is bolted to the camera, so it
>   cancels seating) rather than sim-derived. Dylan's own `12d2c13` fiducial, one step further.
>
> **>>> RETRAINED 2026-08-03 — THE MODEL IS FIXED OFFLINE. Confirm it on hardware next.**
> Dylan labelled 87 positioned + 35 closed on the new worn corpus; retrained and exported.
> Measured against his 87 ground-truth clicks:
>
> | model | eyeL median | eyeR median |
> |-------|-------------|-------------|
> | OLD (pre-2026-08-03) | 0.175 (**224 px** of 1280) | 0.159 (**203 px**) |
> | NEW (trained today)  | 0.018 (**23 px**)          | 0.021 (**27 px**)  |
>
> The old model's 0.17 offline miss matches the ~0.12-0.13 live miss measured on hardware earlier
> the same day — the same failure quantified two independent ways, which is what makes this a
> diagnosis rather than a story.
> - **HONEST CAVEAT:** that A/B is partly **in-sample** (the new model trained on those labels).
>   The fair generalisation figure is the **held-out median 0.0194/0.0197 (~25 px)**. That is NOT
>   comparable to the old model's 14.1 px held-out either: the old number came from a
>   low-diversity corpus whose held-out frames were near-duplicates of its training frames. The new
>   corpus is gaze-diverse and genuinely harder, so a larger held-out error is not a regression.
> - **Gate status, which predicts whether the tracker locks:** band [0.698, 0.971] — `eyeL` median
>   **0.826, 100% in band, margin +0.128**; `eyeR` median **0.741, 100% in band, margin +0.043**.
>   `eyeR`'s margin nearly doubled (+0.023) and is still the thinner one, which is *correct* — see
>   the asymmetry below.
> - **>>> NOT YET CONFIRMED ON HARDWARE.** Every number is offline against saved frames. Next
>   action: wear the rig, `python3 eye_check.py`, expect the lock rate to jump from `eyeL` 0% /
>   `eyeR` 48%. Only then re-run `calib_preflight` and go for the calibration.
>
> **>>> THE 0.09 EYE ASYMMETRY IS REAL GEOMETRY — old open item 2 is CLOSED.** Two independent
> labelling passes, different corpora, different seatings, fresh clicks:
> `OLD seed eyeL 0.810 / eyeR 0.707` (gap **0.102**) vs `NEW seed eyeL 0.812 / eyeR 0.728` (gap
> **0.084**). Reproducing to within 0.02 rules out Dylan's clicking, which is how the previous
> handoff framed it. **`rig.py`'s SYMMETRIC prior does not describe this rig**, and that is also
> why `eyeR` has been chronically marginal: the band assumes both eyes land in the same place.
>
> **`data/*.npz` is GITIGNORED** — models, corpora and seeds live only on disk, so the `.bak` files
> are their only version history. Do not delete them: `canthus_net_pre20260803.npz.bak`,
> `canthus_models_pre20260803.pt.bak`, `canthus_corpus_20260802.npz`,
> `canthus_seed_MIXED_20260803.npz.bak`.
>
> **>>> (done) label the new corpus.** `data/canthus_corpus.npz` = **3000 frames, 1500/eye,
> 1280x800, confirmed worn across all three thirds of the run**, captured with the new `--guided`
> phase prompts. Diversity vs the ungTuided first attempt: `eyeL` u sd 0.008 -> **0.015**, range
> 0.048 -> **0.069**, closed 2% -> **15%**; `eyeR` closed 42% -> **13%**. Then:
> ```bash
> python3 canthus_label.py          # ~20-30 clicks on the inner canthus
> python3 canthus_train.py --train  # torch, in .venv-train
> ```
> Backups: `data/canthus_corpus_20260802.npz` is the original 656 MB corpus, untouched.
>
> ## STATE AS OF 2026-08-03 SESSION 2 (bugfix session, no hardware touched)
> Both things that stopped the last run are fixed and covered by checks; `verify_all --fast` is
> green at **38 checks, 0 failures** (37 + the new `calib-loop input plumbing`). Note the previous
> handoff's "33 checks" was stale — count it, don't quote it.
>
> **AND THE MOST IMPORTANT LINE IN THIS FILE:** that same 38-check release gate printed
> **`ALL CHECKS PASS ✅` against code where `main.py --use-model` crashed on frame one of both eye
> cameras.** Measured this session — the pre-fix suite was run deliberately to see it. A full green
> release gate is not evidence the rig works. Only running the rig is.
>
> **The anchor fix IS verified on real hardware** (2026-08-03): all four cameras enumerate, and
> `CanthusTracker(mirrored=...).track()` ran 20 live frames from each eye cam at ~26 ms/frame with
> no `AttributeError`. That is the crash reproduced-then-cleared on the actual rig, not in sim.
> **Everything else here is still code-path + negative-control only, and no calibration has run.**
> Scan of this session (indices WILL move): `worldL=2 worldR=3 eyeL=0 eyeR=1`, plus the built-in
> FaceTime cam at 4 and an iPhone at 5 — the extras are why `--scan` needs checking, not trusting.
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
> 1. ~~**No calibration samples yet / WASD not registering.**~~ **SOLVED 2026-08-03 (session 2),
>    two independent bugs, neither of them focus and neither of them the key mapping.** Do not
>    re-investigate; do go and run the rig.
>    - **(a) THE BLOCKER: `CanthusTracker.__init__` never created `self.anchor`.** The mount-anchor
>      commit `12d2c13` (07:41) added `self.anchor` / `self._warm` / `self.flexed` to `update()`
>      and set them **only inside the selftest**, which built the tracker with
>      `CanthusTracker.__new__(...)` and assigned every attribute by hand. `git log -S` confirms
>      the line `self.anchor = MountAnchor()` had **never existed in any commit**. So
>      `main.py --use-model` raised `AttributeError` on **frame one of both eye cameras**, while
>      all of `canthus_net`'s checks — including the three new anchor ones — stayed green.
>      Fixed in `__init__`; the selftests now build through `__init__` and only the stub *net* is
>      injected; a new **PRODUCTION PATH** check constructs the tracker exactly as `main.py` does
>      and pushes 15 real frames through `track()`. Verified to go red against the old code.
>    - **(b) The loop threw every keypress away when nothing was tracked.** On a frame where any
>      input was missing, `main.py` polled the InputController **for `.quit` only** and then
>      `continue`d — before the nudge accumulator and before the approve branch. So WASD *and*
>      ENTER were consumed and dropped, which is exactly "keys don't work" plus "no samples
>      stored". Worse, **the raw-keycode HUD added to diagnose it was only in the live branch**,
>      so it read `none yet` forever and pointed at a focus problem that did not exist. The loop
>      now has a single input path with a `live` flag: nudge/undo/reset/quit always work, only
>      APPROVE is gated, and the HUD says `NOT LIVE -- <reason>`.
>    - **THE TIMELINE MATTERS:** `12d2c13` landed at **07:41** and the handoff was written at
>      **07:42**. Every observation in the previous handoff — the `6/6 READY TO CALIBRATE ✅`, the
>      `eyeR` "no plausible landmark" — was made **before** that commit. The rig had been broken
>      for one minute when the notes describing it were written, and nobody ran it again.
> 2. **`eyeR` is the weak eye — but the previous handoff overstated it. MEASURED 2026-08-03.**
>    It is **not** being rejected. Running the real exported model over all 98 seed frames:
>
>    | eye | model u as the gate sees it | gate pass rate | margin above lower cutoff |
>    |-----|-----------------------------|----------------|---------------------------|
>    | eyeL (mirrored) | med 0.815, sd 0.023 | **49/49 = 100%** | **+0.117** |
>    | eyeR | med 0.721, sd 0.028 | **49/49 = 100%** | **+0.023** |
>
>    The gate is `[u_lo - pad, u_hi + pad]` = **[0.698, 0.971]** (`pad=0.08`), so `eyeR`'s median
>    clears it — but by **0.023, against a model sd of 0.028 and a held-out error of ~0.030**.
>    `eyeR` is **marginal, not broken**: it sits under one sigma from the cliff, so it will start
>    dropping frames on live data that is even slightly off-distribution, intermittently, in a way
>    that looks like flakiness rather than a gate. `eyeL` has 5x the headroom.
>    **Do NOT re-label 20 seeds on the strength of the old note** — the "no plausible landmark"
>    report came from `calib_preflight`, and note that with the anchor bug that same code path
>    returns `"model failed on <role>: ..."`, not `"no plausible landmark"`, so that observation is
>    from the pre-`12d2c13` code and describes a real but *different* condition.
>    The open question is unchanged and still worth answering: **why do the two eyes differ by
>    0.09 in mirrored-u** (0.815 vs 0.721) when the carrier is symmetric? Either Dylan clicks a
>    different point on `eyeR`, or the sim's geometry for that camera is wrong — as it already was
>    about mount occlusion. Widening the band hides it; measuring it does not.
>
>    **Weak corroboration, and read the caveat first.** With the rig **ON THE DESK** (so per the
>    settled rule these numbers are meaningless as anatomy — the cams stare at a room and the M12
>    lenses are set for ~3 cm), 15 live frames per eye gave: `eyeL` raw u 0.392 → mirrored 0.608 →
>    **0% in band, all frames `lost`**; `eyeR` raw u 0.769 → **100% in band, all frames `ok`**.
>    The interesting half is the second one: **the gate ACCEPTED a garbage landmark off a picture
>    of a room for `eyeR`, and correctly refused one for `eyeL`.** That is what a band sitting too
>    loose around `eyeR` looks like, and it is the same weakness the seed-frame margin shows
>    (+0.023 vs +0.117) seen from the other side. It is one desk measurement, not proof — but if
>    you re-measure the band, this says the fix is likely to *tighten* `eyeR`, not widen it.
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
> ## >>> READ `CAPABILITIES.md` — the rig/XREAL capability inventory (added 2026-08-03)
> Written because capability was scattered across 78 modules, `rig.py` and several sessions of
> measurements, so it was genuinely hard to know what the rig already does. Every figure is either
> measured on this hardware or read from `rig.py`, and anything **assumed** is labelled as such
> (display FOV ~50°, "no world cams on the One Pro", follow-mode damping). It also lists the
> **UNDER-USED RESOURCES**, which is the point: the unwired IMU (solves three problems at once),
> the world cams as a depth sensor, the 70% of the eye sensor thrown away by 640×480, the free
> pupil head, the mount as a fiducial, and global shutter on all four cameras.
>
> **`software/depth.py` (new)** — the world pair as a METRIC DEPTH SENSOR with uncertainty, which
> is the one capability the glasses fundamentally cannot have. `python3 depth.py --table` prints
> the working envelope; `--selftest` is in `verify_all` (now **40 checks**). Refusals are
> first-class: epipolar violation, **negative disparity** (reversed pair — never `abs()`'d), and
> disparity below the noise floor each return a named cause rather than a confident number.
> `calib_preflight` now reports the dot's depth **±σ and the parallax cost per 10 cm of head
> motion**, instead of a bare millimetre figure.
>
> ## >>> READ `DOMAIN_REFERENCE.md` — eye anatomy / tracking / optics, and what each implies here
> Researched 2026-08-03. Three items change how to read our own past measurements:
> - **THE CANTHUS APPROACH IS THE LITERATURE'S SLIPPAGE ANSWER, NOT A WORKAROUND.** 3D
>   model-based eye trackers compensate for headset slip; **2D video-based ones have NO known
>   solution**, and measured slippage costs **0.8-3.1° of gaze error**. We track a face-fixed
>   landmark to recover the glasses' pose — that IS the slippage term. With a **zip-tied,
>   non-repeatable mount** this is essential rather than optional.
> - **THE 9.9 vs 10.1 px PUPIL RESULT IS STRUCTURAL, NOT A NULL FINDING.** Rendering needs a
>   projection centre: the **entrance pupil** gives the best ANGULAR accuracy but **moves ~10 mm
>   with gaze** (the eye rotates about a point ~10 mm behind the pupil); the **centre of rotation**
>   is gaze-INDEPENDENT. A canthus can only give the gaze-independent one, so pupil information has
>   nowhere to go in our formulation. It cost ~0 to omit — and at 20 m the two viewpoints differ by
>   **0.003°**, so **the choice is well matched to the far-field goal.** It would cost ~44 px at 0.5 m.
> - **DISPLAY FOV IS NOT A CONSTANT.** "Pupil swim" means the angle→pixel mapping varies with where
>   the eye sits in the eyebox, in **every** near-eye display. So `geometry.DISPLAY_FOV_DEG = 50.0`
>   is both unmeasured AND unrepresentable as one scalar. **The saving grace is architectural:** the
>   learned residual takes the EYE-CORNER features as input, which encode exactly the eye-position
>   variable pupil swim depends on — so geometry+residual can learn a position-dependent correction
>   that no fixed constant could. That is a stronger argument for the split than the one we adopted
>   it for.
> - **`display_calib.py` is written, has NEVER been run, and measures real FOV + K1/K2.** It is the
>   highest-value unrun tool in the repo and feeds the second-largest unmodelled error term.
> - Also settled: angle kappa (visual vs optical axis) is **~5° horizontal, ~1.5° vertical** and
>   **varies per person**; the inner canthus is skull-fixed soft tissue, which is why
>   `SOFT_TISSUE_SD=0.15mm` is a "holding still" placeholder that squinting/talking exceeds.
>
> ## >>> READ `LATENCY_AND_TRACKING.md` — WE HAVE BEEN USING THE WRONG CLASS OF TOOL
> Researched 2026-08-03 after a session spent fighting overlay lag and jitter by tuning filters.
> **A filter can only trade lag against jitter. The field's answer is PREDICT then REPROJECT**,
> which removes lag *without* adding jitter because it adds information instead of smoothing it.
> - **Holloway (Presence 6, 1997)**: in an OST-HMD error budget, **system delay is the LARGEST
>   single registration error** — bigger than tracker noise, calibration or distortion — because
>   every other term is roughly constant while the latency term scales with **head angular
>   velocity**. Stand still and lag is invisible; turn and it is the only error you see.
>   **So calibration accuracy is not the bottleneck while latency dominates.**
> - **Modern XR targets < 20 ms motion-to-photon. We measure 62 ms** (was 196 ms). ~3x over, and
>   **no filter tuning closes it** — 56 ms of that is capture-and-process, which no causal filter
>   can remove.
> - **THE FIX, in order of value:** (1) **predict forward by the measured latency** — we know it is
>   62 ms, head motion is predictable at that horizon, and `fraunhoferhhi/pred6dof` is a readable
>   Kalman reference evaluated at 20-100 ms look-ahead; (2) **late-stage reprojection off the
>   XREAL's own IMU** at 100-1000 Hz against our 17.9 fps loop; (3) **then camera speed stops
>   mattering** — do NOT optimise the vision pipeline before doing 1 and 2.
> - Same IMU the project already concluded is the fix for `WorldTracker`'s 0 map points. The
>   latency work and the SLAM work want the same component.
> - **Expected failure mode of prediction:** overshoot at motion onset and reversal. It will feel
>   worse at turn reversals before it feels better overall; the recent literature is about
>   detecting unpredictable motion and shortening the look-ahead then.
>
> ## >>> GEOMETRY IS NOW THE BACKBONE, LEARNING IS A RESIDUAL (2026-08-03) — `software/geometry.py`
> Dylan, watching a trained model throw the dot across the display on the smallest head movement:
> *"use distance and angle of rotation to calculate how much the dot has to move in the opposite
> direction. this must run perfectly before anything else can be done."* Correct, and now done.
>
> **WHY IT FLEW OFF.** `config.poly_degree = 2` over 8 features is a **45-term quadratic**, taking
> over at `min_samples_for_model = 6`. Fitted to samples spanning ~0.02 of frame — which is what
> the real sets spanned — it interpolates the cluster and **extrapolates violently outside it**. A
> polynomial does not know a display pixel is an angle. Wrong prior, not bad tuning.
>
> **MEASURED, in the regime that actually failed** (train on a clustered set, test across the whole
> field — the numbers that matter are the 95th percentile, which IS "flies off"):
>
> | samples | polynomial replaces geometry | **geometry + residual** |
> |---|---|---|
> | 8  | median 383 px, **95th 999 px** | median **122 px**, 95th **319 px** |
> | 15 | median 349 px, **95th 1037 px** | median **172 px**, 95th **352 px** |
> | 35 | median 212 px, 95th 660 px | median 202 px, 95th **399 px** |
>
> A 95th percentile of 999 px on a 1080-tall display is off-screen. **~3x better at low sample
> counts**, which is where every session begins. On a full 240-sample well-spread set the two are
> tied (14.4 vs 13.9 px) — the gain is entirely in the sparse/clustered/extrapolating regime.
>
> - `geometry.geometric_pixel(features, depth_mm=None)` — closed form, **no fitted parameters**:
>   direction from the world pair (**tangent**, not fraction-of-frame: the old bootstrap used
>   70/50 = 1.400 where the correct factor is tan35/tan25 = **1.502**), then a **parallax** term
>   using stereo depth for the ~34 mm eye-to-camera offset. Gain is bounded at ~1.5 by optics.
> - `Calibrator.predict` = geometry + learned residual, residual **clipped to ±0.25** frame.
>   `Calibrator.fit` trains on `Y - geometry` so the two halves cannot double-count.
> - **A SIGN BUG THE ORACLE CAUGHT:** `rig.py` uses **+y UP**, image space uses **+y DOWN**. Mixing
>   them put a constant **+0.353 frame (381 px)** bias in v. Correlation was already +0.999 in u and
>   +0.973 in v — right shape, wrong offset, the signature of a flipped axis. Fixing it took the
>   geometric error from **396 px to 72 px** against the simulator oracle. **Test new geometry
>   against `autosim` before trusting it on hardware** — it found this in one run.
>
> ## >>> ASYMMETRIC LIGHTING: SOLVED BY NORMALISATION, NOT BY RETRAINING (2026-08-03)
> Dylan: *"the light in my house is to my left. it is not a rig problem... should we run a low
> light scan and use it to do more training?"* **No — it was free.**
> **ROOT CAUSE:** the corpus was captured in a brightness band of **139-161** (median 161). A live
> frame at mean 49 is entirely outside everything the model has ever seen, so it mis-locates.
> **MEASURED against the 87 ground-truth clicks, median error px of 1280:**
>
> | normaliser | normal | dark x0.5 | dark x0.34 |
> |---|---|---|---|
> | none | 25 px | **206 px** | **715 px** |
> | mean | 26 px | 25 px | 23 px |
> | **MEDIAN** | **21 px** | **21 px** | **21 px** |
>
> Median-normalising is **flat across a 3x brightness range and beats doing nothing even at full
> brightness.** On by default in `CanthusNet.predict`, target `TRAIN_MEDIAN = 161`.
> **WHY MEDIAN, NOT MEAN:** a blown-out window in one corner drags the MEAN up with pixels carrying
> no information, so mean-normalising under-corrects the dark region that matters. Measured on
> `eyeL`, whose live frame had **mean 39 but median 11**.
>
> **THE HONEST LIMIT — and the part light must still fix.** Normalisation corrects a LEVEL, not a
> GRADIENT. `eyeL` reached **median 11 of 255**, essentially black, and neither a normaliser nor
> more training can recover information that was never captured. CLAHE was *worse* than plain gain
> (0.477/0.423 vs 0.695/0.858), so local contrast is not a free win. `eyeL` also fluctuated
> **mean 94 -> 39 within minutes**, so its exposure is unstable rather than merely low.
> **Any future corpus should deliberately SPAN lighting** — the 139-161 narrowness is what made the
> model brittle to begin with.
>
> ## >>> THE IMU NOW DRIVES THE PREDICTOR — axis map MEASURED, R^2 0.99 (2026-08-03)
> `data/imu_map.npz`, measured by `python3 xreal_imu.py --map`. Result:
> ```
> du <- wy (YAW)    sign +1   |corr| 0.99   fitted/theoretical 0.97   R^2 0.988
> dv <- wx (PITCH)  sign -1   |corr| 0.99   fitted/theoretical 0.73   R^2 0.853
> ```
> Yaw drives horizontal, pitch drives vertical — physically correct — and the **fitted scale
> matches the FOV prediction to 3% on u, which independently confirms `WORLD_FOV = 70°`.** The v
> ratio of 0.73 is not error: the 16:10 sensor's VERTICAL FOV predicts 0.68, so the aspect-corrected
> theory matches and the naive one does not. That is a second, independent confirmation.
>
> **`main.py` now uses the IMU as the predictor's velocity source automatically**, and falls back
> to image motion if the glasses are unplugged or no usable map exists. `load_map()` **REFUSES a
> map with R^2 < 0.3** rather than letting a bad one predict backwards.
>
> **THREE THINGS THAT MADE THIS WORK, each after a failed attempt — do not undo them:**
> 1. **OPTICAL FLOW, NOT THE DOT.** The dot is one small target competing with a whole room; when
>    it mis-detects it jumps to an unrelated object. Across three runs the fitted axis assignment
>    FLIPPED between yaw and pitch at |corr| 0.13 — a fit converging on nothing. Head rotation moves
>    the ENTIRE SCENE, so median sparse flow over ~300 corners is vastly more robust, needs no
>    detector, and does not care if the target leaves frame. **|corr| went 0.13 -> 0.99.**
> 2. **INTEGRATE, DO NOT DIFFERENTIATE.** Differencing dot positions at 17.9 fps against 0.004
>    noise gives velocity SNR ~1.7 (R^2 0.03-0.07). Regressing DISPLACEMENT over a 0.30 s window
>    against the INTEGRATED gyro averages noise down while signal grows linearly.
> 3. **CONSTRAIN THE FIT TO THE PHYSICS.** A free 2x3 least squares has six parameters, and natural
>    head motion couples the axes — it confidently attributed u to ROLL with a coefficient of -2.76
>    where the entire plausible range is |1/FOV| = 0.82. Only FOUR things are unknown (which axis
>    for u, which for v, and two signs); the magnitude follows from the FOV. Fixing the scale turns
>    fitted/theoretical into a **CHECK** instead of a free parameter.
>
> **ON-SCREEN GUIDANCE.** `--map` shows instructions, live head-rate, and flow status **on the
> display**, because terminal output does not reach someone wearing the glasses — that coordination
> failure wasted several runs in one session. It also refreshes on FAILURE, since the first version
> only redrew after a successful sample and so froze exactly when things went wrong.
>
> ## >>> THE XREAL IMU IS WORKING — 1100-1400 Hz, DECODED AND CONFIRMED (2026-08-03)
> `software/xreal_imu.py`. **It is NOT HID.** The glasses present as a **USB NETWORK device**:
> here they came up as `en8`, host `169.254.2.10`, **glasses at `169.254.2.1`, TCP port 52998**,
> pushing a binary stream the moment you connect. (HID interfaces exist — VendorID 0x3318, usage
> pages 0x41 and 0x0c — open fine and produce **no reports**. Dead end; do not spend time there.)
>
> **Wire format:** fixed **134-byte records**, magic `28 36 00 00`, at **~1100-1400 Hz**.
> ```
> offset 0x22 (34)   3x float32 LE   GYRO   rad/s
> offset 0x2e (46)   3x float32 LE   ACCEL  m/s^2
> ```
> **CONFIRMED BY PHYSICS, NOT BY EYE:** stationary, `|accel|` = **9.773 m/s²** against gravity's
> 9.81 — **0.38% off**, which is mounting tilt and bias, not a mis-parse. Gyro reads 0.23 °/s at
> rest. Those two facts are what make this a decode rather than a plausible guess.
>
> **`python3 xreal_imu.py --live`** prints rate, `|a|` vs g, and `|w|` — re-run it after any
> firmware or cable change, because a silently wrong offset would still decode to finite floats.
>
> - **THIS IS 63x THE CAMERA RATE** (1133 Hz vs 17.9 fps) and it is the fast half of the fast/slow
>   split every XR stack uses. `predictor.py` currently estimates velocity from the dot's image
>   motion; the IMU replaces that estimate and **nothing else in the chain changes**.
> - Also the fix for `WorldTracker`'s 0 map points. **One component, two problems.**
> - **NO SOLDERING, NO CARRIER CHANGE** — which matters because the carrier is final. Prefer this
>   over the XIAO IMU in the CAD: the XREAL's is rigid to the **glasses** (where the DISPLAY is,
>   which is what reprojection needs), while a carrier IMU is rigid to the **cameras**.
> - Needs `pip install hidapi` only for the (dead-end) HID diagnostic; the TCP path needs nothing.
>
> ## >>> (superseded) the XREAL IMU is probably readable — CHECK THIS BEFORE WIRING ANYTHING
> Dylan, 2026-08-03: *"the xreal has an imu so it may not need an additional imu."* He is right,
> and it may be BETTER than a carrier-mounted one. **Verified on this Mac:** the glasses enumerate
> as USB `XREAL One Pro`, **VendorID 13080 (0x3318), ProductID 1078**, and expose **HID interfaces
> including `PrimaryUsagePage = 65` (0x41, vendor-defined)** — which is exactly where the
> community Linux/macOS drivers read IMU packets from. No extra hardware, no soldering.
> - Prior art: `SamiMitwalli/One-Pro-IMU-Retriever-Demo` (One Pro specifically, gyro + accel) and
>   `adidoes/xrealair-sdk-macos` (macOS port of the nrealAir driver, reads two HID interfaces).
> - **WHY IT MAY BEAT OUR OWN IMU, and this is the real argument:** the XREAL's IMU is rigid to the
>   GLASSES, which is where the DISPLAY is. Our carrier is **ZIP-TIED** and moves relative to the
>   glasses. Late-stage reprojection needs *display-relative* rotation, so the glasses' own IMU is
>   the more correct reference. A carrier IMU measures the CAMERAS' motion (what VIO wants); the
>   two differ only by carrier flex, which `MountAnchor` already detects.
> - **STILL UNVERIFIED:** that the packets can actually be decoded on the One Pro, and at what rate.
>   The HID interfaces existing is not the same as the data being readable. Try before planning on it.
>
> ## ACCELERATION vs POSITION — Dylan's second question, and he is right
> *"why do we need to know acceleration if we know change in position/distance/angle?"* We do NOT.
> My earlier "an IMU cannot give translation, it drifts" was about an **IMU ALONE**. With cameras
> supplying position and the gyro supplying rotation you FUSE them — that is visual-inertial
> odometry, the standard solution, and it is why the drift argument does not apply here.
> - For the immediate need — **low-latency rotation compensation — only the GYRO is used.** Gyro
>   integrates cleanly over short intervals. No accelerometer, no double integration, no drift problem.
> - **"Can it update in real time?" Yes, and that is the whole point.** IMUs run 100-1000 Hz against
>   our **13 fps** camera loop. Fast gyro for high-rate rotation, slow cameras for absolute position
>   and drift correction. The camera loop never has to get faster.
>
> ## RENDERING ARCHITECTURE — follow mode + own IMU, decided 2026-08-03
> Dylan asked the right question: to paint a goblet onto a cup, in FOLLOW mode the goblet swings
> with your head so software must counter-rotate it, whereas a LOCKED screen appears to do that
> for free. The answer is that you must do it yourself, and the reason is not calibration hygiene:
>
> - **A 3DoF stabiliser cancels ROTATION exactly at any distance, and TRANSLATION not at all**,
>   because the needed shift depends on depth, which it does not know. A 10 cm sideways head move,
>   uncorrected: **0.5 m -> 11.3° = 434 px**; 2 m -> 2.9° = 110 px; **20 m -> 0.29° = 11 px**.
>   So anchor mode cannot do the cup. For the far field it is nearly sufficient.
> - **You cannot tell anchor mode where the cup is.** It holds whatever was drawn relative to
>   wherever it anchored. You still must compute the pixel — and now through a transform you can
>   neither query nor calibrate. That is what breaks both the calibration and the rendering.
> - **THE GLASSES CANNOT GET DEPTH.** An IMU integrates rotation cleanly but translation needs
>   double-integrating acceleration, which drifts quadratically — metres within seconds. Physics,
>   not firmware. (Believed no built-in world cams on the One Pro; the XREAL Eye is a clip-on —
>   VERIFY, not measured.)
> - **THE RIG CAN, and the scaling is the good news.** 67 mm baseline, focal 913 px. Stereo depth
>   error and the depth error REQUIRED for sub-pixel parallax **both scale as d²**, so their ratio
>   is **constant at 0.90 across the whole range** — 1 mm needed / 1 mm available at 0.5 m,
>   1.8 m / 1.6 m at 20 m. The baseline is matched to the task at EVERY distance, not just near.
>   Caveat: assumes 0.25 px disparity precision (optimistic) and leaves only 10% headroom; at a
>   realistic 0.5 px you are ~2 px off. Thin, not comfortable.
> - **LATENCY IS THE REAL ARGUMENT FOR ANCHOR MODE, and it is a strong one.** The measured loop is
>   **73.6 ms/frame**: at 30°/s that is 2.2° = **85 px** of lag, at 100°/s **283 px**, at 200°/s
>   **565 px**. The goblet would visibly swim on every head turn. XREAL's stabiliser runs in the
>   display pipeline at near-zero latency and genuinely does something a 13 fps loop cannot.
> - **>>> THE ARCHITECTURE: keep FOLLOW mode, and do low-latency rotation compensation with YOUR
>   OWN IMU, applied as late as possible before drawing** (late-stage reprojection / timewarp —
>   what every VR/AR system does, for exactly this reason). You get the near-zero-latency rotation
>   cancellation AND keep a transform you own and can calibrate. The CAD already has the IMU mount,
>   and the handoff already concluded an IMU is the fix for `WorldTracker`'s 0 map points —
>   **the same IMU solves both, which makes it the highest-value unbuilt thing in the project.**
> - **Strategic:** the end goal is people and cars FAR away, where translation costs only ~11 px at
>   20 m. So for the actual target **rotation is nearly the entire problem**, and rotation is
>   exactly what an IMU cancels perfectly. The far-field goal is a much easier problem than the cup.
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
> - **`WorldTracker`: THE CORRECTED PAIR ORDER DOES NOT FIX IT — RE-TESTED ON HARDWARE 2026-08-03.**
>   Run with the ordering established that day (**worldL=3, worldR=2**), on live frames, with the
>   head moving: **0 inliers, 2 track ids over 67 frames**, telemetry `{'used': 'imu'}` while no IMU
>   is present. **The reversed pair was never the cause.** That closes the question the earlier note
>   left open as "a cause, not *the* cause" — the correspondence filter and the pair order are both
>   now known-good. **Do not investigate either again.** The fix is an IMU.
> - **XREAL's OWN MOTION TRACKING MUST BE OFF DURING CALIBRATION.** Dylan asked to run "with motion
>   tracking", meaning the glasses'. With their 3DoF tracking active the optics displace the image
>   by a **head-pose-dependent amount** before it reaches the eye, so identical (eye features + dot)
>   map to different physical directions depending on head history — the learned map fits an average
>   and is wrong everywhere. **Head-following IS the "off" state** (screen rigidly attached to the
>   glasses) and is the correct one. Anchor/locked = tracking ON = corrupts the calibration.
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
>   **>>> CONFIRMED WORN 2026-08-03 AND THIS IS NOW THE BLOCKER. It is SEATING, not the model.**
>   Ruled out by measurement, in this order, so nobody repeats it:
>   - **Exposure is FINE**: 1.7% / 1.1% saturated worn (worn reference 5.3/3.3, desk 40.9/20.2) —
>     better than reference. Light was never the problem; stop chasing daylight.
>   - **Focus is FINE**: lashes and iris texture resolve clearly in both frames. (Laplacian read
>     14/13, which is exactly why that metric is useless here — it calls a sharp eye blurry.)
>   - **The model lands OUTSIDE the band on BOTH eyes**: gate-convention u **0.675** (`eyeL`) and
>     **0.665** (`eyeR`) against a band of **[0.698, 0.971]**. In band **0% / 1%** of frames;
>     tracker reached `ok` on only **22% / 6%**.
>   - **Drawing the prediction on the frame shows it is not on a tear duct** — it sits on the lower
>     lid / cheek (`eyeL`) and the upper lid crease (`eyeR`).
>   - **`eyeL`'s eye is CUT OFF at the left frame edge** — the iris is clipped by the border, so
>     the inner canthus is at or past the edge. There is nothing there for a model to find.
>   - **The rig is seated differently than when the 98-frame corpus was captured**, and
>     asymmetrically: `eyeL` reads v 0.788 now vs 0.635 labelled (0.15 LOWER); `eyeR` 0.587 vs
>     0.666 (0.08 HIGHER). That gap, not model quality, is what the band rejects.
>   **Do NOT retrain and do NOT widen the band until seating is fixed** — either would bake this
>   session's bad seating into the model permanently. The brow clamp is the lever; the eye needs to
>   come UP and toward frame centre. Re-measure after each adjustment with `python3 eye_check.py`.
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
> - `main.py --input-test` — calibration-loop input plumbing. Pins that the not-live reason NAMES
>   the failing camera (one lumped string for five faults is what sent a session hunting the world
>   dot when the fault was `eyeR`), and that every documented key maps. Both assertions were run
>   against the old code and confirmed to fail.
>
> ## Guardrails (do NOT break)
> - `software/rig.py` is the single source of truth for camera geometry; `cad/xreal_one_mount.scad`
>   mirrors it (parity checked to 0.0003 mm). Don't reorder `config.feature_names`.
> - Every new module needs a `--selftest` hooked into `verify_all.py`. **`python3 verify_all.py
>   --fast` must end "ALL CHECKS PASS"** (currently 34 checks). Run it after touching shared
>   capture/geometry code — and note it takes ~10 min, so run it in the background.
> - **Build the object under test through its real constructor.** See the 9th pattern entry above:
>   `__new__` + hand-assigned attributes is how a tracker that could not be constructed at all
>   passed every check. Stub the *collaborator*, never the object itself.
> - **Persist settled decisions** — see **RULE ZERO** at the top of this file, which is the one
>   guardrail that outranks the rest: code (a named constant others default to) + the memory
>   file's settled-facts block + a commit, written as each thing lands. Never leave a decision
>   only in the transcript. If you finish a session with findings unsaved, the session did not
>   happen.
> - Be honest about sim vs hardware. Almost everything is sim-validated; what has actually run on
>   real cameras is listed above, and no calibration has ever completed on hardware.

---

See `media/session-2026-07/` for images from the build/overlay work referenced above.
