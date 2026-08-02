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
> **Read first, in this order:**
> 1. The memory file `project_ar_eye_calibration.md` — especially the **`SETTLED HARDWARE FACTS —
>    DO NOT RE-DERIVE`** block at the end. Read it before proposing anything physical.
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
> - In `--calibrate-corners`, drag the ROI box around the **INNER** canthus (tear-duct side).
> - `calib_preflight.py --run` checks all six preconditions and names the failing one. The
>   historically-red row is **`world dot`**: it needs a **dark round dot on white paper**, ~2-3 cm,
>   ~0.5-2 m away, **roughly centred in BOTH world cams**, away from a bright window. Its
>   `dot_geometry()` check rejects a "dot" at an impossible depth or off the epipolar line —
>   two loose cameras each finding their own dark blob otherwise reads green and poisons samples.
> - In the calibration loop: **ENTER** approve · **U** undo last · **Z** cancel nudge · **Q** quit.
>   Expect ~20-40 corrections across the field before the error means anything.
>
> ## Open bugs (neither blocks calibration)
> - **`WorldTracker` built 0 map points** across a 45 s hardware run — no usable stereo
>   correspondences on real frames. Calibration does **not** touch the mesh (it uses
>   `dot_detector` + `eye_tracker`), but the world-locked overlay depends on it entirely.
> - **The accuracy corpus is STALE.** `data/mega_prior*`, `calibration_db.npz`, `meta.db` and the
>   1.88 px `accuracy_map` figure were all generated at `EYE_FOV=90` tracking the **outer**
>   canthus. They no longer describe this rig — regenerate before quoting any accuracy number.
>
> ## Tools added for hardware work
> - `bank_bringup.py` — partial-bank bring-up under a CPU budget: `--scan`, `--run`, `--ramp`
>   (finds how many streams the bus carries), `--roles ROLE=INDEX` override, `--native`,
>   `--no-warm`, `--view`.
> - `calib_preflight.py` — the six calibration preconditions, with the fix printed per failing row.
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
> - **Persist settled decisions**: code (a named constant others default to) + the memory file's
>   settled-facts block + a commit. Never leave a decision only in the transcript.
> - Be honest about sim vs hardware. Almost everything is sim-validated; what has actually run on
>   real cameras is listed above, and no calibration has ever completed on hardware.

---

See `media/session-2026-07/` for images from the build/overlay work referenced above.
