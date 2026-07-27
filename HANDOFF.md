# New-session handoff prompt

Paste the block below into a fresh session to pick up this project. Everything it references
is in this repo (`~/ar-eye-calibration`) or the persistent memory file.

---

> You're continuing the **AR eye-calibration rig** project (`~/ar-eye-calibration`, a git repo).
> It's a modified **XREAL One Pro** that registers AR-overlay pixels to the real world by
> tracking how the glasses sit on the face (eye-corner + pupil cams) plus two forward world
> cams. It's **simulation-first**: every module has a headless `--selftest`, and
> `software/verify_all.py --fast` is the release gate (must end "ALL CHECKS PASS").
>
> **First, read in this order:**
> 1. The memory file `project_ar_eye_calibration.md` (persistent; the full running history).
> 2. `README.md`, then `HARDWARE_BRINGUP.md`, `WIRING.md`, `ASSEMBLY.md`, `ORDER_LIST.md`,
>    `MONKEY_OVERLAY.md`, `SAFETY.md`, `RESTORE.md`.
> 3. `git log --oneline | head -30` for recent work.
>
> **Current state (as of 2026-07-27):**
> - **Software complete + gated:** sim/calibrator stack; the host bring-up layer
>   (`connect.py`, `sync_capture.py`, `autoexpose.py`, `world_mesh.py`, `rig_test.py`,
>   `snapshot.py`, `live_rig.py`) with synchronized capture + world-mesh tracking; the
>   world-locked "turn people into monkeys" overlay (`anchor.py`, `people_track.py`,
>   `avatar.py`, `augment_rig.py`). All have `--selftest`; `verify_all.py --fast` passes.
> - **CAD:** carrier + LED bracket (`cad/xreal_one_mount.scad`). Eye-corner cams were
>   **re-aimed down 6 mm + out 3 mm** (`EC_AIM_DOWN`/`EC_AIM_OUT`) to centre the outer canthus
>   (mounted test showed it edge-riding). CAD↔rig parity holds (0.0003 mm). Printable STLs are
>   in `_printable_stls/` in the backup and regenerate from the CAD.
> - **Electrical verified + documented:** full V/I/R/A audit done. Three wiring rules are now
>   in WIRING.md + ASSEMBLY.md: **(1) IMU on 3.3 V not 5 V**, **(2) rail-monitor ADC via a 2:1
>   divider**, **(3) IR 5 V + divider tap the rail, not the XIAO's 5 V pin.** Parts:
>   MPU-6050 IMU (6-axis), 2N7000 MOSFET, 4× 940 nm LEDs (330-470 Ω, ~48 mA total), SMAJ5.0A
>   TVS, 300 mA polyfuse, SABRENT 60 W hub.
> - **Hardware:** cameras/NoIR confirmed by bench test; carrier v9 printed + fits; the parts
>   order (to Hawaii) is fully vetted (see `ORDER_LIST.md` + the memory). Returning the current
>   XREAL, ordering a new one to Hawaii. Building in Hawaii; final print in **ASA/PETG at home**.
>
> **>>> IMMEDIATE NEXT STEP: rerun the sim so every output MATCHES the current specs. <<<**
> The eye-cam **re-aim** (`EC_AIM_DOWN=6` / `EC_AIM_OUT=3`, applied in BOTH `rig.py` and the CAD)
> plus the electrical/diagram updates are the latest changes. Some `data/` artifacts were built
> on the PRE-re-aim geometry, so bring the whole sim into line with the current `rig.py`:
> ```
> cd software && source .venv/bin/activate
> python3 verify_all.py --fast     # CAD<->rig parity (0.0003mm) + every selftest must PASS
> python3 accuracy_map.py          # per-position accuracy; was 1.88 px median post-re-aim
> ```
> Then **regenerate the geometry-dependent trained artifacts** on the current geometry so the
> data matches the specs (see each script's header for exact flags):
> `pixel_map.py` → `data/pixel_map.npz`; `calibrate.py` → `data/calibration_db.npz`;
> `autotrain.py`/`megarun.py` → the warm-start prior (`meta.db` / `mega_*.npz`, research-scale).
> Confirm parity stays 0.0003 mm and accuracy stays ~1.9 px median (the re-aim is
> accuracy-neutral — the canthus is still the tracked landmark, just re-centred); **report the
> deltas vs those numbers.** Then the build proceeds once parts arrive (PLA test print of the
> re-aimed carrier → verify eye-cam framing + glasses clearance on-head → ASA/PETG final).
>
> **Guardrails (do NOT break):**
> - `software/rig.py` is the single source of truth for camera geometry; `cad/xreal_one_mount.scad`
>   mirrors it (parity checked). Don't reorder `config.feature_names`.
> - Keep everything sim-testable; add a `--selftest` to new modules and hook it into
>   `verify_all.py`. The gate must stay green.
> - Be honest about hardware vs sim: nothing has run on real cameras yet.
> - Eye safety: the IR-meter corneal-irradiance check (`SAFETY.md`) is required before first
>   wear; never rely on the strobe/watchdog alone (Rule 7 — resistor-limited continuous-safe).

---

See `media/session-2026-07/` for images from the build/overlay work referenced above.
