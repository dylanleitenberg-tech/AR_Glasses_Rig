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
> **>>> IMMEDIATE NEXT STEP: ASSEMBLY — the parts have ARRIVED (2026-07). <<<**
> Guide the physical build per `ASSEMBLY.md` + `WIRING.md`, verifying each step and asking for
> photos. Order:
> 1. **PLA test print of the re-aimed carrier FIRST.** The eye-cam re-aim (`EC_AIM_DOWN=6` /
>    `EC_AIM_OUT=3`) is UNVERIFIED on hardware. Mount an eye cam and check on-head that the outer
>    corner centres AND the cam clears the glasses hinge (it sat tight there — see
>    `media/session-2026-07/rig_mounted_on_glasses.jpg`). Only then print the **ASA/PETG final**.
> 2. Prep cameras: NoIR remote test; 940 nm filter on the **2 pupil** cams only; focus + **lock**
>    each M12 lens.
> 3. Populate: M2 self-tap boards to holders; bolt the **LED brackets** to the pupil shelves
>    (2 M2 each); seat the **4 LEDs** in the baffled pockets.
> 4. Wire the strobe/power circuit — **obey the three part-protecting rules (WIRING.md callout):**
>    IMU on **3.3 V not 5 V**; rail-monitor ADC via a **2:1 divider**; IR 5 V + divider **tap the
>    rail, not the XIAO**. 4 LEDs × 330-470 Ω → 300 mA polyfuse → 2N7000 low-side (100 Ω gate +
>    10 kΩ pull-down); 470 µF + 0.1 µF + SMAJ5.0A across the rail.
> 5. Bring-up (software): `connect.py --auto && --identify` → `rig_test.py --run` →
>    `snapshot.py` → `live_rig.py --run`.
> **⚠️ Safety:** measure corneal-plane IR irradiance with the meter (≪ 1 mW/cm²) BEFORE wearing
> with LEDs energized; verify the XIAO firmware fail-safe (gate low = IR off when the host app
> isn't running) before enabling strobe. Multimeter-check continuity + the 5 V rail (4.9-5.1 V)
> before powering IR.
>
> (Optional pre-flight: `verify_all.py --fast` + `accuracy_map.py` to reconfirm the sim matches
> current specs — was 0.0003 mm parity / ~1.9 px median. Regenerate `data/` artifacts only if
> you touch the geometry.)
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
