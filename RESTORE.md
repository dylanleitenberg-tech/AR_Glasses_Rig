# Restore / backup guide — AR eye-calibration project

This is the full project: the **XREAL One Pro eye-calibration rig** — parametric CAD, the
simulation + calibration + capture + world-mesh/overlay software, firmware, wiring/assembly
docs, and the parts list. This file explains how to restore it on any computer.

## What's in the backup archive

- **Everything committed to git** (full history — `.git/` is included).
- `cad/` — OpenSCAD parametric carrier + LED bracket (`xreal_one_mount.scad`).
- `software/` — all Python (sim, calibrator, capture, world_mesh, people_track, avatar,
  augment_rig, verify_all, etc.) + `requirements.txt`.
- `firmware/` — the XIAO IR-strobe firmware.
- Docs: `README.md`, `HARDWARE_BRINGUP.md`, `WIRING.md`, `ASSEMBLY.md`, `ORDER_LIST.md`,
  `EYE_TRACKING.md`, `SAFETY.md`, `KAPPA.md`, `MONKEY_OVERLAY.md`, `RIGID_MOUNT_BUILD.md`, …
- `data/` — the **distilled** artifacts kept (trained prior/identifier `*.npz`, `meta.db`,
  `calibration_db.npz`, `pixel_map.npz`, templates, checkerboard, logs).
- `_printable_stls/` — the ready-to-print parts (also regenerate from `cad/`, see below).

## What's intentionally EXCLUDED (regenerable — to keep the archive small)

| Excluded | Size | How to regenerate |
|---|---|---|
| `software/.venv/` | ~200 MB | `pip install -r requirements.txt` (below) |
| `data/mega_samples.npz` | ~227 MB | `python3 software/megarun.py 100000` (~2.5 h; research only, not needed to build) |
| `data/pixel_sweep_guesses.bin` | ~38 MB | `python3 software/pixel_sweep.py` (research only) |
| `data/*.old_geom_*`, `*.old_fwd12` | ~36 MB | superseded DB backups; not needed |
| `__pycache__/`, `*.pyc` | — | recreated on run |

None of the excluded files are needed to **build the hardware** or run the release gate.

## Restore on another computer

1. **Extract** the archive:
   ```
   tar xzf ar-eye-calibration-backup-*.tar.gz
   cd ar-eye-calibration
   ```
2. It's a live **git repo** with full history:
   ```
   git log --oneline | head        # see the commit history
   ```
3. **Recreate the Python environment** (needs Python 3.9+):
   ```
   cd software
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt        # numpy + opencv
   ```
4. **Verify everything works** (headless, no hardware):
   ```
   python3 verify_all.py --fast           # should end: RELEASE GATE: ALL CHECKS PASS
   ```

## Where to start (entry points)

- **Build the rig:** `ORDER_LIST.md` (parts) → print `cad/` → `ASSEMBLY.md` + `WIRING.md`.
- **Bring up the cameras on a Mac:** `HARDWARE_BRINGUP.md`
  (`connect.py` → `rig_test.py` → `live_rig.py`).
- **The overlay app:** `MONKEY_OVERLAY.md` (`augment_rig.py`).
- **The science / accuracy:** `README.md`, `KAPPA.md`, `software/AUTOTRAIN.md`.

## Regenerate the printable STLs from CAD

```
OSC=~/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD    # or your openscad path
$OSC -D 'part="carrier"'     -o carrier.stl     cad/xreal_one_mount.scad
$OSC -D 'part="led_bracket"' -o led_bracket.stl cad/xreal_one_mount.scad
```

## Making a fresh backup later

From the project root, re-run the same archive command (see the parent folder), or simplest —
because it's all in git, `git bundle create ../ar-eye-cal.bundle --all` makes a single-file,
full-history backup you can restore with `git clone ar-eye-cal.bundle`.
