# Results: SIMULATION only

Everything in this directory is a **simulation** result: no number here is a hardware
measurement. I generated these in July 2026, before the rig existed, to decide whether the
approach was worth building. The hardware now exists and its measurements live in the dated
session write-ups at the repository root, not here.

- **`kappa_separation.png`**, the angle-kappa / perceptual-bias confound, and the
  multi-distance way around it. Generated 2026-07-13 by `python3 software/kappa.py`.
  - *Left:* overlay error vs the number of the user's own corrections. Against geometric
    truth the error plateaus at **~9 px**, the kappa + perceptual-bias confound that a
    single-condition calibration cannot separate.
  - *Right:* registration error across target distances 300–1800 mm. Calibrating at a
    **single distance** mistakes the vergence-accommodation bias for a constant and
    carries **13.0 px** median to other distances; regressing **across distances**
    separates kappa (the constant) from the vergence bias (the slope) and holds
    **2.4 px** median everywhere, no extra hardware.
- **`kappa_run.log`**, the raw program output the plot numbers were taken from.

Reproduce: `cd software && python3 kappa.py` (numpy only; the raw `data/` experiment
outputs referenced elsewhere are generated locally and are not committed).
