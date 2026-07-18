# Pixel-by-Pixel Calibration Sweep

`software/pixel_sweep.py`: the large-scale data-generation run, stated as an algorithm
and grounded in the same physics as the rest of the rig.

## The protocol

100 eye geometries. Place the glasses (the CAD) on each face, use the glasses' own
cameras to make position guesses, and generate 100 different glasses positions on each
face. Then go pixel by pixel: guess a point that lies on a straight line that intersects
the dot, the pixel, and the location where light enters the eye. A user-AI corrects the
guessing-AI until it guesses correctly twice in a row, then the sweep moves to the next
pixel, for every pixel, for each of the 100 eye geometries, and within each eye
geometry, the 100 glasses positions.

The resulting dataset is **100 faces × 100 glasses positions × pixels × avg
guesses-to-correct**; `sum(guess_counts)` *is* that product.

## How each clause is realized (and where it comes from)

| Protocol clause | Implementation | Source of the numbers |
|---|---|---|
| 100 eye geometries from real eye data | `anatomy.sample_subject` | IPD 63±3.8 mm (Dodgson 2004), globe r 12 mm, entrance pupil 10.5 mm ahead of the center of rotation, inter-/outer-canthal distance, canthal tilt, **angle kappa**: all cited in `software/anatomy.py` |
| Faces structured so glasses sit naturally | face frame + rest pose | nominal IPD, vertex distance, **pantoscopic tilt −7°**, canthi from palpebral-fissure anthropometry (`rig.py`, `anatomy.py`) |
| The CAD *is* the glasses; its cameras guess position | `rig.build()` mirrors `cad/xreal_one_mount.scad`: 2 forward world cams at the pupils + 2 inward eye-corner cams | `rig.py` is the single source of truth for the printed bracket |
| 100 glasses positions per face | `seat()` then 100× `slip()`: the realistic slide-down-the-nose envelope (vertical drift + correlated pantoscopic pitch + jitter, occasional re-seat) | `autosim.Simulator` |
| Pixel by pixel; a point on the line through the dot, the pixel, and where light enters the eye | for each display pixel we invert the optic to the **chief ray** (entrance-pupil → world) and pin the point where it meets the page | `truepx_batch` + `invert_pixels` |
| User-AI corrects guessing-AI until right twice in a row | the per-pixel correction loop with a two-in-a-row gate | `GuessingAI` + `UserAI` |
| dataset = 100 × 100 × pixels × avg guesses | `sum(guess_counts)` | the run's output |

## Why a pixel corresponds to a *line*, not a point

Lighting display pixel `(u,v)` sends one angular direction out of the see-through optic.
You perceive the overlay along your eye's **chief ray**: the line from your **entrance
pupil** out into the world. So a pixel maps to a *line in space*, and any real target on
that line registers under the pixel. We pin the unique point where that line crosses the
page (depth `z0 ≈ 480 mm`). The guessing-AI guesses that point; "correct" means the guess
lands on the line (within `--tol-mm`, default 2.5 mm at the page ≈ a few display pixels).

This is the **inverse** of `pixel_map.py` (which asks "given a world dot, what pixel?").
Here we walk pixels and ask "what world point?", which is what "go pixel by pixel and guess
a point on the line" literally requires.

## The guessing-AI actually learns (so the guess count means something)

Two levels, mirroring the rig's warm-start meta-learning:

* a **global prior**: folded in at the end of each face, so later faces start warmer;
* a fast **per-face residual**: reset each face, updated by every correction; it can read
  the eye-corner features (which move with the glasses), so it learns this face's
  irreducible bias (angle kappa, unobservable from outside) *and* its pose-dependent warp.

It **starts at zero** and learns only from this run's corrections, no pre-existing data is
read. Early pixels of a new face need several corrections; once the residual has the warp,
later pixels lock in the minimum 2 guesses. Early faces cost more than late faces. The
guess counts encode that whole learning curve.

## The validation gate, the sim must pass before anything generates

Nothing generates until `selftest()` passes (it runs automatically at the start of every
sweep, and `--selftest` runs it alone):

1. the vectorized oracle equals `autosim.Simulator.ground_truth` to **2.2e-16**;
2. the pixel→world inverse round-trips (`forward(invert(uv)) == uv`) to **3.3e-16** (≈0 px @1080p);
3. recovered chief-ray points are finite and on-page;
4. **physical sensitivity**: sliding the glasses 6 mm down the nose moves the center
   pixel's required world point by 4.5 mm, i.e. the calibration premise is real.

If any check fails, the run refuses to start.

## Running it

```bash
cd ~/ar-eye-calibration/software
python3 pixel_sweep.py --selftest                 # validate the sim only
python3 pixel_sweep.py                             # 100x100, dense 24x24 pixel grid (storable tensor)
python3 pixel_sweep.py --every-pixel               # EVERY native pixel, full 1920x1080
python3 pixel_sweep.py --faces 100 --poses 100 --grid 48 --tol-mm 2.0
# or:  python3 main.py --pixel-sweep [--every-pixel-full]
```

### Two modes

* **Grid mode** (default), strict per-pixel sequential protocol over an `N×N` grid across
  the display field. Saves the full per-pixel tensor `guess_counts[face, position, pixel]`
  to `data/pixel_sweep.npz`. Best for analysis/heatmaps.
* **Every-pixel mode** (`--every-pixel`): the *same* twice-in-a-row rule over **every** one
  of the 1920×1080 native pixels. A pixel already within tolerance deterministically locks
  in exactly 2 guesses (the model only changes on a miss), so the whole field is tractable:
  it's evaluated vectorized per round and only the pixels the AI can't yet predict drive
  extra rounds. The full per-pixel tensor would be ~40 GB, so it stores per-(face,position)
  aggregates `agg[face, position] = [on-page pixels, total guesses, rounds]` to
  `data/pixel_sweep_full.npz`; `sum(agg[...,1])` is the exact total data-piece count. The
  surrogate inverse used to place all native pixels is checked each pose to stay far below
  the tolerance (reported as `surrogate inverse max error`).

`--every-pixel` at 100×100 is a very long run (hours per face at native resolution, use
the default dense grid unless the full tensor is required); it checkpoints every face to
`data/pixel_sweep_full.ckpt.npz` and logs progress, so a crash never costs the whole run.

## Outputs

* `data/pixel_sweep.npz` (grid mode): `guess_counts[face,pos,pixel]`, `grid_uv`,
  `descriptors[face]` (the 8-number eye geometry), `face_avg_guesses`, `tol_mm`, `z0`.
* `data/pixel_sweep_full.npz` (every-pixel): `agg[face,pos,3]`, `descriptors`,
  `face_avg_guesses`, `display`, `total_pieces`.

Nothing here reads or writes the existing `mega_*`, `meta.db`, or `pixel_map.npz` artifacts.
