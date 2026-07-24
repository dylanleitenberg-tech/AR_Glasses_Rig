"""cad_overlap.py — measure REAL solid-on-solid overlaps in the printed carrier CAD.

cad_fit.py / wearable.py check the CAMERAS (boards at rig.py positions) against the cone /
eyeball / face, but nothing checked the printed SOLIDS — rail, brow clamps, booms, holders,
IMU shelf — against EACH OTHER or against the GLASSES BODY the carrier clips onto. This shells
out to the OpenSCAD CLI, renders intersection(A, B) for each pair via cad/overlap_check.scad,
and integrates the resulting binary-STL volume (divergence theorem). Volume ~0 => clear.

Pairs are classified:
  KEEPOUT  — part vs glasses / eyeball / see-through cone: any overlap is a physical
             impossibility (the glasses are real) or blocks vision -> must be ~0.
  DISTINCT — two printed parts that should NOT touch (e.g. a camera holder vs a clamp):
             overlap means the print is fused where it shouldn't be -> must be ~0.
  ANCHOR   — intended structural union (booms root into the rail): overlap is fine.

Usage:  python3 cad_overlap.py            # full matrix
"""
import os
import struct
import subprocess
import sys
import tempfile

OPENSCAD = os.path.expanduser("~/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD")
SCAD = os.path.join(os.path.dirname(__file__), "..", "cad", "overlap_check.scad")

# (A, B, class) — right side + midline; the left is a mirror.
PAIRS = [
    # printed part vs the PHYSICAL glasses body (must be zero — the glasses are real)
    ("rail",   "glasses", "KEEPOUT"), ("clampR", "glasses", "KEEPOUT"),
    ("imu",    "glasses", "KEEPOUT"), ("worldR", "glasses", "KEEPOUT"),
    ("eyeR",   "glasses", "KEEPOUT"), ("pupilR", "glasses", "KEEPOUT"),
    # printed part vs the eye / the see-through cone (blocks vision or touches the eye)
    ("rail",   "coneR", "KEEPOUT"), ("clampR", "coneR", "KEEPOUT"),
    ("worldR", "coneR", "KEEPOUT"), ("eyeR",   "coneR", "KEEPOUT"),
    ("pupilR", "coneR", "KEEPOUT"), ("imu",    "coneR", "KEEPOUT"),
    ("eyeR",   "eyeballR", "KEEPOUT"), ("pupilR", "eyeballR", "KEEPOUT"),
    # printed parts that must NOT fuse with each other
    ("worldR", "eyeR",   "DISTINCT"), ("worldR", "pupilR", "DISTINCT"),
    ("eyeR",   "pupilR", "DISTINCT"),
    ("eyeR",   "clampR", "DISTINCT"), ("pupilR", "clampR", "DISTINCT"),
    ("worldR", "imu",    "DISTINCT"), ("worldR", "worldL", "DISTINCT"),
    # the pupil booms UNITE into one central column and root into the IMU tower (by design,
    # 2026-07-04) — these fusions are intended structure now
    ("pupilR", "pupilL", "ANCHOR"), ("pupilR", "imu", "ANCHOR"),
    # each camera's boom vs its own PCB keep-out (the physical board + back components):
    # the boom's end must NEVER enter it (the old attachment bulged 2.8 mm in)
    ("worldR_boom", "worldR_pcb", "KEEPOUT"),
    ("eyeR_boom",   "eyeR_pcb",   "KEEPOUT"),
    ("pupilR_boom", "pupilR_pcb", "KEEPOUT"),
    # and each boom must FUSE with its holder's boss (one printable body)
    ("worldR_boom", "worldR_hold", "ANCHOR"),
    ("eyeR_boom",   "eyeR_hold",   "ANCHOR"),
    ("pupilR_boom", "pupilR_hold", "ANCHOR"),
    # intended structural unions (report volume, no verdict)
    ("rail", "clampR", "ANCHOR"), ("rail", "imu", "ANCHOR"),
    ("rail", "worldR", "ANCHOR"), ("rail", "eyeR", "ANCHOR"), ("rail", "pupilR", "ANCHOR"),
    # STRAIGHT-UP world riser (2026-07-23): its foot sits under the plate centre, inside the
    # clamp span, so it FUSES with the clamp top by design (rigidity; jaw slot below is clear)
    ("worldR", "clampR", "ANCHOR"),
]

TOL_MM3 = 1.0   # below this, treat as numerical noise / face-grazing


def stl_volume(path):
    """Signed volume of a binary STL via the divergence theorem (mm^3)."""
    if not os.path.exists(path) or os.path.getsize(path) < 84:
        return 0.0
    with open(path, "rb") as f:
        f.seek(80)
        (n,) = struct.unpack("<I", f.read(4))
        vol = 0.0
        for _ in range(n):
            d = f.read(50)
            v = struct.unpack("<12f", d[:48])
            ax, ay, az = v[3], v[4], v[5]
            bx, by, bz = v[6], v[7], v[8]
            cx, cy, cz = v[9], v[10], v[11]
            vol += (ax * (by * cz - bz * cy)
                    - ay * (bx * cz - bz * cx)
                    + az * (bx * cy - by * cx)) / 6.0
    return abs(vol)


def check_pair(a, b, out):
    cmd = [OPENSCAD, "-D", 'part="none"', "-D", f'compA="{a}"', "-D", f'compB="{b}"',
           "--export-format", "binstl", "-o", out, SCAD]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        # an EMPTY intersection makes OpenSCAD exit nonzero — that's the CLEAR case, not an error
        if "top level object is empty" in (r.stderr or ""):
            return 0.0, []
        return None, (r.stderr or "").strip().splitlines()[-1:]
    return stl_volume(out), []


def main():
    only = sys.argv[1:] or None
    bad = 0
    print("== printed-solid overlap matrix (intersection volume, mm^3) ==")
    print(f"  {'A':8s} {'B':10s} {'class':9s} {'volume':>10s}   verdict")
    with tempfile.TemporaryDirectory() as td:
        for a, b, cls in PAIRS:
            if only and a not in only and b not in only:
                continue
            vol, err = check_pair(a, b, os.path.join(td, "pair.stl"))
            if vol is None:
                print(f"  {a:8s} {b:10s} {cls:9s} {'ERROR':>10s}   {err}")
                bad += 1
                continue
            if cls == "ANCHOR":
                verdict = "joined" if vol > TOL_MM3 else "NOT JOINED?!"
                if vol <= TOL_MM3:
                    bad += 1
            else:
                verdict = "clear" if vol <= TOL_MM3 else "OVERLAP <<<"
                if vol > TOL_MM3:
                    bad += 1
            print(f"  {a:8s} {b:10s} {cls:9s} {vol:10.1f}   {verdict}")
    print(f"  => {'ALL CLEAR' if bad == 0 else str(bad) + ' PROBLEM(S) — fix the CAD'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
