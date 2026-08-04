"""verify_all.py — THE RELEASE GATE: one command that verifies the whole device design.

Runs every check the project has (code, simulation contracts, CAD geometry, doc consistency)
and fails loudly on any regression. Run before every print, order, or release:

    python3 verify_all.py            # full gate (~40 min; the overlap matrix dominates)
    python3 verify_all.py --fast     # skips the overlap matrix + STL renders (~3 min)

Sections:
  1 CODE      py_compile everything; module selftests (sim contracts, capture/blink,
              IMU serial, device bridge, binocular oracle, pixel_sweep oracle)
  2 GEOMETRY  rig.py import (occlusion assert), CAD<->rig parity (positions AND aims),
              cad_fit (boards vs cone/eyeball), wearable (CORE cams vs face)
  3 CAD       OpenSCAD CSG compile; full printed-solid overlap matrix (35 pairs);
              STL exports with connected-shell counts (carrier=1, ir_rings=2)
  4 DOCS      stale-fact denylist greps (old positions, dead claims, dead hardware)
"""
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OPENSCAD = os.path.expanduser("~/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD")
SCAD = os.path.join(ROOT, "cad", "xreal_one_mount.scad")
FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout, cwd=HERE)
    return r.returncode, r.stdout + r.stderr


def stl_shells(path):
    with open(path, "rb") as f:
        f.seek(80)
        (n,) = struct.unpack("<I", f.read(4))
        tris = []
        for _ in range(n):
            d = f.read(50)
            v = struct.unpack("<12f", d[:48])
            tris.append(tuple((round(v[i], 4), round(v[i+1], 4), round(v[i+2], 4))
                              for i in (3, 6, 9)))
    parent = {}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for t in tris:
        for v in t:
            parent.setdefault(v, v)
        union(t[0], t[1]); union(t[1], t[2])
    return len({find(t[0]) for t in tris})


def main(fast=False):
    print("== 1 CODE ==")
    rc, out = run("python3 -m py_compile *.py")
    check("py_compile all modules", rc == 0, out.strip()[:120])
    for name, cmd, needle in [
        ("pixel_sweep oracle selftest", "python3 pixel_sweep.py --selftest", "SIM VALID"),
        ("feature contract + device bridge", "python3 main.py --contract-test", "DEVICE BRIDGE OK"),
        ("calib-loop input plumbing (keys survive a not-live frame)",
         "python3 main.py --input-test", "INPUT PLUMBING OK"),
        ("worn-eye diagnostic verdicts (desk vs worn signature)",
         "python3 eye_check.py --selftest", "EYE CHECK OK"),
        ("stereo depth sensor + uncertainty (refuses reversed/epipolar/far)",
         "python3 depth.py --selftest", "DEPTH OK"),
        ("closed-form overlay geometry (counter-rotation sign, bounded gain, parallax)",
         "python3 geometry.py --selftest", "GEOMETRY OK"),
        ("velocity-adaptive smoothing (beats every fixed EMA on jitter AND lag)",
         "python3 smoothing.py --selftest", "SMOOTHING OK"),
        ("latency compensation by prediction (beats filtering AND raw at photon time)",
         "python3 predictor.py --selftest", "PREDICTOR OK"),
        ("capture/blink/fallback", "python3 main.py --capture-test", "PROVEN"),
        ("imu filter selftest", "python3 imu.py", "IMU FILTER OK"),
        ("imu serial + gyro integrator", "python3 imu_serial.py --selftest", "GYRO INTEGRATOR OK ✅"),
        ("binocular oracle", "python3 main.py --binocular-test", "BINOCULAR PHYSICS OK"),
        ("tracked-landmark switch (outer/inner)", "python3 landmark_test.py --selftest",
         "landmark_test selftest: PASS"),
        # hardware bring-up layer (headless selftests — no cameras needed)
        ("partial-bank bring-up + CPU budget", "python3 bank_bringup.py --selftest",
         "BANK BRINGUP OK"),
        ("frame budget + adaptive quality", "python3 perf.py --selftest", "PERF OK"),
        ("calibration preflight checks", "python3 calib_preflight.py --selftest",
         "CALIB PREFLIGHT OK"),
        ("canthus corpus gate (margin, not score)", "python3 canthus_data.py --selftest",
         "CANTHUS DATA OK"),
        ("canthus model runtime + mount anchor", "python3 canthus_net.py --selftest",
         "CANTHUS NET OK"),
        ("canthus auto-label (agreement never accepts alone)",
         "python3 canthus_auto.py --selftest", "CANTHUS AUTO OK"),
        ("canthus label tool (sampling spreads, deterministic)",
         "python3 canthus_label.py --selftest", "CANTHUS LABEL OK"),
        ("sync capture (barrier/jitter)", "python3 sync_capture.py --selftest", "SYNC CAPTURE OK"),
        ("auto-exposure (per-role control)", "python3 autoexpose.py --selftest", "AUTOEXPOSE OK"),
        ("world mesh tracking (VO+IMU)", "python3 world_mesh.py --selftest", "WORLD MESH OK"),
        ("camera connect/classify/persist", "python3 connect.py --selftest", "CONNECT OK"),
        ("rig bring-up metrics", "python3 rig_test.py --selftest", "RIG_TEST OK"),
        ("synchronized snapshot", "python3 snapshot.py --selftest", "SNAPSHOT OK"),
        ("integrated live rig loop", "python3 live_rig.py --selftest", "LIVE RIG OK"),
        # world-locked overlay on moving people (the "monkeys" pipeline)
        ("world-locked anchor projection", "python3 anchor.py --selftest", "ANCHOR OK"),
        ("surface placement + occlusion + persistence",
         "python3 content_anchor.py --selftest", "CONTENT ANCHOR OK"),
        ("world memory (rolling TSDF, forgets)",
         "python3 world_memory.py --selftest", "WORLD MEMORY OK"),
        ("people detect+track 3D", "python3 people_track.py --selftest", "PEOPLE TRACK OK"),
        ("avatar render/compose", "python3 avatar.py --selftest", "AVATAR OK"),
        ("augment: people->locked monkeys", "python3 augment_rig.py --selftest", "AUGMENT RIG OK"),
    ]:
        rc, out = run(cmd)
        check(name, needle in out, "" if needle in out else out.strip().splitlines()[-1][:120] if out.strip() else "no output")

    print("== 2 GEOMETRY ==")
    sys.path.insert(0, HERE)
    import numpy as np
    import rig                                        # import runs the occlusion assert
    check("rig import + occlusion assert", True)
    scad = open(SCAD).read()
    def scadval(name):
        m = re.search(rf'^{name}\s*=\s*\[([^\]]+)\]', scad, re.M)
        return [eval(x, {"ipd": rig.NOMINAL_IPD, "disp_ipd": rig.DISPLAY_IPD})
                for x in m.group(1).split(',')]
    pairs = [("WORLD_SIM", [rig.WC_X, rig.WC_UP, rig.WC_FWD]),
             ("EYE_SIM", [rig.EC_X, rig.EC_UP, rig.EC_FWD]),
             ("EYE2_SIM", [rig.EC2_X, rig.EC2_UP, rig.EC2_FWD]),
             ("PUPIL_SIM", list(rig.PUPIL_POS)),
             ("CANTH_SIM", list(rig.nominal_outer_canthus()[1])),
             ("COR_SIM", list(rig.OPTIC_R + rig.T0))]
    worst = max(max(abs(a - b) for a, b in zip(scadval(n), r)) for n, r in pairs)
    check("CAD<->rig parity (positions + aims)", worst < 5e-3, "worst %.4f mm" % worst)
    check("pantoscopic tilt hardware-achievable", abs(rig.PANTO_DEG) <= 3.5 + 1e-9,
          "PANTO_DEG=%.1f vs One Pro ±3.5° stages" % rig.PANTO_DEG)
    rc, out = run("python3 cad_fit.py")
    check("cad_fit: boards vs cone/eyeball", "ALL FIT" in out)
    rc, out = run("python3 wearable.py")
    core_bad = any(("HITS FACE" in l) and not l.strip().startswith("eye2")
                   for l in out.splitlines())
    check("wearable: CORE cams vs face", not core_bad,
          "eye2 (FULL, unprinted) excluded; eyeR TIGHT is a dry-fit item")

    print("== 3 CAD ==")
    rc, out = run(f"'{OPENSCAD}' -o /tmp/verify.csg '{SCAD}'")
    check("OpenSCAD CSG compile", rc == 0, out.strip()[:120])
    if not fast:
        rc, out = run("python3 cad_overlap.py", timeout=3600)
        check("printed-solid overlap matrix", "ALL CLEAR" in out,
              out.strip().splitlines()[-1][:120])
        for part, expected, fname in [("carrier", 1, "verify_carrier.stl"),
                                      ("ir_ring", 2, "verify_rings.stl")]:
            path = "/tmp/" + fname
            rc, out = run(f"'{OPENSCAD}' -D 'part=\"{part}\"' --export-format binstl "
                          f"-o {path} '{SCAD}'", timeout=1200)
            n = stl_shells(path) if rc == 0 and os.path.exists(path) else -1
            check(f"STL {part}: {expected} connected shell(s)", n == expected, f"got {n}")

    print("== 4 DOCS ==")
    docs = [f for f in os.listdir(ROOT) if f.endswith(".md")
            and f not in ("RESEARCH.md", "KAPPA.md", "PIXEL_SWEEP.md")]
    denylist = [
        (r"±\[32, ?30|\[24, ?-30, ?6\]|\[56, ?10, ?-2\]", "pre-2026-06-22 camera positions"),
        (r"~1\.4 ?px per eye", "optimistic accuracy claim"),
        (r"(?<!NOT )heat.set insert(?!s DON'T|s don't)", "heat-set inserts (design uses self-tap)"),
    ]
    for pat, why in denylist:
        hits = []
        for f in docs:
            body = open(os.path.join(ROOT, f)).read()
            for i, line in enumerate(body.splitlines(), 1):
                if re.search(pat, line) and "SUPERSEDED" not in line and "~~" not in line:
                    hits.append(f"{f}:{i}")
        check(f"docs free of: {why}", not hits, "; ".join(hits[:4]))

    print()
    if FAILS:
        print("=> GATE FAILED (%d): %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("=> RELEASE GATE: ALL CHECKS PASS ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main(fast="--fast" in sys.argv))
