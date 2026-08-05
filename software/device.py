"""device.py, the DeviceModel bridge: one rig the capture AND the geometry share.

Every geometry routine (physics_preset.physics_predict, preset.build_preset,
complete_geometry, match, calibrate) takes a `sim` object that carries the device
constants: each camera's extrinsics + intrinsics, the display optic + its warp, and
the display->camera misregistration (R_dc / t_dc). In simulation those come from
`rig.DEVICE_SEED`; on real hardware they come from the ONE-TIME factory rig
calibration (checkerboard through the optic, Phase 3).

This module does NOT touch the validated core. It builds a normal
`autosim.Simulator` and, when given a calibration, OVERWRITES the device-constant
fields on that instance with measured values. So:

  * `--simulate`            -> plain Simulator (DEVICE_SEED constants), unchanged.
  * real hardware           -> same Simulator class, constants loaded from
                               data/device_calib.json; used purely as geometry
                               (never observe()) since real features come from the
                               cameras via live_features.assemble_features.

The rig (camera<->display) geometry is a fixed one-time CONSTANT, not a per-use
unknown, that is exactly what a Simulator's device fields are: fixed and learnable.

CLI:
  python3 device.py --selftest      round-trip device constants, prove apply works
  python3 device.py --dump-sim      write the sim device constants as a calib schema
"""
import argparse
import json
import os
import sys

import numpy as np

import autosim

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")
DEFAULT_CALIB = os.path.join(_DATA, "device_calib.json")


# ---- (de)serialize one PinholeCamera --------------------------------------
def _cam_dict(c):
    return dict(pos=np.asarray(c.pos).tolist(), R=np.asarray(c.R).tolist(),
                f=float(c.f), k1=float(c.k1), k2=float(c.k2), res=int(c.res),
                fscale=float(c.fscale), cx=float(c.cx), cy=float(c.cy),
                p1=float(c.p1), p2=float(c.p2))


def _apply_cam(c, d):
    c.pos = np.array(d["pos"], float)
    c.R = np.array(d["R"], float)
    c.f = float(d["f"]); c.k1 = float(d["k1"]); c.k2 = float(d["k2"])
    c.res = int(d["res"]); c.fscale = float(d["fscale"])
    c.cx = float(d["cx"]); c.cy = float(d["cy"])
    c.p1 = float(d["p1"]); c.p2 = float(d["p2"])


# ---- (de)serialize the whole device ---------------------------------------
def extract_calibration(sim):
    """Snapshot every device constant of `sim` into a plain dict (JSON-able)."""
    disp = sim.rig["display"]
    return dict(
        version=1,
        world=[_cam_dict(c) for c in sim.rig["world"]],
        eye=[_cam_dict(c) for c in sim.rig["eye"]],
        eye2=[_cam_dict(c) for c in sim.rig["eye2"]],
        pupil=_cam_dict(sim.rig["pupil"]),
        display=dict(half=float(disp.half), k1=float(disp.k1), k2=float(disp.k2),
                     cx=float(disp.cx), cy=float(disp.cy),
                     virtual_dist=(None if disp.virtual_dist is None
                                   else float(disp.virtual_dist)),
                     warp=np.asarray(disp.warp).tolist()),
        R_dc=np.asarray(sim.R_dc).tolist(), t_dc=np.asarray(sim.t_dc).tolist(),
        display_eye=int(sim.rig["display_eye"]),
        optic=np.asarray(sim.rig["optic"]).tolist())


def apply_calibration(sim, cal):
    """Overwrite `sim`'s device constants from a calibration dict; returns `sim`."""
    for c, d in zip(sim.rig["world"], cal["world"]):
        _apply_cam(c, d)
    for c, d in zip(sim.rig["eye"], cal["eye"]):
        _apply_cam(c, d)
    for c, d in zip(sim.rig["eye2"], cal["eye2"]):
        _apply_cam(c, d)
    _apply_cam(sim.rig["pupil"], cal["pupil"])
    disp = sim.rig["display"]; dd = cal["display"]
    disp.half = float(dd["half"]); disp.k1 = float(dd["k1"]); disp.k2 = float(dd["k2"])
    disp.cx = float(dd["cx"]); disp.cy = float(dd["cy"])
    disp.virtual_dist = (None if dd["virtual_dist"] is None else float(dd["virtual_dist"]))
    disp.warp = np.array(dd["warp"], float)
    sim.R_dc = np.array(cal["R_dc"], float)
    sim.t_dc = np.array(cal["t_dc"], float)
    sim.rig["optic"] = np.array(cal["optic"], float)
    sim.rig["display_eye"] = int(cal["display_eye"])
    if hasattr(sim, "_uq_cams"):
        del sim._uq_cams          # invalidate physics_preset's cached un-quantized copies
    return sim


def save_calibration(sim, path=DEFAULT_CALIB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(extract_calibration(sim), fh, indent=2)
    return path


def load_calibration(path=DEFAULT_CALIB):
    with open(path) as fh:
        return json.load(fh)


# ---- the factory ----------------------------------------------------------
def build_device(simulate=True, calib_path=None, seed=0,
                 use_pupil=False, use_stereo=False):
    """Return a Simulator acting as the device.

    simulate=True  -> DEVICE_SEED constants (optionally overlaid by calib_path).
    simulate=False -> constants REQUIRED from calib_path (default device_calib.json);
                      raises if absent (run the one-time rig calibration first).
    """
    sim = autosim.Simulator(seed, use_pupil=use_pupil, use_stereo=use_stereo)
    path = calib_path if calib_path is not None else (None if simulate else DEFAULT_CALIB)
    if path and os.path.exists(path):
        apply_calibration(sim, load_calibration(path))
    elif not simulate:
        raise FileNotFoundError(
            "No device calibration at %s, run the one-time rig calibration "
            "(Phase 3) before live hardware capture." % path)
    return sim


# ----------------------------------------------------------------------
#  Self-test: prove apply_calibration genuinely restores the constants
# ----------------------------------------------------------------------
def _jitter_device(sim, rng):
    """Corrupt every device field so we can prove apply_calibration restores them."""
    for key in ("world", "eye", "eye2"):
        for c in sim.rig[key]:
            c.pos = c.pos + rng.normal(0, 2.0, 3)
            c.R = c.R + rng.normal(0, 0.02, (3, 3))
            c.f *= 1.0 + rng.normal(0, 0.05); c.fscale *= 1.0 + rng.normal(0, 0.05)
            c.k1 += rng.normal(0, 0.02); c.k2 += rng.normal(0, 0.02)
            c.cx += rng.normal(0, 0.01); c.cy += rng.normal(0, 0.01)
    p = sim.rig["pupil"]; p.pos = p.pos + rng.normal(0, 2.0, 3); p.f *= 1.05
    sim.rig["display"].warp = sim.rig["display"].warp + rng.normal(0, 0.01, 4)
    sim.rig["display"].k1 += 0.01
    sim.R_dc = sim.R_dc + rng.normal(0, 0.02, (3, 3))
    sim.t_dc = sim.t_dc + rng.normal(0, 0.5, 3)
    if hasattr(sim, "_uq_cams"):
        del sim._uq_cams


def _max_truth_diff(a, b, n=300, seed=3):
    """Max |ground_truth| difference between two devices over shared subjects/poses."""
    rng = np.random.default_rng(seed)
    a.rng = np.random.default_rng(seed + 1)
    maxd = 0.0; got = 0
    while got < n:
        subj = a.new_subject(); dev = a.seat(); P = a.world_point()
        ga = a.ground_truth(subj, dev, P); gb = b.ground_truth(subj, dev, P)
        if ga is None or gb is None:
            continue
        got += 1
        maxd = max(maxd, float(np.abs(ga[0] - gb[0]).max()),
                   float(np.abs(ga[1] - gb[1]).max()))
    return maxd


def selftest(verbose=True):
    if verbose:
        print("== device calibration round-trip self-test ==")
    ok = True
    for use_pupil, use_stereo in [(False, False), (True, True)]:
        ref = autosim.Simulator(0, use_pupil=use_pupil, use_stereo=use_stereo)
        cal = json.loads(json.dumps(extract_calibration(ref)))   # through JSON

        # a fresh, then CORRUPTED, device: confirm it really differs from ref
        dut = autosim.Simulator(0, use_pupil=use_pupil, use_stereo=use_stereo)
        _jitter_device(dut, np.random.default_rng(99))
        before = _max_truth_diff(ref, dut)

        apply_calibration(dut, cal)                              # restore from JSON
        after = _max_truth_diff(ref, dut)

        # ground_truth is normalized [0,1]; corruption moves it by a large fraction,
        # a correct restore returns it to machine zero.
        passed = before > 0.01 and after < 1e-9
        ok = ok and passed
        if verbose:
            print("  use_pupil=%-5s use_stereo=%-5s  corrupted diff=%.3f (norm), "
                  "restored diff=%.1e  [%s]"
                  % (use_pupil, use_stereo, before, after, "PASS" if passed else "FAIL"))

    # build_device contract: sim default works; hardware without a file refuses
    d = build_device(simulate=True)
    sim_ok = d is not None
    try:
        build_device(simulate=False, calib_path=os.path.join(_DATA, "_no_such.json"))
        hw_ok = False
    except FileNotFoundError:
        hw_ok = True
    ok = ok and sim_ok and hw_ok
    if verbose:
        print("  build_device(simulate=True) -> Simulator [%s];  "
              "(simulate=False, missing file) refuses [%s]"
              % ("PASS" if sim_ok else "FAIL", "PASS" if hw_ok else "FAIL"))
        print("  =>", "DEVICE BRIDGE OK, constants are a portable, loadable artifact ✅"
              if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dump-sim", action="store_true",
                   help="write the simulator's device constants to device_calib.json "
                        "(a schema/template; the real file comes from rig calibration)")
    p.add_argument("--use-pupil", action="store_true")
    p.add_argument("--use-stereo", action="store_true")
    a = p.parse_args(argv)
    if a.dump_sim:
        sim = autosim.Simulator(0, use_pupil=a.use_pupil, use_stereo=a.use_stereo)
        path = save_calibration(sim)
        print("wrote sim device constants ->", path)
        return 0
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
