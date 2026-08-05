"""snapshot.py, grab ONE synchronized frame-set from all rig cameras and save it to disk.

The workhorse capture primitive: every calibration sample, debug capture, and dataset frame is a
synchronized set of the 6 role images plus the metadata that says how trustworthy it is
(timestamps, measured jitter, per-role focus/brightness). It writes:

    data/snapshots/<stamp>/worldL.png worldR.png eyeL.png ... pupilR.png
    data/snapshots/<stamp>/meta.json   # jitter_ms, per-role capture ts + focus + brightness, map

A set whose jitter exceeds the budget is still saved but flagged `synced=false` in meta.json, so
nothing downstream mistakes a smeared set for a clean calibration sample.

    python3 snapshot.py --selftest         # headless: writes a synthetic set, verifies meta
    python3 snapshot.py                     # one synchronized set from the real cameras
    python3 snapshot.py --burst 20          # 20 sets back-to-back (e.g. for a calibration sweep)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from rig_test import focus_score, brightness

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")
SNAP_DIR = os.path.join(_DATA, "snapshots")


def build_meta(sync_frame, role_index, budget_ms=3.0):
    """Assemble the metadata dict for one SyncFrame (no disk I/O, testable)."""
    roles = {}
    for role in sync_frame.frames:
        f = sync_frame.get(role)
        roles[role] = dict(
            index=role_index.get(role) if role_index else None,
            present=f is not None,
            capture_ts=sync_frame.ts.get(role),
            seq=sync_frame.seqs.get(role),
            focus=focus_score(f) if f is not None else None,
            brightness=brightness(f) if f is not None else None,
        )
    return dict(
        stamp=time.strftime("%Y%m%d-%H%M%S"),
        jitter_ms=sync_frame.jitter_ms,
        complete=sync_frame.complete,
        synced=sync_frame.within(budget_ms),
        budget_ms=budget_ms,
        roles=roles,
    )


def write_set(sync_frame, role_index, out_dir, budget_ms=3.0, imwrite=None):
    """Write one synchronized set: a PNG per present role + meta.json. `imwrite(path, frame)`
    defaults to cv2.imwrite; injectable for the selftest. Returns (out_dir, meta)."""
    os.makedirs(out_dir, exist_ok=True)
    if imwrite is None:
        import cv2
        imwrite = cv2.imwrite
    meta = build_meta(sync_frame, role_index, budget_ms)
    for role in sync_frame.frames:
        f = sync_frame.get(role)
        if f is not None:
            imwrite(os.path.join(out_dir, "%s.png" % role), f)
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2, default=_json_default)
    return out_dir, meta


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(o)


def capture(n=1, budget_ms=3.0, fps=100, root=SNAP_DIR, verbose=True):
    """Open the mapped bank and save `n` synchronized sets. Needs opencv + a saved role map."""
    from connect import load_map, validate_map
    from sync_capture import SyncBank
    role_index = load_map()
    if not role_index:
        print("no role map, run:  python3 connect.py --auto")
        return 1
    probs = validate_map(role_index)
    if probs:
        print("role map invalid:", "; ".join(probs)); return 1
    bank = SyncBank(role_index, fps=fps).start()
    saved = 0
    try:
        for _ in range(n):
            fs = bank.sync_frame()
            out = os.path.join(root, time.strftime("%Y%m%d-%H%M%S-") + ("%03d" % (saved)))
            _, meta = write_set(fs, role_index, out, budget_ms)
            saved += 1
            if verbose:
                flag = "OK" if meta["synced"] else ("INCOMPLETE" if not meta["complete"] else "JITTER")
                print("  saved %s  jitter %.2f ms  [%s]" % (out, meta["jitter_ms"], flag))
    finally:
        bank.close()
    return 0


# ==========================================================================
#  Self-test (no hardware): a fake SyncFrame is written + its meta verified.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== snapshot self-test (synthetic sync set, no hardware) ==")
    import tempfile
    from sync_capture import SyncFrame
    rng = np.random.default_rng(0)
    roles = ("worldL", "worldR", "eyeL", "eyeR", "pupilL", "pupilR")
    role_index = {r: i for i, r in enumerate(roles)}
    checks = []

    # a well-synchronized set: timestamps within ~1 ms
    frames = {r: (rng.random((48, 64, 3) if r.startswith("world") else (48, 64)) * 255).astype(np.uint8)
              for r in roles}
    ts = {r: 100.0 + 0.0003 * i for i, r in enumerate(roles)}     # ~1.5 ms spread
    fs = SyncFrame(frames, ts, {r: 5 for r in roles}, time.monotonic)

    written = {}
    def fake_imwrite(path, frame):
        written[os.path.basename(path)] = np.asarray(frame).shape
        return True

    with tempfile.TemporaryDirectory() as td:
        out, meta = write_set(fs, role_index, os.path.join(td, "set0"), budget_ms=3.0,
                              imwrite=fake_imwrite)
        # (1) a PNG written per role + a meta.json
        pngs = all("%s.png" % r in written for r in roles)
        has_meta = os.path.exists(os.path.join(out, "meta.json"))
        checks.append(("writes a PNG per role + meta.json", pngs and has_meta))

        # (2) meta records jitter, marks the set synced, and carries per-role focus/brightness/ts
        with open(os.path.join(out, "meta.json")) as fp:
            m = json.load(fp)
        meta_ok = (m["synced"] and m["complete"] and m["jitter_ms"] < 3.0
                   and all(m["roles"][r]["present"] and m["roles"][r]["focus"] is not None
                           and m["roles"][r]["capture_ts"] is not None for r in roles))
        checks.append(("meta records jitter, synced flag, per-role focus/brightness/ts", meta_ok))

        # (3) an OVER-BUDGET set is saved but flagged synced=false (never mistaken for clean)
        ts_bad = {r: 100.0 + 0.01 * i for i, r in enumerate(roles)}   # ~50 ms spread
        fs_bad = SyncFrame(frames, ts_bad, {r: 6 for r in roles}, time.monotonic)
        _, mbad = write_set(fs_bad, role_index, os.path.join(td, "set1"), budget_ms=3.0,
                            imwrite=fake_imwrite)
        checks.append(("over-budget set saved but flagged synced=false",
                       (not mbad["synced"]) and mbad["jitter_ms"] > 3.0))

        # (4) an INCOMPLETE set (a dropped role) is handled + flagged
        frames_missing = dict(frames); frames_missing["pupilR"] = None
        ts_missing = dict(ts); ts_missing["pupilR"] = None
        fs_miss = SyncFrame(frames_missing, ts_missing, {r: 7 for r in roles}, time.monotonic)
        _, mmiss = write_set(fs_miss, role_index, os.path.join(td, "set2"), imwrite=fake_imwrite)
        checks.append(("incomplete set (missing pupilR) flagged, no crash",
                       (not mmiss["complete"]) and (not mmiss["roles"]["pupilR"]["present"])))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "SNAPSHOT OK, synchronized set + trustworthy metadata ✅"
              if ok else "PROBLEM ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="save synchronized snapshot set(s) from the rig")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--burst", type=int, default=1, help="how many sets to grab")
    ap.add_argument("--budget-ms", type=float, default=3.0)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    sys.exit(capture(n=args.burst, budget_ms=args.budget_ms))
