"""connect.py — plug the 6 rig cameras into the Mac and map each to its ROLE, once.

The whole pipeline addresses cameras by ROLE (worldL/R, eyeL/R, pupilL/R), but macOS hands out
opaque UVC indices in arbitrary order, so something has to decide which index is which camera.
This module does that and PERSISTS it to data/rig_cameras.json, so every other tool
(sync_capture, rig_test, snapshot, capture) just loads the map.

CLASSIFY (no user input): each camera is probed and its frame measured —
  * COLOR vs MONO   : real chroma (mean |channel spread|) → the 2 AR0234 world cams are color,
                      the 4 OV9281 eye/pupil cams are mono (NoIR).
  * RESOLUTION      : the world boards report 1280-class, the OV9281s 640-class.
That splits the 6 into {2 world} + {4 eye/pupil} automatically.

DISAMBIGUATE within a class (needs the human or the LEDs, briefly):
  * world L vs R and eye vs pupil, L vs R can't be told from a static frame. Two ways:
      - IR-STROBE test: pulse the IR LEDs (firmware/ir_strobe) — only the 2 PUPIL cams see a
        bright jump (they point at the IR-lit eye); that separates pupil from eye.
      - assisted: `--identify` shows each camera live, labeled, and you press the role key.
  * The final L/R split uses the assisted step or the physical USB order you confirm.

Runs headless: `--selftest` feeds synthetic color/mono frames through the classifier and proves
the split; the interactive/HW paths need opencv + the cameras.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")
MAP_PATH = os.path.join(_DATA, "rig_cameras.json")

CORE_ROLES = ("worldL", "worldR", "eyeL", "eyeR", "pupilL", "pupilR")


# --------------------------------------------------------------------------
#  Frame classification (pure numpy — the testable core)
# --------------------------------------------------------------------------
def classify_frame(frame):
    """Describe one probed frame: {is_color, res_class, chroma, brightness, w, h}.
    `is_color` = the sensor delivers real chroma (not a mono frame packed into 3 equal channels).
    `res_class` = 1280 (world) or 640 (eye/pupil) by nearest width."""
    a = np.asarray(frame)
    h, w = a.shape[:2]
    if a.ndim == 3 and a.shape[2] >= 3:
        b, g, r = [a[:, :, i].astype(np.float32) for i in range(3)]
        chroma = float((np.abs(r - g) + np.abs(g - b)).mean()) / 255.0
        bright = float(a.mean()) / 255.0
    else:
        chroma = 0.0
        bright = float(a.mean()) / 255.0
    is_color = chroma > 0.02                       # >~5 gray-levels of channel spread => real color
    res_class = 1280 if w >= 960 else 640
    return dict(is_color=is_color, res_class=res_class, chroma=chroma,
                brightness=bright, w=int(w), h=int(h))


def classify_bank(descs):
    """Given {index: classify_frame(...)}, split indices into role CLASSES (not yet L/R):
    returns {'world': [idx...], 'eye_or_pupil': [idx...], 'unknown': [...]}.
    World = color AND 1280-class; the mono 640s are the eye/pupil pool."""
    world, mono, unknown = [], [], []
    for idx, d in descs.items():
        if d["is_color"] and d["res_class"] == 1280:
            world.append(idx)
        elif not d["is_color"]:
            mono.append(idx)
        else:
            unknown.append(idx)
    return {"world": sorted(world), "eye_or_pupil": sorted(mono), "unknown": sorted(unknown)}


def separate_pupils(mono_indices, dark_desc, lit_desc, min_jump=0.06):
    """Tell the 2 PUPIL cams from the 2 EYE-corner cams using an IR-strobe A/B test.
    dark_desc/lit_desc: {index: classify_frame} with the IR LEDs OFF then ON. The pupil cams
    face the IR-lit eye, so their brightness JUMPS; the eye-corner cams barely change.
    Returns (pupil_indices, eye_indices)."""
    jumps = {i: lit_desc[i]["brightness"] - dark_desc[i]["brightness"] for i in mono_indices}
    ordered = sorted(mono_indices, key=lambda i: jumps[i], reverse=True)
    pupils = [i for i in ordered if jumps[i] >= min_jump][:2]
    if len(pupils) < 2:                            # fall back to the 2 biggest jumps
        pupils = ordered[:2]
    eyes = [i for i in mono_indices if i not in pupils]
    return sorted(pupils), sorted(eyes)


# --------------------------------------------------------------------------
#  Persisted role map
# --------------------------------------------------------------------------
def save_map(role_index, path=MAP_PATH, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {"roles": {r: int(i) for r, i in role_index.items()}, "meta": meta or {}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


def load_map(path=MAP_PATH):
    """Return {role: index} or None if not yet mapped."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r: int(i) for r, i in json.load(f)["roles"].items()}


def validate_map(role_index):
    """Sanity-check a role map before it's trusted by the pipeline."""
    problems = []
    for r in role_index:
        if r not in CORE_ROLES:
            problems.append("unknown role %r" % r)
    idxs = list(role_index.values())
    if len(set(idxs)) != len(idxs):
        problems.append("two roles share one camera index")
    missing = [r for r in CORE_ROLES if r not in role_index]
    if missing:
        problems.append("missing roles: %s" % ", ".join(missing))
    return problems


# --------------------------------------------------------------------------
#  Hardware paths (opencv) — auto-classify + assisted identify
# --------------------------------------------------------------------------
def _probe_indices(max_index=10):
    """Open each UVC index, grab a frame, classify it. Returns {index: (frame, desc)}."""
    import cv2
    out = {}
    for i in range(max_index):
        cap = cv2.VideoCapture(i, getattr(cv2, "CAP_AVFOUNDATION", 0))
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)     # request high; sensor caps to its real max
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            out[i] = (frame, classify_frame(frame))
    return out


def auto_map(max_index=10, verbose=True):
    """Best-effort automatic role map from a single probe (no LEDs, no user). Splits world vs
    mono by classification; within each pair assigns L/R by ascending index (confirm/rewire with
    --identify if mirrored). Returns (role_index, notes)."""
    probe = _probe_indices(max_index)
    descs = {i: d for i, (_, d) in probe.items()}
    groups = classify_bank(descs)
    notes = []
    role_index = {}
    world = groups["world"]
    mono = groups["eye_or_pupil"]
    if len(world) >= 2:
        role_index["worldL"], role_index["worldR"] = world[0], world[1]
    else:
        notes.append("found %d world (color/1280) cams, need 2" % len(world))
    # Without an IR strobe we can't split pupil vs eye from one probe; leave that to --identify
    # or separate_pupils(). Assign the mono pool provisionally by index order.
    if len(mono) >= 4:
        role_index.update(dict(zip(("eyeL", "eyeR", "pupilL", "pupilR"), mono[:4])))
        notes.append("mono cams assigned by index order — verify eye vs pupil with --strobe or --identify")
    else:
        notes.append("found %d mono (NoIR/640) cams, need 4" % len(mono))
    if verbose:
        print("auto-map probe: world=%s mono=%s unknown=%s" %
              (world, mono, groups["unknown"]))
        for n in notes:
            print("  note:", n)
    return role_index, notes


def identify_interactive(max_index=10):
    """Assisted mapping: show each camera live and press a role key to assign it. Writes the map."""
    import cv2
    probe = _probe_indices(max_index)
    indices = sorted(probe)
    if not indices:
        print("no cameras found on indices 0..%d" % (max_index - 1))
        return 1
    keymap = {"1": "worldL", "2": "worldR", "3": "eyeL", "4": "eyeR", "5": "pupilL", "6": "pupilR"}
    print("keys: " + "  ".join("%s=%s" % (k, v) for k, v in keymap.items()) + "   n=next  s=save  q=quit")
    role_index = {}
    win = "connect — assign roles"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    pos = 0
    while True:
        idx = indices[pos % len(indices)]
        cap = cv2.VideoCapture(idx, getattr(cv2, "CAP_AVFOUNDATION", 0)) or cv2.VideoCapture(idx)
        ok, frame = cap.read(); cap.release()
        if not ok:
            pos += 1; continue
        d = classify_frame(frame)
        assigned = [r for r, i in role_index.items() if i == idx]
        cv2.putText(frame, "index %d  %s %dp  %s" % (idx, "color" if d["is_color"] else "mono",
                    d["res_class"], ("-> " + assigned[0]) if assigned else "unassigned"),
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        k = chr(cv2.waitKey(30) & 0xFF)
        if k in keymap:
            role_index = {r: i for r, i in role_index.items() if i != idx}   # unbind idx elsewhere
            role_index[keymap[k]] = idx
            print("  index %d -> %s" % (idx, keymap[k]))
        elif k == "n":
            pos += 1
        elif k == "s":
            probs = validate_map(role_index)
            if probs:
                print("  cannot save:", "; ".join(probs)); continue
            save_map(role_index, meta={"method": "identify_interactive"})
            print("  saved", MAP_PATH); break
        elif k in ("q", "\x1b"):
            break
    cv2.destroyAllWindows()
    return 0


# ==========================================================================
#  Self-test (no hardware): synthetic color/mono frames prove the classifier,
#  the class split, the IR-strobe pupil separation, and map persistence.
# ==========================================================================
def _fake_frame(kind, rng):
    if kind == "world":                            # color, 1280-class, real chroma
        f = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
        f[:, :, 0] = np.clip(f[:, :, 0].astype(int) + 40, 0, 255)   # blue-ish scene tint
        return f
    if kind in ("eye", "pupil"):                   # mono 640, packed to 3 equal channels
        g = rng.integers(0, 255, (480, 640), dtype=np.uint8)
        return np.stack([g, g, g], axis=2)
    raise ValueError(kind)


def selftest(verbose=True):
    if verbose:
        print("== connect self-test (synthetic frames, no hardware) ==")
    rng = np.random.default_rng(4)
    checks = []

    # (1) classifier separates color/1280 from mono/640
    cw = classify_frame(_fake_frame("world", rng))
    ce = classify_frame(_fake_frame("eye", rng))
    checks.append(("classify: world=color/1280, eye=mono/640",
                   cw["is_color"] and cw["res_class"] == 1280
                   and (not ce["is_color"]) and ce["res_class"] == 640))

    # (2) class split of a full 6-cam bank (indices in arbitrary order)
    kinds = {0: "eye", 1: "world", 2: "pupil", 3: "eye", 4: "world", 5: "pupil"}
    descs = {i: classify_frame(_fake_frame(k, rng)) for i, k in kinds.items()}
    groups = classify_bank(descs)
    split_ok = (sorted(groups["world"]) == [1, 4]
                and sorted(groups["eye_or_pupil"]) == [0, 2, 3, 5]
                and groups["unknown"] == [])
    checks.append(("6-cam bank splits into 2 world + 4 mono (any index order)", split_ok))

    # (3) IR-strobe A/B separates the 2 pupil cams (they brighten) from the eye cams
    mono = [0, 2, 3, 5]                             # 2,5 are the pupils in `kinds`
    dark = {i: dict(brightness=0.10) for i in mono}
    lit = {i: dict(brightness=0.10 + (0.20 if kinds[i] == "pupil" else 0.01)) for i in mono}
    pupils, eyes = separate_pupils(mono, dark, lit)
    checks.append(("IR-strobe A/B picks the 2 pupil cams (brightness jump)",
                   pupils == [2, 5] and eyes == [0, 3]))

    # (4) map validation catches dup/missing
    good = {r: i for i, r in enumerate(CORE_ROLES)}
    bad_dup = dict(good); bad_dup["worldR"] = bad_dup["worldL"]
    checks.append(("validate_map: accepts a full unique map, rejects a duplicate",
                   validate_map(good) == [] and validate_map(bad_dup)))

    # (5) round-trip persistence
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rig_cameras.json")
        save_map(good, p, meta={"method": "selftest"})
        checks.append(("save_map/load_map round-trips the role map", load_map(p) == good))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "CONNECT OK — auto-classify + IR-strobe pupil split + persistence ✅"
              if ok else "PROBLEM ⚠️")
        print("  on hardware: `python3 connect.py --auto` then `--identify` to confirm L/R, then")
        print("  every tool loads data/rig_cameras.json via connect.load_map().")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="map the 6 rig cameras to roles + persist")
    ap.add_argument("--selftest", action="store_true", help="headless classifier test")
    ap.add_argument("--auto", action="store_true", help="probe + auto role map, save it")
    ap.add_argument("--identify", action="store_true", help="assisted live role assignment")
    ap.add_argument("--show", action="store_true", help="print the saved map")
    ap.add_argument("--max-index", type=int, default=10)
    args = ap.parse_args()
    if args.show:
        print(load_map() or "no map saved yet (run --auto or --identify)")
        sys.exit(0)
    if args.auto:
        ri, _ = auto_map(args.max_index)
        probs = validate_map(ri)
        if probs:
            print("incomplete/invalid:", "; ".join(probs), "\n-> run --identify to finish")
            sys.exit(1)
        print("saved", save_map(ri, meta={"method": "auto_map"}))
        sys.exit(0)
    if args.identify:
        sys.exit(identify_interactive(args.max_index))
    sys.exit(selftest())
