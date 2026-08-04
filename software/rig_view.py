"""Live 4-up view of the whole rig, with every preflight detection drawn on it.

WHY THIS EXISTS
    calib_preflight tells you WHICH row failed, then you adjust something, re-run, and find out
    whether you guessed right. On 2026-08-03 that loop ran a dozen times: the target was flat, then
    too small, then beside a laptop keyboard, then out of frame, then the rig was not on a face.
    Each answer arrived one run too late, and several times the fix addressed a problem that had
    already moved on.

    Preconditions that must ALL hold simultaneously are far easier to satisfy when you can watch
    them at once. This shows the two world cams with the dot detection marked, the two eye cams
    with the model's landmark and its state, and a live PASS/FAIL line for the same geometry tests
    the preflight runs -- so you position the target and the glasses until everything reads green,
    and only then run the preflight to confirm.

    python3 rig_view.py --roles worldL=3 worldR=2 eyeL=0 eyeR=1
"""
import sys

import numpy as np


def _enhance(g, cv2):
    """Gentle lift for the human eye only; detection always runs on the raw frame."""
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    out = np.clip((g.astype(np.float32) - lo) * 235.0 / max(hi - lo, 1e-6) + 10, 0, 255)
    return out.astype(np.uint8)


def run(roles, tile=(480, 300)):
    import cv2
    from cameras import Camera
    from dot_detector import DotDetector
    from calib_preflight import dot_geometry
    from canthus_net import CanthusTracker

    # 640x480, NOT native. Four 1280x800 streams do not fit on this rig's shared USB 2.0 bus:
    # measured, native mode delivers ~1 fps with TWO cameras producing no frames at all, while
    # 640x480 gives ~40 fps and zero misses. This is a settled fact of the hardware
    # (SyncBank.ROLE_MODE encodes it) and I broke it here -- the symptom was exactly as recorded,
    # a crawling view with the world cams apparently "not connected".
    #
    # Nothing is lost for this purpose: the view exists to position the target and the glasses,
    # and every detector downstream is scale-invariant in normalised coordinates.
    # PER-SENSOR modes from cameras.ROLE_MODE — never one size for every role. The eye (OV9281)
    # and world (AR0234) sensors accept DIFFERENT modes, and an unsupported request silently
    # returns native, saturating the bus. Hardcoding 640x400 here gave the world cams 1920x1200
    # and starved half the bank: "only 2 cams are running".
    from cameras import ROLE_MODE
    cams = {r: Camera(i, *ROLE_MODE.get(r, (640, 480)), name=r) for r, i in roles.items()}
    det = DotDetector()
    trk = {"eyeL": CanthusTracker(mirrored=True), "eyeR": CanthusTracker(mirrored=False)}
    win = "rig view — position everything until all four read green.  q quits"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        frames = {r: c.read() for r, c in cams.items()}
        tiles, dots, states = {}, {}, {}

        # JOINT stereo pick, so this view shows what the LIVE LOOP will actually use. Each camera
        # taking its own argmax is how the two cams end up locked onto different furniture.
        _pair = {}
        _fl, _fr = frames.get("worldL"), frames.get("worldR")
        if _fl is not None and _fr is not None:
            try:
                _L, _R = det.detect_pair(_fl, _fr)
                if _L is not None:
                    _pair = {"worldL": _L, "worldR": _R}
            except Exception:
                _pair = {}

        for r in ("worldL", "worldR"):
            f = frames.get(r)
            if f is None:
                tiles[r] = np.zeros((tile[1], tile[0], 3), np.uint8)
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            d = _pair.get(r)
            vis = cv2.cvtColor(_enhance(g, cv2), cv2.COLOR_GRAY2BGR)
            H, W = vis.shape[:2]
            if d is not None:
                dots[r] = d
                cv2.drawMarker(vis, (int(d[0] * W), int(d[1] * H)), (0, 255, 0),
                               cv2.MARKER_CROSS, 70, 4)
                cv2.circle(vis, (int(d[0] * W), int(d[1] * H)), 44, (0, 255, 0), 3)
            txt = "%s  %s" % (r, ("dot %.3f,%.3f" % d) if d is not None else "NO DOT")
            cv2.putText(vis, txt, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if d is not None else (0, 0, 255), 3)
            tiles[r] = cv2.resize(vis, tile)

        for r in ("eyeL", "eyeR"):
            f = frames.get(r)
            if f is None:
                tiles[r] = np.zeros((tile[1], tile[0], 3), np.uint8)
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            uv, score = trk[r].track(g)
            states[r] = score
            vis = cv2.cvtColor(_enhance(g, cv2), cv2.COLOR_GRAY2BGR)
            H, W = vis.shape[:2]
            col = (0, 255, 0) if score >= 1.0 else ((0, 200, 255) if score > 0 else (0, 0, 255))
            if uv is not None:
                cv2.drawMarker(vis, (int(uv[0] * W), int(uv[1] * H)), col,
                               cv2.MARKER_CROSS, 70, 4)
            lab = "OK" if score >= 1.0 else ("HELD" if score > 0 else "NO LANDMARK")
            cv2.putText(vis, "%s  %s%s" % (r, lab,
                                           ("  %.3f,%.3f" % uv) if uv is not None else ""),
                        (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            tiles[r] = cv2.resize(vis, tile)

        # live geometry verdict, same tests the preflight applies
        verdict, vcol = "WORLD DOT: need both cams", (0, 0, 255)
        if "worldL" in dots and "worldR" in dots:
            depth, dv, warns = dot_geometry(dots["worldL"], dots["worldR"])
            if warns:
                verdict = "WORLD DOT: " + warns[0].split(" -- ")[0][:78]
            else:
                verdict = "WORLD DOT OK  depth %s  dv %.3f" % (
                    "inf" if depth is None else "%.0f mm" % depth, dv)
                vcol = (0, 255, 0)

        eyes_ok = all(states.get(r, 0.0) >= 1.0 for r in ("eyeL", "eyeR"))
        grid = np.vstack([np.hstack([tiles["worldL"], tiles["worldR"]]),
                          np.hstack([tiles["eyeL"], tiles["eyeR"]])])
        bar = np.zeros((70, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, verdict, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vcol, 2)
        cv2.putText(bar, "EYES: %s" % ("both OK" if eyes_ok else "not both locked"),
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if eyes_ok else (0, 0, 255), 2)
        cv2.imshow(win, np.vstack([grid, bar]))
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
            break

    for c in cams.values():
        c.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    import argparse
    from bank_bringup import parse_roles
    ap = argparse.ArgumentParser(description="live 4-up rig view with preflight detections")
    ap.add_argument("--roles", nargs="*", default=None, metavar="ROLE=INDEX")
    a = ap.parse_args()
    roles = parse_roles(a.roles) if a.roles else {"worldL": 3, "worldR": 2,
                                                 "eyeL": 0, "eyeR": 1}
    sys.exit(run(roles))
