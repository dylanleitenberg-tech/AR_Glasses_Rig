"""intrinsics.py — per-camera checkerboard calibration, ready for the day the cameras arrive.

Each real camera's detections must be mapped into the canonical normalized PinholeCamera
convention the sim/calibrator use (the P0 contract): that mapping IS the intrinsic calibration
(fx fy cx cy + distortion). This tool captures checkerboard views per ROLE, solves with
cv2.calibrateCamera, and stores the result in data/intrinsics.json — the piece device.py's
one-time factory calibration consumes for the camera side (extrinsics = the separate
checkerboard-in-common-view step, Phase 3).

USAGE (per camera, once, before mounting is fine — intrinsics don't change with position):
    python3 intrinsics.py --role eyeR --index 1            # live capture: SPACE=grab, C=calibrate
    python3 intrinsics.py --role eyeR --index 1 --auto     # auto-grab a view every ~1.5 s
    python3 intrinsics.py --board                          # write a printable checkerboard PNG

Board: 9x6 INNER corners, 20 mm squares (edit BOARD_*). Print data/checkerboard_9x6_20mm.png
at 100% scale (A4/letter, landscape), tape it FLAT to something rigid, verify one square is
20.0 mm with calipers. 15-25 varied views per camera (tilt it, fill corners, near+far).

Quality gate: RMS reprojection < 0.5 px is good for the 640 eye cams; < 0.8 px for the 1280
world cams. Re-run any camera above that with more/better views.
"""
import argparse
import json
import os
import time

BOARD_COLS = 9      # inner corners across
BOARD_ROWS = 6      # inner corners down
SQUARE_MM = 20.0

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(_DATA, "intrinsics.json")


def write_board_png(path=None, px_per_square=120):
    """Printable checkerboard: (BOARD_COLS+1)x(BOARD_ROWS+1) squares + a caliper scale note."""
    import numpy as np
    try:
        import cv2
    except ImportError:
        cv2 = None
    cols, rows = BOARD_COLS + 1, BOARD_ROWS + 1
    img = np.full(((rows + 1) * px_per_square, (cols + 1) * px_per_square), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = (r + 1) * px_per_square, (c + 1) * px_per_square
                # margin row/col keeps a white border for detection + print bleed
                img[y - px_per_square:y, x - px_per_square:x] = 0
    path = path or os.path.join(_DATA, "checkerboard_%dx%d_%dmm.png"
                                % (BOARD_COLS, BOARD_ROWS, int(SQUARE_MM)))
    os.makedirs(_DATA, exist_ok=True)
    if cv2 is not None:
        cv2.putText(img, "print at 100%% scale; 1 square = %.0f mm (verify with calipers)"
                    % SQUARE_MM, (px_per_square // 2, img.shape[0] - px_per_square // 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        cv2.imwrite(path, img)
    else:                                   # PNG via PIL fallback, else PGM
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
        except ImportError:
            path = path.rsplit(".", 1)[0] + ".pgm"
            with open(path, "wb") as f:
                f.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
                f.write(img.tobytes())
    print("wrote %s  (%d x %d inner corners, %.0f mm squares)"
          % (path, BOARD_COLS, BOARD_ROWS, SQUARE_MM))
    return path


def calibrate_role(role, index, width=None, height=None, auto=False, min_views=15):
    """Live capture + solve for one camera role. Saves/updates data/intrinsics.json."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit("camera index %d did not open — run: python3 main.py --list-cams" % index)

    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_MM
    obj_pts, img_pts, shape = [], [], None
    last_grab = 0.0
    print("[%s @ index %d]  SPACE=grab view   C=calibrate (>=%d views)   Q=quit"
          % (role, index, min_views))
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, (BOARD_COLS, BOARD_ROWS), corners, found)
        cv2.putText(disp, "%s: %d views" % (role, len(obj_pts)), (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("intrinsics " + role, disp)
        k = cv2.waitKey(1) & 0xFF
        grab = (k == ord(' ')) or (auto and found and time.time() - last_grab > 1.5)
        if grab and found:
            c2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            obj_pts.append(objp); img_pts.append(c2); last_grab = time.time()
            print("  grabbed view %d" % len(obj_pts))
        if k == ord('c') and len(obj_pts) >= min_views:
            break
        if k == ord('q'):
            if len(obj_pts) < min_views:
                print("aborted (%d views, need %d)" % (len(obj_pts), min_views))
                cap.release(); cv2.destroyAllWindows(); return None
            break
    cap.release(); cv2.destroyAllWindows()

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, shape, None, None)
    entry = {"role": role, "index": index, "width": shape[0], "height": shape[1],
             "rms_px": float(rms), "n_views": len(obj_pts),
             "fx": float(K[0, 0]), "fy": float(K[1, 1]),
             "cx": float(K[0, 2]), "cy": float(K[1, 2]),
             "dist": [float(d) for d in dist.ravel()],
             "board": [BOARD_COLS, BOARD_ROWS, SQUARE_MM],
             "when": time.strftime("%Y-%m-%d %H:%M")}
    gate = 0.8 if shape[0] >= 1280 else 0.5
    print("RMS reprojection: %.3f px  (%s %.1f px gate)"
          % (rms, "PASS, under" if rms < gate else "OVER the", gate))
    data = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)
    data[role] = entry
    os.makedirs(_DATA, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print("saved -> %s  (%d roles calibrated)" % (OUT, len(data)))
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", action="store_true", help="write the printable checkerboard image")
    ap.add_argument("--role", type=str, help="camera role (worldL/worldR/eyeL/eyeR/pupilL/pupilR/eye2L/eye2R)")
    ap.add_argument("--index", type=int, help="UVC device index (see main.py --list-cams)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--auto", action="store_true", help="auto-grab a found board every 1.5 s")
    a = ap.parse_args()
    if a.board:
        write_board_png()
        return
    if not a.role or a.index is None:
        ap.error("--role and --index required (or --board)")
    calibrate_role(a.role, a.index, a.width, a.height, a.auto)


if __name__ == "__main__":
    main()
