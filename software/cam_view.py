"""cam_view.py — plug a USB camera into the laptop and see its live feed.

The simple bring-up tool: confirm a camera enumerates, run the IR remote test on the
NoIR cameras, and set focus. Independent of the rig contract — it just opens a UVC camera
and shows it.

    python3 cam_view.py            # open the first working camera
    python3 cam_view.py 2          # open camera index 2
    python3 cam_view.py --list     # just list which indices respond

In the window:
    0-9   switch to that camera index        s   save a snapshot (PNG, next to this file)
    i     toggle the info + brightness probe  f   flip the image (mirror)
    q/Esc quit

IR REMOTE TEST (proves a camera is NoIR / IR-sensitive): open the feed, point any TV/AC
remote at the lens, hold a button. If a bright white/purple dot flashes at the emitter,
the camera sees near-IR and is good for the eye-tracking role. If nothing shows, that lens
has an IR-cut filter. Watch the CENTER BRIGHTNESS number (press `i`): it spikes when the
camera catches the pulse.
"""
import os
import sys
import time


def _need_cv2():
    try:
        import cv2  # noqa
        return cv2
    except ImportError:
        sys.exit("OpenCV is not installed. In the software venv:\n"
                 "  python3 -m venv .venv && source .venv/bin/activate\n"
                 "  pip install opencv-python\n"
                 "then re-run:  python3 cam_view.py")


def _open(cv2, index):
    """Open a camera index with the Mac-friendly backend, falling back to the default."""
    for backend in (getattr(cv2, "CAP_AVFOUNDATION", 0), cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def list_cameras(cv2, n=8):
    print("probing camera indices 0-%d ..." % (n - 1))
    found = []
    for i in range(n):
        cap = _open(cv2, i)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("  index %d : OK  %dx%d" % (i, w, h))
            found.append(i)
            cap.release()
        else:
            print("  index %d : -" % i)
    if not found:
        print("no cameras responded. Check the USB-C adapter and that the cable is data-capable.")
    return found


def run(index=0):
    cv2 = _need_cv2()
    cap = _open(cv2, index)
    if cap is None:
        print("could not open camera %d; probing for one that works ..." % index)
        found = list_cameras(cv2)
        if not found:
            return 1
        index = found[0]
        cap = _open(cv2, index)

    win = "cam_view — 0-9 switch  s snap  i info  f flip  q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_info, flip = True, False
    t_prev, fps = time.time(), 0.0
    print("showing camera %d. Point a TV remote at the lens and press a button for the IR test." % index)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("read failed on camera %d" % index)
            break
        if flip:
            frame = cv2.flip(frame, 1)
        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        if show_info:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cx, cy, r = w // 2, h // 2, max(8, min(w, h) // 20)
            roi = gray[cy - r:cy + r, cx - r:cx + r]
            cbright = float(roi.mean()) if roi.size else 0.0
            cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 0), 1)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 2 * r, 1)
            for i, txt in enumerate([
                "cam %d   %dx%d   %.0f fps" % (index, w, h, fps),
                "center brightness %5.1f / 255   (peak %d)" % (cbright, int(gray.max())),
                "IR test: remote at lens -> bright dot = NoIR",
            ]):
                cv2.putText(frame, txt, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('i'):
            show_info = not show_info
        elif k == ord('f'):
            flip = not flip
        elif k == ord('s'):
            fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "cam%d_%d.png" % (index, int(now)))
            cv2.imwrite(fn, frame)
            print("saved", fn)
        elif ord('0') <= k <= ord('9'):
            new = k - ord('0')
            if new != index:
                nc = _open(cv2, new)
                if nc is not None:
                    cap.release()
                    cap, index = nc, new
                    print("switched to camera %d" % index)
                else:
                    print("camera %d did not open" % new)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--list" in args:
        sys.exit(0 if list_cameras(_need_cv2()) else 1)
    idx = next((int(a) for a in args if a.isdigit()), 0)
    sys.exit(run(idx))
