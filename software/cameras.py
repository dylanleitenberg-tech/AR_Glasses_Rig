"""Thin wrapper over OpenCV VideoCapture for the world + eye-corner cameras.

The eye-corner cameras want to be *global-shutter* and run fast so the "approve"
snapshot freezes the canthus without motion blur (see BOM). This wrapper exposes a
plain read() for the live loop and a snapshot() that flushes the buffer and grabs
the freshest frame for the high-speed capture at approval time.
"""
import cv2


class Camera:
    def __init__(self, index: int, width: int = 640, height: int = 480,
                 fps: int = 120, name: str = "", mjpg: bool = True):
        self.index = index
        self.name = name or "cam%d" % index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera %s (index %d)"
                               % (self.name, index))
        # MJPG lets the global-shutter UVC cams hit full resolution/fps over USB bandwidth.
        if mjpg:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Small buffer so snapshot() gets a *current* frame, not a stale one.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # VERIFY THE MODE ACTUALLY TOOK — a rejected mode is SILENT.
        #
        # UVC does not error on an unsupported size; it hands back the native one. Asking an
        # AR0234 for 640x400 returns 1920x1200, which is 7.5x the pixels, and four of those on one
        # USB 2.0 bus starves half the bank. The visible symptom is "only 2 cameras are running" --
        # nothing anywhere says the resolution request was ignored. This turns that into a warning
        # at the one place that knows both what was asked and what arrived.
        self.requested = (int(width), int(height))
        self.actual = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.actual != self.requested and self.actual != (0, 0):
            print("!! %s: asked %dx%d, got %dx%d — the mode was IGNORED. That is %.1fx the "
                  "pixels and will starve the shared USB bus. See cameras.ROLE_MODE."
                  % (self.name, self.requested[0], self.requested[1],
                     self.actual[0], self.actual[1],
                     (self.actual[0] * self.actual[1]) /
                     max(1.0, self.requested[0] * self.requested[1])))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def snapshot(self):
        """Flush any buffered frames and return the freshest one (low latency)."""
        for _ in range(2):
            self.cap.grab()
        ok, frame = self.cap.retrieve()
        if not ok:
            return self.read()
        return frame

    def release(self):
        self.cap.release()


def list_cameras(max_index: int = 8):
    """Probe indices and report which ones open. Handy for picking the role->index map."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            found.append((i, ok))
            cap.release()
    return found


# Camera ROLES on the rigid carrier, matching rig.py + Config.feature_names order.
# CANONICAL = 6-camera BINOCULAR CORE: 2 world (outward) + 2 eye-corner + 2 NIR pupil
# (pupilL/pupilR, one per eye). The eye2L/eye2R pair is the SEPARATE 8-cam "FULL" future
# upgrade (stereo eye-corner), kept here so a use_stereo build can map it.
# Resolutions mirror rig.py's sensor models (world AR0234 @ 1280, OV9281 eye/pupil @ 640+).
CORE_ROLES  = ("worldL", "worldR", "eyeL", "eyeR", "pupilL", "pupilR")  # 6-cam binocular CORE
STEREO_ROLES = ("eye2L", "eye2R")                                        # +2 = 8-cam FULL upgrade
ROLES = CORE_ROLES + STEREO_ROLES
ROLE_RES = {"worldL": 1280, "worldR": 1280, "eyeL": 640, "eyeR": 640,
            "pupilL": 640, "pupilR": 640, "eye2L": 640, "eye2R": 640}


# EXPLICIT PER-SENSOR CAPTURE MODES. Do NOT compute these from an aspect ratio.
#
# AN UNSUPPORTED MODE DOES NOT FAIL -- IT SILENTLY RETURNS NATIVE, which then saturates the shared
# USB 2.0 bus and starves the other cameras. Measured 2026-08-03 on this rig:
#
#   OV9281 eye  (1280x800 native): 640x400 -> 640x400 TRUE DOWNSCALE, full FOV
#                                  640x480 -> a CROP at scale 1.00 (~30% of the sensor area)
#   AR0234 world (1920x1200 native): 640x480 -> 640x480 TRUE DOWNSCALE, full FOV
#                                    640x400 -> IGNORED, returns 1920x1200
#                                    1280x800 -> IGNORED, returns 1920x1200
#
# So the two sensors need DIFFERENT modes and neither generalises from the other. Deriving the
# height from a single aspect rule broke this twice in one session: 3/4 gave the eye cams a crop,
# and 10/16 made the world cams silently run native and starve half the bank (only 2 of 4 cameras
# delivered frames).
ROLE_MODE = {
    "worldL": (640, 480), "worldR": (640, 480),          # AR0234 — 640x400 is ignored
    "eyeL":   (640, 400), "eyeR":   (640, 400),          # OV9281 — 640x480 is a crop
    "pupilL": (640, 400), "pupilR": (640, 400),
    "eye2L":  (640, 400), "eye2R":  (640, 400),
}


class CameraBank:
    """Opens the role-mapped camera suite. `role_index` maps a subset of ROLES to UVC device
    indices (use --list-cams to find them). Only the roles you pass are opened: the 8-feature
    WORLD+EYE-CORNER base needs 4 (worldL/R, eyeL/R); the 6-cam binocular CORE adds the 2 NIR
    pupil cams (pupilL/R); the 8-cam FULL upgrade adds the 2 stereo cams (eye2L/R).
    """
    def __init__(self, role_index: dict, fps: int = 100):
        bad = [r for r in role_index if r not in ROLE_RES]
        if bad:
            raise ValueError("unknown camera role(s): %s (valid: %s)" % (bad, ROLES))
        self.cams = {}
        for role, idx in role_index.items():
            w, h = ROLE_MODE[role]
            self.cams[role] = Camera(idx, w, h, fps=fps, name=role)

    def read(self) -> dict:
        """Live read: {role: frame-or-None}."""
        return {role: cam.read() for role, cam in self.cams.items()}

    def snapshot(self) -> dict:
        """Freshest synchronized grab (for the approval capture): {role: frame-or-None}."""
        for cam in self.cams.values():       # grab all first (closest to synchronized)
            cam.cap.grab()
        return {role: (cam.cap.retrieve()[1] if cam.cap.retrieve()[0] else cam.read())
                for role, cam in self.cams.items()}

    def release(self):
        for cam in self.cams.values():
            cam.release()
