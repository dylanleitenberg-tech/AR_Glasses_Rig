"""The AR overlay surface: a black canvas with a red dot at a chosen pixel.

On see-through optics (Path A tethered glasses, or the Path B combiner) black is
transparent, so only the lit pixels appear floating in the world. Put this window
on your AR display (use --fullscreen once it's on the right monitor).

In simulation it also draws a green "truth" target so you can practice the
nudge-to-align loop on your laptop screen with no hardware.
"""
from typing import Optional, Tuple
import cv2
import numpy as np


class Overlay:
    def __init__(self, width: int, height: int, window: str = "AR-overlay",
                 fullscreen: bool = False, dot_radius: int = 8, win_x=None):
        self.w = width
        self.h = height
        self.window = window
        self.dot_radius = dot_radius
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        # MOVE TO THE AR DISPLAY BEFORE GOING FULLSCREEN.
        #
        # setWindowProperty(FULLSCREEN) fills whatever monitor the window currently occupies, and
        # a fresh window opens on the PRIMARY one -- so without this the overlay went fullscreen on
        # the laptop screen and nothing reached the glasses. The canvas is black, which a birdbath
        # display renders as transparent, so see-through was never the problem: the overlay simply
        # was not on the AR display.
        #
        # win_x is the desktop x-coordinate of the AR monitor (its left edge in the arrangement).
        if win_x is not None:
            cv2.moveWindow(window, int(win_x), 0)
        if fullscreen:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window, width, height)
        # mouse: click or click-drag places the dot directly (fast, simultaneous X+Y, which
        # the one-key-at-a-time keyboard can't do). _mouse holds the latest normalized pos;
        # _mouse_dirty is set only on a button press/drag so a bare cursor move doesn't hijack it.
        self._mouse = None
        self._mouse_dirty = False
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        # normalize by the displayed image size so it works windowed AND fullscreen-scaled
        rw, rh = self.w, self.h
        try:
            r = cv2.getWindowImageRect(self.window)
            if r[2] > 0 and r[3] > 0:
                rw, rh = r[2], r[3]
        except Exception:
            pass
        self._mouse = (min(max(x, 0), rw) / rw, min(max(y, 0), rh) / rh)
        if event == cv2.EVENT_LBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            self._mouse_dirty = True

    def take_mouse(self):
        """Return the latest click/drag position (normalized) once, or None if no new click."""
        if self._mouse_dirty:
            self._mouse_dirty = False
            return self._mouse
        return None

    def render(self, red_norm: Tuple[float, float],
               green_norm: Optional[Tuple[float, float]] = None,
               hud: Optional[str] = None) -> int:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        if green_norm is not None:                     # sim-only truth target
            gx, gy = int(green_norm[0] * self.w), int(green_norm[1] * self.h)
            cv2.drawMarker(canvas, (gx, gy), (0, 220, 0),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.circle(canvas, (gx, gy), self.dot_radius + 4, (0, 120, 0), 1)
        rx, ry = int(red_norm[0] * self.w), int(red_norm[1] * self.h)
        cv2.circle(canvas, (rx, ry), self.dot_radius, (0, 0, 255), -1)
        if hud:
            for i, line in enumerate(hud.split("\n")):
                cv2.putText(canvas, line, (16, 28 + 26 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1,
                            cv2.LINE_AA)
        cv2.imshow(self.window, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(self.window)
