"""Unified nudge/approve input: gamepad (pygame) with a keyboard fallback.

The live loop calls poll(key) once per frame, passing the key code from the overlay
window (cv2.waitKey). It returns how much to nudge the red dot this frame and whether
you approved / quit / asked to recalibrate corners.

Keyboard:  w/a/s/d = fine nudge,  W/A/S/D = coarse,  ENTER = approve,
           U = undo last stored sample,  Z = reset/cancel the current nudge,
           R = recalibrate,  Q/ESC = quit
Gamepad:   left stick = nudge,  A = approve,  B = quit,  X = undo,  Y = reset nudge
"""
from dataclasses import dataclass

try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False


@dataclass
class Action:
    dx: float = 0.0
    dy: float = 0.0
    approve: bool = False
    quit: bool = False
    recalibrate: bool = False
    undo: bool = False      # remove the last stored sample (mis-approve fallback)
    reset: bool = False     # cancel the current nudge before approving (mis-nudge fallback)


# cv2.waitKey codes we rely on (ascii). Enter is 13 on macOS/Windows, 10 on some
# Linux builds, so accept both. Arrow keys vary by platform -> we use WASD instead.
_K_ENTER = (13, 10)
_K_ESC = 27


class InputController:
    def __init__(self, nudge_gain: float = 0.0025, deadzone: float = 0.15,
                 approve_button: int = 0, quit_button: int = 1,
                 use_joystick: bool = True):
        self.nudge_gain = nudge_gain
        self.deadzone = deadzone
        self.approve_button = approve_button
        self.quit_button = quit_button
        self.fine_step = nudge_gain          # lowercase wasd = pixel-level nudges
        self.coarse_step = nudge_gain * 6    # UPPERCASE WASD = big jumps
        self.js = None
        if use_joystick and _HAS_PYGAME:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.js = pygame.joystick.Joystick(0)
                self.js.init()

    @property
    def has_joystick(self) -> bool:
        return self.js is not None

    def _dz(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else v

    def poll(self, key: int = -1) -> Action:
        a = Action()

        # ---- keyboard (from the overlay's cv2.waitKey) ----
        # WASD only — arrow-key codes from cv2.waitKey are platform-specific and
        # collide with letter codes, so we don't map them (docs say WASD).
        if key != -1 and key != 255:
            k = key
            fine, coarse = self.fine_step, self.coarse_step
            if k == ord('w'):
                a.dy -= fine
            elif k == ord('W'):
                a.dy -= coarse
            elif k == ord('s'):
                a.dy += fine
            elif k == ord('S'):
                a.dy += coarse
            elif k == ord('a'):
                a.dx -= fine
            elif k == ord('A'):
                a.dx -= coarse
            elif k == ord('d'):
                a.dx += fine
            elif k == ord('D'):
                a.dx += coarse
            elif k in _K_ENTER:
                a.approve = True
            elif k in (ord('u'), ord('U')):
                a.undo = True
            elif k in (ord('z'), ord('Z')):
                a.reset = True
            elif k in (ord('r'), ord('R')):
                a.recalibrate = True
            elif k in (ord('q'), ord('Q'), _K_ESC):
                a.quit = True

        # ---- gamepad ----
        if self.js is not None:
            pygame.event.pump()
            ax = self._dz(self.js.get_axis(0))
            ay = self._dz(self.js.get_axis(1))
            a.dx += ax * self.nudge_gain
            a.dy += ay * self.nudge_gain
            if self.js.get_button(self.approve_button):
                a.approve = True
            if self.js.get_button(self.quit_button):
                a.quit = True
            nb = self.js.get_numbuttons()
            if nb > 2 and self.js.get_button(2):      # X -> undo last sample
                a.undo = True
            if nb > 3 and self.js.get_button(3):      # Y -> reset/cancel current nudge
                a.reset = True
        return a

    def close(self) -> None:
        if _HAS_PYGAME:
            try:
                pygame.quit()
            except Exception:
                pass
