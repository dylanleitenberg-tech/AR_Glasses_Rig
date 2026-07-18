"""Optics + camera geometry for the simulator (numpy only).

- rot_xyz / look_at: rigid-body helpers.
- PinholeCamera: the world camera and the two eye-corner cameras (project a 3D point
  in the glasses frame to a normalized image pixel).
- DisplayOptics: maps a desired viewing DIRECTION (glasses frame) to the display pixel
  that emits it. A see-through AR optic (birdbath/waveguide) is ~collimated, so a pixel
  corresponds to a fixed angular direction; we add mild barrel distortion + a principal
  point, which is the same for every wearer (it's the device). Per-person variation
  enters through WHERE the eye is (parallax) and angle kappa, handled in autosim.

Glasses frame: origin at the optic center, +x right, +y up, +z forward (world).
"""
import numpy as np


def rot_xyz(rx, ry, rz):
    """Rotation matrix from intrinsic-ish X,Y,Z angles in DEGREES (R = Rz Ry Rx)."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def look_at(pos, target, up_hint=(0.0, 1.0, 0.0)):
    """Camera orientation (columns = right, up, look) pointing pos->target."""
    pos = np.asarray(pos, float); target = np.asarray(target, float)
    look = target - pos
    look = look / np.linalg.norm(look)
    up_hint = np.asarray(up_hint, float)
    right = np.cross(up_hint, look)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([0.0, 0.0, 1.0]), look)
    right = right / np.linalg.norm(right)
    up = np.cross(look, right)
    return np.column_stack([right, up, look])


class PinholeCamera:
    """Real camera: pinhole + radial & tangential distortion + per-unit intrinsic error
    (focal scale, principal-point offset) + finite sensor resolution."""

    def __init__(self, pos, R, fov_deg=70.0, k1=0.0, k2=0.0, res=640):
        self.pos = np.asarray(pos, float)
        self.R = np.asarray(R, float)
        self.f = 0.5 / np.tan(np.radians(fov_deg) / 2.0)   # half-size 0.5 spans half-FOV
        self.k1, self.k2, self.res = k1, k2, res
        # per-unit intrinsic errors (set by Simulator from the device tolerance)
        self.fscale = 1.0          # focal-length error (multiplicative)
        self.cx = self.cy = 0.0    # principal-point offset (normalized)
        self.p1 = self.p2 = 0.0    # tangential distortion

    def project(self, X):
        """3D point (glasses frame) -> normalized pixel [0,1]^2, or None if behind."""
        local = self.R.T @ (np.asarray(X, float) - self.pos)
        if local[2] <= 1e-6:
            return None
        f = self.f * self.fscale
        x = f * local[0] / local[2]
        y = f * local[1] / local[2]
        r2 = x * x + y * y
        d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2          # radial distortion
        xd = x * d + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)   # tangential
        yd = y * d + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        u = xd + 0.5 + self.cx
        v = 0.5 - yd + self.cy
        if self.res:                                        # quantize to the sensor grid
            u = round(u * self.res) / self.res
            v = round(v * self.res) / self.res
        return np.array([u, v])


# Sanity bound on the display's normalized radial-distortion coefficients. The XREAL One Pro's
# X-prism optic is mild (k1~0.06, k2~0.02); anything past this magnitude is unphysical for this
# device and would warp the overlay badly, so it's almost certainly a typo/mis-set constant.
# (NB: there is no display.py in this repo — the display optic lives HERE and in rig.py.)
DISPLAY_K_MAX = 0.25


def _check_display_k(k1, k2):
    """Reject EXTREME display distortion constants (guards against a stray/typo'd K1/K2)."""
    if abs(float(k1)) > DISPLAY_K_MAX or abs(float(k2)) > DISPLAY_K_MAX:
        raise ValueError(
            "DisplayOptics distortion out of range: k1=%.3f k2=%.3f (|k| must be <= %.2f). "
            "The One Pro optic is mild (k1~0.06, k2~0.02); a larger value is almost certainly "
            "a mistake." % (float(k1), float(k2), DISPLAY_K_MAX))


class DisplayOptics:
    """See-through AR optic. Virtual image at `virtual_dist` mm (None = collimated/∞).

    A finite virtual image is the real case (birdbath/waveguide focus a few metres out),
    which makes the registered pixel depend on eye position (vergence), not just angle —
    so calibration must account for it, exactly like real hardware.
    """

    def __init__(self, fov_deg=50.0, k1=0.06, k2=0.0, cx=0.0, cy=0.0,
                 virtual_dist=2000.0):
        _check_display_k(k1, k2)   # no extreme K1/K2 (see DISPLAY_K_MAX)
        self.half = np.radians(fov_deg) / 2.0
        self.k1, self.k2 = k1, k2
        self.cx, self.cy = cx, cy
        self.virtual_dist = virtual_dist
        self.warp = np.zeros(4)    # per-unit spatially-varying residual distortion

    def pixel_for_angles(self, az, el):
        """Viewing angles (radians, az=+right, el=+up) -> display pixel [0,1]^2."""
        x = az / self.half
        y = el / self.half
        r2 = x * x + y * y
        dist = 1.0 + self.k1 * r2 + self.k2 * r2 * r2     # barrel distortion
        u = 0.5 + 0.5 * (x * dist) + self.cx
        v = 0.5 - 0.5 * (y * dist) + self.cy
        w = self.warp                                      # spatially-varying residual
        u += w[0] * x * y + w[1] * (x * x - y * y)
        v += w[2] * x * y + w[3] * (x * x - y * y)
        return np.array([u, v])

    def pixel_for_dir(self, d):
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        return self.pixel_for_angles(np.arctan2(d[0], d[2]), np.arctan2(d[1], d[2]))

    def pixel_for_target(self, eye_pos, direction, optic_pos):
        """Pixel so the eye at `eye_pos` perceives the overlay along `direction`.

        Collimated -> just the angle. Finite virtual image -> place the virtual point on
        the line of sight at the focus sphere (radius virtual_dist about the optic) and
        render the angle of THAT point from the optic, giving real vergence parallax.
        """
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        if self.virtual_dist is None:
            return self.pixel_for_dir(d)
        w = np.asarray(eye_pos, float) - np.asarray(optic_pos, float)
        b = 2.0 * np.dot(d, w)
        c = np.dot(w, w) - self.virtual_dist ** 2
        s = (-b + np.sqrt(max(b * b - 4 * c, 0.0))) / 2.0   # forward intersection
        r = (np.asarray(eye_pos, float) + s * d) - np.asarray(optic_pos, float)
        return self.pixel_for_angles(np.arctan2(r[0], r[2]), np.arctan2(r[1], r[2]))
