"""avatar.py — turn a tracked person into a MONKEY drawn on the AR display.

Given a TrackedPerson (people_track) with 3D world anchor points + the world-locked AnchorProjector
(anchor.py), this poses a monkey over them and produces a DrawItem the compositor paints onto the
display. Because every corner of the monkey is a WORLD point re-projected each frame, the monkey
stays glued to the person as both the person and the wearer's head move.

  * POSE / SCALE / ORIENT: the monkey is a camera-facing billboard spanning the person's head→feet
    with a width from their build; it shrinks with distance automatically (the projected head–feet
    span is smaller when they're farther), and the billboard's width axis tracks the view direction
    so it always faces the wearer. (A rigged 3D monkey articulated to body keypoints is the upgrade
    path — feed pose keypoints as extra anchor points; the billboard is the runnable default.)
  * OCCLUSION: DrawItems carry a depth; the compositor paints far→near (painter's algorithm) so a
    nearer monkey correctly covers a farther one.
  * ASSET: `make_monkey_texture()` procedurally builds an RGBA monkey (brown fur, face, ears, eyes)
    so the whole render path runs with no art files. Swap in a real PNG (or a 3D model renderer)
    via `MonkeyAvatar(texture=...)` for "realistic".

Rendering uses cv2 for the perspective warp; the geometry (posing the billboard, depth sort) is
plain numpy. `--selftest` proves the monkey covers the person, scales with distance, sorts by
depth (occlusion), and composites to an otherwise-transparent (black) canvas — no hardware.
"""
import argparse
import sys

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------
#  Asset: a procedural monkey (RGBA). Replace with a PNG / 3D render for realism.
# --------------------------------------------------------------------------
def make_monkey_texture(size=256):
    """RGBA monkey sprite: transparent outside the silhouette, head at the TOP row of the image.
    Warm brown fur with soft radial shading, a tan face + muzzle, and eyes with highlights so it
    reads as a monkey rather than a flat blob. (Still a placeholder for a real art asset / 3D
    render — MonkeyAvatar(texture=<PNG>) swaps it out.)"""
    import cv2
    s = size
    img = np.zeros((s, s, 4), np.uint8)
    fur = (38, 74, 116); fur_dk = (24, 48, 78); belly = (60, 104, 150)   # BGR browns
    face = (120, 162, 205); muzzle = (150, 190, 224); dark = (18, 18, 20); white = (235, 235, 235)

    def blob(cx, cy, ax, ay, color, layer=None):
        tgt = layer if layer is not None else img
        cv2.ellipse(tgt, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360,
                    (color[0], color[1], color[2], 255), -1)

    blob(s * 0.5, s * 0.62, s * 0.27, s * 0.35, fur)      # torso
    blob(s * 0.28, s * 0.58, s * 0.10, s * 0.24, fur)     # arms
    blob(s * 0.72, s * 0.58, s * 0.10, s * 0.24, fur)
    blob(s * 0.5, s * 0.66, s * 0.15, s * 0.22, belly)    # lighter belly
    blob(s * 0.29, s * 0.24, s * 0.07, s * 0.08, fur)     # ears
    blob(s * 0.71, s * 0.24, s * 0.07, s * 0.08, fur)
    blob(s * 0.29, s * 0.24, s * 0.035, s * 0.045, muzzle)
    blob(s * 0.71, s * 0.24, s * 0.035, s * 0.045, muzzle)
    blob(s * 0.5, s * 0.27, s * 0.21, s * 0.21, fur)      # head
    blob(s * 0.5, s * 0.31, s * 0.145, s * 0.155, face)   # face
    blob(s * 0.5, s * 0.37, s * 0.075, s * 0.055, muzzle) # muzzle
    blob(s * 0.47, s * 0.385, s * 0.012, s * 0.012, dark) # nostrils
    blob(s * 0.53, s * 0.385, s * 0.012, s * 0.012, dark)
    for ex in (0.435, 0.565):                              # eyes + highlight
        blob(s * ex, s * 0.28, s * 0.032, s * 0.038, white)
        blob(s * ex, s * 0.285, s * 0.020, s * 0.024, dark)
        blob(s * (ex + 0.006), s * 0.278, s * 0.006, s * 0.006, white)

    # soft radial shading: darken toward the silhouette edge for a rounded look
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    r = np.sqrt(((xx - s / 2) / (s / 2)) ** 2 + ((yy - s * 0.55) / (s * 0.55)) ** 2)
    shade = np.clip(1.15 - 0.5 * r, 0.55, 1.0)[:, :, None]
    rgb = np.clip(img[:, :, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)
    img[:, :, :3] = rgb
    return img


class RenderSmoother:
    """Per-person EMA on the projected billboard quad to kill on-screen jitter (detection noise
    survives the world-space tracker as small quad wobble). Keyed by track id so identities don't
    bleed; `alpha` in (0,1] — lower = smoother but laggier."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = {}

    def smooth(self, tid, quad):
        q = np.asarray(quad, float)
        if tid in self.state:
            self.state[tid] = self.alpha * q + (1 - self.alpha) * self.state[tid]
        else:
            self.state[tid] = q
        return self.state[tid].copy()

    def prune(self, keep_ids):
        for k in list(self.state):
            if k not in keep_ids:
                del self.state[k]


class DrawItem:
    """One thing to composite: a textured quad in NORMALIZED display coords, with a depth for sort."""
    def __init__(self, quad_norm, texture, depth, tid=-1):
        self.quad = np.asarray(quad_norm, float)   # 4x2 [TL,TR,BR,BL], normalized (0..1)
        self.texture = texture                     # HxWx4 RGBA
        self.depth = float(depth)                  # mm from camera (larger = farther)
        self.id = tid


class MonkeyAvatar:
    def __init__(self, texture=None, width_frac=0.55):
        self.texture = texture if texture is not None else make_monkey_texture()
        self.width_frac = width_frac               # monkey width as a fraction of person height

    def render(self, person, projector, pose):
        """TrackedPerson -> DrawItem (or None). The billboard's VERTICAL axis is the person's
        head->feet (both world points, re-projected each frame => world-locked), and its WIDTH is
        the projected shoulder span (world-correct, camera-facing) with a screen-space fallback if a
        shoulder clips the world-camera FOV. Robust when the person's shoulders are cropped but the
        head/feet axis is visible (e.g. a close, tall person)."""
        R_cw, C = pose
        head_uv = projector.project(person.points["head"], pose)
        feet_uv = projector.project(person.points["feet"], pose)
        if head_uv is None or feet_uv is None:
            return None                            # can't see the body axis -> can't place it
        head_uv = np.asarray(head_uv, float); feet_uv = np.asarray(feet_uv, float)
        l = projector.project(person.points["l_shoulder"], pose)
        r = projector.project(person.points["r_shoulder"], pose)
        if l is not None and r is not None:
            hw = (np.asarray(r, float) - np.asarray(l, float)) / 2.0     # world-correct half width
            hw *= self.width_frac / 0.42                                 # shoulders=0.42h -> monkey width
        else:                                                            # fallback: perp to the axis
            axis = feet_uv - head_uv
            perp = np.array([-axis[1], axis[0]]); n = np.linalg.norm(perp)
            perp = perp / n if n > 1e-9 else np.array([1.0, 0.0])
            hw = perp * (np.linalg.norm(axis) * self.width_frac * 0.5)
        quad = np.array([head_uv - hw, head_uv + hw, feet_uv + hw, feet_uv - hw])
        depth = float(np.linalg.norm(person.points["centroid"] - np.asarray(C, float)))
        return DrawItem(quad, self.texture, depth, person.id)

    def render_all(self, people, projector, pose):
        items = [self.render(p, projector, pose) for p in people]
        return [it for it in items if it is not None]


# --------------------------------------------------------------------------
#  Compositor: paint DrawItems onto a transparent (black) display canvas.
# --------------------------------------------------------------------------
def compose(items, display_w=1920, display_h=1080):
    """Painter's algorithm (far->near) alpha-composite. Returns a BGR canvas; black = transparent on
    the see-through AR display, so only the monkeys show and nearer ones occlude farther ones."""
    import cv2
    canvas = np.zeros((display_h, display_w, 3), np.uint8)
    for it in sorted(items, key=lambda d: -d.depth):           # far first
        dst = it.quad * np.array([display_w, display_h])       # normalized -> pixels
        th, tw = it.texture.shape[:2]
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        warped = cv2.warpPerspective(it.texture, M, (display_w, display_h),
                                     flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        rgb = warped[:, :, :3]; a = (warped[:, :, 3:4].astype(np.float32) / 255.0)
        canvas[:] = (rgb.astype(np.float32) * a + canvas.astype(np.float32) * (1 - a)).astype(np.uint8)
    return canvas


# ==========================================================================
#  Self-test (no hardware): monkeys cover people, scale with distance, occlude
#  by depth, and composite onto an otherwise-transparent canvas.
# ==========================================================================
def selftest(verbose=True):
    if verbose:
        print("== avatar self-test (synthetic people + monkey render, no hardware) ==")
    from anchor import AnchorProjector, _make_synthetic_device
    from people_track import TrackedPerson

    display_map, gt = _make_synthetic_device()
    proj = AnchorProjector(display_map)
    avatar = MonkeyAvatar()
    pose = (np.eye(3), np.zeros(3))                 # head at origin looking +z
    DW, DH = 1920, 1080
    checks = []

    # (0) texture is a valid RGBA sprite with a real silhouette, brown fur + a lighter face
    tex = avatar.texture
    op = tex[:, :, 3] > 0
    fur_px = tex[op][:, :3].mean(0)                          # BGR; brown => R>G>B
    face_region = tex[int(0.22 * 256):int(0.40 * 256), int(0.36 * 256):int(0.64 * 256), :3]
    brown = fur_px[2] > fur_px[1] > fur_px[0]
    face_lighter = face_region.mean() > tex[op][:, :3].mean()
    checks.append(("texture is RGBA, brown fur (BGR %s), face lighter than body"
                   % np.round(fur_px).astype(int),
                   tex.shape[2] == 4 and 0.1 < op.mean() < 0.9 and brown and face_lighter))

    # (0b) RenderSmoother cuts on-screen jitter from noisy per-frame quads
    rng0 = np.random.default_rng(0)
    base_quad = np.array([[0.4, 0.3], [0.6, 0.3], [0.6, 0.7], [0.4, 0.7]])
    sm = RenderSmoother(alpha=0.35)
    raw_c, sm_c = [], []
    for _ in range(60):
        noisy = base_quad + rng0.normal(0, 0.01, base_quad.shape)
        raw_c.append(noisy.mean(0)); sm_c.append(sm.smooth(7, noisy).mean(0))
    raw_j = np.linalg.norm(np.diff(raw_c, axis=0), axis=1).mean()
    sm_j = np.linalg.norm(np.diff(sm_c, axis=0), axis=1).mean()
    checks.append(("RenderSmoother reduces frame-to-frame jitter (%.4f -> %.4f)" % (raw_j, sm_j),
                   sm_j < raw_j * 0.6))

    # two people ~1.7 m tall (2.5 m and 4 m away — full body inside the world-cam FOV)
    near = TrackedPerson(0, np.array([0.0, 0.0, 2500.0]), 1700.0)
    far = TrackedPerson(1, np.array([250.0, 0.0, 4000.0]), 1700.0)

    # (1) the monkey billboard covers the person's projected head->feet span
    item = avatar.render(near, proj, pose)
    head_uv = proj.project(near.points["head"], pose)
    feet_uv = proj.project(near.points["feet"], pose)
    ys = item.quad[:, 1]
    covers = ys.min() <= head_uv[1] + 1e-6 and ys.max() >= feet_uv[1] - 1e-6
    checks.append(("monkey billboard spans the person head->feet on the display", covers))

    # (2) SCALE with distance: the near monkey covers more display area than the far one
    def area(it):
        q = it.quad * np.array([DW, DH])
        return 0.5 * abs((q[0, 0] - q[2, 0]) * (q[1, 1] - q[3, 1]) - (q[1, 0] - q[3, 0]) * (q[0, 1] - q[2, 1]))
    near_item = avatar.render(near, proj, pose); far_item = avatar.render(far, proj, pose)
    checks.append(("nearer person -> bigger monkey (%.0f vs %.0f px^2)"
                   % (area(near_item), area(far_item)), area(near_item) > 2 * area(far_item)))

    # (3) DEPTH sort for occlusion: far item is painted before near item
    order = sorted([near_item, far_item], key=lambda d: -d.depth)
    checks.append(("painter order is far->near (occlusion correct)",
                   order[0].id == far.id and order[1].id == near.id))

    # (4) COMPOSITE: canvas has monkey pixels where the near person is, transparent (black) elsewhere
    canvas = compose([near_item, far_item], DW, DH)
    cxpx = int(proj.project(near.points["centroid"], pose)[0] * DW)
    cypx = int(proj.project(near.points["centroid"], pose)[1] * DH)
    lit_at_person = canvas[cypx, cxpx].sum() > 0
    lit_frac = (canvas.sum(axis=2) > 0).mean()
    checks.append(("composited: lit on the person, mostly transparent (%.1f%% lit)"
                   % (100 * lit_frac), lit_at_person and lit_frac < 0.5))

    # (5) WORLD-LOCK end to end: move the head; the monkey re-projects to stay on the person
    yaw = np.radians(10)
    pose2 = (np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]),
             np.array([40.0, 0, 0]))
    item2 = avatar.render(near, proj, pose2)
    person_uv2 = proj.project(near.points["centroid"], pose2)
    cover2 = (item2.quad[:, 0].min() <= person_uv2[0] <= item2.quad[:, 0].max()
              and item2.quad[:, 1].min() <= person_uv2[1] <= item2.quad[:, 1].max())
    moved = np.linalg.norm(item2.quad.mean(0) - item.quad.mean(0)) > 1e-3
    checks.append(("after a head turn the monkey re-projects onto the person (locked, moved on screen)",
                   cover2 and moved))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "AVATAR OK — monkey posed/scaled/oriented, depth-occluded, world-locked ✅"
              if ok else "PROBLEM ⚠️")
        print("  the procedural sprite is a placeholder; MonkeyAvatar(texture=<PNG>) or a rigged 3D")
        print("  model + body keypoints gives 'realistic'. Registration/tracking are asset-agnostic.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="monkey avatar render for tracked people")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save-texture", metavar="PNG", help="write the procedural monkey sprite")
    args = ap.parse_args()
    if args.save_texture:
        import cv2
        cv2.imwrite(args.save_texture, make_monkey_texture())
        print("wrote", args.save_texture); sys.exit(0)
    sys.exit(selftest())
