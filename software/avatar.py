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
    """RGBA monkey sprite: transparent outside the silhouette, head at the TOP row of the image."""
    import cv2
    s = size
    img = np.zeros((s, s, 4), np.uint8)
    fur = (60, 42, 30); face = (120, 150, 190); dark = (20, 20, 20)   # BGR
    def blob(cx, cy, ax, ay, color):
        cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360,
                    (color[0], color[1], color[2], 255), -1)
    blob(s * 0.5, s * 0.62, s * 0.26, s * 0.34, fur)      # torso
    blob(s * 0.30, s * 0.60, s * 0.09, s * 0.22, fur)     # arms
    blob(s * 0.70, s * 0.60, s * 0.09, s * 0.22, fur)
    blob(s * 0.5, s * 0.26, s * 0.20, s * 0.20, fur)      # head
    blob(s * 0.30, s * 0.24, s * 0.06, s * 0.07, fur)     # ears
    blob(s * 0.70, s * 0.24, s * 0.06, s * 0.07, fur)
    blob(s * 0.5, s * 0.30, s * 0.13, s * 0.14, face)     # face
    blob(s * 0.44, s * 0.26, s * 0.025, s * 0.03, dark)   # eyes
    blob(s * 0.56, s * 0.26, s * 0.025, s * 0.03, dark)
    blob(s * 0.5, s * 0.34, s * 0.04, s * 0.03, (90, 110, 150))  # muzzle
    return img


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

    # (0) texture is a valid RGBA sprite with a real silhouette
    tex = avatar.texture
    checks.append(("monkey texture is RGBA with a silhouette (%.0f%% opaque)"
                   % (100 * (tex[:, :, 3] > 0).mean()),
                   tex.shape[2] == 4 and 0.1 < (tex[:, :, 3] > 0).mean() < 0.9))

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
