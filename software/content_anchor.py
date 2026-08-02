"""content_anchor.py — PLACE generated content on a real surface and keep it there.

`anchor.py` answers "given a 3D world point, where on the display does it land right now?" It
cannot answer the question that comes first for generated content: *which* 3D point? Today the
only anchors the runtime produces come from `people_track` — content can be stuck to a detected
person and nothing else. There is no way to put an image on a wall, a label on a table, or a
panel in mid-air and have it stay.

This module closes that gap with four pieces:

  1. PLACEMENT   `ray_from_world_cam` turns a world-camera pixel into a world-frame ray, and
                 `raycast_mesh` intersects that ray with the live Delaunay surface from
                 `world_mesh.visible_mesh`. Point the rig at a spot, get the 3D point on the real
                 surface plus its normal. Placement is done in the WORLD-CAMERA frame, not the
                 display frame, so it needs no inverse of the learned calibrator — the mesh and
                 the cameras already share a frame.

  2. ORIENTATION `SurfaceAnchor` holds a position, a surface normal and an up-hint, and emits four
                 world-space corners. The content therefore lies ON the surface and keeps the
                 surface's orientation — it is not a camera-facing billboard like `avatar.py`.
                 Walk around a poster anchored to a wall and it foreshortens correctly.

  3. OCCLUSION   `is_occluded` casts from the camera centre toward the anchor and reports whether
                 real geometry sits in front of it, so content behind a wall can be suppressed
                 rather than painted over it.

  4. PERSISTENCE `AnchorStore` round-trips anchors to JSON so a placement survives the session.

Projection stays with `anchor.AnchorProjector` — the corners are ordinary world points, so they
inherit the parallax-correct, calibrated path already proven there.

Pure numpy + stdlib; `--selftest` runs headless with no cameras and no cv2.

    python3 content_anchor.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

from anchor import AnchorProjector, _make_synthetic_device, _rot
from world_mesh import DEFAULT_F, DEFAULT_B
from rig import WORLD_RES

EPS = 1e-9


# --------------------------------------------------------------------------
#  1. Placement — world-camera pixel -> world ray -> surface hit
# --------------------------------------------------------------------------
def ray_from_world_cam(uv, pose, f=DEFAULT_F, res=WORLD_RES):
    """World-camera pixel -> (origin, unit direction) in the WORLD frame.

    `pose` is (R_cw, C) from world_mesh: R_cw maps world->cam, C is the camera centre in world.
    The ray starts at C, so a hit distance is a true metric range from the head.
    """
    R_cw, C = pose
    W = res
    H = int(res * 3 // 4)
    d_cam = np.array([(uv[0] - W / 2.0) / f, (uv[1] - H / 2.0) / f, 1.0])
    d_world = R_cw.T @ d_cam                      # cam->world is the transpose
    n = np.linalg.norm(d_world)
    return np.asarray(C, float).copy(), d_world / (n if n > EPS else 1.0)


def ray_triangle(orig, dirn, v0, v1, v2):
    """Moller-Trumbore. Returns distance along `dirn` to the hit, or None.

    Two-sided on purpose: a surface reconstructed from a point cloud has arbitrary winding, so
    culling by facing would drop half the mesh at random.
    """
    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(dirn, e2)
    det = np.dot(e1, p)
    if abs(det) < EPS:
        return None                                # ray parallel to the triangle
    inv = 1.0 / det
    t = orig - v0
    u = np.dot(t, p) * inv
    if u < -EPS or u > 1.0 + EPS:
        return None
    q = np.cross(t, e1)
    v = np.dot(dirn, q) * inv
    if v < -EPS or u + v > 1.0 + EPS:
        return None
    dist = np.dot(e2, q) * inv
    return float(dist) if dist > EPS else None


def raycast_mesh(orig, dirn, verts, faces):
    """Nearest intersection of a ray with the mesh.

    Returns dict(point, normal, distance, face) or None. The normal is flipped to face back
    along the ray so content placed on it always sits on the side you are looking from.
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    if len(faces) == 0:
        return None
    best = None
    for fi, (a, b, c) in enumerate(faces):
        d = ray_triangle(orig, dirn, verts[a], verts[b], verts[c])
        if d is not None and (best is None or d < best[0]):
            best = (d, fi, a, b, c)
    if best is None:
        return None
    d, fi, a, b, c = best
    n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
    ln = np.linalg.norm(n)
    n = n / ln if ln > EPS else np.array([0.0, 0.0, -1.0])
    if np.dot(n, dirn) > 0:                        # face the viewer
        n = -n
    return {"point": orig + d * dirn, "normal": n, "distance": d, "face": fi}


def fit_plane(points):
    """Least-squares plane through >=3 points -> (centroid, unit normal).

    Used to smooth a local surface patch: a single Delaunay triangle's normal is noisy, so
    content placed on it jitters. Averaging over the neighbourhood is far more stable.
    """
    P = np.asarray(points, float)
    if len(P) < 3:
        raise ValueError("fit_plane needs at least 3 points, got %d" % len(P))
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[2] / (np.linalg.norm(vt[2]) or 1.0)   # smallest singular vector = normal


# --------------------------------------------------------------------------
#  2. Orientation — a quad that lies ON the surface
# --------------------------------------------------------------------------
def surface_basis(normal, up_hint=(0.0, 1.0, 0.0)):
    """Orthonormal (right, up) spanning the plane of `normal`.

    `up_hint` is world up by default, so a poster on a wall hangs level rather than rolled to
    some arbitrary angle. When the surface is nearly horizontal (a table) world-up is parallel
    to the normal and useless as a hint, so we fall back to world -z.
    """
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    h = np.asarray(up_hint, float)
    if abs(np.dot(h, n)) > 0.95:                   # hint parallel to the normal -> degenerate
        h = np.array([0.0, 0.0, -1.0])
    right = np.cross(h, n)
    rn = np.linalg.norm(right)
    if rn < EPS:
        right = np.array([1.0, 0.0, 0.0])
        rn = 1.0
    right = right / rn
    up = np.cross(n, right)
    return right, up / (np.linalg.norm(up) or 1.0)


class SurfaceAnchor:
    """Generated content pinned to a real surface: a world-space quad with the surface's pose.

    Unlike the person billboards in avatar.py this does NOT face the camera — it keeps the
    orientation of the surface it was placed on, so it foreshortens as you move, which is what
    sells it as being part of the scene.
    """

    def __init__(self, anchor_id, position, normal, width, height,
                 up_hint=(0.0, 1.0, 0.0), meta=None):
        self.id = anchor_id
        self.position = np.asarray(position, float)
        self.normal = np.asarray(normal, float) / (np.linalg.norm(normal) or 1.0)
        self.width = float(width)
        self.height = float(height)
        self.up_hint = np.asarray(up_hint, float)
        self.meta = dict(meta or {})

    def corners(self):
        """4 world points, order TL, TR, BR, BL — matching avatar.compose's quad convention."""
        right, up = surface_basis(self.normal, self.up_hint)
        w = right * (self.width / 2.0)
        h = up * (self.height / 2.0)
        p = self.position
        return np.array([p - w + h, p + w + h, p + w - h, p - w - h])

    def project(self, projector, pose):
        """Corners -> display quad. Returns (4,2) normalized display pixels, or None if any
        corner is not displayable (behind the display or off the world cameras)."""
        uvs = [projector.project(c, pose) for c in self.corners()]
        if any(u is None for u in uvs):
            return None
        return np.array(uvs, float)

    def depth(self, pose):
        """Range from the camera centre to the anchor, for painter's-algorithm sorting."""
        _, C = pose
        return float(np.linalg.norm(self.position - np.asarray(C, float)))

    def to_dict(self):
        return {"id": self.id, "position": self.position.tolist(),
                "normal": self.normal.tolist(), "width": self.width,
                "height": self.height, "up_hint": self.up_hint.tolist(), "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        return cls(d["id"], d["position"], d["normal"], d["width"], d["height"],
                   d.get("up_hint", (0.0, 1.0, 0.0)), d.get("meta"))


def place_on_surface(anchor_id, uv, pose, verts, faces, width, height,
                     up_hint=(0.0, 1.0, 0.0), patch_radius=0.0, meta=None):
    """THE placement call: aim a world-camera pixel at a real surface, get an anchor.

    `patch_radius` > 0 fits a plane to the mesh vertices within that radius of the hit instead of
    using the single triangle's normal — steadier orientation on a noisy reconstruction.
    Returns None when the ray misses the mesh (nothing real to stick to).
    """
    orig, dirn = ray_from_world_cam(uv, pose)
    hit = raycast_mesh(orig, dirn, verts, faces)
    if hit is None:
        return None
    normal = hit["normal"]
    if patch_radius > 0:
        near = np.asarray(verts, float)
        sel = near[np.linalg.norm(near - hit["point"], axis=1) <= patch_radius]
        if len(sel) >= 3:
            _, n = fit_plane(sel)
            if np.dot(n, dirn) > 0:                # keep it facing the viewer
                n = -n
            normal = n
    return SurfaceAnchor(anchor_id, hit["point"], normal, width, height, up_hint, meta)


# --------------------------------------------------------------------------
#  3. Occlusion — is real geometry in front of the content?
# --------------------------------------------------------------------------
def is_occluded(point, pose, verts, faces, tol=5.0):
    """True when the mesh blocks the line of sight from the camera centre to `point`.

    `tol` (mm) keeps a surface from occluding content placed ON it — the hit and the anchor are
    the same location, so without slack every surface anchor would occlude itself.
    """
    _, C = pose
    C = np.asarray(C, float)
    v = np.asarray(point, float) - C
    dist = np.linalg.norm(v)
    if dist < EPS:
        return False
    hit = raycast_mesh(C, v / dist, verts, faces)
    return hit is not None and hit["distance"] < dist - tol


# --------------------------------------------------------------------------
#  4. Persistence
# --------------------------------------------------------------------------
class AnchorStore:
    """Anchors that outlive the session. Coordinates are in the world frame the mesh built, so
    a store is only meaningful against that same map — `world_id` records which one."""

    def __init__(self, anchors=None, world_id=""):
        self.anchors = list(anchors or [])
        self.world_id = world_id

    def add(self, anchor):
        self.anchors.append(anchor)
        return anchor

    def remove(self, anchor_id):
        n = len(self.anchors)
        self.anchors = [a for a in self.anchors if a.id != anchor_id]
        return len(self.anchors) != n

    def get(self, anchor_id):
        return next((a for a in self.anchors if a.id == anchor_id), None)

    def visible(self, projector, pose, verts=None, faces=None, occlusion=True):
        """[(anchor, quad, depth)] for anchors that are on-display and not occluded, sorted
        far->near so avatar.compose can paint them straight through."""
        out = []
        for a in self.anchors:
            quad = a.project(projector, pose)
            if quad is None:
                continue
            if occlusion and verts is not None and faces is not None and len(faces):
                if is_occluded(a.position, pose, verts, faces):
                    continue
            out.append((a, quad, a.depth(pose)))
        return sorted(out, key=lambda t: -t[2])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"world_id": self.world_id,
                       "anchors": [a.to_dict() for a in self.anchors]}, f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls([SurfaceAnchor.from_dict(a) for a in d.get("anchors", [])],
                   d.get("world_id", ""))


# ==========================================================================
#  Self-test — headless, no cameras, no cv2
# ==========================================================================
def _wall_mesh(z=1500.0, half=600.0, n=5):
    """A flat wall at depth z, triangulated into a grid — stands in for world_mesh output."""
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    verts, faces = [], []
    for j in range(n):
        for i in range(n):
            verts.append([xs[i], ys[j], z])
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; b = a + 1; c = a + n; d = c + 1
            faces.append([a, b, d]); faces.append([a, d, c])
    return np.array(verts, float), np.array(faces, int)


def selftest(verbose=True):
    if verbose:
        print("== content_anchor self-test (synthetic surface + device, no hardware) ==")
    ok = True
    verts, faces = _wall_mesh()
    pose0 = (np.eye(3), np.zeros(3))

    # (1) ray/triangle + raycast: a centre pixel must hit the wall dead ahead at its depth
    orig, dirn = ray_from_world_cam((WORLD_RES / 2, WORLD_RES * 3 // 4 / 2), pose0)
    hit = raycast_mesh(orig, dirn, verts, faces)
    good = hit is not None and abs(hit["distance"] - 1500.0) < 1e-6 and abs(hit["point"][2] - 1500.0) < 1e-6
    print("  centre ray hits the wall at its true depth : %s  (%.3f mm)"
          % ("PASS" if good else "FAIL", hit["distance"] if hit else float("nan")))
    ok &= good

    # normal must face back along the ray (toward the viewer)
    facing = hit is not None and np.dot(hit["normal"], dirn) < 0
    print("  surface normal faces the viewer            : %s" % ("PASS" if facing else "FAIL"))
    ok &= facing

    # (2) an off-centre pixel hits off-centre, and the hit re-projects to that same pixel
    uv = (WORLD_RES / 2 + 140.0, WORLD_RES * 3 // 4 / 2 - 90.0)
    o2, d2 = ray_from_world_cam(uv, pose0)
    h2 = raycast_mesh(o2, d2, verts, faces)
    back = None
    if h2 is not None:
        p = h2["point"]
        back = (DEFAULT_F * p[0] / p[2] + WORLD_RES / 2,
                DEFAULT_F * p[1] / p[2] + (WORLD_RES * 3 // 4) / 2)
    rt = back is not None and abs(back[0] - uv[0]) < 1e-6 and abs(back[1] - uv[1]) < 1e-6
    print("  pixel -> world -> pixel round-trip         : %s" % ("PASS" if rt else "FAIL"))
    ok &= rt

    # (3) placement + WORLD LOCK: place content, then move the head; it must stay on the same
    #     real spot (compare against the synthetic device's ground-truth projection).
    display_map, gt = _make_synthetic_device()
    proj = AnchorProjector(display_map)
    a = place_on_surface("poster", uv, pose0, verts, faces, width=300.0, height=200.0,
                         patch_radius=400.0)
    placed = a is not None
    print("  place_on_surface returns an anchor         : %s" % ("PASS" if placed else "FAIL"))
    ok &= placed

    if placed:
        poses = [pose0,
                 (_rot([0, 1, 0], 5.0), np.array([20.0, 8.0, 15.0])),
                 (_rot([1, 0, 0], -4.0) @ _rot([0, 1, 0], -7.0), np.array([-30.0, -12.0, 25.0]))]
        worst = 0.0
        for pose in poses:
            for c in a.corners():
                p = proj.project(c, pose)
                g = gt(c, pose)
                if p is None or g is None:
                    continue
                worst = max(worst, float(np.hypot(*(np.array(p) - np.array(g))) * 1080))
        lock = worst < 0.01
        print("  world-locked across head moves             : %s  (worst %.4f px)"
              % ("PASS" if lock else "FAIL", worst))
        ok &= lock

        # (4) the quad lies IN the surface plane and is the requested size
        cs = a.corners()
        planar = max(abs(np.dot(c - a.position, a.normal)) for c in cs)
        w = np.linalg.norm(cs[1] - cs[0]); h = np.linalg.norm(cs[3] - cs[0])
        geom = planar < 1e-6 and abs(w - 300.0) < 1e-6 and abs(h - 200.0) < 1e-6
        print("  quad is planar + correctly sized           : %s  (%.1f x %.1f mm, off-plane %.2e)"
              % ("PASS" if geom else "FAIL", w, h, planar))
        ok &= geom

        # a wall anchor must NOT be camera-facing: rotate the head and the projected quad's
        # aspect must change (a billboard's would not).
        q0 = a.project(proj, poses[0]); q2 = a.project(proj, poses[2])
        if q0 is not None and q2 is not None:
            def aspect(q):
                return np.linalg.norm(q[1] - q[0]) / max(np.linalg.norm(q[3] - q[0]), 1e-9)
            fore = abs(aspect(q0) - aspect(q2)) > 1e-4
            print("  foreshortens with view (not a billboard)   : %s" % ("PASS" if fore else "FAIL"))
            ok &= fore

    # (5) occlusion: a near wall between the eye and the anchor hides it; without it, visible
    near_v, near_f = _wall_mesh(z=800.0, half=600.0, n=3)
    far_pt = np.array([0.0, 0.0, 1500.0])
    occ = is_occluded(far_pt, pose0, near_v, near_f)
    vis = not is_occluded(far_pt, pose0, verts, faces)     # its own surface must not occlude it
    print("  occluded by nearer geometry                : %s" % ("PASS" if occ else "FAIL"))
    print("  not self-occluded by its own surface       : %s" % ("PASS" if vis else "FAIL"))
    ok &= occ and vis

    # (6) persistence round-trip
    store = AnchorStore(world_id="selftest")
    if placed:
        store.add(a)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = store.save(os.path.join(td, "anchors.json"))
        back_store = AnchorStore.load(p)
    same = (len(back_store.anchors) == len(store.anchors) and
            (not placed or (np.allclose(back_store.anchors[0].position, a.position) and
                            np.allclose(back_store.anchors[0].normal, a.normal) and
                            back_store.anchors[0].id == a.id)))
    print("  save/load round-trips exactly              : %s" % ("PASS" if same else "FAIL"))
    ok &= same

    # (7) a ray into empty space must return nothing rather than inventing a placement
    miss = place_on_surface("x", (5.0, 5.0), pose0, verts, faces, 100.0, 100.0)
    print("  miss returns None (no phantom anchor)      : %s" % ("PASS" if miss is None else "FAIL"))
    ok &= miss is None

    print("CONTENT ANCHOR OK" if ok else "CONTENT ANCHOR FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="headless geometry checks")
    args = ap.parse_args()
    if args.selftest or len(sys.argv) == 1:
        sys.exit(0 if selftest() else 1)
