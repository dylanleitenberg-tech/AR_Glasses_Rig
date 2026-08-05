"""world_memory.py, a few seconds of remembered world geometry, densely meshed.

`world_mesh.py` reconstructs only what the cameras can see RIGHT NOW: a Delaunay surface over
the sparse ORB features visible in the current frame. Anything that leaves view stops existing,
so content anchored to it pops out the instant you glance away or someone walks past.

This module keeps a SHORT ROLLING MEMORY of the world instead:

 * DENSE, not sparse, `stereo_depth` runs cv2's StereoSGBM over the world pair for a per-pixel
    depth map, and frames are fused into a TSDF volume (Open3D, MIT). Truncated-signed-distance
    fusion is the standard best-in-class surface reconstruction: it averages many noisy depth
    maps into one smooth implicit surface and meshes it with marching cubes. Compared with
    Delaunay-over-features it fills gaps, suppresses per-frame noise, and yields real normals.

 * IT FORGETS, the volume is rebuilt from a ring buffer holding only the last `horizon_s`
    seconds of depth frames. That is deliberate, and it is the honest horizon for THIS rig:
    tracking here is dead-reckoned stereo VO with no loop closure and no relocalization, so
    accumulated pose drift makes anything older than a few seconds actively wrong. A short
    memory is also what removes stale geometry, a person who walks away ages out instead of
    being fused permanently into the wall behind them.

What the memory buys: surfaces persist through brief occlusion, feature dropout and head turns,
so a `content_anchor.SurfaceAnchor` stays put instead of flickering.

Open3D is an OPTIONAL fast path. Without it the module still runs, falling back to the sparse
numpy raycast in `content_anchor`, so the release gate stays green on a machine that lacks it.

    python3 world_memory.py --selftest
"""
import argparse
import sys
import time

import numpy as np

from world_mesh import DEFAULT_F, DEFAULT_B

try:
    import open3d as o3d
    HAVE_OPEN3D = True
except Exception:                                   # noqa: BLE001 - any import problem = fallback
    o3d = None
    HAVE_OPEN3D = False


# --------------------------------------------------------------------------
#  Dense stereo depth (cv2 StereoSGBM)
# --------------------------------------------------------------------------
def stereo_depth(frameL, frameR, f=DEFAULT_F, B=DEFAULT_B, num_disp=96, block=7):
    """Rectified world stereo pair -> per-pixel depth in MILLIMETRES (0 = no measurement).

    Semi-global block matching: a good default that needs no training and no GPU. Depth is
    f*B/disparity, the same relation `world_mesh.triangulate_stereo` uses on sparse features , 
    this just does it everywhere instead of at ORB corners.
    """
    import cv2
    gl = frameL if frameL.ndim == 2 else cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    gr = frameR if frameR.ndim == 2 else cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    num_disp = int(np.ceil(num_disp / 16.0) * 16)               # SGBM requires a multiple of 16
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0       # SGBM returns fixed-point x16
    depth = np.zeros_like(disp)
    good = disp > 0.5                                            # tiny disparity => unbounded depth
    depth[good] = f * B / disp[good]
    return depth


# --------------------------------------------------------------------------
#  The rolling-window fused world
# --------------------------------------------------------------------------
class WorldMemory:
    """Fuses the last `horizon_s` seconds of depth frames into one mesh you can raycast.

    Open3D's TSDF has no "un-integrate", so the window is implemented as a ring buffer of
    keyframes that is re-fused whenever it changes. Re-fusing a few dozen frames is cheap and
    keeps the forgetting exact, rather than approximating it with a decay factor.
    """

    def __init__(self, horizon_s=3.0, max_frames=24, voxel_mm=20.0, trunc_mm=60.0,
                 f=DEFAULT_F, B=DEFAULT_B, max_depth_mm=4000.0, min_interval_s=0.0,
                 edge_mm=150.0, fallback_step=8, fallback_frames=4):
        self.horizon_s = float(horizon_s)
        self.max_frames = int(max_frames)
        self.voxel_mm = float(voxel_mm)
        self.trunc_mm = float(trunc_mm)
        self.f = float(f)
        self.B = float(B)
        self.max_depth_mm = float(max_depth_mm)
        self.min_interval_s = float(min_interval_s)
        self.edge_mm = float(edge_mm)            # depth step that breaks a grid quad
        # Fallback-only: the numpy raycast is O(faces) per ray, so the unfused path is meshed
        # coarsely and over fewer frames to stay tractable. Open3D's BVH has no such limit.
        self.fallback_step = int(fallback_step)
        self.fallback_frames = int(fallback_frames)
        self.frames = []                 # [(t, depth HxW float32 mm, T_wc 4x4)]
        self._dirty = True
        self._mesh = None                # (verts Nx3 mm, faces Mx3)
        self._scene = None               # Open3D BVH over the current mesh
        self._last_t = -np.inf

    # ---- ingest -----------------------------------------------------------
    @staticmethod
    def pose_to_T_wc(R_cw, C):
        """world_mesh pose (R_cw world->cam, C centre in world) -> 4x4 cam->world."""
        T = np.eye(4)
        T[:3, :3] = np.asarray(R_cw, float).T                  # cam->world
        T[:3, 3] = np.asarray(C, float)
        return T

    def integrate(self, depth_mm, pose, t=None):
        """Add one depth frame observed at `pose` = (R_cw, C). Returns True if it was kept."""
        t = time.monotonic() if t is None else float(t)
        if t - self._last_t < self.min_interval_s:
            return False                                       # rate-limit keyframes
        self._last_t = t
        R_cw, C = pose
        self.frames.append((t, np.asarray(depth_mm, np.float32), self.pose_to_T_wc(R_cw, C)))
        self.prune(t)
        self._dirty = True
        return True

    def prune(self, now=None):
        """Drop frames outside the memory horizon (and beyond the frame cap). -> n dropped."""
        now = self.frames[-1][0] if now is None and self.frames else (now or 0.0)
        before = len(self.frames)
        self.frames = [fr for fr in self.frames if now - fr[0] <= self.horizon_s]
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]
        if len(self.frames) != before:
            self._dirty = True
        return before - len(self.frames)

    def age_span(self):
        """Seconds between the oldest and newest remembered frame."""
        return 0.0 if len(self.frames) < 2 else self.frames[-1][0] - self.frames[0][0]

    # ---- fuse -------------------------------------------------------------
    def _grid_mesh(self, depth, T_wc, step):
        """Depth map -> world-space triangles by connecting neighbouring pixels.

        A depth image is already a structured grid, so it meshes directly: each 2x2 pixel block
        with valid, CONTIGUOUS depths becomes two triangles. The contiguity test is what stops
        a depth step (an object edge) from being webbed over with false surface.
        """
        h, w = depth.shape
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        d = depth[np.ix_(ys, xs)]
        gy, gx = np.meshgrid(ys.astype(float), xs.astype(float), indexing="ij")
        X = (gx - w / 2.0) * d / self.f
        Y = (gy - h / 2.0) * d / self.f
        cam = np.stack([X, Y, d], -1).reshape(-1, 3)
        world = cam @ T_wc[:3, :3].T + T_wc[:3, 3]
        H, W = d.shape
        idx = np.arange(H * W).reshape(H, W)
        a, b = idx[:-1, :-1], idx[:-1, 1:]
        c, e = idx[1:, :-1], idx[1:, 1:]
        dv = d > 0
        quad = dv[:-1, :-1] & dv[:-1, 1:] & dv[1:, :-1] & dv[1:, 1:]
        dm = np.maximum.reduce([d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]])
        dn = np.minimum.reduce([d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]])
        quad &= (dm - dn) < self.edge_mm                       # no webbing across depth steps
        if not quad.any():
            return world, np.zeros((0, 3), int)
        f1 = np.stack([a[quad], b[quad], e[quad]], 1)
        f2 = np.stack([a[quad], e[quad], c[quad]], 1)
        return world, np.vstack([f1, f2])

    def _rebuild(self):
        self._mesh = None
        self._scene = None
        self._dirty = False
        if not self.frames:
            return
        if not HAVE_OPEN3D:
            # FALLBACK (no Open3D, no TSDF): mesh each remembered frame's depth grid in world
            # space and keep their UNION. Unfused: no averaging or smoothing, so it is noisier
            # than the TSDF path, but the memory semantics are identical: a region dropped from
            # the newest frame is still carried by an older one, and frames age out.
            V, F = [], []
            base = 0
            for _t, depth, T_wc in self.frames[-self.fallback_frames:]:
                v, f = self._grid_mesh(depth, T_wc, self.fallback_step)
                if len(f):
                    V.append(v); F.append(f + base); base += len(v)
            if V:
                self._mesh = (np.vstack(V), np.vstack(F))
            return
        h, w = self.frames[0][1].shape
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, self.f, self.f, w / 2.0, h / 2.0)
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_mm, sdf_trunc=self.trunc_mm,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        blank = o3d.geometry.Image(np.zeros((h, w, 3), np.uint8))
        for _t, depth, T_wc in self.frames:
            d = np.ascontiguousarray(depth, np.float32)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                blank, o3d.geometry.Image(d), depth_scale=1.0,
                depth_trunc=self.max_depth_mm, convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, np.linalg.inv(T_wc))     # Open3D wants world->cam
        m = vol.extract_triangle_mesh()
        m.compute_vertex_normals()
        verts = np.asarray(m.vertices, float)
        faces = np.asarray(m.triangles, int)
        if len(faces) == 0:
            return
        self._mesh = (verts, faces)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    def mesh(self):
        """(verts Nx3 mm, faces Mx3) fused over the memory window, or (empty, empty)."""
        if self._dirty:
            self._rebuild()
        if self._mesh is None:
            return np.zeros((0, 3)), np.zeros((0, 3), int)
        return self._mesh

    # ---- query ------------------------------------------------------------
    def raycast(self, origins, directions):
        """Cast rays against the remembered surface.

        Returns (hits Nx3 world points, dists N, valid N bool). Uses Open3D's BVH when present
        (vectorized over all rays); otherwise falls back to the numpy triangle loop in
        content_anchor, which is exact but O(faces) per ray.
        """
        origins = np.atleast_2d(np.asarray(origins, float))
        directions = np.atleast_2d(np.asarray(directions, float))
        directions = directions / np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        n = len(origins)
        if self._dirty:
            self._rebuild()
        if self._scene is not None:
            rays = np.hstack([origins, directions]).astype(np.float32)
            ans = self._scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
            t = ans["t_hit"].numpy().astype(float)
            valid = np.isfinite(t)
            pts = origins + directions * np.where(valid, t, 0.0)[:, None]
            return pts, t, valid
        # ---- fallback: sparse numpy raycast
        from content_anchor import raycast_mesh
        verts, faces = self.mesh()
        pts = np.zeros((n, 3)); dists = np.full(n, np.inf); valid = np.zeros(n, bool)
        for i in range(n):
            h = raycast_mesh(origins[i], directions[i], verts, faces)
            if h is not None:
                pts[i] = h["point"]; dists[i] = h["distance"]; valid[i] = True
        return pts, dists, valid

    def stats(self):
        verts, faces = self.mesh()
        return {"frames": len(self.frames), "age_span_s": round(self.age_span(), 3),
                "verts": len(verts), "faces": len(faces), "open3d": HAVE_OPEN3D}


# ==========================================================================
#  Self-test: headless, synthetic depth, no cameras
# ==========================================================================
def _synth_depth(h, w, z_mm, hole=None):
    """A fronto-parallel wall at z_mm, optionally with a rectangular dropout (occluder/no-data)."""
    d = np.full((h, w), float(z_mm), np.float32)
    if hole:
        y0, y1, x0, x1 = hole
        d[y0:y1, x0:x1] = 0.0
        return d
    return d


def selftest(verbose=True):
    if verbose:
        print("== world_memory self-test (synthetic depth, no cameras) ==")
        print("   backend: %s" % ("Open3D TSDF (dense)" if HAVE_OPEN3D else "numpy fallback (sparse)"))
    ok = True
    h, w = 120, 160
    pose = (np.eye(3), np.zeros(3))
    Z = 1500.0

    # (1) fuse a few frames -> a mesh exists and raycasts to the true wall depth
    wm = WorldMemory(horizon_s=3.0, f=300.0)
    for i in range(5):
        wm.integrate(_synth_depth(h, w, Z), pose, t=100.0 + 0.1 * i)
    verts, faces = wm.mesh()
    built = len(faces) > 0
    print("  fuses depth frames into a mesh             : %s  (%d verts, %d faces)"
          % ("PASS" if built else "FAIL", len(verts), len(faces)))
    ok &= built

    pts, dists, valid = wm.raycast([[0, 0, 0]], [[0, 0, 1]])
    acc = bool(valid[0]) and abs(dists[0] - Z) <= 30.0          # within one voxel
    print("  raycast returns the true wall depth        : %s  (%.1f mm vs %.1f)"
          % ("PASS" if acc else "FAIL", dists[0] if valid[0] else float("nan"), Z))
    ok &= acc

    # (2) MEMORY: a frame where the wall centre drops out (occluder passes / features lost).
    #     The surface must SURVIVE there, because earlier frames still remember it.
    wm.integrate(_synth_depth(h, w, Z, hole=(40, 80, 60, 100)), pose, t=100.5)
    pts2, d2, v2 = wm.raycast([[0, 0, 0]], [[0, 0, 1]])
    survived = bool(v2[0]) and abs(d2[0] - Z) <= 30.0
    print("  surface survives a dropout frame           : %s" % ("PASS" if survived else "FAIL"))
    ok &= survived

    # (3) FORGETTING: push past the horizon with only dropout frames -> the remembered
    #     surface must age out rather than persist forever.
    for i in range(12):
        wm.integrate(_synth_depth(h, w, Z, hole=(0, h, 0, w)), pose, t=104.0 + 0.1 * i)
    kept_span = wm.age_span()
    horizon_ok = kept_span <= wm.horizon_s + 1e-6
    print("  memory window bounded to horizon           : %s  (%.2f s <= %.2f s)"
          % ("PASS" if horizon_ok else "FAIL", kept_span, wm.horizon_s))
    ok &= horizon_ok

    _, d3, v3 = wm.raycast([[0, 0, 0]], [[0, 0, 1]])
    forgot = not bool(v3[0])
    print("  stale geometry is forgotten                : %s" % ("PASS" if forgot else "FAIL"))
    ok &= forgot

    # (4) pruning drops old frames and keeps the newest
    wm2 = WorldMemory(horizon_s=1.0, f=300.0)
    for i in range(10):
        wm2.integrate(_synth_depth(h, w, Z), pose, t=200.0 + 0.3 * i)
    within = all(wm2.frames[-1][0] - t <= 1.0 + 1e-9 for t, _, _ in wm2.frames)
    print("  prune keeps only in-horizon frames         : %s  (%d frames)"
          % ("PASS" if within else "FAIL", len(wm2.frames)))
    ok &= within

    # (5) frame cap bounds memory regardless of rate
    wm3 = WorldMemory(horizon_s=1e6, max_frames=5, f=300.0)
    for i in range(20):
        wm3.integrate(_synth_depth(h, w, Z), pose, t=300.0 + i)
    capped = len(wm3.frames) == 5
    print("  frame cap bounds memory                    : %s  (%d)"
          % ("PASS" if capped else "FAIL", len(wm3.frames)))
    ok &= capped

    # (6) a ray into empty space must miss rather than invent a surface
    _, _, vmiss = wm2.raycast([[0, 0, 0]], [[0, 1, 0]])
    print("  ray into empty space misses                : %s" % ("PASS" if not vmiss[0] else "FAIL"))
    ok &= not bool(vmiss[0])

    # (7) pose handling: observe the same wall from a translated camera; the world-frame
    #     surface must land at the SAME world depth, not move with the camera.
    wm4 = WorldMemory(horizon_s=5.0, f=300.0)
    shift = 200.0
    wm4.integrate(_synth_depth(h, w, Z), (np.eye(3), np.zeros(3)), t=400.0)
    wm4.integrate(_synth_depth(h, w, Z - shift), (np.eye(3), np.array([0.0, 0.0, shift])), t=400.1)
    _, d4, v4 = wm4.raycast([[0, 0, 0]], [[0, 0, 1]])
    consistent = bool(v4[0]) and abs(d4[0] - Z) <= 40.0
    print("  poses fuse into ONE world surface          : %s  (%.1f mm vs %.1f)"
          % ("PASS" if consistent else "FAIL", d4[0] if v4[0] else float("nan"), Z))
    ok &= consistent

    print("WORLD MEMORY OK" if ok else "WORLD MEMORY FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or len(sys.argv) == 1:
        sys.exit(0 if selftest() else 1)
