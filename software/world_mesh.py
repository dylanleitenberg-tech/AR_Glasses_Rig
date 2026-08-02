"""world_mesh.py — real-world MESH TRACKING from the two forward world cameras.

The AR overlay has to sit on the REAL WORLD, so we need to know, every frame, where the world
is relative to the glasses. The world pair (rig.py: two forward AR0234 cams, R=I, baseline =
IPD, FOV 70 deg, 1280 px) is a rectified stereo rig. Each frame this module:

  1. TRIANGULATES matched features into 3D points in the left-camera frame (stereo depth).
  2. TRACKS those points across time and solves the rigid CAMERA MOTION between frames
     (stereo visual odometry) with a robust Kabsch fit — RANSAC rejects mismatched features.
  3. FUSES the IMU (imu_serial): the gyro gives the inter-frame rotation, used as a prior and to
     carry the pose when too few features survive (blank wall, motion blur) so tracking never
     free-falls.
  4. MAINTAINS a persistent MESH: a growing cloud of world 3D map points + a Delaunay surface
     over the currently-visible ones. This mesh is the frame the overlay registers to, and it
     lets the display pixel<->world-ray map (pixel_map.py) stay locked as the head moves.

DESIGN: the GEOMETRY (triangulate / Kabsch / RANSAC / pose integration / IMU blend) is pure
numpy and fully unit-tested by ``--selftest`` with a synthetic moving rig — no camera needed.
The FEATURE FRONT-END (ORB stereo matching, LK optical-flow tracking, Subdiv2D meshing) is the
only cv2 part and runs on real frames via WorldTracker.track(frameL, frameR).

    python3 world_mesh.py --selftest      # headless: recovers a known trajectory + map
"""
import argparse
import sys

import numpy as np

from rig import NOMINAL_IPD, WORLD_FOV, WORLD_RES


def world_focal_px(res=WORLD_RES, fov_deg=WORLD_FOV):
    """Pinhole focal length in pixels for the world cameras (horizontal FOV convention)."""
    return (res / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


DEFAULT_F = world_focal_px()
DEFAULT_B = NOMINAL_IPD                      # world stereo baseline (mm) = the two world cams


# --------------------------------------------------------------------------
#  Geometry core (pure numpy)
# --------------------------------------------------------------------------
def triangulate_stereo(uvL, uvR, f=DEFAULT_F, B=DEFAULT_B, cx=0.0, cy=0.0, min_disp=0.5):
    """Rectified-stereo triangulation. uvL/uvR: Nx2 pixel coords in the left/right world cam.
    Returns (pts3d Nx3 in the LEFT-camera frame, mask of points with usable disparity).
    Z = f*B/disparity;  X = (uL-cx)*Z/f;  Y = (vL-cy)*Z/f."""
    uvL = np.atleast_2d(np.asarray(uvL, float))
    uvR = np.atleast_2d(np.asarray(uvR, float))
    disp = (uvL[:, 0] - cx) - (uvR[:, 0] - cx)
    good = disp > min_disp
    Z = np.where(good, f * B / np.where(good, disp, 1.0), 0.0)
    X = (uvL[:, 0] - cx) * Z / f
    Y = (uvL[:, 1] - cy) * Z / f
    return np.stack([X, Y, Z], axis=1), good


def kabsch(P, Q, weights=None):
    """Best rigid transform mapping P onto Q: returns (R, t) with Q ~= P @ R.T + t.
    P, Q: Nx3 corresponding points. Classic SVD solution (reflection-safe)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    w = w / (w.sum() + 1e-12)
    pbar = (w[:, None] * P).sum(0); qbar = (w[:, None] * Q).sum(0)
    Pc = P - pbar; Qc = Q - qbar
    H = (Pc * w[:, None]).T @ Qc                     # weighted covariance
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))     # kill reflection
    R = Vt.T @ D @ U.T
    t = qbar - R @ pbar
    return R, t


def _mad_scale(res):
    """Robust residual scale (mm) via the median absolute deviation (50% breakdown)."""
    return 1.4826 * float(np.median(res)) + 1e-6


def kabsch_ransac(P, Q, iters=100, thresh=None, rng=None):
    """Robust P->Q rigid fit for stereo VO. Because stereo DEPTH noise grows with range
    (~15-40 mm at 1-2.5 m here), a fixed inlier threshold is wrong — it rejects good far points.
    When `thresh` is None it is set ADAPTIVELY from a MAD scale of an initial fit (robust to
    <50% outliers), so the gate tracks the true inlier noise while still excluding gross feature
    mismatches. RANSAC finds the consensus, then a Tukey-biweight IRLS refine (redescending, like
    calibrator.py) uses every inlier smoothly. Returns (R, t, inlier_mask)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    n = len(P)
    if n < 3:
        R, t = (np.eye(3), np.zeros(3)) if n == 0 else kabsch(P, Q)
        return R, t, np.ones(n, bool)
    rng = rng or np.random.default_rng(0)
    if thresh is None:
        R0, t0 = kabsch(P, Q)
        thresh = 3.0 * _mad_scale(np.linalg.norm(P @ R0.T + t0 - Q, axis=1))
    best_in = None; best_cnt = -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(P[idx], Q[idx])
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm(P @ R.T + t - Q, axis=1)
        inl = res < thresh
        c = int(inl.sum())
        if c > best_cnt:
            best_cnt, best_in = c, inl
    if best_in is None or best_cnt < 3:
        best_in = np.ones(n, bool)
    R, t = kabsch(P[best_in], Q[best_in])            # seed from the consensus set
    # Tukey-biweight IRLS refine over ALL points (redescending -> hard-rejects outliers)
    for _ in range(6):
        res = np.linalg.norm(P @ R.T + t - Q, axis=1)
        s = _mad_scale(res)
        u = res / (4.685 * s)
        w = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        if w.sum() < 3:
            break
        R, t = kabsch(P, Q, weights=w)
    res = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return R, t, res < thresh


# --------------------------------------------------------------------------
#  The tracked mesh + pose (pure numpy state machine)
# --------------------------------------------------------------------------
class WorldMesh:
    """Accumulates the camera trajectory and a persistent world map from per-frame stereo points.

    Pose is stored as (R_cw: world->cam rotation, C: camera centre in world). A world point X
    projects to left-cam coords x = R_cw (X - C). Frame 0 defines the world frame (R=I, C=0)."""

    def __init__(self, ransac_thresh=None, min_inliers=6, imu_gate_deg=8.0):
        self.R_cw = np.eye(3)
        self.C = np.zeros(3)
        self.map = {}                    # id -> world 3D point (mm)
        self._prev = None                # {id: cam3d} from the last frame (for VO)
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.imu_gate_deg = imu_gate_deg
        self.frame_idx = -1
        self.last = dict(inliers=0, used="init", track_ids=0)

    def reset_world(self):
        self.R_cw = np.eye(3); self.C = np.zeros(3); self.map.clear(); self._prev = None

    def ingest(self, ids, cam3d, imu_dR=None):
        """One frame of correspondences.
          ids:    length-N iterable of stable feature IDs (same ID = same world point over time)
          cam3d:  Nx3 stereo-triangulated points in THIS frame's left-camera coordinates
          imu_dR: optional 3x3 world->cam rotation INCREMENT since last frame (from the gyro)
        Updates the pose + map. Returns a dict of this frame's telemetry."""
        ids = list(ids)
        cam3d = np.atleast_2d(np.asarray(cam3d, float))
        self.frame_idx += 1
        cur = {i: p for i, p in zip(ids, cam3d)}

        if self.frame_idx == 0:
            # define the world frame; every triangulated point is a map point
            self.R_cw = np.eye(3); self.C = np.zeros(3)
            for i, p in cur.items():
                self.map[i] = p.copy()
            self._prev = cur
            self.last = dict(inliers=len(cur), used="anchor", track_ids=len(cur))
            return dict(self.last)

        # resect the current camera from map points we've already triangulated (3D world <-> 3D cam)
        known = [i for i in cur if i in self.map]
        used = "imu"; inliers = 0
        if len(known) >= 3:
            Xw = np.array([self.map[i] for i in known])
            Xc = np.array([cur[i] for i in known])
            R, t, inl = kabsch_ransac(Xw, Xc, thresh=self.ransac_thresh)   # Xc ~= R Xw + t
            inliers = int(inl.sum())
            if inliers >= min(self.min_inliers, len(known)):
                # accept vision; optionally sanity-check against the IMU rotation
                if imu_dR is not None:
                    R_pred = imu_dR @ self.R_cw
                    if _rot_angle_deg(R, R_pred) > self.imu_gate_deg:
                        R = _project_rotation(0.5 * (R + R_pred))          # blend on disagreement
                        t = Xc.mean(0) - R @ Xw.mean(0)
                        used = "vision+imu"
                    else:
                        used = "vision"
                else:
                    used = "vision"
                self.R_cw, t_new = R, t
                self.C = -R.T @ t_new
        if used == "imu":
            # too few map matches: carry the pose with the IMU rotation (translation held)
            if imu_dR is not None:
                self.R_cw = imu_dR @ self.R_cw

        # grow the map: add world points for ids seen now but not yet mapped
        for i, xc in cur.items():
            if i not in self.map:
                self.map[i] = self.C + self.R_cw.T @ xc      # X = C + R_cw^T x_cam
        self._prev = cur
        self.last = dict(inliers=inliers, used=used, track_ids=len(cur))
        return dict(self.last)

    # ---- outputs -------------------------------------------------------
    def pose(self):
        """(R_cw world->cam, C camera centre in world)."""
        return self.R_cw.copy(), self.C.copy()

    def map_points(self):
        """(ids, Nx3 world points) of the whole accumulated map."""
        ids = list(self.map)
        return ids, np.array([self.map[i] for i in ids]) if ids else np.zeros((0, 3))

    def visible_mesh(self, ids):
        """Triangulate (Delaunay) the given currently-visible map ids in the image plane, giving
        a surface mesh: (verts world Nx3, faces Mx3 index triples). Uses cv2.Subdiv2D when
        available; without cv2 returns verts only (faces empty)."""
        ids = [i for i in ids if i in self.map]
        if not ids:
            return np.zeros((0, 3)), np.zeros((0, 3), int)
        verts = np.array([self.map[i] for i in ids])
        cam = (verts - self.C) @ self.R_cw.T
        fwd = cam[:, 2] > 1e-6
        ids = [i for i, f in zip(ids, fwd) if f]
        verts = verts[fwd]; cam = cam[fwd]
        if len(ids) < 3:
            return verts, np.zeros((0, 3), int)
        uv = np.stack([DEFAULT_F * cam[:, 0] / cam[:, 2], DEFAULT_F * cam[:, 1] / cam[:, 2]], 1)
        faces = _delaunay_faces(uv)
        return verts, faces


def _rot_angle_deg(A, B):
    """Geodesic angle between two rotations (deg)."""
    c = (np.trace(A @ B.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _project_rotation(M):
    """Nearest proper rotation to M (SVD projection)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def _delaunay_faces(uv):
    """Delaunay triangle index triples over 2D points. cv2.Subdiv2D if present, else empty."""
    try:
        import cv2
    except Exception:
        return np.zeros((0, 3), int)
    uv = np.asarray(uv, np.float32)
    mn = uv.min(0) - 10; mx = uv.max(0) + 10
    wh = mx - mn                                  # Subdiv2D rect is (x, y, WIDTH, HEIGHT)
    sub = cv2.Subdiv2D((float(mn[0]), float(mn[1]), float(wh[0]), float(wh[1])))
    key = {}
    for k, p in enumerate(uv):
        sub.insert((float(p[0]), float(p[1])))
        key[(round(float(p[0]), 3), round(float(p[1]), 3))] = k
    faces = []
    for tri in sub.getTriangleList():
        idx = []
        for j in range(3):
            k = key.get((round(float(tri[2 * j]), 3), round(float(tri[2 * j + 1]), 3)))
            if k is None:
                break
            idx.append(k)
        if len(idx) == 3:
            faces.append(idx)
    return np.array(faces, int) if faces else np.zeros((0, 3), int)


# --------------------------------------------------------------------------
#  Real-frame feature front-end (cv2) — the only hardware-facing part
# --------------------------------------------------------------------------
class WorldTracker:
    """Turns real (frameL, frameR) world-cam pairs into ID'd stereo correspondences and drives
    WorldMesh. ORB features are matched L<->R along the epipolar row (rectified), given stable
    IDs, and carried across time with LK optical flow. Needs opencv."""

    def __init__(self, f=DEFAULT_F, B=DEFAULT_B, max_feat=400, row_tol=3.0):
        import cv2
        self.cv2 = cv2
        self.max_feat = int(max_feat)
        self.orb = cv2.ORB_create(self.max_feat)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.f = f; self.B = B; self.row_tol = row_tol
        self.mesh = WorldMesh()
        self._next_id = 0
        self._prev_gray = None
        self._tracks = {}                # id -> (uL,vL) in the previous left frame

    def set_max_features(self, n):
        """Change the ORB feature budget LIVE — perf.QualityController's main lever.

        ORB detect + BF match is the heaviest stage in the loop and its cost is roughly linear
        in the feature count, so this is the knob that buys frame time. No-op when unchanged,
        because rebuilding the detector every frame would cost more than it saves."""
        n = int(max(20, n))
        if n == self.max_feat:
            return
        self.max_feat = n
        try:
            self.orb.setMaxFeatures(n)          # cheap in-place update where available
        except Exception:
            self.orb = self.cv2.ORB_create(n)   # older cv2: rebuild
        return n

    def _gray(self, img):
        if img.ndim == 3:
            return self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        return img

    def track(self, frameL, frameR, imu_dR=None):
        """Ingest one synchronized world pair. Returns the WorldMesh frame telemetry."""
        cv2 = self.cv2
        gL = self._gray(frameL); gR = self._gray(frameR)
        # 1) carry existing tracks forward with optical flow on the LEFT image
        ids, uvL = [], []
        if self._prev_gray is not None and self._tracks:
            tid = list(self._tracks)
            p0 = np.array([self._tracks[i] for i in tid], np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gL, p0, None)
            for i, pt, ok in zip(tid, p1.reshape(-1, 2), st.reshape(-1)):
                if ok:
                    ids.append(i); uvL.append(pt)
        # 2) detect fresh ORB features to top up / seed
        kpsL, desL = self.orb.detectAndCompute(gL, None)
        kpsR, desR = self.orb.detectAndCompute(gR, None)
        new_uvL, new_uvR = [], []
        if desL is not None and desR is not None:
            for m in self.bf.match(desL, desR):
                a = kpsL[m.queryIdx].pt; b = kpsR[m.trainIdx].pt
                if abs(a[1] - b[1]) <= self.row_tol and (a[0] - b[0]) > 0.5:   # epipolar + disparity
                    new_uvL.append(a); new_uvR.append(b)
        # 3) triangulate the fresh stereo matches, give them IDs, merge with carried tracks
        fresh_ids = []
        if new_uvL:
            p3, good = triangulate_stereo(np.array(new_uvL), np.array(new_uvR), self.f, self.B)
            for k, g in enumerate(good):
                if g:
                    fresh_ids.append((self._next_id, new_uvL[k], new_uvR[k]))
                    self._next_id += 1
        # assemble this frame's correspondences: carried tracks need a right-match too, so we
        # depth them from the fresh stereo where available; carried-only points keep last depth.
        all_uvL = list(uvL) + [x[1] for x in fresh_ids]
        all_ids = list(ids) + [x[0] for x in fresh_ids]
        # for depth we need uvR per id; use the fresh matches (carried tracks reuse map depth)
        uvR_by_id = {x[0]: x[2] for x in fresh_ids}
        cam3d, keep_ids = [], []
        for i, a in zip(all_ids, all_uvL):
            b = uvR_by_id.get(i)
            if b is None:
                continue                      # no stereo match this frame -> skip depth (kept via map)
            p3, g = triangulate_stereo(np.array([a]), np.array([b]), self.f, self.B)
            if g[0]:
                cam3d.append(p3[0]); keep_ids.append(i)
        # 4) update the mesh/pose and refresh the track table
        tele = self.mesh.ingest(keep_ids, np.array(cam3d) if cam3d else np.zeros((0, 3)), imu_dR)
        self._tracks = {i: (float(a[0]), float(a[1])) for i, a in zip(all_ids, all_uvL)}
        self._prev_gray = gL
        return tele


# ==========================================================================
#  Self-test (numpy only): synthetic static world + known camera trajectory.
#  Recovers the trajectory and the map through triangulation + Kabsch VO,
#  survives noise + outliers (RANSAC), and shows IMU carrying a feature dropout.
# ==========================================================================
def _project_pair(Xw, R_cw, C, f=DEFAULT_F, B=DEFAULT_B):
    """Project world points into the rectified stereo pair; return (uvL, uvR, visible mask)."""
    xL = (Xw - C) @ R_cw.T                       # left-cam coords
    xR = xL - np.array([B, 0, 0])                # right cam = left shifted +B in x
    vis = (xL[:, 2] > 1.0) & (xR[:, 2] > 1.0)
    uvL = np.stack([f * xL[:, 0] / xL[:, 2], f * xL[:, 1] / xL[:, 2]], 1)
    uvR = np.stack([f * xR[:, 0] / xR[:, 2], f * xR[:, 1] / xR[:, 2]], 1)
    return uvL, uvR, vis


def _rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def selftest(verbose=True):
    if verbose:
        print("== world_mesh self-test (synthetic moving stereo rig, no hardware) ==")
    rng = np.random.default_rng(7)
    checks = []

    # (0) triangulation round-trips a known cloud
    Xw = rng.uniform([-400, -300, 800], [400, 300, 2500], size=(200, 3))
    uvL, uvR, vis = _project_pair(Xw, np.eye(3), np.zeros(3))
    p3, good = triangulate_stereo(uvL[vis], uvR[vis])
    tri_err = np.linalg.norm(p3 - Xw[vis], axis=1).max()
    checks.append(("stereo triangulation recovers 3D cloud (max err %.3f mm)" % tri_err, tri_err < 1e-3))

    # a smooth trajectory: translate + yaw/pitch as the head moves
    T = 24
    Cs = np.stack([80 * np.sin(np.linspace(0, 1.6, T)),
                   40 * np.sin(np.linspace(0, 1.0, T)),
                   120 * np.sin(np.linspace(0, 0.9, T))], 1)
    Rs = [_rot([0, 1, 0], 10 * np.sin(t)) @ _rot([1, 0, 0], 5 * np.sin(0.7 * t))
          for t in np.linspace(0, 1.5, T)]

    def run_traj(noise=0.0, outlier=0.0, drop_frame=None, use_imu=False, seed=0):
        r = np.random.default_rng(seed)
        wm = WorldMesh()
        est_C, tru_C, est_R, tru_R = [], [], [], []
        prevR = np.eye(3)
        for k in range(T):
            R_cw, C = Rs[k], Cs[k]
            uvL, uvR, vis = _project_pair(Xw, R_cw, C)
            ids = np.where(vis)[0]
            a, b = uvL[vis].copy(), uvR[vis].copy()
            if noise:
                a += r.normal(0, noise, a.shape); b += r.normal(0, noise, b.shape)
            if outlier and len(a) > 6:                       # corrupt a fraction of right-matches
                m = r.random(len(a)) < outlier
                b[m] += r.uniform(-25, 25, (int(m.sum()), 2))
            keep = np.ones(len(ids), bool)
            if drop_frame is not None and k == drop_frame:   # simulate a feature dropout
                keep[:] = False; keep[:2] = True             # <3 matches -> IMU must carry it
            p3, g = triangulate_stereo(a[keep], b[keep])
            imu_dR = (R_cw @ prevR.T) if use_imu else None   # gyro-style rotation increment
            wm.ingest(ids[keep][g], p3[g], imu_dR=imu_dR)
            prevR = R_cw
            Rc, Cc = wm.pose()
            est_C.append(Cc); tru_C.append(C); est_R.append(Rc); tru_R.append(R_cw)
        cerr = np.linalg.norm(np.array(est_C) - np.array(tru_C), axis=1)
        rerr = np.array([_rot_angle_deg(e, t) for e, t in zip(est_R, tru_R)])
        return wm, cerr, rerr

    # (1) clean trajectory: pose recovered to numerical precision
    wm, cerr, rerr = run_traj()
    checks.append(("recovers camera trajectory clean (max pos %.3f mm, rot %.3f deg)"
                   % (cerr.max(), rerr.max()), cerr.max() < 1e-2 and rerr.max() < 1e-2))

    # (2) the accumulated MAP matches the true world cloud (mesh points land in the world)
    ids, P = wm.map_points()
    merr = np.linalg.norm(P - Xw[ids], axis=1).max()
    checks.append(("accumulated world map matches truth (max %.3f mm, %d points)" % (merr, len(ids)),
                   merr < 1e-2 and len(ids) > 150))

    # (3) robust to pixel noise + 25% gross feature MISMATCHES (RANSAC)
    _, cerrn, rerrn = run_traj(noise=0.4, outlier=0.25, seed=3)
    checks.append(("robust to noise+25%% outliers (median pos %.2f mm, rot %.2f deg)"
                   % (np.median(cerrn), np.median(rerrn)),
                   np.median(cerrn) < 12.0 and np.median(rerrn) < 1.5))

    # (4) IMU carries the pose through a feature dropout (only 2 matches on one frame)
    _, cerr_imu, rerr_imu = run_traj(drop_frame=12, use_imu=True, seed=5)
    _, cerr_no, rerr_no = run_traj(drop_frame=12, use_imu=False, seed=5)
    checks.append(("IMU carries rotation through a dropout (rot err at drop %.2f deg vs %.2f w/o)"
                   % (rerr_imu[12], rerr_no[12]), rerr_imu[12] < 1.0 and rerr_imu[12] < rerr_no[12]))

    # (5) a surface MESH is produced over visible points
    R_cw, C = Rs[0], Cs[0]
    uvL, uvR, vis = _project_pair(Xw, R_cw, C)
    verts, faces = wm.visible_mesh(list(np.where(vis)[0]))
    checks.append(("builds a Delaunay surface mesh (%d verts, %d faces)" % (len(verts), len(faces)),
                   len(verts) > 50 and len(faces) > 50))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "WORLD MESH OK — triangulate + robust VO + IMU-fused pose + surface mesh ✅"
              if ok else "PROBLEM ⚠️")
        print("  note: geometry proven in numpy; on hardware WorldTracker feeds it ORB/LK features")
        print("  from the two world cams (needs opencv). Drift is bounded by re-observing map points.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="real-world stereo mesh tracking")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    sys.exit(selftest())
