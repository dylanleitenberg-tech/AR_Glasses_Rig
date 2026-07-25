"""people_track.py — find and TRACK people in the world cameras, and put each one in 3D.

For a world-locked "turn people into monkeys" overlay we need, every frame, each visible person's
stable identity and 3D world position (so the monkey stays glued to them as they AND the wearer's
head move). This module:

  1. DETECT people in both world cameras (pluggable detector; the default `CvPeopleDetector` uses
     OpenCV's built-in HOG pedestrian detector + Haar face cascade — no model downloads. A YOLO /
     MediaPipe detector drops into the same `detect(frame) -> [Detection]` seam for quality).
  2. STEREO-MATCH the left/right detections of the same person (rectified: matching image rows,
     positive disparity) and TRIANGULATE the box centre -> depth -> a 3D point, then lift it to the
     WORLD frame with the current mesh pose (so it's an anchor, not a camera-relative blob).
  3. TRACK across time: greedy nearest-neighbour association on the ground plane, an alpha-beta
     (constant-velocity) filter per person, stable IDs, and DEAD-RECKONING through short detector
     dropouts / occlusions so a fast walker's monkey doesn't pop off. New people spawn tracks; ones
     that leave age out.
  4. Derive named ANCHOR POINTS (head, feet, shoulders, centroid) + a metric height + a facing
     estimate so avatar.py can pose a monkey to the body.

The detector is the only hardware/vision-model part; the stereo geometry + tracking core is pure
numpy and fully exercised by `--selftest` with synthetic walking people and a moving head.
"""
import argparse
import sys

import numpy as np

from world_mesh import DEFAULT_F, DEFAULT_B
from rig import WORLD_RES

UP = np.array([0.0, 1.0, 0.0])          # rig world frame: +y is up


class Detection:
    """A person box in ONE world camera, in pixel coords (u,v = centre; w,h = size)."""
    def __init__(self, u, v, w, h, score=1.0, face=None):
        self.u = float(u); self.v = float(v); self.w = float(w); self.h = float(h)
        self.score = float(score)
        self.face = face                # optional (u,v,w,h) face sub-box

    @property
    def center(self):
        return np.array([self.u, self.v])


class TrackedPerson:
    def __init__(self, tid, world_pos, height_mm):
        self.id = tid
        self.pos = np.asarray(world_pos, float)     # world 3D centroid (mm)
        self.vel = np.zeros(3)                       # world velocity (mm/frame)
        self.height = float(height_mm)
        self.facing = np.array([0.0, 0.0, -1.0])     # unit horizontal facing (toward -z default)
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.points = {}
        self._recompute_points(None)

    def _recompute_points(self, cam_center):
        """Named world anchor points from the centroid + height + facing (for the avatar)."""
        h = self.height
        # facing horizontal; right = up x facing (a horizontal perpendicular)
        f = self.facing.copy(); f[1] = 0.0
        n = np.linalg.norm(f); f = f / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
        self.facing = f
        right = np.cross(UP, f); right = right / (np.linalg.norm(right) + 1e-9)
        width = 0.42 * h
        self.points = {
            "centroid": self.pos.copy(),
            "head":  self.pos + UP * (h * 0.5),
            "feet":  self.pos - UP * (h * 0.5),
            "l_shoulder": self.pos + UP * (h * 0.28) - right * (width * 0.5),
            "r_shoulder": self.pos + UP * (h * 0.28) + right * (width * 0.5),
        }

    def predict(self):
        """Advance one frame by the constant-velocity model (dead-reckoning when unseen)."""
        self.pos = self.pos + self.vel
        self.age += 1
        self._recompute_points(None)

    def correct(self, meas_pos, meas_h, alpha=0.6, beta=0.25, cam_center=None):
        """Alpha-beta update toward a measurement; refresh height, facing, anchor points."""
        resid = meas_pos - self.pos
        self.pos = self.pos + alpha * resid
        self.vel = self.vel + beta * resid
        self.height = 0.7 * self.height + 0.3 * meas_h
        # facing: walking direction if moving, else face the wearer
        vh = self.vel.copy(); vh[1] = 0.0
        if np.linalg.norm(vh) > 3.0:
            self.facing = vh / np.linalg.norm(vh)
        elif cam_center is not None:
            d = np.asarray(cam_center, float) - self.pos; d[1] = 0.0
            if np.linalg.norm(d) > 1e-6:
                self.facing = d / np.linalg.norm(d)
        self.hits += 1; self.misses = 0
        self._recompute_points(cam_center)


class PeopleTracker:
    def __init__(self, f=DEFAULT_F, B=DEFAULT_B, res=WORLD_RES,
                 gate_mm=700.0, max_miss=8, min_hits=2, row_tol=0.06, detector=None):
        self.f = f; self.B = B
        self.W = res; self.H = int(res * 3 // 4)
        self.cx = self.W / 2.0; self.cy = self.H / 2.0
        self.gate = gate_mm; self.max_miss = max_miss; self.min_hits = min_hits
        self.row_tol = row_tol * self.H          # vertical match tolerance (px)
        self.detector = detector
        self.tracks = []
        self._next_id = 0

    # ---- stereo -> world measurements -----------------------------------
    def _stereo_match(self, detL, detR):
        """Greedy match each L box to an R box (same row, positive disparity, similar size)."""
        pairs = []
        usedR = set()
        for dl in sorted(detL, key=lambda d: -d.score):
            best = None; best_cost = 1e18
            for j, dr in enumerate(detR):
                if j in usedR:
                    continue
                disp = dl.u - dr.u
                if disp <= 0.5:
                    continue
                if abs(dl.v - dr.v) > self.row_tol:
                    continue
                cost = abs(dl.v - dr.v) + 0.5 * abs(dl.h - dr.h)
                if cost < best_cost:
                    best_cost = cost; best = j
            if best is not None:
                usedR.add(best); pairs.append((dl, detR[best]))
        return pairs

    def _measure(self, dl, dr, pose):
        """Matched box pair -> (world centroid, metric height). Uses the mesh pose to lift the
        camera-frame 3D point into the world frame."""
        R_cw, C = pose
        disp = dl.u - dr.u
        Z = self.f * self.B / disp
        x_cam = np.array([(dl.u - self.cx) * Z / self.f, (dl.v - self.cy) * Z / self.f, Z])
        world = np.asarray(C, float) + R_cw.T @ x_cam
        height = dl.h * Z / self.f                    # box pixel height -> metric height
        return world, height

    # ---- the per-frame update -------------------------------------------
    def update(self, detL, detR, pose):
        """One synchronized frame: left/right detections + mesh pose -> active TrackedPersons.
        On hardware, call update_frames(frameL, frameR, pose) which runs the detector first."""
        R_cw, C = pose
        cam_center = np.asarray(C, float)
        pairs = self._stereo_match(detL, detR)
        meas = [self._measure(dl, dr, pose) for dl, dr in pairs]

        # predict all tracks forward, then associate measurements (greedy nearest on ground plane)
        for t in self.tracks:
            t.predict()
        unmatched = set(range(len(meas)))
        for t in self.tracks:
            best = None; best_d = self.gate
            for i in unmatched:
                d = np.linalg.norm((meas[i][0] - t.pos) * np.array([1, 0.3, 1]))   # weight XZ
                if d < best_d:
                    best_d = d; best = i
            if best is not None:
                t.correct(meas[best][0], meas[best][1], cam_center=cam_center)
                unmatched.discard(best)
            else:
                t.misses += 1

        # spawn tracks for unmatched measurements; age out stale tracks
        for i in unmatched:
            self.tracks.append(TrackedPerson(self._next_id, meas[i][0], meas[i][1]))
            self.tracks[-1].facing = self._toward(cam_center, meas[i][0])
            self.tracks[-1]._recompute_points(cam_center)
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_miss]
        return self.confirmed()

    def update_frames(self, frameL, frameR, pose):
        """Hardware path: detect in both world frames, then update()."""
        if self.detector is None:
            raise RuntimeError("no detector set (use CvPeopleDetector or inject one)")
        return self.update(self.detector.detect(frameL), self.detector.detect(frameR), pose)

    def _toward(self, cam_center, pos):
        d = np.asarray(cam_center, float) - pos; d[1] = 0.0
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])

    def confirmed(self):
        """Tracks stable enough to draw (seen a few times, or briefly coasting)."""
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses <= self.max_miss]


# --------------------------------------------------------------------------
#  Real detector (OpenCV built-ins) — the only hardware/vision-model part
# --------------------------------------------------------------------------
class CvPeopleDetector:
    """People via HOG + faces via Haar (both bundled with opencv-python; no downloads). Returns
    pixel-coord Detections. Swap for a DNN detector by matching detect(frame)->[Detection]."""

    def __init__(self, min_score=0.3):
        import cv2
        self.cv2 = cv2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.min_score = min_score

    def detect(self, frame):
        cv2 = self.cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), scale=1.05)
        faces = self.face.detectMultiScale(gray, 1.1, 4)
        dets = []
        for (x, y, w, h), score in zip(rects, weights):
            if score < self.min_score:
                continue
            fb = None
            for (fx, fy, fw, fh) in faces:                       # attach a face inside this body box
                if x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h:
                    fb = (fx + fw / 2, fy + fh / 2, fw, fh); break
            dets.append(Detection(x + w / 2, y + h / 2, w, h, float(score), fb))
        return dets


# ==========================================================================
#  Self-test (no hardware): synthetic walking people + a moving head prove the
#  stereo 3D recovery, stable IDs, dead-reckoning through a dropout, spawn/age.
# ==========================================================================
def _project_box(world_centroid, height, R_cw, C, f, cx, cy, W, H, dx=0.0):
    """Project a person (centroid + height) to a pixel box in one world cam, or None."""
    head = world_centroid + UP * (height * 0.5)
    feet = world_centroid - UP * (height * 0.5)
    def proj(P):
        x = R_cw @ (P - C) - np.array([dx, 0, 0])
        if x[2] <= 1e-6:
            return None
        return np.array([f * x[0] / x[2] + cx, f * x[1] / x[2] + cy])
    ph, pf, pc = proj(head), proj(feet), proj(world_centroid)
    if ph is None or pf is None or pc is None:
        return None
    boxh = abs(pf[1] - ph[1]); boxw = 0.42 * boxh
    if not (0 <= pc[0] <= W and 0 <= pc[1] <= H):
        return None
    return Detection(pc[0], pc[1], boxw, boxh, 1.0)


def selftest(verbose=True):
    if verbose:
        print("== people_track self-test (synthetic walking people + moving head) ==")
    f = DEFAULT_F; B = DEFAULT_B; W = WORLD_RES; H = int(W * 3 // 4); cx = W / 2; cy = H / 2
    trk = PeopleTracker()
    checks = []

    # three people ~1.7 m tall, walking with different velocities, 2-4 m in front
    people = [dict(pos=np.array([-500.0, 0, 1600.0]), vel=np.array([25.0, 0, 5.0]), h=1700.0),
              dict(pos=np.array([300.0, 0, 2600.0]), vel=np.array([-18.0, 0, 0.0]), h=1650.0),
              dict(pos=np.array([-100.0, 0, 3400.0]), vel=np.array([0.0, 0, -20.0]), h=1750.0)]
    poses = []
    def head_pose(k):
        yaw = 6 * np.sin(k * 0.15)
        th = np.radians(yaw)
        R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        C = np.array([10 * np.sin(k * 0.2), 0.0, 5 * k * 0.0])
        return R, C

    id_history = {i: set() for i in range(3)}
    pos_err = []
    drop_recovered = None
    for k in range(30):
        R_cw, C = head_pose(k)
        detL, detR, truth = [], [], []
        for pi, p in enumerate(people):
            p["pos"] = p["pos"] + p["vel"]
            truth.append(p["pos"].copy())
            dl = _project_box(p["pos"], p["h"], R_cw, C, f, cx, cy, W, H, 0.0)
            dr = _project_box(p["pos"], p["h"], R_cw, C, f, cx, cy, W, H, B)
            # simulate a 2-frame detector DROPOUT of person 0 (occlusion) at k=15,16
            if pi == 0 and k in (15, 16):
                dl = dr = None
            if dl is not None and dr is not None:
                detL.append(dl); detR.append(dr)
        tracks = trk.update(detL, detR, (R_cw, C))
        # nearest track to each truth -> record id + position error
        for pi, tp in enumerate(truth):
            near = min(tracks, key=lambda t: np.linalg.norm(t.pos - tp), default=None)
            if near is not None and np.linalg.norm(near.pos - tp) < 400:
                id_history[pi].add(near.id)
                pos_err.append(np.linalg.norm((near.pos - tp) * np.array([1, 0.3, 1])))
        if k == 17:      # just after person 0's dropout ended -> was it coasted + kept?
            t0 = [t for t in tracks if t.id in id_history[0]]
            drop_recovered = len(t0) > 0 and np.linalg.norm(t0[0].pos - truth[0]) < 500

    # (1) 3D recovery accurate
    checks.append(("stereo 3D world position recovered (median err %.0f mm)" % np.median(pos_err),
                   np.median(pos_err) < 150))
    # (2) stable IDs: each person kept a single ID for essentially the whole run
    stable = all(len(ids) == 1 for ids in id_history.values())
    checks.append(("IDs stable through motion + head turn (per-person id counts %s)"
                   % [len(v) for v in id_history.values()], stable))
    # (3) dead-reckoned through the 2-frame dropout and re-locked
    checks.append(("track survives a 2-frame occlusion (dead-reckoned, same ID)", bool(drop_recovered)))
    # (4) exactly three people tracked (no phantom tracks)
    final = trk.confirmed()
    checks.append(("tracks the right count (3 people, %d tracks)" % len(final), len(final) == 3))
    # (5) anchor points sane on a track (head above feet; shoulders flank centroid; height ~1.7 m)
    t = final[0]
    head_up = t.points["head"][1] > t.points["feet"][1]
    hgt_ok = 1400 < t.height < 2000
    checks.append(("anchor points sane (head above feet, height %.0f mm)" % t.height,
                   head_up and hgt_ok))

    ok = all(v for _, v in checks)
    if verbose:
        for name, v in checks:
            print("  [%s] %s" % ("PASS" if v else "FAIL", name))
        print("  =>", "PEOPLE TRACK OK — stereo 3D + stable IDs + dropout coasting ✅"
              if ok else "PROBLEM ⚠️")
        print("  default detector = OpenCV HOG+Haar (no downloads); swap in YOLO/MediaPipe at the")
        print("  detect() seam for better recall/pose. Geometry+tracking here are model-independent.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="detect + track people in 3D from the world cams")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    sys.exit(selftest())
