"""MediaPipe hai camera và weighted triangulation trong frame D435i."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import yaml

from .perception import Perception, rotation_distance, slerp_rotation


HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))


@dataclass(frozen=True)
class PerceptionResult:
    timed_frame: object
    frame: object
    inference_ms: float


class AsyncPerception:
    """Mỗi camera có MediaPipe instance/thread riêng, luôn xử lý frame mới nhất."""

    def __init__(self, capture, cfg, orientation_source=None, pose_enabled=True,
                 force_hand_side=None, num_hands=2):
        self.capture = capture
        orientation_cfg = dict(cfg.get("orientation", {}))
        if orientation_source is not None:
            orientation_cfg["source"] = orientation_source
        self.engine = Perception(
            cfg["models"]["pose"], cfg["models"]["hand"], num_hands=num_hands,
            min_conf=cfg["models"]["min_conf"],
            depth_cfg=cfg["camera"].get("realsense"), orientation_cfg=orientation_cfg,
            tracking_cfg=cfg.get("models", {}), pose_enabled=pose_enabled,
            force_hand_side=force_hand_side)
        self._lock = threading.Lock()
        self._latest = None
        self._history = deque(maxlen=int(cfg.get("fusion", {}).get("inference_history", 20)))
        self._last_frame_id = -1
        self._running = False
        self._error = None
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"inference-{self.capture.camera_id}", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        try:
            while self._running:
                item = self.capture.latest()
                if item is None or item.frame_id == self._last_frame_id:
                    time.sleep(0.002); continue
                self._last_frame_id = item.frame_id
                t0 = time.monotonic()
                frame = self.engine.process(
                    item.sample.bgr, t=item.host_timestamp_s,
                    depth_m=item.sample.depth_m, depth_intrinsics=item.sample.intrinsics)
                result = PerceptionResult(item, frame, 1000.0 * (time.monotonic() - t0))
                with self._lock:
                    self._latest = result
                    self._history.append(result)
        except Exception as exc:
            self._error = exc
            self._running = False

    def latest(self):
        with self._lock:
            return self._latest

    def history(self):
        with self._lock:
            return tuple(self._history)

    @property
    def error(self):
        return self._error

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.engine.close()


def best_hand(frame, side="right"):
    candidates = [x for x in frame.hand_observations if x.side == side]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.handedness_score * x.association_score * max(x.border_score, 0.05))


def closest_to_primary(primary: AsyncPerception, secondary: AsyncPerception, window_s=0.12):
    """Chọn cặp timestamp gần nhất trong cửa sổ mới, chấp nhận trễ vài frame.

    Đây vẫn là bounded/latest processing: history chỉ để pairing, không tạo hàng
    đợi inference. Độ trễ sẽ được tối ưu sau khi hình học fusion đã đúng.
    """
    hp, hs = primary.history(), secondary.history()
    if not hp or not hs:
        return None, None, None
    newest = max(hp[-1].timed_frame.host_timestamp_s, hs[-1].timed_frame.host_timestamp_s)
    recent_p = [x for x in hp if newest - x.timed_frame.host_timestamp_s <= window_s] or [hp[-1]]
    recent_s = [x for x in hs if newest - x.timed_frame.host_timestamp_s <= window_s] or [hs[-1]]
    p, s = min(((a, b) for a in recent_p for b in recent_s),
               key=lambda pair: (abs(pair[0].timed_frame.host_timestamp_s -
                                     pair[1].timed_frame.host_timestamp_s),
                                 -max(pair[0].timed_frame.host_timestamp_s,
                                      pair[1].timed_frame.host_timestamp_s)))
    delta = abs(s.timed_frame.host_timestamp_s - p.timed_frame.host_timestamp_s)
    return p, s, delta


def observation_confidence(hand):
    if hand is None:
        return 0.0
    # handedness_score là độ chắc nhãn Left/Right, không phải detection confidence.
    # Nó thường rất thấp khi camera nhìn mu/cạnh tay dù 21 landmark vẫn chính xác.
    area_score = float(np.clip(hand.palm_area / 0.003, 0.0, 1.0))
    image_quality = ((0.35 + 0.65 * hand.border_score) *
                     (0.35 + 0.65 * area_score))
    return float(np.clip(hand.association_score * image_quality, 0.0, 1.0))


def load_stereo_calibration(path):
    data = yaml.safe_load(Path(path).read_text())
    if not data.get("quality", {}).get("accepted", False):
        raise ValueError(f"Calibration chua PASS: {path}")
    return data


def _camera(data, name):
    c = data[name]
    return np.asarray(c["K"], float), np.asarray(c["distortion"], float)


def undistort_normalized(points_px, K, dist):
    return cv2.undistortPoints(np.asarray(points_px, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)


def weighted_dlt_point(uv_d, uv_l, P_d, P_l, weight_d=1.0, weight_l=1.0):
    ud, vd = uv_d; ul, vl = uv_l
    A = np.vstack((weight_d * (ud * P_d[2] - P_d[0]),
                   weight_d * (vd * P_d[2] - P_d[1]),
                   weight_l * (ul * P_l[2] - P_l[0]),
                   weight_l * (vl * P_l[2] - P_l[1])))
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-10:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_hand(calibration, points_d_px, points_l_px, weight_d=1.0, weight_l=1.0):
    Kd, dd = _camera(calibration, "d435i")
    Kl, dl = _camera(calibration, "laptop")
    T = calibration["T_laptop_from_d435i"]
    R, t = np.asarray(T["R"], float), np.asarray(T["t_m"], float).reshape(3)
    nd = undistort_normalized(points_d_px, Kd, dd)
    nl = undistort_normalized(points_l_px, Kl, dl)
    Pd = np.c_[np.eye(3), np.zeros(3)]
    Pl = np.c_[R, t]
    points = np.asarray([weighted_dlt_point(a, b, Pd, Pl, weight_d, weight_l)
                         for a, b in zip(nd, nl)])
    rd, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), Kd, dd)
    rl, _ = cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, Kl, dl)
    rd, rl = rd.reshape(-1, 2), rl.reshape(-1, 2)
    err_d = np.linalg.norm(rd - np.asarray(points_d_px), axis=1)
    err_l = np.linalg.norm(rl - np.asarray(points_l_px), axis=1)
    center_l_d = -R.T @ t
    rays_d = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-9)
    rays_l = points - center_l_d
    rays_l /= np.maximum(np.linalg.norm(rays_l, axis=1, keepdims=True), 1e-9)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(rays_d * rays_l, axis=1), -1.0, 1.0)))
    return {"points_d435i": points, "reprojection_d435i_px": rd,
            "reprojection_laptop_px": rl, "error_d435i_px": err_d,
            "error_laptop_px": err_l, "ray_angle_deg": angles}


class RobustHandFusion:
    """Refine 21 điểm bằng reprojection + RGB-D + bone + temporal robust loss."""

    def __init__(self, calibration, cfg=None):
        self.calibration = calibration
        self.cfg = cfg or {}
        self.Kd, self.dd = _camera(calibration, "d435i")
        self.Kl, self.dl = _camera(calibration, "laptop")
        T = calibration["T_laptop_from_d435i"]
        self.R = np.asarray(T["R"], float)
        self.rvec = cv2.Rodrigues(self.R)[0]
        self.t = np.asarray(T["t_m"], float).reshape(3)
        self.previous = None
        self.previous_t = None
        self.bone_lengths = None

    @staticmethod
    def _bone_lengths(points):
        p = np.asarray(points, float)
        return np.asarray([np.linalg.norm(p[b] - p[a]) for a, b in HAND_EDGES])

    def _project(self, points):
        d, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.Kd, self.dd)
        l, _ = cv2.projectPoints(points, self.rvec, self.t, self.Kl, self.dl)
        return d.reshape(-1, 2), l.reshape(-1, 2)

    def refine(self, triangulated, points_d_px, points_l_px, depth_points=None,
               depth_confidence=None, timestamp=None, confidence_d=1.0,
               confidence_l=1.0, sync_confidence=1.0):
        from scipy.optimize import least_squares

        started = time.monotonic()
        x0 = np.asarray(triangulated, float).reshape(21, 3).copy()
        if not np.all(np.isfinite(x0)):
            raise ValueError("Triangulation chua du 21 diem huu han")
        x0[:, :2] = np.clip(x0[:, :2], -2.99, 2.99)
        x0[:, 2] = np.clip(x0[:, 2], 0.151, 5.99)
        depth = (np.full_like(x0, np.nan) if depth_points is None
                 else np.asarray(depth_points, float).reshape(21, 3))
        dconf = (np.zeros(21) if depth_confidence is None
                 else np.clip(np.asarray(depth_confidence, float).reshape(21), 0.0, 1.0))
        valid_depth = np.all(np.isfinite(depth), axis=1) & (dconf > 0.02)

        reference = depth if np.count_nonzero(valid_depth) >= 12 else x0
        initial_bones = self._bone_lengths(reference)
        if self.bone_lengths is None:
            self.bone_lengths = initial_bones

        c = self.cfg
        sigma_px = float(c.get("reprojection_sigma_px", 3.0))
        sigma_depth = float(c.get("depth_sigma_m", 0.018))
        sigma_bone = float(c.get("bone_sigma_m", 0.006))
        wd = np.sqrt(float(c.get("lambda_reprojection", 1.0)) *
                     max(float(confidence_d * sync_confidence), 1e-4)) / sigma_px
        wl = np.sqrt(float(c.get("lambda_reprojection", 1.0)) *
                     max(float(confidence_l * sync_confidence), 1e-4)) / sigma_px
        wdepth = np.sqrt(float(c.get("lambda_depth", 2.0)) * dconf) / sigma_depth
        wbone = np.sqrt(float(c.get("lambda_bone", 0.7))) / sigma_bone
        prev = self.previous
        if prev is not None and timestamp is not None and self.previous_t is not None:
            dt = max(float(timestamp - self.previous_t), 0.0)
            temporal_sigma = (float(c.get("temporal_base_sigma_m", 0.008)) +
                              float(c.get("temporal_speed_m_s", 1.5)) * dt)
            wtemporal = np.sqrt(float(c.get("lambda_temporal", 0.35))) / temporal_sigma
        else:
            wtemporal = 0.0

        pd, pl = np.asarray(points_d_px, float), np.asarray(points_l_px, float)
        raw_rd, raw_rl = self._project(x0)
        raw_err_d = np.linalg.norm(raw_rd - pd, axis=1)
        raw_err_l = np.linalg.norm(raw_rl - pl, axis=1)
        raw_depth_error = (np.linalg.norm(x0[valid_depth] - depth[valid_depth], axis=1)
                           if np.any(valid_depth) else np.array([]))

        def residual(flat):
            X = flat.reshape(21, 3)
            rd, rl = self._project(X)
            out = [wd * (rd - pd).ravel(), wl * (rl - pl).ravel()]
            if np.any(valid_depth):
                out.append(((X[valid_depth] - depth[valid_depth]) *
                            wdepth[valid_depth, None]).ravel())
            lengths = self._bone_lengths(X)
            out.append(wbone * (lengths - self.bone_lengths))
            if prev is not None and wtemporal > 0:
                out.append((wtemporal * (X - prev)).ravel())
            return np.concatenate(out)

        # Mỗi reprojection/depth chỉ phụ thuộc một landmark; mỗi bone chỉ phụ
        # thuộc hai đầu mút. Khai báo sparsity tránh finite-difference Jacobian
        # dày 63xN (chậm hơn một bậc độ lớn trên CPU laptop).
        from scipy.sparse import lil_matrix
        n_depth = int(np.count_nonzero(valid_depth))
        n_temporal = 63 if prev is not None and wtemporal > 0 else 0
        n_rows = 84 + 3 * n_depth + len(HAND_EDGES) + n_temporal
        sparsity = lil_matrix((n_rows, 63), dtype=np.int8)
        row = 0
        for _camera_index in range(2):
            for i in range(21):
                sparsity[row:row + 2, 3*i:3*i + 3] = 1
                row += 2
        for i in np.flatnonzero(valid_depth):
            sparsity[row:row + 3, 3*i:3*i + 3] = 1
            row += 3
        for a, b in HAND_EDGES:
            sparsity[row, 3*a:3*a + 3] = 1
            sparsity[row, 3*b:3*b + 3] = 1
            row += 1
        if n_temporal:
            for i in range(63):
                sparsity[row + i, i] = 1

        lower = np.tile([-3.0, -3.0, 0.15], 21)
        upper = np.tile([3.0, 3.0, 6.0], 21)
        result = least_squares(
            residual, x0.ravel(), bounds=(lower, upper), loss=str(c.get("loss", "huber")),
            f_scale=float(c.get("loss_scale", 1.0)), max_nfev=int(c.get("max_nfev", 12)),
            jac_sparsity=sparsity.tocsr(), tr_solver="lsmr")
        usable = bool(np.all(np.isfinite(result.x)))
        points = result.x.reshape(21, 3) if usable else x0
        rd, rl = self._project(points)
        err_d = np.linalg.norm(rd - pd, axis=1)
        err_l = np.linalg.norm(rl - pl, axis=1)
        depth_error = (np.linalg.norm(points[valid_depth] - depth[valid_depth], axis=1)
                       if np.any(valid_depth) else np.array([]))
        bone_error = np.abs(self._bone_lengths(points) - self.bone_lengths)
        per_point_reproj = np.maximum(err_d, err_l)
        depth_score = np.zeros(21)
        if np.any(valid_depth):
            depth_score[valid_depth] = (dconf[valid_depth] *
                np.exp(-np.linalg.norm(points[valid_depth] - depth[valid_depth], axis=1) /
                       max(float(c.get("depth_confidence_scale_m", 0.04)), 1e-6)))
        view_score = (np.sqrt(max(confidence_d * confidence_l, 0.0)) * sync_confidence *
                      np.exp(-0.5 * (per_point_reproj /
                                     max(float(c.get("reprojection_confidence_px", 8.0)), 1e-6)) ** 2))
        point_confidence = np.clip(1.0 - (1.0 - depth_score) * (1.0 - view_score), 0.0, 1.0)
        rejected = point_confidence < float(c.get("min_point_confidence", 0.20))
        if usable:
            if timestamp is None or self.previous_t is None or timestamp >= self.previous_t:
                self.previous, self.previous_t = points.copy(), timestamp
            alpha = float(c.get("bone_adapt_alpha", 0.01))
            if np.all(np.isfinite(initial_bones)):
                self.bone_lengths = (1.0 - alpha) * self.bone_lengths + alpha * initial_bones
        reproj = np.r_[err_d, err_l]
        return {
            "points_d435i": points,
            "reprojection_d435i_px": rd, "reprojection_laptop_px": rl,
            "error_d435i_px": err_d, "error_laptop_px": err_l,
            "mode": "FULL_RGBD" if np.any(valid_depth) else "RGB_ONLY",
            "success": usable, "converged": bool(result.success),
            "cost": float(result.cost), "nfev": int(result.nfev),
            "solve_ms": 1000.0 * (time.monotonic() - started),
            "reprojection_median_px": float(np.median(reproj)),
            "reprojection_p95_px": float(np.percentile(reproj, 95)),
            "depth_median_m": float(np.median(depth_error)) if depth_error.size else float("nan"),
            "bone_median_m": float(np.median(bone_error)),
            "depth_points": int(np.count_nonzero(valid_depth)),
            "point_confidence": point_confidence,
            "confidence_median": float(np.median(point_confidence)),
            "rejected_points": int(np.count_nonzero(rejected)),
            "raw_reprojection_p95_px": float(np.percentile(np.r_[raw_err_d, raw_err_l], 95)),
            "raw_depth_median_m": (float(np.median(raw_depth_error)) if raw_depth_error.size
                                   else float("nan")),
        }


class OrientationFusion:
    """Chọn giữa hai giả thuyết palm-normal và fusion có hysteresis trên SO(3)."""

    FLIP = np.diag([1.0, -1.0, -1.0])

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.previous = None
        self.branch = 0
        self.pending_branch = None
        self.pending_count = 0
        self.bad_count = 0

    @staticmethod
    def _deg(a, b):
        return np.rad2deg(rotation_distance(a, b))

    def update(self, fused_R, depth_R=None, laptop_R=None, fused_confidence=1.0,
               depth_confidence=0.0, laptop_confidence=0.0):
        c = self.cfg
        observations = []
        if depth_R is not None and depth_confidence > 0:
            observations.append((np.asarray(depth_R, float), float(depth_confidence), "depth"))
        if laptop_R is not None and laptop_confidence > 0:
            observations.append((np.asarray(laptop_R, float), float(laptop_confidence), "laptop"))
        if fused_R is None or fused_confidence < float(c.get("min_fused_confidence", 0.2)):
            self.bad_count += 1
            hold = int(c.get("hold_frames", 8))
            state = "HOLD" if self.previous is not None and self.bad_count <= hold else "LOST"
            if state == "LOST": self.previous = None
            return {"R": self.previous, "state": state, "confidence": 0.0,
                    "branch": self.branch, "sources": []}

        base = np.asarray(fused_R, float)
        candidates = (base, base @ self.FLIP)
        temporal_weight = float(c.get("temporal_weight", 1.5)) if self.previous is not None else 0.0
        scores = []
        for candidate in candidates:
            score = temporal_weight * (self._deg(self.previous, candidate) / 90.0) ** 2 if self.previous is not None else 0.0
            for obs, weight, _ in observations:
                score += weight * (self._deg(obs, candidate) / 90.0) ** 2
            scores.append(score)
        proposed = int(np.argmin(scores))
        # Index 0/1 là tương đối với raw frame và có thể hoán đổi khi detector
        # flip, nên phải đổi index ngay nếu orientation vật lý vẫn liên tục.
        # Hysteresis chỉ áp dụng cho một bước nhảy vật lý lớn của candidate tốt nhất.
        proposed_jump = self._deg(self.previous, candidates[proposed]) if self.previous is not None else 0.0
        if self.previous is not None and proposed_jump > float(c.get("physical_switch_deg", 100)):
            if self.pending_branch == proposed: self.pending_count += 1
            else: self.pending_branch, self.pending_count = proposed, 1
            if self.pending_count >= int(c.get("switch_frames", 3)):
                self.branch = proposed
                self.pending_branch, self.pending_count = None, 0
        else:
            self.branch = proposed
            self.pending_branch, self.pending_count = None, 0

        chosen = candidates[self.branch]
        max_agree = float(c.get("max_source_disagreement_deg", 70))
        accepted = [(chosen, max(float(fused_confidence), 0.05), "landmarks")]
        for obs, weight, name in observations:
            if self._deg(chosen, obs) <= max_agree:
                accepted.append((obs, weight, name))
        # Sequential weighted-SLERP avoids quaternion sign ambiguity and không
        # trung bình một observation bất đồng lớn.
        fused = accepted[0][0]
        total = accepted[0][1]
        for obs, weight, _ in accepted[1:]:
            fused = slerp_rotation(fused, obs, weight / max(total + weight, 1e-9))
            total += weight
        if self.previous is not None:
            fused = slerp_rotation(self.previous, fused, float(c.get("output_smoothing", 0.65)))
        agreement = [self._deg(fused, obs) for obs, _, _ in accepted]
        confidence = float(np.clip((total / max(1.0 + len(observations), 1.0)) *
                                   np.exp(-np.mean(agreement) / 90.0), 0.0, 1.0))
        self.previous = fused
        self.bad_count = 0
        state = "TRACKING" if len(accepted) >= 2 and confidence >= float(c.get("tracking_confidence", .35)) else "DEGRADED"
        return {"R": fused, "state": state, "confidence": confidence,
                "branch": self.branch, "sources": [x[2] for x in accepted],
                "candidate_scores": scores}
