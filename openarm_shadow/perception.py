"""MediaPipe Pose + Hand (Tasks API, chạy CPU) -> quan sát tay người trong khung thân.

Đầu ra cho mỗi tay người ("right" / "left" = tay phải/trái THẬT của người):
    s, e, w : vai, khuỷu, cổ tay (m) trong khung thân (x trước, y trái, z lên, gốc giữa hai vai)
    H       : 3x3 hướng bàn tay trong khung thân (cột x = hướng ngón, y = út→trỏ, z = x×y) hoặc None
    grip    : độ mở kẹp 0..1 (khoảng cách đầu ngón cái–trỏ / chiều dài bàn tay) hoặc None
    conf    : độ tin cậy cho 3 nhóm khớp: upper (J1–J2), fore (J3–J4), hand (J5–J7 + kẹp)

Lưu ý: chạy MediaPipe trên ảnh GỐC (không lật gương). Lật gương chỉ để hiển thị.
Giả định: điểm 3D "world" của Pose và Hand đều có hướng trục theo camera (gốc khác nhau),
nên hướng bàn tay có thể đổi sang khung thân bằng cùng ma trận quay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import make_frame, unit

# chỉ số MediaPipe Pose
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
ARM_IDX = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}
# chỉ số MediaPipe Hand
H_WRIST, H_THUMB_TIP, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 4, 5, 8, 9, 17
FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


@dataclass
class ArmObs:
    s: np.ndarray | None = None
    e: np.ndarray | None = None
    w: np.ndarray | None = None
    H: np.ndarray | None = None
    grip: float | None = None
    conf: dict = field(default_factory=lambda: {"upper": 0.0, "fore": 0.0, "hand": 0.0})
    hand_points_cam: np.ndarray | None = None  # (21,3), metric, cùng camera frame RealSense; NaN nếu thiếu
    hand_point_confidence: np.ndarray | None = None
    hand_depth_valid: int = 0
    hand_depth_confidence: float = 0.0
    hand_depth_mode: str = "NONE"
    hand_R_cam: np.ndarray | None = None
    hand_center_cam: np.ndarray | None = None
    hand_axes_px: np.ndarray | None = None  # origin,x,y,z endpoints trong pixel RealSense RGB
    hand_open_fingers: int = 0
    hand_orientation_mode: str = "NONE"

    @property
    def u(self):
        return None if self.s is None else unit(self.e - self.s)

    @property
    def l(self):
        return None if self.e is None else unit(self.w - self.e)


@dataclass
class HandDetection2D:
    landmarks: np.ndarray
    side: str | None
    handedness: str | None
    handedness_score: float
    association_score: float
    border_score: float
    palm_area: float


@dataclass
class Frame:
    arms: dict                     # "right"/"left" -> ArmObs
    pose_2d: np.ndarray | None     # (33, 3) x, y chuẩn hoá + visibility, để vẽ
    hands_2d: list                 # [(21, 2) array, side hoặc None]
    body_R: np.ndarray | None      # cột = trục khung thân trong toạ độ camera
    t: float = 0.0
    depth_used: dict = field(default_factory=dict)  # số landmark depth hợp lệ theo tay (0..3)
    hand_depth: dict = field(default_factory=dict)  # diagnostics 21 điểm bàn tay theo tay người
    body_origin: np.ndarray | None = None
    hand_observations: list = field(default_factory=list)
    tracking: dict = field(default_factory=dict)


def _intrinsic(intr, name):
    return float(intr[name] if isinstance(intr, dict) else getattr(intr, name))


def sample_depth_with_confidence(depth_m, u, v, radius=3, min_m=0.3, max_m=6.0,
                                 max_delta_m=0.15):
    """Trả về (depth median, confidence) của cụm depth gần pixel landmark."""
    if depth_m is None or depth_m.ndim != 2 or not np.isfinite(u + v):
        return None, 0.0
    h, w = depth_m.shape
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None, 0.0
    r = max(0, int(radius))
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    patch = np.asarray(depth_m[y0:y1, x0:x1], float)
    valid = np.isfinite(patch) & (patch >= min_m) & (patch <= max_m)
    if not valid.any():
        return None, 0.0
    center = float(depth_m[y, x])
    center_valid = np.isfinite(center) and min_m <= center <= max_m
    if not center_valid:
        yy, xx = np.nonzero(valid)
        nearest = np.argmin((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2)
        center = float(patch[yy[nearest], xx[nearest]])
    cluster = patch[valid & (np.abs(patch - center) <= max_delta_m)]
    if not cluster.size:
        return center, 0.15
    z = float(np.median(cluster))
    mad = float(np.median(np.abs(cluster - z)))
    support = cluster.size / max(patch.size, 1)
    confidence = np.sqrt(support) * np.exp(-mad / 0.02) * (1.0 if center_valid else 0.75)
    return z, float(np.clip(confidence, 0.0, 1.0))


def sample_depth(depth_m, u, v, radius=3, min_m=0.3, max_m=6.0, max_delta_m=0.15):
    """API cũ: chỉ trả depth để các caller body-pose và test hiện tại không đổi."""
    return sample_depth_with_confidence(depth_m, u, v, radius, min_m, max_m, max_delta_m)[0]


def deproject_pixel(intr, u, v, depth_m):
    """Pixel RGB + depth (m) -> điểm [x phải, y xuống, z trước] trong camera."""
    if not isinstance(intr, dict):
        try:
            import pyrealsense2 as rs
            return np.asarray(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m)),
                              dtype=float)
        except (ImportError, RuntimeError, TypeError, ValueError):
            # Fallback pinhole giúp test/offline intrinsics vẫn dùng được; RealSense thật đi qua SDK ở trên.
            pass
    fx, fy = _intrinsic(intr, "fx"), _intrinsic(intr, "fy")
    ppx, ppy = _intrinsic(intr, "ppx"), _intrinsic(intr, "ppy")
    z = float(depth_m)
    return np.array([(u - ppx) * z / fx, (v - ppy) * z / fy, z])


def fit_metric_depth(relative_z, measured_z, confidence=None, previous=None, min_points=4):
    """Fit z_metric = a*z_mediapipe + b; trả (a,b) hoặc previous khi dữ liệu chưa đủ.

    MediaPipe cho hình dạng depth tương đối. RealSense cung cấp scale/offset metric. Fit robust này cho phép
    khôi phục landmark nằm trên lỗ stereo mà vẫn giữ mọi điểm trong cùng camera frame metric.
    """
    rz = np.asarray(relative_z, float)
    mz = np.asarray(measured_z, float)
    cw = np.ones_like(rz) if confidence is None else np.asarray(confidence, float)
    valid = np.isfinite(rz) & np.isfinite(mz) & (cw > 0.05)
    model = None
    if np.count_nonzero(valid) >= int(min_points) and np.ptp(rz[valid]) > 1e-3:
        X = np.column_stack([rz[valid], np.ones(np.count_nonzero(valid))])
        y, w = mz[valid], np.sqrt(cw[valid])
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        residual = y - X @ beta
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        keep = np.abs(residual - med) <= max(0.015, 3.0 * mad)
        if np.count_nonzero(keep) >= int(min_points):
            beta = np.linalg.lstsq(X[keep] * w[keep, None], y[keep] * w[keep], rcond=None)[0]
        a, b = map(float, beta)
        if 0.03 <= a <= 3.0 and np.isfinite(a + b):
            model = (a, b)
    # GMH-D style anchor: z của wrist (index 0) đặt scale metric ban đầu khi chưa fit đủ điểm.
    if model is None and len(mz) and np.isfinite(mz[0]):
        a = float(np.clip(mz[0], 0.03, 3.0))
        model = (a, float(mz[0] - a * rz[0]))
    if model is None:
        return previous
    if previous is not None and np.all(np.isfinite(previous)):
        # Model đổi chậm hơn từng pixel depth, tránh toàn bộ bàn tay co/giãn khi số điểm hợp lệ thay đổi.
        model = tuple(0.7 * float(p) + 0.3 * float(n) for p, n in zip(previous, model))
    return model


def fuse_hand_landmarks(hand_landmarks, depth_m, intr, cfg, previous_model=None):
    """MediaPipe 2D + RealSense depth -> 21 điểm metric cùng camera frame và diagnostics."""
    h, w = depth_m.shape
    uv = np.array([[p.x * w, p.y * h] for p in hand_landmarks], dtype=float)
    relative_z = np.array([p.z for p in hand_landmarks], dtype=float)
    measured = np.full(len(hand_landmarks), np.nan)
    direct_conf = np.zeros(len(hand_landmarks), dtype=float)
    radius = int(cfg.get("hand_patch_radius", 2))
    for i, (u, v) in enumerate(uv):
        measured[i], direct_conf[i] = sample_depth_with_confidence(
            depth_m, u, v, radius=radius, min_m=cfg.get("min_depth_m", 0.3),
            max_m=cfg.get("max_depth_m", 6.0), max_delta_m=cfg.get("max_local_delta_m", 0.15))
    min_points = int(cfg.get("hand_min_direct_points", 4))
    model = fit_metric_depth(relative_z, measured, direct_conf, previous_model, min_points)
    predicted = np.full_like(measured, np.nan)
    if model is not None:
        predicted = model[0] * relative_z + model[1]
    fused_z = predicted.copy()
    direct = np.isfinite(measured)
    only_direct = direct & ~np.isfinite(fused_z)
    fused_z[only_direct] = measured[only_direct]
    both = direct & np.isfinite(predicted)
    fused_z[both] = direct_conf[both] * measured[both] + (1.0 - direct_conf[both]) * predicted[both]
    points = np.full((len(hand_landmarks), 3), np.nan)
    for i, z in enumerate(fused_z):
        if np.isfinite(z) and cfg.get("min_depth_m", 0.3) <= z <= cfg.get("max_depth_m", 6.0):
            points[i] = deproject_pixel(intr, uv[i, 0], uv[i, 1], z)
    n_direct = int(np.count_nonzero(direct))
    n_fused = int(np.count_nonzero(np.all(np.isfinite(points), axis=1)))
    mean_conf = float(np.mean(direct_conf[direct])) if n_direct else 0.0
    model_conf = 0.35 * mean_conf * min(1.0, n_direct / max(min_points, 1)) if model is not None else 0.0
    point_conf = np.where(direct, np.maximum(direct_conf, model_conf), model_conf)
    point_conf[~np.all(np.isfinite(points), axis=1)] = 0.0
    mode = "DEPTH" if n_direct == len(hand_landmarks) else ("FUSED" if n_fused else "NONE")
    info = {"direct": n_direct, "fused": n_fused, "confidence": mean_conf, "mode": mode,
            "model": model, "point_confidence": point_conf}
    return points, info


def project_point(intr, point):
    """Điểm camera-frame -> pixel RGB, ưu tiên projection có distortion của librealsense."""
    p = np.asarray(point, float)
    if not np.all(np.isfinite(p)) or p[2] <= 0:
        return None
    if not isinstance(intr, dict):
        try:
            import pyrealsense2 as rs
            return np.asarray(rs.rs2_project_point_to_pixel(intr, p.tolist()), dtype=float)
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass
    fx, fy = _intrinsic(intr, "fx"), _intrinsic(intr, "fy")
    ppx, ppy = _intrinsic(intr, "ppx"), _intrinsic(intr, "ppy")
    return np.array([p[0] * fx / p[2] + ppx, p[1] * fy / p[2] + ppy])


def project_rgb_hand_axes(hand_landmarks, hand_R_cam, width, height):
    """Chiếu khung bàn tay lên RGB bằng mô hình weak-perspective.

    Không cần depth hay camera intrinsics: hướng x/y lấy từ khung camera của
    MediaPipe, còn độ dài trục tự co giãn theo kích thước bàn tay trên ảnh.
    """
    uv = np.array([[p.x * width, p.y * height] for p in hand_landmarks], dtype=float)
    if uv.shape != (21, 2) or not np.all(np.isfinite(uv)):
        return None
    R = np.asarray(hand_R_cam, dtype=float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return None
    palm_ids = (H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP)
    origin = np.mean(uv[list(palm_ids)], axis=0)
    palm_px = max(np.linalg.norm(uv[H_MIDDLE_MCP] - uv[H_WRIST]),
                  np.linalg.norm(uv[H_INDEX_MCP] - uv[H_PINKY_MCP]))
    axis_px = float(np.clip(1.25 * palm_px, 35.0, 110.0))
    endpoints = origin[None, :] + axis_px * R[:2, :].T
    return np.vstack((origin, endpoints))


def open_finger_count(points):
    """Đếm bốn ngón chính đang duỗi từ hình học 3D; không dùng thumb vì biến thiên lớn."""
    p = np.asarray(points, float)
    if p.shape != (21, 3):
        return 0
    count = 0
    for mcp, pip, dip, tip in FINGER_CHAINS:
        ids = (H_WRIST, mcp, pip, dip, tip)
        if not np.all(np.isfinite(p[list(ids)])):
            continue
        chain = (np.linalg.norm(p[pip] - p[mcp]) + np.linalg.norm(p[dip] - p[pip]) +
                 np.linalg.norm(p[tip] - p[dip]))
        straight = np.linalg.norm(p[tip] - p[mcp]) / max(chain, 1e-6)
        reach = np.linalg.norm(p[tip] - p[H_WRIST]) / max(np.linalg.norm(p[mcp] - p[H_WRIST]), 1e-6)
        count += int(straight > 0.78 and reach > 1.30)
    return count


def fit_palm_plane(depth_m, intr, hand_landmarks, cfg):
    """Fit plane robust trong polygon lòng bàn tay; trả normal, center và diagnostics."""
    h, w = depth_m.shape
    uv = np.array([[p.x * w, p.y * h] for p in hand_landmarks], dtype=float)
    palm_ids = [H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP]
    poly = cv2.convexHull(np.round(uv[palm_ids]).astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    area = int(mask.sum())
    if area < 20:
        return None, None, {"inliers": 0, "rms_m": float("inf")}
    erode_px = max(1, int(np.sqrt(area) * 0.08))
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    mask = cv2.erode(mask, kernel)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None, None, {"inliers": 0, "rms_m": float("inf")}
    z = np.asarray(depth_m[ys, xs], float)
    valid = np.isfinite(z) & (z >= cfg.get("min_depth_m", 0.3)) & (z <= cfg.get("max_depth_m", 6.0))
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if not len(z):
        return None, None, {"inliers": 0, "rms_m": float("inf")}
    med = float(np.median(z))
    band = float(cfg.get("palm_depth_band_m", 0.06))
    keep = np.abs(z - med) <= band
    xs, ys, z = xs[keep], ys[keep], z[keep]
    # Giới hạn chi phí deproject nhưng vẫn phủ đều ROI.
    if len(z) > 400:
        take = np.linspace(0, len(z) - 1, 400).astype(int)
        xs, ys, z = xs[take], ys[take], z[take]
    min_points = int(cfg.get("palm_min_points", 35))
    if len(z) < min_points:
        return None, None, {"inliers": int(len(z)), "rms_m": float("inf")}
    pts = np.array([deproject_pixel(intr, x, y, zz) for x, y, zz in zip(xs, ys, z)])
    center = np.mean(pts, axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    normal = unit(vt[-1])
    residual = np.abs((pts - center) @ normal)
    robust = residual <= max(0.008, 3.0 * float(np.median(residual)))
    if np.count_nonzero(robust) >= min_points:
        pts = pts[robust]
        center = np.mean(pts, axis=0)
        _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
        normal = unit(vt[-1])
    rms = float(np.sqrt(np.mean(((pts - center) @ normal) ** 2)))
    if rms > float(cfg.get("palm_plane_max_rms_m", 0.012)):
        return None, center, {"inliers": int(len(pts)), "rms_m": rms}
    return normal, center, {"inliers": int(len(pts)), "rms_m": rms}


def palm_frame_from_depth(points, plane_normal=None, plane_quality=0.0, side=None):
    """Frame camera của tay: x hướng ngón, y út->trỏ, z=x×y; plane chỉ tinh chỉnh pháp tuyến."""
    p = np.asarray(points, float)
    ids = [H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP]
    if p.shape != (21, 3) or not np.all(np.isfinite(p[ids])):
        return None, None
    center = np.mean(p[ids], axis=0)
    x = unit(0.5 * (p[H_MIDDLE_MCP] + p[13]) - p[H_WRIST])
    across = p[H_INDEX_MCP] - p[H_PINKY_MCP]
    y_land = unit(across - (across @ x) * x)
    z_land = unit(np.cross(x, y_land))
    # Hai bàn tay có chirality đối nhau. Bù dấu để z đều hướng ra khỏi lòng bàn tay.
    if side == "right":
        z_land = -z_land
    z = z_land
    if plane_normal is not None and np.linalg.norm(plane_normal) > 0.5:
        pn = unit(plane_normal)
        if pn @ z_land < 0:
            pn = -pn
        weight = float(np.clip(plane_quality, 0.0, 0.85))
        z = unit((1.0 - weight) * z_land + weight * pn)
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    R = np.column_stack([x, y, z])
    return R, center


def matrix_to_quaternion(R):
    """Rotation matrix -> quaternion [w,x,y,z]."""
    R = np.asarray(R, float)
    q = np.empty(4)
    tr = float(np.trace(R))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[:] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[:] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                    (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[:] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q)


def quaternion_to_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def slerp_rotation(R0, R1, fraction):
    q0, q1 = matrix_to_quaternion(R0), matrix_to_quaternion(R1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1.0, 1.0)
    a = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return quaternion_to_matrix(q / np.linalg.norm(q))
    theta = np.arccos(dot)
    q = (np.sin((1.0 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
    return quaternion_to_matrix(q)


def rotation_distance(R0, R1):
    d = np.asarray(R0).T @ np.asarray(R1)
    return float(np.arccos(np.clip((np.trace(d) - 1.0) * 0.5, -1.0, 1.0)))


def hand_frame(hw, side=None):
    """Khung bàn tay từ MediaPipe world landmarks, cùng quy ước chirality với depth."""
    x = unit(hw[H_MIDDLE_MCP] - hw[H_WRIST])
    across = hw[H_INDEX_MCP] - hw[H_PINKY_MCP]
    y_land = unit(across - (across @ x) * x)
    z = unit(np.cross(x, y_land))
    if side == "right":
        z = -z
    # Dựng lại cả y và x để frame luôn trực chuẩn, tay phải và tay trái đều det=+1.
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    return np.column_stack([x, y, z])


def body_frame(W, vis, min_hip_vis=0.5):
    """Khung thân từ điểm world của Pose. Thiếu hông (ngồi, bị bàn che) thì dùng 'lên' của camera."""
    if min(vis[L_HIP], vis[R_HIP]) >= min_hip_vis:
        bottom = 0.5 * (W[L_HIP] + W[R_HIP])
    else:
        bottom = 0.5 * (W[L_SH] + W[R_SH]) + np.array([0.0, 0.5, 0.0])   # y của MediaPipe hướng xuống
    return make_frame(W[L_SH], W[R_SH], bottom)


class Perception:
    def __init__(self, pose_model, hand_model, num_hands=2, min_conf=0.5, depth_cfg=None,
                 orientation_cfg=None, tracking_cfg=None, pose_enabled=True,
                 force_hand_side=None):
        from pathlib import Path
        needed_models = (hand_model,) if not pose_enabled else (pose_model, hand_model)
        missing = [str(m) for m in needed_models if not Path(m).is_file()]
        if missing:
            raise SystemExit("Chưa có model MediaPipe: " + ", ".join(missing) +
                             "\nChạy: bash scripts/download_models.sh")
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        RM = vision.RunningMode.VIDEO
        self.pose_enabled = bool(pose_enabled)
        self.force_hand_side = force_hand_side
        self.pose = None
        if self.pose_enabled:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(pose_model)), running_mode=RM,
                num_poses=1, min_pose_detection_confidence=min_conf,
                min_pose_presence_confidence=min_conf, min_tracking_confidence=min_conf))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(hand_model)), running_mode=RM,
            num_hands=num_hands, min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf))
        self._last_ts = -1
        self.depth_cfg = depth_cfg or {}
        self.orientation_cfg = orientation_cfg or {}
        self.tracking_cfg = tracking_cfg or {}
        self.orientation_source = str(self.orientation_cfg.get("source", "depth_required")).lower()
        if self.orientation_source not in {"depth_required", "rgb_world"}:
            raise ValueError(
                "orientation.source phải là 'depth_required' hoặc 'rgb_world', "
                f"không phải {self.orientation_source!r}"
            )
        self._hand_depth_model = {"right": None, "left": None}
        self._hand_depth_misses = {"right": 0, "left": 0}
        self._hand_orientation = {"right": None, "left": None}
        self._orientation_tracking = {"right": False, "left": False}
        self._orientation_good = {"right": 0, "left": 0}
        self._orientation_bad = {"right": 0, "left": 0}
        self._body_R = None
        self._last_pose = None
        self._pose_misses = 0
        self._pose_frame_count = 0

    def close(self):
        if self.pose is not None:
            self.pose.close()
        self.hands.close()

    def _process_hand_only(self, hres, bgr, t, depth_m, depth_intrinsics):
        """D435 phụ: chỉ track một tay phải + depth, không phụ thuộc Pose toàn thân."""
        h, w = bgr.shape[:2]
        arms = {"right": ArmObs(), "left": ArmObs()}
        raw = list(hres.hand_landmarks or [])
        if not raw:
            for side in arms:
                self._stabilize_orientation(side, None, "NONE")
            return Frame(arms, None, [], None, t, tracking={
                "pose": "DISABLED", "raw_hands": 0, "associated_hands": 0})

        # Chế độ này được dùng cho MVP một tay phải và HandLandmarker chỉ xuất
        # một candidate. Không để Pose D435 chập chờn quyết định có giữ tay hay không.
        hi = 0
        side = self.force_hand_side or "right"
        hl = raw[hi]
        h2 = np.array([[p.x, p.y] for p in hl])
        hw = np.array([[p.x, p.y, p.z] for p in hres.hand_world_landmarks[hi]])
        category = hres.handedness[hi][0] if hres.handedness and hi < len(hres.handedness) else None
        handedness = getattr(category, "category_name", None) if category is not None else None
        hand_score = float(getattr(category, "score", 1.0)) if category is not None else 1.0
        edge = float(np.min(np.r_[h2[:, 0], h2[:, 1], 1.0 - h2[:, 0], 1.0 - h2[:, 1]]))
        border_score = float(np.clip(edge / 0.08, 0.0, 1.0))
        palm = (h2[[H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP]] * [w, h]).astype(np.float32)
        palm_area = float(abs(cv2.contourArea(cv2.convexHull(palm))) / max(w * h, 1))
        detection = HandDetection2D(h2, side, handedness, hand_score, 1.0,
                                    border_score, palm_area)
        ob = arms[side]
        ob.grip = float(np.linalg.norm(hw[H_THUMB_TIP] - hw[H_INDEX_TIP]) /
                        max(np.linalg.norm(hw[H_MIDDLE_MCP] - hw[H_WRIST]), 1e-6))
        ob.conf["hand"] = hand_score

        if self.orientation_source == "depth_required" and depth_m is not None and depth_intrinsics is not None:
            hold_frames = int(self.depth_cfg.get("hand_model_hold_frames", 6))
            previous_model = (self._hand_depth_model[side]
                              if self._hand_depth_misses[side] < hold_frames else None)
            pts, dinfo = fuse_hand_landmarks(hl, depth_m, depth_intrinsics,
                                             self.depth_cfg, previous_model)
            if dinfo["direct"]:
                self._hand_depth_misses[side] = 0
            else:
                self._hand_depth_misses[side] += 1
            if dinfo["model"] is not None and dinfo["direct"]:
                self._hand_depth_model[side] = dinfo["model"]
            if self._hand_depth_misses[side] >= hold_frames:
                self._hand_depth_model[side] = None
            ob.hand_points_cam = pts
            ob.hand_point_confidence = dinfo["point_confidence"]
            ob.hand_depth_valid = dinfo["direct"]
            ob.hand_depth_confidence = dinfo["confidence"]
            ob.hand_depth_mode = dinfo["mode"]
            open_count = open_finger_count(pts)
            ob.hand_open_fingers = open_count
            raw_R, center = palm_frame_from_depth(pts, side=side)
            plane_info = {"inliers": 0, "rms_m": float("inf")}
            source = "LANDMARK"
            if open_count >= int(self.orientation_cfg.get("min_open_fingers", 3)):
                plane_n, plane_center, plane_info = fit_palm_plane(
                    depth_m, depth_intrinsics, hl, self.depth_cfg)
                max_rms = float(self.depth_cfg.get("palm_plane_max_rms_m", 0.012))
                plane_quality = 0.0
                if plane_n is not None:
                    support = min(1.0, plane_info["inliers"] /
                                  max(3 * int(self.depth_cfg.get("palm_min_points", 35)), 1))
                    plane_quality = support * np.exp(-plane_info["rms_m"] / max(max_rms, 1e-6))
                    source = "PLANE"
                raw_R, center = palm_frame_from_depth(pts, plane_n, plane_quality, side=side)
                if plane_center is not None:
                    center = plane_center
            stable_R, orientation_mode = self._stabilize_orientation(side, raw_R, source)
            ob.hand_orientation_mode = orientation_mode
            if stable_R is not None:
                ob.H = stable_R
                ob.hand_R_cam, ob.hand_center_cam = stable_R, center
                if center is not None:
                    axis_len = float(self.depth_cfg.get("palm_axis_length_m", 0.08))
                    axis_points = [center] + [center + stable_R[:, j] * axis_len for j in range(3)]
                    axes_px = [project_point(depth_intrinsics, p) for p in axis_points]
                    if all(p is not None for p in axes_px):
                        ob.hand_axes_px = np.asarray(axes_px)
            dinfo.update({"open_fingers": open_count, "orientation": orientation_mode,
                          "plane_inliers": plane_info["inliers"],
                          "plane_rms_m": plane_info["rms_m"]})
            hand_depth = {side: {k: v for k, v in dinfo.items()
                                 if k not in ("model", "point_confidence")}}
        else:
            raw_R = hand_frame(hw, side=side)
            stable_R, orientation_mode = self._stabilize_orientation(side, raw_R, "RGB_WORLD")
            ob.hand_orientation_mode = orientation_mode
            if stable_R is not None:
                ob.H = stable_R
                ob.hand_R_cam = stable_R
                ob.hand_axes_px = project_rgb_hand_axes(hl, stable_R, w, h)
            hand_depth = {side: {"mode": "RGB_ONLY", "confidence": hand_score,
                                 "orientation": orientation_mode}}
        return Frame(arms, None, [(h2, side)], None, t, {side: 0}, hand_depth,
                     None, [detection], tracking={
                         "pose": "DISABLED", "raw_hands": 1, "associated_hands": 1})

    def _hold_orientation(self, side):
        self._orientation_tracking[side] = False
        self._orientation_good[side] = 0
        self._orientation_bad[side] += 1
        if (self._hand_orientation[side] is not None and
                self._orientation_bad[side] <= int(self.orientation_cfg.get("hold_frames", 6))):
            return self._hand_orientation[side], "HOLD"
        self._hand_orientation[side] = None
        return None, "NONE"

    def _stabilize_orientation(self, side, raw_R, source):
        if raw_R is None:
            return self._hold_orientation(side)
        prev = self._hand_orientation[side]
        if prev is not None:
            jump = np.rad2deg(rotation_distance(prev, raw_R))
            if jump > float(self.orientation_cfg.get("max_jump_deg", 70)):
                return self._hold_orientation(side)
        if not self._orientation_tracking[side]:
            self._orientation_good[side] += 1
            if self._orientation_good[side] < int(self.orientation_cfg.get("reacquire_frames", 2)):
                if prev is not None:
                    return prev, "HOLD"
                return None, "WAIT"
            self._orientation_tracking[side] = True
        self._orientation_bad[side] = 0
        self._orientation_good[side] = 0
        alpha = float(self.orientation_cfg.get("smoothing", 0.55))
        if prev is not None:
            fast_at = float(self.orientation_cfg.get("fast_angle_deg", 75))
            fast_alpha = float(self.orientation_cfg.get("fast_smoothing", 0.90))
            alpha += (fast_alpha - alpha) * min(jump / max(fast_at, 1e-3), 1.0)
        stable = raw_R if prev is None else slerp_rotation(prev, raw_R, alpha)
        self._hand_orientation[side] = stable
        return stable, source

    def process(self, bgr, t=None, depth_m=None, depth_intrinsics=None) -> Frame:
        """bgr: ảnh OpenCV gốc (không lật). t: thời điểm (s), mặc định time.monotonic()."""
        t = time.monotonic() if t is None else t
        ts = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._pose_frame_count += 1
        pose_interval = max(1, int(self.tracking_cfg.get("pose_interval", 1)))
        run_pose = (self.pose is not None and
                    (self._last_pose is None or self._pose_frame_count % pose_interval == 0))
        pres = self.pose.detect_for_video(img, ts) if run_pose else None
        hres = self.hands.detect_for_video(img, ts)
        h, w = bgr.shape[:2]

        if not self.pose_enabled:
            return self._process_hand_only(hres, bgr, t, depth_m, depth_intrinsics)

        arms = {"right": ArmObs(), "left": ArmObs()}
        pose_held = False
        if pres is not None and pres.pose_landmarks:
            P = pres.pose_landmarks[0]
            pose_2d = np.array([[p.x, p.y, p.visibility or 0.0] for p in P])
            W = np.array([[p.x, p.y, p.z] for p in pres.pose_world_landmarks[0]])
            self._last_pose = (pose_2d.copy(), W.copy())
            self._pose_misses = 0
        else:
            if run_pose:
                self._pose_misses += 1
            hold = int(self.tracking_cfg.get("pose_hold_frames", 6))
            if self._last_pose is None or self._pose_misses > hold:
                for side in arms:
                    self._stabilize_orientation(side, None, "NONE")
                return Frame(arms, None, [], None, t, tracking={
                    "pose": "LOST", "raw_hands": len(hres.hand_landmarks or []),
                    "associated_hands": 0,
                })
            pose_2d, W = (x.copy() for x in self._last_pose)
            pose_held = True
        vis = pose_2d[:, 2]
        R_body, origin = body_frame(W, vis)

        # RealSense: lấy 3D metric tại landmark RGB. Chỉ dùng cho một tay khi đủ vai/khuỷu/cổ tay;
        # nếu một điểm depth mất thì giữ trọn bộ MediaPipe để không trộn hai hệ tọa độ.
        depth_points = {}
        R_depth = origin_depth = None
        if depth_m is not None and depth_intrinsics is not None:
            dc = self.depth_cfg
            ids = {L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP}
            for idx in ids:
                z = sample_depth(
                    depth_m, pose_2d[idx, 0] * w, pose_2d[idx, 1] * h,
                    radius=dc.get("patch_radius", 3), min_m=dc.get("min_depth_m", 0.3),
                    max_m=dc.get("max_depth_m", 6.0), max_delta_m=dc.get("max_local_delta_m", 0.15),
                )
                if z is not None:
                    depth_points[idx] = deproject_pixel(
                        depth_intrinsics, pose_2d[idx, 0] * w, pose_2d[idx, 1] * h, z)
            if L_SH in depth_points and R_SH in depth_points:
                D = W.copy()
                D[L_SH], D[R_SH] = depth_points[L_SH], depth_points[R_SH]
                dvis = vis.copy()
                if L_HIP in depth_points and R_HIP in depth_points:
                    D[L_HIP], D[R_HIP] = depth_points[L_HIP], depth_points[R_HIP]
                else:
                    dvis[L_HIP] = dvis[R_HIP] = 0.0
                R_depth, origin_depth = body_frame(D, dvis)

        # Ghép one-to-one bàn tay với cổ tay gần nhất trong ảnh (không tin nhãn
        # handedness). Bản cũ duyệt tuần tự nên candidate đầu tiên có thể chiếm
        # sai một side và làm candidate đúng phía sau bị loại.
        sh_px = np.linalg.norm((pose_2d[L_SH, :2] - pose_2d[R_SH, :2]) * [w, h])
        raw_hands = [np.array([[p.x, p.y] for p in hl])
                     for hl in (hres.hand_landmarks or [])]
        match_limit = float(self.tracking_cfg.get(
            "hand_association_radius_shoulders", 0.50)) * max(sh_px, 1.0)
        pairs = []
        for i, h2 in enumerate(raw_hands):
            for side in ("right", "left"):
                wr = pose_2d[ARM_IDX[side][2], :2]
                d = float(np.linalg.norm((h2[H_WRIST] - wr) * [w, h]))
                if d < match_limit:
                    pairs.append((d, i, side))
        hand_of, side_of, used_hands = {}, {}, set()
        if self.force_hand_side is not None and len(raw_hands) == 1:
            # MVP một người/một tay phải: Pose vẫn dùng cho vai-khuỷu-cổ tay,
            # nhưng không được phép xoá HandLandmarker chỉ vì pose wrist lệch.
            hand_of[self.force_hand_side] = 0
            side_of[0] = (self.force_hand_side, 0.0)
            used_hands.add(0)
        else:
            for distance, i, side in sorted(pairs):
                if i in used_hands or side in hand_of:
                    continue
                hand_of[side] = i
                side_of[i] = (side, distance)
                used_hands.add(i)

        hands_2d, hand_observations = [], []
        for i, h2 in enumerate(raw_hands):
            best, best_d = side_of.get(i, (None, match_limit))
            hands_2d.append((h2, best))
            category = hres.handedness[i][0] if hres.handedness and i < len(hres.handedness) else None
            handedness = getattr(category, "category_name", None) if category is not None else None
            hand_score = float(getattr(category, "score", 1.0)) if category is not None else 1.0
            edge = float(np.min(np.r_[h2[:, 0], h2[:, 1], 1.0 - h2[:, 0], 1.0 - h2[:, 1]]))
            border_score = float(np.clip(edge / 0.08, 0.0, 1.0))
            palm = (h2[[H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP]] * [w, h]).astype(np.float32)
            palm_area = float(abs(cv2.contourArea(cv2.convexHull(palm))) / max(w * h, 1))
            assoc = (0.0 if best is None else
                     float(np.exp(-0.5 * (best_d / max(0.6 * match_limit, 1e-6)) ** 2)))
            hand_observations.append(HandDetection2D(
                h2, best, handedness, hand_score, assoc, border_score, palm_area))

        body_candidate = R_depth if R_depth is not None else R_body
        if self._body_R is None:
            self._body_R = body_candidate
        elif np.rad2deg(rotation_distance(self._body_R, body_candidate)) <= float(
                self.orientation_cfg.get("body_max_jump_deg", 45)):
            self._body_R = slerp_rotation(
                self._body_R, body_candidate, float(self.orientation_cfg.get("body_smoothing", 0.35)))
        active_R = self._body_R
        active_origin = origin_depth if origin_depth is not None else origin
        depth_used, hand_depth = {}, {}
        for side, (i_s, i_e, i_w) in ARM_IDX.items():
            ob = arms[side]
            have_depth = R_depth is not None and all(i in depth_points for i in (i_s, i_e, i_w))
            if have_depth:
                td = lambda v: active_R.T @ (v - origin_depth)
                ob.s, ob.e, ob.w = td(depth_points[i_s]), td(depth_points[i_e]), td(depth_points[i_w])
                depth_used[side] = 3
            else:
                ta = lambda v: active_R.T @ (v - origin)
                ob.s, ob.e, ob.w = ta(W[i_s]), ta(W[i_e]), ta(W[i_w])
                depth_used[side] = 0
            ob.conf["upper"] = float(min(vis[i_s], vis[i_e]))
            ob.conf["fore"] = float(min(vis[i_e], vis[i_w]))
            if side in hand_of:
                hi = hand_of[side]
                hw = np.array([[p.x, p.y, p.z] for p in hres.hand_world_landmarks[hi]])
                hand_len = np.linalg.norm(hw[H_MIDDLE_MCP] - hw[H_WRIST])
                ob.grip = float(np.linalg.norm(hw[H_THUMB_TIP] - hw[H_INDEX_TIP]) / max(hand_len, 1e-6))
                score = hres.handedness[hi][0].score if hres.handedness else 1.0
                ob.conf["hand"] = float(min(score, vis[i_w]))
                if self.orientation_source == "depth_required" and depth_m is not None and depth_intrinsics is not None:
                    hold_frames = int(self.depth_cfg.get("hand_model_hold_frames", 6))
                    previous_model = (self._hand_depth_model[side]
                                      if self._hand_depth_misses[side] < hold_frames else None)
                    pts, dinfo = fuse_hand_landmarks(hres.hand_landmarks[hi], depth_m, depth_intrinsics,
                                                     self.depth_cfg, previous_model)
                    if dinfo["direct"]:
                        self._hand_depth_misses[side] = 0
                    else:
                        self._hand_depth_misses[side] += 1
                    if dinfo["model"] is not None and dinfo["direct"]:
                        self._hand_depth_model[side] = dinfo["model"]
                    if self._hand_depth_misses[side] >= hold_frames:
                        self._hand_depth_model[side] = None
                    ob.hand_points_cam = pts
                    ob.hand_point_confidence = dinfo["point_confidence"]
                    ob.hand_depth_valid = dinfo["direct"]
                    ob.hand_depth_confidence = dinfo["confidence"]
                    ob.hand_depth_mode = dinfo["mode"]
                    open_count = open_finger_count(pts)
                    ob.hand_open_fingers = open_count
                    raw_R, center = palm_frame_from_depth(pts, side=side)
                    plane_info = {"inliers": 0, "rms_m": float("inf")}
                    source = "LANDMARK"
                    if open_count >= int(self.orientation_cfg.get("min_open_fingers", 3)):
                        plane_n, plane_center, plane_info = fit_palm_plane(
                            depth_m, depth_intrinsics, hres.hand_landmarks[hi], self.depth_cfg)
                        max_rms = float(self.depth_cfg.get("palm_plane_max_rms_m", 0.012))
                        plane_quality = 0.0
                        if plane_n is not None:
                            support = min(1.0, plane_info["inliers"] /
                                          max(3 * int(self.depth_cfg.get("palm_min_points", 35)), 1))
                            plane_quality = support * np.exp(-plane_info["rms_m"] / max(max_rms, 1e-6))
                            source = "PLANE"
                        raw_R, center = palm_frame_from_depth(pts, plane_n, plane_quality, side=side)
                        if plane_center is not None:
                            center = plane_center
                    stable_R, orientation_mode = self._stabilize_orientation(side, raw_R, source)
                    ob.hand_orientation_mode = orientation_mode
                    if stable_R is not None:
                        ob.H = active_R.T @ stable_R
                        ob.hand_R_cam, ob.hand_center_cam = stable_R, center
                        if center is not None:
                            axis_len = float(self.depth_cfg.get("palm_axis_length_m", 0.08))
                            axis_points = [center] + [center + stable_R[:, j] * axis_len for j in range(3)]
                            axes_px = [project_point(depth_intrinsics, p) for p in axis_points]
                            if all(p is not None for p in axes_px):
                                ob.hand_axes_px = np.asarray(axes_px)
                    dinfo.update({"open_fingers": open_count, "orientation": orientation_mode,
                                  "plane_inliers": plane_info["inliers"], "plane_rms_m": plane_info["rms_m"]})
                    hand_depth[side] = {k: v for k, v in dinfo.items()
                                        if k not in ("model", "point_confidence")}
                elif self.orientation_source == "rgb_world":
                    # MediaPipe hand world landmarks có scale tương đối nhưng orientation đủ để
                    # retarget J5-J7. Mode được chọn khi khởi động và không đổi sang depth giữa phiên.
                    open_count = open_finger_count(hw)
                    ob.hand_open_fingers = open_count
                    raw_R = hand_frame(hw, side=side)
                    stable_R, orientation_mode = self._stabilize_orientation(side, raw_R, "RGB_WORLD")
                    ob.hand_orientation_mode = orientation_mode
                    if stable_R is not None:
                        ob.H = active_R.T @ stable_R
                        ob.hand_R_cam = stable_R
                        ob.hand_axes_px = project_rgb_hand_axes(
                            hres.hand_landmarks[hi], stable_R, w, h)
                    hand_depth[side] = {
                        "direct": 0, "fused": 0, "confidence": ob.conf["hand"],
                        "mode": "RGB_ONLY", "open_fingers": open_count,
                        "orientation": orientation_mode, "plane_inliers": 0,
                        "plane_rms_m": float("inf"),
                    }
                else:
                    # depth_required là strict: mất depth thì HOLD/NONE, tuyệt đối không
                    # dùng hand_world_landmarks làm orientation fallback.
                    self._stabilize_orientation(side, None, "NONE")
            else:
                self._stabilize_orientation(side, None, "NONE")
        return Frame(arms, pose_2d, hands_2d, active_R, t, depth_used, hand_depth,
                     active_origin, hand_observations, tracking={
                         "pose": "HELD" if pose_held else "LIVE",
                         "raw_hands": len(raw_hands),
                         "associated_hands": len(hand_of),
                     })
