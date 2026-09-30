import numpy as np
import pytest

from openarm_shadow.perception import (deproject_pixel, fit_metric_depth, fuse_hand_landmarks,
                                       fit_palm_plane, hand_frame, open_finger_count, palm_frame_from_depth,
                                       project_rgb_hand_axes, rotation_distance, sample_depth,
                                       sample_depth_with_confidence,
                                       slerp_rotation)


def test_deproject_center_and_offset():
    intr = {"fx": 500.0, "fy": 400.0, "ppx": 320.0, "ppy": 240.0}
    assert np.allclose(deproject_pixel(intr, 320, 240, 2.0), [0, 0, 2])
    assert np.allclose(deproject_pixel(intr, 370, 200, 2.0), [0.2, -0.2, 2])


def test_sample_depth_uses_local_cluster_not_background():
    depth = np.full((11, 11), 3.0, dtype=np.float32)
    depth[4:7, 4:7] = 1.2
    depth[5, 5] = 1.21
    assert sample_depth(depth, 5, 5, radius=3, max_delta_m=0.15) == pytest.approx(1.2, abs=0.02)


def test_sample_depth_finds_nearest_when_center_is_hole():
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[3:6, 3:6] = 1.5
    depth[4, 4] = 0
    assert sample_depth(depth, 4, 4, radius=2) == pytest.approx(1.5)


def test_sample_depth_rejects_invalid_range():
    depth = np.full((5, 5), 8.0, dtype=np.float32)
    assert sample_depth(depth, 2, 2, max_m=6.0) is None


def test_sample_depth_reports_quality():
    depth = np.full((9, 9), 1.25, dtype=np.float32)
    z, confidence = sample_depth_with_confidence(depth, 4, 4, radius=2)
    assert z == pytest.approx(1.25)
    assert confidence > 0.95


def test_fit_metric_depth_rejects_outlier():
    relative = np.linspace(-0.1, 0.1, 9)
    measured = 0.8 * relative + 1.2
    measured[4] = 2.5
    model = fit_metric_depth(relative, measured, np.ones(9), min_points=4)
    assert model[0] == pytest.approx(0.8, abs=0.05)
    assert model[1] == pytest.approx(1.2, abs=0.02)


def test_fit_metric_depth_uses_wrist_anchor_when_sparse():
    relative = np.array([0.0, -0.02, -0.04])
    measured = np.array([1.4, np.nan, np.nan])
    assert fit_metric_depth(relative, measured, min_points=4) == pytest.approx((1.4, 1.4))


class _Landmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def test_fused_hand_points_stay_in_camera_metric_frame():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    # Mô phỏng lỗ stereo ở nửa số landmark; model vẫn phải dựng đủ 21 điểm trong D455 frame.
    landmarks = [_Landmark(0.3 + 0.01 * i, 0.5, -0.002 * i) for i in range(21)]
    for i, p in enumerate(landmarks):
        if i % 2:
            x, y = int(round(p.x * 100)), int(round(p.y * 100))
            depth[y, x] = 0.0
    intr = {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0}
    cfg = {"hand_patch_radius": 0, "min_depth_m": 0.3, "max_depth_m": 3.0,
           "max_local_delta_m": 0.15, "hand_min_direct_points": 4}
    points, info = fuse_hand_landmarks(landmarks, depth, intr, cfg)
    assert info["direct"] >= 4
    assert info["fused"] == 21
    assert np.all(np.isfinite(points))
    assert np.all((points[:, 2] > 0.3) & (points[:, 2] < 3.0))


def _open_hand_points():
    p = np.full((21, 3), np.nan)
    p[0] = [0.0, 0.0, 1.0]
    # Bốn ngón thẳng theo +x, trải từ index (y âm) tới pinky (y dương).
    for chain, yy, base_x in zip(((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
                                 (-0.035, -0.012, 0.015, 0.04), (0.07, 0.08, 0.078, 0.065)):
        for j, idx in enumerate(chain):
            p[idx] = [base_x + 0.03 * j, yy, 1.0]
    return p


def test_open_hand_and_palm_frame_are_well_formed():
    points = _open_hand_points()
    assert open_finger_count(points) == 4
    R, center = palm_frame_from_depth(points, np.array([0.0, 0.0, 1.0]), 0.8)
    assert center.shape == (3,)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    # Plane normal phải được đổi dấu để nhất quán với landmark normal, không lật frame 180 độ.
    assert R[2, 2] < -0.9


def test_right_hand_chirality_flips_palm_normal_outward():
    points = _open_hand_points()
    left, _ = palm_frame_from_depth(points, side="left")
    right, _ = palm_frame_from_depth(points, side="right")
    assert left[:, 2] @ right[:, 2] < -0.99
    assert np.linalg.det(right) == pytest.approx(1.0, abs=1e-6)


def test_rgb_world_hand_frame_is_a_rotation():
    points = _open_hand_points()
    for side in ("left", "right"):
        R = hand_frame(points, side=side)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_rgb_world_right_hand_chirality_matches_depth_convention():
    points = _open_hand_points()
    left = hand_frame(points, side="left")
    right = hand_frame(points, side="right")
    depth_right, _ = palm_frame_from_depth(points, side="right")
    assert left[:, 2] @ right[:, 2] < -0.99
    assert right[:, 2] @ depth_right[:, 2] > 0.99


def test_rgb_hand_axes_are_projected_at_palm_scale():
    landmarks = [_Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    landmarks[0] = _Landmark(0.50, 0.60, 0.0)
    landmarks[5] = _Landmark(0.45, 0.50, 0.0)
    landmarks[9] = _Landmark(0.50, 0.45, 0.0)
    landmarks[17] = _Landmark(0.55, 0.50, 0.0)
    axes = project_rgb_hand_axes(landmarks, np.eye(3), 640, 480)
    assert axes.shape == (4, 2)
    assert np.all(np.isfinite(axes))
    assert axes[1, 0] > axes[0, 0]
    assert axes[2, 1] > axes[0, 1]


def test_fit_palm_plane_from_depth_roi():
    depth = np.full((120, 120), 1.0, dtype=np.float32)
    coords = [(0.50, 0.75), (0.35, 0.38), (0.50, 0.32), (0.60, 0.36), (0.70, 0.42)]
    landmarks = [_Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    for idx, xy in zip((0, 5, 9, 13, 17), coords):
        landmarks[idx] = _Landmark(*xy, 0.0)
    intr = {"fx": 120.0, "fy": 120.0, "ppx": 60.0, "ppy": 60.0}
    normal, center, info = fit_palm_plane(
        depth, intr, landmarks,
        {"min_depth_m": 0.3, "max_depth_m": 3.0, "palm_min_points": 35,
         "palm_depth_band_m": 0.06, "palm_plane_max_rms_m": 0.012})
    assert info["inliers"] >= 35
    assert info["rms_m"] < 1e-5
    assert center[2] == pytest.approx(1.0)
    assert abs(normal[2]) > 0.99


def test_slerp_rotation_has_expected_half_angle():
    R0 = np.eye(3)
    R1 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    halfway = slerp_rotation(R0, R1, 0.5)
    assert np.rad2deg(rotation_distance(R0, halfway)) == pytest.approx(45.0, abs=1e-6)
    assert np.allclose(halfway.T @ halfway, np.eye(3), atol=1e-6)
