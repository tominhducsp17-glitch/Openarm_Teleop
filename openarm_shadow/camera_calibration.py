"""ChArUco utilities cho calibration D435i RGB <-> camera laptop."""
from __future__ import annotations

import cv2
import numpy as np


def aruco_dictionary(name="DICT_5X5_1000"):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV khong co ArUco dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_charuco_board(cfg):
    dictionary = aruco_dictionary(cfg.get("dictionary", "DICT_5X5_1000"))
    return cv2.aruco.CharucoBoard(
        (int(cfg["squares_x"]), int(cfg["squares_y"])),
        float(cfg["square_length_m"]), float(cfg["marker_length_m"]), dictionary)


def camera_matrix_from_realsense(intrinsics):
    return np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                     [0.0, intrinsics.fy, intrinsics.ppy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def detect_charuco(detector, bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if corners is None or ids is None:
        return np.empty((0, 2), np.float32), np.empty((0,), np.int32), marker_corners, marker_ids
    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(np.int32), marker_corners, marker_ids


def common_charuco(board, corners_a, ids_a, corners_b, ids_b, min_common=8):
    lookup_a = {int(i): p for i, p in zip(ids_a, corners_a)}
    lookup_b = {int(i): p for i, p in zip(ids_b, corners_b)}
    common = sorted(set(lookup_a) & set(lookup_b))
    if len(common) < min_common:
        return None
    board_points = np.asarray(board.getChessboardCorners(), np.float32)
    obj = board_points[common].reshape(-1, 1, 3)
    img_a = np.asarray([lookup_a[i] for i in common], np.float32).reshape(-1, 1, 2)
    img_b = np.asarray([lookup_b[i] for i in common], np.float32).reshape(-1, 1, 2)
    return obj, img_a, img_b, np.asarray(common, np.int32)


def calibrate_laptop(board, observations, image_size):
    object_points, image_points = [], []
    board_points = np.asarray(board.getChessboardCorners(), np.float32)
    for corners, ids in observations:
        if len(ids) < 6:
            continue
        object_points.append(board_points[ids].reshape(-1, 1, 3))
        image_points.append(np.asarray(corners, np.float32).reshape(-1, 1, 2))
    if len(object_points) < 6:
        raise ValueError("Can it nhat 6 anh laptop hop le de calibrate intrinsic")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, tuple(image_size), None, None, flags=0)
    return float(rms), K, dist.reshape(-1), rvecs, tvecs


def _skew(v):
    x, y, z = np.asarray(v, float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _epipolar_distances(points_a, points_b, K_a, dist_a, K_b, dist_b, R_a_to_b, t_a_to_b):
    """Khoảng cách point-line đối xứng sau khi bỏ lens distortion.

    Fundamental matrix chỉ mô tả đúng đường thẳng epipolar trong ảnh pinhole.
    Vì vậy phải undistort corner trước; đo trực tiếp raw pixel sẽ phạt nhầm
    distortion ở rìa ảnh, đặc biệt với webcam laptop.
    """
    K_a, K_b = np.asarray(K_a, float), np.asarray(K_b, float)
    E = _skew(t_a_to_b) @ np.asarray(R_a_to_b, float)
    F = np.linalg.inv(K_b).T @ E @ np.linalg.inv(K_a)
    values = []
    for a, b in zip(points_a, points_b):
        aa = cv2.undistortPoints(np.asarray(a, np.float64), K_a, dist_a, P=K_a).reshape(-1, 2)
        bb = cv2.undistortPoints(np.asarray(b, np.float64), K_b, dist_b, P=K_b).reshape(-1, 2)
        la = cv2.computeCorrespondEpilines(aa.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        lb = cv2.computeCorrespondEpilines(bb.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        values.extend(np.abs(la[:, 0] * bb[:, 0] + la[:, 1] * bb[:, 1] + la[:, 2]) /
                      np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-9))
        values.extend(np.abs(lb[:, 0] * aa[:, 0] + lb[:, 1] * aa[:, 1] + lb[:, 2]) /
                      np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-9))
    return np.asarray(values, float)


def stereo_calibrate_fixed(object_points, image_d435i, image_laptop, image_size,
                           K_d435i, dist_d435i, K_laptop, dist_laptop):
    if len(object_points) < 6:
        raise ValueError("Can it nhat 6 cap anh co du corner chung de stereo calibrate")
    result = cv2.stereoCalibrate(
        object_points, image_d435i, image_laptop,
        np.asarray(K_d435i, np.float64), np.asarray(dist_d435i, np.float64),
        np.asarray(K_laptop, np.float64), np.asarray(dist_laptop, np.float64),
        tuple(image_size), flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-9))
    rms, _, _, _, _, R_d_to_l, t_d_to_l, E, F = result
    R_l_to_d = R_d_to_l.T
    t_l_to_d = -R_l_to_d @ t_d_to_l.reshape(3, 1)
    epi = _epipolar_distances(
        image_d435i, image_laptop, K_d435i, dist_d435i, K_laptop, dist_laptop,
        R_d_to_l, t_d_to_l)
    return {
        "rms": float(rms),
        "R_d435i_to_laptop": R_d_to_l,
        "t_d435i_to_laptop_m": t_d_to_l.reshape(3),
        "R_laptop_to_d435i": R_l_to_d,
        "t_laptop_to_d435i_m": t_l_to_d.reshape(3),
        "E": E,
        "F": F,
        "epipolar_median_px": float(np.median(epi)),
        "epipolar_p95_px": float(np.percentile(epi, 95)),
    }
