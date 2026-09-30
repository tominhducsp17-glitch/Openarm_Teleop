import cv2
import numpy as np

from openarm_shadow.camera_calibration import (common_charuco, detect_charuco,
                                                make_charuco_board, stereo_calibrate_fixed)


CFG = {"dictionary": "DICT_5X5_1000", "squares_x": 7, "squares_y": 5,
       "square_length_m": 0.04, "marker_length_m": 0.03}


def test_generated_charuco_is_detected_and_common_points_keep_ids():
    board = make_charuco_board(CFG)
    image = board.generateImage((1400, 1000), marginSize=30, borderBits=1)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detect_charuco(detector, bgr)
    assert len(ids) >= 20
    common = common_charuco(board, corners, ids, corners + 3, ids, min_common=10)
    assert common is not None
    obj, a, b, common_ids = common
    assert obj.shape[0] == a.shape[0] == b.shape[0] == len(common_ids)
    assert np.allclose(b - a, 3)


def test_stereo_calibration_recovers_known_baseline():
    board = make_charuco_board(CFG)
    obj0 = np.asarray(board.getChessboardCorners(), np.float32).reshape(-1, 1, 3)
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    R_d_to_l = cv2.Rodrigues(np.array([0.0, 0.03, 0.0]))[0]
    t_d_to_l = np.array([[0.24], [0.01], [0.0]])
    objects, image_d, image_l = [], [], []
    for i in range(10):
        rvec_d = np.array([0.03 * i, -0.02 * i, 0.01 * i])
        R_board_d = cv2.Rodrigues(rvec_d)[0]
        t_board_d = np.array([[0.02 * (i % 3)], [-0.02 * (i % 2)], [0.8 + 0.04 * i]])
        pd, _ = cv2.projectPoints(obj0, rvec_d, t_board_d, K, dist)
        R_board_l = R_d_to_l @ R_board_d
        t_board_l = R_d_to_l @ t_board_d + t_d_to_l
        pl, _ = cv2.projectPoints(obj0, cv2.Rodrigues(R_board_l)[0], t_board_l, K, dist)
        objects.append(obj0.copy()); image_d.append(pd.astype(np.float32)); image_l.append(pl.astype(np.float32))
    result = stereo_calibrate_fixed(objects, image_d, image_l, (640, 480), K, dist, K, dist)
    assert result["rms"] < 1e-3
    np.testing.assert_allclose(np.linalg.norm(result["t_d435i_to_laptop_m"]),
                               np.linalg.norm(t_d_to_l), atol=2e-3)
