import cv2
import numpy as np

from openarm_shadow.multiview import (closest_to_primary, OrientationFusion,
                                      RobustHandFusion, triangulate_hand)


def test_triangulate_hand_recovers_points_and_low_reprojection_error():
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    R = cv2.Rodrigues(np.array([0., .08, 0.]))[0]
    t = np.array([.28, .01, .02])
    cal = {
        "d435i": {"K": K.tolist(), "distortion": [0, 0, 0, 0, 0]},
        "laptop": {"K": K.tolist(), "distortion": [0, 0, 0, 0, 0]},
        "T_laptop_from_d435i": {"R": R.tolist(), "t_m": t.tolist()},
    }
    points = np.array([[.01*i, .005*(i % 4), 1.0 + .01*i] for i in range(21)])
    pd, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, np.zeros(5))
    pl, _ = cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, K, np.zeros(5))
    result = triangulate_hand(cal, pd.reshape(-1, 2), pl.reshape(-1, 2), .9, .7)
    np.testing.assert_allclose(result["points_d435i"], points, atol=1e-5)
    assert np.max(result["error_d435i_px"]) < 1e-4
    assert np.max(result["error_laptop_px"]) < 1e-4
    assert np.median(result["ray_angle_deg"]) > 5


def test_closest_pair_uses_latest_primary_and_nearest_secondary_timestamp():
    class Item:
        def __init__(self, t):
            self.timed_frame = type("F", (), {"host_timestamp_s": t})()
    class Worker:
        def __init__(self, times): self.items = tuple(Item(t) for t in times)
        def history(self): return self.items
    p, s, dt = closest_to_primary(Worker([1.0, 2.0]), Worker([1.7, 1.98, 2.2]), window_s=.25)
    assert p.timed_frame.host_timestamp_s == 2.0
    assert s.timed_frame.host_timestamp_s == 1.98
    np.testing.assert_allclose(dt, .02, atol=1e-12)


def test_robust_rgbd_refinement_improves_noisy_triangulation():
    rng = np.random.default_rng(4)
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    R = cv2.Rodrigues(np.array([0., .08, 0.]))[0]
    t = np.array([.28, .01, .02])
    cal = {
        "d435i": {"K": K.tolist(), "distortion": [0, 0, 0, 0, 0]},
        "laptop": {"K": K.tolist(), "distortion": [0, 0, 0, 0, 0]},
        "T_laptop_from_d435i": {"R": R.tolist(), "t_m": t.tolist()},
    }
    truth = np.array([[.008*i, .008*((i % 4)-1.5), .9 + .004*i] for i in range(21)])
    pd, _ = cv2.projectPoints(truth, np.zeros(3), np.zeros(3), K, np.zeros(5))
    pl, _ = cv2.projectPoints(truth, cv2.Rodrigues(R)[0], t, K, np.zeros(5))
    pd, pl = pd.reshape(-1, 2), pl.reshape(-1, 2)
    noisy_l = pl + rng.normal(0, 5.0, pl.shape)
    tri = triangulate_hand(cal, pd, noisy_l)["points_d435i"]
    depth = truth + rng.normal(0, .003, truth.shape)
    fusion = RobustHandFusion(cal, {"max_nfev": 30, "lambda_depth": 3.0})
    refined = fusion.refine(tri, pd, noisy_l, depth, np.full(21, .9), timestamp=1.0)
    raw_rmse = np.sqrt(np.mean((tri - truth) ** 2))
    refined_rmse = np.sqrt(np.mean((refined["points_d435i"] - truth) ** 2))
    assert refined["mode"] == "FULL_RGBD"
    assert refined_rmse < raw_rmse
    assert refined["depth_points"] == 21


def test_orientation_fusion_keeps_normal_branch_when_landmark_frame_flips():
    fusion = OrientationFusion({"switch_frames": 2, "output_smoothing": 1.0})
    R = np.eye(3)
    first = fusion.update(R, depth_R=R, laptop_R=R, depth_confidence=.9, laptop_confidence=.8)
    assert first["state"] == "TRACKING"
    # Landmark frame bất chợt lật y,z; candidate còn lại vẫn chính là R cũ.
    flipped_input = R @ OrientationFusion.FLIP
    second = fusion.update(flipped_input, depth_R=R, laptop_R=R,
                           depth_confidence=.9, laptop_confidence=.8)
    assert np.rad2deg(np.arccos(np.clip((np.trace(second["R"]) - 1) / 2, -1, 1))) < 1e-5
    assert second["state"] == "TRACKING"


def test_orientation_fusion_holds_then_loses_when_landmarks_missing():
    fusion = OrientationFusion({"hold_frames": 2})
    fusion.update(np.eye(3), fused_confidence=1.0)
    assert fusion.update(None)["state"] == "HOLD"
    assert fusion.update(None)["state"] == "HOLD"
    assert fusion.update(None)["state"] == "LOST"
