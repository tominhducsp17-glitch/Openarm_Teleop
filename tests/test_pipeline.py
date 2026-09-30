"""Kiểm tra pipeline + vẽ bằng dữ liệu người giả lập (không cần camera / MediaPipe)."""
import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.perception import ArmObs, Frame
from openarm_shadow.pipeline import ShadowPipeline
from openarm_shadow.viz import draw_robot


def fake_frame(pipe, q_true, t, noise=0.0, open_fingers=4, rng=np.random.default_rng(0)):
    arms = {}
    for s, kin in pipe.kins.items():
        k = kin.keypoints(q_true[s])
        # người "to" gấp 1.3 lần robot: chỉ hướng là quan trọng
        s_, e_, w_ = 1.3 * k["shoulder"], 1.3 * k["elbow"], 1.3 * k["wrist"]
        u = kin.axis_world(q_true[s], 3); l = kin.axis_world(q_true[s], 5)
        e_ = s_ + 0.3 * u + rng.normal(0, noise, 3)
        w_ = e_ + 0.27 * l + rng.normal(0, noise, 3)
        H = kin.R0(q_true[s], 7) @ pipe.rt[s].R_offset.T
        arms[s] = ArmObs(s=s_, e=e_, w=w_, H=H, grip=0.6,
                         conf={"upper": 0.9, "fore": 0.9, "hand": 0.9},
                         hand_R_cam=np.diag([1.0, -1.0, -1.0]),
                         hand_open_fingers=open_fingers)
    return Frame(arms, None, [], np.eye(3), t)


def test_pipeline_tracks_slow_motion():
    cfg = load_config()
    pipe = ShadowPipeline(cfg)
    q0 = {s: np.zeros(7) for s in pipe.robot_sides}
    pipe.seed({s: np.zeros(8) for s in pipe.robot_sides})
    q_true = None
    for i in range(150):
        t = i / 30
        a = min(t / 3, 1.0)
        q_true = {"right": np.deg2rad([40 * a, 20 * a, 10 * a, 60 * a, 10 * a, 10 * a, 5 * a]),
                  "left": np.deg2rad([-40 * a, -20 * a, -10 * a, 60 * a, -10 * a, 10 * a, -5 * a])}
        out = pipe.step(fake_frame(pipe, q_true, t, noise=0.003))
    for s in pipe.robot_sides:
        err = np.rad2deg(np.abs(out[s][:7] - q_true[s]))
        assert err.max() < 4.0, (s, err)
        assert 0 <= out[s][7] <= 1
    img = draw_robot(pipe.kins, {s: np.append(q_true[s], 0.5) for s in pipe.robot_sides})
    assert img.shape == (480, 480, 3)


def test_mirror_mode_swaps_arms():
    cfg = load_config()
    cfg["mapping"]["mode"] = "mirror"
    pipe = ShadowPipeline(cfg)
    q_true = {"right": np.deg2rad([30, 20, 0, 50, 0, 0, 0]), "left": np.deg2rad([-30, -20, 0, 50, 0, 0, 0])}
    for i in range(60):
        out = pipe.step(fake_frame(pipe, q_true, i / 30))
    # người dùng tay phải -> robot tay trái nhận tư thế đối xứng của tay phải người
    assert np.allclose(np.rad2deg(out["left"][:4]), np.rad2deg(q_true["left"][:4]), atol=2.0)


def test_auto_calibrates_stable_neutral_hand():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    cfg["calibration"]["hand_auto"]["hold_s"] = 0.2
    cfg["calibration"]["hand_auto"]["min_samples"] = 3
    pipe = ShadowPipeline(cfg)
    q = {"right": np.zeros(7)}
    done = []
    for i in range(10):
        done += pipe.auto_calibrate_hand_neutral(fake_frame(pipe, q, i * 0.05))
    assert done == ["right"]
    assert pipe.hand_calibrated["right"] and pipe.calib_progress["right"] == 1.0
    assert pipe.calib_ready_now["right"]

    # READY là điều kiện sống: bỏ tư thế calib thì countdown auto-sync phải bị huỷ.
    pipe.auto_calibrate_hand_neutral(fake_frame(pipe, q, 1.0, open_fingers=2))
    assert not pipe.calib_ready_now["right"]


def test_auto_calibration_accepts_visible_arm_pose_and_zeros_wrist_reference():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    cfg["calibration"]["hand_auto"]["hold_s"] = 0.2
    cfg["calibration"]["hand_auto"]["min_samples"] = 3
    pipe = ShadowPipeline(cfg)
    q = {"right": np.deg2rad([20, 10, 5, 25, 25, -10, 20])}
    fixed_H = fake_frame(pipe, q, 0.0).arms["right"].H.copy()
    for i in range(10):
        fr = fake_frame(pipe, q, i * 0.05)
        fr.arms["right"].H = fixed_H
        pipe.auto_calibrate_hand_neutral(fr)
    assert pipe.hand_calibrated["right"]
    out = None
    for i in range(60):
        fr = fake_frame(pipe, q, 1.0 + i / 30)
        fr.arms["right"].H = fixed_H
        out = pipe.step(fr)
    assert np.max(np.abs(np.rad2deg(out["right"][4:7]))) < 2.0


def test_auto_calibration_waits_for_open_hand():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    pipe = ShadowPipeline(cfg)
    q = {"right": np.deg2rad([30, 20, 0, 50, 0, 0, 0])}
    for i in range(30):
        pipe.auto_calibrate_hand_neutral(fake_frame(pipe, q, i * 0.05, open_fingers=2))
    assert not pipe.hand_calibrated["right"]


def test_auto_calibration_waits_for_palm_facing_camera():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    pipe = ShadowPipeline(cfg)
    q = {"right": np.zeros(7)}
    for i in range(30):
        fr = fake_frame(pipe, q, i * 0.05)
        fr.arms["right"].hand_R_cam = np.eye(3)  # normal +z: quay lưng bàn tay về camera
        pipe.auto_calibrate_hand_neutral(fr)
    assert not pipe.hand_calibrated["right"]
    assert pipe.calib_hint["right"] == "long ban tay nhin camera"
