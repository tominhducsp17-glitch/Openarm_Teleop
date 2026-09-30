import numpy as np
import pytest

from openarm_shadow.filters import JointFilter, OneEuro
from openarm_shadow.geometry import make_frame, rot, seg_seg_distance, sp1, sp2, unit
from openarm_shadow.kinematics import ArmKinematics
from openarm_shadow.retarget import ArmRetargeter, mirror_rotation, mirror_vector
from openarm_shadow.safety import SafetyGate
from openarm_shadow.config import load_config

rng = np.random.default_rng(0)


def rand_unit():
    return unit(rng.normal(size=3))


# ---------------- hình học ----------------
def test_sp1_recovers_angle():
    for _ in range(200):
        k, p = rand_unit(), rng.normal(size=3)
        th = rng.uniform(-np.pi, np.pi)
        t, deg = sp1(p, rot(k, th) @ p, k)
        assert not deg
        assert np.allclose(rot(k, t) @ p, rot(k, th) @ p, atol=1e-9)


def test_sp2_zero_error_when_reachable():
    for _ in range(200):
        k1, k2, p2 = rand_unit(), rand_unit(), rand_unit()
        t1, t2 = rng.uniform(-np.pi, np.pi, 2)
        p1 = rot(k1, -t1) @ rot(k2, t2) @ p2      # đảm bảo có nghiệm
        sols, err = sp2(p1, p2, k1, k2)
        assert err < 1e-8
        for a, b in sols:
            assert np.linalg.norm(rot(k1, a) @ p1 - rot(k2, b) @ p2) < 1e-6


def test_make_frame_axes():
    R, p = make_frame([0, 0.2, 1.4], [0, -0.2, 1.4], [0, 0, 0.9])
    assert np.allclose(R, np.eye(3), atol=1e-9)
    assert np.allclose(p, [0, 0, 1.4])


def test_seg_distance():
    d = seg_seg_distance(np.array([0, 0, 0.]), np.array([1, 0, 0.]), np.array([0.5, 1, 0.]), np.array([0.5, 1, 1.]))
    assert d == pytest.approx(1.0)


# ---------------- động học ----------------
@pytest.mark.parametrize("side", ["right", "left"])
def test_consecutive_axes_perpendicular(side):
    kin = ArmKinematics(side)
    for _ in range(20):
        q = rng.uniform(kin.lower, kin.upper)
        for i in range(1, 7):
            assert abs(kin.axis_world(q, i) @ kin.axis_world(q, i + 1)) < 1e-6


def test_frame_convention():
    kin = ArmKinematics("right")
    k0 = kin.keypoints(np.zeros(7))
    assert k0["wrist"][2] < k0["elbow"][2] < k0["shoulder"][2]          # tay thả xuôi
    # Ở zero-pose, tâm J7 -> đầu kẹp phải nằm trên trục dọc; dùng gốc J6
    # làm "wrist" sẽ tạo một độ gãy giả 37.5 mm trong hình que.
    hand = k0["tool"] - k0["wrist"]
    assert abs(hand[0]) < 1e-6 and abs(hand[1]) < 1e-6 and hand[2] < 0
    q = np.zeros(7); q[3] = np.pi / 2
    k1 = kin.keypoints(q)
    assert k1["wrist"][0] - k1["elbow"][0] > 0.15                          # gập khuỷu -> cẳng tay ra +x (trước)
    assert kin.keypoints(np.zeros(7))["shoulder"][1] < 0                   # tay phải ở phía −y


def test_display_zero_pose_is_collinear():
    """Offset lắp motor không được tạo độ gập giả trong stick-model."""
    kin = ArmKinematics("right")
    k = kin.display_keypoints(np.zeros(7))
    upper = unit(k["elbow"] - k["shoulder"])
    fore = unit(k["wrist"] - k["elbow"])
    hand = unit(k["tool"] - k["wrist"])
    assert np.allclose(upper, fore, atol=1e-6)
    assert np.allclose(fore, hand, atol=1e-6)


# ---------------- retarget ----------------
@pytest.mark.parametrize("side", ["right", "left"])
def test_retarget_roundtrip(side):
    kin = ArmKinematics(side)
    rt = ArmRetargeter(kin)
    for _ in range(300):
        q = rng.uniform(kin.lower, kin.upper)
        q[3] = rng.uniform(0.4, 2.3)
        u = kin.axis_world(q, 3) * kin.limb_sign[3]
        l = kin.axis_world(q, 5) * kin.limb_sign[5]
        H = kin.R0(q, 7) @ rt.R_offset.T
        qs, info = rt.solve(u, l, H, q + rng.normal(0, 0.03, 7))
        assert info.err_upper_deg < 1e-4 and info.err_fore_deg < 1e-4 and info.err_hand_deg < 1e-3
        assert np.abs(qs - q).max() < 1e-6


def test_human_neutral_gives_zero_pose():
    kin = ArmKinematics("right")
    rt = ArmRetargeter(kin)
    down = np.array([0, 0, -1.0])
    from openarm_shadow.retarget import DEFAULT_HAND_NEUTRAL
    q, info = rt.solve(down, down, DEFAULT_HAND_NEUTRAL, np.zeros(7))
    assert np.abs(q).max() < 1e-4   # URDF làm tròn π/2 thành 1.5708


def test_forward_raise_moves_j1():
    kin = ArmKinematics("right")
    rt = ArmRetargeter(kin)
    fwd = unit([1, 0, -1.0])        # cánh tay nâng ra trước 45 độ
    q, _ = rt.solve(fwd, fwd, None, np.zeros(7))
    assert np.rad2deg(q[0]) == pytest.approx(45, abs=1e-3)
    assert abs(q[1]) < 1e-4


def test_side_raise_moves_j2_positive_on_right_negative_on_left():
    for side, sgn in (("right", 1), ("left", -1)):
        kin = ArmKinematics(side)
        rt = ArmRetargeter(kin)
        out = unit([0, -sgn * 1.0, -1.0])   # dang tay ra ngoài 45 độ
        q, _ = rt.solve(out, out, None, np.zeros(7))
        assert np.sign(q[1]) == sgn and abs(np.rad2deg(abs(q[1])) - 45) < 1e-3


def test_straight_arm_holds_j3():
    kin = ArmKinematics("right")
    rt = ArmRetargeter(kin)
    prev = np.zeros(7); prev[2] = 0.3
    d = unit([1, -0.2, -0.5])
    q, info = rt.solve(d, d, None, prev)
    assert info.elbow_straight and q[2] == pytest.approx(0.3)


def test_limits_respected():
    kin = ArmKinematics("right")
    rt = ArmRetargeter(kin)
    back = unit([-1, 0.3, 0.2])       # đưa tay ra sau quá giới hạn
    q, info = rt.solve(back, back, None, np.zeros(7))
    assert np.all(q >= kin.lower - 1e-9) and np.all(q <= kin.upper + 1e-9)


def test_mirror_rotation_is_proper():
    from openarm_shadow.retarget import DEFAULT_HAND_NEUTRAL
    Hm = mirror_rotation(DEFAULT_HAND_NEUTRAL)
    assert np.linalg.det(Hm) == pytest.approx(1.0)
    assert np.allclose(mirror_vector([1, 2, 3]), [1, -2, 3])


# ---------------- lọc ----------------
def test_one_euro_converges():
    f = OneEuro(1.0, 0.0)
    y = [f(1.0, t * 0.03) for t in range(200)]
    assert y[-1] == pytest.approx(1.0, abs=1e-3)


def test_joint_filter_holds_low_conf_and_rejects_jump():
    jf = JointFilter(2, 5.0, 0.0, [0.0, 0.0], jump_deg=30, jump_hold_s=0.2, min_conf=0.6)
    for k in range(10):
        out, held = jf(np.array([0.1, 0.1]), np.array([1, 1]), k * 0.03)
    out, held = jf(np.array([0.1, 0.1]), np.array([1, 0.1]), 0.33)      # khớp 2 tin cậy thấp
    assert held[1] and not held[0]
    out2, held = jf(np.array([2.0, 0.1]), np.array([1, 1]), 0.36)       # nhảy 109 độ trong 1 khung
    assert held[0] and out2[0] == pytest.approx(out[0])


def test_joint_filter_can_accept_large_step_immediately():
    jf = JointFilter(1, 1000.0, 0.0, [0.0], jump_deg=30, min_conf=0.0,
                     reject_jumps=False)
    jf(np.array([0.0]), np.array([1.0]), 0.0)
    out, held = jf(np.array([2.0]), np.array([1.0]), 0.01)
    assert not held[0]
    assert out[0] > 1.5


# ---------------- an toàn ----------------
def make_gate(velocity_limit_enabled=True):
    cfg = load_config()
    cfg["safety"]["velocity_limit_enabled"] = velocity_limit_enabled
    if velocity_limit_enabled and cfg["safety"]["engage_blend_s"] <= 0:
        cfg["safety"]["engage_blend_s"] = 1.5
    kins = {s: ArmKinematics(s) for s in ("right", "left")}
    g = SafetyGate(kins, cfg["safety"])
    g.reset({s: np.zeros(8) for s in kins})
    return g


def test_gate_holds_until_engaged_and_limits_velocity():
    g = make_gate()
    tgt = {s: np.array([1.0, 0, 0, 0, 0, 0, 0, 0.5]) for s in g.sides}
    g.set_target(tgt, 0.0)
    assert np.allclose(g.step(0.01, 0.0)["right"], 0)
    g.engage(0.0)
    t = 0.0
    for _ in range(300):
        t += 0.01
        g.set_target(tgt, t)
        cmd = g.step(0.01, t)
    assert cmd["right"][0] <= np.deg2rad(45) * 3.0 + 1e-9        # không nhanh hơn 45 độ/s
    assert cmd["right"][0] > 0.5


def test_gate_can_send_target_without_velocity_or_step_limit():
    g = make_gate(velocity_limit_enabled=False)
    tgt = {s: np.array([0.8, 0, 0, 0, 0, 0, 0, 0.5]) for s in g.sides}
    g.engage(0.0)
    g.set_target(tgt, 0.01)
    cmd = g.step(0.01, 0.01)
    assert cmd["right"][0] == pytest.approx(0.8)


def test_gate_deadman():
    g = make_gate()
    g.engage(0.0)
    g.set_target({s: np.full(8, 0.5) for s in g.sides}, 0.0)
    g.step(0.01, 0.01)
    before = {s: v.copy() for s, v in g.cmd.items()}
    after = g.step(0.01, 1.0)                  # 1 s không có mục tiêu mới
    assert all(np.allclose(before[s], after[s]) for s in g.sides)
    assert "dead-man" in g.status


def test_gate_blocks_arm_collision():
    g = make_gate()
    g.lo = {s: np.full(7, -np.pi) for s in g.sides}
    g.hi = {s: np.full(7, np.pi) for s in g.sides}
    g.engage(-10)
    # hai tay đưa ra trước rồi khép vào giữa (J2 khép, J4 gập) -> hai bàn tay gặp nhau
    tgt = {"right": np.array([1.2, -0.6, 0, 1.2, 0, 0, 0, 0.5]),
           "left": np.array([-1.2, 0.6, 0, 1.2, 0, 0, 0, 0.5])}
    t = 0.0
    for _ in range(600):
        t += 0.01
        g.set_target(tgt, t)
        g.step(0.01, t)
    assert g.min_arm_distance(g.cmd) >= g.col_margin - 1e-6 or "va chạm" in g.status


def test_gate_not_stuck_near_collision():
    """Đang sát ngưỡng va chạm, mục tiêu về tư thế nghỉ: gate phải thoát ra được, không đứng im mãi."""
    g = make_gate()
    M = np.array([-1, -1, -1, 1, -1, -1, -1, 1])
    near = np.append(np.deg2rad([69, -5, -15, 109, 0, 0, 0]), 0.5)
    g.reset({"right": near, "left": M * near})
    g.engage(-10)
    rest = {s: np.zeros(8) for s in g.sides}
    t = 0.0
    for _ in range(400):
        t += 0.01
        g.set_target(rest, t)
        g.step(0.01, t)
    assert np.abs(g.cmd["right"][:7]).max() < np.deg2rad(2)
    assert g.min_arm_distance(g.cmd) >= g.col_margin - 1e-6
