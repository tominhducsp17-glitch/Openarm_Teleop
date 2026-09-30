"""Kiểm tra backend CAN với openarm_can giả lập: lọc số đọc rác, 2 lần đọc khớp nhau, dừng khi hỏng liên tục."""
import sys
import time
import types

import numpy as np
import pytest

from openarm_shadow.config import load_config

GARBAGE = -12.4676   # giá trị giải mã từ phản hồi 0x55 của Damiao đã gặp ở taichi_player


def make_fake_openarm_can():
    oa = types.ModuleType("openarm_can")

    class _E:
        def __getattr__(self, k):
            return k

    oa.MotorType = _E()
    oa.CallbackMode = _E()

    class MITParam:
        def __init__(self, kp, kd, q, dq, tau):
            self.kp, self.kd, self.q = kp, kd, q

    class _Stats:
        def __init__(self, m):
            self.m = m

        def seconds_since_response(self):
            return time.monotonic() - self.m.t

    class _Motor:
        def __init__(self, q):
            self.q, self.t, self.glitch = q, time.monotonic(), []   # glitch: các giá trị rác sẽ trả về lần tới

        def get_position(self):
            return self.glitch.pop(0) if self.glitch else self.q

    class _Comp:
        def __init__(self, qs):
            self.ms = [_Motor(q) for q in qs]

        def get_motors(self):
            return self.ms

        def get_link_stats(self, i):
            return _Stats(self.ms[i])

        def mit_control_all(self, ps):
            for m, p in zip(self.ms, ps):
                if p.kp > 0:
                    m.q += 0.5 * (p.q - m.q)
                m.t = time.monotonic()

    class OpenArm:
        instances = []

        def __init__(self, iface, fd):
            self.a, self.g, self.enabled = None, _Comp([0.0]), False
            OpenArm.instances.append(self)

        def init_arm_motors(self, t, a, b):
            self.a = _Comp([0.02, 0.05, 0.0, 0.1, 0.0, 0.0, 0.3])

        def init_gripper_motor(self, *a):
            pass

        def set_callback_mode_all(self, m):
            pass

        def refresh_all(self):
            for m in self.a.ms + self.g.ms:
                m.t = time.monotonic()

        def recv_all(self, t=500):
            pass

        def enable_all(self):
            self.enabled = True

        def disable_all(self):
            self.enabled = False

        def get_arm(self):
            return self.a

        def get_gripper(self):
            return self.g

    oa.MITParam, oa.OpenArm = MITParam, OpenArm
    return oa


@pytest.fixture
def robot(monkeypatch):
    oa = make_fake_openarm_can()
    monkeypatch.setitem(sys.modules, "openarm_can", oa)
    from openarm_shadow.robot.openarm_can_robot import OpenArmCANRobot
    cfg = load_config()
    r = OpenArmCANRobot(cfg["robot"], ["right"])
    return r, oa.OpenArm.instances[-1]


def test_connect_ignores_single_garbage_read(robot):
    r, hw = robot
    hw.a.ms[0].glitch = [GARBAGE]
    q = r.connect()
    assert q["right"][0] == pytest.approx(0.02)


def test_connect_fails_when_reads_never_agree(robot):
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    hw.a.ms[1].glitch = [0.05 + 0.1 * (i % 2) for i in range(100)]   # dao động 0.1 rad mỗi lần đọc
    with pytest.raises(RobotFault):
        r.connect()


def test_send_holds_last_good_value_on_single_glitch(robot):
    r, hw = robot
    q0 = r.connect()["right"]
    r.enable()
    hw.a.ms[3].glitch = [GARBAGE]
    r.send({"right": q0})
    assert np.isfinite(r.arms["right"].q_motor).all()
    assert abs(r.read()["right"][3] - q0[3]) < 0.05
    assert r.rejected_reads()["right"] == 1
    r.send({"right": q0})                  # lần sau đọc tốt -> hết trạng thái hỏng
    assert np.isnan(r.arms["right"].bad_since).all()


def test_persistent_garbage_faults_but_not_while_returning(robot):
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    q0 = r.connect()["right"]
    r.enable()
    hw.a.ms[2].glitch = [GARBAGE] * 1000
    r.returning = True
    for _ in range(30):                    # 0.3 s đọc hỏng khi đang về: không dừng
        r.send({"right": q0})
        time.sleep(0.01)
    r.returning = False
    with pytest.raises(RobotFault):
        r.send({"right": q0})


def test_enable_refuses_if_pose_changed(robot):
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    r.connect()
    hw.a.ms[0].q += np.deg2rad(10)         # tay bị cầm di chuyển sau khi đọc
    with pytest.raises(RobotFault):
        r.enable()
    assert not hw.enabled


def test_poll_reads_without_enabling(robot):
    r, hw = robot
    r.connect()
    hw.a.ms[0].q = 0.1                     # người cầm tay robot nâng J1 lên (motor tắt)
    assert r.poll()["right"][0] == pytest.approx(0.1) and not hw.enabled
    hw.a.ms[0].q = 0.6                     # nhảy lớn giữa hai lần đọc: lần đầu bị nghi là rác...
    assert r.poll()["right"][0] == pytest.approx(0.1)
    r.poll()
    assert r.poll()["right"][0] == pytest.approx(0.6)   # ...đọc giống nhau 3 lần thì chấp nhận


def test_enable_refuses_when_zero_is_wrong(robot):
    """Tay thả xuôi nhưng đọc J1 = 178° (zero motor sai, như tay trái ngày 28/09): không được bật motor."""
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    hw.a.ms[0].q = np.deg2rad(178.2)
    r.connect()
    assert r.out_of_range()
    with pytest.raises(RobotFault):
        r.enable()
    assert not hw.enabled


def test_encoder_limits_follow_software_zero_offset(monkeypatch):
    """Zero J4 âm phải được giữ nguyên, không bị chốt encoder clip về 0°."""
    oa = make_fake_openarm_can()
    monkeypatch.setitem(sys.modules, "openarm_can", oa)
    from openarm_shadow.robot.openarm_can_robot import OpenArmCANRobot
    cfg = load_config()["robot"]
    cfg["urdf_to_motor"]["right"]["offset_deg"] = [0, 0, 0, -4.5, 0, 0, 0]
    r = OpenArmCANRobot(cfg, ["right"])
    arm = r.arms["right"]
    assert np.rad2deg(arm.mlo[3]) == pytest.approx(-4.5)
    assert np.rad2deg(arm.to_motor(np.zeros(7))[3]) == pytest.approx(-4.5)
