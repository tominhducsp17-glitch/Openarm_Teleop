"""Backend OpenArm v1.0 thật qua openarm_can (CAN-FD, SavvyCAN/PEAK -> can0/can1).

CHƯA CHẠY TRÊN ROBOT THẬT. Trước khi dùng phải làm các bước trong docs/SAFETY.md, nhất là
kiểm tra quy ước góc URDF ↔ góc motor (robot.urdf_to_motor trong config).

API openarm_can dùng ở đây đã đối chiếu với enactic/openarm_can commit f340d4b (16/09/2026).

Lọc số đọc rác (cùng cách với taichi_player v3.2 của nhóm): ở chế độ CallbackMode.STATE, openarm_can coi mọi
gói có DLC >= 8 từ recv id là gói trạng thái, nên phản hồi ghi tham số của Damiao (byte đầu 0x55) bị giải mã
thành góc rác, vd q = -12.4676 rad. Nếu số rác này thành lệnh vị trí, tay sẽ giật mạnh. Vì vậy:
- xả hết gói còn trên bus sau khi đổi chế độ và sau khi enable;
- trước khi enable phải có 2 lần đọc liên tiếp khớp nhau, và khớp với tư thế lúc connect();
- mỗi lần đọc: |q| > max_abs_rad hoặc nhảy > max_jump_rad so với lần tốt gần nhất -> bỏ, giữ giá trị tốt gần nhất;
- một khớp đọc hỏng liên tục quá bad_hold_s -> RobotFault (trừ khi đang về tư thế nghỉ, xem `returning`).
"""
from __future__ import annotations

import time

import numpy as np

MOTOR_TYPES = ["DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"]
SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]


class RobotFault(RuntimeError):
    pass


class _Arm:
    def __init__(self, side, rcfg):
        import openarm_can as oa
        self.oa = oa
        self.side = side
        self.iface = rcfg["interfaces"][side]
        m = rcfg["urdf_to_motor"][side]
        self.sign = np.asarray(m["sign"], float)
        self.offset = np.deg2rad(np.asarray(m["offset_deg"], float))
        # motor_limits_deg mô tả dải cơ khí theo góc URDF. Khi encoder có offset
        # phần mềm, giới hạn cuối ở miền encoder cũng phải được biến đổi tương tự;
        # nếu không, vd J4 zero=-4.5° sẽ bị clip thành 0° ngay lúc enable.
        lim_urdf = np.deg2rad(np.asarray(rcfg["motor_limits_deg"][side], float))
        lim_motor = self.sign[:, None] * lim_urdf + self.offset[:, None]
        self.mlo, self.mhi = np.min(lim_motor, axis=1), np.max(lim_motor, axis=1)
        self.kp = np.asarray(rcfg["kp"], float)
        self.kd = np.asarray(rcfg["kd"], float)
        g = rcfg["gripper"]
        self.grip_on = bool(g["enabled"])
        self.g_open, self.g_closed = np.deg2rad(g["open_deg"]), np.deg2rad(g["closed_deg"])
        self.g_kp, self.g_kd = float(g["kp"]), float(g["kd"])
        self.stale_s = float(rcfg.get("feedback_timeout_s", 0.1))
        rf = rcfg.get("read_filter", {})
        self.max_abs = float(rf.get("max_abs_rad", 3.7))
        self.max_jump = float(rf.get("max_jump_rad", 0.35))
        self.bad_hold_s = float(rf.get("bad_hold_s", 0.2))
        self.consistent_tol = np.deg2rad(float(rf.get("consistent_tol_deg", 1.0)))

        self.arm = oa.OpenArm(self.iface, True)
        self.arm.init_arm_motors([getattr(oa.MotorType, t) for t in MOTOR_TYPES], SEND_IDS, RECV_IDS)
        self.arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
        self.arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.q_motor = np.full(7, np.nan)      # giá trị tốt gần nhất (sau lọc)
        self.q_raw = np.full(7, np.nan)        # giá trị đọc thô gần nhất
        self.bad_since = np.full(7, np.nan)    # thời điểm khớp bắt đầu đọc hỏng liên tục
        self.n_rejected = 0                    # số lần bỏ số đọc rác (để in khi kết thúc)
        self.cand = np.full(7, np.nan)         # giá trị "nhảy" đang chờ xác nhận
        self.cand_n = np.zeros(7, int)
        self.g_motor = np.nan
        self.drain()

    # ---- đổi đơn vị ----
    def to_motor(self, q_urdf):
        return self.sign * np.asarray(q_urdf) + self.offset

    def to_urdf(self, q_motor):
        return (np.asarray(q_motor) - self.offset) * self.sign

    def grip_to_motor(self, f):
        return self.g_closed + f * (self.g_open - self.g_closed)

    def grip_from_motor(self, g):
        span = self.g_open - self.g_closed
        return float(np.clip((g - self.g_closed) / span, 0, 1)) if abs(span) > 1e-6 else 0.5

    # ---- đọc ----
    def drain(self, n=5):
        """Xả các gói còn trên bus (vd phản hồi ghi tham số 0x55) để chúng không bị đọc như gói trạng thái."""
        for _ in range(n):
            self.arm.recv_all(2000)

    def _read_raw(self):
        q = np.array([m.get_position() for m in self.arm.get_arm().get_motors()], float)
        gm = self.arm.get_gripper().get_motors()
        g = float(gm[0].get_position()) if gm else np.nan
        return q, g

    def _update(self, now=None):
        """Đọc góc và lọc số đọc rác. q_motor chỉ nhận giá trị hợp lệ; khớp hỏng giữ giá trị tốt gần nhất."""
        now = time.monotonic() if now is None else now
        raw, g = self._read_raw()
        sane = np.isfinite(raw) & (np.abs(raw) <= self.max_abs)
        have = np.isfinite(self.q_motor)
        jump = np.abs(raw - np.where(have, self.q_motor, raw))
        ok = sane & (~have | (jump <= self.max_jump))
        # Số rác xuất hiện lẻ tẻ; một bước nhảy đọc được giống nhau 3 lần liên tiếp là chuyển động thật
        # (vd tay bị cầm di chuyển giữa hai lần đọc thưa ở --dry-run) -> chấp nhận.
        jumped = sane & ~ok
        same = jumped & (np.abs(raw - np.nan_to_num(self.cand, nan=np.inf)) <= 0.05)
        self.cand_n = np.where(same, self.cand_n + 1, np.where(jumped, 1, 0))
        self.cand = np.where(jumped, raw, np.nan)
        confirmed = self.cand_n >= 3
        ok |= confirmed
        self.cand_n[confirmed] = 0
        self.n_rejected += int(np.count_nonzero(~ok & np.isfinite(raw)))
        self.q_raw = raw
        self.q_motor = np.where(ok, raw, self.q_motor)
        self.bad_since = np.where(ok, np.nan, np.where(np.isnan(self.bad_since), now, self.bad_since))
        if np.isfinite(g) and abs(g) <= self.max_abs:
            self.g_motor = g

    def check_fresh(self, tolerate_bad=False):
        comp = self.arm.get_arm()
        if hasattr(comp, "get_link_stats"):
            stale = [i + 1 for i in range(7) if comp.get_link_stats(i).seconds_since_response() > self.stale_s]
            if stale:
                raise RobotFault(f"{self.side}: mất phản hồi khớp {stale}")
        if not np.all(np.isfinite(self.q_motor)):
            raise RobotFault(f"{self.side}: chưa đọc được góc hợp lệ {self.q_motor}")
        if tolerate_bad:
            return
        bad = np.isfinite(self.bad_since) & (time.monotonic() - np.nan_to_num(self.bad_since, nan=np.inf)
                                             > self.bad_hold_s)
        if bad.any():
            raise RobotFault(f"{self.side}: khớp {list(np.flatnonzero(bad) + 1)} đọc hỏng liên tục "
                             f"> {self.bad_hold_s} s (thô: {np.round(self.q_raw, 3)})")

    def refresh(self):
        self.arm.refresh_all()
        self.arm.recv_all(2000)
        self._update()

    def read_consistent(self, tries=20):
        """Đọc tới khi được 2 lần liên tiếp hợp lệ và khớp nhau (motor phải đứng yên). Trả về góc motor."""
        prev = None
        for _ in range(tries):
            self.arm.refresh_all()
            self.arm.recv_all(2000)
            raw, g = self._read_raw()
            valid = bool(np.all(np.isfinite(raw)) and np.all(np.abs(raw) <= self.max_abs))
            if valid and prev is not None and np.max(np.abs(raw - prev)) <= self.consistent_tol:
                self.q_motor, self.q_raw = raw.copy(), raw.copy()
                self.bad_since[:] = np.nan
                if np.isfinite(g) and abs(g) <= self.max_abs:
                    self.g_motor = g
                return raw
            prev = raw if valid else None
            time.sleep(0.02)
        raise RobotFault(f"{self.side}: không có 2 lần đọc khớp nhau sau {tries} lần (thô gần nhất: "
                         f"{np.round(raw, 3)}). Tay có đang bị cầm/di chuyển không?")

    def state(self):
        return np.append(self.to_urdf(self.q_motor), self.grip_from_motor(self.g_motor))


class OpenArmCANRobot:
    def __init__(self, rcfg: dict, sides):
        self.rcfg = rcfg
        self.sides = list(sides)
        self.arms = {s: _Arm(s, rcfg) for s in self.sides}
        self.enabled = False
        self.pre_enabled = False  # enable_all chỉ để đánh thức encoder, chưa gửi MIT target
        self.returning = False          # True khi đang về tư thế nghỉ: đọc hỏng không dừng giữa chừng
        self.t_enable = 0.0
        self.q_connect = {}
        self.gain_ramp_s = float(rcfg.get("gain_ramp_s", 1.0))
        self.gravity = None
        gc = rcfg.get("gravity_comp", {})
        if gc.get("enabled"):
            from .gravity import GravityModel
            self.gravity = GravityModel(gc["urdf"], self.sides)

    def connect(self):
        """Đọc tư thế khi motor còn tắt (2 lần đọc khớp nhau). Trả về trạng thái theo góc URDF."""
        for s, a in self.arms.items():
            a.drain()
            try:
                self.q_connect[s] = a.read_consistent().copy()
                a.check_fresh()
            except RobotFault:
                # Một số firmware ngừng trả lời refresh sau disable_all(). Đánh
                # thức motor ở chế độ mềm (chưa có MIT target), rồi mới đọc pose.
                # Đây cũng là hành vi của tools/bringup/read_joints.py --enable.
                try:
                    a.arm.enable_all()
                    time.sleep(0.1)
                    a.drain()
                    self.q_connect[s] = a.read_consistent().copy()
                    a.check_fresh()
                    self.pre_enabled = True
                except Exception:
                    a.arm.disable_all()
                    raise
        bad = self.out_of_range()
        if bad:
            print("CẢNH BÁO: góc motor nằm ngoài giới hạn, zero của motor có thể sai -> KHÔNG được bật motor:")
            for line in bad:
                print("   ", line)
        return self.read()

    def out_of_range(self, tol_deg=5.0):
        """Các khớp có góc encoder nằm ngoài dải cơ khí đã đổi qua sign/offset (± tol).

        Tay thả xuôi mà đọc ra vd J1 = 178° nghĩa là zero của motor sai (chưa hiệu chuẩn hoặc hiệu chuẩn bị mất).
        Nếu vẫn bật motor, lệnh đầu tiên bị kẹp vào giới hạn và tay sẽ quay một góc rất lớn."""
        tol = np.deg2rad(tol_deg)
        out = []
        for s, a in self.arms.items():
            q = a.q_motor
            for i in np.flatnonzero((q < a.mlo - tol) | (q > a.mhi + tol)):
                out.append(f"{s} J{i + 1}: {np.rad2deg(q[i]):7.1f}° (giới hạn motor "
                           f"{np.rad2deg(a.mlo[i]):.0f}..{np.rad2deg(a.mhi[i]):.0f}°)")
        return out

    def read(self):
        return {s: a.state() for s, a in self.arms.items()}

    def poll(self):
        """Đọc lại góc khi motor đang TẮT (chế độ --dry-run). Không tạo mô-men."""
        for a in self.arms.values():
            a.refresh()
            a.check_fresh()
        return self.read()

    def enable(self):
        """Bật motor. Từ chối nếu tư thế đã khác lúc connect() (lệnh đầu tiên = tư thế lúc connect)."""
        tol = np.deg2rad(float(self.rcfg.get("read_filter", {}).get("enable_pose_tol_deg", 3.0)))
        bad = self.out_of_range()
        if bad:
            raise RobotFault("Không bật motor: góc đo nằm ngoài giới hạn (zero motor sai?): " + "; ".join(bad) +
                             ". Hiệu chuẩn lại zero trước (docs/SAFETY.md).")
        for s, a in self.arms.items():
            q = a.read_consistent()
            d = np.abs(q - self.q_connect.get(s, q))
            if np.max(d) > tol:
                raise RobotFault(f"{s}: tư thế đã đổi {np.round(np.rad2deg(d), 1)} độ kể từ lúc đọc đầu. "
                                 "Để tay yên rồi chạy lại.")
        try:
            for a in self.arms.values():
                a.arm.enable_all()
                time.sleep(0.05)
                a.drain()                        # xả phản hồi của lệnh enable
                a.refresh()                      # đọc lại, qua bộ lọc
                a.check_fresh()
        except Exception:
            for a in self.arms.values():
                a.arm.disable_all()
            raise
        self.enabled, self.t_enable = True, time.monotonic()
        self.pre_enabled = False

    def send(self, cmd):
        if not self.enabled:
            raise RobotFault("send() khi chưa enable")
        ramp = min(1.0, (time.monotonic() - self.t_enable) / self.gain_ramp_s)
        tau = self.gravity.torques({s: np.asarray(cmd[s][:7]) for s in self.sides}) if self.gravity else None
        oa = next(iter(self.arms.values())).oa
        for s, a in self.arms.items():
            q_m = np.clip(a.to_motor(cmd[s][:7]), a.mlo, a.mhi)
            ff = np.zeros(7) if tau is None else a.sign * tau[s]
            a.arm.get_arm().mit_control_all(
                [oa.MITParam(float(a.kp[i] * ramp), float(a.kd[i]), float(q_m[i]), 0.0, float(ff[i]))
                 for i in range(7)])
            if a.grip_on and np.isfinite(cmd[s][7]):
                a.arm.get_gripper().mit_control_all(
                    [oa.MITParam(a.g_kp * ramp, a.g_kd, float(a.grip_to_motor(cmd[s][7])), 0.0, 0.0)])
            a.arm.recv_all(1000)
            a._update()
            a.check_fresh(tolerate_bad=self.returning)

    def rejected_reads(self):
        return {s: a.n_rejected for s, a in self.arms.items()}

    def relax(self, seconds=1.0):
        """Chuyển sang giảm chấn (kp = 0) rồi tắt motor. Gọi khi tay đã về tư thế nghỉ."""
        if not self.enabled:
            return
        oa = next(iter(self.arms.values())).oa
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            for a in self.arms.values():
                a.arm.get_arm().mit_control_all(
                    [oa.MITParam(0.0, float(a.kd[i]), float(a.q_motor[i]), 0.0, 0.0) for i in range(7)])
                a.arm.recv_all(1000)
                a._update()
            time.sleep(0.01)
        for a in self.arms.values():
            a.arm.disable_all()
            a.arm.recv_all(2000)
        self.enabled = False

    def close(self):
        try:
            self.relax()
        finally:
            if self.pre_enabled:
                for a in self.arms.values():
                    a.arm.disable_all()
                    a.arm.recv_all(2000)
                self.pre_enabled = False
            self.enabled = False
            n = self.rejected_reads()
            if any(n.values()):
                print("Số lần bỏ số đọc rác:", n)
