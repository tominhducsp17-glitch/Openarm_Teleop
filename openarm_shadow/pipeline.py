"""Nối perception -> retarget -> lọc. Đầu ra: mục tiêu 8 phần tử cho mỗi tay robot
(7 góc URDF theo rad + độ mở kẹp 0..1; NaN = giữ nguyên khớp đó)."""
from __future__ import annotations

import numpy as np

from .filters import EMA, JointFilter
from .geometry import angle_between, orthonormalize, unit
from .kinematics import ArmKinematics
from .perception import ArmObs, Frame
from .retarget import ArmRetargeter, mirror_rotation, mirror_vector


class ShadowPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mp = cfg["mapping"]
        self.robot_sides = list(mp["robot_arms"])
        self.mode = mp["mode"]                                  # "direct" hoặc "mirror"
        self.kins = {s: ArmKinematics(s) for s in self.robot_sides}
        rc = cfg["retarget"]
        self.rt = {s: ArmRetargeter(self.kins[s], rc["elbow_straight_deg"]) for s in self.robot_sides}
        fc = cfg["filter"]
        self.filt = {
            s: JointFilter(8, fc["min_cutoff"], fc["beta"], fc["deadband_deg"], fc["jump_deg"],
                           fc["jump_hold_s"], fc["min_conf"], fc.get("reject_jumps", True))
            for s in self.robot_sides
        }
        self.lm_ema = {s: EMA(fc["landmark_ema_alpha"]) for s in self.robot_sides}
        g = cfg["grip"]
        self.grip_pinch, self.grip_open = g["pinch_ratio"], g["open_ratio"]
        self.q_prev = {s: np.zeros(7) for s in self.robot_sides}
        self.last_info = {}
        cc = cfg.get("calibration", {}).get("hand_auto", {})
        self.auto_calib_enabled = bool(cc.get("enabled", True))
        self.auto_calib_hold_s = float(cc.get("hold_s", 0.6))
        self.auto_calib_min_samples = int(cc.get("min_samples", 5))
        self.auto_calib_motion = np.deg2rad(float(cc.get("max_motion_deg", 12)))
        self.auto_calib_min_open = int(cc.get("min_open_fingers", 4))
        self.auto_calib_upper_down = np.deg2rad(float(cc.get("max_upper_from_down_deg", 55)))
        self.auto_calib_fore_down = np.deg2rad(float(cc.get("max_fore_from_down_deg", 55)))
        self.auto_calib_palm_camera = np.deg2rad(float(cc.get("max_palm_from_camera_deg", 50)))
        self.hand_calibrated = {s: False for s in self.robot_sides}
        self.calib_ready_now = {s: False for s in self.robot_sides}
        self.calib_progress = {s: 0.0 for s in self.robot_sides}
        self.calib_hint = {s: "dua tay vao khung" for s in self.robot_sides}
        self._calib_samples = {s: [] for s in self.robot_sides}
        self._calib_start = {s: None for s in self.robot_sides}
        self._calib_prev = {s: None for s in self.robot_sides}

    # ------------------------------------------------------------------
    def human_side_for(self, robot_side):
        if self.mode == "direct":
            return robot_side                     # tay phải người -> tay phải robot
        return "left" if robot_side == "right" else "right"

    def _obs_for_robot(self, frame: Frame, robot_side) -> ArmObs:
        ob = frame.arms[self.human_side_for(robot_side)]
        if self.mode == "direct" or ob.s is None:
            return ob
        m = ArmObs(s=mirror_vector(ob.s), e=mirror_vector(ob.e), w=mirror_vector(ob.w),
                   H=mirror_rotation(ob.H), grip=ob.grip, conf=dict(ob.conf),
                   hand_points_cam=ob.hand_points_cam, hand_point_confidence=ob.hand_point_confidence,
                   hand_depth_valid=ob.hand_depth_valid,
                   hand_depth_confidence=ob.hand_depth_confidence, hand_depth_mode=ob.hand_depth_mode,
                   hand_R_cam=ob.hand_R_cam, hand_center_cam=ob.hand_center_cam,
                   hand_axes_px=ob.hand_axes_px, hand_open_fingers=ob.hand_open_fingers,
                   hand_orientation_mode=ob.hand_orientation_mode)
        return m

    def _neutral_reference(self, side, ob):
        """Giữ J1–J4 theo tư thế tay hiện tại, đặt J5–J7=0 để calib ở vị trí dễ thấy camera."""
        u = unit(ob.e - ob.s)
        l = unit(ob.w - ob.e)
        q_ref, _ = self.rt[side].solve(u, l, None, self.q_prev[side])
        q_ref[4:7] = 0.0
        return q_ref

    def calibrate_hand_neutral(self, frame: Frame):
        """Calib hướng tay tại tư thế hiện tại; J5–J7 của robot được xem là trung tính."""
        done = []
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            if ob.H is not None and ob.s is not None and ob.e is not None and ob.w is not None:
                self.rt[s].set_hand_neutral(ob.H, self._neutral_reference(s, ob))
                self.hand_calibrated[s] = True
                self.calib_progress[s] = 1.0
                self.calib_hint[s] = "READY"
                self._reset_auto_calib(s, keep_progress=True)
                done.append(s)
        return done

    def _reset_auto_calib(self, side, keep_progress=False):
        self._calib_samples[side] = []
        self._calib_start[side] = None
        self._calib_prev[side] = None
        if not keep_progress:
            self.calib_progress[side] = 0.0

    def _calib_pose_status(self, ob):
        down = np.array([0.0, 0.0, -1.0])
        toward_camera = np.array([0.0, 0.0, -1.0])
        valid = ob.H is not None and ob.s is not None and ob.e is not None and ob.w is not None
        hint = "dua tay vao khung"
        if valid and ob.conf["hand"] < self.cfg["filter"]["min_conf"]:
            valid, hint = False, "tay chua ro"
        if valid and ob.hand_open_fingers < self.auto_calib_min_open:
            valid, hint = False, "xoe ban tay"
        if valid and ob.hand_R_cam is None:
            valid, hint = False, "cho depth ban tay"
        u = None
        if valid:
            u, l = unit(ob.e - ob.s), unit(ob.w - ob.e)
            if angle_between(u, down) > self.auto_calib_upper_down:
                valid, hint = False, "tha canh tay xuong"
            elif angle_between(l, down) > self.auto_calib_fore_down:
                valid, hint = False, "tha co tay xuong"
            elif angle_between(ob.hand_R_cam[:, 2], toward_camera) > self.auto_calib_palm_camera:
                valid, hint = False, "long ban tay nhin camera"
        return valid, hint, u

    def auto_calibrate_hand_neutral(self, frame: Frame):
        """Tự calib khi tay buông xuống hơi dang, khuỷu thả lỏng, lòng bàn tay nhìn camera.

        Chỉ gọi khi SafetyGate chưa engage để thay offset cổ tay không làm lệnh nhảy trong lúc follow.
        """
        done = []
        if not self.auto_calib_enabled:
            return done
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            valid, hint, u = self._calib_pose_status(ob)
            self.calib_ready_now[s] = bool(valid)
            if self.hand_calibrated[s]:
                self.calib_hint[s] = "READY" if valid else hint
                continue
            if not valid:
                self.calib_hint[s] = hint
                self._reset_auto_calib(s)
                continue
            self.calib_hint[s] = "giu yen"
            prev = self._calib_prev[s]
            if prev is not None:
                prev_u, prev_H = prev
                dH = np.asarray(prev_H).T @ np.asarray(ob.H)
                hand_motion = np.arccos(np.clip((np.trace(dH) - 1) / 2, -1, 1))
                if angle_between(prev_u, u) > self.auto_calib_motion or hand_motion > self.auto_calib_motion:
                    self.calib_hint[s] = "giu yen"
                    self._reset_auto_calib(s)
            if self._calib_start[s] is None:
                self._calib_start[s] = frame.t
            self._calib_samples[s].append(np.asarray(ob.H).copy())
            self._calib_prev[s] = (u.copy(), np.asarray(ob.H).copy())
            elapsed = max(0.0, frame.t - self._calib_start[s])
            self.calib_progress[s] = min(1.0, elapsed / max(self.auto_calib_hold_s, 1e-3))
            if elapsed >= self.auto_calib_hold_s and len(self._calib_samples[s]) >= self.auto_calib_min_samples:
                H = orthonormalize(np.mean(self._calib_samples[s], axis=0))
                self.rt[s].set_hand_neutral(H, self._neutral_reference(s, ob))
                self.hand_calibrated[s] = True
                self.calib_hint[s] = "READY"
                self._reset_auto_calib(s, keep_progress=True)
                done.append(s)
        return done

    def seed(self, q_meas: dict):
        """Đặt nghiệm tham chiếu = tư thế robot hiện tại (để chọn nghiệm gần nhất)."""
        for s in self.robot_sides:
            self.q_prev[s] = np.asarray(q_meas[s][:7], float).copy()

    def step(self, frame: Frame):
        targets = {}
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            fc = self.cfg["filter"]
            c_up, c_fo, c_ha = ob.conf["upper"], ob.conf["fore"], ob.conf["hand"]
            ok_up, ok_fo, ok_ha = c_up >= fc["min_conf"], c_fo >= fc["min_conf"], c_ha >= fc["min_conf"]
            u = l = H = None
            if ob.s is not None and ok_up:
                pts = self.lm_ema[s](np.stack([ob.s, ob.e, ob.w]))
                s_, e_, w_ = pts
                u = e_ - s_
                if ok_fo:
                    l = w_ - e_
            if ok_ha and ob.H is not None:
                H = ob.H
            q, info = self.rt[s].solve(u, l, H, self.q_prev[s])
            self.q_prev[s] = q
            self.last_info[s] = info
            grip = np.nan
            if ob.grip is not None and ok_ha:
                grip = np.clip((ob.grip - self.grip_pinch) / (self.grip_open - self.grip_pinch), 0, 1)
            raw = np.append(q, grip)
            conf = np.array([c_up, c_up, c_fo, c_fo, c_ha, c_ha, c_ha, c_ha])
            if u is None:
                conf[:2] = 0
            if l is None:
                conf[2:4] = 0
            if H is None:
                conf[4:7] = 0
            if info.elbow_straight:
                conf[2] = 0          # J3 không xác định khi tay thẳng -> giữ
            out, _ = self.filt[s](raw, conf, frame.t)
            targets[s] = out
        return targets
