"""Lớp an toàn đứng giữa pipeline và robot. Mọi lệnh tới robot (thật hay mô phỏng) đều đi qua đây.

- Giới hạn khớp mềm (hẹp hơn giới hạn URDF), giới hạn vận tốc từng khớp.
- Ly hợp (engage/disengage): chỉ bám người khi đã engage; mỗi lần engage tốc độ tăng dần (smoothstep).
- Dead-man: không có mục tiêu mới quá `deadman_s` giây -> đứng yên tại chỗ.
- Chống hai tay va nhau: mỗi đoạn tay là một capsule; bước nào làm khoảng cách xuống dưới
  ngưỡng và gần hơn trước thì bị bỏ (robot dừng tại chỗ, không nhảy).
Giá trị lệnh: 7 góc khớp theo URDF (rad) + phần tử thứ 8 = độ mở kẹp [0..1].
"""
from __future__ import annotations

import itertools

import numpy as np

from .geometry import seg_seg_distance
from .kinematics import ArmKinematics


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class SafetyGate:
    def __init__(self, kins: dict[str, ArmKinematics], cfg: dict):
        self.kins = kins
        self.sides = list(kins)
        self.max_vel = np.deg2rad(np.asarray(cfg["max_vel_deg_s"], float))
        self.grip_vel = float(cfg.get("grip_vel_per_s", 1.5))
        self.velocity_limit_enabled = bool(cfg.get("velocity_limit_enabled", True))
        self.deadman_s = float(cfg["deadman_s"])
        self.blend_s = float(cfg["engage_blend_s"])
        self.lo, self.hi = {}, {}
        for s, kin in kins.items():
            lim = np.deg2rad(np.asarray(cfg["soft_limits_deg"][s], float))
            self.lo[s] = np.maximum(lim[:, 0], kin.lower)
            self.hi[s] = np.minimum(lim[:, 1], kin.upper)
        col = cfg.get("self_collision", {})
        self.col_on = bool(col.get("enabled", True)) and len(self.sides) == 2
        self.col_margin = float(col.get("margin_m", 0.02))
        r = col.get("radius_m", {})
        self.radius = [r.get("upper", 0.05), r.get("fore", 0.045), r.get("hand", 0.05)]
        self.cmd = None
        self.target = None
        self.t_target = -np.inf
        self.engaged = False
        self.t_engage = 0.0
        self.status = "idle"

    # ------------------------------------------------------------------
    def reset(self, q_meas: dict):
        """Gọi khi kết nối robot: lệnh bắt đầu đúng bằng tư thế đo được (không giật)."""
        self.cmd = {s: np.asarray(q_meas[s], float).copy() for s in self.sides}
        self.target = None
        self.engaged = False

    def engage(self, now):
        self.engaged, self.t_engage = True, now

    def disengage(self):
        self.engaged = False

    def set_target(self, targets: dict, now):
        """targets[side]: mảng 8 phần tử, NaN = giữ khớp đó."""
        self.target = {s: np.asarray(v, float).copy() for s, v in targets.items() if s in self.sides}
        self.t_target = now

    # ------------------------------------------------------------------
    def min_arm_distance(self, q: dict):
        if not self.col_on:
            return np.inf
        segs = {}
        for s in self.sides:
            k = self.kins[s].keypoints(q[s][:7])
            segs[s] = [(k["shoulder"], k["elbow"]), (k["elbow"], k["wrist"]), (k["wrist"], k["tool"])]
        a, b = self.sides
        d = np.inf
        for (i, sa), (j, sb) in itertools.product(enumerate(segs[a]), enumerate(segs[b])):
            if i == 0 and j == 0:
                continue          # hai cánh tay trên luôn cách xa nhau bởi thân
            dd = seg_seg_distance(*sa, *sb) - self.radius[i] - self.radius[j]
            d = min(d, dd)
        return d

    def step(self, dt, now):
        if self.cmd is None:
            raise RuntimeError("SafetyGate.reset() chưa được gọi")
        if not self.engaged or self.target is None:
            self.status = "hold (chưa engage)" if not self.engaged else "hold (chưa có mục tiêu)"
            return self.cmd
        if now - self.t_target > self.deadman_s:
            self.status = "hold (dead-man: mất mục tiêu)"
            return self.cmd
        ramp = (1.0 if self.blend_s <= 0 else
                max(0.05, smoothstep((now - self.t_engage) / self.blend_s)))
        new = {}
        for s in self.sides:
            cur = self.cmd[s]
            goal = self.target.get(s, cur)
            goal = np.where(np.isfinite(goal), goal, cur)
            goal[:7] = np.clip(goal[:7], self.lo[s], self.hi[s])
            goal[7] = np.clip(goal[7], 0.0, 1.0)
            if self.velocity_limit_enabled:
                vmax = np.append(self.max_vel, self.grip_vel) * ramp * dt
                new[s] = cur + np.clip(goal - cur, -vmax, vmax)
            else:
                new[s] = goal.copy()
        if self.col_on:
            d_new, d_cur = self.min_arm_distance(new), self.min_arm_distance(self.cmd)
            if d_new < self.col_margin and d_new < d_cur:
                # Bước đầy đủ làm hai tay xích lại quá gần. Không đứng im cả tay (dễ bị kẹt ở gần vùng va chạm),
                # mà chỉ cho đi từng khớp nào không làm khoảng cách giảm xuống dưới ngưỡng.
                part, d_part, moved = {k: v.copy() for k, v in self.cmd.items()}, d_cur, False
                for s in self.sides:
                    for i in np.flatnonzero(new[s] != part[s]):
                        trial = {k: v.copy() for k, v in part.items()}
                        trial[s][i] = new[s][i]
                        d_t = d_part if i == 7 else self.min_arm_distance(trial)   # kẹp không ảnh hưởng
                        if d_t >= self.col_margin or d_t >= d_part:
                            part, d_part, moved = trial, d_t, moved or i < 7
                self.cmd = part
                self.status = (f"tránh va chạm: chỉ đi khớp an toàn, cách {d_part * 100:.1f} cm" if moved
                               else f"hold (sắp va chạm, cách {d_new * 100:.1f} cm)")
                return self.cmd
        self.cmd = new
        self.status = "follow" if ramp >= 1.0 else f"engage {ramp * 100:.0f}%"
        return self.cmd
