"""Lọc nhiễu theo từng khớp.

- One Euro filter (Casiez, Roussel, Vogel, CHI 2012): êm khi đứng yên, nhanh khi di chuyển.
- Vùng chết có trễ (hysteresis): rung nhỏ không tới robot.
- Bỏ bước nhảy: đổi quá `jump_deg` trong một khung coi là lỗi tracking, giữ tới khi lặp lại đủ lâu.
- Độ tin cậy riêng từng khớp: khớp nào tin cậy thấp thì chỉ khớp đó đứng yên.
Các ý tưởng xử lý lấy cảm hứng từ repo Im-ma/Teleoperation-Arm (filters.mjs) và bài
Hand Shadowing (arXiv 2603.11383, EMA hai tầng); code ở đây tự viết.
"""
from __future__ import annotations

import math

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x


class JointFilter:
    """Lọc một vector góc (rad). Cấu hình theo từng phần tử."""

    def __init__(self, n, min_cutoff, beta, deadband_deg, jump_deg=35.0, jump_hold_s=0.2,
                 min_conf=0.6, reject_jumps=True):
        as_list = lambda v: list(v) if np.ndim(v) else [v] * n
        self.n = n
        self.f = [OneEuro(mc, b) for mc, b in zip(as_list(min_cutoff), as_list(beta))]
        self.dead = np.deg2rad(as_list(deadband_deg))
        self.jump = np.deg2rad(jump_deg)
        self.jump_hold_s = jump_hold_s
        self.reject_jumps = bool(reject_jumps)
        self.min_conf = min_conf
        self.reset()

    def reset(self):
        for f in self.f:
            f.reset()
        self.out = [None] * self.n
        self.raw = [None] * self.n      # (giá trị thô cuối, thời điểm)
        self.jump_since = [None] * self.n
        self.held = np.ones(self.n, bool)

    def __call__(self, x, conf, t):
        """x: mảng n góc (có thể NaN), conf: mảng n độ tin cậy [0..1]. Trả về (out, held)."""
        for i in range(self.n):
            xi = x[i]
            if not np.isfinite(xi) or conf[i] < self.min_conf:
                self.held[i] = True
                self.jump_since[i] = None
                continue
            if (self.reject_jumps and self.raw[i] is not None and
                    abs(xi - self.raw[i]) > self.jump):
                if self.jump_since[i] is None:
                    self.jump_since[i] = t
                if t - self.jump_since[i] < self.jump_hold_s:
                    self.held[i] = True
                    continue          # chờ xem bước nhảy có lặp lại không
                self.f[i].reset()     # nhảy thật (lặp lại đủ lâu): bắt đầu lại bộ lọc
            self.jump_since[i] = None
            self.raw[i] = xi
            y = self.f[i](xi, t)
            o = self.out[i]
            if o is None or abs(y - o) > self.dead[i]:
                self.out[i] = y if o is None else y - math.copysign(self.dead[i], y - o)
            self.held[i] = False
        out = np.array([np.nan if o is None else o for o in self.out])
        return out, self.held.copy()


class EMA:
    """Làm mượt mũ đơn giản: y = a·x + (1−a)·y_prev (Hand Shadowing dùng a = 0.8 cho điểm mốc)."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None

    def __call__(self, x):
        x = np.asarray(x, float)
        self.y = x if self.y is None or x.shape != self.y.shape else self.alpha * x + (1 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None
