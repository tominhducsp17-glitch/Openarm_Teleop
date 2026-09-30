"""Ánh xạ tay người → 7 góc khớp OpenArm theo ý tưởng SEW-Mimic (căn HƯỚNG, không căn vị trí).

Ý tưởng (theo phần đã kiểm chứng của arXiv 2602.01632): lấy hướng cánh tay trên u = unit(e − s),
hướng cẳng tay l = unit(w − e) và hướng bàn tay H, rồi giải góc khớp robot để các đoạn tay robot
cùng hướng. Cách chia khớp dưới đây là thiết kế của repo này cho OpenArm v1.0:

    J1, J2 : đưa trục J3 (chạy dọc cánh tay trên) trùng u           -> SP2
    J3, J4 : đưa trục J5 (chạy dọc cẳng tay) trùng l                -> SP2
    J5, J6 : đưa trục J7 trùng trục tương ứng của hướng tay mong muốn -> SP2
    J7     : xoay nốt quanh trục J7 cho khớp hướng tay                -> SP1

Điều kiện: các trục khớp liên tiếp vuông góc (OpenArm v1.0 thoả, xem scripts/check_kinematics.py).
Vì chỉ dùng hướng, chiều dài tay người/robot khác nhau không ảnh hưởng.
Tất cả vector đầu vào phải ở khung "thân": x trước, y trái, z lên (trùng khung world của URDF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import angle_between, rot, sp1, sp2, unit, wrap
from .kinematics import ArmKinematics

# Hướng bàn tay "trung tính" mặc định: tay thả xuôi, lòng bàn tay hướng vào đùi.
#   cột x = hướng ngón tay (cổ tay → gốc ngón giữa) = xuống
#   cột y = hướng từ gốc ngón út sang gốc ngón trỏ  = ra trước
#   cột z = x × y
_X = np.array([0.0, 0.0, -1.0])
_Y = np.array([1.0, 0.0, 0.0])
DEFAULT_HAND_NEUTRAL = np.column_stack([_X, _Y, np.cross(_X, _Y)])


@dataclass
class RetargetInfo:
    err_upper_deg: float = float("nan")
    err_fore_deg: float = float("nan")
    err_hand_deg: float = float("nan")
    elbow_straight: bool = False
    clamped: list = field(default_factory=list)


class ArmRetargeter:
    def __init__(self, kin: ArmKinematics, elbow_straight_deg: float = 12.0,
                 q_neutral=None, hand_neutral=None):
        self.kin = kin
        self.elbow_straight = np.deg2rad(elbow_straight_deg)
        self.q_neutral = np.zeros(7) if q_neutral is None else np.asarray(q_neutral, float)
        self.set_hand_neutral(DEFAULT_HAND_NEUTRAL if hand_neutral is None else hand_neutral)

    def set_hand_neutral(self, H_neutral, q_reference=None):
        """Căn hướng tay người hiện tại với một tư thế robot có J5–J7 trung tính."""
        q_ref = self.q_neutral if q_reference is None else np.asarray(q_reference, float)
        self.R_offset = np.asarray(H_neutral).T @ self.kin.R0(q_ref, 7)

    # ------------------------------------------------------------------
    def _pick(self, cands, q_prev, ia, ib):
        """Chọn nghiệm trong giới hạn khớp và gần q_prev nhất; đưa góc về đúng chu kỳ 2π."""
        lo, hi = self.kin.lower, self.kin.upper
        best, best_score = None, np.inf
        for qa, qb in cands:
            pair = []
            viol = 0.0
            for q, i in ((qa, ia), (qb, ib)):
                opts = [wrap(q) + k * 2 * np.pi for k in (-1, 0, 1)]
                # ưu tiên trong giới hạn, sau đó gần q_prev
                opts.sort(key=lambda x: (max(lo[i] - x, 0) + max(x - hi[i], 0), abs(x - q_prev[i])))
                pair.append(opts[0])
                viol += max(lo[i] - opts[0], 0) + max(opts[0] - hi[i], 0)
            score = 100.0 * viol + abs(pair[0] - q_prev[ia]) + abs(pair[1] - q_prev[ib])
            if score < best_score:
                best, best_score = pair, score
        return best

    def align_axis(self, i, q, v, q_prev):
        """Giải (q_{i-2}, q_{i-1}) để trục khớp i trong world trùng hướng v. i dùng chỉ số 1..7."""
        a, b = i - 2, i - 1
        ja, jb, ji = self.kin.joints[a - 1], self.kin.joints[b - 1], self.kin.joints[i - 1]
        F = self.kin.R0(q, a - 1)
        v_loc = ja.R_local.T @ F.T @ unit(v)
        p2 = jb.R_local @ ji.R_local @ ji.axis
        k2 = jb.R_local @ jb.axis
        sols, _ = sp2(v_loc, p2, ja.axis, k2)
        cands = [(-t1, t2) for t1, t2 in sols]
        qa, qb = self._pick(cands, q_prev, a - 1, b - 1)
        q = q.copy()
        q[a - 1], q[b - 1] = qa, qb
        return q

    # ------------------------------------------------------------------
    def solve(self, u=None, l=None, H=None, q_prev=None):
        """u, l: hướng cánh tay trên / cẳng tay (khung thân). H: 3x3 hướng bàn tay hoặc None.

        Phần nào thiếu (None) thì giữ nguyên các khớp tương ứng của q_prev.
        Trả về (q 7 góc URDF, RetargetInfo).
        """
        kin = self.kin
        q_prev = np.zeros(7) if q_prev is None else np.asarray(q_prev, float)
        q = q_prev.copy()
        info = RetargetInfo()

        if u is not None:
            q = self.align_axis(3, q, kin.limb_sign[3] * unit(u), q_prev)
        if l is not None and u is not None:
            if angle_between(u, l) < self.elbow_straight:
                # tay gần thẳng: xoay cánh tay (J3) không xác định -> giữ J3, chỉ giải J4
                info.elbow_straight = True
                q[2] = q_prev[2]
                j4, j5 = kin.joints[3], kin.joints[4]
                target = j4.R_local.T @ kin.R0(q, 3).T @ (kin.limb_sign[5] * unit(l))
                t, _ = sp1(j5.R_local @ j5.axis, target, j4.axis)
                q[3] = self._pick([(q[2], t)], q_prev, 2, 3)[1]
            else:
                q = self.align_axis(5, q, kin.limb_sign[5] * unit(l), q_prev)
        if H is not None and l is not None:
            R_des = np.asarray(H) @ self.R_offset
            j7 = kin.joints[6]
            q = self.align_axis(7, q, R_des @ j7.axis, q_prev)
            M = (kin.R0(q, 6) @ j7.R_local).T @ R_des
            e = unit(np.cross(j7.axis, [1.0, 0.3, 0.2]))
            t, _ = sp1(e, M @ e, j7.axis)
            q[6] = self._pick([(q[5], t)], q_prev, 5, 6)[1]

        q_c = kin.clamp(q)
        info.clamped = [i + 1 for i in range(7) if abs(q_c[i] - q[i]) > 1e-9]
        if u is not None:
            info.err_upper_deg = np.rad2deg(angle_between(kin.axis_world(q_c, 3) * kin.limb_sign[3], u))
        if l is not None:
            info.err_fore_deg = np.rad2deg(angle_between(kin.axis_world(q_c, 5) * kin.limb_sign[5], l))
        if H is not None:
            R_err = (np.asarray(H) @ self.R_offset).T @ kin.R0(q_c, 7)
            info.err_hand_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
        return q_c, info


def mirror_vector(v):
    """Phản chiếu qua mặt phẳng dọc giữa thân (y → −y): tay phải người điều khiển tay trái robot."""
    return None if v is None else np.asarray(v) * np.array([1.0, -1.0, 1.0])


def mirror_rotation(H):
    if H is None:
        return None
    M = np.diag([1.0, -1.0, 1.0])
    # Phản chiếu cột x (hướng ngón) và y (út→trỏ); z = x × y phải tính lại nên đổi dấu
    return M @ np.asarray(H) @ np.diag([1.0, 1.0, -1.0])
