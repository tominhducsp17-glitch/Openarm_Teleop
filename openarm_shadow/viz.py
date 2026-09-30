"""Vẽ bằng OpenCV: khung xương người trên ảnh camera + hình que robot (nhìn trước và nhìn ngang)."""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
COL = {"right": (255, 140, 0), "left": (0, 140, 255)}   # BGR: phải = xanh dương, trái = cam


def ascii_text(s: str) -> str:
    """cv2.putText chỉ vẽ được ASCII: bỏ dấu tiếng Việt (vd trạng thái SafetyGate) để không hiện '???'."""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return s.encode("ascii", "replace").decode()


def draw_human(img, frame, draw_hand_axes=True):
    h, w = img.shape[:2]
    th = max(2, w // 250)           # nét dày theo kích thước ảnh (1280 px -> 5 px)
    if frame.pose_2d is not None:
        P = frame.pose_2d
        for a, b in POSE_EDGES:
            if min(P[a, 2], P[b, 2]) > 0.3:
                cv2.line(img, (int(P[a, 0] * w), int(P[a, 1] * h)), (int(P[b, 0] * w), int(P[b, 1] * h)),
                         (0, 220, 0), th)
        for i in (11, 12, 13, 14, 15, 16):
            if P[i, 2] > 0.3:
                cv2.circle(img, (int(P[i, 0] * w), int(P[i, 1] * h)), th + 3, (0, 255, 255), -1)
    for h2, side in frame.hands_2d:
        c = COL.get(side, (160, 160, 160))
        for a, b in HAND_EDGES:
            cv2.line(img, (int(h2[a, 0] * w), int(h2[a, 1] * h)), (int(h2[b, 0] * w), int(h2[b, 1] * h)), c,
                     max(1, th - 2))
    # Palm frame metric từ RealSense: x đỏ (hướng ngón), y xanh lá (út->trỏ), z xanh dương (pháp tuyến).
    axis_colors = ((0, 0, 255), (0, 220, 0), (255, 0, 0))
    if draw_hand_axes:
        for ob in frame.arms.values():
            if ob.hand_axes_px is None:
                continue
            origin = tuple(np.round(ob.hand_axes_px[0]).astype(int))
            cv2.circle(img, origin, th + 2, (255, 255, 255), -1)
            for endpoint, color in zip(ob.hand_axes_px[1:], axis_colors):
                cv2.arrowedLine(img, origin, tuple(np.round(endpoint).astype(int)), color,
                                max(2, th - 1), tipLength=0.2)
    return img


def draw_robot(kins, q: dict, size=(480, 480), q_target: dict | None = None, title="",
               q_meas: dict | None = None):
    """Hai hình chiếu: trái = nhìn từ phía trước robot (ngang = y), phải = nhìn từ bên phải (ngang = x).

    q: lệnh (nét đậm màu) · q_target: mục tiêu (nét mảnh xám) · q_meas: góc đo từ robot thật (nét xanh lá)."""
    W, H = size
    canvas = np.full((H, W, 3), 245, np.uint8)
    half = W // 2
    # Tầm với mỗi tay ~0.55 m quanh vai (vai ở y = ±0.15, z = 0.7): mỗi hình chiếu phải chứa y trong ±0.75 m,
    # z trong 0.1..1.3 m, để tay dang ngang hoặc giơ lên đầu không bị cắt.
    scale = min(half / 1.5, H / 1.4)

    def proj(p, view):
        horiz = p[1] if view == 0 else p[0]     # nhìn từ phía trước robot: tay phải robot ở bên trái ảnh
        cx = half // 2 + view * half
        return int(cx + horiz * scale), int(H * 0.45 + (0.7 - p[2]) * scale)

    def draw_frame(origin, R, view, label, length=0.075):
        colors = ((0, 0, 255), (0, 180, 0), (255, 0, 0))  # x đỏ, y xanh lá, z xanh dương
        po = proj(origin, view)
        cv2.circle(canvas, po, 3, (30, 30, 30), -1)
        for j, color in enumerate(colors):
            cv2.arrowedLine(canvas, po, proj(origin + R[:, j] * length, view), color, 2, tipLength=0.22)
        cv2.putText(canvas, label, (po[0] + 4, po[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    for view in (0, 1):
        cv2.putText(canvas, "truoc" if view == 0 else "ben phai", (view * half + 8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        for s, kin in kins.items():
            for qq, thick, col in ((q_target, 1, (180, 180, 180)), (q, 3, COL[s]), (q_meas, 2, (0, 170, 0))):
                if qq is None or s not in qq or not np.all(np.isfinite(qq[s][:7])):
                    continue
                k = kin.display_keypoints(qq[s][:7])
                pts = [kin.p_base, k["shoulder"], k["elbow"], k["wrist"], k["tool"]]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, proj(a, view), proj(b, view), col, thick)
                for p in pts[1:4]:
                    cv2.circle(canvas, proj(p, view), 4 if thick > 1 else 2, col, -1)
            # Chỉ vẽ frame của lệnh hiện tại để không chồng ba bộ target/cmd/measured.
            if q is not None and s in q and np.all(np.isfinite(q[s][:7])):
                qs = q[s][:7]
                k = kin.display_keypoints(qs)
                draw_frame(0.5 * (k["shoulder"] + k["elbow"]), kin.R0(qs, 3), view, "U")
                draw_frame(0.5 * (k["elbow"] + k["wrist"]), kin.R0(qs, 5), view, "F")
                draw_frame(0.5 * (k["wrist"] + k["tool"]), kin.R_tool(qs), view, "H")
    cv2.line(canvas, (half, 0), (half, H), (200, 200, 200), 1)
    if title:
        cv2.putText(canvas, ascii_text(title), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    return canvas


def draw_fused_human_arm(arm, hand_R_body=None, state="", size=(720, 480)):
    """Hai hình chiếu 3D của tay người trong body frame, kèm các frame B/U/F/H."""
    W, H = size
    canvas = np.full((H, W, 3), 245, np.uint8)
    half = W // 2
    scale = min(half / 1.1, H / 1.3)

    def proj(p, view):
        p = np.asarray(p, float)
        horizontal = p[1] if view == 0 else p[0]
        return (int(view * half + half * 0.5 + horizontal * scale),
                int(H * 0.42 - p[2] * scale))

    def direction_frame(v):
        x = np.asarray(v, float)
        n = np.linalg.norm(x)
        if n < 1e-8:
            return np.eye(3)
        x /= n
        ref = np.array([0.0, 0.0, 1.0])
        if abs(x @ ref) > 0.92:
            ref = np.array([0.0, 1.0, 0.0])
        y = np.cross(ref, x); y /= max(np.linalg.norm(y), 1e-9)
        z = np.cross(x, y)
        return np.column_stack((x, y, z))

    def axes(origin, R, view, label, length=.07):
        colors = ((0, 0, 255), (0, 180, 0), (255, 0, 0))
        po = proj(origin, view)
        for j, color in enumerate(colors):
            cv2.arrowedLine(canvas, po, proj(origin + R[:, j] * length, view),
                            color, 2, tipLength=.22)
        cv2.putText(canvas, label, (po[0] + 4, po[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1)

    valid = arm is not None and all(x is not None for x in (arm.s, arm.e, arm.w))
    for view in (0, 1):
        cv2.putText(canvas, "front" if view == 0 else "right", (view * half + 8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 80, 80), 1)
        axes(np.zeros(3), np.eye(3), view, "B")
        if valid:
            points = [np.asarray(arm.s), np.asarray(arm.e), np.asarray(arm.w)]
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(canvas, proj(a, view), proj(b, view), (255, 140, 0), 5)
            for p in points:
                cv2.circle(canvas, proj(p, view), 6, (0, 180, 255), -1)
            axes(.5 * (points[0] + points[1]), direction_frame(points[1] - points[0]), view, "U")
            axes(.5 * (points[1] + points[2]), direction_frame(points[2] - points[1]), view, "F")
            if hand_R_body is not None and np.all(np.isfinite(hand_R_body)):
                axes(points[2], np.asarray(hand_R_body), view, "H", .09)
    title = f"FUSED HUMAN ARM | {state}" if state else "FUSED HUMAN ARM"
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (30, 30, 30), 2, cv2.LINE_AA)
    if not valid:
        cv2.putText(canvas, "waiting for laptop Pose", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 100, 220), 1)
    cv2.line(canvas, (half, 0), (half, H), (200, 200, 200), 1)
    return canvas


def put_lines(img, lines, org=(10, 24), color=(255, 255, 255)):
    x, y = org
    for ln in lines:
        ln = ascii_text(ln)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 22
    return img


def side_by_side(cam, robot):
    h = cam.shape[0]
    r = cv2.resize(robot, (int(robot.shape[1] * h / robot.shape[0]), h))
    return np.hstack([cam, r])
