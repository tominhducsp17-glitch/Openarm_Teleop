"""Vòng chạy chính: camera -> pipeline -> SafetyGate -> robot (sim hoặc OpenArm thật).

Hai luồng:
- luồng chính: đọc camera, MediaPipe, retarget, lọc, đặt mục tiêu cho SafetyGate, vẽ.
- luồng điều khiển: chạy đều `control_hz`, bước SafetyGate (giới hạn vận tốc, dead-man, va chạm)
  rồi gửi lệnh xuống robot.

Phím: SPACE = engage / nhả (ly hợp) · c = hiệu chuẩn hướng bàn tay trung tính · p = về tư thế nghỉ
      q hoặc ESC = về tư thế nghỉ rồi thoát.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .perception import Perception
from .pipeline import ShadowPipeline
from .robot import make_robot
from .safety import SafetyGate
from .sources import open_source
from .viz import draw_human, draw_robot, put_lines, side_by_side


class Controller(threading.Thread):
    def __init__(self, robot, gate, hz):
        super().__init__(daemon=True)
        self.robot, self.gate, self.dt = robot, gate, 1.0 / hz
        self.lock = threading.Lock()
        self.running = True
        self.error = None
        self.cmd = None

    def run(self):
        t_prev = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                with self.lock:
                    cmd = self.gate.step(now - t_prev, now)
                    self.cmd = {s: v.copy() for s, v in cmd.items()}
                t_prev = now
                self.robot.send(self.cmd)
                time.sleep(max(0.0, self.dt - (time.monotonic() - now)))
        except Exception as e:     # lỗi phần cứng: dừng vòng điều khiển, luồng chính sẽ thoát an toàn
            self.error = e
            self.running = False


def park(robot, gate, rest, vel_deg_s, timeout=25.0):
    """Đưa hai tay về tư thế nghỉ với tốc độ thấp (chạy sau khi đã dừng luồng điều khiển)."""
    saved = gate.max_vel.copy()
    gate.max_vel = np.full(7, np.deg2rad(vel_deg_s))
    gate.engage(time.monotonic() - 10)     # bỏ qua pha tăng tốc
    if hasattr(robot, "returning"):
        robot.returning = True             # đang về: số đọc rác chỉ bị bỏ qua, không dừng giữa chừng
    try:
        t0 = t_prev = time.monotonic()
        while time.monotonic() - t0 < timeout:
            now = time.monotonic()
            tgt = {s: np.append(rest, np.nan) for s in gate.sides}
            gate.set_target(tgt, now)
            cmd = gate.step(now - t_prev, now)
            t_prev = now
            robot.send(cmd)
            if max(np.max(np.abs(cmd[s][:7] - rest)) for s in gate.sides) < np.deg2rad(1.0):
                break
            time.sleep(0.01)
    finally:
        if hasattr(robot, "returning"):
            robot.returning = False
        gate.max_vel = saved
        gate.disengage()


def run(cfg, source, robot_kind="sim", record=None, show=True, dry_run=False):
    """dry_run (chỉ với robot_kind="openarm"): đọc góc robot thật, KHÔNG bật motor. Lệnh đi vào robot mô phỏng;
    hình vẽ có thêm nét xanh lá = tư thế đo từ robot thật. Dùng để kiểm tra can0/can1 và chiều từng khớp
    bằng cách cầm tay robot di chuyển, trước khi chạy thật."""
    cap = open_source(source, cfg)
    perc = Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"],
                      depth_cfg=cfg["camera"].get("realsense"), orientation_cfg=cfg.get("orientation"))
    pipe = ShadowPipeline(cfg)
    real = None
    if robot_kind == "openarm" and dry_run:
        real = make_robot("openarm", cfg, pipe.robot_sides)
        q_meas = real.connect()
        from .robot.sim import SimRobot
        robot = SimRobot(pipe.robot_sides, q0=q_meas)
        robot_kind = "sim"
    else:
        robot = make_robot(robot_kind, cfg, pipe.robot_sides)
        q_meas = robot.connect()
    print("Tư thế đo được (độ, URDF):")
    for s, q in q_meas.items():
        print(f"  {s:5s}", np.round(np.rad2deg(q[:7]), 1), " kẹp", round(float(q[7]), 2))
    gate = SafetyGate(pipe.kins, cfg["safety"])
    gate.reset(q_meas)
    pipe.seed(q_meas)

    if robot_kind == "openarm":
        print("\nROBOT THẬT. Kiểm tra: E-stop trong tay, không ai trong tầm với, tay đang thả xuôi.")
        if input("Gõ 'yes' để bật motor: ").strip().lower() != "yes":
            robot.close()
            raise SystemExit("Huỷ.")
        robot.enable()

    ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
    ctl.start()
    rest = np.deg2rad(np.asarray(cfg["robot"]["rest_pose_deg"], float))
    log = {"t": [], **{f"target_{s}": [] for s in pipe.robot_sides}, **{f"cmd_{s}": [] for s in pipe.robot_sides}}
    fps_t, fps = time.monotonic(), 0.0
    auto_engage_s = float(cfg.get("calibration", {}).get("hand_auto", {}).get("auto_engage_sim_s", 3.0))
    ready_since = None
    auto_engage_used = False
    auto_countdown = None
    msg = ("GIU READY 3s: tu dong sync | SPACE: dung/chay thu cong | c: calib lai | q: thoat"
           if robot_kind == "sim" else
           "SPACE: engage | c: hieu chuan tay | p: ve nghi | q: thoat")
    if real is not None:
        msg = "DRY RUN: motor TAT. Xanh la = robot that. " + msg
    if show:
        cv2.namedWindow("openarm_shadow", cv2.WINDOW_NORMAL)   # kéo giãn được cửa sổ
    try:
        while ctl.running:
            ok, sample = cap.read()
            if not ok:
                break
            frame_bgr = sample.bgr
            fr = perc.process(frame_bgr, depth_m=sample.depth_m, depth_intrinsics=sample.intrinsics)
            with ctl.lock:
                engaged = gate.engaged
            auto_done = [] if engaged else pipe.auto_calibrate_hand_neutral(fr)
            if auto_done:
                print("Tự động hiệu chuẩn tay trung tính cho:", auto_done)
            ready_live = all(pipe.hand_calibrated[s] and pipe.calib_ready_now[s] for s in pipe.robot_sides)
            now = time.monotonic()
            if robot_kind == "sim" and not engaged and not auto_engage_used:
                if ready_live:
                    ready_since = now if ready_since is None else ready_since
                    auto_countdown = max(0.0, auto_engage_s - (now - ready_since))
                    if auto_countdown <= 0.0:
                        q_now = robot.read()
                        pipe.seed(q_now)
                        with ctl.lock:
                            gate.engage(now)
                        engaged = True
                        auto_engage_used = True
                        auto_countdown = None
                        print("Simulation tự đồng bộ sau khi READY đủ", auto_engage_s, "giây")
                else:
                    ready_since = None
                    auto_countdown = None
            targets = pipe.step(fr)
            with ctl.lock:
                gate.set_target(targets, time.monotonic())
                cmd = {s: v.copy() for s, v in gate.cmd.items()}
                status = gate.status
            if record:
                log["t"].append(fr.t)
                for s in pipe.robot_sides:
                    log[f"target_{s}"].append(targets[s])
                    log[f"cmd_{s}"].append(cmd[s])
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
            fps_t = now
            if show:
                cam = draw_human(frame_bgr.copy(), fr)
                if cfg["camera"]["mirror_display"]:
                    cam = cv2.flip(cam, 1)
                ready = ready_live
                cx, cy = cam.shape[1] - 28, 28
                if ready and not engaged:
                    cv2.circle(cam, (cx, cy), 15, (0, 220, 0), -1)
                    ready_text = (f"AUTO SYNC {auto_countdown:.1f}s" if auto_countdown is not None
                                  else "READY")
                    cv2.putText(cam, ready_text, (max(8, cx - 175), cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                elif ready:
                    cv2.circle(cam, (cx, cy), 15, (255, 200, 0), -1)
                    cv2.putText(cam, "FOLLOW", (max(8, cx - 90), cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                else:
                    cv2.circle(cam, (cx, cy), 15, (0, 180, 255), 2)
                    cv2.putText(cam, "CALIB: ARM DOWN + PALM TO CAM", (max(8, cx - 285), cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)
                lines = [f"{fps:4.1f} fps | {status}", msg]
                if sample.depth_m is not None:
                    ds = " ".join(f"{s}:{fr.depth_used.get(s, 0)}/3" for s in pipe.robot_sides)
                    lines.append("RealSense depth vai/khuyu/co tay " + ds + " (3/3 = dang dung depth)")
                    for human_side, di in fr.hand_depth.items():
                        lines.append(f"{human_side} hand: {di['mode']} depth {di['direct']}/21 "
                                     f"fused {di['fused']}/21 conf {di['confidence']:.2f}")
                        rms = di.get("plane_rms_m", float("inf"))
                        rms_text = f"{1000*rms:.1f}mm" if np.isfinite(rms) else "--"
                        lines.append(f"  orient {di.get('orientation', 'NONE')} open {di.get('open_fingers', 0)}/4 "
                                     f"plane {di.get('plane_inliers', 0)} rms {rms_text}")
                elif cfg.get("orientation", {}).get("source") == "rgb_world":
                    for human_side, ob in fr.arms.items():
                        if ob.hand_orientation_mode != "NONE":
                            lines.append(f"{human_side} hand: RGB_ONLY orient {ob.hand_orientation_mode} "
                                         f"open {ob.hand_open_fingers}/4")
                cs = " ".join(f"{s}:OK" if pipe.hand_calibrated[s] else
                              f"{s}:{100 * pipe.calib_progress[s]:.0f}% [{pipe.calib_hint[s]}]"
                              for s in pipe.robot_sides)
                lines.append("Auto calib tay: " + cs)
                for s in pipe.robot_sides:
                    inf = pipe.last_info.get(s)
                    if inf is not None:
                        lines.append(f"{s}: err u {inf.err_upper_deg:5.1f} l {inf.err_fore_deg:5.1f} "
                                     f"tay {inf.err_hand_deg:5.1f} deg" + (" [thang]" if inf.elbow_straight else ""))
                    if s in targets and np.all(np.isfinite(targets[s][4:7])):
                        wt = np.rad2deg(targets[s][4:7])
                        wc = np.rad2deg(cmd[s][4:7])
                        lines.append(f"{s} J5-7 target {wt[0]:5.1f} {wt[1]:5.1f} {wt[2]:5.1f} | "
                                     f"cmd {wc[0]:5.1f} {wc[1]:5.1f} {wc[2]:5.1f}")
                q_real = None
                if real is not None:
                    q_real = real.poll()
                    for s in pipe.robot_sides:
                        lines.append(f"{s} that (URDF, do): " +
                                     " ".join(f"{v:5.0f}" for v in np.rad2deg(q_real[s][:7])))
                elif robot_kind == "openarm":
                    q_real = robot.read()
                put_lines(cam, lines)
                rob = draw_robot(pipe.kins, cmd, q_target=targets, q_meas=q_real,
                                 title="lenh (dam) / muc tieu (mo)" + (" / do that (xanh la)" if q_real else ""))
                cv2.imshow("openarm_shadow", side_by_side(cam, rob))
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    auto_engage_used = True
                    ready_since = None
                    with ctl.lock:
                        if gate.engaged:
                            gate.disengage()
                        else:
                            q_now = robot.read()
                            pipe.seed(q_now)
                            gate.engage(time.monotonic())
                elif k == ord("c"):
                    print("Hiệu chuẩn tay trung tính cho:", pipe.calibrate_hand_neutral(fr) or "không thấy bàn tay")
                elif k == ord("p"):
                    with ctl.lock:
                        gate.disengage()
                    ctl.running = False
                    ctl.join()
                    park(robot, gate, rest, cfg["robot"]["park_vel_deg_s"])
                    ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
                    ctl.start()
                elif k in (ord("q"), 27):
                    break
    finally:
        ctl.running = False
        ctl.join(timeout=1.0)
        if ctl.error:
            print("LỖI vòng điều khiển:", ctl.error)
        try:
            if robot_kind == "openarm" and ctl.error is None:
                print("Về tư thế nghỉ...")
                park(robot, gate, rest, cfg["robot"]["park_vel_deg_s"])
        finally:
            robot.close()
            if real is not None:
                real.close()
            perc.close()
            cap.close()
            cv2.destroyAllWindows()
            if record and log["t"]:
                np.savez(record, **{k: np.asarray(v) for k, v in log.items()})
                print("Đã lưu", record)
