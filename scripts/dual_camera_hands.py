#!/usr/bin/env python3
"""Debug Lượt 3-4: MediaPipe hai camera + triangulation tay phải, chưa chạy robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_shadow.app import Controller
from openarm_shadow.config import load_config
from openarm_shadow.multicam import AsyncCapture
from openarm_shadow.multiview import (AsyncPerception, best_hand, closest_to_primary, load_stereo_calibration,
                                      observation_confidence, OrientationFusion,
                                      RobustHandFusion, triangulate_hand)
from openarm_shadow.perception import ArmObs, Frame, palm_frame_from_depth
from openarm_shadow.pipeline import ShadowPipeline
from openarm_shadow.robot import make_robot
from openarm_shadow.safety import SafetyGate
from openarm_shadow.sources import OpenCVSource, RealSenseSource
from openarm_shadow.viz import draw_fused_human_arm, draw_human


def put(image, lines, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 1, cv2.LINE_AA)


def draw_axes(image, center, R, K, dist, length=.08):
    points = np.vstack((center, center + R[:, 0]*length,
                        center + R[:, 1]*length, center + R[:, 2]*length))
    uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, dist)
    uv = np.round(uv.reshape(-1, 2)).astype(int)
    colors = ((0, 0, 255), (0, 220, 0), (255, 0, 0))
    for endpoint, color in zip(uv[1:], colors):
            cv2.arrowedLine(image, tuple(uv[0]), tuple(endpoint), color, 3, tipLength=.2)


def draw_reprojection_points(image, points, radius, color):
    """Vẽ diagnostics an toàn; nghiệm DLT suy biến có thể chiếu ra vô cực."""
    h, w = image.shape[:2]
    for point in np.asarray(points, float).reshape(-1, 2):
        if not np.all(np.isfinite(point)):
            continue
        x, y = float(point[0]), float(point[1])
        if -w <= x <= 2 * w and -h <= y <= 2 * h:
            cv2.circle(image, (int(round(x)), int(round(y))), radius, color, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=["config/fusion_d435i_laptop.yaml"])
    ap.add_argument("--log", default=None, help="JSONL observations")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--robot", choices=["openarm"], default=None,
                    help="gửi fused pose tới tay phải OpenArm thật")
    args = ap.parse_args(); cfg = load_config(args.config); fc = cfg["fusion"]
    calibration = load_stereo_calibration(fc["calibration_file"])
    refiner = RobustHandFusion(calibration, fc.get("robust"))
    use_robust = bool(fc.get("robust", {}).get("realtime_enabled", False))
    orientation_fusion = OrientationFusion(fc.get("orientation_fusion"))
    cap_d = AsyncCapture(fc["primary_id"], RealSenseSource(cfg)).start()
    cap_l = AsyncCapture(fc["laptop_id"], OpenCVSource(
        fc["laptop_index"], fc["laptop_width"], fc["laptop_height"], fc.get("laptop_fps", 30))).start()
    # Laptop chính diện giữ Pose + định danh tay phải. D435 bên phải chỉ chạy
    # một HandLandmarker + depth; không để Pose D435 chập chờn xoá bàn tay.
    det_d = AsyncPerception(cap_d, cfg, orientation_source="depth_required",
                            pose_enabled=False, force_hand_side="right", num_hands=1).start()
    det_l = AsyncPerception(cap_l, cfg, orientation_source="rgb_world",
                            pose_enabled=True, force_hand_side="right", num_hands=1).start()
    log = open(args.log, "w") if args.log else None
    start = time.monotonic(); last_pair = (-1, -1)
    last_refined_ids, refined = (-1, -1), None
    last_orientation_ids = (-1, -1)
    orientation = None
    orientation_center_d = None
    Kd = np.asarray(calibration["d435i"]["K"], float)
    dd = np.asarray(calibration["d435i"]["distortion"], float)
    Kl = np.asarray(calibration["laptop"]["K"], float)
    dl = np.asarray(calibration["laptop"]["distortion"], float)
    R_d_to_l = np.asarray(calibration["T_laptop_from_d435i"]["R"], float)
    t_d_to_l = np.asarray(calibration["T_laptop_from_d435i"]["t_m"], float)
    R_l_to_d = np.asarray(calibration["T_d435i_from_laptop"]["R"], float)
    pipe = robot = gate = ctl = None
    if args.robot == "openarm":
        cfg["mapping"]["robot_arms"] = ["right"]
        pipe = ShadowPipeline(cfg)
        robot = make_robot("openarm", cfg, pipe.robot_sides)
        q_meas = robot.connect()
        print("Tư thế robot phải (URDF, độ):", np.round(np.rad2deg(q_meas["right"][:7]), 1))
        gate = SafetyGate(pipe.kins, cfg["safety"])
        gate.reset(q_meas)
        pipe.seed(q_meas)
        print("ROBOT THẬT: gõ yes để bật motor; sau đó giữ zero-pose để auto calib, SPACE để engage.")
        if input("Bật motor? ").strip().lower() != "yes":
            robot.close()
            raise SystemExit("Huỷ bật robot")
        robot.enable()
        ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
        ctl.start()
    cv2.namedWindow("dual_camera_hands", cv2.WINDOW_NORMAL)
    cv2.namedWindow("fused_arm_3d", cv2.WINDOW_NORMAL)
    try:
        while True:
            # Laptop là nhịp tracking chính; chọn frame D435 gần timestamp laptop nhất.
            b, a, pair_delta = closest_to_primary(
                det_l, det_d, float(fc.get("pair_history_window_s", 0.12)))
            if det_d.error or det_l.error:
                raise RuntimeError(f"inference: D435i={det_d.error!r}, laptop={det_l.error!r}")
            if a is None or b is None:
                time.sleep(.005); continue
            hd, hl = best_hand(a.frame), best_hand(b.frame)
            cd, cl = observation_confidence(hd), observation_confidence(hl)
            dt_ms = 1000.0 * pair_delta
            triangulated = None
            ids = (a.timed_frame.frame_id, b.timed_frame.frame_id)
            if hd is not None and hl is not None and dt_ms <= float(fc.get("max_pair_delta_ms", 50)):
                h_d, w_d = a.timed_frame.sample.bgr.shape[:2]
                h_l, w_l = b.timed_frame.sample.bgr.shape[:2]
                pd = hd.landmarks * [w_d, h_d]; pl = hl.landmarks * [w_l, h_l]
                sync = np.exp(-.5 * (dt_ms / 25.0) ** 2)
                solve_started = time.monotonic()
                triangulated = triangulate_hand(
                    calibration, pd, pl, max(cd * sync, .05), max(cl * sync, .05))
                if ids != last_refined_ids:
                    arm = a.frame.arms["right"]
                    if use_robust:
                        refined = refiner.refine(
                            triangulated["points_d435i"], pd, pl,
                            depth_points=arm.hand_points_cam,
                            depth_confidence=arm.hand_point_confidence,
                            timestamp=a.timed_frame.host_timestamp_s,
                            confidence_d=cd, confidence_l=cl, sync_confidence=sync)
                    else:
                        err_d = triangulated["error_d435i_px"]
                        err_l = triangulated["error_laptop_px"]
                        per_point = np.maximum(err_d, err_l)
                        point_conf = (np.sqrt(max(cd * cl, 0.0)) * sync *
                                      np.exp(-0.5 * (per_point / 8.0) ** 2))
                        refined = {
                            **triangulated,
                            "mode": "WEIGHTED_DLT_REALTIME", "success": True,
                            "converged": True, "cost": 0.0, "nfev": 0,
                            "solve_ms": 1000.0 * (time.monotonic() - solve_started),
                            "reprojection_median_px": float(np.median(np.r_[err_d, err_l])),
                            "reprojection_p95_px": float(np.percentile(np.r_[err_d, err_l], 95)),
                            "depth_median_m": float("nan"), "bone_median_m": float("nan"),
                            "depth_points": int(np.count_nonzero(arm.hand_point_confidence > 0.02))
                            if arm.hand_point_confidence is not None else 0,
                            "point_confidence": np.clip(point_conf, 0.0, 1.0),
                            "confidence_median": float(np.median(point_conf)),
                            "rejected_points": int(np.count_nonzero(point_conf < 0.20)),
                        }
                    fused_R, center = palm_frame_from_depth(refined["points_d435i"], side="right")
                    palm_conf = float(np.median(refined["point_confidence"][[0, 5, 9, 17]]))
                    depth_R = arm.hand_R_cam
                    laptop_local_R = b.frame.arms["right"].hand_R_cam
                    laptop_R = None if laptop_local_R is None else R_l_to_d @ laptop_local_R
                    orientation = orientation_fusion.update(
                        fused_R, depth_R=depth_R, laptop_R=laptop_R,
                        fused_confidence=palm_conf,
                        depth_confidence=arm.hand_depth_confidence,
                        laptop_confidence=cl)
                    orientation_center_d = np.mean(
                        refined["points_d435i"][[0, 5, 9, 17]], axis=0)
                    last_refined_ids = ids
                    last_orientation_ids = ids
            if triangulated is None and ids != last_orientation_ids:
                orientation = orientation_fusion.update(None)
                last_orientation_ids = ids
            # Chỉ vẽ fused frame bên dưới; trục local từng camera trước đây làm
            # người dùng thấy hai hệ trục chồng nhau như một frame bị duplicate.
            left = draw_human(a.timed_frame.sample.bgr.copy(), a.frame, draw_hand_axes=False)
            right = draw_human(b.timed_frame.sample.bgr.copy(), b.frame, draw_hand_axes=False)
            if triangulated is not None and refined is not None:
                draw_reprojection_points(left, triangulated["reprojection_d435i_px"],
                                         2, (160, 160, 160))
                draw_reprojection_points(right, triangulated["reprojection_laptop_px"],
                                         2, (160, 160, 160))
                draw_reprojection_points(left, refined["reprojection_d435i_px"],
                                         3, (0, 255, 0))
                draw_reprojection_points(right, refined["reprojection_laptop_px"],
                                         3, (0, 255, 0))
                mode = (f"{refined['mode']} reproj med/p95 "
                        f"{refined['reprojection_median_px']:.2f}/{refined['reprojection_p95_px']:.2f}px")
                angle = (f"depth {refined['depth_points']}/21 {1000*refined['depth_median_m']:.1f}mm | "
                         f"solve {refined['solve_ms']:.0f}ms")
            else:
                state = (orientation or {}).get("state", "NONE")
                mode, angle = f"{state}: missing hand or bad sync", ""
            if (orientation is not None and orientation["R"] is not None and
                    orientation_center_d is not None):
                draw_axes(left, orientation_center_d, orientation["R"], Kd, dd)
                center_l = R_d_to_l @ orientation_center_d + t_d_to_l
                draw_axes(right, center_l, R_d_to_l @ orientation["R"], Kl, dl)
                angle += (f" | {orientation['state']} c={orientation['confidence']:.2f} "
                          f"b={orientation['branch']}")
            td, tl = a.frame.tracking, b.frame.tracking
            put(left, [f"D435i cap {cap_d.stats['fps']:.1f}fps conf {cd:.2f} infer {a.inference_ms:.0f}ms",
                       f"detect pose={td.get('pose')} hand={td.get('raw_hands', 0)} assoc={td.get('associated_hands', 0)}",
                       mode, angle])
            put(right, [f"Laptop cap {cap_l.stats['fps']:.1f}fps conf {cl:.2f} infer {b.inference_ms:.0f}ms",
                        f"detect pose={tl.get('pose')} hand={tl.get('raw_hands', 0)} assoc={tl.get('associated_hands', 0)}",
                        f"pair dt {dt_ms:.1f}ms"])
            canvas = np.hstack((left, right))
            cv2.imshow("dual_camera_hands", canvas)
            arm_laptop = b.frame.arms["right"]
            hand_R_body = None
            if (orientation is not None and orientation["R"] is not None and
                    b.frame.body_R is not None):
                hand_R_body = b.frame.body_R.T @ R_d_to_l @ orientation["R"]
            sim = draw_fused_human_arm(
                arm_laptop, hand_R_body,
                state=(orientation or {}).get("state", "NONE"))
            cv2.imshow("fused_arm_3d", sim)

            targets = None
            if pipe is not None:
                src = arm_laptop
                ostate = (orientation or {}).get("state", "NONE")
                hand_conf = 1.0 if ostate == "TRACKING" else (0.7 if ostate == "DEGRADED" else 0.0)
                hand_R_laptop = (None if orientation is None or orientation["R"] is None
                                 else R_d_to_l @ orientation["R"])
                fused_ob = ArmObs(
                    s=src.s, e=src.e, w=src.w, H=hand_R_body, grip=src.grip,
                    conf={"upper": src.conf["upper"], "fore": src.conf["fore"],
                          "hand": hand_conf},
                    hand_R_cam=hand_R_laptop,
                    hand_open_fingers=src.hand_open_fingers,
                    hand_orientation_mode=ostate,
                )
                fused_frame = Frame(
                    {"right": fused_ob, "left": ArmObs()}, b.frame.pose_2d,
                    b.frame.hands_2d, b.frame.body_R, b.frame.t,
                    body_origin=b.frame.body_origin, tracking={
                        "fusion_state": ostate})
                if not gate.engaged:
                    completed = pipe.auto_calibrate_hand_neutral(fused_frame)
                    if completed:
                        print("Auto calib fused tay trung tính:", completed)
                targets = pipe.step(fused_frame)
                with ctl.lock:
                    gate.set_target(targets, time.monotonic())
                    cmd = {s: v.copy() for s, v in gate.cmd.items()}
                    robot_status = gate.status
                cv2.putText(sim, f"robot: {robot_status} | calib: " +
                            ("OK" if pipe.hand_calibrated["right"] else
                             f"{100*pipe.calib_progress['right']:.0f}% {pipe.calib_hint['right']}"),
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .52, (30, 30, 30), 1,
                            cv2.LINE_AA)
                cv2.imshow("fused_arm_3d", sim)
            if log is not None and ids != last_pair:
                row = {"t": time.monotonic(), "d435i_frame": ids[0], "laptop_frame": ids[1],
                       "dt_ms": dt_ms, "confidence": {"d435i": cd, "laptop": cl},
                       "tracking": {"d435i": td, "laptop": tl},
                       "right_d435i": None if hd is None else hd.landmarks.tolist(),
                       "right_laptop": None if hl is None else hl.landmarks.tolist(),
                       "triangulated_d435i": None if triangulated is None else
                       triangulated["points_d435i"].tolist(),
                       "points_d435i": None if triangulated is None or refined is None else
                       refined["points_d435i"].tolist(),
                       "fusion": None if triangulated is None or refined is None else {
                           k: (v.item() if isinstance(v, np.generic) else v)
                           for k, v in refined.items()
                           if np.ndim(v) == 0},
                       "orientation": None if orientation is None else {
                           k: v for k, v in orientation.items() if k != "R"}}
                log.write(json.dumps(row) + "\n"); log.flush(); last_pair = ids
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and gate is not None:
                with ctl.lock:
                    if gate.engaged:
                        gate.disengage()
                        print("Robot HOLD")
                    elif not pipe.hand_calibrated["right"]:
                        print("Chưa engage: cần auto calib tay phải đạt OK")
                    else:
                        pipe.seed(robot.read())
                        gate.engage(time.monotonic())
                        print("Robot ENGAGED")
            elif key == ord("c") and pipe is not None:
                done = pipe.calibrate_hand_neutral(fused_frame)
                print("Calib tay trung tính:", done or "chưa đủ pose")
            elif key in (ord("q"), 27):
                break
            if args.seconds and time.monotonic() - start >= args.seconds: break
    finally:
        if ctl is not None:
            ctl.running = False
            ctl.join(timeout=1.0)
        if robot is not None:
            robot.close()
        det_d.close(); det_l.close(); cap_d.close(); cap_l.close()
        if log is not None: log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
