#!/usr/bin/env python3
"""Tạo bảng và calibration ChArUco cho D435i RGB + camera laptop RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_shadow.camera_calibration import (calibrate_laptop, camera_matrix_from_realsense,
    common_charuco, detect_charuco, make_charuco_board, stereo_calibrate_fixed)
from openarm_shadow.config import load_config
from openarm_shadow.multicam import AsyncCapture, latest_pair
from openarm_shadow.sources import OpenCVSource, RealSenseSource


def board_config(cfg):
    return cfg["fusion"]["calibration"]["charuco"]


def generate_board(cfg, output):
    bc = board_config(cfg)
    board = make_charuco_board(bc)
    dpi = int(bc.get("print_dpi", 300))
    page_w_mm, page_h_mm = 297.0, 210.0  # A4 landscape
    px_per_mm = dpi / 25.4
    page_size = (int(round(page_w_mm * px_per_mm)), int(round(page_h_mm * px_per_mm)))
    board_w_mm = 1000.0 * bc["squares_x"] * bc["square_length_m"]
    board_h_mm = 1000.0 * bc["squares_y"] * bc["square_length_m"]
    board_size = (int(round(board_w_mm * px_per_mm)), int(round(board_h_mm * px_per_mm)))
    if board_size[0] > page_size[0] or board_size[1] > page_size[1]:
        raise SystemExit("Board khong vua trang A4; giam squares hoac square_length_m")
    image = np.full((page_size[1], page_size[0]), 255, np.uint8)
    rendered = board.generateImage(board_size, marginSize=0, borderBits=1)
    x0, y0 = (page_size[0] - board_size[0]) // 2, (page_size[1] - board_size[1]) // 2
    image[y0:y0 + board_size[1], x0:x0 + board_size[0]] = rendered
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise SystemExit(f"Khong ghi duoc {output}")
    meta = {
        "paper": "A4 landscape", "dpi": dpi, "print_scale_percent": 100,
        "outer_board_width_mm": board_w_mm, "outer_board_height_mm": board_h_mm,
        **bc,
    }
    output.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Da tao {output}")
    print(f"In A4 ngang, scale 100%/Actual size; do lai vien board = {board_w_mm:.0f} x {board_h_mm:.0f} mm")


def draw_detection(image, corners, ids, label, color):
    out = image.copy()
    if len(ids):
        cv2.aruco.drawDetectedCornersCharuco(out, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), color)
    cv2.putText(out, f"{label}: {len(ids)} corners", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, f"{label}: {len(ids)} corners", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 1, cv2.LINE_AA)
    return out


def observation_novel(previous, corners, ids, threshold_px):
    if previous is None:
        return True
    prev_corners, prev_ids = previous
    a = {int(i): p for i, p in zip(prev_ids, prev_corners)}
    b = {int(i): p for i, p in zip(ids, corners)}
    common = set(a) & set(b)
    if len(common) < 6:
        return True
    rms = np.sqrt(np.mean([np.sum((a[i] - b[i]) ** 2) for i in common]))
    return rms >= threshold_px


def capture(cfg, session_dir):
    fc, cc = cfg["fusion"], cfg["fusion"]["calibration"]
    board = make_charuco_board(cc["charuco"])
    detector = cv2.aruco.CharucoDetector(board)
    root = Path(session_dir)
    (root / "d435i").mkdir(parents=True, exist_ok=True)
    (root / "laptop").mkdir(parents=True, exist_ok=True)
    primary = AsyncCapture(fc["primary_id"], RealSenseSource(cfg)).start()
    laptop = AsyncCapture(fc["laptop_id"], OpenCVSource(
        fc["laptop_index"], fc["laptop_width"], fc["laptop_height"], fc.get("laptop_fps", 30))).start()
    target = int(cc.get("target_pairs", 30))
    min_common = int(cc.get("min_common_corners", 10))
    interval = float(cc.get("capture_interval_s", 0.55))
    novelty = float(cc.get("novelty_px", 12))
    max_dt = float(fc.get("max_pair_delta_ms", 50))
    records, last_ids, last_obs, last_save = [], (-1, -1), None, 0.0
    intr = None
    cv2.namedWindow("charuco_calibration", cv2.WINDOW_NORMAL)
    try:
        while len(records) < target:
            d, l, delta = latest_pair(primary, laptop)
            if d is None or l is None:
                time.sleep(0.005); continue
            cd, idd, _, _ = detect_charuco(detector, d.sample.bgr)
            cl, idl, _, _ = detect_charuco(detector, l.sample.bgr)
            common = common_charuco(board, cd, idd, cl, idl, min_common=min_common)
            good = common is not None and 1000.0 * delta <= max_dt
            now = time.monotonic()
            fresh = d.frame_id != last_ids[0] and l.frame_id != last_ids[1]
            changed = observation_novel(last_obs, cl, idl, novelty)
            if good and fresh and changed and now - last_save >= interval:
                idx = len(records)
                dn, ln = f"d435i/{idx:03d}.png", f"laptop/{idx:03d}.png"
                cv2.imwrite(str(root / dn), d.sample.bgr)
                cv2.imwrite(str(root / ln), l.sample.bgr)
                records.append({"index": idx, "d435i": dn, "laptop": ln,
                                "d435i_frame": d.frame_id, "laptop_frame": l.frame_id,
                                "pair_delta_ms": 1000.0 * delta,
                                "common_corners": int(len(common[3]))})
                last_ids, last_obs, last_save = (d.frame_id, l.frame_id), (cl.copy(), idl.copy()), now
                intr = d.sample.intrinsics
            left = draw_detection(d.sample.bgr, cd, idd, "D435i", (0, 255, 0) if good else (0, 165, 255))
            right = draw_detection(l.sample.bgr, cl, idl, "Laptop", (0, 255, 0) if good else (0, 165, 255))
            status = f"AUTO CAPTURE {len(records)}/{target} | common {0 if common is None else len(common[3])} | dt {1000*delta:.1f}ms"
            canvas = np.hstack((left, right))
            cv2.putText(canvas, status, (10, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, status, (10, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 255, 0) if good else (0, 165, 255), 1, cv2.LINE_AA)
            cv2.imshow("charuco_calibration", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        primary.close(); laptop.close(); cv2.destroyAllWindows()
    if intr is None or len(records) < 6:
        raise SystemExit(f"Chi thu duoc {len(records)} cap hop le; can it nhat 6")
    metadata = {
        "records": records,
        "image_size": [int(cfg["camera"]["realsense"]["width"]), int(cfg["camera"]["realsense"]["height"])],
        "d435i": {
            "serial": cfg["camera"]["realsense"].get("serial"),
            "K": camera_matrix_from_realsense(intr).tolist(),
            "distortion": list(intr.coeffs), "distortion_model": str(intr.model),
        },
    }
    (root / "session.json").write_text(json.dumps(metadata, indent=2))
    print(f"Da thu {len(records)} cap vao {root}")
    return root


def solve(cfg, session_dir, output):
    root = Path(session_dir)
    meta = json.loads((root / "session.json").read_text())
    board = make_charuco_board(board_config(cfg)); detector = cv2.aruco.CharucoDetector(board)
    laptop_obs, obj, img_d, img_l, used = [], [], [], [], []
    for rec in meta["records"]:
        d = cv2.imread(str(root / rec["d435i"])); l = cv2.imread(str(root / rec["laptop"]))
        cd, idd, _, _ = detect_charuco(detector, d)
        cl, idl, _, _ = detect_charuco(detector, l)
        laptop_obs.append((cl, idl))
        common = common_charuco(board, cd, idd, cl, idl,
                                int(cfg["fusion"]["calibration"].get("min_common_corners", 10)))
        if common is not None:
            o, a, b, _ = common; obj.append(o); img_d.append(a); img_l.append(b); used.append(rec["index"])
    size = tuple(meta["image_size"])
    laptop_rms, K_l, dist_l, _, _ = calibrate_laptop(board, laptop_obs, size)
    K_d = np.asarray(meta["d435i"]["K"], float); dist_d = np.asarray(meta["d435i"]["distortion"], float)
    stereo = stereo_calibrate_fixed(obj, img_d, img_l, size, K_d, dist_d, K_l, dist_l)
    baseline = float(np.linalg.norm(stereo["t_laptop_to_d435i_m"]))
    limits = cfg["fusion"]["calibration"].get("quality", {})
    reasons = []
    if laptop_rms > float(limits.get("max_laptop_rms_px", 1.2)):
        reasons.append("laptop intrinsic RMS qua cao")
    if stereo["rms"] > float(limits.get("max_stereo_rms_px", 1.5)):
        reasons.append("stereo RMS qua cao")
    if stereo["epipolar_p95_px"] > float(limits.get("max_epipolar_p95_px", 2.0)):
        reasons.append("epipolar p95 qua cao")
    if not (float(limits.get("min_baseline_m", 0.05)) <= baseline <=
            float(limits.get("max_baseline_m", 2.0))):
        reasons.append("baseline khong hop ly")
    accepted = not reasons
    result = {
        "schema_version": 1,
        "reference_frame": "d435i_color_optical_frame",
        "image_size": list(size),
        "board": board_config(cfg),
        "d435i": {**meta["d435i"], "resolution": list(size)},
        "laptop": {"index": cfg["fusion"]["laptop_index"], "resolution": list(size),
                   "K": K_l.tolist(), "distortion": dist_l.tolist(),
                   "intrinsic_rms_px": laptop_rms},
        "T_d435i_from_laptop": {
            "R": stereo["R_laptop_to_d435i"].tolist(),
            "t_m": stereo["t_laptop_to_d435i_m"].tolist(),
        },
        "T_laptop_from_d435i": {
            "R": stereo["R_d435i_to_laptop"].tolist(),
            "t_m": stereo["t_d435i_to_laptop_m"].tolist(),
        },
        "quality": {"pairs_used": len(used), "pair_indices": used,
                    "stereo_rms_px": stereo["rms"],
                    "epipolar_median_px": stereo["epipolar_median_px"],
                    "epipolar_p95_px": stereo["epipolar_p95_px"],
                    "baseline_m": baseline, "accepted": accepted,
                    "rejection_reasons": reasons},
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"Da luu calibration: {output}")
    print(f"Laptop intrinsic RMS: {laptop_rms:.3f}px")
    print(f"Stereo RMS: {stereo['rms']:.3f}px | epipolar median/p95: "
          f"{stereo['epipolar_median_px']:.3f}/{stereo['epipolar_p95_px']:.3f}px | baseline {baseline:.3f}m")
    print("CALIBRATION:", "PASS" if accepted else "FAIL - " + "; ".join(reasons))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=["config/fusion_d435i_laptop.yaml"])
    sub = ap.add_subparsers(dest="command", required=True)
    bp = sub.add_parser("board"); bp.add_argument("--output", default="docs/charuco_d435i_laptop_a4.png")
    rp = sub.add_parser("run")
    rp.add_argument("--session", default="captures/calibration_d435i_laptop")
    rp.add_argument("--output", default="config/calibration/d435i_laptop.yaml")
    sp = sub.add_parser("solve")
    sp.add_argument("--session", default="captures/calibration_d435i_laptop")
    sp.add_argument("--output", default="config/calibration/d435i_laptop.yaml")
    args = ap.parse_args(); cfg = load_config(args.config)
    if args.command == "board": generate_board(cfg, args.output)
    elif args.command == "run":
        result = solve(cfg, capture(cfg, args.session), args.output)
        if not result["quality"]["accepted"]:
            raise SystemExit(2)
    else:
        result = solve(cfg, args.session, args.output)
        if not result["quality"]["accepted"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
