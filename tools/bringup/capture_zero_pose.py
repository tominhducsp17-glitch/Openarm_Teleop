#!/usr/bin/env python3
"""Capture zero-pose PHẦN MỀM khi motor tắt; không ghi zero vào motor.

Ví dụ, từ thư mục gốc repo:
  python tools/bringup/capture_zero_pose.py --iface can0 --side right \
    --config config/d455_wrist_real.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openarm_common import link_ok, make_openarm, read_positions
from openarm_shadow.config import load_config
from openarm_shadow.zero_calibration import ZeroCalibrationError, estimate_zero, write_zero_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iface", default="can0")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--config", default="config/d455_wrist_real.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--max-std-deg", type=float, default=0.25)
    ap.add_argument("--max-span-deg", type=float, default=0.8)
    ap.add_argument("--yes", action="store_true", help="bỏ bước gõ ZERO")
    args = ap.parse_args()
    if args.seconds < 1.0 or args.hz < 10:
        raise SystemExit("Cần --seconds >= 1 và --hz >= 10")

    cfg = load_config(args.config)
    sign = cfg["robot"]["urdf_to_motor"][args.side]["sign"]
    output = args.output or cfg["robot"].get("zero_calibration_file")
    if not output:
        raise SystemExit("Profile chưa đặt robot.zero_calibration_file; hãy truyền --output")
    output = Path(output)
    output = output if output.is_absolute() else ROOT / output

    if not args.yes:
        print("Motor phải TẮT. Đưa toàn bộ tay robot về zero-pose cơ khí đã định nghĩa và giữ yên.")
        ans = input("Gõ ZERO để lấy mẫu (không ghi gì vào motor): ")
        if ans.strip() != "ZERO":
            raise SystemExit("Huỷ.")

    arm = make_openarm(args.iface, with_gripper=False)
    # Warm-up: frame đầu thường chưa đủ cả 7 motor.
    deadline = time.monotonic() + 2.0
    while True:
        q = np.asarray(read_positions(arm, with_gripper=False), float)
        ok = link_ok(arm, max_age_s=0.2)
        if np.all(np.isfinite(q)) and (ok is None or all(ok)):
            break
        if time.monotonic() >= deadline:
            raise SystemExit(f"Không đọc được đủ 7 khớp trên {args.iface}: {np.rad2deg(q)} link={ok}")
        time.sleep(0.02)

    n = int(round(args.seconds * args.hz))
    samples = []
    dt = 1.0 / args.hz
    print(f"Đang lấy {n} mẫu trong {args.seconds:.1f} s — giữ tay đứng yên...")
    for _ in range(n):
        t0 = time.monotonic()
        q = np.asarray(read_positions(arm, with_gripper=False), float)
        ok = link_ok(arm, max_age_s=0.2)
        if not np.all(np.isfinite(q)) or (ok is not None and not all(ok)):
            raise SystemExit(f"Mất phản hồi CAN khi lấy mẫu: q={np.rad2deg(q)} link={ok}")
        if np.any(np.abs(q) > math.radians(220)):
            raise SystemExit("Có góc encoder bất thường >220°, không ghi calibration")
        samples.append(q)
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    try:
        offset, std, span, count = estimate_zero(
            samples, max_std_deg=args.max_std_deg, max_span_deg=args.max_span_deg)
    except ZeroCalibrationError as exc:
        raise SystemExit(f"Không ghi calibration: {exc}") from exc

    write_zero_file(output, args.side, args.iface, sign, offset, std, span, len(samples))
    print("Đã ghi:", output)
    print("offset_deg:", np.round(np.rad2deg(offset), 3).tolist())
    print("std_deg:   ", np.round(np.rad2deg(std), 4).tolist())
    print("span_deg:  ", np.round(np.rad2deg(span), 4).tolist())
    print("inliers:   ", count.tolist())


if __name__ == "__main__":
    main()
