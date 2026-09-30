#!/usr/bin/env python3
"""Lượt 1 fusion: capture đồng thời D435i RGB-D và RGB camera laptop.

Không chạy MediaPipe, không mở CAN và không điều khiển robot.
Phím: q/ESC thoát. Dùng --record-dir để ghi RGB, depth PNG và metadata CSV.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_shadow.config import load_config
from openarm_shadow.multicam import AsyncCapture, latest_pair
from openarm_shadow.sources import OpenCVSource, RealSenseSource


def _fit_height(image, height=480):
    scale = height / image.shape[0]
    return cv2.resize(image, (int(round(image.shape[1] * scale)), height))


def _overlay(image, lines, color=(0, 255, 0)):
    out = image.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 1, cv2.LINE_AA)
    return out


class PairRecorder:
    """Ghi dữ liệu ở thread riêng để disk I/O không chặn capture."""

    def __init__(self, root):
        self.root = Path(root)
        for sub in ("d435i_rgb", "d435i_depth_mm", "laptop_rgb"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue(maxsize=16)
        self.skipped = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def put(self, primary, laptop, delta_s):
        try:
            self.q.put_nowait((primary, laptop, delta_s))
        except queue.Full:
            self.skipped += 1

    def _loop(self):
        metadata = self.root / "frames.csv"
        with metadata.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["d435i_frame", "d435i_host_s", "d435i_device_s",
                             "laptop_frame", "laptop_host_s", "pair_delta_ms"])
            while self.running or not self.q.empty():
                try:
                    p, l, dt = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                tag = f"{p.frame_id:06d}_{l.frame_id:06d}"
                cv2.imwrite(str(self.root / "d435i_rgb" / f"{tag}.jpg"), p.sample.bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(str(self.root / "laptop_rgb" / f"{tag}.jpg"), l.sample.bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                if p.sample.depth_m is not None:
                    depth_mm = np.clip(np.nan_to_num(p.sample.depth_m) * 1000.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(self.root / "d435i_depth_mm" / f"{tag}.png"), depth_mm)
                writer.writerow([p.frame_id, f"{p.host_timestamp_s:.9f}",
                                 "" if p.sample.timestamp_s is None else f"{p.sample.timestamp_s:.9f}",
                                 l.frame_id, f"{l.host_timestamp_s:.9f}", f"{1000*dt:.3f}"])
                f.flush()

    def close(self):
        self.running = False
        self.thread.join(timeout=20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=["config/fusion_d435i_laptop.yaml"])
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--record-seconds", type=float, default=0.0,
                    help="tự thoát sau N giây; 0 = chạy đến khi nhấn q")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["fusion"]
    primary = AsyncCapture(fc["primary_id"], RealSenseSource(cfg)).start()
    laptop_source = OpenCVSource(fc["laptop_index"], fc["laptop_width"], fc["laptop_height"],
                                 fc.get("laptop_fps", 30))
    laptop = AsyncCapture(fc["laptop_id"], laptop_source).start()
    recorder = PairRecorder(args.record_dir) if args.record_dir else None
    last_recorded = (-1, -1)
    started = time.monotonic()
    cv2.namedWindow("dual_camera_capture", cv2.WINDOW_NORMAL)
    try:
        while True:
            p, l, delta = latest_pair(primary, laptop)
            ps, ls = primary.stats, laptop.stats
            if ps["error"] or ls["error"]:
                raise RuntimeError(f"capture error: D435i={ps['error']!r}, laptop={ls['error']!r}")
            if p is None or l is None:
                time.sleep(0.005)
                continue
            dt_ms = 1000.0 * delta
            limit = float(fc.get("max_pair_delta_ms", 50))
            pair_color = (0, 255, 0) if dt_ms <= limit else (0, 165, 255)
            left = _overlay(_fit_height(p.sample.bgr), [
                f"D435i RGB-D | frame {p.frame_id} | {ps['fps']:.1f} fps",
                f"host {p.host_timestamp_s:.3f} | dropped {ps['dropped']}",
            ])
            right = _overlay(_fit_height(l.sample.bgr), [
                f"Laptop RGB | frame {l.frame_id} | {ls['fps']:.1f} fps",
                f"host {l.host_timestamp_s:.3f} | dropped {ls['dropped']}",
                f"latest-pair dt {dt_ms:.1f} ms",
            ], pair_color)
            cv2.imshow("dual_camera_capture", np.hstack((left, right)))
            ids = (p.frame_id, l.frame_id)
            # Không tái sử dụng một frame cũ khi chỉ camera còn lại vừa cập nhật.
            # Lượt pairing tối ưu theo history sẽ được thực hiện ở Lượt 4.
            if (recorder is not None and p.frame_id != last_recorded[0]
                    and l.frame_id != last_recorded[1]):
                recorder.put(p, l, delta)
                last_recorded = ids
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.record_seconds > 0 and time.monotonic() - started >= args.record_seconds:
                break
    finally:
        primary.close()
        laptop.close()
        if recorder is not None:
            recorder.close()
            print(f"Da luu {args.record_dir}; recorder skipped {recorder.skipped} cap frame")
        cv2.destroyAllWindows()
        print("D435i stats:", primary.stats)
        print("Laptop stats:", laptop.stats)


if __name__ == "__main__":
    main()
