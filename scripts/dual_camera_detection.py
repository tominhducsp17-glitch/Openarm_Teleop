#!/usr/bin/env python3
"""Kiểm tra độ ổn định Pose/Hands hai camera, không calibration/fusion/robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_shadow.config import load_config
from openarm_shadow.multicam import AsyncCapture
from openarm_shadow.multiview import AsyncPerception, best_hand, observation_confidence
from openarm_shadow.sources import OpenCVSource, RealSenseSource
from openarm_shadow.viz import draw_human


def put(image, lines, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        y = 25 + 22 * i
        cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    color, 1, cv2.LINE_AA)


def panel(result, capture, name):
    image = draw_human(result.timed_frame.sample.bgr.copy(), result.frame)
    track = result.frame.tracking
    hand = best_hand(result.frame)
    confidence = observation_confidence(hand)
    stats = capture.stats
    put(image, [
        f"{name} cap {stats['fps']:.1f}fps infer {result.inference_ms:.0f}ms",
        (f"pose={track.get('pose')} raw_hand={track.get('raw_hands', 0)} "
         f"assoc={track.get('associated_hands', 0)} right_conf={confidence:.2f}"),
        "MUC TIEU: raw_hand>=1, assoc>=1 lien tuc | q: thoat",
    ], (0, 255, 0) if hand is not None else (0, 180, 255))
    return image, hand, confidence, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=["config/fusion_d435i_laptop.yaml"])
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--log", default="captures/detection_stability.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["fusion"]
    cap_d = AsyncCapture(fc["primary_id"], RealSenseSource(cfg)).start()
    cap_l = AsyncCapture(fc["laptop_id"], OpenCVSource(
        fc["laptop_index"], fc["laptop_width"], fc["laptop_height"],
        fc.get("laptop_fps", 30))).start()
    det_d = AsyncPerception(cap_d, cfg, orientation_source="depth_required",
                            pose_enabled=False, force_hand_side="right", num_hands=1).start()
    det_l = AsyncPerception(cap_l, cfg, orientation_source="rgb_world",
                            pose_enabled=True, force_hand_side="right", num_hands=1).start()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    start = time.monotonic()
    last_ids = (-1, -1)
    cv2.namedWindow("dual_camera_detection", cv2.WINDOW_NORMAL)
    try:
        while True:
            a, b = det_d.latest(), det_l.latest()
            if det_d.error or det_l.error:
                raise RuntimeError(f"inference: D435i={det_d.error!r}, laptop={det_l.error!r}")
            if a is None or b is None:
                time.sleep(.005)
                continue
            left, hd, cd, td = panel(a, cap_d, "D435i")
            right, hl, cl, tl = panel(b, cap_l, "Laptop-right-view")
            if left.shape[0] != right.shape[0]:
                scale = left.shape[0] / right.shape[0]
                right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
            cv2.imshow("dual_camera_detection", np.hstack((left, right)))
            ids = (a.timed_frame.frame_id, b.timed_frame.frame_id)
            if ids != last_ids:
                row = {
                    "t": time.monotonic(), "frame_ids": ids,
                    "d435i": {"present": hd is not None, "confidence": cd,
                               "inference_ms": a.inference_ms, **td},
                    "laptop": {"present": hl is not None, "confidence": cl,
                               "inference_ms": b.inference_ms, **tl},
                    "capture": {"d435i": cap_d.stats, "laptop": cap_l.stats},
                }
                # Exceptions are only populated on failure and are not JSON serializable.
                for value in row["capture"].values():
                    value["error"] = None if value["error"] is None else repr(value["error"])
                log.write(json.dumps(row) + "\n")
                log.flush()
                last_ids = ids
            if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                break
            if args.seconds and time.monotonic() - start >= args.seconds:
                break
    finally:
        det_d.close(); det_l.close(); cap_d.close(); cap_l.close()
        log.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
