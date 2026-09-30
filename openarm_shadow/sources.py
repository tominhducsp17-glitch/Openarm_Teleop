"""Nguồn ảnh cho teleop: OpenCV thông thường hoặc Intel RealSense RGB-D."""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass
class CameraSample:
    bgr: np.ndarray
    depth_m: np.ndarray | None = None
    intrinsics: object | None = None
    timestamp_s: float | None = None


class OpenCVSource:
    def __init__(self, source, width, height, fps=None):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"Không mở được nguồn video: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self):
        ok, bgr = self.cap.read()
        return ok, CameraSample(bgr, timestamp_s=time.monotonic()) if ok else None

    def close(self):
        self.cap.release()


class RealSenseSource:
    """RealSense D4xx: RGB và depth đã lọc, đồng bộ, align theo pixel của RGB."""

    def __init__(self, cfg):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit("Thiếu pyrealsense2. Cài: pip install pyrealsense2") from e
        self.rs = rs
        self.pipe = rs.pipeline()
        rcfg = cfg["camera"].get("realsense", {})
        width = int(rcfg.get("width", 640))
        height = int(rcfg.get("height", 480))
        fps = int(rcfg.get("fps", 30))
        serial = rcfg.get("serial")
        rsc = rs.config()
        if serial:
            rsc.enable_device(str(serial))
        rsc.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        rsc.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            profile = self.pipe.start(rsc)
        except RuntimeError as e:
            raise SystemExit(f"Không mở được RealSense RGB-D {width}x{height}@{fps}: {e}") from e
        device = profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        if rcfg.get("require_usb3", True) and self.usb_type.startswith("2"):
            self.pipe.stop()
            raise SystemExit(
                f"{self.device_name} đang nối ở USB {self.usb_type} (USB 2.x). "
                f"RGB-D {width}x{height}@{fps} cần dây/cổng USB 3.x truyền dữ liệu; "
                "hãy đổi dây rồi cắm lại camera."
            )
        self.align = rs.align(rs.stream.color)
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter() if rcfg.get("spatial_filter", True) else None
        self.temporal = rs.temporal_filter() if rcfg.get("temporal_filter", True) else None

    def read(self):
        try:
            frames = self.align.process(self.pipe.wait_for_frames(5000))
        except RuntimeError:
            return False, None
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None
        # Spatial filter chạy tốt hơn trong disparity domain nhưng khá nặng. Profile
        # realtime có thể tắt nó; khi đó temporal chạy thẳng trên depth và tránh hai
        # phép đổi depth↔disparity không cần thiết.
        if self.spatial is not None:
            depth_frame = self.depth_to_disparity.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            if self.temporal is not None:
                depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)
        elif self.temporal is not None:
            depth_frame = self.temporal.process(depth_frame)
        bgr = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        # Sau rs.align(depth -> color), profile của depth đã nằm trong pixel space của RGB.
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        timestamp_s = float(color_frame.get_timestamp()) * 1e-3
        return True, CameraSample(bgr, depth_m, intr, timestamp_s)

    def close(self):
        self.pipe.stop()


def open_source(source, cfg):
    required = str(cfg.get("camera", {}).get("required_source", "")).lower()
    realsense_names = {"realsense", "rs", "d435", "d435i", "d455"}
    is_realsense = str(source).lower() in realsense_names
    if required in realsense_names and not is_realsense:
        raise SystemExit("Cấu hình này yêu cầu Intel RealSense RGB-D; hãy chạy với --source realsense")
    if is_realsense:
        return RealSenseSource(cfg)
    c = cfg["camera"]
    return OpenCVSource(source, c["width"], c["height"])
