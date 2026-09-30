"""Capture nhiều camera độc lập bằng latest-frame buffer.

Module này chỉ lo I/O và timestamp; không chạy MediaPipe, fusion hay robot.
Mỗi camera có một thread riêng để camera chậm không tạo backlog cho camera kia.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

from .sources import CameraSample


@dataclass(frozen=True)
class TimedFrame:
    camera_id: str
    frame_id: int
    host_timestamp_s: float
    sample: CameraSample


class AsyncCapture:
    """Đọc source trong background và chỉ giữ frame mới nhất."""

    def __init__(self, camera_id, source):
        self.camera_id = str(camera_id)
        self.source = source
        self._lock = threading.Lock()
        self._latest = None
        self._consumed_id = -1
        self._running = False
        self._thread = None
        self._error = None
        self._frame_id = 0
        self._dropped = 0
        self._fps = 0.0
        self._capture_times = deque(maxlen=90)

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"capture-{self.camera_id}", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        try:
            while self._running:
                ok, sample = self.source.read()
                now = time.monotonic()
                if not ok or sample is None:
                    if self._running:
                        time.sleep(0.005)
                    continue
                with self._lock:
                    if self._latest is not None and self._latest.frame_id > self._consumed_id:
                        self._dropped += 1
                    self._frame_id += 1
                    self._latest = TimedFrame(self.camera_id, self._frame_id, now, sample)
                    self._capture_times.append(now)
                    if len(self._capture_times) > 1:
                        self._fps = ((len(self._capture_times) - 1) /
                                     max(self._capture_times[-1] - self._capture_times[0], 1e-6))
        except Exception as exc:  # surfaced to the UI/main thread via stats
            self._error = exc
            self._running = False

    def latest(self):
        with self._lock:
            frame = self._latest
            if frame is not None:
                self._consumed_id = frame.frame_id
            return frame

    @property
    def stats(self):
        with self._lock:
            return {
                "fps": self._fps,
                "frames": self._frame_id,
                "dropped": self._dropped,
                "error": self._error,
            }

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        self.source.close()


def latest_pair(primary: AsyncCapture, secondary: AsyncCapture):
    """Trả hai frame mới nhất và độ lệch host-arrival timestamp (giây)."""
    a, b = primary.latest(), secondary.latest()
    if a is None or b is None:
        return a, b, None
    return a, b, abs(a.host_timestamp_s - b.host_timestamp_s)
