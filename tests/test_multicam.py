import time

import numpy as np

from openarm_shadow.multicam import AsyncCapture, latest_pair
from openarm_shadow.sources import CameraSample


class _FakeSource:
    def __init__(self):
        self.closed = False

    def read(self):
        time.sleep(0.003)
        return True, CameraSample(np.zeros((2, 3, 3), np.uint8), timestamp_s=time.monotonic())

    def close(self):
        self.closed = True


def test_async_capture_keeps_latest_frame_and_reports_pair_delta():
    sa, sb = _FakeSource(), _FakeSource()
    a, b = AsyncCapture("a", sa).start(), AsyncCapture("b", sb).start()
    try:
        time.sleep(0.03)
        fa, fb, dt = latest_pair(a, b)
        assert fa.camera_id == "a" and fb.camera_id == "b"
        assert fa.frame_id > 1 and fb.frame_id > 1
        assert dt >= 0
        assert a.stats["fps"] > 0 and b.stats["fps"] > 0
        assert a.stats["dropped"] > 0
    finally:
        a.close()
        b.close()
    assert sa.closed and sb.closed
