import numpy as np
import pytest

from openarm_shadow.zero_calibration import ZeroCalibrationError, estimate_zero


def test_estimate_zero_rejects_outlier_and_recovers_median():
    rng = np.random.default_rng(3)
    true = np.deg2rad([1, -2, 3, -4, 5, -6, 7.0])
    samples = true + rng.normal(0, np.deg2rad(0.02), (100, 7))
    samples[5, 3] = np.deg2rad(170)  # frame rác kiểu phản hồi 0x55
    med, std, span, count = estimate_zero(samples)
    assert np.allclose(med, true, atol=np.deg2rad(0.02))
    assert count[3] == 99
    assert np.all(np.rad2deg(std) < 0.05)
    assert np.all(np.rad2deg(span) < 0.2)


def test_estimate_zero_refuses_moving_robot():
    samples = np.zeros((100, 7))
    samples[:, 5] = np.deg2rad(np.linspace(-2, 2, 100))
    with pytest.raises(ZeroCalibrationError, match="Robot chưa đứng yên"):
        estimate_zero(samples)
