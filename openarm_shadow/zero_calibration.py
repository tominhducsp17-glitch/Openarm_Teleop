"""Ước lượng và lưu zero-pose phần mềm từ encoder OpenArm."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


class ZeroCalibrationError(ValueError):
    pass


def estimate_zero(samples, max_std_deg=0.25, max_span_deg=0.8, min_inlier_ratio=0.8):
    """Trả về median/std/span/count cho 7 encoder, sau khi loại outlier độc lập từng khớp."""
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[1] != 7 or x.shape[0] < 10:
        raise ZeroCalibrationError(f"Cần ít nhất 10 mẫu x 7 khớp, nhận {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ZeroCalibrationError("Mẫu encoder có NaN/Inf")

    med0 = np.median(x, axis=0)
    mad = np.median(np.abs(x - med0), axis=0)
    # Sàn 0.1° tránh MAD=0 làm loại số lượng tử hợp lệ; 6*MAD loại frame rác 0x55.
    threshold = np.maximum(np.deg2rad(0.1), 6.0 * mad)
    good = np.abs(x - med0) <= threshold
    need = int(np.ceil(min_inlier_ratio * len(x)))

    med = np.empty(7)
    std = np.empty(7)
    span = np.empty(7)
    count = good.sum(axis=0)
    for j in range(7):
        if count[j] < need:
            raise ZeroCalibrationError(
                f"J{j + 1}: chỉ {count[j]}/{len(x)} mẫu ổn định (cần {need})")
        v = x[good[:, j], j]
        med[j] = np.median(v)
        std[j] = np.std(v)
        span[j] = np.ptp(v)

    std_deg = np.rad2deg(std)
    span_deg = np.rad2deg(span)
    bad = np.flatnonzero((std_deg > max_std_deg) | (span_deg > max_span_deg))
    if len(bad):
        detail = ", ".join(
            f"J{i + 1} std={std_deg[i]:.2f}° span={span_deg[i]:.2f}°" for i in bad)
        raise ZeroCalibrationError("Robot chưa đứng yên: " + detail)
    return med, std, span, count


def write_zero_file(path, side, iface, sign, offset_rad, std_rad, span_rad, samples):
    """Ghi atomic file YAML được load_config() merge sau profile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "zero_calibration": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "interface": iface,
            "samples": int(samples),
            "std_deg": np.round(np.rad2deg(std_rad), 4).tolist(),
            "span_deg": np.round(np.rad2deg(span_rad), 4).tolist(),
        },
        "robot": {
            "urdf_to_motor": {
                side: {
                    "sign": [int(v) for v in sign],
                    "offset_deg": np.round(np.rad2deg(offset_rad), 4).tolist(),
                }
            }
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    tmp.replace(path)

