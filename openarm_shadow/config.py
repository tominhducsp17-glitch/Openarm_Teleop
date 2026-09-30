from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "default.yaml"


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None):
    """Đọc default, một/nhiều profile, rồi calibration zero của robot.

    Khi truyền nhiều profile, profile đứng sau ghi đè profile đứng trước. Nhờ đó
    cấu hình camera (ví dụ D435i/USB2) độc lập với cấu hình robot thật.
    """
    cfg = yaml.safe_load(DEFAULT.read_text())
    paths = [] if path is None else ([path] if isinstance(path, (str, Path)) else list(path))
    for profile_path in paths:
        cfg = _merge(cfg, yaml.safe_load(Path(profile_path).read_text()))
    zero_file = cfg.get("robot", {}).get("zero_calibration_file")
    if zero_file:
        zp = Path(zero_file)
        zp = zp if zp.is_absolute() else ROOT / zp
        if not zp.is_file():
            raise FileNotFoundError(f"Không thấy file zero calibration: {zp}")
        cfg = _merge(cfg, yaml.safe_load(zp.read_text()))
    for k in ("pose", "hand"):
        p = Path(cfg["models"][k])
        cfg["models"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg
