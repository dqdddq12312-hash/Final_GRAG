from __future__ import annotations

import pandas as pd

from .constants import DATA_DIR, VARIANTS


def latency_per_variant() -> pd.DataFrame:
    """§5.3.7 — per-variant median/mean runtime từ snapshot CSV.

    Đọc `data/latency_per_variant_observed.csv` (snapshotted từ lần chạy pipeline
    gốc khi LastWriteTime metadata còn hợp lệ). Đây là nguồn chính thức cho số
    liệu trong thesis vì mtime ở `variant_runs/` đã bị touch bởi refactor.

    Returns: DataFrame[variant, n, median_minutes, mean_minutes, stdev_minutes, total_minutes]
    """
    path = DATA_DIR / "latency_per_variant_observed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"latency snapshot not found at {path}; rerun pipeline to regenerate"
        )
    raw = pd.read_csv(path)
    raw = raw[raw["duration_min"].notna()].copy()
    raw["duration_min"] = raw["duration_min"].astype(float)
    raw = raw[(raw["duration_min"] > 1.0) & (raw["duration_min"] <= 90.0)]

    rows = []
    for v in VARIANTS:
        sub = raw.loc[raw["variant"] == v, "duration_min"]
        if sub.empty:
            rows.append({
                "variant": v, "n": 0,
                "median_minutes": 0.0, "mean_minutes": 0.0,
                "stdev_minutes": 0.0, "total_minutes": 0.0,
            })
            continue
        rows.append({
            "variant": v,
            "n": int(sub.count()),
            "median_minutes": round(float(sub.median()), 2),
            "mean_minutes": round(float(sub.mean()), 2),
            "stdev_minutes": round(float(sub.std(ddof=1)) if len(sub) >= 2 else 0.0, 2),
            "total_minutes": round(float(sub.sum()), 2),
        })
    return pd.DataFrame(rows)
