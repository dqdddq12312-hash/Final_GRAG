import pandas as pd

from .constants import LATENCY_OBSERVED_CSV, VARIANTS


def latency_per_variant() -> pd.DataFrame:
    """Đọc latency snapshot CSV và tính median/mean/stdev/total runtime theo từng variant (§5.3.7).
    Dùng snapshot thay vì variant_runs/ vì mtime tại đó đã bị touch khi refactor pipeline.
    """
    # Bước 1: Load snapshot; fail-fast với message rõ nếu file chưa được tạo
    if not LATENCY_OBSERVED_CSV.exists():
        raise FileNotFoundError(
            f"latency snapshot not found at {LATENCY_OBSERVED_CSV}; rerun pipeline to regenerate"
        )
    raw = pd.read_csv(LATENCY_OBSERVED_CSV)

    # Bước 2: Lọc NaN và outlier — run <1 phút thường là cache hit / lỗi; >90 phút bất thường
    raw = raw[raw["duration_min"].notna()].copy()
    raw["duration_min"] = raw["duration_min"].astype(float)
    raw = raw[(raw["duration_min"] > 1.0) & (raw["duration_min"] <= 90.0)]

    # Bước 3: Tính stats theo từng variant; điền 0 cho variant không có run hợp lệ
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
            # ddof=1 (sample std); fallback 0.0 khi chỉ có 1 run
            "stdev_minutes": round(float(sub.std(ddof=1)) if len(sub) >= 2 else 0.0, 2),
            "total_minutes": round(float(sub.sum()), 2),
        })
    return pd.DataFrame(rows)
