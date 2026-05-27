from math import sqrt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import sklearn.metrics
import statsmodels.stats.contingency_tables

def mcnemar_exact(b: int, c: int) -> dict:
    """Tính two-sided exact McNemar test (binomial, p=0.5) qua statsmodels,
    trả về dict với p_value và b−c để caller dùng cho bảng kết quả.
    """
    b, c = int(b), int(c)
    n = b + c
    # Không có discordant pair → test không định nghĩa; p_value quy ước là 1.0
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "b_minus_c": 0}
    # statsmodels yêu cầu ma trận 2×2; b đặt ở [0,1], c ở [1,0] (diagonal = 0)
    table = np.array([[0, b], [c, 0]])
    result = statsmodels.stats.contingency_tables.mcnemar(
        table, exact=True, correction=False
    )
    # min(1.0, ...) để tránh float rounding vượt 1.0 trong một số edge case
    p_two = min(1.0, float(result.pvalue))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_two, "b_minus_c": b - c}

def cohen_kappa(y1, y2, labels: list[str]) -> float:
    """Tính Cohen's kappa cho hai dãy rating; các cặp có label ngoài tập labels
    bị loại trước khi tính để khớp legacy behaviour.
    """
    y1 = list(y1)
    y2 = list(y2)
    # Fail-fast: length mismatch = lỗi caller, không nên trả NaN âm thầm
    if len(y1) != len(y2):
        raise ValueError(f"length mismatch: {len(y1)} vs {len(y2)}")
    if not labels or not y1:
        return float("nan")
    # Lọc cặp ngoài label_set để không nhầm label lạ thành một category hợp lệ
    label_set = set(labels)
    pairs = [(a, b) for a, b in zip(y1, y2) if a in label_set and b in label_set]
    if not pairs:
        return float("nan")
    a_vals, b_vals = zip(*pairs)
    try:
        return float(sklearn.metrics.cohen_kappa_score(a_vals, b_vals, labels=labels))
    except (ValueError, ZeroDivisionError):
        # ZeroDivisionError xảy ra khi chỉ có 1 label duy nhất trong tất cả pairs
        return float("nan")

def confusion_matrix(y1, y2, labels: list[str]) -> pd.DataFrame:
    """Tính confusion matrix K×K dưới dạng DataFrame; index = y1 (predicted/arbiter), columns = y2 (actual/human)."""
    cm = sklearn.metrics.confusion_matrix(y1, y2, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

def cluster_bootstrap_per_report_diff(
    df: pd.DataFrame,
    var_a_correct: str,
    var_b_correct: str,
    B: int = 1000,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap của (var_a − var_b) accuracy diff,
    resample theo report (cluster) với replacement để tôn trọng cấu trúc phân cụm.
    """
    # Bước 1: Tính per-report mean diff — đơn vị resample là report, không phải case
    # để tránh underestimate SE khi các case trong cùng report tương quan nhau
    means_a = df.groupby("report_id")[var_a_correct].mean()
    means_b = df.groupby("report_id")[var_b_correct].mean()
    diff_per_cluster = (means_a - means_b).values
    point = float(diff_per_cluster.mean())

    # Bước 2: Bootstrap percentile CI qua scipy; seed cố định để kết quả reproducible
    res = scipy.stats.bootstrap(
        (diff_per_cluster,),
        np.mean,
        n_resamples=B,
        method="percentile",
        random_state=seed,
    )
    ci_lo = float(res.confidence_interval.low)
    ci_hi = float(res.confidence_interval.high)
    return {
        "point": point,
        "se": float(res.standard_error),
        "ci_95": [ci_lo, ci_hi],
        "B": B,
        "n_clusters": int(len(diff_per_cluster)),
    }

def power_mde_paired(
    n: int,
    p_disc: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict:
    """Tính minimum detectable effect (MDE) cho paired McNemar theo Connor (1987),
    invert n → delta bằng brentq thay vì bisection thủ công.
    """
    # Tính z-scores tương ứng với alpha/2 (two-sided) và target power
    z_a = float(scipy.stats.norm.ppf(1 - alpha / 2))
    z_b = float(scipy.stats.norm.ppf(target_power))

    def _objective(delta: float) -> float:
        # Guard để tránh sqrt âm khi delta² >= p_disc (ngoài vùng nghiệm)
        if delta * delta >= p_disc:
            return float("inf")
        # Connor (1987): n_req(delta) = (z_a√p_disc + z_b√(p_disc−δ²))² / δ²
        # _objective = 0 khi n_req(delta) = n, tức là delta là MDE cần tìm
        term = z_a * sqrt(p_disc) + z_b * sqrt(p_disc - delta ** 2)
        return (term ** 2) / (delta ** 2) - n

    # brentq tìm delta trong (0, √p_disc) với độ chính xác < 1e-10
    mde = float(scipy.optimize.brentq(_objective, 1e-8, sqrt(p_disc) - 1e-8))
    return {
        "n": int(n),
        "p_disc": round(p_disc, 4),
        "alpha": alpha,
        "target_power": target_power,
        "mde": round(mde, 4),
        "mde_pp": round(mde * 100, 2),
    }

def _bootstrap_ci_simple(correct: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI đơn giản (non-cluster) cho mean của mảng binary correct; trả về (point, ci_lo, ci_hi)."""
    # Mảng rỗng không có nghĩa thống kê; trả về 0 để caller xử lý riêng
    if len(correct) == 0:
        return 0.0, 0.0, 0.0
    res = scipy.stats.bootstrap(
        (correct,),
        np.mean,
        n_resamples=n_boot,
        method="percentile",
        random_state=seed,
    )
    return (
        float(correct.mean()),
        float(res.confidence_interval.low),
        float(res.confidence_interval.high),
    )
