from __future__ import annotations

from math import sqrt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import sklearn.metrics
import statsmodels.stats.contingency_tables


def mcnemar_exact(b: int, c: int) -> dict:
    """Two-sided exact McNemar (binomial, p = 0.5).

    Args:
        b: count where method-A correct, method-B wrong
        c: count where method-A wrong,   method-B correct

    Returns:
        {b, c, n_discordant, p_value (two-sided), b_minus_c}

    Delegates to statsmodels.stats.contingency_tables.mcnemar(exact=True,
    correction=False). For n_discordant == 0, p_value is 1.0.
    """
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "b_minus_c": 0}
    table = np.array([[0, b], [c, 0]])
    result = statsmodels.stats.contingency_tables.mcnemar(
        table, exact=True, correction=False
    )
    p_two = min(1.0, float(result.pvalue))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_two, "b_minus_c": b - c}


def cohen_kappa(y1, y2, labels: list[str]) -> float:
    """Cohen's kappa for two list-like rater outputs.

    Args:
        y1, y2: list/array of category labels (same length).
        labels: explicit label set — items outside this set are dropped
                before computation (matches legacy behaviour).

    Returns:
        kappa (float). nan if denominator zero or no valid pairs.

    Delegates to sklearn.metrics.cohen_kappa_score.
    """
    y1 = list(y1)
    y2 = list(y2)
    if len(y1) != len(y2):
        raise ValueError(f"length mismatch: {len(y1)} vs {len(y2)}")
    if not labels or not y1:
        return float("nan")
    label_set = set(labels)
    pairs = [(a, b) for a, b in zip(y1, y2) if a in label_set and b in label_set]
    if not pairs:
        return float("nan")
    a_vals, b_vals = zip(*pairs)
    try:
        return float(sklearn.metrics.cohen_kappa_score(a_vals, b_vals, labels=labels))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def confusion_matrix(y1, y2, labels: list[str]) -> pd.DataFrame:
    """K×K confusion matrix as DataFrame[index=y1_label, cols=y2_label]."""
    cm = sklearn.metrics.confusion_matrix(y1, y2, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def cluster_bootstrap_per_report_diff(
    df: pd.DataFrame,
    var_a_correct: str,
    var_b_correct: str,
    B: int = 1000,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap of (var_a − var_b) using per-report means.

    Resamples reports (clusters) with replacement, then averages the
    per-report mean accuracy difference. Delegates to scipy.stats.bootstrap.
    """
    means_a = df.groupby("report_id")[var_a_correct].mean()
    means_b = df.groupby("report_id")[var_b_correct].mean()
    diff_per_cluster = (means_a - means_b).values
    point = float(diff_per_cluster.mean())
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
    """Minimum-detectable effect for paired McNemar via Connor (1987) formula.

    The Connor closed form for the sample size needed to detect a difference
    delta = p1 − p2 in two paired proportions, given paired-discordant rate
    p_disc, alpha (two-sided) and target power, is:

        n_req = (z_{a/2} √p_disc + z_b √(p_disc − delta²))² / delta²

    Inverts for delta given n using scipy.optimize.brentq (replaces manual
    60-iteration bisection). Error < 1e-10, well within tolerance of ±1.5 pp.

    Args:
        n:            total paired sample size.
        p_disc:       paired-discordance rate (b + c) / n from data.
        alpha:        two-sided significance level (default 0.05).
        target_power: desired power (default 0.80).

    Returns: {n, p_disc, alpha, target_power, mde, mde_pp}
    """
    z_a = float(scipy.stats.norm.ppf(1 - alpha / 2))
    z_b = float(scipy.stats.norm.ppf(target_power))

    def _objective(delta: float) -> float:
        if delta * delta >= p_disc:
            return float("inf")
        term = z_a * sqrt(p_disc) + z_b * sqrt(p_disc - delta ** 2)
        return (term ** 2) / (delta ** 2) - n

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
    """Simple non-cluster bootstrap of mean. Returns (point, ci_lo, ci_hi)."""
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
