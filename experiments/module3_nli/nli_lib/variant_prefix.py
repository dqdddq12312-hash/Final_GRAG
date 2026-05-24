from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats

from .constants import STATUS_LABELS, VARIANTS
from ._utils import _status_col
from .stats_tests import cohen_kappa, mcnemar_exact, cluster_bootstrap_per_report_diff

def per_variant_marginals(pairing: pd.DataFrame) -> pd.DataFrame:
    """Status marginal distribution per variant.

    Returns: DataFrame[variant, n_total, n_pass, n_partial, n_no_evidence, rate_*]
    """
    n = len(pairing)
    rows = []
    for v in VARIANTS:
        counts = pairing[_status_col(pairing, v)].value_counts()
        rows.append({
            "variant": v,
            "n_total": n,
            "n_pass": int(counts.get("pass", 0)),
            "n_partial": int(counts.get("partial", 0)),
            "n_no_evidence": int(counts.get("no_evidence", 0)),
            "rate_pass": round(counts.get("pass", 0) / n, 4) if n else 0,
            "rate_partial": round(counts.get("partial", 0) / n, 4) if n else 0,
            "rate_no_evidence": round(counts.get("no_evidence", 0) / n, 4) if n else 0,
        })
    return pd.DataFrame(rows)

def pairwise_kappa_variants(pairing: pd.DataFrame) -> pd.DataFrame:
    """Inter-variant Cohen's κ on status labels.

    Returns: DataFrame[variant_a, variant_b, n, agreement_rate, cohen_kappa]
    """
    rows = []
    for i, va in enumerate(VARIANTS):
        for vb in VARIANTS[i + 1:]:
            col_a, col_b = _status_col(pairing, va), _status_col(pairing, vb)
            sa = pairing[col_a].astype(str).tolist()
            sb = pairing[col_b].astype(str).tolist()
            k = cohen_kappa(sa, sb, labels=STATUS_LABELS)
            agreement = float((pairing[col_a] == pairing[col_b]).mean())
            rows.append({
                "variant_a": va, "variant_b": vb,
                "n": int(len(pairing)),
                "agreement_rate": round(agreement, 4),
                "cohen_kappa": round(k, 4) if not pd.isna(k) else float("nan"),
            })
    return pd.DataFrame(rows)

def e1_metrics(pairing_with_gt: pd.DataFrame) -> dict:
    """V_new vs A0 head-to-head (E1).

    Returns:
        {n, n_reports, a0_acc, v_new_acc, v_new_minus_a0_pp,
         only_v_new_correct, only_a0_correct, both_correct, neither_correct,
         mcnemar, bootstrap_diff_per_report, win_rate_v_new}
    """
    df = pairing_with_gt.dropna(subset=["a0_correct", "v_new_correct"]).copy()
    a0 = df["a0_correct"].astype(bool).values
    vn = df["v_new_correct"].astype(bool).values
    only_vn = int((vn & ~a0).sum())
    only_a0 = int((a0 & ~vn).sum())
    both = int((vn & a0).sum())
    neither = int((~vn & ~a0).sum())
    mc = mcnemar_exact(only_vn, only_a0)
    df["_a0_int"] = a0.astype(int)
    df["_vn_int"] = vn.astype(int)
    boot = cluster_bootstrap_per_report_diff(df, "_vn_int", "_a0_int", B=1000)
    return {
        "n": int(len(df)),
        "n_reports": int(df["report_id"].nunique()),
        "both_correct": both,
        "neither_correct": neither,
        "only_v_new_correct": only_vn,
        "only_a0_correct": only_a0,
        "a0_acc": float(a0.mean()),
        "v_new_acc": float(vn.mean()),
        "v_new_minus_a0_pp": round(100.0 * (vn.mean() - a0.mean()), 2),
        "win_rate_v_new": round(only_vn / (only_vn + only_a0), 4) if (only_vn + only_a0) else float("nan"),
        "mcnemar": mc,
        "bootstrap_diff_per_report": boot,
    }

def e2b_factorial(pairing_with_gt: pd.DataFrame, B: int = 1000, seed: int = 42) -> dict:
    """2×2 factorial decomposition (REORDER × HINT) by per-report mean + bootstrap.

    A0  = no NLI         (REORDER=off, HINT=off)
    A1  = REORDER only   (REORDER=on,  HINT=off)
    A2  = HINT only      (REORDER=off, HINT=on)
    V_new = REORDER+HINT (REORDER=on,  HINT=on)

    Main effects (per-report mean accuracy):
        REORDER     = (m_a1 + m_v_new)/2 − (m_a0 + m_a2)/2
        HINT        = (m_a2 + m_v_new)/2 − (m_a0 + m_a1)/2
        INTERACTION = (m_v_new − m_a1) − (m_a2 − m_a0)
    Bootstrap (resample reports, scipy.stats.bootstrap) gives CI +
    percentile two-sided p-value.

    Returns dict with main_effect_reorder / main_effect_hint /
    interaction_reorder_x_hint plus per_variant_acc per-report-mean.
    """
    df = pairing_with_gt.dropna(subset=[f"{v}_correct" for v in VARIANTS]).copy()
    means: dict[str, dict[str, float]] = {}
    for rid, sub in df.groupby("report_id"):
        means[rid] = {v: float(sub[f"{v}_correct"].astype(bool).mean()) for v in VARIANTS}
    reports = sorted(means.keys())
    m = {v: np.array([means[r][v] for r in reports]) for v in VARIANTS}
    reorder_per = (m["a1"] + m["v_new"]) / 2 - (m["a0"] + m["a2"]) / 2
    hint_per    = (m["a2"] + m["v_new"]) / 2 - (m["a0"] + m["a1"]) / 2
    inter_per   = (m["v_new"] - m["a1"]) - (m["a2"] - m["a0"])

    n = len(reports)

    def _boot(values: np.ndarray) -> dict:
        res = scipy.stats.bootstrap(
            (values,),
            np.mean,
            n_resamples=B,
            method="percentile",
            random_state=seed,
        )
        boot_distrib = res.bootstrap_distribution
        ci_lo = float(res.confidence_interval.low)
        ci_hi = float(res.confidence_interval.high)
        p = 2.0 * min(float(np.mean(boot_distrib <= 0)), float(np.mean(boot_distrib >= 0)))
        return {
            "point_pp": round(100.0 * float(values.mean()), 4),
            "se_pp": round(100.0 * float(res.standard_error), 4),
            "ci_95_pp": [round(100.0 * ci_lo, 4), round(100.0 * ci_hi, 4)],
            "p_value": round(min(p, 1.0), 4),
            "B": B,
            "n_reports": n,
        }

    return {
        "n_reports": n,
        "main_effect_reorder": _boot(reorder_per),
        "main_effect_hint": _boot(hint_per),
        "interaction_reorder_x_hint": _boot(inter_per),
        "per_variant_mean_of_means": {v: round(float(m[v].mean()), 4) for v in VARIANTS},
    }

def per_phase_breakdown(pairing_with_gt: pd.DataFrame) -> pd.DataFrame:
    """Per-phase × per-variant accuracy.

    Returns: DataFrame[phase, variant, n_total, n_correct, rate_correct]
    """
    correct_cols = [f"{v}_correct" for v in VARIANTS]
    agg = (
        pairing_with_gt.groupby("phase")[correct_cols]
        .apply(lambda g: pd.DataFrame({
            "n_total": [int(g[c].dropna().shape[0]) for c in correct_cols],
            "n_correct": [int(g[c].dropna().astype(bool).sum()) for c in correct_cols],
        }, index=VARIANTS))
        .reset_index(level=1)
        .rename(columns={"level_1": "variant"})
        .reset_index()
    )
    agg["rate_correct"] = (agg["n_correct"] / agg["n_total"]).round(4).where(agg["n_total"] > 0, float("nan"))
    return agg[["phase", "variant", "n_total", "n_correct", "rate_correct"]]
