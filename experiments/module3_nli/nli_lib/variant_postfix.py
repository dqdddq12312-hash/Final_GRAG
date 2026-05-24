from __future__ import annotations

from typing import Optional

import pandas as pd

from .constants import STATUS_LABELS, VARIANTS
from ._utils import _iter_buckets
from .stats_tests import cohen_kappa, mcnemar_exact, _bootstrap_ci_simple

def build_accuracy_table_postfix(df_sample: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Post-fix 150-case (or 137-case) table: variant × phase accuracy + bootstrap CI + κ vs human.

    Args:
        df_sample: from load_postfix_sample(); has `{v}_post` + `human_correct_status` + `bucket`.

    Returns: DataFrame[view, variant, phase, n, accuracy, ci_low, ci_high, kappa]
    """
    h = "human_correct_status"
    rows = []
    for v in VARIANTS:
        for b, sub in _iter_buckets(df_sample):
            n = len(sub)
            if n == 0:
                continue
            correct = (sub[f"{v}_post"] == sub[h]).astype(int).values
            acc, lo, hi = _bootstrap_ci_simple(correct)
            k = cohen_kappa(sub[f"{v}_post"].tolist(), sub[h].tolist(), labels=STATUS_LABELS)
            rows.append({
                "view": label, "variant": v, "phase": b, "n": n,
                "accuracy": round(acc, 3),
                "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                "kappa": round(k, 3) if not pd.isna(k) else float("nan"),
            })
    return pd.DataFrame(rows)

def build_lift_table_postfix(df_sample: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Bug-fix lift: pre/post accuracy per variant per phase + McNemar paired test.

    Returns: DataFrame[view, variant, phase, n, pre_acc, post_acc, lift_pp,
                       helped_b, hurt_c, mcnemar_p]
    """
    h = "human_correct_status"
    rows = []
    for v in VARIANTS:
        for b, sub in _iter_buckets(df_sample):
            n = len(sub)
            if n == 0:
                continue
            c_pre  = sub[f"{v}_pre"]  == sub[h]
            c_post = sub[f"{v}_post"] == sub[h]
            n_pre  = sub[f"{v}_pre"].notna().sum()
            n_post = sub[f"{v}_post"].notna().sum()
            pre_acc  = c_pre.sum()  / n_pre  if n_pre  else 0
            post_acc = c_post.sum() / n_post if n_post else 0
            lift = (post_acc - pre_acc) * 100
            bb = int((c_post & ~c_pre).sum())
            cc = int((c_pre & ~c_post).sum())
            mc = mcnemar_exact(bb, cc)
            rows.append({
                "view": label, "variant": v, "phase": b, "n": n,
                "pre_acc": round(pre_acc, 3),
                "post_acc": round(post_acc, 3),
                "lift_pp": round(lift, 2),
                "helped_b": bb, "hurt_c": cc,
                "mcnemar_p": round(mc["p_value"], 4),
            })
    return pd.DataFrame(rows)

def build_mcnemar_table_postfix(df_sample: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Pairwise McNemar between variants on post-fix sample.

    Returns: DataFrame[view, pair, scope, n, b, c, b_minus_c, p_value, significant]
    """
    h = "human_correct_status"
    rows = []
    pairs = [("a0", "a1"), ("a0", "a2"), ("a0", "v_new"),
             ("a1", "a2"), ("a1", "v_new"), ("a2", "v_new")]
    for v1, v2 in pairs:
        for b, sub in _iter_buckets(df_sample, buckets=["ALL", "phase5_3-3"]):
            c1 = sub[f"{v1}_post"] == sub[h]
            c2 = sub[f"{v2}_post"] == sub[h]
            bb = int((c1 & ~c2).sum())
            cc = int((~c1 & c2).sum())
            mc = mcnemar_exact(bb, cc)
            rows.append({
                "view": label, "pair": f"{v1} vs {v2}", "scope": b, "n": len(sub),
                "b": bb, "c": cc, "b_minus_c": bb - cc,
                "p_value": round(mc["p_value"], 4),
                "significant": mc["p_value"] < 0.05,
            })
    return pd.DataFrame(rows)

def hint_over_prediction_summary(df_sample: pd.DataFrame) -> pd.DataFrame:
    """§5.3.4 — HINT over-predicts `no_evidence`.

    Computes: for each variant, the ratio (predicted_no_evidence / actual_no_evidence).
    Thesis: V_new ratio = 2.31×, A2 ratio = 2.25×.

    Returns: DataFrame[variant, n_actual_ne, n_predicted_ne, ratio]
    """
    h = "human_correct_status"
    n_actual_ne = (df_sample[h] == "no_evidence").sum()
    rows = []
    for v in VARIANTS:
        n_pred_ne = (df_sample[f"{v}_post"] == "no_evidence").sum()
        ratio = round(n_pred_ne / n_actual_ne, 3) if n_actual_ne else float("nan")
        rows.append({
            "variant": v,
            "n_actual_no_evidence": int(n_actual_ne),
            "n_predicted_no_evidence": int(n_pred_ne),
            "ratio": ratio,
        })
    return pd.DataFrame(rows)

def kappa_pre_post_panel(df_sample: pd.DataFrame) -> pd.DataFrame:
    """§5.3.6 — κ (pre vs post) for each variant × phase.

    Returns: DataFrame[variant, phase, n, kappa_pre, kappa_post, delta_kappa]
    """
    h = "human_correct_status"
    rows = []
    for v in VARIANTS:
        for b, sub in _iter_buckets(df_sample):
            n = len(sub)
            if n == 0:
                continue
            k_pre = cohen_kappa(sub[f"{v}_pre"].tolist(), sub[h].tolist(), labels=STATUS_LABELS)
            k_post = cohen_kappa(sub[f"{v}_post"].tolist(), sub[h].tolist(), labels=STATUS_LABELS)
            rows.append({
                "variant": v, "phase": b, "n": n,
                "kappa_pre": round(k_pre, 4) if not pd.isna(k_pre) else float("nan"),
                "kappa_post": round(k_post, 4) if not pd.isna(k_post) else float("nan"),
                "delta_kappa": round(k_post - k_pre, 4) if not (pd.isna(k_post) or pd.isna(k_pre)) else float("nan"),
            })
    return pd.DataFrame(rows)
