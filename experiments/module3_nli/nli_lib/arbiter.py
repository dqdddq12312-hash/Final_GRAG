from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ARBITER_JOIN_KEYS, STATUS_LABELS
from ._utils import _normalize_material_topic_inplace, _normalize_status_inplace
from .stats_tests import cohen_kappa, confusion_matrix

# Kappa interpretation
_KAPPA_THRESHOLDS = [
    (0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
    (0.60, "moderate"), (0.80, "substantial"),
]

def _interpret_kappa(k: float) -> str:
    """Landis–Koch (1977) kappa interpretation."""
    if pd.isna(k):
        return "n/a"
    for upper, label in _KAPPA_THRESHOLDS:
        if k < upper:
            return label
    return "almost perfect"

# §6 — Arbiter functions
def merge_human_with_arbiter(human_df: pd.DataFrame, arbiter_df: pd.DataFrame, arbiter_label: str) -> pd.DataFrame:
    """Inner-merge human review with one arbiter table on canonical join keys.

    Args:
        human_df: from load_human_review_postfix() / load_human_review_prefix() (n=120).
        arbiter_df: from load_adjudicated('v1'|'v2'|'v3') / load_adjudicated_prefix(...).
        arbiter_label: 'v1', 'v2', or 'v3' (used for output column suffix).

    Returns:
        DataFrame[case_id, ..., human_correct_status, arbiter_{label}_status]
        with `material_topic` normalised to '__none__' and `human_correct_status`
        lowercased+stripped.
    """
    h = human_df.copy()
    a = arbiter_df.copy()
    _normalize_material_topic_inplace(h, a)
    _normalize_status_inplace(h, "human_correct_status")
    keep = ARBITER_JOIN_KEYS + ["correct_status"]
    return h.merge(
        a[keep].rename(columns={"correct_status": f"arbiter_{arbiter_label}_status"}),
        on=ARBITER_JOIN_KEYS,
        how="left",
    )

def arbiter_vs_human_kappa(
    human_df: pd.DataFrame,
    arbiter_df: pd.DataFrame,
    arbiter_label: str = "v2",
) -> dict:
    """Cohen's kappa + agreement + confusion + marginals for one arbiter vs human.

    Args:
        human_df: load_human_review_postfix() / load_human_review_prefix() (120 rows).
        arbiter_df: load_adjudicated('v1'/'v2'/'v3') / load_adjudicated_prefix(...).
        arbiter_label: cosmetic, used as key in returned dict.

    Returns:
        {n, agreement_rate, cohen_kappa, human_marginals, arbiter_marginals,
         confusion_matrix}
    """
    m = merge_human_with_arbiter(human_df, arbiter_df, arbiter_label)
    col = f"arbiter_{arbiter_label}_status"
    m = m.dropna(subset=[col]).copy()
    if len(m) == 0:
        return {"n": 0, "agreement_rate": float("nan"), "cohen_kappa": float("nan")}
    y_h = m["human_correct_status"].tolist()
    y_a = m[col].tolist()
    k = cohen_kappa(y_h, y_a, labels=STATUS_LABELS)
    agreement = float((m["human_correct_status"] == m[col]).mean())
    cm = confusion_matrix(y_a, y_h, labels=STATUS_LABELS)
    return {
        "n": int(len(m)),
        "agreement_rate": agreement,
        "cohen_kappa": k,
        "human_marginals": pd.Series(y_h).value_counts().to_dict(),
        "arbiter_marginals": pd.Series(y_a).value_counts().to_dict(),
        "confusion_matrix": cm,
    }

def arbiter_marginal_distribution(
    human_df: pd.DataFrame,
    adjudicated_v1: pd.DataFrame,
    adjudicated_v2: pd.DataFrame,
    adjudicated_v3: pd.DataFrame,
) -> pd.DataFrame:
    """§5.4.2 marginal table — rows=source, cols=status (n, n_pass, ..., pct_pass, ...).

    Merges tất cả ba arbiter outputs (v1/v2/v3) lên pool 120-case human.
    Drops NaN per source riêng lẻ nên mỗi row báo n riêng.
    """
    h = human_df.copy()
    _normalize_material_topic_inplace(h)
    _normalize_status_inplace(h, "human_correct_status")

    rows = []
    h_status = h["human_correct_status"].tolist()
    h_vc = pd.Series(h_status).value_counts().to_dict()
    rows.append({
        "source": "Human",
        "n": len(h_status),
        **{f"n_{s}": int(h_vc.get(s, 0)) for s in STATUS_LABELS},
        **{f"pct_{s}": round(100.0 * h_vc.get(s, 0) / len(h_status), 1) for s in STATUS_LABELS},
    })

    for label, adj_df in [("Arbiter v1", adjudicated_v1), ("Arbiter v2", adjudicated_v2),
                          ("Arbiter v3", adjudicated_v3)]:
        version = label.lower().split()[-1]
        res = arbiter_vs_human_kappa(human_df, adj_df, arbiter_label=version)
        n = res["n"]
        vc = res.get("arbiter_marginals", {})
        rows.append({
            "source": label,
            "n": n,
            **{f"n_{s}": int(vc.get(s, 0)) for s in STATUS_LABELS},
            **{f"pct_{s}": round(100.0 * vc.get(s, 0) / n, 1) if n else 0.0 for s in STATUS_LABELS},
        })
    return pd.DataFrame(rows)

def arbiter_kappa_panel(
    human_df: pd.DataFrame,
    adjudicated_v1: pd.DataFrame,
    adjudicated_v2: pd.DataFrame,
    adjudicated_v3: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side panel: kappa(arbiter_vi, human) for i ∈ {1, 2, 3}.

    v1/v2: metrics trên n=120 (canonical).
    v3: metrics trên giao 3-way (v1 ∩ v2 ∩ v3 ∩ human có verdict) — §5.4.3.

    Returns:
        DataFrame[arbiter, n, agreement_rate, cohen_kappa, interpretation]
    """
    rows = []
    for label, adj_df in [("v1", adjudicated_v1), ("v2", adjudicated_v2)]:
        r = arbiter_vs_human_kappa(human_df, adj_df, arbiter_label=label)
        rows.append({
            "arbiter": label,
            "n": r["n"],
            "agreement_rate": round(r["agreement_rate"], 4),
            "cohen_kappa": round(r["cohen_kappa"], 4),
            "interpretation": _interpret_kappa(r["cohen_kappa"]),
        })

    m = merge_human_with_arbiter(human_df, adjudicated_v1, "v1")
    m = m.merge(
        adjudicated_v2[ARBITER_JOIN_KEYS + ["correct_status"]].rename(columns={"correct_status": "arbiter_v2_status"}),
        on=ARBITER_JOIN_KEYS, how="left",
    )
    m = m.merge(
        adjudicated_v3[ARBITER_JOIN_KEYS + ["correct_status"]].rename(columns={"correct_status": "arbiter_v3_status"}),
        on=ARBITER_JOIN_KEYS, how="left",
    )
    m = m.dropna(subset=["arbiter_v1_status", "arbiter_v2_status", "arbiter_v3_status"]).copy()
    y_h = m["human_correct_status"].tolist()
    y_a3 = m["arbiter_v3_status"].tolist()
    k3 = cohen_kappa(y_h, y_a3, labels=STATUS_LABELS)
    agreement3 = float((m["human_correct_status"] == m["arbiter_v3_status"]).mean())
    rows.append({
        "arbiter": "v3",
        "n": len(m),
        "agreement_rate": round(agreement3, 4),
        "cohen_kappa": round(k3, 4),
        "interpretation": _interpret_kappa(k3),
    })

    return pd.DataFrame(rows)

def sign_reversal_e1(
    pairing: pd.DataFrame,
    adjudicated_v1: pd.DataFrame,
    adjudicated_v2: pd.DataFrame,
) -> dict:
    """§5.4.4 — recompute V_new vs A0 (HINT main effect) under v1 GT and v2 GT.

    Args:
        pairing: full pairing DataFrame.
        adjudicated_v1, adjudicated_v2: arbiter outputs.

    Returns:
        {n_disagreements, e1_v1_pp, e1_v2_pp, sign_reversed}
        where e1_X_pp = (V_new_correct% - A0_correct%) under arbiter X GT.
    """
    a1 = adjudicated_v1.copy()
    a2 = adjudicated_v2.copy()
    p = pairing.copy()
    _normalize_material_topic_inplace(p, a1, a2)

    m1 = p.merge(a1[ARBITER_JOIN_KEYS + ["correct_status"]].rename(columns={"correct_status": "gt_v1"}), on=ARBITER_JOIN_KEYS, how="left")
    m2 = p.merge(a2[ARBITER_JOIN_KEYS + ["correct_status"]].rename(columns={"correct_status": "gt_v2"}), on=ARBITER_JOIN_KEYS, how="left")

    common_mask = m1["gt_v1"].notna() & m2["gt_v2"].notna()
    n_disc = int(common_mask.sum())
    e1 = {}
    for gt_col, label in [("gt_v1", "v1"), ("gt_v2", "v2")]:
        df = (m1 if label == "v1" else m2).loc[common_mask].copy()
        df["a0_correct"] = df["a0_status"] == df[gt_col]
        df["v_new_correct"] = df["v_new_status"] == df[gt_col]
        a0_acc = df["a0_correct"].mean()
        vn_acc = df["v_new_correct"].mean()
        e1[label] = 100.0 * (vn_acc - a0_acc)
    return {
        "n_disagreements": n_disc,
        "e1_v1_pp": round(e1["v1"], 2),
        "e1_v2_pp": round(e1["v2"], 2),
        "sign_reversed": bool(np.sign(e1["v1"]) != np.sign(e1["v2"])),
    }
