from typing import Optional

import pandas as pd

from .constants import STATUS_LABELS, VARIANTS
from ._utils import _iter_buckets
from .stats_tests import cohen_kappa, mcnemar_exact, _bootstrap_ci_simple

def build_accuracy_table_postfix(df_sample: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Xây bảng accuracy post-fix (variant × phase): accuracy, bootstrap CI 95% và κ vs human."""
    h = "human_correct_status"
    rows = []
    for v in VARIANTS:
        for b, sub in _iter_buckets(df_sample):
            n = len(sub)
            if n == 0:
                continue
            # accuracy = tỷ lệ variant _post khớp với human_correct_status
            correct = (sub[f"{v}_post"] == sub[h]).astype(int).values
            acc, lo, hi = _bootstrap_ci_simple(correct)
            # kappa vs human để đo agreement vượt ra ngoài accuracy đơn thuần
            k = cohen_kappa(sub[f"{v}_post"].tolist(), sub[h].tolist(), labels=STATUS_LABELS)
            rows.append({
                "view": label, "variant": v, "phase": b, "n": n,
                "accuracy": round(acc, 3),
                "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                "kappa": round(k, 3) if not pd.isna(k) else float("nan"),
            })
    return pd.DataFrame(rows)

def build_lift_table_postfix(df_sample: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Tính lift (post − pre accuracy, pp) do bug fix cho từng variant × phase, kèm McNemar paired test."""
    h = "human_correct_status"
    rows = []
    for v in VARIANTS:
        for b, sub in _iter_buckets(df_sample):
            n = len(sub)
            if n == 0:
                continue
            c_pre  = sub[f"{v}_pre"]  == sub[h]
            c_post = sub[f"{v}_post"] == sub[h]
            # Dùng notna count làm denominator vì cột _pre có thể có NaN ở một số case
            n_pre  = sub[f"{v}_pre"].notna().sum()
            n_post = sub[f"{v}_post"].notna().sum()
            pre_acc  = c_pre.sum()  / n_pre  if n_pre  else 0
            post_acc = c_post.sum() / n_post if n_post else 0
            lift = (post_acc - pre_acc) * 100
            # b = helped (pre sai → post đúng); c = hurt (pre đúng → post sai)
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
    """McNemar pairwise giữa các cặp variant trên post-fix sample, trên scope ALL và phase5_3-3."""
    h = "human_correct_status"
    rows = []
    # 6 cặp tổ hợp không lặp từ 4 variants
    pairs = [("a0", "a1"), ("a0", "a2"), ("a0", "v_new"),
             ("a1", "a2"), ("a1", "v_new"), ("a2", "v_new")]
    for v1, v2 in pairs:
        # Chỉ dùng scope ALL và phase5_3-3 vì đây là hai granularity có ý nghĩa phân tích
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
    """Đo mức độ over-predict 'no_evidence' của từng variant so với human (§5.3.4);
    ratio > 1 nghĩa là variant dự đoán nhiều no_evidence hơn thực tế.
    """
    h = "human_correct_status"
    # n_actual_ne tính một lần vì là denominator chung cho tất cả variants
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
    """Bảng κ(variant, human) trước và sau bug fix cho mỗi variant × phase, với delta_kappa (§5.3.6)."""
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
                # delta_kappa = NaN nếu một trong hai kappa không tính được
                "delta_kappa": round(k_post - k_pre, 4) if not (pd.isna(k_post) or pd.isna(k_pre)) else float("nan"),
            })
    return pd.DataFrame(rows)
