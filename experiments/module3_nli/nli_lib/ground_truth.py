from typing import Optional

import numpy as np
import pandas as pd

from .constants import ARBITER_JOIN_KEYS, STATUS_LABELS, VARIANTS
from ._utils import _normalize_material_topic_inplace, _normalize_status_inplace, _status_col

def build_hybrid_gt(human_120: pd.DataFrame, adjudicated_v2: pd.DataFrame, pairing_post: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Xây ground truth lai: 120 case do human review được ưu tiên, phần còn lại lấy từ arbiter v2.
    Đây là GT chuẩn cho TN2/TN3 — lựa chọn arbiter v2 thay vì v1/v3 được biện luận trong TN4 §5.4.
    """
    # Bước 1: Chuẩn hoá material_topic và human_correct_status, loại status ngoài tập hợp lệ
    h = human_120.copy()
    adj = adjudicated_v2.copy()
    _normalize_material_topic_inplace(h, adj)
    _normalize_status_inplace(h, "human_correct_status")
    h = h[h["human_correct_status"].isin(STATUS_LABELS)].copy()

    # Bước 2: Thu hẹp cột, đổi tên thành gt_status → h_part là phần human của GT
    h_part = h[ARBITER_JOIN_KEYS + ["human_correct_status"]].rename(
        columns={"human_correct_status": "gt_status"}
    ).copy()

    # Bước 3: Tính {v}_correct_gt cho human rows bằng cách so sánh status variant với gt_status
    # — chỉ làm được khi pairing_post được truyền vào; nếu không thì để NA
    if pairing_post is not None:
        pp = pairing_post.copy()
        _normalize_material_topic_inplace(pp, h_part)
        status_cols = [_status_col(pp, v) for v in VARIANTS]
        pp_sub = pp[ARBITER_JOIN_KEYS + status_cols].drop_duplicates(subset=ARBITER_JOIN_KEYS)
        h_part = h_part.merge(pp_sub, on=ARBITER_JOIN_KEYS, how="left")
        for v in VARIANTS:
            scol = _status_col(h_part, v)
            if scol in h_part.columns:
                h_part[f"{v}_correct_gt"] = (h_part[scol] == h_part["gt_status"]).astype("boolean")
            else:
                h_part[f"{v}_correct_gt"] = pd.NA
        h_part = h_part[ARBITER_JOIN_KEYS + ["gt_status"] + [f"{v}_correct_gt" for v in VARIANTS]]
    else:
        for v in VARIANTS:
            h_part[f"{v}_correct_gt"] = pd.NA
    h_part["gt_source"] = "human"

    # Bước 4: Chuẩn bị phần arbiter — rename correct_status → gt_status,
    # đổi tên {v}_correct → {v}_correct_gt để khớp schema với h_part
    keep_cols = ARBITER_JOIN_KEYS + ["correct_status"] + [f"{v}_correct" for v in VARIANTS]
    keep_cols = [c for c in keep_cols if c in adj.columns]
    adj_full = adj[keep_cols].rename(columns={"correct_status": "gt_status"}).copy()
    for v in VARIANTS:
        if f"{v}_correct" in adj_full.columns:
            adj_full[f"{v}_correct_gt"] = adj_full[f"{v}_correct"].astype("boolean")
            adj_full = adj_full.drop(columns=[f"{v}_correct"])

    # Bước 5: Loại các row đã được human review khỏi phần arbiter, rồi concat hai phần
    h_keys = set(map(tuple, h_part[ARBITER_JOIN_KEYS].values.tolist()))
    adj_keys_tuples = [tuple(t) for t in adj_full[ARBITER_JOIN_KEYS].values.tolist()]
    adj_mask = [t not in h_keys for t in adj_keys_tuples]
    adj_kept = adj_full.loc[adj_mask].copy()
    adj_kept["gt_source"] = "arbiter_v2"
    return pd.concat([h_part, adj_kept], ignore_index=True)

def apply_gt_to_pairing(pairing: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Ghép hybrid GT vào pairing và tính gt_status cuối:
    ưu tiên GT từ human/arbiter; fallback về consensus (a0_status) trên các row agreement.
    """
    # Bước 1: Merge GT vào pairing; gt_status từ GT được giữ riêng dưới tên gt_status_arb
    # để phân biệt với gt_status cuối (bao gồm cả consensus rows)
    p = pairing.copy()
    g = gt.copy()
    _normalize_material_topic_inplace(p, g)
    p = p.merge(g, on=ARBITER_JOIN_KEYS, how="left", validate="m:1")
    p = p.rename(columns={"gt_status": "gt_status_arb"})

    # Bước 2: Xác định agreement mask — dùng cột tính sẵn nếu có, tính lại nếu không
    if "any_disagree_primary" in p.columns:
        agree_mask = ~p["any_disagree_primary"].astype(bool)
    else:
        cols = [_status_col(p, v) for v in ("a0", "a1", "a2", "v_new")]
        agree_mask = (p[cols[0]] == p[cols[1]]) & (p[cols[1]] == p[cols[2]]) & (p[cols[2]] == p[cols[3]])

    # Bước 3: Điền gt_status cuối — ưu tiên GT từ hybrid; fallback consensus (a0) cho agreement rows
    consensus_col = _status_col(p, "a0")
    p["gt_status"] = np.where(p["gt_status_arb"].notna(), p["gt_status_arb"], np.where(agree_mask, p[consensus_col], np.nan))
    p["gt_source"] = p["gt_source"].fillna("")
    p.loc[agree_mask & (p["gt_source"] == ""), "gt_source"] = "consensus"
    return p

def derive_correct_flags(pairing_with_gt: pd.DataFrame) -> pd.DataFrame:
    """Tính cờ `{variant}_correct` cho mỗi variant theo quy tắc TN3 §5.3:
    ưu tiên `systems_correct` do arbiter v2 đánh dấu (`{v}_correct_gt`),
    fallback so sánh status với gt_status cho row human/consensus.
    """
    p = pairing_with_gt.copy()
    # Row không có GT (gt_status = NaN) phải cho ra NaN, không phải False
    has_gt = p["gt_status"].notna()
    for v in VARIANTS:
        carried = f"{v}_correct_gt"
        scol = _status_col(p, v)
        status_match = p[scol] == p["gt_status"]
        # Ưu tiên carried flag từ arbiter v2 (systems_correct); fallback so sánh status
        arr = np.where(p[carried].notna(), p[carried].astype("boolean").astype(float), status_match.astype(float))
        p[f"{v}_correct"] = np.where(has_gt, arr, np.nan)
    return p

def filter_universe(pairing: pd.DataFrame) -> pd.DataFrame:
    """Lọc chỉ giữ rows mà tất cả variants đều có status nằm trong STATUS_LABELS — khớp universe filter của run_statistical_analysis.py."""
    m = pd.Series(True, index=pairing.index)
    for v in VARIANTS:
        m &= pairing[_status_col(pairing, v)].isin(STATUS_LABELS)
    # reset_index để downstream code có thể dùng positional index an toàn sau khi lọc
    return pairing[m].reset_index(drop=True).copy()
