from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .constants import ARBITER_JOIN_KEYS, STATUS_LABELS, VARIANTS
from ._utils import _normalize_material_topic_inplace, _normalize_status_inplace, _status_col

def build_hybrid_gt(
    human_120: pd.DataFrame,
    adjudicated_v2: pd.DataFrame,
    pairing_post: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Hybrid GT = human (120 reviewed cases) + arbiter v2 (~2430 remaining).

    This is the canonical GT for TN2 / TN3 (both pre- and post-fix). The
    choice of arbiter v2 (over v1 / v3) is justified by TN4 §5.4 (see
    `arbiter_kappa_panel` and `sign_reversal_e1`).

    For human-reviewed rows, `{v}_correct_gt` is computed by status comparison
    (`variant_status == human_correct_status`) — this matches the legacy hybrid
    GT in `run_statistical_analysis.py::_build_hybrid_adjudicated`.

    For arbiter rows, `{v}_correct_gt` is carried directly from
    `adjudicated_v2.csv` (which derives it from `systems_correct_json`).

    Args:
        human_120: load_human_review_postfix() (T1' redo) hoặc
                   load_human_review_prefix() (pilot); có `human_correct_status`.
        adjudicated_v2: load_adjudicated('v2') (post-fix) hoặc
                        load_adjudicated_prefix('v2') (pilot); có `correct_status`.
        pairing_post: optional load_pairing(pre_fix=False) used to derive
                      human-row `{v}_correct_gt` via status comparison. If None,
                      callers should still get usable hybrid_gt but the human
                      rows will lack `{v}_correct_gt` columns.

    Returns:
        DataFrame[JOIN_KEYS + gt_status + gt_source + {v}_correct_gt].
    """
    h = human_120.copy()
    adj = adjudicated_v2.copy()
    _normalize_material_topic_inplace(h, adj)
    _normalize_status_inplace(h, "human_correct_status")
    h = h[h["human_correct_status"].isin(STATUS_LABELS)].copy()

    h_part = h[ARBITER_JOIN_KEYS + ["human_correct_status"]].rename(
        columns={"human_correct_status": "gt_status"}
    ).copy()
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

    keep_cols = ARBITER_JOIN_KEYS + ["correct_status"] + [f"{v}_correct" for v in VARIANTS]
    keep_cols = [c for c in keep_cols if c in adj.columns]
    adj_full = adj[keep_cols].rename(columns={"correct_status": "gt_status"}).copy()
    for v in VARIANTS:
        if f"{v}_correct" in adj_full.columns:
            adj_full[f"{v}_correct_gt"] = adj_full[f"{v}_correct"].astype("boolean")
            adj_full = adj_full.drop(columns=[f"{v}_correct"])
    h_keys = set(map(tuple, h_part[ARBITER_JOIN_KEYS].values.tolist()))
    adj_keys_tuples = [tuple(t) for t in adj_full[ARBITER_JOIN_KEYS].values.tolist()]
    adj_mask = [t not in h_keys for t in adj_keys_tuples]
    adj_kept = adj_full.loc[adj_mask].copy()
    adj_kept["gt_source"] = "arbiter_v2"
    return pd.concat([h_part, adj_kept], ignore_index=True)

def apply_gt_to_pairing(pairing: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Merge GT onto pairing.

    Adds:
        - gt_status_arb : the hybrid GT status (NaN on agreement rows)
        - gt_status     : final GT — hybrid GT where available, else a0_status
                          (consensus on agreement rows).
        - gt_source     : 'human', 'arbiter_v2', or 'consensus'.
        - {v}_correct_gt: carried from `build_hybrid_gt`. For human rows this is
                          a status comparison; for arbiter v2 rows it carries
                          the `systems_correct` flag. Used by `derive_correct_flags`
                          in strict mode (TN3).

    Returns: pairing with gt_status + gt_source + {v}_correct_gt columns.
    """
    p = pairing.copy()
    g = gt.copy()
    _normalize_material_topic_inplace(p, g)
    p = p.merge(g, on=ARBITER_JOIN_KEYS, how="left", validate="m:1")
    p = p.rename(columns={"gt_status": "gt_status_arb"})
    if "any_disagree_primary" in p.columns:
        agree_mask = ~p["any_disagree_primary"].astype(bool)
    else:
        cols = [_status_col(p, v) for v in ("a0", "a1", "a2", "v_new")]
        agree_mask = (p[cols[0]] == p[cols[1]]) & (p[cols[1]] == p[cols[2]]) & (p[cols[2]] == p[cols[3]])
    consensus_col = _status_col(p, "a0")
    p["gt_status"] = np.where(p["gt_status_arb"].notna(), p["gt_status_arb"], np.where(agree_mask, p[consensus_col], np.nan))
    p["gt_source"] = p["gt_source"].fillna("")
    p.loc[agree_mask & (p["gt_source"] == ""), "gt_source"] = "consensus"
    return p

def derive_correct_flags(pairing_with_gt: pd.DataFrame) -> pd.DataFrame:
    """Add `{variant}_correct` boolean column for each variant on pairing_with_gt.

    Correctness convention (TN3 §5.3, factorial decomposition):
    Honours the arbiter's `systems_correct` list when the GT row was supplied
    by arbiter v2 inside the hybrid GT: a variant is correct only if the
    arbiter explicitly listed its system as correct (carried as
    `{v}_correct_gt` by `build_hybrid_gt`). For human/consensus rows, falls
    back to status comparison. Matches legacy `run_statistical_analysis.py`.

    Rows with no GT have NaN.
    """
    p = pairing_with_gt.copy()
    has_gt = p["gt_status"].notna()
    for v in VARIANTS:
        carried = f"{v}_correct_gt"
        scol = _status_col(p, v)
        status_match = p[scol] == p["gt_status"]
        arr = np.where(p[carried].notna(), p[carried].astype("boolean").astype(float), status_match.astype(float))
        p[f"{v}_correct"] = np.where(has_gt, arr, np.nan)
    return p

def filter_universe(pairing: pd.DataFrame) -> pd.DataFrame:
    """Restrict to rows where every variant has a valid status in STATUS_LABELS.

    Matches the universe filter in legacy run_statistical_analysis.py.
    """
    m = pd.Series(True, index=pairing.index)
    for v in VARIANTS:
        m &= pairing[_status_col(pairing, v)].isin(STATUS_LABELS)
    return pairing[m].reset_index(drop=True).copy()
