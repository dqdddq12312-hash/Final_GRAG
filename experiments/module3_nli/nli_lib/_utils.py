"""Internal helper functions shared across submodules."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .constants import MATERIAL_TOPIC_NA_SENTINEL, PHASE_BUCKETS


def _normalize_material_topic_inplace(*dfs: pd.DataFrame) -> None:
    """Replace NaN/None material_topic with sentinel '__none__' (so merge keys align)."""
    for d in dfs:
        if "material_topic" in d.columns:
            d["material_topic"] = d["material_topic"].fillna(MATERIAL_TOPIC_NA_SENTINEL)


def _normalize_status_inplace(df: pd.DataFrame, col: str) -> None:
    """Strip + lowercase a status column in place."""
    df[col] = df[col].astype(str).str.strip().str.lower()


def _status_col(df: pd.DataFrame, v: str) -> str:
    """Return ``{v}_status_resolved`` if present, else ``{v}_status``.

    Pairing tables produced by ``run_pairing.py`` always carry the resolved
    column; older snapshots may only have the raw status.
    """
    resolved = f"{v}_status_resolved"
    return resolved if resolved in df.columns else f"{v}_status"


def _iter_buckets(df: pd.DataFrame, buckets: Optional[list[str]] = None, bucket_col: str = "bucket"):
    """Yield ``(label, sub_df)`` over the named post-fix bucket views.

    ``ALL`` returns the full frame; other labels filter by ``bucket_col``.
    ``buckets`` defaults to :data:`PHASE_BUCKETS` (ALL + the three phase
    buckets used by the post-fix accuracy / lift / kappa-panel builders).
    """
    for b in buckets or PHASE_BUCKETS:
        yield b, df if b == "ALL" else df[df[bucket_col] == b]
