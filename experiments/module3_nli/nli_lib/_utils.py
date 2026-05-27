from typing import Optional
import pandas as pd

from .constants import MATERIAL_TOPIC_NA_SENTINEL, PHASE_BUCKETS

def _normalize_material_topic_inplace(*dfs: pd.DataFrame) -> None:
    """Chuẩn hóa topic rỗng để khóa merge giữa các artifact khớp nhau."""
    for d in dfs:
        if "material_topic" in d.columns:
            d["material_topic"] = d["material_topic"].fillna(MATERIAL_TOPIC_NA_SENTINEL)

def _normalize_status_inplace(df: pd.DataFrame, col: str) -> None:
    """Đưa nhãn status về dạng canonical trước khi so sánh."""
    df[col] = df[col].astype(str).str.strip().str.lower()

def _status_col(df: pd.DataFrame, v: str) -> str:
    """Chọn cột status phù hợp giữa artifact mới và snapshot cũ.

    Bảng pairing mới ưu tiên ``{v}_status_resolved``; snapshot cũ dùng
    ``{v}_status``.
    """
    resolved = f"{v}_status_resolved"
    return resolved if resolved in df.columns else f"{v}_status"

def _iter_buckets(df: pd.DataFrame, buckets: Optional[list[str]] = None, bucket_col: str = "bucket"):
    """Duyệt các lát cắt phase dùng cho bảng metric post-fix.

    ``ALL`` giữ toàn bộ bảng; các bucket còn lại lọc theo ``bucket_col``.
    """
    for b in buckets or PHASE_BUCKETS:
        yield b, df if b == "ALL" else df[df[bucket_col] == b]
