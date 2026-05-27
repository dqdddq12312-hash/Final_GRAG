import pandas as pd

from .constants import (
    REPORTS_MAIN_TXT,
    HUMAN_REVIEW_V2_CSV,
    SAMPLE_MAIN_CSV,
    PRE_FIX_PAIRING_CSV,
    VARIANTS,
)


def load_reports_main() -> list[str]:
    """Đọc danh sách report ID dùng trong thí nghiệm NLI chính (TN3/TN4) từ reports_main.txt."""
    return REPORTS_MAIN_TXT.read_text(encoding="utf-8").split()


def load_postfix_sample(view: str = "14-report") -> pd.DataFrame:
    """Load 150-case post-fix sample với cột {v}_pre/{v}_post (status trước/sau topic-bug fix)
    và bucket (tách phase5_3-3 riêng); lọc theo view '14-report' hoặc '15-report'.
    """
    # Bước 1: Load human review + sample_main; rename _status_resolved → _post
    # (_post = status sau khi topic-bug đã được fix)
    hr = pd.read_csv(HUMAN_REVIEW_V2_CSV, dtype={"case_id": str})
    sm = pd.read_csv(SAMPLE_MAIN_CSV, dtype={"case_id": str})
    sm = sm.rename(columns={f"{v}_status_resolved": f"{v}_post" for v in VARIANTS})

    # Bước 2: Đọc pairing pre-fix, trích cột _status_resolved → rename → _pre, merge vào sample_main
    # join_keys lấy giao các cột tồn tại ở cả hai bảng để tránh lỗi khi schema lệch version
    pre = pd.read_csv(PRE_FIX_PAIRING_CSV, low_memory=False)
    join_keys = [
        c for c in [
            "report_id", "disclosure_id", "material_topic",
            "requirement_id", "occurrence_idx",
        ] if c in pre.columns and c in sm.columns
    ]
    pre_cols = [c for c in pre.columns if c.endswith("_status_resolved")]
    pre_subset = pre[join_keys + pre_cols].drop_duplicates(subset=join_keys).copy()
    pre_subset = pre_subset.rename(columns={c: c.replace("_status_resolved", "_pre") for c in pre_cols})
    sm = sm.merge(pre_subset, on=join_keys, how="left")

    # Bước 3: Join human review với sample_main theo case_id; validate="1:1" để bắt duplicate
    df = hr.merge(sm, on="case_id", how="left", validate="1:1")

    # Bước 4: Tạo cột bucket — tách phase5+disclosure_id=3-3 thành bucket riêng
    # vì 3-3 (Material Topics disclosure) có đặc tính khác với các disclosure phase5 thông thường
    is_p5_33 = (df["phase"] == "phase5") & (df["disclosure_id"] == "3-3")
    df["bucket"] = df["phase"]
    df.loc[is_p5_33, "bucket"] = "phase5_3-3"

    # Bước 5: Lọc theo view; raise rõ nếu view không hợp lệ
    if view in {"14-report", "14"}:
        df = df[~df["report_id"].isin({"Energean2024"})].copy()
    elif view not in {"15-report", "15"}:
        raise ValueError(f"view must be '14' or '15' (or '14-report'/'15-report'), got {view}")
    return df
