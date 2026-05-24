from __future__ import annotations

import pandas as pd

from .constants import DATA_DIR, PRE_FIX_DIR, VARIANTS


def load_reports_main() -> list[str]:
    """Report IDs used by the main NLI experiments (TN3 / TN4)."""
    return (DATA_DIR / "reports_main.txt").read_text(encoding="utf-8").split()


def load_postfix_sample(view: str = "14-report") -> pd.DataFrame:
    """Load 150-case post-fix sample with `{v}_pre`, `{v}_post`, `human_correct_status` per row.

    Args:
        view: '14-report' (excludes Energean2024, n=137) or '15-report' (all 150).

    Returns: DataFrame with columns:
        - case_id, report_id, phase, disclosure_id, material_topic, requirement_id, occurrence_idx
        - bucket (phase or 'phase5_3-3')
        - {v}_pre, {v}_post (status_resolved before/after topic-bug fix)
        - human_correct_status
    """
    hr = pd.read_csv(DATA_DIR / "human_review_v2.csv", dtype={"case_id": str})
    sm = pd.read_csv(DATA_DIR / "sample_main.csv", dtype={"case_id": str})
    sm = sm.rename(columns={f"{v}_status_resolved": f"{v}_post" for v in VARIANTS})

    pre = pd.read_csv(PRE_FIX_DIR / "pairing_pre_topic_fix.csv", low_memory=False)
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
    df = hr.merge(sm, on="case_id", how="left", validate="1:1")

    is_p5_33 = (df["phase"] == "phase5") & (df["disclosure_id"] == "3-3")
    df["bucket"] = df["phase"]
    df.loc[is_p5_33, "bucket"] = "phase5_3-3"

    if view in {"14-report", "14"}:
        df = df[~df["report_id"].isin({"Energean2024"})].copy()
    elif view not in {"15-report", "15"}:
        raise ValueError(f"view must be '14' or '15' (or '14-report'/'15-report'), got {view}")
    return df
