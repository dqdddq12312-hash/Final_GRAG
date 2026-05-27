import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arbiter_lib import (  # noqa: E402
    PRIMARY_VARIANTS,
    RATIONALE_MAX_CHARS,
    _parse_citations,
    _truncate,
    build_anon_layout,
    collect_evidence_chunks,
    load_chunks,
    load_registry,
    lookup_disclosure_name,
    lookup_requirement,
)
from nli_lib.constants import (  # noqa: E402
    ARBITER_JOIN_KEYS,
    DISAGREEMENTS_CSV,
    HUMAN_T1_DIR,
    HUMAN_TEMPLATE_CSV,
    HUMAN_TEMPLATE_MD,
    adjudicated_csv,
)

DISAGREE_CSV = DISAGREEMENTS_CSV
ADJUDICATED_CSV = adjudicated_csv("v1")
TEMPLATE_CSV = HUMAN_TEMPLATE_CSV
TEMPLATE_MD = HUMAN_TEMPLATE_MD

N_TARGET = 120
MIN_PER_STRATUM = 2
SEED = 42

# Stratified allocation (largest-residual / Hamilton method)
def _largest_residual_allocation(counts: dict[str, int], n_total: int, floor: int) -> dict[str, int]:
    """Phân bổ n_total slot theo tỷ lệ stratum, đảm bảo mỗi stratum ≥ min(floor, size),
    làm tròn bằng phương pháp largest-residual (Hamilton method).
    """
    keys = np.array(list(counts.keys()))
    avail = np.array([counts[k] for k in keys], dtype=float)

    # Bước 1: Gán floor guarantee cho mỗi stratum trước khi phân bổ tỷ lệ
    base = np.minimum(floor, avail).astype(int)
    remaining = max(n_total - int(base.sum()), 0)
    if remaining == 0:
        return dict(zip(keys, base.tolist()))

    # Bước 2: Phân bổ quota còn lại tỷ lệ theo kích thước stratum sau khi trừ floor
    avail_after = avail - base
    total_after = avail_after.sum()
    if total_after == 0:
        return dict(zip(keys, base.tolist()))

    raw = remaining * avail_after / total_after
    floored = np.floor(raw).astype(int)
    diff = remaining - int(floored.sum())

    # Bước 3: Phân phối slot dư bằng largest-residual — ưu tiên stratum có phần thập phân lớn nhất,
    # tie-break theo avail_after để stratum lớn hơn được ưu tiên khi residual bằng nhau
    fractions = raw - floored
    order = np.lexsort((avail_after, fractions))[::-1]
    extras = np.zeros(len(keys), dtype=int)
    for idx in order:
        if diff <= 0:
            break
        if floored[idx] + extras[idx] < avail_after[idx]:
            extras[idx] += 1
            diff -= 1

    alloc = base + floored + extras
    alloc = np.minimum(alloc, avail.astype(int))
    return dict(zip(keys.tolist(), alloc.tolist()))

def stratified_sample(df: pd.DataFrame, stratum_col: str, n_total: int, floor: int, seed: int) -> pd.DataFrame:
    """Sample n_total case từ df theo stratum_col bằng phân bổ largest-residual, đảm bảo ≥ floor case/stratum."""
    rng = np.random.default_rng(seed)
    counts = df[stratum_col].value_counts().to_dict()
    alloc = _largest_residual_allocation(counts, n_total, floor)

    parts: list[pd.DataFrame] = []
    for stratum, n in alloc.items():
        if n <= 0:
            continue
        pool = df[df[stratum_col] == stratum]
        n_draw = min(n, len(pool))
        # Seed riêng per-stratum từ rng để đảm bảo reproducible dù thứ tự stratum thay đổi
        parts.append(pool.sample(n=n_draw, random_state=int(rng.integers(0, 2**31 - 1))))

    out = pd.concat(parts, ignore_index=True)
    # stable sort để giữ thứ tự ban đầu trong cùng stratum sau khi gộp
    return out.sort_values(stratum_col, kind="stable").reset_index(drop=True)

# Markdown rendering (per-case review aid)
def render_anonymized_systems(row: dict, var_to_sys: dict[str, int]) -> dict[int, dict]:
    """Xây dict {system_number → verdict info} cho một row, ẩn tên variant theo hoán vị anonymized."""
    out: dict[int, dict] = {}
    for variant in PRIMARY_VARIANTS:
        sys_no = var_to_sys[variant]
        out[sys_no] = {
            "status": str(row.get(f"{variant}_status", "")),
            "decision_path": str(row.get(f"{variant}_decision_path", "")),
            "rationale": str(row.get(f"{variant}_rationale", "")),
            "citations": _parse_citations(row.get(f"{variant}_citations_json")),
        }
    return out

def render_case_md(case_id: int, ctx: dict) -> str:
    """Render một case thành Markdown review aid với context đầy đủ: requirement, evidence, anonymized verdicts."""
    req = ctx["request"]
    sys_view = ctx["sys_view"]
    ev = ctx["evidence_chunks"]

    pieces: list[str] = []

    # Section 1: Case header + metadata
    pieces.append(f"## Case {case_id:03d}\n")
    pieces.append(f"- **Report**: `{req['report_id']}`")
    pieces.append(f"- **Phase**: `{req['phase']}`")
    pieces.append(f"- **Standard**: `{req['standard_id']}`")
    pieces.append(f"- **Disclosure**: `{req['disclosure_id']}` — {req['disclosure_name']}")
    mt = req["material_topic"]
    # __none__ là sentinel cho disclosure không thuộc material topic cụ thể
    if mt == "__none__":
        pieces.append("- **Material topic**: *(non-topical / general)*")
    else:
        pieces.append(f"- **Material topic**: `{mt}`")
    pieces.append(f"- **Requirement-id**: `{req['requirement_id']}`")
    pieces.append(f"- **Disagreement pattern (a0|a1|a2|v_new correct)**: `{ctx['stratum']}`")
    pieces.append("")
    pieces.append("### Requirement text\n")
    pieces.append(f"> {req['requirement_text']}")
    pieces.append("")

    # Section 2: Evidence chunks được cite bởi ít nhất một system
    pieces.append("### Evidence chunks (cited by at least one system)\n")
    if not ev:
        pieces.append("> *(No chunks were cited by any of the 4 systems.)*\n")
    else:
        for c in ev:
            page = c.get("page_start")
            section = (c.get("section_label") or "").strip()
            head = f"**[{c['chunk_id']}**"
            if page is not None:
                head += f" — p.{page}"
            if section:
                head += f" — {section}"
            head += "]"
            pieces.append(head)
            pieces.append("")
            pieces.append(f"> {c['content_text']}")
            pieces.append("")

    # Section 3: Anonymized verdicts — annotator không biết variant nào là System nào
    pieces.append("### Anonymized system verdicts\n")
    for sys_no in (1, 2, 3, 4):
        v = sys_view[sys_no]
        cites = ", ".join(v["citations"]) or "*(none)*"
        rationale = _truncate(v["rationale"], RATIONALE_MAX_CHARS)
        pieces.append(f"**System {sys_no}** — status: `{v['status']}` — decision_path: `{v['decision_path']}`")
        pieces.append(f"  - citations: {cites}")
        pieces.append(f"  - rationale: {rationale}")
        pieces.append("")
    pieces.append("---\n")
    return "\n".join(pieces)

def _build_stratum_label(row: dict) -> str:
    """Tạo stratum label dạng '0110' từ correctness flags của 4 variants (a0|a1|a2|v_new)."""
    return "".join(str(int(bool(row[f"{v}_correct"]))) for v in PRIMARY_VARIANTS)

def main() -> None:
    """Driver chính: load data → stratified sample 120 case → build context → xuất CSV template + Markdown review aid."""
    # Bước 1: Load disagreements + adjudicated_v1, merge theo join keys, loại row lỗi arbiter
    print("Loading disagreements + adjudicated_v1")
    df_d = pd.read_csv(DISAGREE_CSV)
    df_a = pd.read_csv(ADJUDICATED_CSV)
    df_d["material_topic"] = df_d["material_topic"].fillna("__none__")
    df_a["material_topic"] = df_a["material_topic"].fillna("__none__")

    df = df_d.merge(df_a, on=ARBITER_JOIN_KEYS, how="inner", suffixes=("", "_arb"))
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    print(f"  {len(df)} arbitrated disagreements available for sampling")

    # Bước 2: Gán stratum label theo correctness pattern 4 variants, rồi stratified sample
    df["stratum"] = df.apply(_build_stratum_label, axis=1)
    print(f"  {df['stratum'].nunique()} distinct strata")

    sample = stratified_sample(df, "stratum", N_TARGET, MIN_PER_STRATUM, SEED).reset_index(drop=True)
    sample.insert(0, "case_id", [f"{i:03d}" for i in range(1, len(sample) + 1)])
    print(f"Sampled {len(sample)} rows")

    # Bước 3: Build context (registry lookup, evidence chunks, anonymization) cho từng case
    req_map, disc_map = load_registry()
    contexts: list[dict] = []
    for _, row in sample.iterrows():
        rd = row.to_dict()
        var_to_sys, _ = build_anon_layout(rd)
        sys_view = render_anonymized_systems(rd, var_to_sys)
        chunks = load_chunks(rd["report_id"])
        evidence = collect_evidence_chunks(rd, chunks)
        req_text = lookup_requirement(req_map, rd["standard_id"], rd["disclosure_id"], rd["requirement_id"]) or ""
        disc_name = lookup_disclosure_name(disc_map, rd["standard_id"], rd["disclosure_id"])
        contexts.append({
            "case_id": rd["case_id"],
            "stratum": rd["stratum"],
            "request": {
                "report_id": rd["report_id"], "phase": rd["phase"],
                "standard_id": rd["standard_id"], "disclosure_id": rd["disclosure_id"],
                "disclosure_name": disc_name, "material_topic": rd["material_topic"],
                "requirement_id": rd["requirement_id"], "requirement_text": req_text,
            },
            "sys_view": sys_view,
            "evidence_chunks": evidence,
        })

    # Bước 4: Xuất template CSV (annotator điền human_correct_status/…) + Markdown review aid (read-only)
    HUMAN_T1_DIR.mkdir(parents=True, exist_ok=True)
    src_cols = [
        "case_id", "stratum", "report_id", "phase", "standard_id", "disclosure_id",
        "material_topic", "requirement_id", "occurrence_idx",
    ]
    template = sample[src_cols].copy()
    for col in ("human_correct_status", "human_systems_correct", "human_notes"):
        template[col] = ""
    template.to_csv(TEMPLATE_CSV, index=False, encoding="utf-8")
    print(f"Wrote template CSV -> {TEMPLATE_CSV.name} ({len(template)} rows)")

    md_chunks = [render_case_md(int(ctx["case_id"]), ctx) for ctx in contexts]
    TEMPLATE_MD.write_text("".join(md_chunks), encoding="utf-8")
    print(f"Wrote review aid Markdown -> {TEMPLATE_MD.name}")

if __name__ == "__main__":
    main()
