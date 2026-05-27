"""GRAG Compliance Demo Viewer.

Single-file Streamlit app. Two modes:
  1. Audit Report  — render one (variant, report) compliance_report.json
  2. Experiment Overview — show pre-rendered figures + key tables

Reads JSON/CSV/PNG artefacts in place; no Ollama / Zilliz / re-run required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants (trust repo layout)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "module3_nli"
VARIANT_RUNS_DIR = EXP_DIR / "variant_runs"
CHUNKS_DIR = REPO_ROOT / "metadata" / "report_units"
FIGURES_DIR = EXP_DIR / "outputs" / "figures"

# Tạm thời ẩn tab Experiment Overview — bật lại khi cần demo thí nghiệm.
SHOW_EXPERIMENT_OVERVIEW = False

VARIANT_LABELS = {
    "a0": "A0 — Baseline (no NLI)",
    "a1": "A1 — REORDER only",
    "a2": "A2 — HINT only",
    "v_new": "V_new — REORDER + HINT",
}

PHASE_TITLES = {
    "phase1": "1 · Claim & SOU",
    "phase2": "2 · Reporting Principles",
    "phase3": "3 · GRI 2 General",
    "phase4": "4 · Material Topics",
    "phase5": "5 · GRI 3 per Topic",
    "phase6": "6 · Topic Standards",
    "phase7": "7 · Omissions",
    "phase8": "8 · Content Index",
    "phase8_light": "8 · Content Index (light)",
    "phase9": "9 · Notify GRI",
    "final": "Final",
}

STATUS_COLOR = {
    "pass": "#16a34a",
    "fail": "#dc2626",
    "partial": "#f59e0b",
    "external_verification": "#6366f1",
    "skipped": "#9ca3af",
    "no_evidence": "#dc2626",
    "omitted": "#6b7280",
    "omitted_valid": "#16a34a",
    "omitted_invalid": "#dc2626",
    "reported": "#16a34a",
    "missing": "#dc2626",
    "external_reference": "#6366f1",
    "omitted_invalid_reason": "#dc2626",
    "invalid": "#dc2626",
    "valid": "#16a34a",
    "missing_no_omission": "#dc2626",
}

# ---------------------------------------------------------------------------
# Static thesis tables (Chapter 4 — TN4 / TN3 stage).
# Hard-coded từ thesis vì các CSV trong outputs/ là kết quả intermediate
# với rounding/seed khác bản thesis cuối; dùng số liệu thesis làm canonical.
# ---------------------------------------------------------------------------

TN4_KAPPA_PANEL_DF = pd.DataFrame(
    [
        ("v1 (free-form, 3 dòng)", 120, "0,2397", "Fair"),
        ("v2 (anti-hedge decision tree)", 120, "0,3507", "Fair (best)"),
        ("v3 (broad partial, ≥ 1/5 conditions)", 83, "0,1491", "Slight (failed)"),
    ],
    columns=["Arbiter", "n", "κ vs human", "Diễn giải"],
)

TN4_SIGN_REVERSAL_DF = pd.DataFrame(
    [
        ("v1 (free-form, 3 dòng)", 2550, "−9,02", "V_new kém A0 rõ rệt"),
        ("v2 (anti-hedge decision tree)", 2550, "+5,49", "V_new trội A0"),
    ],
    columns=["Arbiter prompt", "n", "δ_E1 (pp)", "Hướng kết luận"],
)

TN3_FACTORIAL_DF = pd.DataFrame(
    [
        ("REORDER main", "−0,14 pp", "[−0,67; +0,33]", "0,578",
         "Không có tác động phát hiện được"),
        ("HINT main", "+1,20 pp", "[−0,31; +2,86]", "0,130",
         "Tín hiệu nhỏ dương, dưới mức ý nghĩa"),
        ("REORDER × HINT", "−0,32 pp", "[−1,13; +0,45]", "0,410",
         "Hai cơ chế parallel, không synergy"),
    ],
    columns=["Hiệu ứng", "Ước lượng", "95% CI", "p", "Diễn giải"],
)

TN3_POSTFIX_ACC_DF = pd.DataFrame(
    [
        ("A0 (no NLI)",          "0,577", "[0,496; 0,664]",
         "0,677", "0,586", "0,405", "0,378 (fair)"),
        ("A1 (REORDER only)",    "0,635", "[0,555; 0,715]",
         "0,692", "0,690", "0,500", "0,456 (moderate)"),
        ("A2 (HINT only)",       "0,474", "[0,394; 0,555]",
         "0,523", "0,414", "0,452", "0,246 (fair)"),
        ("V_new (REORDER+HINT)", "0,467", "[0,380; 0,555]",
         "0,508", "0,345", "0,500", "0,238 (fair)"),
    ],
    columns=["Biến thể", "Acc", "95% CI", "Ph.5 3-3", "Ph.3", "Ph.6",
             "κ vs human"],
)

TN3_BUGFIX_LIFT_DF = pd.DataFrame(
    [
        ("A0 (no NLI)",          "0,518", "0,577", "+5,84 pp",  "+12,31 pp", "0,2153"),
        ("A1 (REORDER only)",    "0,504", "0,635", "+13,14 pp", "+26,15 pp", "0,0014"),
        ("A2 (HINT only)",       "0,409", "0,474", "+6,57 pp",  "+10,77 pp", "0,0784"),
        ("V_new (REORDER+HINT)", "0,416", "0,467", "+5,11 pp",  "+9,23 pp",  "0,2295"),
    ],
    columns=["Biến thể", "Pre acc", "Post acc", "Lift ALL",
             "Lift Ph.5 3-3", "McNemar p"],
)

# ---------------------------------------------------------------------------
# Cached loaders (boundary — validate fail-fast, downstream trusts)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_runs() -> dict[str, list[str]]:
    """Return {variant_id: [report_id, ...]} for available post-fix runs."""
    runs: dict[str, list[str]] = {}
    for variant in VARIANT_LABELS:
        vdir = VARIANT_RUNS_DIR / variant
        if not vdir.exists():
            continue
        runs[variant] = sorted(
            p.name for p in vdir.iterdir()
            if (p / "compliance_report.json").exists()
        )
    return runs


@st.cache_data(show_spinner=False)
def load_report(variant: str, report_id: str) -> dict:
    path = VARIANT_RUNS_DIR / variant / report_id / "compliance_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_chunks(report_id: str) -> dict[str, dict]:
    """Load {chunk_id: chunk_dict} for offline citation drill-down."""
    path = CHUNKS_DIR / report_id / "report_chunks.json"
    if not path.exists():
        return {}
    chunks = json.loads(path.read_text(encoding="utf-8"))
    return {c["chunk_id"]: c for c in chunks}


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------


def badge_html(status: str, *, big: bool = False) -> str:
    color = STATUS_COLOR.get(status, "#6b7280")
    pad = "6px 14px" if big else "2px 8px"
    size = "1.1em" if big else "0.8em"
    label = status.upper().replace("_", " ")
    return (
        f'<span style="background:{color};color:white;padding:{pad};'
        f'border-radius:6px;font-size:{size};font-weight:700;'
        f'white-space:nowrap;letter-spacing:0.5px;">{label}</span>'
    )


def render_phase_pills(all_status: dict[str, str], evaluated: list[str]) -> None:
    eval_set = set(evaluated)
    phases = [p for p in all_status if p != "final"]
    cols = st.columns(len(phases))
    for col, phase in zip(cols, phases):
        status = all_status[phase]
        marker = (
            '<div style="font-size:0.7em;color:#dc2626;">● gate</div>'
            if phase in eval_set
            else '<div style="font-size:0.7em;color:transparent;">.</div>'
        )
        col.markdown(
            f'<div style="text-align:center">'
            f'<div style="font-size:0.75em;color:#6b7280;margin-bottom:2px">'
            f'{PHASE_TITLES.get(phase, phase)}</div>'
            f'{badge_html(status)}'
            f'{marker}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_findings(findings: list[str]) -> None:
    if not findings:
        st.caption("_no findings_")
        return
    for f in findings:
        st.markdown(f"- {f}")


# ---------------------------------------------------------------------------
# Citation drill-down
# ---------------------------------------------------------------------------


def render_citation(chunk_id: str, chunks: dict[str, dict]) -> None:
    chunk = chunks.get(chunk_id)
    if chunk is None:
        st.markdown(f"- `{chunk_id}` _(chunk text not in local cache)_")
        return
    pages = ", ".join(map(str, chunk.get("page_numbers") or [])) or "—"
    heading = chunk.get("section_label") or " > ".join(chunk.get("heading_path") or [])
    label = f"📄 `{chunk_id}`  · page {pages}  · {heading or '—'}"
    with st.expander(label):
        st.markdown(f"_{chunk.get('chunk_type','text')}, {chunk.get('token_count','?')} tokens_")
        st.write(chunk.get("content_text", "(empty)"))


# ---------------------------------------------------------------------------
# Phase-specific renderers (trust schema in compliance_report.json)
# ---------------------------------------------------------------------------


def render_disclosure_verdicts(
    verdicts: list[dict], chunks: dict[str, dict], key_prefix: str
) -> None:
    """Phase 3/5/6 — list disclosures → drill-down one → its requirements."""
    if not verdicts:
        st.caption("_no disclosures evaluated in this phase_")
        return

    df = pd.DataFrame(
        [
            {
                "Disclosure": v["disclosure_id"],
                "Standard": v["standard_id"],
                "Material topic": v.get("material_topic") or "—",
                "Verdict": v["overall"],
                "#req": len(v["requirement_verdicts"]),
                "Unrecoverable": "⚠" if v.get("evidence_unrecoverable") else "",
            }
            for v in verdicts
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)

    options = list(range(len(verdicts)))
    def fmt(i: int) -> str:
        v = verdicts[i]
        mt = f" ({v['material_topic']})" if v.get("material_topic") else ""
        return f"{v['standard_id']} {v['disclosure_id']}{mt} — {v['overall']}"

    idx = st.selectbox(
        "Inspect a disclosure", options, format_func=fmt, key=f"{key_prefix}_sel"
    )
    sel = verdicts[idx]
    st.markdown(
        f"**{sel['standard_id']} — {sel['disclosure_id']}** &nbsp; "
        f"verdict: {badge_html(sel['overall'])}",
        unsafe_allow_html=True,
    )
    if sel.get("notes"):
        st.caption(sel["notes"])

    for req in sel["requirement_verdicts"]:
        conf = req.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        header = (
            f"Req **{req['requirement_id']}** &nbsp; {badge_html(req['status'])} "
            f"&nbsp; <span style='color:#6b7280;font-size:0.85em'>"
            f"source={req.get('source','llm')} · conf={conf_str} · "
            f"path={req.get('decision_path','standard')}</span>"
        )
        with st.expander(req["requirement_id"] + "  ·  " + req["status"]):
            st.markdown(header, unsafe_allow_html=True)
            st.markdown("**Rationale.** " + (req.get("rationale") or "_(none)_"))
            cits = req.get("citations") or []
            if cits:
                st.markdown(f"**Citations ({len(cits)}):**")
                for cid in cits:
                    render_citation(cid, chunks)
            else:
                st.caption("_no citations_")


def render_phase_4(artifacts: dict, materiality_map: dict | None) -> None:
    mm = materiality_map or artifacts.get("materiality_map") or {}
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Material topics**")
        topics = mm.get("material_topics") or []
        if topics:
            st.markdown("\n".join(f"- {t}" for t in topics))
        else:
            st.caption("_none_")
        non_mat = mm.get("non_material_topics") or []
        if non_mat:
            st.markdown("**Non-material topics (with explanation)**")
            st.dataframe(pd.DataFrame(non_mat), width="stretch", hide_index=True)
    with col_b:
        st.markdown("**Base-topic assignment (GRI 11)**")
        assignment = mm.get("base_topic_assignment") or {}
        if assignment:
            rows = [
                {"GRI 11": k, "Material topic(s)": ", ".join(v) if isinstance(v, list) else str(v)}
                for k, v in assignment.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("_no GRI 11 mapping_")


def render_phase_7(artifacts: dict) -> None:
    table = artifacts.get("reconcile_table") or []
    if not table:
        st.caption("_no rows in reconcile_table_")
        return
    df = pd.DataFrame(table)
    cols = [
        "standard_id", "disclosure_id", "source_phase", "verdict",
        "reason", "reason_valid", "explanation_non_empty",
        "requires_not_applicable", "hard_fail_blocked",
    ]
    show_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[show_cols], width="stretch", hide_index=True)

    st.markdown("**Verdict counts**")
    counts = df["verdict"].value_counts().rename_axis("verdict").reset_index(name="n")
    st.dataframe(counts, width="stretch", hide_index=True)


def render_phase_8(artifacts: dict) -> None:
    checklist = artifacts["checklist"]
    appendix = artifacts.get("appendix_used", "Appendix")
    n_pass = artifacts.get("n_pass", sum(c["present"] for c in checklist))
    n_items = artifacts.get("n_items", len(checklist))
    st.markdown(f"**{appendix}** &nbsp; · &nbsp; **{n_pass}/{n_items}** items pass")
    rows = [
        {
            "Item": it["item_id"],
            "Name": it["item_name"],
            "Status": "✅ pass" if it["present"] else "❌ fail",
            "Source field": it.get("evidence_field", ""),
            "Preview": it.get("value_preview", ""),
        }
        for it in checklist
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_phase_final(artifacts: dict) -> None:
    overall_pass = artifacts["overall_pass"]
    st.markdown(
        f"**Overall pass:** {badge_html('pass' if overall_pass else 'fail')}",
        unsafe_allow_html=True,
    )
    evaluated = artifacts.get("evaluated_phases") or []
    if evaluated:
        st.markdown("**Evaluated (gating) phases:** " + ", ".join(evaluated))

    blockers = artifacts.get("blockers") or []
    if blockers:
        st.markdown("**Blockers**")
        for b in blockers:
            st.markdown(
                f"- **{b['phase']}** {badge_html(b['status'])} "
                f"&nbsp; {b['n_findings']} findings",
                unsafe_allow_html=True,
            )
            for line in (b.get("preview") or []):
                st.markdown(f"  - {line}")

    sd = artifacts.get("scope_disclaimer")
    if sd:
        with st.expander("Scope disclaimer (what was / wasn't checked)"):
            st.markdown(f"**Sector standard.** {sd.get('sector_standard','—')}")
            if sd.get("checked"):
                st.markdown("**Checked:**")
                for c in sd["checked"]:
                    st.markdown(f"- {c}")
            if sd.get("not_checked"):
                st.markdown("**Not checked:**")
                for c in sd["not_checked"]:
                    st.markdown(f"- {c}")
            if sd.get("recommendation"):
                st.info(sd["recommendation"])


def render_phase_generic(artifacts: dict) -> None:
    """Fallback renderer for phase 1 / 2 / 9 (simple structures)."""
    if not artifacts:
        st.caption("_no artifacts_")
        return
    for key, val in artifacts.items():
        if isinstance(val, list) and val and not isinstance(val[0], (dict, list)):
            st.markdown(f"**{key}**")
            for v in val:
                st.markdown(f"- {v}")
        elif isinstance(val, (str, int, float, bool)) or val is None:
            st.markdown(f"**{key}.** {val}")
        else:
            with st.expander(f"{key} (raw)"):
                st.json(val)


def render_phase(
    phase_key: str,
    pr: dict,
    materiality_map: dict | None,
    chunks: dict[str, dict],
) -> None:
    st.markdown(
        f"### {PHASE_TITLES.get(phase_key, phase_key)} &nbsp; "
        f"{badge_html(pr['status'])}",
        unsafe_allow_html=True,
    )
    st.markdown("**Findings**")
    render_findings(pr.get("findings") or [])

    st.divider()
    artifacts = pr.get("artifacts") or {}

    if phase_key in ("phase3", "phase5", "phase6"):
        render_disclosure_verdicts(
            artifacts.get("disclosure_verdicts") or [],
            chunks,
            key_prefix=phase_key,
        )
    elif phase_key == "phase4":
        render_phase_4(artifacts, materiality_map)
    elif phase_key == "phase7":
        render_phase_7(artifacts)
    elif phase_key in ("phase8", "phase8_light"):
        render_phase_8(artifacts)
    elif phase_key == "final":
        render_phase_final(artifacts)
    else:
        render_phase_generic(artifacts)


# ---------------------------------------------------------------------------
# Page: Audit Report
# ---------------------------------------------------------------------------


def render_audit_header(report: dict) -> None:
    overall = report["overall"]
    badge = badge_html("pass" if overall["overall_pass"] else "fail", big=True)
    sector = ", ".join(report.get("sector_standards") or []) or "—"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"## 📄 {report['report_id']}",
        )
        st.markdown(
            f"**Claim pathway.** `{overall['claim_pathway']}` &nbsp;&nbsp; "
            f"**Sector.** {sector}",
        )
    with right:
        st.markdown(
            f"<div style='text-align:right;margin-top:18px'>{badge}</div>",
            unsafe_allow_html=True,
        )


def render_audit_page(variant: str, report_id: str) -> None:
    report = load_report(variant, report_id)
    chunks = load_chunks(report_id)
    overall = report["overall"]
    phase_results = report["phase_results"]
    materiality_map = report.get("materiality_map")

    render_audit_header(report)

    evaluated = overall.get("evaluated_phases") or {}
    evaluated_list = (
        list(evaluated.keys()) if isinstance(evaluated, dict) else list(evaluated)
    )
    # Final aggregator may carry the authoritative full list
    final_eval = (
        phase_results.get("final", {}).get("artifacts", {}).get("evaluated_phases")
    )
    if final_eval:
        evaluated_list = list(final_eval)

    st.markdown("**Phase status**")
    render_phase_pills(overall["all_phase_statuses"], evaluated_list)
    st.caption("● gate = phase gates overall pass/fail for this claim pathway")

    st.divider()

    phase_keys = [p for p in phase_results.keys()]
    tab_labels = [PHASE_TITLES.get(p, p) for p in phase_keys]
    tabs = st.tabs(tab_labels)
    for tab, phase_key in zip(tabs, phase_keys):
        with tab:
            render_phase(phase_key, phase_results[phase_key], materiality_map, chunks)

    st.divider()
    with st.expander("Raw JSON (compliance_report.json)"):
        st.json(report, expanded=False)


# ---------------------------------------------------------------------------
# Page: Experiment Overview
# ---------------------------------------------------------------------------


def render_experiment_page() -> None:
    st.markdown("## Experiment Overview")
    st.markdown(
        "Các bảng và biểu đồ chính từ Chương 4 của khóa luận "
        "(TN4 — calibrate arbiter, TN3 — NLI augmentation)."
    )

    st.markdown("### TN4 — Calibration arbiter LLM-as-judge")

    st.markdown(
        "**κ panel giữa ba arbiter và human reviewer trên 120 disagreement case**"
        " &nbsp; — Δκ(v2 − v1) = +0,111."
    )
    st.dataframe(TN4_KAPPA_PANEL_DF, width="stretch", hide_index=True)

    st.markdown(
        "**Sign reversal E1 (V_new − A0) trên cùng 2.550 disagreement case "
        "pre-fix** &nbsp; — swing = 14,5 pp giữa hai prompt arbiter."
    )
    st.dataframe(TN4_SIGN_REVERSAL_DF, width="stretch", hide_index=True)

    st.markdown(
        "**Phân bố nhãn marginal của ba arbiter (v1, v2, v3) và human "
        "reviewer trên cùng 120 disagreement case**"
    )
    st.image(
        str(FIGURES_DIR / "F5_arbiter_marginal.png"),
        caption=(
            "v3 chỉ chạy được 83 case trước khi đạt giới hạn RPD của OpenAI. "
            "v1/v3 over-use partial (71,7% và 73,5% vs human 36,7%); "
            "v2 over-use no_evidence (41,7% vs human 17,5%)."
        ),
        width="stretch",
    )

    st.divider()
    st.markdown("### TN3 — Hiệu quả NLI augmentation")

    st.markdown(
        "**Phân rã factorial 2 × 2 pre-fix dưới hybrid silver "
        "(n = 8.620, cluster bootstrap CI theo 14 báo cáo)**"
    )
    st.dataframe(TN3_FACTORIAL_DF, width="stretch", hide_index=True)

    st.markdown(
        "**Post-fix variant accuracy trên 137 case GT v2 human-only**"
    )
    st.dataframe(TN3_POSTFIX_ACC_DF, width="stretch", hide_index=True)

    st.markdown(
        "**Bug-fix lift paired pre/post trên 137 case GT v2 "
        "(post − pre accuracy)**"
    )
    st.dataframe(TN3_BUGFIX_LIFT_DF, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="GRAG Compliance Demo",
        page_icon="📋",
        layout="wide",
    )

    st.sidebar.markdown("## GRAG Compliance Demo")
    view_modes = ["Audit Report"]
    if SHOW_EXPERIMENT_OVERVIEW:
        view_modes.append("Experiment Overview")
    mode = st.sidebar.radio("View", view_modes, index=0)

    if mode == "Experiment Overview":
        render_experiment_page()
    else:
        runs = list_runs()
        variant = st.sidebar.selectbox(
            "Variant",
            list(runs.keys()),
            format_func=lambda v: VARIANT_LABELS.get(v, v),
        )
        report_id = st.sidebar.selectbox("Report", runs[variant])
        st.sidebar.caption(
            "Reads `compliance_report.json` from "
            f"`experiments/module3_nli/variant_runs/{variant}/{report_id}/`"
        )
        render_audit_page(variant, report_id)

    st.sidebar.divider()
    st.sidebar.caption(
        "Built on `compliance_report.json` (module3.v2). "
        "Citation drill-down uses local `metadata/report_units/<report>/report_chunks.json` "
        "— no Zilliz / Ollama needed."
    )


if __name__ == "__main__":
    main()
