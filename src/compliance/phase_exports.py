from datetime import datetime, timezone
from . import config

# Helpers 
def _common_header(state, phase_key):
    """Field chung cho mọi per-phase export."""
    pr_dict = state.get("phase_results", {}) or {}
    pr = pr_dict.get(phase_key)
    return {
        "report_id": state.get("report_id"),
        "output_version": config.OUTPUT_VERSION,
        "phase": phase_key,
        "phase_status": pr.status if pr is not None else None,
        "findings": list(pr.findings) if pr is not None else [],
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }

def _get_artifacts(state, phase_key):
    pr = (state.get("phase_results") or {}).get(phase_key)
    if pr is None:
        return {}
    return dict(pr.artifacts)

def _format_req_verdicts(rvs):
    """Flatten requirement_verdicts list cho phase 3/5/6 export."""
    return [
        {
            "requirement_id": rv.get("requirement_id", ""),
            "status": rv.get("status", ""),
            "rationale": rv.get("rationale", ""),
            "citations": list(rv.get("citations") or []),
            "confidence": rv.get("confidence"),
            "source": rv.get("source", ""),
        }
        for rv in rvs
    ]

def _format_coverage(cov):
    """Flatten coverage dict cho phase 3/5 export."""
    return {
        "present": bool(cov.get("present")),
        "omitted": bool(cov.get("omitted")),
        "location_raw": cov.get("location_raw"),
        "is_location_supported": cov.get("is_location_supported"),
        "no_pack_reason": cov.get("no_pack_reason"),
    }

# Phase 3 — Verify GRI 2 (2-1 ... 2-30)
def build_phase3_export(state):
    artifacts = _get_artifacts(state, "phase3")
    coverage = artifacts.get("coverage") or {}
    dv_list = artifacts.get("disclosure_verdicts") or []
    dv_by_id = {dv["disclosure_id"]: dv for dv in dv_list}

    all_ids = [f"2-{n}" for n in range(1, 31)]
    disclosures_out = []
    for did in all_ids:
        cov = coverage.get(did) or {}
        dv = dv_by_id.get(did, {})
        disclosures_out.append({
            "disclosure_id": did,
            "overall": dv.get("overall") or ("omitted" if cov.get("omitted") else None),
            "notes": dv.get("notes", ""),
            "evidence_unrecoverable": bool(dv.get("evidence_unrecoverable", False)),
            "coverage": _format_coverage(cov),
            "requirements": _format_req_verdicts(dv.get("requirement_verdicts") or []),
        })

    header = _common_header(state, "phase3")
    return {
        "report_id": header["report_id"],
        "standard": "GRI 2",
        "phase_status": header["phase_status"],
        "findings": header["findings"],
        "export_timestamp": header["export_timestamp"],
        "disclosures": disclosures_out,
    }

# Phase 4 — Determine material topics
def build_phase4_export(state):
    artifacts = _get_artifacts(state, "phase4")
    header = _common_header(state, "phase4")
    return {
        **header,
        "reported_material_topics": list(artifacts.get("reported_material_topics") or []),
        "reported_non_material_topics": list(artifacts.get("reported_non_material_topics") or []),
        "matches": {
            "tier1": list(artifacts.get("tier1_matches") or []),
            "tier2": list(artifacts.get("tier2_matches") or []),
            "tier3": list(artifacts.get("tier3_matches") or []),
        },
        "non_material_match_index": artifacts.get("non_material_match_index") or {},
        "materiality_map": artifacts.get("materiality_map") or {},
    }

# Phase 5 — GRI 3 disclosures per material topic (3-1, 3-2, 3-3)
def build_phase5_export(state):
    artifacts = _get_artifacts(state, "phase5")
    header = _common_header(state, "phase5")

    coverage = artifacts.get("coverage") or {}
    disclosures_out = []
    for dv in artifacts.get("disclosure_verdicts") or []:
        did = dv.get("disclosure_id")
        cov = coverage.get(did) or {}
        disclosures_out.append({
            "disclosure_id": did,
            "material_topic": dv.get("material_topic"),
            "overall": dv.get("overall"),
            "notes": dv.get("notes", ""),
            "evidence_unrecoverable": bool(dv.get("evidence_unrecoverable", False)),
            "coverage": _format_coverage(cov),
            "requirements": _format_req_verdicts(dv.get("requirement_verdicts") or []),
        })

    return {
        **header,
        "standard": "GRI 3",
        "n_material_topics": artifacts.get("n_material_topics", 0),
        "subreq_rollup": artifacts.get("subreq_rollup") or {},
        "structural_3_3_check": artifacts.get("structural_3_3_check") or [],
        "disclosures": disclosures_out,
    }

# Phase 6 — Topic Standard Disclosures per material topic
def build_phase6_export(state):
    artifacts = _get_artifacts(state, "phase6")
    header = _common_header(state, "phase6")

    disclosures_out = []
    for dv in artifacts.get("disclosure_verdicts") or []:
        disclosures_out.append({
            "disclosure_id": dv.get("disclosure_id"),
            "material_topic": dv.get("material_topic"),
            "overall": dv.get("overall"),
            "notes": dv.get("notes", ""),
            "evidence_unrecoverable": bool(dv.get("evidence_unrecoverable", False)),
            "requirements": _format_req_verdicts(dv.get("requirement_verdicts") or []),
        })

    return {
        **header,
        "topic_disclosure_map": artifacts.get("topic_disclosure_map") or {},
        "disclosures": disclosures_out,
        "reconcile_5b": artifacts.get("reconcile_5b") or [],
        "reconcile_5b_skipped_topics": artifacts.get("reconcile_5b_skipped_topics") or [],
        "non_material_mt_names": list(artifacts.get("non_material_mt_names") or []),
    }

# Phase 7 — Reasons for Omission
def build_phase7_export(state):
    artifacts = _get_artifacts(state, "phase7")
    header = _common_header(state, "phase7")
    return {
        **header,
        "missing_list": artifacts.get("missing_list") or [],
        "reported_omissions": artifacts.get("reported_omissions") or [],
        "invalid_reasons": artifacts.get("invalid_reasons") or [],
        "hard_fails": artifacts.get("hard_fails") or [],
        "reconcile_table": artifacts.get("reconcile_table") or [],
    }

# Phase 8 — GRI Content Index validation
def build_phase8_export(state):
    """Handle cả `phase8` (Appendix 1) lẫn `phase8_light` (Appendix 2).

    Variant nào trong state.phase_results sẽ được export — cả 2 ghi cùng
    filename (`content_index_results.json`) vì 1 pathway chỉ produce 1 trong 2.
    """
    phase_key = "phase8" if "phase8" in (state.get("phase_results") or {}) else "phase8_light"
    artifacts = _get_artifacts(state, phase_key)
    header = _common_header(state, phase_key)
    return {
        **header,
        "appendix_used": artifacts.get("appendix_used"),
        "n_items": artifacts.get("n_items", 0),
        "n_pass": artifacts.get("n_pass", 0),
        "checklist": artifacts.get("checklist") or [],
    }

# Registry cho io.write_phase_exports
# phase8 + phase8_light dùng chung builder + filename.
PHASE_EXPORT_BUILDERS = {
    "phase3": build_phase3_export,
    "phase4": build_phase4_export,
    "phase5": build_phase5_export,
    "phase6": build_phase6_export,
    "phase7": build_phase7_export,
    "phase8": build_phase8_export,
    "phase8_light": build_phase8_export,
}

__all__ = [
    "PHASE_EXPORT_BUILDERS",
    "build_phase3_export",
    "build_phase4_export",
    "build_phase5_export",
    "build_phase6_export",
    "build_phase7_export",
    "build_phase8_export",
]
