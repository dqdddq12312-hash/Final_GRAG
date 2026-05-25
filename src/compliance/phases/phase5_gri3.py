import logging

from src.compliance import judges, registries, retrieval, rules
from src.compliance.phases.phase3_gri2 import (
    _classify_no_pack_reason,
    aggregate_disclosure_overall,
    aggregate_phase_status,
)
from src.compliance.state import (
    DisclosureBatchVerdict,
    DisclosureVerdict,
    PendingOmission,
    PhaseResult,
)

logger = logging.getLogger(__name__)

# Universal GRI 3 disclosure — 1 row / report, material_topic=None
EXPECTED_UNIVERSAL = ("3-1", "3-2")
# Per-material-topic disclosure — 1 row / material_topic
PER_MT_DISCLOSURE_ID = "3-3"
GRI_3_PREFIX = "GRI 3:"
STANDARD_ID_SHORT = "GRI 3"

# GCI lookup helpers
def _resolve_disclosure_name(reqs, d_id):
    """Mirror phase3_gri2._resolve_disclosure_name."""
    for r in reqs:
        name = r.get("disclosure_name") or r.get("Disclosure Name")
        if name:
            return str(name).strip()
    return d_id

def _find_universal_row(gci, d_id):
    """First universal GCI row cho 1 GRI 3 disclosure.

    Universal = gri_standard startswith "GRI 3:" AND disclosure_id == d_id
    AND material_topic IS None (treat empty/blank string là None defensive).
    """
    for r in gci:
        mt = r.get("material_topic")
        if isinstance(mt, str) and not mt.strip():
            mt = None
        if (
            (r.get("gri_standard") or "").startswith(GRI_3_PREFIX)
            and r.get("disclosure_id") == d_id
            and mt is None
        ):
            return r
    return None

def _find_3_3_rows_for(gci, material_topic):
    """Mọi 3-3 GCI row cho 1 material topic (preserve order)."""
    return [
        r
        for r in gci
        if (r.get("gri_standard") or "").startswith(GRI_3_PREFIX)
        and r.get("disclosure_id") == "3-3"
        and r.get("material_topic") == material_topic
    ]

def _attempt_pack(row, all_chunks):
    """Build page pack cho row nếu hợp lệ; ngược lại trả [].

    Anchor trên raw GCI field giống Phase 3 step 3 (omission, external_ref,
    blank location_raw, empty page_list).
    """
    if row is None:
        return []
    if (
        rules.is_disclosure_omitted(row)
        or (row.get("location_type") or "") == "external_document_reference"
        or not str(row.get("location_raw") or "").strip()
        or not (row.get("page_list") or [])
    ):
        return []
    return retrieval.build_page_pack(row, all_chunks)

# Step helpers
def _check_universal_coverage(gci):
    """Check coverage 3-1/3-2; trả (coverage, findings, hard_fail, pending, universal_rows)."""
    coverage = {}
    coverage_findings = []
    hard_fail = False
    pending = []
    universal_rows = {}

    for d_id in EXPECTED_UNIVERSAL:
        row = _find_universal_row(gci, d_id)
        universal_rows[d_id] = row
        present = row is not None
        omitted = bool(row and rules.is_disclosure_omitted(row))
        coverage[d_id] = {
            "present": present,
            "omitted": omitted,
            "location_raw": (row or {}).get("location_raw"),
            "page_list": (row or {}).get("page_list") or [],
            "is_location_supported": (row or {}).get("is_location_supported"),
            "location_type": (row or {}).get("location_type"),
        }
        if not present:
            req_letter = "a" if d_id == "3-1" else "b"
            coverage_findings.append(
                f"{d_id}: missing from GCI (Req 4-{req_letter} violation)"
            )
        elif omitted:
            # 3-1/3-2 ∈ GRI3_HARD_FAIL_OMIT_BLOCKED
            hard_fail = True
            coverage_findings.append(
                f"hard-fail: {d_id} is omitted but in "
                f"GRI3_HARD_FAIL_OMIT_BLOCKED"
            )
            pending.append(
                PendingOmission(
                    source_phase="phase5",
                    standard_id=STANDARD_ID_SHORT,
                    disclosure_id=d_id,
                    material_topic=None,
                    omission_reason=row.get("omission_reason"),
                    omission_explanation=row.get("omission_explanation"),
                    row=row,
                )
            )
        else:
            # Appendix-1 vi parity
            location_type = (row.get("location_type") or "")
            location_raw = (row.get("location_raw") or "")
            if (
                location_type != "external_document_reference"
                and not str(location_raw).strip()
            ):
                coverage_findings.append(
                    f"{d_id}: present but missing location "
                    f"(Appendix-1 vi violation)"
                )

    return coverage, coverage_findings, hard_fail, pending, universal_rows

def _check_3_3_structural(gci, material_topics):
    """Structural check 3-3 per-MT (sub-plan 03b internalised).

    Trả (structural_3_3_check, rows_3_3_per_mt, coverage_findings, pending).
    3-3 IS omittable per material topic — omission KHÔNG hard-fail (Phase 7 validate).
    """
    structural_3_3_check = []
    rows_3_3_per_mt = {}
    coverage_findings = []
    pending = []

    for mt in material_topics:
        rows = _find_3_3_rows_for(gci, mt)
        rows_3_3_per_mt[mt] = rows
        n = len(rows)
        if n == 1:
            status_topic = "ok"
        elif n == 0:
            status_topic = "missing"
            coverage_findings.append(f"material topic '{mt}' has no 3-3 row (Req 4-c)")
        else:
            status_topic = "duplicate"
            coverage_findings.append(
                f"material topic '{mt}' has {n} 3-3 rows (expected 1) (Req 4-c)"
            )
        structural_3_3_check.append({
            "material_topic": mt,
            "n_3_3_rows": n,
            "status": status_topic,
            "evidence_disclosure_ids": [r.get("disclosure_id") for r in rows],
            "evidence_pages": [r.get("page_list") or [] for r in rows],
            "evidence_location_raw": [r.get("location_raw") for r in rows],
        })
        for r in rows:
            if rules.is_disclosure_omitted(r):
                pending.append(
                    PendingOmission(
                        source_phase="phase5",
                        standard_id=STANDARD_ID_SHORT,
                        disclosure_id="3-3",
                        material_topic=mt,
                        omission_reason=r.get("omission_reason"),
                        omission_explanation=r.get("omission_explanation"),
                        row=r,
                    )
                )

    return structural_3_3_check, rows_3_3_per_mt, coverage_findings, pending

def _build_judge_units(universal_rows, rows_3_3_per_mt, material_topics, registry, all_chunks):
    """Build judge_unit list cho 3-1/3-2 (universal) và 3-3 (per-MT row)."""
    judge_units = []
    reg_3_3 = registry.get("3-3", [])

    for d_id in EXPECTED_UNIVERSAL:
        row = universal_rows[d_id]
        pack = _attempt_pack(row, all_chunks)
        no_pack_reason = _classify_no_pack_reason(row, pack)
        judge_units.append({
            "d_id": d_id,
            "material_topic": None,
            "registry_reqs": registry.get(d_id, []),
            "row": row,
            "pack": pack,
            "no_pack_reason": no_pack_reason,
        })

    # 3-3: 1 unit / (mt, row). Khi mt có 0 row → emit deterministic
    # missing_disclosure unit để disclosure_verdicts vẫn có row cho (mt, 3-3).
    for mt in material_topics:
        rows = rows_3_3_per_mt[mt]
        if not rows:
            judge_units.append({
                "d_id": "3-3",
                "material_topic": mt,
                "registry_reqs": reg_3_3,
                "row": None,
                "pack": [],
                "no_pack_reason": "missing_disclosure",
            })
            continue
        for row in rows:
            pack = _attempt_pack(row, all_chunks)
            no_pack_reason = _classify_no_pack_reason(row, pack)
            judge_units.append({
                "d_id": "3-3",
                "material_topic": mt,
                "registry_reqs": reg_3_3,
                "row": row,
                "pack": pack,
                "no_pack_reason": no_pack_reason,
            })

    return judge_units

def _build_dedupe(judge_units):
    """Tạo unique_judge_keys + fan_out_map từ judge_units.

    Key schema:
    - 3-1/3-2: (disc,)
    - 3-3 real: (disc, pages_tuple, material_topic) — mt trong key vì đi vào
      prompt (bugfix tn3_postfix: key cũ chỉ có (disc, pages) → mọi topic
      trùng page range nhận IDENTICAL verdict, không condition được trên mt).
    - 3-3 deterministic (no_pack_reason set): ("3-3-det", idx) — unique per unit.
    """
    unique_judge_keys = {}
    fan_out_map = {}

    for idx, unit in enumerate(judge_units):
        if unit["d_id"] in EXPECTED_UNIVERSAL:
            dedupe_key = (unit["d_id"],)
        elif unit["d_id"] == "3-3" and unit["no_pack_reason"] is None:
            row = unit["row"]
            pages_tuple = tuple(sorted(int(p) for p in (row.get("page_list") or [])))
            dedupe_key = ("3-3", pages_tuple, unit["material_topic"])
        else:
            dedupe_key = ("3-3-det", idx)

        unique_judge_keys.setdefault(dedupe_key, unit)
        fan_out_map.setdefault(dedupe_key, []).append(unit)

    return unique_judge_keys, fan_out_map

async def _judge_unit(dedupe_key, unit, report_id):
    """Gọi LLM judge cho 1 unique unit; skip nếu deterministic (no_pack_reason set)."""
    d_id = unit["d_id"]
    if unit["no_pack_reason"] is not None:
        return dedupe_key, None
    d_name = _resolve_disclosure_name(unit["registry_reqs"], d_id)
    try:
        batch = await judges.evidence_judge.run_disclosure_batch(
            disclosure_id=d_id,
            disclosure_name=d_name,
            page_pack=unit["pack"],
            registry=unit["registry_reqs"],
            anchor=None,
            report_id=report_id,
            material_topic=unit["material_topic"],
        )
        return dedupe_key, batch
    except Exception as e:
        logger.exception("phase5 judge_one failed for %s: %s", dedupe_key, e)
        return dedupe_key, {"_error": repr(e)}

def _assemble_disclosure_verdicts(judged_by_key, fan_out_map):
    """Map judged batches → DisclosureVerdict list; trả (verdicts, has_fail, has_partial)."""
    disclosure_verdicts = []
    has_fail = False
    has_partial_or_no_evidence_non_omitted = False

    for dedupe_key, batch in judged_by_key.items():
        for unit in fan_out_map[dedupe_key]:
            row = unit["row"]
            d_id = unit["d_id"]
            mt = unit["material_topic"]
            no_pack_reason = unit["no_pack_reason"]

            # Exception path
            if isinstance(batch, dict) and "_error" in batch:
                has_fail = True
                disclosure_verdicts.append(
                    DisclosureVerdict(
                        disclosure_id=d_id,
                        standard_id=STANDARD_ID_SHORT,
                        requirement_verdicts=[],
                        overall="fail",
                        notes=f"exception: {batch['_error']}",
                        material_topic=mt,
                    )
                )
                continue

            # No-pack / deterministic path
            if batch is None or (
                isinstance(batch, DisclosureBatchVerdict)
                and not batch.requirement_verdicts
            ):
                unrecoverable = False
                is_omitted = bool(row and rules.is_disclosure_omitted(row))
                if is_omitted or no_pack_reason == "omitted":
                    overall = "omitted"
                elif no_pack_reason == "external_document_reference":
                    overall = "external_verification"
                elif no_pack_reason in {"missing_disclosure", "unsupported_location"}:
                    overall = "fail"
                    has_fail = True
                elif no_pack_reason in {"page_list_unparseable", "page_list_no_matching_chunks"}:
                    overall = "no_evidence"
                    unrecoverable = True
                else:
                    overall = "fail"
                    has_fail = True

                notes_parts = (
                    [f"no page pack: {no_pack_reason}"]
                    if no_pack_reason
                    else ["no page pack"]
                )
                if unrecoverable:
                    notes_parts.append("system limitation - human review required")
                disclosure_verdicts.append(
                    DisclosureVerdict(
                        disclosure_id=d_id,
                        standard_id=STANDARD_ID_SHORT,
                        requirement_verdicts=[],
                        overall=overall,
                        notes="; ".join(notes_parts),
                        evidence_unrecoverable=unrecoverable,
                        material_topic=mt,
                    )
                )
                continue

            # Normal LLM path
            statuses = [v.status for v in batch.requirement_verdicts]
            overall = aggregate_disclosure_overall(statuses)
            is_omitted = bool(row and rules.is_disclosure_omitted(row))
            if overall == "fail":
                has_fail = True
            elif overall in {"partial", "no_evidence"} and not is_omitted:
                has_partial_or_no_evidence_non_omitted = True

            disclosure_verdicts.append(
                DisclosureVerdict(
                    disclosure_id=d_id,
                    standard_id=STANDARD_ID_SHORT,
                    requirement_verdicts=list(batch.requirement_verdicts),
                    overall=overall,
                    notes="",
                    evidence_unrecoverable=False,
                    material_topic=mt,
                )
            )

    return disclosure_verdicts, has_fail, has_partial_or_no_evidence_non_omitted

def _build_subreq_rollup(disclosure_verdicts, material_topics, structural_3_3_check):
    """Tổng hợp 3 sub-req rollup (4-a/4-b/4-c) từ disclosure verdicts."""
    dv_3_1 = next(
        (dv for dv in disclosure_verdicts if dv.disclosure_id == "3-1" and dv.material_topic is None),
        None,
    )
    dv_3_2 = next(
        (dv for dv in disclosure_verdicts if dv.disclosure_id == "3-2" and dv.material_topic is None),
        None,
    )
    dvs_3_3 = [dv for dv in disclosure_verdicts if dv.disclosure_id == "3-3"]

    overall_4a = dv_3_1.overall if dv_3_1 else "fail"
    overall_4b = dv_3_2.overall if dv_3_2 else "fail"

    structural_3_3_fail = any(s["status"] != "ok" for s in structural_3_3_check)
    n_pass_4c = sum(1 for dv in dvs_3_3 if dv.overall == "pass")
    n_fail_4c = sum(1 for dv in dvs_3_3 if dv.overall == "fail")
    n_partial_4c = sum(
        1
        for dv in dvs_3_3
        if dv.overall in {"partial", "no_evidence"}
        and not dv.evidence_unrecoverable
    )
    n_omitted_4c = sum(1 for dv in dvs_3_3 if dv.overall == "omitted")
    n_extver_4c = sum(1 for dv in dvs_3_3 if dv.overall == "external_verification")
    n_unrec_4c = sum(1 for dv in dvs_3_3 if dv.evidence_unrecoverable)

    if structural_3_3_fail or n_fail_4c > 0:
        overall_4c = "fail"
    elif n_partial_4c > 0:
        overall_4c = "partial"
    else:
        overall_4c = "pass"

    return {
        "4-a": {
            "disclosure_id": "3-1",
            "overall": overall_4a,
            "n_reqs": len(dv_3_1.requirement_verdicts) if dv_3_1 else 0,
        },
        "4-b": {
            "disclosure_id": "3-2",
            "overall": overall_4b,
            "n_reqs": len(dv_3_2.requirement_verdicts) if dv_3_2 else 0,
        },
        "4-c": {
            "disclosure_id": "3-3",
            "overall": overall_4c,
            "n_topics": len(material_topics),
            "n_pass": n_pass_4c,
            "n_partial": n_partial_4c,
            "n_fail": n_fail_4c,
            "n_omitted": n_omitted_4c,
            "n_external_verification": n_extver_4c,
            "n_unrecoverable": n_unrec_4c,
            "structural_status": "ok" if not structural_3_3_fail else "violations",
        },
    }

def _summary_findings(disclosure_verdicts, pending):
    """Surface fail/partial verdict thành finding cho compliance_summary.csv (mirror Phase 3)."""
    def _label(dv):
        return (
            f"{dv.disclosure_id}({dv.material_topic})"
            if dv.material_topic
            else dv.disclosure_id
        )

    pending_keys = {(po.disclosure_id, po.material_topic) for po in pending}
    failed_dvs = [dv for dv in disclosure_verdicts if dv.overall == "fail"]
    partial_dvs = [
        dv
        for dv in disclosure_verdicts
        if dv.overall in {"partial", "no_evidence"}
        and not dv.evidence_unrecoverable
        and (dv.disclosure_id, dv.material_topic) not in pending_keys
    ]
    findings = []
    if failed_dvs:
        findings.append(
            f"4-x: {len(failed_dvs)} disclosure(s) failed: "
            + ", ".join(_label(dv) for dv in failed_dvs)
        )
    if partial_dvs:
        findings.append(
            f"4-x: {len(partial_dvs)} disclosure(s) partial / no_evidence: "
            + ", ".join(_label(dv) for dv in partial_dvs)
        )
    return findings

def _count_cache_hits(disclosure_verdicts):
    """Đếm cache hit trong requirement verdicts."""
    return sum(
        1
        for dv in disclosure_verdicts
        for rv in dv.requirement_verdicts
        if rv.source == "cache"
    )

# Phase node
async def phase5_gri3_node(state):
    """Phase 5 entry-point — xem module docstring."""
    inputs = state.get("inputs") or {}
    gci = list(inputs.get("gci") or [])
    all_chunks = list(inputs.get("all_chunks_for_report") or [])
    report_id = state["report_id"]
    materiality_map = state.get("materiality_map")

    # Defensive: Phase 4 phải populate materiality_map
    if materiality_map is None:
        pr = PhaseResult(
            phase="phase5",
            status="fail",
            findings=["phase5: materiality_map not populated by Phase 4"],
            artifacts={"n_material_topics": 0},
        )
        return {"phase_results": {**state.get("phase_results", {}), "phase5": pr}}

    material_topics = list(materiality_map.material_topics)

    # Empty material_topics → Phase 4 đã fail 3-a; Phase 5 không có gì
    # verify cho 4-c, emit clear finding
    if not material_topics:
        pr = PhaseResult(
            phase="phase5",
            status="fail",
            findings=[
                "phase5: skipped - Phase 4 reported zero material topics "
                "(4-c has nothing to verify)"
            ],
            artifacts={"n_material_topics": 0},
        )
        return {"phase_results": {**state.get("phase_results", {}), "phase5": pr}}

    # Bước 1: build registry GRI 3 / shall
    registry = registries.build_disclosure_registry("GRI 3", "shall")

    # Bước 2: coverage + omission collect
    coverage, univ_findings, hard_fail, pending, universal_rows = _check_universal_coverage(gci)
    struct_check, rows_3_3_per_mt, struct_findings, pending_3_3 = _check_3_3_structural(gci, material_topics)
    pending = pending + pending_3_3
    coverage_findings = univ_findings + struct_findings

    coverage["structural_3_3_check"] = struct_check
    coverage_check_passed = (not hard_fail) and (len(coverage_findings) == 0)

    # Bước 3-4: build judge units + dedupe
    judge_units = _build_judge_units(universal_rows, rows_3_3_per_mt, material_topics, registry, all_chunks)
    unique_judge_keys, fan_out_map = _build_dedupe(judge_units)

    # Bước 5: sequential LLM judge (anchor=None)
    judged = []
    for dedupe_key, unit in unique_judge_keys.items():
        judged.append(await _judge_unit(dedupe_key, unit, report_id))
    judged_by_key = dict(judged)

    # Bước 6: build DisclosureVerdicts (fan-out cho 3-3 dedupe)
    disclosure_verdicts, has_fail, has_partial_or_no_evidence_non_omitted = (
        _assemble_disclosure_verdicts(judged_by_key, fan_out_map)
    )

    # Bước 7: aggregate (3 sub-req rollup + phase status)
    subreq_rollup = _build_subreq_rollup(disclosure_verdicts, material_topics, struct_check)
    status = aggregate_phase_status(
        coverage_check_passed=coverage_check_passed,
        hard_fail=hard_fail,
        has_fail=has_fail,
        has_partial_or_no_evidence_non_omitted=has_partial_or_no_evidence_non_omitted,
    )

    # Bước 8: summary findings + cache-hit counter
    findings = _summary_findings(disclosure_verdicts, pending)
    n_cache_hits = _count_cache_hits(disclosure_verdicts)

    pr = PhaseResult(
        phase="phase5",
        status=status,
        findings=[*findings, *coverage_findings],
        artifacts={
            "coverage": coverage,
            "coverage_check_passed": coverage_check_passed,
            "coverage_findings": coverage_findings,
            "structural_3_3_check": struct_check,
            "subreq_rollup": subreq_rollup,
            "disclosure_verdicts": [dv.model_dump() for dv in disclosure_verdicts],
            "n_material_topics": len(material_topics),
            "n_cache_hits": n_cache_hits,
        },
    )
    return {
        "phase_results": {**state.get("phase_results", {}), "phase5": pr},
        "pending_omissions": [*state.get("pending_omissions", []), *pending],
    }

__all__ = [
    "EXPECTED_UNIVERSAL",
    "PER_MT_DISCLOSURE_ID",
    "GRI_3_PREFIX",
    "STANDARD_ID_SHORT",
    "phase5_gri3_node",
]
