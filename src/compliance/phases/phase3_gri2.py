import logging

from src.compliance import config, judges, registries, retrieval, rules
from src.compliance.state import (
    DisclosureBatchVerdict,
    DisclosureVerdict,
    PendingOmission,
    PhaseResult,
)

logger = logging.getLogger(__name__)

ANCHOR_AWARE = config.ANCHOR_INJECTION_WHITELIST
EXPECTED = {f"2-{i}" for i in range(1, 31)}
EXPECTED_SORTED = tuple(sorted(EXPECTED, key=lambda d: int(d.split("-")[1])))

# Helpers
def _gci_row_for(gci, d_id):
    """Tìm GCI row có disclosure_id == d_id"""
    for r in gci:
        if str(r.get("disclosure_id") or "").strip() == d_id:
            return r
    return None

def _classify_no_pack_reason(gci_row, pack):
    """Phân loại lý do không build được page pack cho disclosure."""
    if pack:
        return None
    if gci_row is None:
        return "missing_disclosure"
    if rules.is_disclosure_omitted(gci_row):
        return "omitted"
    location_type = (gci_row.get("location_type") or "")
    if location_type == "external_document_reference":
        return "external_document_reference"
    location_raw = (gci_row.get("location_raw") or "")
    if not str(location_raw).strip():
        return "unsupported_location"
    page_list = gci_row.get("page_list") or []
    if not page_list:
        return "page_list_unparseable"
    return "page_list_no_matching_chunks"

def _coverage_check(gci, coverage):
    """Kiểm tra 30 disclosure GRI 2 có mặt + valid trong GCI."""
    coverage_findings = []
    hard_fail = False

    for d_id in EXPECTED_SORTED:
        info = coverage[d_id]
        gci_row = _gci_row_for(gci, d_id)

        if not info["present"] and not info["omitted"]:
            coverage_findings.append(
                f"{d_id}: missing from GCI and not declared as omitted"
            )
            continue

        if info["omitted"] and d_id in config.GRI2_HARD_FAIL_OMIT_BLOCKED:
            hard_fail = True
            coverage_findings.append(
                f"hard-fail: {d_id} is omitted but \u2208 GRI2_HARD_FAIL_OMIT_BLOCKED"
            )

        if info["present"] and not info["omitted"]:
            location_type = (gci_row or {}).get("location_type") or ""
            location_raw = (info.get("location_raw") or "")
            if (
                location_type != "external_document_reference"
                and not str(location_raw).strip()
            ):
                coverage_findings.append(
                    f"{d_id}: missing location without omission "
                    f"(Appendix-1 vi violation)"
                )

    coverage_check_passed = (not hard_fail) and (len(coverage_findings) == 0)
    return coverage_check_passed, coverage_findings, hard_fail

def _resolve_disclosure_name(reqs, d_id):
    """Trả tên disclosure từ registry, fallback về d_id."""
    for r in reqs:
        name = r.get("disclosure_name") or r.get("Disclosure Name")
        if name:
            return str(name).strip()
    return d_id

# Aggregators
def aggregate_disclosure_overall(statuses):
    """Tổng hợp list status requirement → overall disclosure status."""
    if not statuses:
        return "no_evidence"
    if any(s == "fail" for s in statuses):
        return "fail"
    if any(s == "partial" for s in statuses):
        return "partial"
    if all(s == "no_evidence" for s in statuses):
        return "no_evidence"
    if any(s == "no_evidence" for s in statuses):
        return "partial"
    return "pass"

def aggregate_phase_status(*, coverage_check_passed, hard_fail, has_fail, has_partial_or_no_evidence_non_omitted):
    """Tổng hợp phase status từ aggregated disclosure verdicts + coverage check."""
    if hard_fail or (not coverage_check_passed) or has_fail:
        return "fail"
    if has_partial_or_no_evidence_non_omitted:
        return "partial"
    return "pass"

# Step helpers
def _collect_pending_omissions(gci):
    """Collect PendingOmission cho tất cả GRI 2 disclosure bị omit."""
    pending = []
    for d_id in EXPECTED_SORTED:
        gci_row = _gci_row_for(gci, d_id)
        if gci_row and rules.is_disclosure_omitted(gci_row):
            pending.append(
                PendingOmission(
                    source_phase="phase3",
                    standard_id="GRI 2",
                    disclosure_id=d_id,
                    material_topic=None,
                    omission_reason=gci_row.get("omission_reason"),
                    omission_explanation=gci_row.get("omission_explanation"),
                    row=gci_row,
                )
            )
    return pending

def _build_page_packs(gci, all_chunks, coverage):
    """Build page pack cho mỗi disclosure + ghi no_pack_reason vào coverage."""
    page_packs = {}
    no_pack_reasons = {}
    for d_id in EXPECTED:
        gci_row = _gci_row_for(gci, d_id)
        location_raw = (gci_row or {}).get("location_raw") or ""
        if (
            gci_row is not None
            and not rules.is_disclosure_omitted(gci_row)
            and (gci_row.get("location_type") or "") != "external_document_reference"
            and str(location_raw).strip()
            and (gci_row.get("page_list") or [])
        ):
            page_packs[d_id] = retrieval.build_page_pack(gci_row, all_chunks)
        else:
            page_packs[d_id] = []

        reason = _classify_no_pack_reason(gci_row, page_packs[d_id])
        if reason:
            no_pack_reasons[d_id] = reason
            coverage[d_id]["no_pack_reason"] = reason
    return page_packs, no_pack_reasons

async def _run_anchor(page_packs, registry, report_id):
    """Run scope anchor judge trên disclosure 2-2; trả (batch_2_2, anchor, findings)."""
    findings = []
    anchor = None
    if page_packs.get("2-2"):
        try:
            batch_2_2, anchor = await judges.evidence_judge.run_anchor(
                page_pack=page_packs["2-2"],
                registry=registry.get("2-2", []),
                report_id=report_id,
            )
        except Exception as e:
            logger.exception("phase3 anchor judge failed: %s", e)
            findings.append(
                f"anchor judge raised \u2014 falling back to anchor=None: {e!r}"
            )
            batch_2_2 = DisclosureBatchVerdict(
                disclosure_id="2-2", requirement_verdicts=[]
            )
    else:
        findings.append("disclosure 2-2 has no page pack \u2014 scope anchor unavailable")
        batch_2_2 = DisclosureBatchVerdict(disclosure_id="2-2", requirement_verdicts=[])
    return batch_2_2, anchor, findings

async def _judge_others(page_packs, registry, anchor, report_id, batch_2_2):
    """Sequential LLM judge cho 29 disclosure còn lại (skip 2-2)."""
    async def _judge_one(d_id):
        pack = page_packs.get(d_id) or []
        if not pack:
            return d_id, None
        d_name = _resolve_disclosure_name(registry.get(d_id, []), d_id)
        try:
            batch = await judges.evidence_judge.run_disclosure_batch(
                disclosure_id=d_id,
                disclosure_name=d_name,
                page_pack=pack,
                registry=registry.get(d_id, []),
                anchor=anchor if d_id in ANCHOR_AWARE else None,
                report_id=report_id,
            )
            return d_id, batch
        except Exception as e:
            logger.exception("phase3 judge_one failed for %s: %s", d_id, e)
            return d_id, {"_error": repr(e)}

    all_batches = {"2-2": batch_2_2}
    for d_id in EXPECTED_SORTED:
        if d_id == "2-2":
            continue
        result_id, batch = await _judge_one(d_id)
        all_batches[result_id] = batch
    return all_batches

def _assemble_disclosure_verdicts(all_batches, pending, no_pack_reasons):
    """Map batches → DisclosureVerdict list; trả (verdicts, has_fail, has_partial)."""
    omitted_ids = {po.disclosure_id for po in pending}
    disclosure_verdicts = []
    has_fail = False
    has_partial_or_no_evidence_non_omitted = False

    for d_id in EXPECTED_SORTED:
        batch = all_batches.get(d_id)
        is_omitted = d_id in omitted_ids
        no_pack_reason = no_pack_reasons.get(d_id)

        # Exception path
        if isinstance(batch, dict) and "_error" in batch:
            has_fail = True
            disclosure_verdicts.append(
                DisclosureVerdict(
                    disclosure_id=d_id,
                    standard_id="GRI 2",
                    requirement_verdicts=[],
                    overall="fail",
                    notes=f"exception: {batch['_error']}",
                )
            )
            continue

        # No-pack / empty-batch path.
        # Reporter-violation buckets → fail: missing_disclosure / unsupported_location.
        # System-limitation buckets → no_evidence + evidence_unrecoverable=True:
        #   page_list_unparseable / page_list_no_matching_chunks.
        # Else (pack có nhưng LLM trả về empty requirement_verdicts) → fail.
        if batch is None or (
            isinstance(batch, DisclosureBatchVerdict)
            and not batch.requirement_verdicts
        ):
            unrecoverable = False
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
                # Pack tồn tại nhưng LLM trả về empty requirement list.
                overall = "fail"
                has_fail = True
            notes = f"no page pack: {no_pack_reason}" if no_pack_reason else "no requirement verdicts (empty batch)"
            if unrecoverable:
                notes += "; system limitation \u2014 human review required"
            disclosure_verdicts.append(
                DisclosureVerdict(
                    disclosure_id=d_id,
                    standard_id="GRI 2",
                    requirement_verdicts=[],
                    overall=overall,
                    notes=notes,
                    evidence_unrecoverable=unrecoverable,
                )
            )
            continue

        # Normal path
        statuses = [v.status for v in batch.requirement_verdicts]
        overall = aggregate_disclosure_overall(statuses)
        if overall == "fail":
            has_fail = True
        elif overall in {"partial", "no_evidence"} and not is_omitted:
            has_partial_or_no_evidence_non_omitted = True

        disclosure_verdicts.append(
            DisclosureVerdict(
                disclosure_id=d_id,
                standard_id="GRI 2",
                requirement_verdicts=list(batch.requirement_verdicts),
                overall=overall,
                evidence_unrecoverable=False,
            )
        )

    return disclosure_verdicts, has_fail, has_partial_or_no_evidence_non_omitted

def _summary_findings(disclosure_verdicts, pending):
    """Surface fail/partial verdict thành finding cho compliance_summary.csv."""
    omitted_ids = {po.disclosure_id for po in pending}
    findings = []
    failed_dvs = [dv for dv in disclosure_verdicts if dv.overall == "fail"]
    partial_dvs = [
        dv
        for dv in disclosure_verdicts
        if dv.overall in {"partial", "no_evidence"}
        and dv.disclosure_id not in omitted_ids
        and not dv.evidence_unrecoverable
    ]
    if failed_dvs:
        findings.append(
            f"3-c: {len(failed_dvs)} disclosure(s) failed: "
            + ", ".join(dv.disclosure_id for dv in failed_dvs)
        )
    if partial_dvs:
        findings.append(
            f"3-c: {len(partial_dvs)} disclosure(s) partial / no_evidence "
            f"(non-omitted): "
            + ", ".join(dv.disclosure_id for dv in partial_dvs)
        )
    return findings

def _info_findings(disclosure_verdicts):
    """Tóm tắt condition_not_applicable / reported_non_existence cho operator."""
    n_cna = sum(
        1
        for dv in disclosure_verdicts
        for rv in dv.requirement_verdicts
        if rv.decision_path == "condition_not_applicable"
    )
    n_rne = sum(
        1
        for dv in disclosure_verdicts
        for rv in dv.requirement_verdicts
        if rv.decision_path == "reported_non_existence"
    )
    findings = []
    if n_cna:
        findings.append(
            f"3-info: {n_cna} requirement(s) marked condition_not_applicable "
            f"(condition explicitly negated in evidence)"
        )
    if n_rne:
        findings.append(
            f"3-info: {n_rne} requirement(s) marked reported_non_existence "
            f"(item explicitly stated as non-existent in evidence)"
        )
    return findings

# Phase node
async def phase3_gri2_node(state):
    """Judge 30 GRI 2 disclosures + scope anchor (2-2)."""
    inputs = state.get("inputs") or {}
    gci = list(inputs.get("gci") or [])
    all_chunks = list(inputs.get("all_chunks_for_report") or [])
    report_id = state["report_id"]

    # Bước 1: registry + coverage + pending omissions
    registry = registries.build_disclosure_registry("GRI 2", "shall")
    coverage = rules.check_disclosure_presence(gci, expected_ids=EXPECTED)
    pending = _collect_pending_omissions(gci)
    coverage_check_passed, coverage_findings, hard_fail = _coverage_check(gci, coverage)

    # Bước 2: page packs
    page_packs, no_pack_reasons = _build_page_packs(gci, all_chunks, coverage)

    # Bước 3: scope anchor (2-2) → anchor context cho downstream disclosures
    batch_2_2, anchor, anchor_findings = await _run_anchor(page_packs, registry, report_id)

    # Bước 4: sequential judge cho 29 disclosure còn lại
    all_batches = await _judge_others(page_packs, registry, anchor, report_id, batch_2_2)

    # Bước 5: assemble verdicts + phase status
    disclosure_verdicts, has_fail, has_partial = _assemble_disclosure_verdicts(
        all_batches, pending, no_pack_reasons
    )
    status = aggregate_phase_status(
        coverage_check_passed=coverage_check_passed,
        hard_fail=hard_fail,
        has_fail=has_fail,
        has_partial_or_no_evidence_non_omitted=has_partial,
    )

    findings = (
        anchor_findings
        + _summary_findings(disclosure_verdicts, pending)
        + _info_findings(disclosure_verdicts)
    )

    n_cache_hits = sum(
        1
        for dv in disclosure_verdicts
        for rv in dv.requirement_verdicts
        if rv.source == "cache"
    )

    pr = PhaseResult(
        phase="phase3",
        status=status,
        findings=[*findings, *coverage_findings],
        artifacts={
            "coverage": coverage,
            "coverage_check_passed": coverage_check_passed,
            "coverage_findings": coverage_findings,
            "scope_anchor": anchor.model_dump() if anchor is not None else None,
            "disclosure_verdicts": [dv.model_dump() for dv in disclosure_verdicts],
            "n_cache_hits": n_cache_hits,
        },
    )

    return {
        "scope_anchor": anchor,
        "phase_results": {**state.get("phase_results", {}), "phase3": pr},
        "pending_omissions": [*state.get("pending_omissions", []), *pending],
    }

__all__ = [
    "ANCHOR_AWARE",
    "EXPECTED",
    "aggregate_disclosure_overall",
    "aggregate_phase_status",
    "phase3_gri2_node",
]
