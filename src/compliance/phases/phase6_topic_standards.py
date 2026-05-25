"""Phase 6 — Topic Standard Disclosures per Material Topic.

Phase LLM lớn nhất: cover mọi topic-standard disclosure (GRI 1xx/2xx/3xx/4xx)
reported dưới mỗi material topic + GRI 11 promoted-disclosure reconcile
(5-b strict 'Not applicable').

Pipeline:
    1. Topic ↔ disclosure map (group GCI by material_topic; loại
       disclosure_id == '3-3' — Phase 5 cover — và row mt is None —
       universal Phase 3 judge).
    2. Per-row classify (mirror Phase 3, NO silent skip):
       - judge bucket   → non-omitted, in-report page, registry shall non-empty.
       - deterministic  → omitted / external_ref / unsupported_location /
         page_list_*  → DisclosureVerdict không LLM call.
       - out-of-scope   → registry shall rỗng (vd sector add-on B/C);
         emit finding, no DisclosureVerdict.
    3. Dedupe judge bucket by (gri_standard, disc, sorted(page_list)).
    4. Sequential judge call cho unit unique (await per item).
    5. Fan-out unique-unit verdict ra mỗi (mt, disclosure) qua
       DisclosureVerdict.material_topic.
    6. Build expected_for_material từ
       materiality_map.material_topic_to_base_topics. Multi-base mt dedupe
       theo (std, disc).
    7. 5-b reconcile (5 bucket): reported / omitted_valid /
       external_reference / omitted_invalid_reason / missing.
       omitted_invalid_reason → PendingOmission(source_phase='phase6_5b').
    8. Non-material cross-check — mọi GCI row có material_topic resolve
       (qua base_topic_assignment hoặc non_material_match_index) sang
       non-material base topic → PendingOmission(
       source_phase='phase6_non_material_cross_check').
    9. Phase verdict — reuse phase3 aggregate_disclosure_overall +
       aggregate_phase_status (mix pass + no_evidence → partial).

`phase6_5a` literal trong PendingOmission.source_phase RESERVED tương lai
(Phase 7 có thể cần validate mọi topic-standard omission); Phase 6 KHÔNG
emit hiện tại — xem sub-plan 04 "5-bii scope" decision.
"""

import logging

from src.compliance import judges, registries, retrieval, rules
from src.compliance.phases.phase3_gri2 import (
    _classify_no_pack_reason,
    _resolve_disclosure_name,
    aggregate_disclosure_overall,
    aggregate_phase_status,
)
from src.compliance.state import (
    DisclosureVerdict,
    PendingOmission,
    PhaseResult,
)

logger = logging.getLogger(__name__)


# no_pack_reason → (overall, evidence_unrecoverable, contributes_to_has_fail)
# Reporter-violation buckets → fail; system-limitation buckets → no_evidence
# + unrecoverable=True; omitted / external_ref → bucket riêng (không fail).
_NO_PACK_OUTCOME = {
    "omitted": ("omitted", False, False),
    "external_document_reference": ("external_verification", False, False),
    "missing_disclosure": ("fail", False, True),
    "unsupported_location": ("fail", False, True),
    "page_list_unparseable": ("no_evidence", True, False),
    "page_list_no_matching_chunks": ("no_evidence", True, False),
}


# === Helpers ===

def _build_topic_disclosure_map(gci):
    """Group GCI row theo material_topic; loại mt None + disclosure_id 3-3.

    mt None = universal Phase 3 judge; 3-3 = Phase 5 structural cover.
    """
    topic_groups = registries.group_gci_by_material_topic(gci)
    out = {}
    for mt, rows in topic_groups.items():
        if mt is None:
            continue
        out[mt] = [r for r in rows if r.get("disclosure_id") != "3-3"]
    return out


def _classify_rows(topic_disclosure_map, all_chunks):
    """Per-row classify thành judge_inputs vs deterministic_verdicts.

    Return tuple `(judge_inputs, deterministic_verdicts, findings, has_fail)`.
    Branch precedence: blank disclosure_id → deterministic fail; no_pack_reason
    ≠ None → deterministic verdict theo `_NO_PACK_OUTCOME`; còn lại → judge bucket
    (nếu registry shall non-empty) hoặc out-of-scope finding.
    """
    registry_cache = {}
    findings = []
    judge_inputs = []
    deterministic_verdicts = []
    has_fail = False

    def get_reqs_for(row):
        # GCI gri_standard long ("GRI 305: Emissions 2016"); registry
        # short ("GRI 305"). Strip description sau colon đầu.
        raw = (row.get("standard_id") or row.get("gri_standard") or "").strip()
        if not raw:
            return []
        std_key = raw.split(":")[0].strip()
        if std_key not in registry_cache:
            registry_cache[std_key] = registries.build_disclosure_registry(std_key)
        return registry_cache[std_key].get(row.get("disclosure_id") or "", [])

    for mt, rows in topic_disclosure_map.items():
        for row in rows:
            std = (row.get("gri_standard") or "").strip()
            disc_id = row.get("disclosure_id") or ""
            mt_value = row.get("material_topic")

            # Blank disclosure_id short-circuit (data error). PHẢI trước
            # mọi classify path: row không id không lookup registry, không
            # judge, không reconcile. Step 7 sẽ leave expected là missing.
            if not str(disc_id).strip():
                has_fail = True
                findings.append(
                    f"phase6: GCI row under standard {std!r} (mt={mt_value!r}) "
                    f"has missing/blank disclosure_id \u2014 data error; "
                    f"counted as topic-level fail and the topic's expected "
                    f"disclosures will be reported as missing"
                )
                deterministic_verdicts.append(
                    DisclosureVerdict(
                        disclosure_id="",
                        standard_id=std,
                        requirement_verdicts=[],
                        overall="fail",
                        notes="GCI row has missing/blank disclosure_id \u2014 cannot judge",
                        material_topic=mt_value,
                    )
                )
                continue

            # Decide attempt build_page_pack — anchor GCI raw field,
            # precedence giống Phase 3 step 3.
            location_raw = row.get("location_raw") or ""
            attempted = (
                not rules.is_disclosure_omitted(row)
                and (row.get("location_type") or "") != "external_document_reference"
                and bool(str(location_raw).strip())
                and bool(row.get("page_list") or [])
            )
            pack = retrieval.build_page_pack(row, all_chunks) if attempted else []

            no_pack_reason = _classify_no_pack_reason(row, pack)

            # Order matter: deterministic verdict KHÔNG cần shall registry —
            # short-circuit trước LLM. Chỉ judge bucket cần registry;
            # out-of-scope finding chỉ apply khi no_pack_reason is None.
            if no_pack_reason is not None:
                overall, unrecoverable, fail_inc = _NO_PACK_OUTCOME.get(
                    no_pack_reason, ("fail", False, True)  # defensive
                )
                if fail_inc:
                    has_fail = True
                notes = f"no page pack: {no_pack_reason}"
                if unrecoverable:
                    notes += "; system limitation \u2014 human review required"
                deterministic_verdicts.append(
                    DisclosureVerdict(
                        disclosure_id=disc_id,
                        standard_id=std,
                        requirement_verdicts=[],
                        overall=overall,
                        notes=notes,
                        evidence_unrecoverable=unrecoverable,
                        material_topic=mt_value,
                    )
                )
                continue

            # no_pack_reason is None → judge bucket. Cần shall registry
            # non-empty; nếu không, row out-of-scope (vd sector add-on B/C,
            # parser-emitted unknown standard).
            reqs = get_reqs_for(row)
            if not reqs:
                findings.append(
                    f"phase6: disclosure {std}/{disc_id} (mt={mt_value!r}) "
                    f"has no shall registry \u2014 skipped (out-of-scope)"
                )
                continue

            judge_inputs.append((row, pack, reqs))

    return judge_inputs, deterministic_verdicts, findings, has_fail


def _dedupe_judge_inputs(judge_inputs):
    """Dedupe by (gri_standard, disc, sorted(page_list)).

    Same disclosure under multi-mt với cùng page list → judge ONCE,
    fan-out ra mỗi (mt, disc). Return `(unique_units, fan_out_map)`.
    """
    unique_units = {}
    fan_out_map = {}
    for row, pack, reqs in judge_inputs:
        key = (
            (row.get("gri_standard") or "").strip(),
            row.get("disclosure_id") or "",
            tuple(sorted(int(p) for p in (row.get("page_list") or []))),
        )
        unique_units.setdefault(key, {"row": row, "pack": pack, "reqs": reqs})
        fan_out_map.setdefault(key, []).append(row.get("material_topic"))
    return unique_units, fan_out_map


async def _judge_unique_units(unique_units, report_id):
    """Sequential await run_disclosure_batch cho mỗi unique unit."""
    async def judge_one(key, unit):
        row = unit["row"]
        d_id = row.get("disclosure_id") or ""
        d_name = _resolve_disclosure_name(unit["reqs"], d_id)
        try:
            batch = await judges.evidence_judge.run_disclosure_batch(
                disclosure_id=d_id,
                disclosure_name=d_name,
                page_pack=unit["pack"],
                registry=unit["reqs"],
                anchor=None,
                report_id=report_id,
            )
            return key, batch
        except Exception as e:
            logger.exception("phase6 judge_one failed for %s: %s", key, e)
            return key, {"_error": repr(e)}

    judged = []
    for key, unit in unique_units.items():
        judged.append(await judge_one(key, unit))
    return dict(judged)


def _assemble_fan_out_verdicts(judged_by_key, fan_out_map):
    """Fan-out unique-unit batch ra mỗi (mt, disc).

    Reuse phase3 aggregate_disclosure_overall: mix pass + no_evidence → partial.
    Return `(verdicts, has_fail, has_partial_or_no_evidence_non_omitted)`.
    """
    verdicts = []
    has_fail = False
    has_partial = False

    for key, batch in judged_by_key.items():
        std_id, disc_id, _pages_tuple = key

        # Exception path — fail mọi fan-out target deterministic.
        if isinstance(batch, dict) and "_error" in batch:
            has_fail = True
            for mt in fan_out_map[key]:
                verdicts.append(
                    DisclosureVerdict(
                        disclosure_id=disc_id,
                        standard_id=std_id,
                        requirement_verdicts=[],
                        overall="fail",
                        notes=f"exception: {batch['_error']}",
                        material_topic=mt,
                    )
                )
            continue

        statuses = [v.status for v in batch.requirement_verdicts]
        overall = aggregate_disclosure_overall(statuses)
        if overall == "fail":
            has_fail = True
        elif overall in {"partial", "no_evidence"}:
            # Topic-standard row đến judge bucket by construction là
            # non-omitted + non-unrecoverable → partial/no_evidence count
            # vào phase-status downgrade.
            has_partial = True

        for mt in fan_out_map[key]:
            verdicts.append(
                DisclosureVerdict(
                    disclosure_id=disc_id,
                    standard_id=std_id,
                    requirement_verdicts=list(batch.requirement_verdicts),
                    overall=overall,
                    notes="",
                    evidence_unrecoverable=False,
                    material_topic=mt,
                )
            )

    return verdicts, has_fail, has_partial


def _build_expected_for_material(materiality_map):
    """Build expected_for_material driven by material_topic_to_base_topics.

    Multi-base mt dedupe across base. ("GRI 3", "3-3") intentional skip:
    3-3 là Management of MT, deferred Phase 5 (external_verification stub
    v2). Step 1 đã loại 3-3 row khỏi topic_disclosure_map → không produce
    DisclosureVerdict; nếu vẫn để vào expected, Step 7 reconcile sẽ tìm
    GCI row un-omitted với omission_reason=None → label
    omitted_invalid_reason (BUG #1 — 20 false-positive PendingOmission
    Bangchak2025). Filter ở đây giữ Phase 6 responsibility narrow
    ("topic-standard disclosures") + catalogue untouched.

    Material topic Phase 4 matcher trả base-topic list rỗng → emit advisory
    (skipped_5b_topics): không violation, có thể legitimate ngoài GRI 11.
    Return `(expected_for_material, skipped_5b_topics)`.
    """
    catalogue = registries.build_gri11_catalogue()
    expected_for_material = {}
    skipped_5b_topics = []

    for mt, base_ids in materiality_map.material_topic_to_base_topics.items():
        if not base_ids:
            skipped_5b_topics.append({
                "material_topic": mt,
                "reason": "no_gri11_base_topic_mapping",
            })
            continue
        seen = set()
        bucket = []
        for base_id in base_ids:
            base = catalogue.get(base_id)
            if not base:
                continue
            for std_id, disc_id in base.get("promoted_disclosures", []):
                pair = (str(std_id), str(disc_id))
                if pair == ("GRI 3", "3-3"):
                    continue  # Phase 5 own 3-3
                if pair in seen:
                    continue
                seen.add(pair)
                bucket.append(pair)
        if bucket:
            expected_for_material[mt] = bucket

    return expected_for_material, skipped_5b_topics


def _make_5b_entry(mt, short_std, disc_id, status, *, reason=None, explanation=None):
    """5-b reconcile dict — common shape (preview truncate explanation 200 char)."""
    return {
        "material_topic": mt,
        "expected_standard_id": short_std,
        "expected_disclosure_id": disc_id,
        "status": status,
        "omission_reason": reason,
        "omission_explanation_preview": (explanation[:200] if explanation else None),
    }


def _reconcile_5b(expected_for_material, disclosure_verdicts, gci):
    """5-bucket reconcile qua DisclosureVerdict.material_topic (no string parse).

    NORMALIZATION: GCI carry gri_standard long ("GRI 305: Emissions 2016")
    còn catalogue promoted_disclosures dùng short ("GRI 305"). Không
    normalize → `triple in reported_set` silent fail cho mọi topic-standard
    → ALL classify missing (regression: 109 false-positive 5-b-i Bangchak2025).

    BUG #2 FIX: deterministic verdict overall ∈ {omitted, external_verification}
    KHÔNG count "reported" — phải flow vào GCI-lookup branch để reconcile
    classify đúng (omitted_valid / omitted_invalid_reason / external_reference).
    Không filter, vd Bangchak2025 302-2 (omitted với "Information unavailable")
    bị label reported, 5-b-ii violation bị giấu.

    Return `(reconcile_5b, pending_5b)`.
    """
    reported_set = {
        (registries.short_std_id(dv.standard_id), dv.disclosure_id, dv.material_topic or "")
        for dv in disclosure_verdicts
        if dv.material_topic is not None
        and dv.overall not in {"omitted", "external_verification"}
    }

    reconcile = []
    pending = []

    for mt, expected_list in expected_for_material.items():
        for std_id, disc_id in expected_list:
            # expected_for_material source build_gri11_catalogue đã emit
            # short-form std_id; normalize defensive cho symmetric contract.
            short_std = registries.short_std_id(std_id)
            triple = (short_std, disc_id, mt)

            if triple in reported_set:
                reconcile.append(_make_5b_entry(mt, short_std, disc_id, "reported"))
                continue

            # Không trong reported_set — lookup GCI row by (std, disc, mt).
            # Compare short form để GCI row gri_standard "GRI 305: Emissions
            # 2016" vẫn match expected std_id "GRI 305".
            gci_row = next(
                (
                    r
                    for r in gci
                    if registries.short_std_id(r) == short_std
                    and r.get("disclosure_id") == disc_id
                    and r.get("material_topic") == mt
                ),
                None,
            )

            if gci_row is None:
                reconcile.append(_make_5b_entry(mt, short_std, disc_id, "missing"))
                continue

            # External reference bucket — Phase 3 parity. NOT missing,
            # NOT invalid omission; treat external_verification (no phase impact).
            if (
                (gci_row.get("location_type") or "") == "external_document_reference"
                and not rules.is_disclosure_omitted(gci_row)
            ):
                reconcile.append(
                    _make_5b_entry(mt, short_std, disc_id, "external_reference")
                )
                continue

            reason = (gci_row.get("omission_reason") or "").strip() or None
            explanation = (gci_row.get("omission_explanation") or "").strip()
            if rules.is_not_applicable(reason) and explanation:
                reconcile.append(
                    _make_5b_entry(
                        mt, short_std, disc_id, "omitted_valid",
                        reason=reason, explanation=explanation,
                    )
                )
            else:
                reconcile.append(
                    _make_5b_entry(
                        mt, short_std, disc_id, "omitted_invalid_reason",
                        reason=reason, explanation=explanation or None,
                    )
                )
                pending.append(
                    PendingOmission(
                        source_phase="phase6_5b",
                        standard_id=short_std,
                        disclosure_id=disc_id,
                        material_topic=mt,
                        omission_reason=reason,
                        omission_explanation=explanation or None,
                        row=gci_row,
                    )
                )

    return reconcile, pending


def _collect_non_material_mt_names(materiality_map):
    """Tập material_topic name resolve sang non-material base topic."""
    base_assignment = materiality_map.base_topic_assignment
    non_material_base_ids = {
        b for b, owners in base_assignment.items() if "non_material" in owners
    }
    names = set()
    for mt, base_ids in materiality_map.material_topic_to_base_topics.items():
        if any(b in non_material_base_ids for b in base_ids):
            names.add(mt)
    # Belt-and-braces: catch trường hợp hiếm GCI material_topic literal
    # bằng nm.topics text (vd "Topic 11.7 ..."). Most GCI well-formed không vậy.
    names |= set(materiality_map.non_material_match_index.keys())
    return names


def _collect_non_material_pending(gci, non_material_mt_names):
    """Emit PendingOmission(phase6_non_material_cross_check) cho mỗi GCI
    row có material_topic là non-material."""
    pending = []
    for row in gci:
        mt_raw = row.get("material_topic")
        if mt_raw and mt_raw in non_material_mt_names:
            pending.append(
                PendingOmission(
                    source_phase="phase6_non_material_cross_check",
                    # Normalize short form cho downstream join (vd Phase 7 _short_std)
                    standard_id=registries.short_std_id(row),
                    disclosure_id=row.get("disclosure_id") or "",
                    material_topic=mt_raw,
                    omission_reason=row.get("omission_reason"),
                    omission_explanation=row.get("omission_explanation"),
                    row=row,
                )
            )
    return pending


def _phase_status_and_findings(has_fail, has_partial, reconcile_5b, skipped_5b_topics):
    """Tổng hợp phase status + findings của Bước 9.

    Return `(status, step9_findings)`. step9_findings emit theo thứ tự
    5-a → 5-b-i → 5-b-ii → 5-b advisory.
    """
    findings = []
    a_status = aggregate_phase_status(
        coverage_check_passed=True,    # Phase 6 không có coverage check
        hard_fail=False,               # Phase 6 không có hard-fail rule
        has_fail=has_fail,
        has_partial_or_no_evidence_non_omitted=has_partial,
    )
    if a_status == "fail":
        findings.append(
            "5-a: at least one topic-standard disclosure has a failing "
            "requirement (or an unrecoverable system error / unsupported "
            "location)"
        )
    elif a_status == "partial":
        findings.append(
            "5-a: at least one topic-standard disclosure rolled up to "
            "partial / no_evidence"
        )

    # 5-b-i: mọi GRI 11 promoted disclosure cho mt phải reported, validly
    # omitted, hoặc externally referenced. Chỉ missing fail.
    b_i_fail = [r for r in reconcile_5b if r["status"] == "missing"]
    if b_i_fail:
        findings.append(
            f"5-b-i: {len(b_i_fail)} GRI 11 promoted disclosures for "
            f"material topics are neither reported nor omitted nor "
            f"externally referenced"
        )

    # 5-b-ii: mọi 5-b omission row dùng 'Not applicable' + explanation
    # non-empty. omitted_invalid_reason fail bucket.
    b_ii_fail = [r for r in reconcile_5b if r["status"] == "omitted_invalid_reason"]
    if b_ii_fail:
        findings.append(
            f"5-b-ii: {len(b_ii_fail)} GRI 11 expected disclosures use a "
            f"reason other than 'Not applicable' (or have an empty "
            f"explanation)"
        )

    # 5-b (advisory): topic skip 5-b reconcile vì Phase 4 không trả GRI 11
    # base-topic mapping. KHÔNG fail — có thể legitimate ngoài GRI 11.
    if skipped_5b_topics:
        findings.append(
            f"5-b (advisory): GRI 11 expected-disclosure reconcile skipped for "
            f"{len(skipped_5b_topics)} material topic(s) with no GRI 11 "
            f"base-topic mapping — manual review required: "
            f"{[t['material_topic'] for t in skipped_5b_topics]}"
        )

    if a_status == "fail" or b_i_fail or b_ii_fail:
        status = "fail"
    elif a_status == "partial":
        status = "partial"
    else:
        status = "pass"

    return status, findings


# === Phase node ===

async def phase6_topic_standards_node(state):
    """Phase 6 entry-point — xem module docstring."""
    inputs = state.get("inputs") or {}
    gci = list(inputs.get("gci") or [])
    all_chunks = list(inputs.get("all_chunks_for_report") or [])
    report_id = state["report_id"]
    materiality_map = state.get("materiality_map")

    # Defensive: Phase 4 phải populate materiality_map. Không có →
    # 5-b reconcile + non-material cross-check không có input.
    if materiality_map is None:
        pr = PhaseResult(
            phase="phase6",
            status="fail",
            findings=["phase6: materiality_map not populated by Phase 4"],
            artifacts={},
        )
        return {"phase_results": {**state.get("phase_results", {}), "phase6": pr}}

    # Bước 1: topic ↔ disclosure map
    topic_disclosure_map = _build_topic_disclosure_map(gci)

    # Bước 2: per-row classify + registry build (no silent skip)
    judge_inputs, deterministic_verdicts, classify_findings, has_fail = (
        _classify_rows(topic_disclosure_map, all_chunks)
    )

    # Bước 3: dedupe judge bucket
    unique_units, fan_out_map = _dedupe_judge_inputs(judge_inputs)

    # Bước 4: 5-a evidence loop — async fan-out qua unit unique
    judged_by_key = await _judge_unique_units(unique_units, report_id)

    # Bước 5: fan-out unique-unit verdict ra mỗi (mt, disc)
    judge_verdicts, judge_has_fail, has_partial = _assemble_fan_out_verdicts(
        judged_by_key, fan_out_map
    )
    has_fail = has_fail or judge_has_fail
    disclosure_verdicts = [*deterministic_verdicts, *judge_verdicts]

    # Bước 6: build expected_for_material
    expected_for_material, skipped_5b_topics = _build_expected_for_material(
        materiality_map
    )

    # Bước 7: 5-b reconcile (5-bucket status)
    reconcile_5b, pending_5b = _reconcile_5b(
        expected_for_material, disclosure_verdicts, gci
    )

    # Bước 8: non-material cross-check
    non_material_mt_names = _collect_non_material_mt_names(materiality_map)
    pending_non_material = _collect_non_material_pending(gci, non_material_mt_names)

    # Bước 9: phase verdict — partial-aware via Phase 3 roll-up
    status, step9_findings = _phase_status_and_findings(
        has_fail, has_partial, reconcile_5b, skipped_5b_topics
    )
    findings = [*classify_findings, *step9_findings]
    pending = [*pending_5b, *pending_non_material]

    # Post-hoc cache-hit counter
    n_cache_hits = sum(
        1
        for dv in disclosure_verdicts
        for rv in dv.requirement_verdicts
        if rv.source == "cache"
    )

    pr = PhaseResult(
        phase="phase6",
        status=status,
        findings=findings,
        artifacts={
            "topic_disclosure_map": {
                mt: [r.get("disclosure_id") for r in rows]
                for mt, rows in topic_disclosure_map.items()
            },
            "dedupe_units": {
                str(k): {"fan_out": fan_out_map[k], "row": v["row"]}
                for k, v in unique_units.items()
            },
            "disclosure_verdicts": [dv.model_dump() for dv in disclosure_verdicts],
            "reconcile_5b": reconcile_5b,
            "expected_for_material": {
                mt: [list(pair) for pair in pairs]
                for mt, pairs in expected_for_material.items()
            },
            "reconcile_5b_skipped_topics": skipped_5b_topics,
            "non_material_mt_names": sorted(non_material_mt_names),
            "n_cache_hits": n_cache_hits,
        },
    )
    return {
        "phase_results": {**state.get("phase_results", {}), "phase6": pr},
        "pending_omissions": [*state.get("pending_omissions", []), *pending],
    }


__all__ = ["phase6_topic_standards_node"]
