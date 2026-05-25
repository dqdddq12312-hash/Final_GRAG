from src.compliance import config, registries, rules
from src.compliance.registries import short_std_id as _short_std
from src.compliance.state import PhaseResult

def phase7_omissions_node(state):
    """Reconcile omission list — pure rule, không cần LLM hay Milvus.

    Bước 1: Build expected_disclosures từ Phase 4 5-b (GRI 11 promoted) +
            Phase 6 topic-standard reported disclosures.
    Bước 2: Collect reported_omissions từ GCI.
    Bước 3: Reconcile 6-a-i (expected nhưng không reported/omitted) +
            6-a-ii (omission row có reason/explanation không hợp lệ).
    Bước 4 (Step 5): Hard-fail 6-b(2) theo config.OMIT_BLOCKED.
    Bước 5: Build reconcile_table cho notebook render.
    """
    inputs = state["inputs"]
    gci = inputs["gci"]
    materiality_map = state.get("materiality_map")

    # Cache catalogue 1 lần — reuse ở Bước 1a + source_phase labelling.
    catalogue = registries.build_gri11_catalogue()

    # === Bước 1: Build expected_disclosures ===
    # expected = (Phase 4 5-b GRI 11 promoted) ∪ (Phase 6 topic-standard)
    # reported = GCI rows NOT omitted (short-form key)
    # missing_list = expected − reported
    expected_disclosures = set()  # (short_standard_id, disclosure_id)
    phase4_expected = set()       # subset thuộc Phase 4 5-b

    # 1a. Phase 4 5-b expected — GRI 11 promoted cho material topics
    if materiality_map is not None:
        for base_id, owners in materiality_map.base_topic_assignment.items():
            if all(o in {"non_material", "missing"} for o in owners):
                continue
            for std_id, disc_id in catalogue.get(base_id, {}).get("promoted_disclosures", []):
                expected_disclosures.add((std_id, disc_id))
                phase4_expected.add((std_id, disc_id))

    # 1b. Phase 6 expected — topic-standard reported dưới mọi material topic
    # (loại 3-3 vì Phase 5 handle). _short_std() collapse "GRI 302: Energy
    # 2016" → "GRI 302" để khớp catalogue promoted_disclosures.
    for row in gci:
        if row.get("material_topic") and row.get("disclosure_id") != "3-3":
            expected_disclosures.add((_short_std(row), row["disclosure_id"]))

    reported_disclosures = {
        (_short_std(r), r["disclosure_id"])
        for r in gci
        if not rules.is_disclosure_omitted(r)
    }

    missing_pairs = expected_disclosures - reported_disclosures
    missing_list = []
    # `None` slipping vào std_id khi gold materiality_map có base không
    # parent standard match — convert "" để sortable (Python 3 cấm None < str).
    for std_id, disc_id in sorted(missing_pairs, key=lambda p: (p[0] or "", p[1] or "")):
        source = "phase4_5b" if (std_id, disc_id) in phase4_expected else "phase6_expected"
        missing_list.append({
            "standard_id": std_id,
            "disclosure_id": disc_id,
            "source_phase": source,
        })

    # === Bước 2: reported_omissions — user-facing 6-a list ===
    # Mọi GCI row có rules.is_disclosure_omitted() True. Disclosure-level
    # only — omission_requirement_omitted KHÔNG parse. standard_id normalize.
    reported_omissions = []
    for row in gci:
        if rules.is_disclosure_omitted(row):
            reported_omissions.append({
                "standard_id": _short_std(row),
                "disclosure_id": row["disclosure_id"],
                "omission_reason": row.get("omission_reason"),
                "omission_explanation": row.get("omission_explanation") or "",
                "row": row,
            })

    # === Bước 3: Reconcile 6-a-i và 6-a-ii ===
    # GRI 11 5-b strict mirror Phase 6: GRI 11 promoted CHỈ dùng "Not
    # applicable" → reconcile_table.verdict align với phase6.reconcile_5b.

    def _is_5b_expected(std_id, disc_id):
        """Kiểm tra (std_id, disc_id) thuộc Phase 4 GRI 11 5-b expected set."""
        return (std_id, disc_id) in phase4_expected

    def _is_reason_valid(std_id, disc_id, reason):
        """Validate omission reason — GRI 11 5-b chỉ cho phép 'Not applicable'."""
        if _is_5b_expected(std_id, disc_id):
            return rules.is_not_applicable(reason)
        return rules.validate_omission_reason(reason)

    def _why_invalid(r):
        """Trả chuỗi mô tả lý do omission không hợp lệ, hoặc '' nếu hợp lệ."""
        std_id, disc_id = r["standard_id"], r["disclosure_id"]
        is_5b = _is_5b_expected(std_id, disc_id)
        if not _is_reason_valid(std_id, disc_id, r["omission_reason"]):
            if is_5b:
                return (
                    f"GRI 11 5-b: reason {r['omission_reason']!r} is not "
                    f"'Not applicable' (GRI 11 promoted disclosures may only "
                    f"use 'Not applicable')"
                )
            return f"reason {r['omission_reason']!r} not in VALID_OMISSION_REASONS"
        if not r["omission_explanation"].strip():
            return "omission_explanation is empty"
        return ""

    reported_omitted_set = {
        (r["standard_id"], r["disclosure_id"]) for r in reported_omissions
    }

    # 6-a-i: expected mà không reported, không omission row
    a_i = [
        m for m in missing_list
        if (m["standard_id"], m["disclosure_id"]) not in reported_omitted_set
    ]

    # 6-a-ii: reported_omissions có invalid reason (6-b(1)) HOẶC explanation rỗng (6-b(3))
    a_ii = [
        {**r, "why_invalid": _why_invalid(r)}
        for r in reported_omissions
        if not _is_reason_valid(
            r["standard_id"], r["disclosure_id"], r["omission_reason"]
        )
        or not r["omission_explanation"].strip()
    ]

    invalid_reasons = a_ii  # alias dùng trong artifacts

    # === Bước 4: Hard-fails 6-b(2) ===
    # Scope = config.OMIT_BLOCKED.
    # Phase 5 cũng flag {3-1, 3-2} omission qua coverage layer — cả 2 path
    # gây overall_pass=False (defense in depth).
    hard_fail_blocklist = config.OMIT_BLOCKED
    hard_fails = [
        {"disclosure_id": r["disclosure_id"], "omission_reason": r["omission_reason"]}
        for r in reported_omissions
        if r["disclosure_id"] in hard_fail_blocklist
    ]

    # Phase verdict
    findings = []
    if hard_fails:
        findings.append(
            f"hard-fail: {len(hard_fails)} disclosures in GRI2_HARD_FAIL_OMIT_BLOCKED "
            f"are omitted: {[h['disclosure_id'] for h in hard_fails]}"
        )
    if a_i:
        findings.append(
            f"6-a-i: {len(a_i)} expected disclosures not reported and not omitted"
        )
    if a_ii:
        n_5b_invalid = sum(
            1
            for r in a_ii
            if _is_5b_expected(r["standard_id"], r["disclosure_id"])
        )
        suffix = (
            f" (incl. {n_5b_invalid} GRI 11 5-b strict 'Not applicable' violation"
            f"{'s' if n_5b_invalid != 1 else ''})"
            if n_5b_invalid
            else ""
        )
        findings.append(
            f"6-a-ii: {len(a_ii)} omission rows have invalid reason or empty explanation"
            + suffix
        )

    status = "pass" if (not hard_fails and not a_i and not a_ii) else "fail"

    # === Bước 5: Reconcile table ===
    # 1 row / missing_list entry. Notebook render:
    # green=valid, yellow=invalid, gray=missing_no_omission, red=hard_fail.
    reconcile_table = []
    for m in missing_list:
        key = (m["standard_id"], m["disclosure_id"])
        match = next(
            (r for r in reported_omissions if (r["standard_id"], r["disclosure_id"]) == key),
            None,
        )
        # Phase 4 5-b dùng GRI 11 strict via _is_reason_valid;
        # `requires_not_applicable` flag surface lý do "Information
        # unavailable" valid GRI 1 nhưng reject ở đây.
        requires_not_applicable = m["source_phase"] == "phase4_5b"
        if match is not None:
            reason_valid = _is_reason_valid(
                m["standard_id"], m["disclosure_id"], match["omission_reason"]
            )
            expl_ok = bool(match["omission_explanation"].strip())
            verdict = "valid" if (reason_valid and expl_ok) else "invalid"
        else:
            reason_valid = False
            expl_ok = False
            verdict = "missing_no_omission"

        reconcile_table.append({
            "standard_id": m["standard_id"],
            "disclosure_id": m["disclosure_id"],
            "source_phase": m["source_phase"],
            "requires_not_applicable": requires_not_applicable,
            "in_missing_list": True,
            "has_omission_row": match is not None,
            "reason": match["omission_reason"] if match else None,
            "reason_valid": reason_valid,
            "explanation_non_empty": expl_ok,
            "hard_fail_blocked": m["disclosure_id"] in hard_fail_blocklist,
            "verdict": verdict,
        })

    pr = PhaseResult(
        phase="phase7",
        status=status,
        findings=findings,
        artifacts={
            "missing_list": missing_list,
            # Strip "row" (raw GCI dict) khỏi persisted artifacts để JSON-serializable
            "reported_omissions": [
                {k: v for k, v in r.items() if k != "row"} for r in reported_omissions
            ],
            "invalid_reasons": [
                {k: v for k, v in r.items() if k != "row"} for r in invalid_reasons
            ],
            "hard_fails": hard_fails,
            "reconcile_table": reconcile_table,
        },
    )
    return {"phase_results": {**state.get("phase_results", {}), "phase7": pr}}

__all__ = ["phase7_omissions_node"]
