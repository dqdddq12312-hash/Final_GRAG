from src.compliance import config, io
from src.compliance.state import PhaseResult, ScopeDisclaimer

def _build_scope_disclaimer(state):
    """Build ScopeDisclaimer mô tả phạm vi Module 3 đã kiểm tra và chưa kiểm tra.

    `checked` — những gì Module 3 verify được. `not_checked` — gap đã được
    document rõ. Nếu sector standard ngoài supported set, prepend note vào
    not_checked để operator biết cần manual review.
    """
    sector_list = state.get("sector_standards") or []
    sector_standard = sector_list[0] if sector_list else None

    checked = [
        "Universal standards (GRI 1/2/3) shall requirements",
        "Topic standards (GRI 1xx/2xx/3xx/4xx) shall requirements",
        "GRI 11 promoted disclosure mapping (Phase 4 + Phase 6 5-b)",
        "GRI 3 disclosures per material topic — Req 4-a/4-b/4-c (Phase 5)",
    ]

    not_checked = [
        "GRI 11 Additional sector recommendations (B) — shall bullets layered onto topic-standard disclosures",
        "GRI 11 Additional sector disclosures (C, e.g., 11.1.16 CapEx investment allocation)",
        "Per-requirement omission tracking (Module 3 is disclosure-level only)",
    ]

    if sector_standard and sector_standard not in config.SUPPORTED_SECTOR_STANDARDS:
        not_checked.insert(
            0,
            f"Sector standard '{sector_standard}' (only GRI 11 is supported by Module 3)",
        )

    if sector_standard == "GRI 11: Oil and Gas Sector 2021":
        recommendation = (
            "Manual review of GRI 11 PDF sections 11.1–11.22 required for sector-specific add-ons."
        )
    else:
        recommendation = (
            "No supported sector standard — sector-specific verification skipped."
        )

    return ScopeDisclaimer(
        sector_standard=sector_standard,
        checked=checked,
        not_checked=not_checked,
        recommendation=recommendation,
    )

def final_aggregator_node(state):
    """Tổng hợp overall pass/fail, build ScopeDisclaimer, ghi output file.

    Khi `overall_pass=False`, `findings` enumerate phase fail kèm số finding
    để dashboard thấy failure surface mà không cần re-walk `phase_results`.
    """
    pathway = state.get("claim_pathway", "unclear")
    pr_dict = state.get("phase_results", {})

    if pathway == "in_accordance":
        evaluated = ("phase3", "phase4", "phase5", "phase6", "phase7", "phase8")
    elif pathway == "with_reference":
        evaluated = ("phase1", "phase8_light")
    else:
        evaluated = ()

    if evaluated:
        overall_pass = all(
            pr_dict.get(p) is not None and pr_dict[p].status == "pass"
            for p in evaluated
        )
    else:
        overall_pass = False

    scope_disclaimer = _build_scope_disclaimer(state)

    findings = []
    blockers = []
    if not overall_pass:
        if not evaluated:
            findings.append(
                f"overall_pass=False: claim_pathway={pathway!r} cannot be "
                f"evaluated by Module 3"
            )
        else:
            failing = [p for p in evaluated if pr_dict.get(p) is None or pr_dict[p].status != "pass"]
            for p in failing:
                pr_p = pr_dict.get(p)
                if pr_p is None:
                    findings.append(
                        f"{p}: missing PhaseResult (pipeline did not run this phase)"
                    )
                    blockers.append(
                        {"phase": p, "status": "missing", "n_findings": 0, "preview": []}
                    )
                    continue
                pf_findings = list(pr_p.findings)
                findings.append(
                    f"{p} ({pr_p.status}): {len(pf_findings)} finding"
                    f"{'s' if len(pf_findings) != 1 else ''}"
                )
                blockers.append(
                    {
                        "phase": p,
                        "status": pr_p.status,
                        "n_findings": len(pf_findings),
                        "preview": pf_findings[:3],
                    }
                )

    pr = PhaseResult(
        phase="final",
        status="pass" if overall_pass else "fail",
        findings=findings,
        artifacts={
            "overall_pass": overall_pass,
            "scope_disclaimer": scope_disclaimer.model_dump(),
            "blockers": blockers,
            "evaluated_phases": list(evaluated),
            "output_paths": {},
        },
    )

    updated_phase_results = {**pr_dict, "final": pr}
    state_with_final = {**state, "phase_results": updated_phase_results}

    # Two-pass write: pass 1 ghi file, io.write_outputs trả về dict paths
    # (compliance_report.json + compliance_summary.csv). Pass 2 ghi lại SAU KHI
    # output_paths đã được populate vào artifacts — để compliance_report.json
    # chứa đường dẫn đầy đủ cho notebook "Outputs: [JSON] | [CSV]" link.
    # KHÔNG được bỏ bất kỳ pass nào (pass 2 mới embed output_paths).
    paths = io.write_outputs(state_with_final)
    pr.artifacts["output_paths"] = paths
    io.write_outputs(state_with_final)

    return {"phase_results": updated_phase_results}

__all__ = ["final_aggregator_node"]
