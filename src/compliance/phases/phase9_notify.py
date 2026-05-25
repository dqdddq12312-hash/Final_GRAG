from src.compliance.state import PhaseResult

_CHECKLIST = (
    "Confirm Phase 1 claim_pathway and Phase 8 GRI Content Index match the "
    "claim made on the cover of the report.",
    "Email reportregistration@globalreporting.org with the report PDF, "
    "reporting period, and the chosen claim ('in_accordance' or 'with_reference').",
    "Attach the GRI Content Index page list (from Phase 8 artifacts) so GRI "
    "can locate every disclosure in the PDF.",
    "Retain the GRI confirmation email in the report's audit folder.",
    "If the report is restated, repeat the notification within 30 days of "
    "the restated edition.",
)

_NOTE = (
    "Phase 9 emits an operator-facing checklist only. Notifying GRI is a "
    "manual step performed by the reporting organization."
)

def phase9_notify_node(state):
    """Stub PhaseResult — emit notify-GRI checklist cho operator."""
    pr = PhaseResult(
        phase="phase9",
        status="external_verification",
        findings=[],
        artifacts={
            "checklist": list(_CHECKLIST),
            "note": _NOTE,
        },
    )
    return {"phase_results": {**state.get("phase_results", {}), "phase9": pr}}

__all__ = ["phase9_notify_node"]
