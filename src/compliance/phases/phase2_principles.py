from src.compliance import config
from src.compliance.state import PhaseResult

_NOTE = (
    "Reporting principles judging is deferred to external assurance in v2. "
    "Only the canonical 8-principle checklist is emitted here so an auditor "
    "can review them manually."
)

def phase2_principles_node(state):
    """Emit danh sách 8 GRI reporting principles — deferred to external assurance."""
    pr = PhaseResult(
        phase="phase2",
        status="external_verification",
        findings=[],
        artifacts={
            "principles": list(config.REPORTING_PRINCIPLES),
            "note": _NOTE,
        },
    )
    return {"phase_results": {**state.get("phase_results", {}), "phase2": pr}}

__all__ = ["phase2_principles_node"]
