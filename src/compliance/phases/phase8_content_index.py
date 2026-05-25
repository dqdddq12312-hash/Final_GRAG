from src.compliance import rules
from src.compliance.state import PhaseResult

def _build_phase_result(state, phase_key, appendix_label, checklist):
    """Tạo PhaseResult từ kết quả checklist Appendix 1 hoặc 2."""
    findings = [
        f"item {item['item_id']}: {item['item_name']} — missing or invalid"
        for item in checklist
        if not item["present"]
    ]
    pr = PhaseResult(
        phase=phase_key,
        status="pass" if not findings else "fail",
        findings=findings,
        artifacts={
            "checklist": checklist,
            "appendix_used": appendix_label,
            "n_items": len(checklist),
            "n_pass": sum(1 for item in checklist if item["present"]),
        },
    )
    return {"phase_results": {**state.get("phase_results", {}), phase_key: pr}}

def phase8_content_index_node(state):
    """Full variant (in_accordance) — Appendix 1, item i–xii."""
    return _build_phase_result(
        state, "phase8", "Appendix 1", rules.appendix1_checklist(state)
    )

def phase8_content_index_light_node(state):
    """Light variant (with_reference) — Appendix 2, item i–vi."""
    return _build_phase_result(
        state, "phase8_light", "Appendix 2", rules.appendix2_checklist(state)
    )

__all__ = ["phase8_content_index_node", "phase8_content_index_light_node"]
