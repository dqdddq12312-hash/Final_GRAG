from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.compliance.phases.final_aggregator import final_aggregator_node
from src.compliance.phases.phase1_claim import phase1_claim_node
from src.compliance.phases.phase2_principles import phase2_principles_node
from src.compliance.phases.phase3_gri2 import phase3_gri2_node
from src.compliance.phases.phase4_material_topics import phase4_material_topics_node
from src.compliance.phases.phase5_gri3 import phase5_gri3_node
from src.compliance.phases.phase6_topic_standards import phase6_topic_standards_node
from src.compliance.phases.phase7_omissions import phase7_omissions_node
from src.compliance.phases.phase8_content_index import (
    phase8_content_index_light_node,
    phase8_content_index_node,
)
from src.compliance.phases.phase9_notify import phase9_notify_node
from src.compliance.state import ComplianceState


def _route_after_phase1(state):
    """Route theo Phase 1 normalized verdict."""
    pathway = state.get("claim_pathway", "unclear")
    if pathway == "in_accordance":
        return "phase2"
    if pathway == "with_reference":
        return "phase8_light"
    return "final"

def build_compliance_graph(checkpoint=False):
    """Build và compile LangGraph state machine cho toàn bộ compliance pipeline."""
    g = StateGraph(ComplianceState)

    g.add_node("phase1", phase1_claim_node)
    g.add_node("phase2", phase2_principles_node)
    g.add_node("phase3", phase3_gri2_node)
    g.add_node("phase4", phase4_material_topics_node)
    g.add_node("phase5", phase5_gri3_node)
    g.add_node("phase6", phase6_topic_standards_node)
    g.add_node("phase7", phase7_omissions_node)
    g.add_node("phase8", phase8_content_index_node)
    g.add_node("phase8_light", phase8_content_index_light_node)
    g.add_node("phase9", phase9_notify_node)
    g.add_node("phase9_light", phase9_notify_node)
    g.add_node("final", final_aggregator_node)

    g.set_entry_point("phase1")
    g.add_conditional_edges(
        "phase1",
        _route_after_phase1,
        {
            "phase2": "phase2",
            "phase8_light": "phase8_light",
            "final": "final",
        },
    )

    g.add_edge("phase2", "phase3")
    g.add_edge("phase3", "phase4")
    g.add_edge("phase4", "phase5")
    g.add_edge("phase5", "phase6")
    g.add_edge("phase6", "phase7")
    g.add_edge("phase7", "phase8")
    g.add_edge("phase8", "phase9")
    g.add_edge("phase9", "final")

    g.add_edge("phase8_light", "phase9_light")
    g.add_edge("phase9_light", "final")

    g.add_edge("final", END)

    saver = MemorySaver() if checkpoint else None
    return g.compile(checkpointer=saver)

__all__ = ["build_compliance_graph"]
