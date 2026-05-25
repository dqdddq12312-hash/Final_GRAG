from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# Phán quyết cấp requirement
class RequirementVerdict(BaseModel):
    requirement_id: str
    status: Literal[
        "pass",
        "partial",
        "fail",
        "no_evidence",
        "omitted_valid",
        "omitted_invalid",
    ]
    rationale: str = Field(default="", max_length=400)
    citations: list[str] = Field(default_factory=list, max_length=5)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    source: Literal["llm", "cache"] = "llm"
    decision_path: Literal[
        "standard",
        "condition_not_applicable", # tuyên bố không đáp ứng điều kiện -> pass
        "reported_non_existence", # tuyên bố không tồn tại -> pass
    ] = "standard"

# Kết quả thô của một lần gọi evidence judge cho một disclosure
class DisclosureBatchVerdict(BaseModel):
    disclosure_id: str
    requirement_verdicts: list[RequirementVerdict]

# Phán quyết của disclosure
class DisclosureVerdict(BaseModel):
    disclosure_id: str
    standard_id: str
    requirement_verdicts: list[RequirementVerdict]
    overall: Literal[
        "pass",
        "fail",
        "partial",
        "omitted",
        "no_evidence",
        "external_verification",
    ]
    notes: str = ""
    evidence_unrecoverable: bool = False
    material_topic: Optional[str] = None

# GRI 2 scope anchor (GRI 2-2)
class ScopeAnchor(BaseModel):
    entities: list[str] # danh sách tên pháp nhân/đơn vị
    consolidation_basis: str # cơ sở hợp nhất
    entities_sources: list[str] # danh sách id chunk_ids
    differs_from_financial: bool # true nếu report khác với financial report

# Materiality map 
class NonMaterialEntry(BaseModel):
    material_topic: str 
    topics: str
    Explanation: str  # Capital E — convention từ JSON input GCI

class MaterialityMap(BaseModel):
    material_topics: list[str]
    non_material_topics: list[NonMaterialEntry]
    base_topic_assignment: dict[str, list[str]]
    material_topic_to_base_topics: dict[str, list[str]]
    non_material_match_index: dict[str, list[str]]
    unmatched_reported_topics: list[str]

# Pending omission ở cấp disclosure
class PendingOmission(BaseModel):
    source_phase: Literal[
        "phase3", # GRI 2 (2-1 -> 2.30)
        "phase5", # GRI 3 (3-1/3-2 hard-failed hoặc 3-3)
        "phase6_5a", # Chưa dùng
        "phase6_5b", # omission_reason != "Not applicable"
        "phase6_non_material_cross_check", # Topic non-material nhưng vẫn có row GCI
    ]
    standard_id: str
    disclosure_id: str
    material_topic: Optional[str] = None
    omission_reason: Optional[str]
    omission_explanation: Optional[str]
    row: dict

# Kết quả của 1 phase
class PhaseResult(BaseModel):
    phase: str
    status: Literal[
        "pass",
        "fail",
        "partial",
        "external_verification",
        "skipped",
    ]
    findings: list[str]
    artifacts: dict

# Phạm vi adudit
class ScopeDisclaimer(BaseModel):
    sector_standard: Optional[str]
    checked: list[str]
    not_checked: list[str]
    recommendation: str

# State của LangGraph
class ComplianceState(TypedDict, total=False):
    report_id: str
    inputs: dict
    claim_pathway: Literal["in_accordance", "with_reference", "unclear"]
    sector_standards: list[str]
    scope_anchor: Optional[ScopeAnchor]
    materiality_map: Optional[MaterialityMap]
    pending_omissions: list[PendingOmission]
    phase_results: dict[str, PhaseResult]
