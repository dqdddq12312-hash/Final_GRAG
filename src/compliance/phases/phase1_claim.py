from src.compliance import config, rules
from src.compliance.state import PhaseResult

# Helpers 
def _normalize_sector_standard(raw):
    """Trả canonical name nếu raw chứa 1 supported sector standard, ngược lại None."""
    s = raw.strip().lower()
    for canonical in config.SUPPORTED_SECTOR_STANDARDS:
        if canonical.lower() in s:
            return canonical
    return None

# Phase node
def phase1_claim_node(state):
    """Xác định claim pathway (in_accordance / with_reference / unclear) và normalize sector standards."""
    sou = (state.get("inputs") or {}).get("sou") or {}

    # Bước 1: claim pathway — chỉ phụ thuộc statement_of_use.
    ok, reasons = rules.validate_sou(sou)
    pathway = (sou.get("statement_of_use") or "").strip()
    if pathway not in {"in_accordance", "with_reference"}:
        pathway = "unclear"
    findings = list(reasons)

    # Bước 2: normalize sector_standards cho downstream Phase 4
    raw_sector = sou.get("sector_standards") or []
    if isinstance(raw_sector, str):
        raw_entries = [s.strip() for s in raw_sector.split(",") if s.strip()]
    elif isinstance(raw_sector, list):
        raw_entries = [str(s).strip() for s in raw_sector if str(s).strip()]
    else:
        raw_entries = []

    sector_standards = []
    for raw in raw_entries:
        canonical = _normalize_sector_standard(raw)
        if canonical is not None:
            sector_standards.append(canonical)
        else:
            sector_standards.append(raw)
            findings.append(
                f"Sector standard {raw!r} not supported by Module 3 — Phase 4 "
                f"base-topic reconciliation will be skipped"
            )

    status = "pass" if ok else "fail"

    pr = PhaseResult(
        phase="phase1",
        status=status,
        findings=findings,
        artifacts={
            "claim_pathway": pathway,
            "gri1_used": sou.get("gri1_used"),
            "sector_standards": sector_standards,
        },
    )

    return {
        "claim_pathway": pathway,
        "sector_standards": sector_standards,
        "phase_results": {**state.get("phase_results", {}), "phase1": pr},
    }

__all__ = ["phase1_claim_node", "_normalize_sector_standard"]
