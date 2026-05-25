from . import config

# Validation SOU (Statement of Use)
def validate_sou(sou, expected_pathway=None):
    """Xác định claim pathway"""
    reasons = []
    pathway = (sou.get("statement_of_use") or "").strip()
    if pathway not in {"in_accordance", "with_reference"}:
        reasons.append(
            f"statement_of_use must be 'in_accordance' or 'with_reference', got {pathway!r}"
        )
    if expected_pathway and pathway != expected_pathway:
        reasons.append(
            f"statement_of_use mismatch: expected {expected_pathway!r}, got {pathway!r}"
        )
    return (len(reasons) == 0, reasons)

# Omission rules 
def is_not_applicable(reason):
    """Kiểm tra reason có phải là "Not applicable"."""
    if reason is None:
        return False
    return str(reason).strip().casefold() == "Not applicable".casefold()

def validate_omission_reason(reason):
    """Kiểm tra reason có phải là 1 trong 4 GRI-1-valid omission reason."""
    if reason is None:
        return False
    r = str(reason).strip().casefold()
    return r in {valid.casefold() for valid in config.VALID_OMISSION_REASONS}

def is_disclosure_omitted(row):
    """Kiểm tra row GCI có omission_reason không."""
    reason = row.get("omission_reason")
    if reason is None:
        return False
    return bool(str(reason).strip())

# Coverage check 
def _presence_entry(row):
    """Trả về dict trạng thái presence cho một disclosure — row=None nghĩa là thiếu hoàn toàn."""
    if row is None:
        return {
            "present": False,
            "omitted": False,
            "location_raw": None,
            "page_list": [],
            "is_location_supported": 0,
        }
    return {
        "present": True,
        "omitted": is_disclosure_omitted(row),
        "location_raw": row.get("location_raw"),
        "page_list": list(row.get("page_list") or []),
        "is_location_supported": int(row.get("is_location_supported", 0) or 0),
    }

def check_disclosure_presence(gci, expected_ids):
    """Từ GRI content index, kiểm tra tình trạng các disclosure theo expected_ids."""
    by_id = {}
    for row in gci:
        d_id = str(row.get("disclosure_id") or "").strip()
        if d_id in expected_ids:
            by_id.setdefault(d_id, row)  # giữ dòng đầu tiên nếu trùng ID

    result = {}
    for d_id in expected_ids:
        row = by_id.get(d_id)
        result[d_id] = _presence_entry(row)
    return result

# Appendix checklists (GRI 1 Appendix 1 & 2)
_APPENDIX1_ITEMS = [
    ("i",    "Title of the report"),
    ("ii",   "Statement of use (in_accordance)"),
    ("iii",  "GRI 1 used"),
    ("iv",   "List of applicable sector standards"),
    ("v",    "List of material topics"),
    ("vi",   "List of non-material sector topics with explanation"),
    ("vii",  "List of reported disclosures with disclosure IDs"),
    ("viii", "GRI Standard titles for reported disclosures"),
    ("ix",   "List of omitted disclosures"),
    ("x",    "Sector reference numbers (gri_sector_standard_ref_no)"),
    ("xi",   "Location of each disclosure"),
    ("xii",  "Reasons for omission"),
]

_APPENDIX2_ITEMS = [
    ("i",   "Title of the report"),
    ("ii",  "Statement of use (with_reference)"),
    ("iii", "GRI 1 used"),
    ("iv",  "List of reported disclosures with disclosure IDs"),
    ("v",   "GRI Standard titles for reported disclosures"),
    ("vi",  "Location of each disclosure"),
]

# Phase 8: Content Index validation
def _parse_sector_standards(sou: dict) -> list:
    ss_raw = sou.get("sector_standards") or []
    if isinstance(ss_raw, str):
        return [s.strip() for s in ss_raw.split(",") if s.strip()]
    return list(ss_raw)

def _check_title(sou: dict, state: dict) -> tuple:
    title = sou.get("title") or state.get("report_id") or ""
    return bool(title), str(title)

def _check_sou_pathway(sou: dict, expected: str) -> tuple:
    su = sou.get("statement_of_use")
    return su == expected, str(su or "")

def _check_gri1_version(sou: dict) -> tuple:
    g1 = sou.get("gri1_used")
    return g1 == "GRI 1: Foundation 2021", str(g1 or "")

def _check_sector_standards(ss: list) -> tuple:
    return bool(ss), ", ".join(ss)[:120]

def _check_material_topics(materiality_map, gci: list) -> tuple:
    mts = []
    if materiality_map is not None:
        mts = list(getattr(materiality_map, "material_topics", []) or [])
    if not mts:
        mts = sorted({r["material_topic"] for r in gci if r.get("material_topic")})
    preview = f"{len(mts)} topics: {', '.join(mts[:5])}{'...' if len(mts) > 5 else ''}"
    return bool(mts), preview

def _check_non_material(non_material: list) -> tuple:
    # List rỗng = tất cả sector topics đều material → N/A, pass.
    # List non-empty → mọi entry phải có Explanation.
    if not non_material:
        return True, "0 entries (all sector topics are material — N/A)"
    ok = all((r.get("Explanation") or "").strip() for r in non_material)
    return ok, f"{len(non_material)} entries"

def _check_disclosure_ids(gci: list) -> tuple:
    # Module 2 luôn populate disclosure_id; thiếu = row malformed.
    n = sum(1 for r in gci if r.get("disclosure_id"))
    return bool(gci) and n == len(gci), f"{n} / {len(gci)} rows have disclosure_id"

def _check_gri_standard(gci: list) -> tuple:
    n = sum(1 for r in gci if r.get("gri_standard"))
    return bool(gci) and n == len(gci), f"{n} / {len(gci)} rows have gri_standard"

def _check_omission_list(omitted: list) -> tuple:
    # Structural presence check — Phase 7 đảm nhận semantic completeness.
    return True, f"{len(omitted)} omission rows"

def _check_sector_refs(ss: list, gci: list) -> tuple:
    # N/A khi không có sector standard được claim.
    if not ss:
        return True, "N/A (no sector standard claimed)"
    n_ref = sum(1 for r in gci if r.get("gri_sector_standard_ref_no"))
    ok = any(r.get("gri_sector_standard_ref_no") for r in gci)
    return ok, f"{n_ref} rows have sector ref"

def _check_location(gci: list) -> tuple:
    n_loc = sum(1 for r in gci if r.get("location_raw"))
    ok = bool(gci) and all(
        r.get("location_raw") or is_disclosure_omitted(r) for r in gci
    )
    return ok, f"{n_loc} / {len(gci)} rows have location_raw"

def _check_omission_reasons(omitted: list) -> tuple:
    # all([]) = True khi không có omission — hành vi đúng.
    n_valid = sum(1 for r in omitted if validate_omission_reason(r.get("omission_reason")))
    ok = all(validate_omission_reason(r.get("omission_reason")) for r in omitted)
    return ok, f"{n_valid} / {len(omitted)} valid reasons"

# Public checklist builders
def appendix1_checklist(state: dict) -> list:
    """GRI 1 Appendix 1 (in_accordance) — checklist 12 item."""
    sou          = (state.get("inputs") or {}).get("sou") or {}
    gci          = (state.get("inputs") or {}).get("gci") or []
    non_material = (state.get("inputs") or {}).get("non_material") or []
    materiality_map = state.get("materiality_map")

    def row(idx, field, ok, preview):
        item_id, item_name = _APPENDIX1_ITEMS[idx]
        return {"item_id": item_id, "item_name": item_name,
                "present": ok, "evidence_field": field, "value_preview": preview}

    ss = _parse_sector_standards(sou)
    omitted = [r for r in gci if is_disclosure_omitted(r)]

    return [
        row(0, "state.report_id / sou.title", *_check_title(sou, state)),
        row(1, "inputs.sou.statement_of_use", *_check_sou_pathway(sou, "in_accordance")),
        row(2, "inputs.sou.gri1_used", *_check_gri1_version(sou)),
        row(3, "inputs.sou.sector_standards", *_check_sector_standards(ss)),
        row(4, "materiality_map.material_topics", *_check_material_topics(materiality_map, gci)),
        row(5, "inputs.non_material", *_check_non_material(non_material)),
        row(6, "inputs.gci[*].disclosure_id", *_check_disclosure_ids(gci)),
        row(7, "inputs.gci[*].gri_standard", *_check_gri_standard(gci)),
        row(8, "inputs.gci[*] where omission_reason != null", *_check_omission_list(omitted)),
        row(9, "inputs.gci[*].gri_sector_standard_ref_no", *_check_sector_refs(ss, gci)),
        row(10, "inputs.gci[*].location_raw", *_check_location(gci)),
        row(11, "inputs.gci[*].omission_reason where omitted", *_check_omission_reasons(omitted)),
    ]

def appendix2_checklist(state: dict) -> list:
    """GRI 1 Appendix 2 (with_reference) — checklist 6 item."""
    sou = (state.get("inputs") or {}).get("sou") or {}
    gci = (state.get("inputs") or {}).get("gci") or []

    def row(idx, field, ok, preview):
        item_id, item_name = _APPENDIX2_ITEMS[idx]
        return {"item_id": item_id, "item_name": item_name,
                "present": ok, "evidence_field": field, "value_preview": preview}

    return [
        row(0, "state.report_id / sou.title", *_check_title(sou, state)),
        row(1, "inputs.sou.statement_of_use", *_check_sou_pathway(sou, "with_reference")),
        row(2, "inputs.sou.gri1_used", *_check_gri1_version(sou)),
        row(3, "inputs.gci[*].disclosure_id", *_check_disclosure_ids(gci)),
        row(4, "inputs.gci[*].gri_standard", *_check_gri_standard(gci)),
        row(5, "inputs.gci[*].location_raw", *_check_location(gci)),
    ]

__all__ = [
    "validate_sou",
    "validate_omission_reason",
    "is_disclosure_omitted",
    "check_disclosure_presence",
    "appendix1_checklist",
    "appendix2_checklist",
]
