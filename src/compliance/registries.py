import json
import re

from . import config

_CONDITIONAL_RE = re.compile(r"^\s*if\b", re.IGNORECASE)
_GRI_STD_RE = re.compile(r"^\s*GRI\s+(\d+)")
_GRI_UNITS_CACHE = None
_GRI11_CACHE = None

# Loaders 
def _load_gri_units():
    global _GRI_UNITS_CACHE
    if _GRI_UNITS_CACHE is None:
        with open(config.GRI_UNITS_JSON, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(
                f"{config.GRI_UNITS_JSON} root must be a list, got {type(records)}"
            )
        _GRI_UNITS_CACHE = records
    return _GRI_UNITS_CACHE

def _load_gri11():
    global _GRI11_CACHE
    if _GRI11_CACHE is None:
        with open(config.GRI11_SECTOR_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{config.GRI11_SECTOR_JSON} root must be a list")
        _GRI11_CACHE = data
    return _GRI11_CACHE

# Helper
def short_std_id(value):
    """Trả standard_id dạng ngắn (vd "GRI 305") từ string hoặc GCI row dict."""
    if value is None:
        return ""
    if isinstance(value, dict):
        raw = value.get("gri_standard") or value.get("standard_id") or ""
    else:
        raw = value

    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    match = _GRI_STD_RE.match(raw_str)
    return f"GRI {match.group(1)}" if match else raw_str

# Registry builders 
def build_disclosure_registry(standard_id, req_type=None):
    """Group requirement của 1 standard theo disclosure_id.

    Trả `{disclosure_id: [req_row, ...]}`. Mỗi req_row có thêm field
    `is_conditional` (bool) dựa trên text bắt đầu bằng "if".
    """
    rows = _load_gri_units()
    out = {}
    for row in rows:
        if str(row.get("standard_id") or row.get("standards_id") or "").strip() != standard_id:
            continue
        if req_type is not None:
            if (row.get("requirement_type") or "").strip() != req_type:
                continue
        d_id = str(row.get("disclosure_id") or "").strip()
        if not d_id:
            continue

        req_text = row.get("requirement_text") or ""
        # Copy để không mutate cache gốc
        entry = dict(row)
        entry["is_conditional"] = bool(_CONDITIONAL_RE.match(req_text))
        out.setdefault(d_id, []).append(entry)

    return out

def build_gri11_catalogue():
    """Nhóm disclosure theo base_topic_id cho GRI 11.

    Trả `{base_topic_id: {"topic_name": str, "promoted_disclosures": [[std_id, d_id], ...]}}`.
    """
    rows = _load_gri11()
    out = {}
    for row in rows:
        t_name = str(row.get("topic_name") or "").strip()
        bt_id = str(row.get("base_topic_id") or "").strip()
        s_id = str(row.get("standard_id") or row.get("standards_id") or "").strip()
        d_id = str(row.get("disclosure_id") or "").strip()

        if bt_id not in out:
            out[bt_id] = {
                "topic_name": t_name,
                "promoted_disclosures": [],
            }
        out[bt_id]["promoted_disclosures"].append([s_id, d_id])

    return out

def group_gci_by_material_topic(gci):
    """Nhóm GCI row theo material_topic.

    Trả `{material_topic: [row, ...]}`. Chuỗi rỗng được normalize về None.
    """
    out = {}
    for row in gci:
        mt = row.get("material_topic")
        if isinstance(mt, str) and not mt.strip():
            mt = None
        out.setdefault(mt, []).append(row)
    return out

def group_requirements_by_parent(reqs):
    """Nhóm requirement theo parent_requirement.

    Trả `[(parent_key, [req, ...]), ...]`. Requirement không có parent
    được trả từng cái riêng dưới key "".
    """
    result = []
    parent_groups: dict[str, list] = {}

    for req in reqs:
        raw_parent = req.get("parent_requirement")
        parent_key = (raw_parent or "").strip()
        if not parent_key:
            result.append(("", [req]))
        else:
            parent_groups.setdefault(parent_key, []).append(req)

    for parent_key, children in parent_groups.items():
        result.append((parent_key, children))

    return result

__all__ = [
    "build_disclosure_registry",
    "build_gri11_catalogue",
    "group_gci_by_material_topic",
    "group_requirements_by_parent",
    "short_std_id",
    "_CONDITIONAL_RE",
]
