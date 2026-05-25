import re
import unicodedata

from rapidfuzz import fuzz

from src.compliance import config, registries, judges
from src.compliance.state import MaterialityMap, NonMaterialEntry, PhaseResult

SECTOR_REF_RE = re.compile(r"\b11\.(\d+)\.\d+\b")
NM_TOPIC_ID_RE = re.compile(r"\b11\.(\d+)\b")

_PLURAL_TO_SINGULAR = (
    ("practices", "practice"),
    ("emissions", "emission"),
)

def _normalize_topic_name(s):
    """Normalize topic name cho fuzzy matching: 
    NFKC → lower → collapse whitespace → de-pluralize."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[\u00A0\s]+", " ", s)
    s = re.sub(r"[^\w\s&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for plural, singular in _PLURAL_TO_SINGULAR:
        s = re.sub(rf"\b{plural}\b", singular, s)
    return s

def _accept_llm_verdict(verdict, catalogue):
    """Parse và validate LLM verdict theo threshold confidence.

    Trả (accepted: bool, base_topic_id: str, confidence: float).
    """
    v_base = verdict.get("base_topic_id")
    v_conf = float(verdict.get("confidence", 0.0))
    accepted = (
        v_base not in (None, "none")
        and v_base in catalogue
        and v_conf >= config.LLM_MATCHER_MIN_CONFIDENCE
    )
    return accepted, v_base, v_conf

def _match_material_topic(mt, mt_rows, catalogue, catalogue_norm, skip_3_b_i):
    """3-tier matching cho một material topic.

    Trả (bases, tier1_rows, tier2_row, tier3_row).
    Chính xác một trong {tier1_rows, tier2_row, tier3_row} khác None khi có match/audit,
    còn lại là None.
    """
    # Tier 1: sector_ref short-circuit, multi-base aware (A.3)
    ref_bases = []
    tier1_rows = []
    for r in mt_rows:
        raw = (r.get("gri_sector_standard_ref_no") or "").strip()
        if not raw:
            continue
        for part in re.split(r"[,;]", raw):
            m = SECTOR_REF_RE.search(part.strip())
            if not m:
                continue
            base_id = f"11.{m.group(1)}"
            if base_id not in catalogue or base_id in ref_bases:
                continue
            ref_bases.append(base_id)
            tier1_rows.append({
                "material_topic": mt,
                "base_topic_id": base_id,
                "evidence_disclosure_id": r.get("disclosure_id"),
                "evidence_ref_part": part.strip(),
            })
    if ref_bases:
        return ref_bases, tier1_rows, None, None

    # Tier 2: rapidfuzz trên normalized name
    mt_norm = _normalize_topic_name(mt)
    best_id, best_score = None, 0
    for base_id, cat_norm in catalogue_norm.items():
        score = fuzz.WRatio(mt_norm, cat_norm)
        if score > best_score:
            best_id, best_score = base_id, score

    if best_score >= config.FUZZY_THRESHOLD and best_id:
        tier2_row = {
            "material_topic": mt,
            "base_topic_id": best_id,
            "similarity": best_score,
        }
        return [best_id], None, tier2_row, None

    # Tier 3: residual LLM matcher (A.5 — confidence guarded)
    if skip_3_b_i:
        return [], None, None, None

    # Pre-screen — best fuzzy thấp xa threshold → topic ngoài GRI 11 scope
    if best_score < config.FUZZY_LOW_SKIP_LLM:
        tier3_row = {
            "material_topic": mt,
            "base_topic_id": None,
            "confidence": 0.0,
            "rationale": (
                f"skipped LLM matcher: best Tier 2 fuzzy score "
                f"{best_score} < FUZZY_LOW_SKIP_LLM "
                f"({config.FUZZY_LOW_SKIP_LLM}); topic outside GRI 11 scope"
            ),
            "candidates": [],
            "accepted": False,
            "source": "fuzzy_low_skip",
            "best_fuzzy_score": best_score,
            "best_fuzzy_base_id": best_id,
        }
        return [], None, None, tier3_row

    verdict = judges.topic_matcher.match(mt, catalogue)
    accepted, v_base, v_conf = _accept_llm_verdict(verdict, catalogue)
    tier3_row = {
        "material_topic": mt,
        "base_topic_id": v_base,
        "confidence": v_conf,
        "rationale": verdict.get("rationale", ""),
        "candidates": verdict.get("candidates", []),
        "accepted": accepted,
        "source": "llm",
        "best_fuzzy_score": best_score,
        "best_fuzzy_base_id": best_id,
    }
    bases = [v_base] if accepted else []
    return bases, None, None, tier3_row

def _classify_all_material_topics(material_topics, gci, catalogue, catalogue_norm, skip_3_b_i):
    """Phân loại tất cả material topics qua 3 tiers.

    Trả (tier1_matches, tier2_matches, tier3_matches,
         base_topic_assignment, material_topic_to_base_topics, unmatched_reported).
    """
    base_topic_assignment = {b: [] for b in catalogue}
    material_topic_to_base_topics = {}
    unmatched_reported = []
    tier1_matches, tier2_matches, tier3_matches = [], [], []

    for mt in material_topics:
        # Sort disclosure 3-3 lên đầu — canonical link giữa material topic và sector ref (B.5)
        mt_rows = [r for r in gci if r.get("material_topic") == mt]
        mt_rows.sort(key=lambda r: 0 if r.get("disclosure_id") == "3-3" else 1)

        bases, t1_rows, t2_row, t3_row = _match_material_topic(
            mt, mt_rows, catalogue, catalogue_norm, skip_3_b_i
        )

        if t1_rows:
            tier1_matches.extend(t1_rows)
        if t2_row:
            tier2_matches.append(t2_row)
        if t3_row:
            tier3_matches.append(t3_row)

        material_topic_to_base_topics[mt] = bases
        if bases:
            for base_id in bases:
                base_topic_assignment[base_id].append(mt)
        else:
            unmatched_reported.append(mt)

    return (
        tier1_matches, tier2_matches, tier3_matches,
        base_topic_assignment, material_topic_to_base_topics, unmatched_reported,
    )

def _match_nm_entry(topics_text, topics_norm, catalogue, catalogue_norm, skip_3_b_i):
    """4-tier cascade matching cho một non-material entry.

    Trả list base_topic_id đã match (có thể rỗng).
    """
    matched = []

    # Tier A: parse base_topic_id từ topics_text (canonical)
    for m in NM_TOPIC_ID_RE.finditer(topics_text):
        base = f"11.{m.group(1)}"
        if base in catalogue and base not in matched:
            matched.append(base)

    # Tier B: catalogue topic_name là substring của topics_text
    if not matched and topics_norm:
        for base_id, cat_norm in catalogue_norm.items():
            if cat_norm and cat_norm in topics_norm and base_id not in matched:
                matched.append(base_id)

    # Tier C: rapidfuzz fallback — tier_c_best_score cần cho pre-screen Tier D
    tier_c_best_score = 0
    if not matched and topics_norm:
        best_id, best_score = None, 0
        for base_id, cat_norm in catalogue_norm.items():
            score = fuzz.WRatio(topics_norm, cat_norm)
            if score > best_score:
                best_id, best_score = base_id, score
        tier_c_best_score = best_score
        if best_score >= config.FUZZY_THRESHOLD and best_id:
            matched.append(best_id)

    # Tier D: LLM fallback (skip khi không phải GRI 11, hoặc Tier C score quá thấp)
    if (
        not matched
        and not skip_3_b_i
        and tier_c_best_score >= config.FUZZY_LOW_SKIP_LLM
    ):
        v = judges.topic_matcher.match(topics_text, catalogue)
        accepted, v_base, _ = _accept_llm_verdict(v, catalogue)
        if accepted:
            matched.append(v_base)

    return matched

def _process_non_material(non_material_entries, catalogue, catalogue_norm, skip_3_b_i,
                           base_topic_assignment, findings):
    """Cascade matching cho tất cả NM entries + cập nhật base_topic_assignment.

    Trả (non_material_match_index, nm_unmapped).
    """
    non_material_match_index = {}
    nm_unmapped = []

    for nm in non_material_entries:
        topics_text = nm.topics or ""
        topics_norm = _normalize_topic_name(topics_text)

        matched = _match_nm_entry(topics_text, topics_norm, catalogue, catalogue_norm, skip_3_b_i)
        non_material_match_index[topics_text] = matched

        # Apply matches với conflict detection — "missing" sentinel chưa tồn tại ở bước này
        for base in matched:
            existing = base_topic_assignment[base]
            material_owners = [x for x in existing if x != "non_material"]
            if material_owners:
                findings.append(
                    f"contradiction: non-material entry '{topics_text}' covers base topic "
                    f"{base}, which is already a material topic "
                    f"({', '.join(material_owners)})"
                )
                # Material claim wins — không add 'non_material'
            elif "non_material" not in existing:
                base_topic_assignment[base].append("non_material")

        if not matched:
            findings.append(
                f"3-b-ii: non-material entry '{topics_text}' không map được vào "
                f"bất kỳ GRI 11 base topic nào"
            )
            nm_unmapped.append(topics_text)

    return non_material_match_index, nm_unmapped

def phase4_material_topics_node(state):
    """Xác định mapping material topic → GRI 11 base topic (3-tier) + non-material cascade."""
    inputs = state["inputs"]
    gci = inputs["gci"]
    non_material = inputs["non_material"]
    sector_standards = state.get("sector_standards", []) or []

    findings = []

    skip_3_b_i = config.GRI_11_CANONICAL not in sector_standards
    if skip_3_b_i:
        findings.append(
            "GRI 11 not in sector_standards — base-topic reconciliation (3-b-i) skipped"
        )

    # Bước 1: build catalogue 22 base topic 11.1–11.22
    catalogue = registries.build_gri11_catalogue()
    # A1 + A6: pre-compute normalized names một lần — tránh normalize 22 string lặp trong mọi inner loop
    catalogue_norm = {b: _normalize_topic_name(info["topic_name"]) for b, info in catalogue.items()}

    # Bước 2: reported material topics (distinct, non-null)
    material_topics = sorted({r["material_topic"] for r in gci if r.get("material_topic")})

    # Bước 3: reported non-material topics
    non_material_entries = [NonMaterialEntry(**row) for row in non_material]

    # Bước 4: 3-tier matcher (per material topic; first hit wins)
    (
        tier1_matches, tier2_matches, tier3_matches,
        base_topic_assignment, material_topic_to_base_topics, unmatched_reported,
    ) = _classify_all_material_topics(material_topics, gci, catalogue, catalogue_norm, skip_3_b_i)

    # Bước 5: non-material cascade (A.4 + A.7)
    non_material_match_index, nm_unmapped = _process_non_material(
        non_material_entries, catalogue, catalogue_norm, skip_3_b_i, base_topic_assignment, findings
    )

    # Bước 6: mark uncovered base topic là 'missing' (singleton sentinel)
    for base_id, owners in base_topic_assignment.items():
        if not owners:
            base_topic_assignment[base_id] = ["missing"]

    materiality_map = MaterialityMap(
        material_topics=material_topics,
        non_material_topics=non_material_entries,
        base_topic_assignment=base_topic_assignment,
        material_topic_to_base_topics=material_topic_to_base_topics,
        non_material_match_index=non_material_match_index,
        unmatched_reported_topics=unmatched_reported,
    )

    # Bước 7: Phase verdict (NO 3-c — đã chuyển Phase 5)
    a_pass = bool(material_topics)
    if not a_pass:
        findings.append("3-a: no material topics reported")

    if not skip_3_b_i:
        missing_bases = [b for b, v in base_topic_assignment.items() if v == ["missing"]]
        b_i_pass = not missing_bases
        if missing_bases:
            findings.append(
                f"3-b-i: GRI 11 base topics missing from materiality assessment: {missing_bases}"
            )
    else:
        b_i_pass = True  # skipped; finding đã emit lúc đầu

    b_ii_explanations_ok = all((nm.Explanation or "").strip() for nm in non_material_entries)
    if not b_ii_explanations_ok:
        findings.append("3-b-ii: at least one non-material entry has empty Explanation")

    b_ii_pass = b_ii_explanations_ok and not nm_unmapped

    # Advisory: reported material topic ngoài GRI 11 scope — chỉ surface, không hạ status
    if not skip_3_b_i and unmatched_reported:
        findings.append(
            f"3-b-i (advisory): {len(unmatched_reported)} reported "
            f"material topic(s) outside GRI 11 scope — manual review "
            f"required: {unmatched_reported}"
        )

    status = "pass" if (a_pass and b_i_pass and b_ii_pass) else "fail"

    pr = PhaseResult(
        phase="phase4",
        status=status,
        findings=findings,
        artifacts={
            "gri11_catalogue": catalogue,
            "reported_material_topics": material_topics,
            "reported_non_material_topics": [nm.model_dump() for nm in non_material_entries],
            "tier1_matches": tier1_matches,
            "tier2_matches": tier2_matches,
            "tier3_matches": tier3_matches,
            "non_material_match_index": non_material_match_index,
            "materiality_map": materiality_map.model_dump(),
        },
    )
    return {
        "materiality_map": materiality_map,
        "phase_results": {**state.get("phase_results", {}), "phase4": pr},
    }

__all__ = ["phase4_material_topics_node", "_normalize_topic_name"]
