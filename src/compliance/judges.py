import json
from pydantic import BaseModel
from types import SimpleNamespace

from . import cache as cache_mod
from . import config
from . import llm as llm_mod
from . import nli as nli_mod
from . import registries
from .prompts import PROMPT_ANCHOR, PROMPT_EVIDENCE_BATCH, PROMPT_MATCHER
from .state import (
    DisclosureBatchVerdict,
    RequirementVerdict,
    ScopeAnchor,
)

# Schema nội bộ cho anchor combo 
class _AnchorComboPayload(BaseModel):
    """Schema cho LLM call đồng thời trả verdict 2-2 + scope anchor."""

    disclosure_batch_verdict: DisclosureBatchVerdict
    scope_anchor: ScopeAnchor

# Prompt builders 
def _format_requirements_block(grouped):
    """Chuyển danh sách requirement đã nhóm thành đoạn REQUIREMENTS trong prompt judge.

    Input `grouped` là list cặp (parent_stem, leaves) từ
    `registries.group_requirements_by_parent`: requirement độc lập có stem rỗng;
    sub-requirement chung một stem cha được liệt kê lùi dòng dưới

    Requirement điều kiện (`is_conditional=True`) được gắn tiền tố
    `[CONDITIONAL]` để LLM bật nhánh xử lý "if X, then …" trong prompt header.
    """
    lines = []
    for stem, leaves in grouped:
        if stem:
            lines.append(f"- Parent stem: {stem}")
            for leaf in leaves:
                cond_prefix = "[CONDITIONAL] " if leaf.get("is_conditional") else ""
                lines.append(
                    f"    - {cond_prefix}{leaf.get('requirement_id')}: {leaf.get('requirement_text','').strip()}"
                )
        else:
            for leaf in leaves:
                cond_prefix = "[CONDITIONAL] " if leaf.get("is_conditional") else ""
                lines.append(
                    f"- {cond_prefix}{leaf.get('requirement_id')}: {leaf.get('requirement_text','').strip()}"
                )
    return "\n".join(lines) if lines else "(no requirements)"

def _format_evidence_block(page_pack, char_budget=config.EVIDENCE_BLOCK_CHAR_BUDGET):
    """Format page_pack thành block EVIDENCE CHUNKS cho prompt judge.

    Mỗi chunk một dòng `[chunk_id | pages=…] text`; dừng và cắt khi vượt
    `char_budget` (mặc định `config.EVIDENCE_BLOCK_CHAR_BUDGET`).
    """
    out = []
    used = 0
    for c in page_pack:
        cid = c.get("chunk_id", "")
        pages = c.get("page_numbers") or []
        text = (c.get("content_text") or "").strip().replace("\n", " ")
        block = f"[{cid} | pages={pages}] {text}"
        if used + len(block) > char_budget:
            block = block[: max(0, char_budget - used)] + "…"
            out.append(block)
            break
        out.append(block)
        used += len(block) + 1
    return "\n".join(out) if out else "(empty pack)"

def _format_anchor_block(anchor):
    if anchor is None:
        return ""
    return (
        "SCOPE ANCHOR (from disclosure 2-2 — use as authoritative entity scope):\n"
        + json.dumps(anchor.model_dump(), default=str, ensure_ascii=False)
        + "\n"
    )

def _format_material_topic_block(material_topic):
    """Tạo đoạn MATERIAL TOPIC IN SCOPE cho prompt judge (disclosure theo topic).

    Trả rỗng nếu không có `material_topic`; ngược lại gắn tên topic và nhắc LLM
    chỉ xét evidence liên quan topic đó.
    """
    if not material_topic:
        return ""
    return (
        f'MATERIAL TOPIC IN SCOPE: "{material_topic}"\n\n'
        f'This disclosure is reported PER material topic. Throughout the '
        f'REQUIREMENTS below, every reference to "the topic", "the actions", '
        f'"the impacts", or "the material topic" refers SPECIFICALLY to '
        f'"{material_topic}". The EVIDENCE CHUNKS may contain text about '
        f'several material topics; consider ONLY information directly '
        f'relevant to "{material_topic}" — content discussing other topics '
        f'is OFF-SCOPE for this judgment.\n\n'
    )

# Verdict sanitiser
def _normalize_chunk_id(citation):
    """Strip rendering suffix `"<chunk_id> | pages=[...]"` của prompt.

    Evidence block format `[<chunk_id> | pages=[18]] ...`; qwen3 thỉnh
    thoảng cite blob `"<chunk_id> | p.18"` thay vì id. Strip từ `|` đầu,
    trim bracket/whitespace.
    """
    if not citation:
        return ""
    s = str(citation)
    bar = s.find("|")
    if bar >= 0:
        s = s[:bar]
    return s.strip().strip("[]").strip()

def _sanitize_verdicts(verdicts, page_pack, *, max_citations=config.MAX_CITATIONS_PER_REQUIREMENT):
    pack_ids = {str(c.get("chunk_id") or "") for c in page_pack}
    pack_ids.discard("")
    cleaned = []
    for v in verdicts:
        seen = set()
        ordered = []
        for raw in v.citations or []:
            cid = _normalize_chunk_id(raw)
            if not cid or cid in seen:
                continue
            if pack_ids and cid not in pack_ids:
                # Hallucinated chunk_id — skip silently
                continue
            seen.add(cid)
            ordered.append(cid)
            if len(ordered) >= max_citations:
                break

        dp = v.decision_path
        if dp != "standard":
            # Invariant 1: non-standard path requires status="pass"
            if v.status != "pass":
                dp = "standard"
            # Invariant 2: non-standard path requires >= 1 citation
            elif not ordered:
                dp = "standard"

        cleaned.append(
            v.model_copy(update={"citations": ordered, "decision_path": dp})
        )
    return cleaned

# Evidence judge 
def _chunked(seq, size):
    """Chia seq thành các batch kích thước `size`. size <= 0 → 1 batch nguyên."""
    if size <= 0:
        return [list(seq)]
    return [seq[i: i + size] for i in range(0, len(seq), size)]


def _merge_in_registry_order(llm_verdicts, registry):
    """Sắp xếp verdict theo registry order.

    Requirement thiếu trong llm_verdicts → thêm placeholder no_evidence để
    downstream rely full coverage.
    """
    by_id = {v.requirement_id: v for v in llm_verdicts}
    out = []
    for req in registry:
        rid = str(req.get("requirement_id") or "")
        v = by_id.get(rid)
        if v is None:
            v = RequirementVerdict(
                requirement_id=rid,
                status="no_evidence",
                rationale="No verdict produced.",
                citations=[],
                source="llm",
            )
        out.append(v)
    return out

async def _run_disclosure_batch_impl(*, disclosure_id, disclosure_name,
                                       page_pack, registry, anchor, report_id,
                                       material_topic=None):
    judge_cache = cache_mod.JudgeCache(report_id)
    judge_cache.load()

    anchor_dump = anchor.model_dump() if anchor is not None else None
    anchor_h = cache_mod.anchor_hash(anchor_dump)
    chunk_ids_sorted = tuple(sorted(c.get("chunk_id", "") for c in page_pack))
    evidence_h = cache_mod.evidence_text_hash(page_pack)

    # Bước 1: NLI coverage matrix — entailment cho mọi (req, chunk).
    # Reorder + hint inject conditional theo ablation flag
    # (NLI_REORDER_ENABLED / NLI_HINT_ENABLED) → variant A1/A2/V_new share
    # code path qua env-var override.
    coverage = nli_mod.build_coverage_matrix(registry, page_pack)
    ordered_pack = (
        nli_mod.reorder_pack_by_coverage(page_pack, coverage)
        if config.NLI_REORDER_ENABLED else page_pack
    )

    # Bước 2: Batch mọi req qua LLM judge
    llm_verdicts = []
    use_anchor_in_prompt = (
        anchor is not None and disclosure_id in config.ANCHOR_INJECTION_WHITELIST
    )

    batches = _chunked(registry, config.BATCH_JUDGE_MAX_REQUIREMENTS)
    for batch in batches:
        grouped = registries.group_requirements_by_parent(batch)
        coverage_hint = (
            nli_mod.format_coverage_hint(coverage, batch)
            if config.NLI_HINT_ENABLED else ""
        )
        coverage_hint_block = (coverage_hint + "\n") if coverage_hint else ""
        prompt = PROMPT_EVIDENCE_BATCH.format(
            disclosure_id=disclosure_id,
            disclosure_name=disclosure_name,
            material_topic_block=_format_material_topic_block(material_topic),
            anchor_block=_format_anchor_block(anchor) if use_anchor_in_prompt else "",
            coverage_hint_block=coverage_hint_block,
            requirements_block=_format_requirements_block(grouped),
            evidence_block=_format_evidence_block(ordered_pack),
        )

        # Stable cache key per requirement-batch.
        # v0.v2 PR1.F1-B: requirement_text_hash invalidate khi registry edit.
        # v3.1: PROMPT_VERSION bump invalidate stale CE-era cache.
        # E2b: suffix encode NLI flag → mỗi ablation variant cache isolated
        #      dù share cùng _judge_cache.jsonl.
        _pv_evidence = (
            config.PROMPT_VERSION_EVIDENCE
            + f"_r{int(config.NLI_REORDER_ENABLED)}h{int(config.NLI_HINT_ENABLED)}"
        )
        batch_req_ids = sorted(str(r.get("requirement_id") or "") for r in batch)
        batch_key = cache_mod.judge_cache_key(
            model=config.OLLAMA_LLM_MODEL,
            prompt_version=_pv_evidence,
            requirement_id="|".join(batch_req_ids),
            chunk_ids_sorted=chunk_ids_sorted,
            evidence_text_hash=evidence_h,
            anchor_hash=anchor_h if use_anchor_in_prompt else None,
            requirement_text_hash=cache_mod.requirement_text_hash(batch),
            material_topic=material_topic,
        )

        async def _compute():
            llm = llm_mod.make_judge_llm()
            verdict = await llm_mod._call_llm_with_retry(
                llm, prompt, schema=DisclosureBatchVerdict
            )
            for rv in verdict.requirement_verdicts:
                if not rv.source:
                    rv.source = "llm"
            return DisclosureBatchVerdict(
                disclosure_id=verdict.disclosure_id,
                requirement_verdicts=_sanitize_verdicts(
                    verdict.requirement_verdicts, ordered_pack
                ),
            )

        cached_or_fresh = await judge_cache.get_or_compute(
            batch_key, _compute, DisclosureBatchVerdict
        )
        llm_verdicts.extend(cached_or_fresh.requirement_verdicts)

    # Bước 3: Merge preserve registry order
    merged = _merge_in_registry_order(llm_verdicts, registry)

    return DisclosureBatchVerdict(
        disclosure_id=disclosure_id, requirement_verdicts=merged
    )

# Anchor judge 
async def _run_anchor_impl(*, page_pack, registry, report_id):
    """1 LLM call (thinking ON) trả DisclosureBatchVerdict_2_2 + ScopeAnchor.

    NLI coverage matrix build trên full pack để reorder evidence + per-req
    entailment hint — cùng pipeline run_disclosure_batch.
    """
    judge_cache = cache_mod.JudgeCache(report_id)
    judge_cache.load()

    # NLI coverage cho 2-2 requirement × pack chunk. Reorder + hint
    # conditional theo ablation flag.
    coverage = nli_mod.build_coverage_matrix(registry, page_pack)
    ordered_pack = (
        nli_mod.reorder_pack_by_coverage(page_pack, coverage)
        if config.NLI_REORDER_ENABLED else page_pack
    )
    coverage_hint = (
        nli_mod.format_coverage_hint(coverage, registry)
        if config.NLI_HINT_ENABLED else ""
    )
    coverage_hint_block = (coverage_hint + "\n") if coverage_hint else ""

    # E2b: suffix encode NLI flag → isolated cache per variant
    _pv_anchor = (
        config.PROMPT_VERSION_ANCHOR
        + f"_r{int(config.NLI_REORDER_ENABLED)}h{int(config.NLI_HINT_ENABLED)}"
    )
    chunk_ids_sorted = tuple(sorted(c.get("chunk_id", "") for c in page_pack))
    evidence_h = cache_mod.evidence_text_hash(page_pack)
    batch_req_ids = sorted(str(r.get("requirement_id") or "") for r in registry)
    batch_key = cache_mod.judge_cache_key(
        model=config.OLLAMA_LLM_MODEL,
        prompt_version=_pv_anchor,
        requirement_id="anchor:" + "|".join(batch_req_ids),
        chunk_ids_sorted=chunk_ids_sorted,
        evidence_text_hash=evidence_h,
        anchor_hash=None,
        requirement_text_hash=cache_mod.requirement_text_hash(registry),
    )

    grouped = registries.group_requirements_by_parent(registry)
    prompt = PROMPT_ANCHOR.format(
        coverage_hint_block=coverage_hint_block,
        requirements_block=_format_requirements_block(grouped),
        evidence_block=_format_evidence_block(ordered_pack),
    )

    async def _compute():
        llm = llm_mod.make_anchor_llm()
        combo = await llm_mod._call_llm_with_retry(
            llm, prompt, schema=_AnchorComboPayload
        )
        # Fix-6: scrub anchor batch citation giống regular evidence
        # judge → cached anchor verdict ra cache_dir clean.
        # ScopeAnchor.entities_sources cũng filter vs pack (anchor-side
        # hallucination observed trên small report).
        cleaned_batch = DisclosureBatchVerdict(
            disclosure_id=combo.disclosure_batch_verdict.disclosure_id,
            requirement_verdicts=_sanitize_verdicts(
                combo.disclosure_batch_verdict.requirement_verdicts,
                page_pack,
            ),
        )
        pack_ids = {str(c.get("chunk_id") or "") for c in page_pack}
        pack_ids.discard("")
        seen = set()
        cleaned_sources = []
        for raw in combo.scope_anchor.entities_sources or []:
            cid = _normalize_chunk_id(raw)
            if not cid or cid in seen:
                continue
            if pack_ids and cid not in pack_ids:
                continue
            seen.add(cid)
            cleaned_sources.append(cid)
        cleaned_anchor = combo.scope_anchor.model_copy(
            update={"entities_sources": cleaned_sources}
        )
        return _AnchorComboPayload(
            disclosure_batch_verdict=cleaned_batch,
            scope_anchor=cleaned_anchor,
        )

    combo = await judge_cache.get_or_compute(
        batch_key, _compute, _AnchorComboPayload
    )
    return combo.disclosure_batch_verdict, combo.scope_anchor

# Public namespace handles 
evidence_judge = SimpleNamespace(
    run_disclosure_batch=_run_disclosure_batch_impl,
    run_anchor=_run_anchor_impl,
)

# Topic matcher (Phase 4)
def _topic_matcher_match(material_topic, gri11_catalogue):
    """Sync LLM call mapping reported topic → GRI 11 base topic.

    Trả {base_topic_id, confidence, rationale, candidates}. Khi JSON parse
    fail hoặc LLM error sau 1 re-prompt → conservative
    {base_topic_id: "none", confidence: 0.0, rationale: "", candidates: []}.

    Parsing pipeline dùng `llm._parse_llm_json` để strip qwen3 thinking-mode
    `<think>...</think>` block trước JSON validate.
    """
    from pydantic import Field, model_validator

    class _MatcherPayload(BaseModel):
        base_topic_id: str
        confidence: float = Field(ge=0, le=1)
        rationale: str = Field(default="")
        candidates: list[dict] = Field(default_factory=list)

        @model_validator(mode="before")
        @classmethod
        def _coerce_score_alias(cls, data):
            """Normalize field name + cap length trước Pydantic validate.

            1. Accept `score` alias cho `confidence` — qwen3:14b thỉnh
               thoảng output sai field name.
            2. Truncate `rationale` 300 char + `candidates` 3 item — với
               1500-token budget LLM có thể gen rationale dài hơn gây
               re-prompt trước đây.
            """
            if isinstance(data, dict):
                data = dict(data)
                if "confidence" not in data and "score" in data:
                    data["confidence"] = data.pop("score")
                if "rationale" in data and isinstance(data["rationale"], str):
                    data["rationale"] = data["rationale"][:300]
                if "candidates" in data and isinstance(data["candidates"], list):
                    data["candidates"] = data["candidates"][:3]
            return data

    catalogue_lines = [
        f"- {base_id}: {info.get('topic_name', '')}"
        for base_id, info in gri11_catalogue.items()
    ]
    catalogue_block = "\n".join(catalogue_lines) if catalogue_lines else "(empty)"

    prompt = PROMPT_MATCHER.format(
        material_topic=material_topic,
        catalogue_block=catalogue_block,
    )

    llm = llm_mod.make_matcher_llm()
    payload = None
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        payload = llm_mod._parse_llm_json(content, _MatcherPayload)
    except Exception as e:
        # 1 re-prompt với prefix "valid JSON only"
        try:
            reprompt = (
                "The previous output was malformed; emit valid JSON only "
                "matching the requested schema. No prose, no code fences.\n\n"
                + prompt
            )
            response2 = llm.invoke(reprompt)
            content2 = getattr(response2, "content", response2)
            payload = llm_mod._parse_llm_json(content2, _MatcherPayload)
        except Exception:
            return {
                "base_topic_id": "none",
                "confidence": 0.0,
                "rationale": "",
                "candidates": [],
            }
    return {
        "base_topic_id": payload.base_topic_id,
        "confidence": float(payload.confidence),
        "rationale": payload.rationale,
        "candidates": payload.candidates,
    }

topic_matcher = SimpleNamespace(match=_topic_matcher_match)

__all__ = [
    "PROMPT_EVIDENCE_BATCH",
    "PROMPT_ANCHOR",
    "PROMPT_MATCHER",
    "evidence_judge",
    "topic_matcher",
]
