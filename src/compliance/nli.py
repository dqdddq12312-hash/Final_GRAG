import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from sentence_transformers import CrossEncoder
from . import config

logger = logging.getLogger(__name__)

# Index cho output softmax của NLI model (contradiction, entailment, neutral)
_IDX_CONTRADICTION = 0
_IDX_ENTAILMENT = 1
_IDX_NEUTRAL = 2

@dataclass(frozen=True)
class CoverageScore:
    """NLI triple đã softmax cho một cặp (requirement, chunk)."""

    entail: float
    contradict: float
    neutral: float

@lru_cache(maxsize=1)
def _nli_model():
    env_device = os.environ.get("NLI_DEVICE", "").strip().lower()
    device = env_device if env_device else "cpu"
    return CrossEncoder(config.NLI_MODEL, device=device)

def build_coverage_matrix(reqs, chunks, batch_size=32):
    """Tính NLI entailment cho mọi cặp (req, chunk) trong 1 batched call.

    Trả dict `{req_id: {chunk_id: CoverageScore}}`.
    """
    if not reqs or not chunks:
        return {}

    model = _nli_model() # cross-encoder/nli-deberta-v3-base

    req_ids = []
    chunk_ids = []
    pairs = []

    # Tạo cặp (chunk_text, req_text) cho mọi (requirement, chunk)
    for req in reqs:
        req_id = str(req.get("requirement_id") or "").strip()
        req_text = str(req.get("requirement_text") or "").strip()
        if not req_id or not req_text:
            continue
        for chunk in chunks:
            cid = str(chunk.get("chunk_id") or "").strip()
            chunk_text = (chunk.get("content_text") or "").strip().replace("\n", " ")
            if not cid or not chunk_text:
                continue
            req_ids.append(req_id)
            chunk_ids.append(cid)
            pairs.append((chunk_text, req_text))

    if not pairs:
        return {}

    try:
        scores_raw = model.predict(pairs, apply_softmax=True, batch_size=batch_size) # sentence_transformers. CrossEncoder
    except Exception as e:
        logger.warning("NLI model.predict failed: %s — returning empty matrix", e)
        return {}

    # Tạo ma trận NLI scores
    matrix = {}
    for i, row in enumerate(scores_raw):
        score = CoverageScore(
            entail=float(row[_IDX_ENTAILMENT]),
            contradict=float(row[_IDX_CONTRADICTION]),
            neutral=float(row[_IDX_NEUTRAL]),
        )
        matrix.setdefault(req_ids[i], {})[chunk_ids[i]] = score

    return matrix

def reorder_pack_by_coverage(pack, matrix):
    """Sort chunks theo max entailment → giảm positional bias cho LLM judge."""
    if not matrix or not pack:
        return list(pack)

    def _max_entail(chunk):
        cid = str(chunk.get("chunk_id") or "")
        best = 0.0
        for req_scores in matrix.values():
            s = req_scores.get(cid)
            if s is not None and s.entail > best:
                best = s.entail
        return best

    return sorted(pack, key=_max_entail, reverse=True)

def _entail_score(item):
    chunk_id, score = item
    return score.entail

def format_coverage_hint(matrix, reqs, top_k=3, min_entail=0.30):
    """NLI matrix → coverage hint block cho LLM prompt.

    Với mỗi requirement, liệt kê tối đa `top_k` chunk có entailment cao nhất
    (≥ `min_entail`). Block này là advisory — LLM không auto-pass/fail dựa trên
    score NLI.
    """
    if not matrix:
        return ""

    lines = []
    for req in reqs:
        req_id = str(req.get("requirement_id") or "").strip()
        if not req_id or req_id not in matrix:
            continue
        req_text = str(req.get("requirement_text") or "").strip()
        req_label = req_text[:80] + ("…" if len(req_text) > 80 else "")

        req_scores = matrix[req_id]
        ranked = sorted(req_scores.items(), key=_entail_score, reverse=True)

        top = []
        for chunk_id, score in ranked:
            if len(top) >= top_k: # 3 chunks
                break
            if score.entail >= min_entail: # 0.30
                top.append((chunk_id, score))

        if top:
            chunk_labels = [f"{cid} (entail={s.entail:.2f})" for cid, s in top]
            parts = ", ".join(chunk_labels)
            lines.append(f"- {req_id} ({req_label}): {parts}")
        else:
            lines.append(f"- {req_id} ({req_label}): no evidence above threshold")

    if not lines:
        return ""

    header = (
        "EVIDENCE COVERAGE HINTS — NLI entailment scores (advisory only; "
        "high score does not auto-pass, low score does not auto-fail):\n"
    )
    return header + "\n".join(lines)

__all__ = [
    "CoverageScore",
    "build_coverage_matrix",
    "reorder_pack_by_coverage",
    "format_coverage_hint",
]
