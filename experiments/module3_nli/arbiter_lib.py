import asyncio
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compliance.registries import short_std_id  # noqa: E402

REGISTRY_JSON = REPO_ROOT / "metadata" / "gri_units" / "gri_units_with_sector_metadata.json"
REPORT_UNITS_DIR = REPO_ROOT / "metadata" / "report_units"

PRIMARY_VARIANTS: tuple[str, str, str, str] = ("a0", "a1", "a2", "v_new")
VALID_STATUSES = {"pass", "partial", "no_evidence"}

# LLM call constants
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 600
MAX_EVIDENCE_CHUNKS = 8
MAX_CHUNK_CHARS = 1200
RATIONALE_MAX_CHARS = 1200
CONCURRENCY = 3
MAX_ATTEMPTS = 8
RETRY_BACKOFF_CAP_S = 60.0
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.IGNORECASE)

# Registry + lookup
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

def _parse_year(standard_long: str) -> int | None:
    m = _YEAR_RE.search(standard_long)
    return int(m.group(1)) if m else None

def load_registry() -> tuple[dict, dict]:
    """Đọc gri_units_with_sector_metadata.json, xây hai dict tra cứu:
    req_map (std, disc, req, year) → requirement_text và disc_map (std, disc) → tên disclosure.
    """
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        rows: list[dict] = json.load(f)

    req_map: dict[tuple, str] = {}
    disc_map: dict[tuple, str] = {}
    for r in rows:
        std = (r.get("standard_id") or "").strip()
        disc = (r.get("disclosure_id") or "").strip()
        req = (r.get("requirement_id") or "").strip()
        year = r.get("year")
        text = (r.get("requirement_text") or "").strip()
        # Bỏ qua row thiếu required fields — không raise vì registry có thể có row incomplete
        if not (std and disc and req and text):
            continue
        req_map[(std, disc, req, year)] = text
        # setdefault với year=None làm fallback khi caller không có thông tin năm
        req_map.setdefault((std, disc, req, None), text)
        d_name = (r.get("disclosure_name") or "").strip()
        if d_name:
            disc_map.setdefault((std, disc), d_name)
    return req_map, disc_map

def lookup_requirement(req_map: dict, standard_long: str, disclosure_id: str, requirement_id: str) -> str | None:
    """Tra cứu requirement text theo standard dài, ưu tiên match có year trước khi fallback year=None."""
    std_short = short_std_id(standard_long)
    year = _parse_year(standard_long)
    if year is not None:
        text = req_map.get((std_short, disclosure_id, requirement_id, year))
        if text:
            return text
    return req_map.get((std_short, disclosure_id, requirement_id, None))

def lookup_disclosure_name(disc_map: dict, standard_long: str, disclosure_id: str) -> str:
    """Tra cứu tên disclosure theo standard dài và disclosure_id; trả về chuỗi rỗng nếu không tìm thấy."""
    return disc_map.get((short_std_id(standard_long), disclosure_id), "")

# Evidence chunks
_chunk_cache: dict[str, dict[str, dict]] = {}

def load_chunks(report_id: str) -> dict[str, dict]:
    """Load report_chunks.json cho một report; có in-memory cache để tránh đọc lại khi cùng report_id được gọi nhiều lần."""
    if report_id in _chunk_cache:
        return _chunk_cache[report_id]
    path = REPORT_UNITS_DIR / report_id / "report_chunks.json"
    if not path.exists():
        # Cache kết quả rỗng để không thử đọc lại ở các call sau
        _chunk_cache[report_id] = {}
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows: list[dict] = json.load(f)
    out = {r["chunk_id"]: r for r in rows}
    _chunk_cache[report_id] = out
    return out

def _parse_citations(value: Any) -> list[str]:
    """Parse cột citations JSON từ DataFrame row; trả về list chunk_id hoặc [] nếu NaN/rỗng."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    v = value.strip()
    if not v:
        return []
    parsed = json.loads(v)
    return [str(x) for x in parsed]

def collect_evidence_chunks(row: dict, chunk_index: dict[str, dict]) -> list[dict]:
    """Thu thập và xếp hạng evidence chunks được cite bởi các variant, ưu tiên chunk được nhiều variant cite và được cite sớm."""
    # counts và first_seen để rank theo (frequency DESC, first_variant_idx ASC, chunk_id)
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for variant_idx, variant in enumerate(PRIMARY_VARIANTS):
        for c in _parse_citations(row.get(f"{variant}_citations_json")):
            counts[c] = counts.get(c, 0) + 1
            first_seen.setdefault(c, variant_idx)

    ordered = sorted(counts.keys(), key=lambda c: (-counts[c], first_seen.get(c, 99), c))

    out: list[dict] = []
    for cid in ordered[:MAX_EVIDENCE_CHUNKS]:
        chunk = chunk_index.get(cid)
        if not chunk:
            out.append({"chunk_id": cid, "content_text": "[chunk text missing]"})
            continue
        # Truncate text để tránh vượt MAX_CHUNK_CHARS và làm prompt quá dài
        text = (chunk.get("content_text") or "").strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rstrip() + " […truncated…]"
        out.append({
            "chunk_id": cid,
            "content_text": text,
            "page_start": chunk.get("page_start"),
            "section_label": chunk.get("section_label") or "",
        })
    return out

def _truncate(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " […]"

# Anonymization (deterministic SHA256 → System 1/2/3/4 layout)
def deterministic_permutation(seed_str: str) -> list[str]:
    """Tạo hoán vị cố định của PRIMARY_VARIANTS từ seed_str bằng SHA256 để anonymize thứ tự system."""
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    # Gán 4 byte làm sort key cho mỗi variant để hoán vị phụ thuộc duy nhất vào seed
    keyed = [(h[i * 4: i * 4 + 4], v) for i, v in enumerate(PRIMARY_VARIANTS)]
    keyed.sort(key=lambda kv: kv[0])
    return [v for _, v in keyed]

def build_anon_layout(row: dict) -> tuple[dict[str, int], dict[int, str]]:
    """Xây variant↔system mapping cho một case: variant_to_system (str→int) và system_to_variant (int→str)."""
    # Seed kết hợp toàn bộ join keys để mỗi case có hoán vị riêng, không đoán được thứ tự
    seed = "|".join([
        str(row["report_id"]), str(row["phase"]), str(row["standard_id"]),
        str(row["disclosure_id"]), str(row["material_topic"]),
        str(row["requirement_id"]), str(row["occurrence_idx"]),
    ])
    permuted = deterministic_permutation(seed)
    variant_to_system = {v: i + 1 for i, v in enumerate(permuted)}
    system_to_variant = {i + 1: v for i, v in enumerate(permuted)}
    return variant_to_system, system_to_variant

# LLM call: JSON parsing + validation
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_arbiter_json(raw: str) -> dict:
    """Parse JSON từ LLM response; strip markdown fences nếu có, fallback extract {...} nếu json.loads thất bại."""
    if not raw:
        raise ValueError("empty response")
    text = raw.strip()
    # Strip markdown code fences (```json ... ```) mà LLM đôi khi bọc response
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: tìm {...} đầu tiên trong text khi LLM thêm prefix/suffix không phải JSON
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))

def validate_arbiter_payload(payload: Any) -> tuple[str, list[int], str]:
    """Validate schema dict từ LLM: correct_status ∈ VALID_STATUSES, systems_correct là list[int] trong 1–4, rationale không rỗng."""
    if not isinstance(payload, dict):
        raise ValueError(f"payload is not an object: {type(payload).__name__}")
    status = str(payload.get("correct_status", "")).strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid correct_status: {status!r}")
    sys_correct_raw = payload.get("systems_correct", [])
    if not isinstance(sys_correct_raw, list):
        raise ValueError("systems_correct must be a list")
    # Dedup nhưng vẫn validate từng phần tử để bắt lỗi LLM dùng sai kiểu
    sys_correct: list[int] = []
    for x in sys_correct_raw:
        try:
            n = int(x)
        except (TypeError, ValueError):
            raise ValueError(f"systems_correct contains non-int: {x!r}")
        if n not in (1, 2, 3, 4):
            raise ValueError(f"systems_correct contains out-of-range: {n}")
        if n not in sys_correct:
            sys_correct.append(n)
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("rationale is empty")
    return status, sys_correct, rationale

# LLM call: retry với exponential backoff + jitter
async def _call_openai_once(client, prompt: str, system_msg: str) -> tuple[str, list[int], str, str]:
    """Gọi OpenAI một lần (không retry): parse response → validate schema; raise ngay nếu bất kỳ bước nào fail."""
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    payload = parse_arbiter_json(raw)
    status, sys_correct, rationale = validate_arbiter_payload(payload)
    # canonical_raw chuẩn hoá schema về 3 key cố định để cache key luôn nhất quán
    canonical_raw = json.dumps(
        {"correct_status": status, "systems_correct": sys_correct, "rationale": rationale},
        ensure_ascii=False,
    )
    return status, sys_correct, rationale, canonical_raw

def _extract_retry_after_seconds(exc: Exception) -> float | None:
    """Đọc gợi ý chờ từ 429: ưu tiên header `Retry-After`, fallback parse message."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                try:
                    return float(raw)
                except ValueError:
                    pass
    m = _RETRY_AFTER_RE.search(str(exc))
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    return value / 1000.0 if unit == "ms" else value

async def call_with_retry(client, prompt: str, system_msg: str) -> tuple[str, list[int], str, str]:
    """Retry _call_openai_once với exponential backoff + jitter cho các lỗi transient.
    Lỗi validation (JSON malformed, schema sai) raise ngay vì temperature=0 → retry không đổi kết quả.
    """
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    transient = (RateLimitError, APITimeoutError, APIConnectionError)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await _call_openai_once(client, prompt, system_msg)
        except transient as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            hint = _extract_retry_after_seconds(exc)
            # Exponential backoff: 2^(attempt-1)s, cap RETRY_BACKOFF_CAP_S
            # Cộng jitter random [0,1) để tránh nhiều worker cùng wake một lúc (thundering herd)
            backoff = min(2 ** (attempt - 1), RETRY_BACKOFF_CAP_S) + random.random()
            delay = max(hint if hint is not None else 0.0, backoff)
            print(f"  Retry {attempt}/{MAX_ATTEMPTS} after {delay:.1f}s ({type(exc).__name__}: {str(exc)[:120]})")
            await asyncio.sleep(delay)

# Cache I/O (JSONL: 1 row = 1 cache entry, key = prompt_sha256)
def load_cache(path: Path) -> dict[str, dict]:
    """Load cache JSONL; skip line bị lỗi JSON để không bị crash khi async appender ghi dở."""
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = row.get("prompt_sha256")
            if k:
                cache[k] = row
    return cache

def append_cache(path: Path, row: dict) -> None:
    """Thêm một cache entry vào JSONL file; tạo thư mục nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# Worker chính: cache check → call_with_retry → cache write → build result row
def _build_result_row(
    req: dict,
    status: str,
    sys_correct: list[int],
    rationale: str,
    sha: str,
    cache_hit: bool,
    error: str = "",
) -> dict:
    """Xây result dict từ một adjudication, map system_number → variant để điền {v}_correct."""
    # var_to_sys ánh xạ variant → system number theo hoán vị của case này
    var_to_sys = req["variant_to_system"]
    return {
        "report_id": req["report_id"],
        "phase": req["phase"],
        "standard_id": req["standard_id"],
        "disclosure_id": req["disclosure_id"],
        "material_topic": req["material_topic"],
        "requirement_id": req["requirement_id"],
        "occurrence_idx": req["occurrence_idx"],
        "correct_status": status,
        "systems_correct_json": json.dumps(sys_correct),
        "a0_correct": var_to_sys["a0"] in sys_correct,
        "a1_correct": var_to_sys["a1"] in sys_correct,
        "a2_correct": var_to_sys["a2"] in sys_correct,
        "v_new_correct": var_to_sys["v_new"] in sys_correct,
        "n_systems_correct": len(sys_correct),
        "arbiter_rationale": rationale,
        "n_evidence_chunks": len(req["evidence_chunks"]),
        "prompt_sha256": sha,
        "cache_hit": cache_hit,
        "error": error,
    }

async def adjudicate_one(
    req: dict,
    prompt: str,
    client,
    cache: dict[str, dict],
    cache_path: Path,
    system_msg: str,
    sem: asyncio.Semaphore,
    counters: dict[str, int],
) -> dict:
    """Worker chính cho một disagreement row: cache check → LLM call → cache write → trả về result row."""
    sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # Bước 1: Cache hit — validate lại payload để bắt cache row bị corrupt từ run trước
    if sha in cache:
        try:
            payload = json.loads(cache[sha]["raw_response"])
            status, sys_correct, rationale = validate_arbiter_payload(payload)
            counters["cache_hits"] += 1
            return _build_result_row(req, status, sys_correct, rationale, sha, cache_hit=True)
        except Exception as exc:
            print(f"  Cache row {sha[:8]} invalid ({exc}); re-calling")

    # Bước 2: LLM call trong semaphore để giới hạn concurrency; ghi cache ngay sau khi thành công
    async with sem:
        try:
            status, sys_correct, rationale, canonical_raw = await call_with_retry(
                client, prompt, system_msg,
            )
        except Exception as exc:
            counters["errors"] += 1
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  Row {req['row_idx']} failed after retries: {err_msg}")
            return _build_result_row(req, "", [], "", sha, cache_hit=False, error=err_msg)

        counters["live_calls"] += 1
        cache_row = {
            "prompt_sha256": sha,
            "raw_response": canonical_raw,
            "model": MODEL,
            "ts": time.time(),
        }
        cache[sha] = cache_row
        append_cache(cache_path, cache_row)
        return _build_result_row(req, status, sys_correct, rationale, sha, cache_hit=False)
