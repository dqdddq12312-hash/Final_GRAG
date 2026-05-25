import asyncio
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel

from . import config


# Hash helpers 
def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def evidence_text_hash(chunks):
    """Hash ổn định của nội dung evidence pack.

    Sort theo `chunk_id` để order-insensitive: cùng evidence list reorder
    phải cho cùng hash.
    """
    parts = []
    for c in sorted(chunks, key=lambda x: x.get("chunk_id", "")):
        parts.append(c.get("chunk_id", ""))
        parts.append(c.get("content_text", ""))
    return _sha256_text("\u241e".join(parts))  # \u241e = unit separator

def anchor_hash(anchor_dump):
    """Hash ổn định của ScopeAnchor dump. Trả None nếu anchor_dump là None."""
    if anchor_dump is None:
        return None
    return _sha256_text(json.dumps(anchor_dump, sort_keys=True, default=str))

def requirement_text_hash(reqs):
    """Hash ổn định của requirement texts trong batch.

    Sort theo `requirement_id` để reorder không đổi hash. Cho phép invalidate
    đúng entry khi text 1 requirement thay đổi mà không phải bump global
    prompt version.

    Mỗi entry trong `reqs` cần có `requirement_id` + `requirement_text`
    (hoặc `text`); thiếu coi như "".
    """
    parts = []
    for r in sorted(reqs, key=lambda x: str(x.get("requirement_id") or "")):
        req_id = str(r.get("requirement_id") or "")
        req_text = str(r.get("requirement_text") or r.get("text") or "")
        parts.append(f"{req_id}\u241f{req_text}")
    return _sha256_text("\u241e".join(parts))

def judge_cache_key(
    *,
    model,
    prompt_version,
    requirement_id,
    chunk_ids_sorted,
    evidence_text_hash,
    anchor_hash=None,
    requirement_text_hash=None,
    material_topic=None,
):
    """Key SHA256 64-char canonical cho cache.

    Tính deterministic là toàn bộ contract — không `time.time()`, không
    `id()`, không randomness per-process. Cùng input → cùng key.

    `requirement_text_hash`: khi cung cấp, hash của text requirement được
    fold vào key để chỉnh registry text chỉ invalidate entry liên quan,
    không phải bump global PROMPT_VERSION_EVIDENCE.

    `material_topic`: optional per-topic scope binder cho GRI 3-3. Khi là
    None, field bị OMIT khỏi payload — cache hit cho các disclosure không
    gắn topic (3-1, 3-2, mọi phase-6 disclosure) vẫn hợp lệ. Khi cung cấp,
    field được fold vào payload — mỗi topic sinh 1 cache entry riêng ngay
    cả khi chung page_list.
    """
    payload_dict = {
        "model": model,
        "prompt_version": prompt_version,
        "requirement_id": requirement_id,
        "chunk_ids_sorted": list(chunk_ids_sorted),
        "evidence_text_hash": evidence_text_hash,
        "anchor_hash": anchor_hash,
        "requirement_text_hash": requirement_text_hash,
    }
    if material_topic:
        # Chỉ thêm khi truthy → giữ cache hit cho disclosure topic-agnostic
        payload_dict["material_topic"] = material_topic
    payload = json.dumps(
        payload_dict,
        sort_keys=True,
        ensure_ascii=False,
    )
    return _sha256_text(payload)

# === JudgeCache ===
class JudgeCache:
    """Cache verdict per-report, append-only.

    Ghi atomic (1 dòng JSON / entry); đọc là dict lookup sau khi `load()`
    populate `_mem`. `get_or_compute` serialize per-key dưới `asyncio.Lock`
    để 2 fan-out đồng thời không cùng compute() 1 key.
    """

    def __init__(self, report_id):
        self.report_id = report_id
        self.path = (
            Path(config.REPORT_UNITS_DIR)
            / report_id
            / config.JUDGE_CACHE_FILENAME
        )
        self._mem = {}
        self._locks = {}
        self._loaded = False

    # File I/O

    def load(self):
        """Idempotent load JSONL vào `_mem`. An toàn gọi lặp."""
        if self._loaded:
            return self._mem
        self._mem = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = row.get("key")
                    verdict = row.get("verdict")
                    if key and isinstance(verdict, dict):
                        # Last write wins khi có duplicate key
                        self._mem[key] = verdict
        self._loaded = True
        return self._mem

    def save_entry(self, key, verdict):
        """Append 1 entry vào JSONL và update `_mem`."""
        if isinstance(verdict, BaseModel):
            verdict_payload = verdict.model_dump()
        else:
            verdict_payload = dict(verdict)
        row = {"key": key, "verdict": verdict_payload}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        except OSError:
            pass
        self._mem[key] = verdict_payload

    def clear(self):
        """Xóa cache cho report này — notebook cell 0.7 dùng."""
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        self._mem = {}
        self._locks = {}
        self._loaded = True

    # Async access
    def _lock_for(self, key):
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def get_or_compute(self, key, compute, schema):
        """Atomic per-key get-or-compute.

        Hit → trả `schema.model_validate(...)` của dict cached với
        `source="cache"` được inject (KHÔNG mutate dict đã persist).
        Miss → chạy `compute()`, validate, persist, trả.
        """
        if not self._loaded:
            self.load()

        # Fast path: đã trong memory
        cached = self._mem.get(key)
        if cached is not None:
            return self._materialize_cached(cached, schema)

        async with self._lock_for(key):
            # Double-check: task khác có thể đã populate trong khi ta đợi lock
            cached = self._mem.get(key)
            if cached is not None:
                return self._materialize_cached(cached, schema)

            verdict = await compute()
            if not isinstance(verdict, schema):
                verdict = schema.model_validate(verdict)
            self.save_entry(key, verdict)
            return verdict

    @staticmethod
    def _materialize_cached(cached_dump, schema):
        """Re-hydrate dict đã persist về schema, mark source='cache'.

        Source tag chỉ apply cho payload kiểu RequirementVerdict (top-level
        `source` field) và cho từng child verdict trong DisclosureBatchVerdict
        (`requirement_verdicts[i].source`). Tag mutate trên copy, không động
        record persist.
        """
        copy = dict(cached_dump)
        if "source" in copy and isinstance(copy.get("source"), str):
            copy["source"] = "cache"
        rvs = copy.get("requirement_verdicts")
        if isinstance(rvs, list):
            new_rvs = []
            for rv in rvs:
                if isinstance(rv, dict):
                    rv2 = dict(rv)
                    if "source" in rv2:
                        rv2["source"] = "cache"
                    new_rvs.append(rv2)
                else:
                    new_rvs.append(rv)
            copy["requirement_verdicts"] = new_rvs
        return schema.model_validate(copy)


__all__ = [
    "judge_cache_key",
    "evidence_text_hash",
    "anchor_hash",
    "requirement_text_hash",
    "JudgeCache",
]
