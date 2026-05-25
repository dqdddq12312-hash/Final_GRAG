import asyncio
import json
import logging
import re
from functools import lru_cache

from . import config

logger = logging.getLogger(__name__)

# LLM factory 
def _build_chat_ollama(num_predict, enable_thinking):
    """Build ChatOllama instance — lazy import để test mock được."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_LLM_MODEL,
        temperature=config.OLLAMA_TEMPERATURE,
        num_predict=num_predict,
        num_ctx=config.OLLAMA_NUM_CTX,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        timeout=config.OLLAMA_REQUEST_TIMEOUT_S,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

@lru_cache(maxsize=1)
def make_judge_llm():
    """ChatOllama cho EvidenceJudge."""
    return _build_chat_ollama(
        num_predict=config.OLLAMA_NUM_PREDICT,
        enable_thinking=config.ENABLE_THINKING_JUDGE,
    )

@lru_cache(maxsize=1)
def make_anchor_llm():
    """ChatOllama cho ScopeAnchor."""
    return _build_chat_ollama(
        num_predict=config.OLLAMA_NUM_PREDICT_ANCHOR,
        enable_thinking=config.ENABLE_THINKING_ANCHOR,
    )

@lru_cache(maxsize=1)
def make_matcher_llm():
    """ChatOllama cho TopicMatcher."""
    return _build_chat_ollama(
        num_predict=config.OLLAMA_NUM_PREDICT_MATCHER,
        enable_thinking=config.ENABLE_THINKING_MATCHER,
    )

# Xử lý LLM output 
# qwen3 thinking-mode bọc output trong <think>...</think> trước JSON thực
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)

def _strip_json_fence(text):
    m = _FENCE_RE.match(text or "")
    return m.group(1) if m else (text or "")

def _strip_think_block(text):
    result = _THINK_RE.sub("", text or "").strip()
    # Bắt trường hợp tag mở không đóng (truncated response)
    m = _THINK_OPEN_RE.search(result)
    if m:
        result = result[: m.start()].strip()
    return result

def _extract_first_json(text):
    """Tách JSON object/array đầu tiên trong text (bỏ qua prose trước/sau)."""
    s = (text or "").strip()
    if not s:
        return s
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(s[i:])
                return s[i : i + end]
            except json.JSONDecodeError:
                continue
    return s

def _parse_llm_json(content, schema):
    """Parse JSON từ LLM response: strip think block → strip fence → extract → validate."""
    text = "" if content is None else str(content)
    text = _strip_think_block(text)
    text = _strip_json_fence(text)
    text = _extract_first_json(text)
    payload = json.loads(text)
    return schema.model_validate(payload)

# Retry + re-prompt 
class LLMOutputUnparseableError(RuntimeError):
    """Raise khi LLM trả output unparseable cả lần đầu lẫn sau re-prompt."""

    def __init__(self, schema_name, content_preview, underlying):
        self.schema_name = schema_name
        self.content_preview = content_preview
        self.underlying = underlying
        super().__init__(
            f"LLM output unparseable for {schema_name} after re-prompt "
            f"({type(underlying).__name__}: {str(underlying)[:120]}); "
            f"content_preview={content_preview!r}"
        )

def _preview(content, n=200):
    """Single-line, length-capped preview cho log."""
    s = "" if content is None else str(content)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s[:n] if len(s) <= n else s[:n] + "…"

def _classify_transient(exc):
    """True nếu exc đáng retry (timeout / network error)."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    name = type(exc).__name__
    if name in {"ConnectError", "ReadError", "ReadTimeout", "ConnectTimeout"}:
        return True
    return False

def _build_reprompt(messages):
    """Prefix messages với nudge "valid JSON only", giữ nguyên kiểu input."""
    prefix = (
        "The previous output was malformed; emit valid JSON only "
        "matching the requested schema. No prose, no code fences.\n\n"
    )
    if isinstance(messages, str):
        return prefix + messages
    if isinstance(messages, list):
        try:
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=prefix.strip())] + list(messages)
        except ImportError:
            return [{"role": "system", "content": prefix.strip()}] + list(messages)
    return messages

async def _call_llm_with_retry(
    llm,
    messages,
    *,
    schema=None,
    max_attempts=config.OLLAMA_RETRY_ATTEMPTS,
    backoff_s=config.OLLAMA_RETRY_BACKOFF_S,
):
    """Gọi LLM với retry khi gặp transient error và re-prompt khi JSON invalid.

    Flow:
    1. Gọi `llm.ainvoke(messages)`.
    2. Nếu `schema` được cung cấp, parse JSON và validate qua Pydantic.
    3. Nếu parse/validate fail → re-prompt 1 lần với nudge "valid JSON only".
    4. Nếu re-prompt cũng fail → raise `LLMOutputUnparseableError`.
    5. Nếu transient error (timeout/network) → exponential backoff và retry
       tối đa `max_attempts` lần.
    """
    from pydantic import ValidationError

    attempt = 0
    while True:
        try:
            response = await llm.ainvoke(messages)
            if schema is None:
                return response
            content = getattr(response, "content", response)
            try:
                return _parse_llm_json(content, schema)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    "LLM produced invalid JSON for schema %s: %r "
                    "(content_preview=%r) — re-prompting once",
                    schema.__name__, str(e)[:200], _preview(content),
                )
                response2 = await llm.ainvoke(_build_reprompt(messages))
                content2 = getattr(response2, "content", response2)
                try:
                    return _parse_llm_json(content2, schema)
                except (json.JSONDecodeError, ValidationError) as e2:
                    preview2 = _preview(content2)
                    logger.error(
                        "LLM re-prompt also produced invalid JSON for schema %s: "
                        "%r (content_preview=%r) — giving up",
                        schema.__name__, str(e2)[:200], preview2,
                    )
                    raise LLMOutputUnparseableError(
                        schema_name=schema.__name__,
                        content_preview=preview2,
                        underlying=e2,
                    ) from e2
        except BaseException as e:
            if not _classify_transient(e) or attempt >= max_attempts:
                raise
            sleep_s = backoff_s[min(attempt, len(backoff_s) - 1)]
            logger.warning(
                "LLM transient error %s — retrying in %ss (attempt %d/%d)",
                type(e).__name__, sleep_s, attempt + 1, max_attempts,
            )
            await asyncio.sleep(sleep_s)
            attempt += 1
            continue

__all__ = [
    "make_judge_llm",
    "make_anchor_llm",
    "make_matcher_llm",
    "_call_llm_with_retry",
    "_strip_json_fence",
    "LLMOutputUnparseableError",
]
