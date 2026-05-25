import os
from pathlib import Path

# LLM / Ollama 
OLLAMA_LLM_MODEL = "qwen3:14b"
OLLAMA_KEEP_ALIVE = -1  # Giữ model trong VRAM
OLLAMA_TEMPERATURE = 0.0
OLLAMA_NUM_PREDICT = 2500
OLLAMA_NUM_CTX = 16384
OLLAMA_REQUEST_TIMEOUT_S = 180       # per-call hard timeout
OLLAMA_RETRY_ATTEMPTS = 2            # khi timeout / connection error
OLLAMA_RETRY_BACKOFF_S = (1, 4)      # exponential backoff

ENABLE_THINKING_JUDGE = True
ENABLE_THINKING_ANCHOR = True
ENABLE_THINKING_MATCHER = True

OLLAMA_NUM_PREDICT_ANCHOR = 2000
OLLAMA_NUM_PREDICT_MATCHER = 1500

# Prompt versions
PROMPT_VERSION_EVIDENCE = "v3.1"
PROMPT_VERSION_ANCHOR = "v2.1"

# Vector DB — Zilliz
ZILLIZ_URI = os.getenv("ZILLIZ_CLOUD_URI", "")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_CLOUD_API_KEY", "")

COLLECTION_REPORT_CHUNKS = "report_chunks"

# NLI
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_REORDER_ENABLED = os.environ.get("NLI_REORDER_ENABLED", "1") != "0"
NLI_HINT_ENABLED = os.environ.get("NLI_HINT_ENABLED", "1") != "0"

# Verdict post-processing 
MAX_CITATIONS_PER_REQUIREMENT = 5
EVIDENCE_BLOCK_CHAR_BUDGET = 20000

# Whitelist disclosure_id được inject ScopeAnchor vào prompt
ANCHOR_INJECTION_WHITELIST = {"2-7", "2-8", "2-21", "2-27", "2-30"}

# Disclosure-batch judge 
BATCH_JUDGE_MAX_REQUIREMENTS = 6

# Phase 4 — Materiality matching 
FUZZY_THRESHOLD = 90
LLM_MATCHER_MIN_CONFIDENCE = 0.6
FUZZY_LOW_SKIP_LLM = 50

SUPPORTED_SECTOR_STANDARDS = {"GRI 11: Oil and Gas Sector 2021"}
GRI_11_CANONICAL = "GRI 11: Oil and Gas Sector 2021"

# Compliance rules 
VALID_OMISSION_REASONS = (
    "Not applicable",
    "Legal prohibitions",
    "Confidentiality constraints",
    "Information unavailable/incomplete",
)
GRI2_HARD_FAIL_OMIT_BLOCKED = {"2-1", "2-2", "2-3", "2-4", "2-5"}
GRI3_HARD_FAIL_OMIT_BLOCKED = {"3-1", "3-2"}
OMIT_BLOCKED = GRI2_HARD_FAIL_OMIT_BLOCKED | GRI3_HARD_FAIL_OMIT_BLOCKED

REPORTING_PRINCIPLES = (
    "Accuracy",
    "Balance",
    "Clarity",
    "Comparability",
    "Completeness",
    "Sustainability context",
    "Timeliness",
    "Verifiability",
)

# Output schema
OUTPUT_VERSION = "module3.v2"

# Filesystem layout 
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_UNITS_DIR = str(REPO_ROOT / "metadata" / "report_units")
GRI_UNITS_JSON = str(
    REPO_ROOT / "metadata" / "gri_units" / "gri_units_with_sector_metadata.json"
)
GRI11_SECTOR_JSON = str(
    REPO_ROOT / "metadata" / "sector_standards" / "gri_11" / "gri_11_sector_metadata.json"
)

JUDGE_CACHE_FILENAME = "_judge_cache.jsonl"
COMPLIANCE_REPORT_FILENAME = "compliance_report.json"
COMPLIANCE_SUMMARY_FILENAME = "compliance_summary.csv"

OUTPUT_DIR = str(REPO_ROOT / "output")
PHASE_EXPORT_FILENAMES = {
    "phase3": "gri2_results.json",
    "phase4": "material_topics_results.json",
    "phase5": "gri3_results.json",
    "phase6": "topic_standards_results.json",
    "phase7": "omissions_results.json",
    "phase8": "content_index_results.json",
    "phase8_light": "content_index_results.json",
}
