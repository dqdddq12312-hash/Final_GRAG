from pathlib import Path

# ROOT
EXP_DIR  = Path(__file__).parent.parent
DATA_DIR = EXP_DIR / "data"
OUT_DIR  = EXP_DIR / "outputs"

TABLES_DIR  = OUT_DIR / "tables"
STATS_DIR   = OUT_DIR / "stats"
FIGURES_DIR = OUT_DIR / "figures"

VARIANT_RUNS_DIR = EXP_DIR / "variant_runs"

# Data subdirectories
CONFIG_DIR   = DATA_DIR / "00_config"
PAIRING_DIR  = DATA_DIR / "01_pairing_postfix"
ARBITER_DIR  = DATA_DIR / "02_arbiter_postfix"
HUMAN_T1_DIR = DATA_DIR / "03_human_T1_postfix_120"
HUMAN_T3_DIR = DATA_DIR / "04_human_T3_GTv2_150"
LATENCY_DIR  = DATA_DIR / "05_latency"
PRE_FIX_DIR  = DATA_DIR / "90_pilot_prefix"   # pilot pre-fix archive

# File constants
REPORTS_MAIN_TXT      = CONFIG_DIR   / "reports_main.txt"

PAIRING_CSV           = PAIRING_DIR  / "pairing.csv"
DISAGREEMENTS_CSV     = PAIRING_DIR  / "disagreements.csv"
VERDICTS_LONG_CSV     = PAIRING_DIR  / "verdicts_long.csv"
DISCLOSURE_INDEX_CSV  = PAIRING_DIR  / "disclosure_index.csv"

HUMAN_REVIEW_DONE_CSV = HUMAN_T1_DIR / "human_review_completed.csv"
HUMAN_TEMPLATE_CSV    = HUMAN_T1_DIR / "human_review_template.csv"
HUMAN_TEMPLATE_MD     = HUMAN_T1_DIR / "human_review_template.md"

HUMAN_REVIEW_V2_CSV   = HUMAN_T3_DIR / "human_review_v2.csv"
SAMPLE_MAIN_CSV       = HUMAN_T3_DIR / "sample_main.csv"

LATENCY_OBSERVED_CSV  = LATENCY_DIR  / "latency_per_variant_observed.csv"

PRE_FIX_PAIRING_CSV   = PRE_FIX_DIR  / "pairing_pre_topic_fix.csv"
PRE_FIX_HUMAN_CSV     = PRE_FIX_DIR  / "human_review_prefix.csv"
PRE_FIX_ADJ_V2_CSV    = PRE_FIX_DIR  / "adjudicated_v2_prefix.csv"

def adjudicated_csv(version: str) -> Path:
    """Path đến adjudicated_{version}.csv trong 02_arbiter_postfix/."""
    return ARBITER_DIR / f"adjudicated_{version}.csv"

def arbiter_cache_jsonl(version: str) -> Path:
    """Path đến arbiter_cache_{version}.jsonl trong 02_arbiter_postfix/."""
    return ARBITER_DIR / f"arbiter_cache_{version}.jsonl"

def arbiter_stats_json(version: str) -> Path:
    """Path đến arbiter_stats_{version}.json trong 02_arbiter_postfix/."""
    return ARBITER_DIR / f"arbiter_stats_{version}.json"

# Variants
VARIANTS = ["a0", "a1", "a2", "v_new"]
VARIANT_LABELS = {
    "a0":    "A0 (no NLI)",
    "a1":    "A1 (REORDER only)",
    "a2":    "A2 (HINT only)",
    "v_new": "V_new (REORDER + HINT)",
}

# Status / Join keys
STATUS_LABELS = ["pass", "partial", "no_evidence"]

# Canonical join keys
ARBITER_JOIN_KEYS = [
    "report_id", "phase", "standard_id", "disclosure_id",
    "material_topic", "requirement_id", "occurrence_idx",
]
MATERIAL_TOPIC_NA_SENTINEL = "__none__"

# Phase buckets
PHASE_BUCKETS = ["ALL", "phase3", "phase5_3-3", "phase6"]
