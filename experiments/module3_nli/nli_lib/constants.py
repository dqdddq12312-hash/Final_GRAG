"""Module-wide constants: paths, variant labels, status labels, join keys."""
from __future__ import annotations

from pathlib import Path

# PATHS 
EXP_DIR  = Path(__file__).parent.parent
DATA_DIR = EXP_DIR / "data"
PRE_FIX_DIR = DATA_DIR / "pre_fix"  # snapshot of pre-topic-fix artifacts (TN3 pilot)
OUT_DIR  = EXP_DIR / "outputs"
TABLES_DIR  = OUT_DIR / "tables"
STATS_DIR   = OUT_DIR / "stats"
FIGURES_DIR = OUT_DIR / "figures"

VARIANT_RUNS_DIR = EXP_DIR / "variant_runs"

# VARIANTS
VARIANTS = ["a0", "a1", "a2", "v_new"]
VARIANT_LABELS = {
    "a0":    "A0 (no NLI)",
    "a1":    "A1 (REORDER only)",
    "a2":    "A2 (HINT only)",
    "v_new": "V_new (REORDER + HINT)",
}

# STATUS / JOIN KEYS
STATUS_LABELS = ["pass", "partial", "no_evidence"]

# Canonical join keys (matches legacy compute_kappa_v2.py / build_pairing.py)
ARBITER_JOIN_KEYS = [
    "report_id", "phase", "standard_id", "disclosure_id",
    "material_topic", "requirement_id", "occurrence_idx",
]
MATERIAL_TOPIC_NA_SENTINEL = "__none__"

# Phase buckets used by post-fix accuracy / lift / kappa-panel builders
PHASE_BUCKETS = ["ALL", "phase3", "phase5_3-3", "phase6"]
