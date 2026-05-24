"""TN3 / TN4 — Module 3 NLI factorial library (package).

Một thư viện chung cho 4 notebook phân tích:
  - tn3_prefix_factorial.ipynb     — TN3 §5.3.1 + Phụ lục D (pre-fix factorial, pilot)
  - tn3_postfix_factorial.ipynb    — TN3 §5.3.1 (post-fix factorial, T2')
  - tn3_postfix_bugfix.ipynb       — TN3 §5.3.2-5.3.8 (post-fix bug-fix lift, T3)
  - tn4_groundtruth.ipynb          — TN4 §5.4 (arbiter calibration, GT methodology)

Package layout (one submodule per section):
  constants       — paths, VARIANTS, STATUS_LABELS, ARBITER_JOIN_KEYS, PHASE_BUCKETS
  _utils          — private normalisation / iteration helpers
  loading         — load_reports_main, load_postfix_sample (pure I/O)
  ground_truth    — §2 Ground Truth Construction
  stats_tests     — §3 Statistical Tests
  variant_prefix  — §4a Variant Analyses (TN3 factorial)
  variant_postfix — §4b Variant Analyses (TN3 post-fix bug-fix lift)
  latency         — §5 Latency Analysis
  arbiter         — §6 Arbiter / GT Quality (TN4)

Thin CSV loaders (load_pairing, load_adjudicated, load_human_review_*) đã được
inline trực tiếp vào từng notebook dưới dạng pd.read_csv(...) để đơn giản hóa.
"""
from __future__ import annotations

# ── constants (exported via wildcard) ────────────────────────────────────────
from .constants import (
    EXP_DIR,
    DATA_DIR,
    PRE_FIX_DIR,
    OUT_DIR,
    TABLES_DIR,
    STATS_DIR,
    FIGURES_DIR,
    VARIANT_RUNS_DIR,
    VARIANTS,
    VARIANT_LABELS,
    STATUS_LABELS,
    ARBITER_JOIN_KEYS,
    MATERIAL_TOPIC_NA_SENTINEL,
    PHASE_BUCKETS,
)

# ── private helpers (re-exported so notebooks can use lib._xxx) ───────────────
from ._utils import (
    _normalize_material_topic_inplace,
    _normalize_status_inplace,
    _status_col,
    _iter_buckets,
)

# ── §1 Data Loading ───────────────────────────────────────────────────────────
from .loading import (
    load_reports_main,
    load_postfix_sample,
)

# ── §2 Ground Truth Construction ─────────────────────────────────────────────
from .ground_truth import (
    build_hybrid_gt,
    apply_gt_to_pairing,
    derive_correct_flags,
    filter_universe,
)

# ── §3 Statistical Tests ──────────────────────────────────────────────────────
from .stats_tests import (
    mcnemar_exact,
    cohen_kappa,
    confusion_matrix,
    cluster_bootstrap_per_report_diff,
    power_mde_paired,
    _bootstrap_ci_simple,
)

# ── §4a Variant Analyses — pre-fix factorial ──────────────────────────────────
from .variant_prefix import (
    per_variant_marginals,
    pairwise_kappa_variants,
    e1_metrics,
    e2b_factorial,
    per_phase_breakdown,
)

# ── §4b Variant Analyses — post-fix bug-fix lift ──────────────────────────────
from .variant_postfix import (
    build_accuracy_table_postfix,
    build_lift_table_postfix,
    build_mcnemar_table_postfix,
    hint_over_prediction_summary,
    kappa_pre_post_panel,
)

# ── §5 Latency Analysis ───────────────────────────────────────────────────────
from .latency import latency_per_variant

# ── §6 Arbiter / GT Quality ───────────────────────────────────────────────────
from .arbiter import (
    merge_human_with_arbiter,
    arbiter_vs_human_kappa,
    arbiter_marginal_distribution,
    arbiter_kappa_panel,
    sign_reversal_e1,
    _interpret_kappa,
)

# ── Public API (matches original nli_lib.py __all__) ─────────────────────────
__all__ = [
    # Paths
    "EXP_DIR", "DATA_DIR", "PRE_FIX_DIR", "OUT_DIR", "TABLES_DIR", "STATS_DIR", "FIGURES_DIR",
    "VARIANTS", "VARIANT_LABELS", "STATUS_LABELS",
    # §1 Data Loading
    "load_reports_main",
    # §2 GT Construction
    "build_hybrid_gt", "apply_gt_to_pairing", "derive_correct_flags", "filter_universe",
    # §3 Statistical Tests
    "mcnemar_exact", "cohen_kappa", "confusion_matrix",
    "cluster_bootstrap_per_report_diff", "power_mde_paired",
    # §4 Variant Analyses
    "per_variant_marginals", "pairwise_kappa_variants",
    "e1_metrics", "e2b_factorial",
    "per_phase_breakdown",
    "load_postfix_sample", "build_accuracy_table_postfix",
    "build_lift_table_postfix", "build_mcnemar_table_postfix",
    "hint_over_prediction_summary", "kappa_pre_post_panel",
    # §5 Latency Analysis
    "latency_per_variant",
    # §6 Arbiter / GT Quality
    "merge_human_with_arbiter", "arbiter_vs_human_kappa",
    "arbiter_marginal_distribution", "arbiter_kappa_panel",
    "sign_reversal_e1",
    # constants / helpers
    "ARBITER_JOIN_KEYS", "MATERIAL_TOPIC_NA_SENTINEL", "_interpret_kappa",
]
