"""Verify Đợt A — kết quả không đổi so với artifact đã lưu."""
import json
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pt

sys.path.insert(0, str(Path(__file__).parent))
import nli_lib as lib

PASS = "[PASS]"
FAIL = "[FAIL]"
all_ok = True


def check_df(name, actual, csv_path):
    global all_ok
    saved = pd.read_csv(csv_path)
    try:
        pt.assert_frame_equal(actual.reset_index(drop=True), saved.reset_index(drop=True))
        print(f"{PASS} {name}")
    except AssertionError as e:
        print(f"{FAIL} {name}: {e}")
        all_ok = False


def check_json(name, actual_dict, json_path):
    global all_ok
    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)
    if actual_dict == saved:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}: {actual_dict} != {saved}")
        all_ok = False


# ── TN4 ──────────────────────────────────────────────────────────────────────
print("=== TN4 ===")
human = pd.read_csv(lib.DATA_DIR / "human_review_completed.csv", low_memory=False)
adj_v1 = pd.read_csv(lib.DATA_DIR / "adjudicated_v1.csv", low_memory=False)
adj_v2 = pd.read_csv(lib.DATA_DIR / "adjudicated_v2.csv", low_memory=False)
adj_v3 = pd.read_csv(lib.DATA_DIR / "adjudicated_v3.csv", low_memory=False)
pairing_post = pd.read_csv(lib.DATA_DIR / "pairing.csv", low_memory=False)

kappa_panel = lib.arbiter_kappa_panel(human, adj_v1, adj_v2, adj_v3)
marg = lib.arbiter_marginal_distribution(human, adj_v1, adj_v2, adj_v3)
r3 = lib.arbiter_vs_human_kappa(human, adj_v3, arbiter_label="v3")
sr = lib.sign_reversal_e1(pairing_post, adj_v1, adj_v2)

check_df("kappa_panel", kappa_panel, lib.TABLES_DIR / "T_TN4_kappa_panel.csv")
check_df("marginal_distribution", marg, lib.TABLES_DIR / "T_TN4_marginal_distribution.csv")
check_json("sign_reversal", sr, lib.STATS_DIR / "TN4_sign_reversal.json")

with (lib.STATS_DIR / "TN4_v3_failed.json").open() as f:
    saved_v3 = json.load(f)
ok_v3 = r3["n"] == saved_v3["n"] and abs(r3["cohen_kappa"] - saved_v3["cohen_kappa"]) < 1e-6
print(f"{PASS if ok_v3 else FAIL} arbiter_vs_human_kappa(v3)")
if not ok_v3:
    all_ok = False

# ── TN3 post-fix factorial ────────────────────────────────────────────────────
print("\n=== TN3 post-fix factorial ===")
human_post = pd.read_csv(lib.DATA_DIR / "human_review_completed.csv", low_memory=False)
adj_v2_post = pd.read_csv(lib.DATA_DIR / "adjudicated_v2.csv", low_memory=False)
gt_hybrid = lib.build_hybrid_gt(human_post, adj_v2_post, pairing_post=pairing_post)
p_hybrid = lib.derive_correct_flags(
    lib.filter_universe(lib.apply_gt_to_pairing(pairing_post, gt_hybrid))
)

check_df("TN3post marginals", lib.per_variant_marginals(p_hybrid), lib.TABLES_DIR / "T_TN3post_T2_variant_marginals.csv")
check_df("TN3post kappa_pair", lib.pairwise_kappa_variants(p_hybrid), lib.TABLES_DIR / "T_TN3post_T3_pairwise_kappa.csv")
# T6 lưu dạng pivot phase × variant (rate_correct), không phải long format
ppb = lib.per_phase_breakdown(p_hybrid)
ppb_pivot = ppb.pivot(index="phase", columns="variant", values="rate_correct").reset_index()
ppb_pivot.columns.name = None
check_df("TN3post per_phase", ppb_pivot, lib.TABLES_DIR / "T_TN3post_T6_per_phase.csv")

# ── TN3 pre-fix factorial ─────────────────────────────────────────────────────
print("\n=== TN3 pre-fix factorial ===")
pairing_pre = pd.read_csv(lib.PRE_FIX_DIR / "pairing_pre_topic_fix.csv", low_memory=False)
human_v1 = pd.read_csv(lib.PRE_FIX_DIR / "human_review_prefix.csv", low_memory=False)
adj_v2_pre = pd.read_csv(lib.PRE_FIX_DIR / "adjudicated_v2_prefix.csv", low_memory=False)

gt_pre = lib.build_hybrid_gt(human_v1, adj_v2_pre, pairing_post=pairing_post)
p_pre = lib.derive_correct_flags(
    lib.filter_universe(lib.apply_gt_to_pairing(pairing_pre, gt_pre))
)

check_df("TN3pre kappa_pair", lib.pairwise_kappa_variants(p_pre), lib.TABLES_DIR / "T_TN3pre_T3_pairwise_kappa.csv")
ppb_pre = lib.per_phase_breakdown(p_pre)
ppb_pre_pivot = ppb_pre.pivot(index="phase", columns="variant", values="rate_correct").reset_index()
ppb_pre_pivot.columns.name = None
check_df("TN3pre per_phase", ppb_pre_pivot, lib.TABLES_DIR / "T_TN3pre_T6_per_phase.csv")

# ── TN3 post-fix bugfix (T3) ──────────────────────────────────────────────────
print("\n=== TN3 bugfix (T3) ===")
df_14 = lib.load_postfix_sample(view="14")
df_15 = lib.load_postfix_sample(view="15")

acc_14 = lib.build_accuracy_table_postfix(df_14, label="14-report")
acc_15 = lib.build_accuracy_table_postfix(df_15, label="15-report")
acc_all = pd.concat([acc_14, acc_15], ignore_index=True)
check_df("T9 postfix_accuracy", acc_all, lib.TABLES_DIR / "T9_postfix_accuracy.csv")

lift_14 = lib.build_lift_table_postfix(df_14, label="14-report")
lift_15 = lib.build_lift_table_postfix(df_15, label="15-report")
lift_all = pd.concat([lift_14, lift_15], ignore_index=True)
check_df("T10 postfix_lift", lift_all, lib.TABLES_DIR / "T10_postfix_lift.csv")

lat = lib.latency_per_variant()
check_df("T13 latency", lat, lib.TABLES_DIR / "T13_latency_per_variant.csv")

print()
print("=" * 60)
print(f"{'ALL PASS' if all_ok else 'SOME FAILED'}")
print("=" * 60)
sys.exit(0 if all_ok else 1)
