"""
Step 2.3 LC — Ranking Validation  [LIQUID COOLING]
====================================================
Phase 2: IS-Match Score (OS2) · Thesis Cap. 5.2 · RQ1 principale

Validates that IS-Match Score provides superior ranking quality vs.
RI_static baseline (Heriot-Watt / WHER D2) for LC DC–manufacturing pairs.

Metrics:
  1. NDCG@k  — Normalized Discounted Cumulative Gain
               Ground truth relevance: RI_static from CE-HEAT LC dataset.
               k = min(9, n_plants) since 9 plants per DC.
               Measures ranking quality of IS-Match vs RI_temporal-only.
  2. precision@3 — fraction of top-3 IS-Match pairs that are in top-3 RI_static
  3. RI > 0.5 rate — binary feasibility fraction: IS-Match vs RI_static threshold

  Expected result (Cap. 5.2):
    - LowT_60C: IS-Match ≈ RI_static (T_compat dominates both)
    - MidT/HighT: IS-Match promotes pairs with higher Exergy_DT_norm
      → NDCG(IS-Match) ≥ NDCG(RI_temporal) for mid/high-T tiers
    - Exergy_DT_norm uniform (~0.75) for LC → discrimination via ΔTC_norm

  Sensitivity analysis:
    - Vary β, γ, δ weights → stability of NDCG ranking
    - Group by: DC scale, process T, upgrade technology

References:
  Baseline: WHER (D2), Heriot-Watt — static RI index
  Metric:   NDCG from scikit-learn (ndcg_score)
  Ref B6:   Hatsor & Jelnov (2024) for ε-parameter
"""

# Decision active: D6 — NDCG@9, precision@3 validated on LC pairs only.


from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _dcg(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel / discounts))


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k (pure numpy). y_true/y_score shape: (1, n_docs)."""
    true_1d  = y_true[0]
    score_1d = y_score[0]
    rank_idx  = np.argsort(score_1d)[::-1]
    ideal_idx = np.argsort(true_1d)[::-1]
    dcg_val   = _dcg(true_1d[rank_idx],  k)
    idcg_val  = _dcg(true_1d[ideal_idx], k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RESULTS_DIR   = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SCORES_CSV    = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
# Dataset comes from step_2_2 (results/) or committed copy (data/)
DATASET_CSV   = RESULTS_DIR / "step_2_2_ce_heat_dataset_lc.csv"
if not DATASET_CSV.exists():
    for _alt in [RESULTS_DIR / "ce_heat_inspired_dataset_lc.csv",
                 DATA_DIR    / "ce_heat_inspired_dataset_lc.csv",
                 BASE_DIR    / "ce_heat_inspired_dataset_lc.csv"]:
        if _alt.exists():
            DATASET_CSV = _alt
            break
OUT_METRICS   = RESULTS_DIR / "step_2_3_ranking_metrics_lc.csv"
OUT_SENS      = RESULTS_DIR / "step_2_3_sensitivity_ranking_lc.csv"
SENS_IS_MATCH = RESULTS_DIR / "step_2_1_sensitivity_lc.csv"

NDCG_K = 9  # 9 plants per DC query


def compute_ndcg_for_dc(
    dc_name: str,
    df_scores: pd.DataFrame,
    df_dataset: pd.DataFrame,
    score_col: str,
    relevance_col: str = "ri_static",
    k: int = NDCG_K,
) -> float:
    """NDCG@k treating each DC as one query, 9 plants as documents."""
    plants = df_scores[df_scores["dc_name"] == dc_name]["process_name"].tolist()
    scores  = []
    rel_gt  = []
    for proc in plants:
        s_row = df_scores[(df_scores["dc_name"]==dc_name) & (df_scores["process_name"]==proc)]
        d_row = df_dataset[(df_dataset["dc_name"]==dc_name) & (df_dataset["process_name"]==proc)]
        if s_row.empty or d_row.empty:
            continue
        scores.append(float(s_row[score_col].mean()))
        rel_gt.append(float(d_row[relevance_col].mean()))
    if len(scores) < 2:
        return float("nan")
    # ndcg_score expects shape (1, n_docs)
    return float(ndcg_score(
        np.array([rel_gt]), np.array([scores]),
        k=min(k, len(scores)),
    ))


def precision_at_k(
    dc_name: str,
    df_scores: pd.DataFrame,
    df_dataset: pd.DataFrame,
    score_col: str,
    relevance_col: str = "ri_static",
    k: int = 3,
) -> float:
    """Fraction of top-k by score_col that are also top-k by relevance_col."""
    sub_s = df_scores[df_scores["dc_name"] == dc_name].copy()
    sub_d = df_dataset[df_dataset["dc_name"] == dc_name].copy()
    merged = sub_s.merge(sub_d[["process_name", relevance_col]], on="process_name", how="inner")
    if len(merged) < k:
        return float("nan")
    top_k_score = set(merged.nlargest(k, score_col)["process_name"])
    top_k_rel   = set(merged.nlargest(k, relevance_col)["process_name"])
    return len(top_k_score & top_k_rel) / k


def compute_all_metrics(
    df_scores: pd.DataFrame,
    df_dataset: pd.DataFrame,
    score_col: str,
    label: str,
) -> pd.DataFrame:
    """Compute NDCG@k, precision@3, RI>0.5 rate per DC and overall."""
    rows = []
    dcs = df_scores["dc_name"].unique().tolist()

    for dc in dcs:
        ndcg = compute_ndcg_for_dc(dc, df_scores, df_dataset, score_col)
        p3   = precision_at_k(dc, df_scores, df_dataset, score_col, k=3)

        sub_s = df_scores[df_scores["dc_name"] == dc]
        sub_d = df_dataset[df_dataset["dc_name"] == dc]
        merged = sub_s.merge(sub_d[["process_name","ri_static","label_feasible"]], on="process_name")
        n_feasible_is_match = int((merged[score_col] >= 0.30).sum())  # marginal+ threshold
        n_feasible_baseline = int(sub_d["label_feasible"].sum())
        rows.append({
            "method":              label,
            "dc_name":             dc,
            "ndcg_at_k":          round(ndcg, 4),
            "precision_at_3":      round(p3,   4),
            "n_marginal_plus":     n_feasible_is_match,
            "n_feasible_baseline": n_feasible_baseline,
            "ri_gt05_rate_pct":    round(100 * n_feasible_baseline / max(len(sub_d), 1), 1),
        })

    # Overall row (mean across DCs)
    df_rows = pd.DataFrame(rows)
    overall = df_rows[["ndcg_at_k","precision_at_3","n_marginal_plus",
                        "n_feasible_baseline","ri_gt05_rate_pct"]].mean()
    rows.append({
        "method":              label,
        "dc_name":             "ALL",
        "ndcg_at_k":          round(float(overall["ndcg_at_k"]),   4),
        "precision_at_3":      round(float(overall["precision_at_3"]), 4),
        "n_marginal_plus":     round(float(overall["n_marginal_plus"]), 1),
        "n_feasible_baseline": round(float(overall["n_feasible_baseline"]), 1),
        "ri_gt05_rate_pct":    round(float(overall["ri_gt05_rate_pct"]), 1),
    })
    return pd.DataFrame(rows)


def run_weight_sensitivity(
    df_sens_is: pd.DataFrame,
    df_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """NDCG@k for each (β,γ,δ) weight combination in sensitivity grid."""
    rows = []
    for (beta, gamma, delta), grp in df_sens_is.groupby(["beta","gamma","delta"]):
        scores_pivot = grp.pivot_table(index="dc_name", columns="process_name",
                                       values="is_match_score")
        ndcg_vals = []
        for dc in scores_pivot.index:
            s_row = scores_pivot.loc[dc].dropna()
            d_sub = df_dataset[df_dataset["dc_name"] == dc].set_index("process_name")
            common = s_row.index.intersection(d_sub.index)
            if len(common) < 2:
                continue
            rel = d_sub.loc[common, "ri_static"].values
            scr = s_row.loc[common].values
            ndcg_vals.append(float(ndcg_score(
                np.array([rel]), np.array([scr]),
                k=min(NDCG_K, len(common))
            )))
        if ndcg_vals:
            rows.append({
                "beta": beta, "gamma": gamma, "delta": delta,
                "ndcg_mean": round(np.mean(ndcg_vals), 4),
                "ndcg_min":  round(np.min(ndcg_vals),  4),
                "ndcg_max":  round(np.max(ndcg_vals),  4),
            })
    return pd.DataFrame(rows).sort_values("ndcg_mean", ascending=False)


def main() -> None:
    # ── Load inputs ────────────────────────────────────────────────────────────
    for p in [SCORES_CSV, DATASET_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    df_scores  = pd.read_csv(SCORES_CSV)
    df_dataset = pd.read_csv(DATASET_CSV)

    # Merge dataset ri_static into scores for RI_temporal comparison
    df_ri = df_dataset[["dc_name","process_name","ri_static","label_feasible"]].copy()
    df_ri_scores = df_ri.rename(columns={"ri_static": "is_match_score"})

    logger.info("=" * 70)
    logger.info("  Step 2.3 LC — Ranking Validation  (NDCG@%d, precision@3)", NDCG_K)
    logger.info("=" * 70)

    # ── Compute metrics for both methods ───────────────────────────────────────
    metrics_is  = compute_all_metrics(df_scores,    df_dataset, "is_match_score", "IS-Match_LC")
    metrics_ri  = compute_all_metrics(df_ri_scores, df_dataset, "is_match_score", "RI_static_baseline")

    df_metrics = pd.concat([metrics_is, metrics_ri], ignore_index=True)
    df_metrics.to_csv(OUT_METRICS, index=False)
    logger.info(f"  Metrics → {OUT_METRICS}")

    # ── Print comparison table ─────────────────────────────────────────────────
    logger.info("")
    logger.info("  Comparison: IS-Match LC  vs.  RI_static (Heriot-Watt baseline)")
    logger.info("  " + "-" * 64)
    logger.info(f"  {'DC':<16} {'NDCG@9 IS-Match':>16} {'NDCG@9 RI_stat':>15} "
                f"{'P@3 IS-Match':>14} {'P@3 RI_stat':>12}")
    logger.info("  " + "-" * 64)
    for dc in list(df_scores["dc_name"].unique()) + ["ALL"]:
        row_is = metrics_is[metrics_is["dc_name"]==dc]
        row_ri = metrics_ri[metrics_ri["dc_name"]==dc]
        if row_is.empty or row_ri.empty:
            continue
        ndcg_is = float(row_is["ndcg_at_k"].iloc[0])
        ndcg_ri = float(row_ri["ndcg_at_k"].iloc[0])
        p3_is   = float(row_is["precision_at_3"].iloc[0])
        p3_ri   = float(row_ri["precision_at_3"].iloc[0])
        flag = "↑" if ndcg_is >= ndcg_ri else "↓"
        logger.info(f"  {dc:<16} {ndcg_is:>16.4f} {ndcg_ri:>15.4f} "
                    f"{p3_is:>14.4f} {p3_ri:>12.4f}  {flag}")

    # ── Tier breakdown ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("  IS-Match tier distribution:")
    for dc in df_scores["dc_name"].unique():
        sub = df_scores[df_scores["dc_name"]==dc]
        tc  = sub["tier"].value_counts().to_dict()
        logger.info(f"    {dc:<16}: high={tc.get('high_priority',0)} "
                    f"marginal={tc.get('marginal',0)} "
                    f"not_viable={tc.get('not_viable',0)}")
    logger.info("=" * 70)

    # ── Sensitivity over weight grid ───────────────────────────────────────────
    if SENS_IS_MATCH.exists():
        df_sens_is = pd.read_csv(SENS_IS_MATCH)
        df_weight_sens = run_weight_sensitivity(df_sens_is, df_dataset)
        df_weight_sens.to_csv(OUT_SENS, index=False)
        logger.info(f"  Weight sensitivity → {OUT_SENS}  ({len(df_weight_sens)} rows)")
        logger.info("  NDCG@k stability over (β,γ,δ) grid:")
        logger.info(f"  {'β':>6} {'γ':>6} {'δ':>6} {'NDCG_mean':>12} {'NDCG_min':>10} {'NDCG_max':>10}")
        for _, row in df_weight_sens.head(8).iterrows():
            logger.info(f"  {row['beta']:>6.2f} {row['gamma']:>6.2f} {row['delta']:>6.2f} "
                        f"{row['ndcg_mean']:>12.4f} {row['ndcg_min']:>10.4f} {row['ndcg_max']:>10.4f}")

    # ── Thesis note ────────────────────────────────────────────────────────────
    print("\n  THESIS NOTE (Cap. 5.2 — RQ1)