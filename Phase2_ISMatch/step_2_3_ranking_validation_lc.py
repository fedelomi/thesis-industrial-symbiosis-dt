"""
Step 2.3 LC - Ranking Validation  [LIQUID COOLING]
====================================================
Phase 2: IS-Match Score (OS2) - Thesis Cap. 5 - RQ1.

Validates that the IS-Match Score ranks DC-manufacturing pairs consistently with
the established RI_static baseline (WHER / Heriot-Watt static recovery index), and
that adding the exergy and transaction-cost terms does not degrade ranking quality
relative to the temporal recovery index alone.

RANKING UNIT (reconstruction decision, 2026-05-30)
--------------------------------------------------
The ranking unit is the PLANT-LEVEL pair: each DC is a query of 9 candidate plants
(3 plants per temperature tier), for 27 DC-plant pairs in total. This is the finest
unit the dataset supports and the only one where the validation metrics discriminate.

  Why not 3 tiers per DC: ranking only 3 items makes top-3 the whole set, so
  precision@3 is 1.000 by construction. That is the trivial quick-fix this file
  replaces.

  Per-DC vs pooled: within any single DC the three LowT plants top both IS-Match and
  RI_static, so the per-DC top-3 of 9 plants agree (precision@3 = 1.0 per DC, a
  genuine 3-of-9 agreement rather than a construction artefact). The discriminating
  metric is the POOLED precision@3 over the 27 pairs: there the two indices weight
  capacity differently (IS-Match uses capacity utilisation Q_available / Q_nominal,
  near 0.8 for every DC scale, while RI_static uses capacity match to plant demand,
  RI_quantity, which penalises the small Edge DC against large plants). RI_static
  therefore ranks the Mid and Hyperscale LowT plants above the Edge LowT plants while
  IS-Match rates them near-equal, so the pooled top-3 sets only partially overlap and
  the pooled precision@3 falls below 1.0. The pooled value is the headline.

Methods compared (both ranked against the RI_static ground truth):
  1. IS-Match (full)        score = beta*RI_temporal + gamma*Exergy_DT_norm - delta*DTC_norm
  2. RI_temporal (baseline) score = RI_temporal alone (the WHER temporal index)
  The per-plant scores are computed with the SAME functions as step_2_1 (imported
  here as the single source of the scoring formula) applied to each plant's own
  attributes (distance, shift pattern, demand, transaction-cost baseline).

Metrics:
  - NDCG@9       graded relevance = RI_static; per DC (within-query quality) and
                 pooled over the 27 pairs.
  - precision@3  fraction of the top-3 by score that are in the top-3 by RI_static;
                 per DC and pooled (the pooled value is the discriminating metric).
  - n_marginal_plus    count of per-plant IS-Match >= 0.30 (marginal-or-better tier).
  - n_feasible_baseline  count of dataset feasible flags (RI_static-based).

Sensitivity:
  - NDCG@9 (per-DC mean) recomputed across the (beta, gamma, delta) weight grid to
    confirm the ranking quality is stable to the weight choice. Deterministic (no RNG).

References:
  Baseline: WHER (D2), Heriot-Watt - static RI index (RI_static, in the dataset).
  Metric:   NDCG (graded relevance), precision@k (pure numpy).
  Ref B6:   Hatsor & Jelnov (2024) for the epsilon-parameter gate (step_2_0).
"""

# Decision active: D6 - NDCG@9, precision@3 validated on LC pairs only.


from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Import step_2_1 as the single source of the IS-Match scoring formula.
# A direct import is used (not importlib.exec_module) so the @dataclass decorators
# inside step_2_1 resolve their module correctly.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import step_2_1_is_match_score_lc as s21  # noqa: E402

DATASET_CSV = RESULTS_DIR / "step_2_2_ce_heat_dataset_lc.csv"
if not DATASET_CSV.exists():
    for _alt in [RESULTS_DIR / "ce_heat_inspired_dataset_lc.csv",
                 DATA_DIR    / "ce_heat_inspired_dataset_lc.csv",
                 BASE_DIR    / "ce_heat_inspired_dataset_lc.csv"]:
        if _alt.exists():
            DATASET_CSV = _alt
            break

OUT_METRICS = RESULTS_DIR / "step_2_3_ranking_metrics_lc.csv"
OUT_SENS    = RESULTS_DIR / "step_2_3_sensitivity_ranking_lc.csv"

NDCG_K       = 9     # 9 plants per DC query
PRECISION_K  = 3     # top-3 agreement
MARGINAL_MIN = 0.30


def _dcg(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel / discounts))


def ndcg_at_k(rel_by_score: np.ndarray, rel_ideal: np.ndarray, k: int) -> float:
    """NDCG@k. rel_by_score: relevances ordered by the method score (desc);
    rel_ideal: relevances ordered by the ground truth (desc)."""
    dcg  = _dcg(rel_by_score, k)
    idcg = _dcg(rel_ideal,    k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(names_by_score: list[str], names_by_rel: list[str], k: int) -> float:
    """Fraction of the top-k by score that are also top-k by relevance."""
    if len(names_by_score) < k or len(names_by_rel) < k:
        return float("nan")
    return len(set(names_by_score[:k]) & set(names_by_rel[:k])) / k


def compute_plant_scores(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-plant IS-Match (full) and RI_temporal (baseline) for the 27 DC-plant pairs.

    Uses the imported step_2_1 functions on each plant's own attributes. The exergy
    term is per-DC (uniform across plants of a DC) so it shifts a DC's plants equally;
    the discriminating per-plant terms are RI_temporal (distance, shift, demand) and
    the transaction-cost penalty.
    """
    if not s21.PHASE1_LC_CSV.exists():
        raise FileNotFoundError(f"Phase 1 LC CSV not found: {s21.PHASE1_LC_CSV}")
    df_dt = pd.read_csv(s21.PHASE1_LC_CSV)
    exergy_max = df_dt.groupby(s21.COL_SCENARIO)[s21.COL_EXERGY].max().to_dict()

    # A stable global key per pair so pooled top-k sets are unambiguous.
    rows: list[dict] = []
    for dc_name in df_dt[s21.COL_SCENARIO].unique():
        sub   = df_dt[df_dt[s21.COL_SCENARIO] == dc_name].reset_index(drop=True)
        q_nom = s21.Q_NOMINAL_LC.get(dc_name, float(sub[s21.COL_Q_AVAIL].max()))
        n     = len(sub)
        ex_norm = float(np.clip(sub[s21.COL_EXERGY].mean() / exergy_max[dc_name], 0.0, 1.0))
        t_arr = sub[s21.COL_T_SUPPLY].to_numpy(float)
        q_arr = sub[s21.COL_Q_AVAIL].to_numpy(float)
        a_arr = sub[s21.COL_T_AVAIL].to_numpy(float)

        for _, p in df_dataset[df_dataset["dc_name"] == dc_name].iterrows():
            spec = s21.ManufacturingSpec(
                process_name=str(p["plant_name"]),
                t_req_c=float(p["plant_t_req_c"]),
                distance_km=float(p["distance_km"]),
                shift_type=str(p["plant_shift_pattern"]),
                q_demand_kw=float(p["plant_q_demand_kw"]),
                delta_tc_baseline=float(p["plant_delta_tc_baseline"]),
            )
            w        = s21.make_shift_weights(n, spec.shift_type)
            ri_temp  = s21.compute_ri_temporal(
                s21.compute_ri_series(t_arr, q_arr, q_nom, a_arr, spec), w)
            dtc      = float(np.clip(spec.delta_tc_baseline, 0.0, 1.0))
            is_match = s21.compute_is_match(ri_temp, ex_norm, dtc)
            rows.append({
                "pair_key":       f"{dc_name}|{spec.process_name}",
                "dc_name":        dc_name,
                "plant_name":     spec.process_name,
                "plant_tier":     str(p["plant_tier"]),
                "ri_temporal":    round(ri_temp, 6),
                "exergy_dt_norm": round(ex_norm, 6),
                "delta_tc_norm":  round(dtc, 6),
                "is_match":       round(is_match, 6),
                "ri_static":      float(p["RI_static"]),
                "feasible":       int(p["feasible"]),
            })
    return pd.DataFrame(rows)


def _rank_metrics(grp: pd.DataFrame, score_col: str, k_ndcg: int) -> tuple[float, float]:
    """NDCG@k and precision@PRECISION_K of one ranking group against RI_static.

    Ties on the score are broken by RI_static descending so the ranking is fully
    deterministic and the metric is conservative (it does not reward an arbitrary
    tie order)."""
    by_score = grp.sort_values([score_col, "ri_static"], ascending=False)
    by_rel   = grp.sort_values("ri_static", ascending=False)
    ndcg = ndcg_at_k(by_score["ri_static"].to_numpy(),
                     by_rel["ri_static"].to_numpy(), k_ndcg)
    p_at = precision_at_k(by_score["pair_key"].tolist(),
                          by_rel["pair_key"].tolist(), PRECISION_K)
    return round(ndcg, 4), round(p_at, 4)


def compute_all_metrics(df_plants: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    """Per-DC NDCG@9 + precision@3, plus a pooled ALL_pooled row over the 27 pairs."""
    rows: list[dict] = []
    for dc in df_plants["dc_name"].unique():
        grp = df_plants[df_plants["dc_name"] == dc]
        ndcg, p3 = _rank_metrics(grp, score_col, NDCG_K)
        rows.append({
            "method":              label,
            "scope":               dc,
            "ndcg_at_9":           ndcg,
            "precision_at_3":      p3,
            "n_marginal_plus":     int((grp["is_match"] >= MARGINAL_MIN).sum()),
            "n_feasible_baseline": int(grp["feasible"].sum()),
            "n_pairs":             int(len(grp)),
        })
    ndcg_all, p3_all = _rank_metrics(df_plants, score_col, NDCG_K)
    rows.append({
        "method":              label,
        "scope":               "ALL_pooled",
        "ndcg_at_9":           ndcg_all,
        "precision_at_3":      p3_all,
        "n_marginal_plus":     int((df_plants["is_match"] >= MARGINAL_MIN).sum()),
        "n_feasible_baseline": int(df_plants["feasible"].sum()),
        "n_pairs":             int(len(df_plants)),
    })
    return pd.DataFrame(rows)


def run_weight_sensitivity(df_plants: pd.DataFrame) -> pd.DataFrame:
    """Per-DC-mean NDCG@9 across the (beta, gamma, delta) weight grid. Deterministic."""
    base = df_plants[["dc_name", "pair_key", "ri_temporal", "exergy_dt_norm",
                      "delta_tc_norm", "ri_static"]].copy()
    grid = []
    for beta, gamma in itertools.product([0.2, 0.3, 0.4, 0.5], repeat=2):
        delta = round(1.0 - beta - gamma, 4)
        if delta < 0.0:
            continue
        grid.append((beta, gamma, delta))

    rows = []
    for beta, gamma, delta in grid:
        scored = base.copy()
        scored["score"] = np.clip(
            beta * scored["ri_temporal"] + gamma * scored["exergy_dt_norm"]
            - delta * scored["delta_tc_norm"], 0.0, 1.0)
        per_dc = []
        for dc in scored["dc_name"].unique():
            grp = scored[scored["dc_name"] == dc]
            by_score = grp.sort_values(["score", "ri_static"], ascending=False)
            by_rel   = grp.sort_values("ri_static", ascending=False)
            per_dc.append(ndcg_at_k(by_score["ri_static"].to_numpy(),
                                    by_rel["ri_static"].to_numpy(), NDCG_K))
        rows.append({
            "beta": beta, "gamma": gamma, "delta": delta,
            "ndcg_mean": round(float(np.mean(per_dc)), 4),
            "ndcg_min":  round(float(np.min(per_dc)), 4),
            "ndcg_max":  round(float(np.max(per_dc)), 4),
        })
    return (pd.DataFrame(rows)
            .sort_values(["ndcg_mean", "beta", "gamma"], ascending=[False, True, True])
            .reset_index(drop=True))


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")
    df_dataset = pd.read_csv(DATASET_CSV)

    df_plants = compute_plant_scores(df_dataset)

    logger.info("=" * 70)
    logger.info("  Step 2.3 LC - Ranking Validation  (NDCG@%d, precision@%d)",
                NDCG_K, PRECISION_K)
    logger.info("  Ranking unit: 9 plants per DC (27 DC-plant pairs; pooled is headline)")
    logger.info("=" * 70)

    metrics_is = compute_all_metrics(df_plants, "is_match",    "IS-Match_LC")
    metrics_ri = compute_all_metrics(df_plants, "ri_temporal", "RI_temporal_baseline")
    df_metrics = pd.concat([metrics_is, metrics_ri], ignore_index=True)
    df_metrics.to_csv(OUT_METRICS, index=False)
    logger.info(f"  Metrics -> {OUT_METRICS}")

    # Comparison table
    logger.info("")
    logger.info("  IS-Match (full)  vs  RI_temporal (baseline)   [ground truth: RI_static]")
    logger.info("  " + "-" * 66)
    logger.info(f"  {'scope':<14} {'NDCG@9 IS':>10} {'NDCG@9 RI':>10} "
                f"{'P@3 IS':>8} {'P@3 RI':>8}")
    logger.info("  " + "-" * 66)
    for scope in list(df_plants["dc_name"].unique()) + ["ALL_pooled"]:
        is_row = metrics_is[metrics_is["scope"] == scope].iloc[0]
        rt_row = metrics_ri[metrics_ri["scope"] == scope].iloc[0]
        logger.info(f"  {scope:<14} {is_row['ndcg_at_9']:>10.4f} {rt_row['ndcg_at_9']:>10.4f} "
                    f"{is_row['precision_at_3']:>8.4f} {rt_row['precision_at_3']:>8.4f}")

    # Transparency: print the pooled top-3 sets that drive the discriminating metric
    pooled_is  = df_plants.sort_values(["is_match", "ri_static"], ascending=False)
    pooled_rel = df_plants.sort_values("ri_static", ascending=False)
    logger.info("")
    logger.info("  Pooled top-3 by IS-Match : %s", pooled_is.head(3)["pair_key"].tolist())
    logger.info("  Pooled top-3 by RI_static: %s", pooled_rel.head(3)["pair_key"].tolist())
    logger.info("  Pooled precision@3 (headline, discriminating): %.4f",
                float(metrics_is[metrics_is["scope"] == "ALL_pooled"]["precision_at_3"].iloc[0]))

    # Tier distribution from the tier-level canonical scores (step_2_1 output)
    scores_csv = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
    if scores_csv.exists():
        df_scores = pd.read_csv(scores_csv)
        logger.info("")
        logger.info("  IS-Match tier distribution (tier-level, step_2_1):")
        for dc in df_scores["dc_name"].unique():
            tc = df_scores[df_scores["dc_name"] == dc]["tier"].value_counts().to_dict()
            logger.info(f"    {dc:<16}: high={tc.get('high_priority',0)} "
                        f"marginal={tc.get('marginal',0)} not_viable={tc.get('not_viable',0)}")
    logger.info("=" * 70)

    # Weight sensitivity (per-DC-mean NDCG@9 over the grid)
    df_weight_sens = run_weight_sensitivity(df_plants)
    df_weight_sens.to_csv(OUT_SENS, index=False)
    logger.info(f"  Weight sensitivity -> {OUT_SENS}  ({len(df_weight_sens)} rows)")
    logger.info("  NDCG@9 stability over (beta, gamma, delta) grid:")
    logger.info(f"  {'beta':>6} {'gamma':>6} {'delta':>6} {'NDCG_mean':>11} "
                f"{'NDCG_min':>10} {'NDCG_max':>10}")
    for _, row in df_weight_sens.head(8).iterrows():
        logger.info(f"  {row['beta']:>6.2f} {row['gamma']:>6.2f} {row['delta']:>6.2f} "
                    f"{row['ndcg_mean']:>11.4f} {row['ndcg_min']:>10.4f} {row['ndcg_max']:>10.4f}")

    print("\n  THESIS NOTE (Cap. 5, RQ1): IS-Match ranking validated against the RI_static "
          "baseline at the plant level (9 plants per DC, 27 pooled; pooled precision@3 is "
          "the discriminating metric).")


if __name__ == "__main__":
    main()
