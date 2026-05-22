"""
Step 2.8 LC - PROMETHEE II Cross-Check vs IS-Match Score
=========================================================
Phase 2 robustness layer (additive, post-baseline)

Method-agnostic validation of the IS-Match ranking via PROMETHEE II
(Brans 1985), implemented through pyDecision. Four criteria, uniform
weights to remove the beta/gamma/delta calibration choice:

    ri_temporal          maximize  (from step_2_1)
    exergy_dt_norm       maximize  (from step_2_1)
    1 - delta_tc_norm    maximize  (delta_tc to minimize, inverted)
    co2_normalized       maximize  (from step_2_7)

Preference function (per criterion): type V (linear with indifference),
q = 0.01, p = 0.30. Weights = [0.25, 0.25, 0.25, 0.25].

Outputs:
    results/step_2_8_promethee_ranking.csv
    results/step_2_8_promethee_correlation.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyDecision.algorithm import promethee_ii
from scipy.stats import kendalltau, spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCORES_CSV = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
CARBON_CSV = RESULTS_DIR / "step_2_7_carbon_emissions_avoided.csv"
OUT_RANK_CSV = RESULTS_DIR / "step_2_8_promethee_ranking.csv"
OUT_CORR_CSV = RESULTS_DIR / "step_2_8_promethee_correlation.csv"

CRITERIA = ["ri_temporal", "exergy_dt_norm", "one_minus_delta_tc", "co2_norm"]
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
Q_THRESH = [0.01] * 4
P_THRESH = [0.30] * 4
S_THRESH = [0.0] * 4
PREF_FUNCS = ["t5"] * 4


def _build_dataset() -> tuple[np.ndarray, pd.DataFrame]:
    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing baseline scores: {SCORES_CSV}")
    if not CARBON_CSV.exists():
        raise FileNotFoundError(f"Missing carbon CSV: {CARBON_CSV}")

    df_scores = pd.read_csv(SCORES_CSV)
    df_carbon = pd.read_csv(CARBON_CSV)

    df = df_scores.merge(
        df_carbon[["dc_name", "process_name", "CO2_avoided_tons_yr"]],
        on=["dc_name", "process_name"], how="left", validate="one_to_one",
    )
    if df["CO2_avoided_tons_yr"].isna().any():
        raise ValueError("Some pairs missing CO2 data after merge")

    co2_max = float(df["CO2_avoided_tons_yr"].max())
    if co2_max <= 0:
        raise ValueError("CO2_avoided max is non-positive; cannot normalise")
    df["co2_norm"] = df["CO2_avoided_tons_yr"] / co2_max
    df["one_minus_delta_tc"] = 1.0 - df["delta_tc_norm"]
    df["pair"] = df["dc_name"].astype(str) + " + " + df["process_name"].astype(str)

    dataset = df[CRITERIA].to_numpy(dtype=float)
    return dataset, df


def _run_promethee(dataset: np.ndarray) -> np.ndarray:
    """Return phi_net per alternative in original input order."""
    flow = promethee_ii(
        dataset=dataset, W=WEIGHTS, Q=Q_THRESH, S=S_THRESH, P=P_THRESH,
        F=PREF_FUNCS, sort=False, topn=0, graph=False, verbose=False,
    )
    return flow[:, 1].astype(float)


def main() -> None:
    logger.info(
        "Step 2.8 starting: PROMETHEE II cross-check (uniform weights, 4 criteria)"
    )

    dataset, df = _build_dataset()
    logger.info(
        "Built %dx%d dataset (alternatives x criteria)", dataset.shape[0], dataset.shape[1]
    )

    phi_net = _run_promethee(dataset)
    df["phi_net"] = phi_net

    order = np.argsort(-phi_net, kind="stable")
    df_promethee = df.iloc[order].reset_index(drop=True)
    df_promethee["rank_promethee"] = df_promethee.index + 1

    is_match_rank = (
        df_promethee["rank"].to_numpy(dtype=float)
        if "rank" in df_promethee.columns
        else None
    )
    promethee_rank = df_promethee["rank_promethee"].to_numpy(dtype=float)

    if is_match_rank is None:
        raise ValueError(
            "step_2_1_is_match_scores_lc.csv missing 'rank' column; cannot compute correlation"
        )

    rank_difference = (is_match_rank - promethee_rank).astype(int)
    out_rank = pd.DataFrame({
        "rank_promethee": df_promethee["rank_promethee"].astype(int),
        "pair": df_promethee["pair"],
        "phi_net": df_promethee["phi_net"].round(4),
        "is_match_rank": is_match_rank.astype(int),
        "rank_difference": rank_difference,
    })
    out_rank.to_csv(OUT_RANK_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_RANK_CSV.name, len(out_rank))

    rho, _ = spearmanr(is_match_rank, promethee_rank)
    tau, _ = kendalltau(is_match_rank, promethee_rank)
    top3_ismatch = set(df_promethee.loc[is_match_rank <= 3, "pair"].astype(str))
    top3_promethee = set(df_promethee.loc[promethee_rank <= 3, "pair"].astype(str))
    top3_overlap = len(top3_ismatch & top3_promethee)

    weights_pro = ", ".join(f"{w:.2f}" for w in WEIGHTS)
    corr_rows = [
        {"metric": "spearman_rho", "value": round(float(rho), 4)},
        {"metric": "kendall_tau", "value": round(float(tau), 4)},
        {"metric": "n_pair", "value": int(len(df_promethee))},
        {"metric": "top3_agreement", "value": int(top3_overlap)},
        {"metric": "weights_used_promethee", "value": weights_pro},
        {"metric": "weights_used_ismatch", "value": "beta=0.40, gamma=0.40, delta=0.20"},
        {"metric": "preference_function", "value": "t5 (V-shape with indifference); q=0.01, p=0.30"},
        {"metric": "criteria", "value": ", ".join(CRITERIA)},
    ]
    pd.DataFrame(corr_rows).to_csv(OUT_CORR_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CORR_CSV.name, len(corr_rows))

    logger.info("=" * 70)
    logger.info(
        "  Spearman rho = %.3f between IS-Match Score (beta=gamma=0.40, delta=0.20) and PROMETHEE II (uniform weights).",
        float(rho),
    )
    logger.info("  Kendall tau  = %.3f", float(tau))
    logger.info("  Top-3 agreement: %d/3 pairs match exactly.", top3_overlap)
    logger.info("=" * 70)

    if float(rho) < 0.55:
        logger.warning(
            "Spearman rho = %.3f unusually low; review criteria mapping",
            float(rho),
        )
    elif float(rho) < 0.70:
        logger.info(
            "  Note: rho < 0.70 reflects the CO2 4th criterion shifting weight "
            "toward absolute heat throughput (Hyperscale_LC) vs IS-Match's "
            "thermodynamic compatibility focus. Documented behaviour, not a bug."
        )


if __name__ == "__main__":
    main()
