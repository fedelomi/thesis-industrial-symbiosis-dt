"""
Step 2.5 LC - Sobol Sensitivity on IS-Match Weights
====================================================
Phase 2 robustness layer (additive, post-baseline)

Quantifies how much each of the three weights beta, gamma and delta
contributes to the variance of the IS-Match Score across the 9 LC pairs.

Method:
    Saltelli sampling (N=1024, calc_second_order=False) on
        beta  in [0.20, 0.60]
        gamma in [0.20, 0.60]
        delta in [0.00, 0.40]
    Each sample is renormalised to sum=1 before scoring.

    Two response variables are analysed:
        Y_aggregate         = mean IS-Match Score across the 9 pairs
        top3_stability_pct  = 1 if perturbed top-3 set equals baseline
                              top-3 set (LowT_60C with Edge, Mid and
                              Hyperscale LC), else 0

Outputs:
    results/step_2_5_sobol_weights.csv
        metric, parameter, S1, S1_conf, ST, ST_conf

Additivity sanity check: sum(S1) for Y_aggregate should be ~ 1.0
(when calc_second_order=False, residual is interactions captured in ST).

Reference: Sobol (1993); Saltelli et al. (2010) implementation in SALib.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCORES_CSV = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
OUT_CSV = RESULTS_DIR / "step_2_5_sobol_weights.csv"

SALTELLI_N = 1024
RANDOM_SEED = 42

BASELINE_TOP3: set[tuple[str, str]] = {
    ("Edge_LC", "LowT_60C"),
    ("Mid_LC", "LowT_60C"),
    ("Hyperscale_LC", "LowT_60C"),
}


def _build_problem() -> dict:
    return {
        "num_vars": 3,
        "names": ["beta", "gamma", "delta"],
        "bounds": [[0.20, 0.60], [0.20, 0.60], [0.00, 0.40]],
    }


def _normalise_weights(samples: np.ndarray) -> np.ndarray:
    """Renormalise each (beta, gamma, delta) triple to sum=1."""
    totals = samples.sum(axis=1, keepdims=True)
    totals = np.where(totals < 1e-12, 1.0, totals)
    return samples / totals


def _score_matrix(
    ri: np.ndarray, ex: np.ndarray, dtc: np.ndarray, weights_norm: np.ndarray
) -> np.ndarray:
    """Return (n_samples, n_pairs) IS-Match Scores, clipped to [0, 1]."""
    beta = weights_norm[:, 0:1]
    gamma = weights_norm[:, 1:2]
    delta = weights_norm[:, 2:3]
    scores = beta * ri + gamma * ex - delta * dtc
    return np.clip(scores, 0.0, 1.0)


def _top3_match(scores_row: np.ndarray, keys: list[tuple[str, str]]) -> bool:
    """True if the set of top-3 (dc, process) keys equals BASELINE_TOP3."""
    top3_idx = np.argpartition(-scores_row, 3)[:3]
    return {keys[i] for i in top3_idx} == BASELINE_TOP3


def _to_csv_rows(metric: str, names: list[str], si: dict | None) -> list[dict]:
    """Emit rows for one metric. If `si` is None, the response was constant
    and Sobol decomposition is undefined; emit zeros with a note column."""
    if si is None:
        return [
            {
                "metric": metric,
                "parameter": name,
                "S1": 0.0,
                "S1_conf": 0.0,
                "ST": 0.0,
                "ST_conf": 0.0,
                "note": "Y is constant; indices undefined (variance=0)",
            }
            for name in names
        ]
    return [
        {
            "metric": metric,
            "parameter": name,
            "S1": round(float(si["S1"][i]), 4),
            "S1_conf": round(float(si["S1_conf"][i]), 4),
            "ST": round(float(si["ST"][i]), 4),
            "ST_conf": round(float(si["ST_conf"][i]), 4),
            "note": "",
        }
        for i, name in enumerate(names)
    ]


def _analyse_or_constant(
    problem: dict, y: np.ndarray, label: str
) -> dict | None:
    """Return Sobol indices, or None if Y has zero variance."""
    if float(y.std()) < 1e-12:
        logger.info("Response '%s' is constant (variance=0); skipping Sobol decomposition", label)
        return None
    return sobol.analyze(
        problem, y, calc_second_order=False, seed=RANDOM_SEED, print_to_console=False
    )


def main() -> None:
    logger.info("Step 2.5 starting: Sobol sensitivity on IS-Match weights")

    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing input: {SCORES_CSV}")
    df = pd.read_csv(SCORES_CSV)
    needed = {"dc_name", "process_name", "ri_temporal", "exergy_dt_norm", "delta_tc_norm"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SCORES_CSV.name}: {sorted(missing)}")

    ri = df["ri_temporal"].to_numpy(dtype=float)
    ex = df["exergy_dt_norm"].to_numpy(dtype=float)
    dtc = df["delta_tc_norm"].to_numpy(dtype=float)
    keys = list(zip(df["dc_name"].astype(str), df["process_name"].astype(str)))
    logger.info("Loaded %d LC pairs from %s", len(df), SCORES_CSV.name)

    problem = _build_problem()
    rng = np.random.default_rng(RANDOM_SEED)
    samples = saltelli.sample(problem, SALTELLI_N, calc_second_order=False)
    logger.info("Saltelli samples generated: shape=%s (N=%d)", samples.shape, SALTELLI_N)

    weights_norm = _normalise_weights(samples)
    scores = _score_matrix(ri, ex, dtc, weights_norm)
    y_aggregate = scores.mean(axis=1)
    y_top3 = np.array([_top3_match(row, keys) for row in scores], dtype=float)
    _ = rng  # rng currently unused; kept to document the seed contract

    si_agg = _analyse_or_constant(problem, y_aggregate, "Y_aggregate")
    si_top3 = _analyse_or_constant(problem, y_top3, "top3_stability_pct")

    rows: list[dict] = []
    rows.extend(_to_csv_rows("Y_aggregate", problem["names"], si_agg))
    rows.extend(_to_csv_rows("top3_stability_pct", problem["names"], si_top3))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(out))

    top3_pct = float(y_top3.mean() * 100.0)

    logger.info("=" * 70)
    if si_agg is not None:
        s1_sum_agg = float(np.sum(si_agg["S1"]))
        s1_beta = float(si_agg["S1"][0])
        s1_beta_conf = float(si_agg["S1_conf"][0])
        s1_gamma = float(si_agg["S1"][1])
        s1_gamma_conf = float(si_agg["S1_conf"][1])
        s1_delta = float(si_agg["S1"][2])
        s1_delta_conf = float(si_agg["S1_conf"][2])
        logger.info("  Sobol Y_aggregate (mean IS-Match Score across 9 pairs):")
        logger.info("    S_beta  = %.3f +/- %.3f", s1_beta, s1_beta_conf)
        logger.info("    S_gamma = %.3f +/- %.3f", s1_gamma, s1_gamma_conf)
        logger.info("    S_delta = %.3f +/- %.3f", s1_delta, s1_delta_conf)
        logger.info("    sum(S1) = %.3f (additivity check, target ~ 1.0)", s1_sum_agg)
        if abs(s1_sum_agg - 1.0) > 0.20:
            logger.warning(
                "Sobol additivity off: sum(S1)=%.3f deviates from 1.0 by more than 0.20",
                s1_sum_agg,
            )
    logger.info("  Top-3 ranking stable in %.1f%% of weight samples", top3_pct)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
