"""
Step 4.5 - Bootstrap CI on PoA and Gap A0 vs D
================================================
Phase 4 robustness layer (additive, no retraining)

Re-uses the per-seed-per-scenario ablation output to compute paired
bootstrap CI95 (10000 resamples, seed 42) on:

  - PoA mean, Shapley fairness mean, convergence rate mean
  - for 4 scenario groups: A0/9LC, A0/S_AIR_M, D/9LC, D/S_AIR_M

and PAIRED bootstrap on the A0-vs-D gap (same resample indices reused
across the two configs so the gap CIs are coherent). Significance flag
is True iff the gap CI95 excludes 0.

Inputs (read-only):
    results/ablation_results.csv   (200 rows = 2 configs x 10 scenarios x 10 seeds)

Outputs:
    results/robustness/step_4_5_bootstrap_ci.csv
    results/robustness/step_4_5_gap_ci.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RESULTS_DIR / "ablation_results.csv"
OUT_PER_GROUP = ROBUSTNESS_DIR / "step_4_5_bootstrap_ci.csv"
OUT_GAP = ROBUSTNESS_DIR / "step_4_5_gap_ci.csv"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5

SCENARIO_GROUPS: dict[str, list[str]] = {
    "9LC": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
    "S_AIR_M": ["S_AIR_M"],
}
METRICS = ("price_of_anarchy", "shapley_fairness", "convergence_rate")
METRIC_LABEL = {
    "price_of_anarchy": "PoA",
    "shapley_fairness": "Shapley_fairness",
    "convergence_rate": "conv_rate",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _extract_vector(df: pd.DataFrame, config: str, scenarios: list[str], metric: str) -> np.ndarray:
    sub = df[(df["config"] == config) & (df["scenario"].isin(scenarios))][metric].to_numpy(dtype=float)
    return sub[np.isfinite(sub)]


def _bootstrap_means(vec: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return vec[indices].mean(axis=1)


def _paired_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def main() -> None:
    logger.info(
        "Step 4.5 starting: paired bootstrap CI on PoA, Shapley_fairness and conv_rate"
    )
    df = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d rows from %s", len(df), INPUT_CSV.name)

    rows_per_group: list[dict] = []
    rows_gap: list[dict] = []

    for group_name, scen_list in SCENARIO_GROUPS.items():
        sub_all = df[df["scenario"].isin(scen_list)]
        n_a0 = int(sub_all[sub_all["config"] == "A0"].shape[0])
        n_d = int(sub_all[sub_all["config"] == "D"].shape[0])
        n_paired = min(n_a0, n_d)
        if n_a0 != n_d:
            logger.warning(
                "Group %s: A0 and D have different sample counts (%d vs %d); "
                "paired bootstrap uses min=%d", group_name, n_a0, n_d, n_paired,
            )
        indices = _paired_indices(n_paired, N_BOOTSTRAP, RANDOM_SEED)

        for metric in METRICS:
            v_a0 = _extract_vector(df, "A0", scen_list, metric)[:n_paired]
            v_d = _extract_vector(df, "D", scen_list, metric)[:n_paired]
            m_a0 = _bootstrap_means(v_a0, indices) if v_a0.size else np.array([np.nan])
            m_d = _bootstrap_means(v_d, indices) if v_d.size else np.array([np.nan])
            for cfg, vec, m in (("A0", v_a0, m_a0), ("D", v_d, m_d)):
                rows_per_group.append({
                    "config": cfg,
                    "scenario_group": group_name,
                    "metric": METRIC_LABEL[metric],
                    "mean": round(float(np.mean(vec)) if vec.size else float("nan"), 4),
                    "std": round(float(np.std(vec, ddof=1)) if vec.size > 1 else 0.0, 4),
                    "ci95_lo": round(float(np.percentile(m, CI_LO_PCT)), 4),
                    "ci95_hi": round(float(np.percentile(m, CI_HI_PCT)), 4),
                    "n_bootstrap": N_BOOTSTRAP,
                    "n_observations": int(vec.size),
                })
            if v_a0.size and v_d.size:
                gap = m_d - m_a0
                lo = float(np.percentile(gap, CI_LO_PCT))
                hi = float(np.percentile(gap, CI_HI_PCT))
                rows_gap.append({
                    "scenario_group": group_name,
                    "metric": METRIC_LABEL[metric],
                    "gap_D_minus_A0_mean": round(float(gap.mean()), 4),
                    "gap_ci95_lo": round(lo, 4),
                    "gap_ci95_hi": round(hi, 4),
                    "significant_at_95": bool(lo > 0 or hi < 0),
                    "n_bootstrap": N_BOOTSTRAP,
                })

    df_per = pd.DataFrame(rows_per_group)
    df_per.to_csv(OUT_PER_GROUP, index=False)
    logger.info("Wrote %s (%d rows)", OUT_PER_GROUP.name, len(df_per))

    df_gap = pd.DataFrame(rows_gap)
    df_gap.to_csv(OUT_GAP, index=False)
    logger.info("Wrote %s (%d rows)", OUT_GAP.name, len(df_gap))

    logger.info("=" * 70)
    for _, r in df_gap.iterrows():
        sig = "sig" if r["significant_at_95"] else "ns"
        logger.info(
            "  %s %s gap D-A0 = %+.4f  CI95 [%+.4f, %+.4f]  (%s)",
            r["scenario_group"], r["metric"],
            r["gap_D_minus_A0_mean"], r["gap_ci95_lo"], r["gap_ci95_hi"], sig,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
