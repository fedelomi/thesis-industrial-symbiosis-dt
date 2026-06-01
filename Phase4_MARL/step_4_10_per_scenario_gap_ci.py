"""
step_4_10_per_scenario_gap_ci.py — Per-scenario paired bootstrap CI on PoA / Shapley gaps.

Inputs:  results/ablation_results.csv (200 rows: 2 configs x 10 scenarios x 10 seeds)
Outputs: results/robustness/step_4_10_per_scenario_gap_ci.csv
         results/robustness/step_4_10_per_subgroup_gap_ci.csv
Gate:    none (informational; complements step_4_5 aggregated gap CI)

Goal: complement the aggregated gap CI (step_4_5) which collapses all 9 LC
scenarios into a single estimate. Here we keep scenarios separated and ask
whether D vs A0 differences are localised in a sub-population. We also
aggregate by DC scale (Edge / Mid / Hyperscale) and by manufacturing
temperature band (LowT / MidT / HighT) since the 9 LC scenarios sit on a
3x3 grid (D2 scope decision in implementation-decisions).

Per-scenario sample size is n=10 (one paired diff per seed). Per-subgroup
n=30 (3 scenarios x 10 seeds). The 9LC aggregated reference is n=90.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "robustness"

# 3 x 3 grid of LC scenarios, per D2 scope decision.
# Convention: S1..S9 indexed as (scale, T_band) with scale in {Edge, Mid, Hyperscale}
# and T_band in {LowT, MidT, HighT}.
SCALE_GROUPS: dict[str, tuple[str, ...]] = {
    "Edge_LC": ("S1", "S2", "S3"),
    "Mid_LC": ("S4", "S5", "S6"),
    "Hyperscale_LC": ("S7", "S8", "S9"),
}
TBAND_GROUPS: dict[str, tuple[str, ...]] = {
    "LowT_60C": ("S1", "S4", "S7"),
    "MidT_90C": ("S2", "S5", "S8"),
    "HighT_130C": ("S3", "S6", "S9"),
}
NINE_LC: tuple[str, ...] = tuple(f"S{i}" for i in range(1, 10))


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    seed: int = 42
    skip_existing: bool = False
    n_bootstrap: int = 10000
    ci_level: float = 0.95
    metrics: tuple[str, ...] = ("price_of_anarchy", "shapley_fairness", "convergence_rate")


def _paired_diffs(
    df: pd.DataFrame, metric: str, scenarios: tuple[str, ...]
) -> np.ndarray:
    """
    Per-(scenario, seed) paired differences D - A0 for scenarios in the tuple.
    Returns 1D array of length n_seeds * n_scenarios.
    """
    sub = df[df["scenario"].isin(scenarios)].copy()
    pivot = sub.pivot_table(
        index=["scenario", "seed"], columns="config", values=metric, aggfunc="first"
    )
    pivot = pivot.dropna(subset=["A0", "D"])
    return (pivot["D"] - pivot["A0"]).to_numpy()


def _paired_bootstrap_ci(
    diffs: np.ndarray, n_bootstrap: int, ci_level: float, rng: np.random.Generator
) -> tuple[float, float, float, float, bool]:
    """
    Paired bootstrap CI on the mean of `diffs`.

    Returns (mean, ci_lo, ci_hi, std, significant_at_ci_level).
    Significance: CI does not contain zero.
    """
    n = diffs.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), False
    if n == 1:
        return float(diffs[0]), float("nan"), float("nan"), 0.0, False

    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()

    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    mean = float(diffs.mean())
    sd = float(boot_means.std(ddof=1))
    sig = (lo > 0.0) or (hi < 0.0)  # CI does not contain zero
    return mean, lo, hi, sd, bool(sig)


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the per-scenario gap CI dataframe."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "step_4_10_per_scenario_gap_ci.csv"
    sub_path = OUT_DIR / "step_4_10_per_subgroup_gap_ci.csv"

    if cfg.skip_existing and out_path.exists() and sub_path.exists():
        LOG.info("Skipping (outputs exist): %s, %s", out_path, sub_path)
        return pd.read_csv(out_path)

    src = RESULTS / "ablation_results.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing input: {src}")
    df = pd.read_csv(src)
    LOG.info("Loaded %d rows from %s", len(df), src)

    rng = np.random.default_rng(cfg.seed)

    # 1. Per-scenario CI for the 9 LC scenarios
    per_scenario: list[dict] = []
    for scen in NINE_LC:
        for metric in cfg.metrics:
            diffs = _paired_diffs(df, metric, (scen,))
            mean, lo, hi, sd, sig = _paired_bootstrap_ci(
                diffs, cfg.n_bootstrap, cfg.ci_level, rng
            )
            per_scenario.append(
                {
                    "scenario": scen,
                    "metric": metric,
                    "n_paired": int(diffs.size),
                    "gap_D_minus_A0_mean": round(mean, 6),
                    "ci95_lo": round(lo, 6) if np.isfinite(lo) else float("nan"),
                    "ci95_hi": round(hi, 6) if np.isfinite(hi) else float("nan"),
                    "bootstrap_std": round(sd, 6),
                    "significant_at_95": sig,
                    "n_bootstrap": cfg.n_bootstrap,
                }
            )

    per_scenario_df = pd.DataFrame(per_scenario)

    # 2. Per-subgroup CI (DC scale + T band)
    per_subgroup: list[dict] = []

    def _add_subgroup(
        partition: str, label: str, scenarios: tuple[str, ...]
    ) -> None:
        for metric in cfg.metrics:
            diffs = _paired_diffs(df, metric, scenarios)
            mean, lo, hi, sd, sig = _paired_bootstrap_ci(
                diffs, cfg.n_bootstrap, cfg.ci_level, rng
            )
            per_subgroup.append(
                {
                    "partition": partition,
                    "subgroup": label,
                    "scenarios": " ".join(scenarios),
                    "metric": metric,
                    "n_paired": int(diffs.size),
                    "gap_D_minus_A0_mean": round(mean, 6),
                    "ci95_lo": round(lo, 6) if np.isfinite(lo) else float("nan"),
                    "ci95_hi": round(hi, 6) if np.isfinite(hi) else float("nan"),
                    "bootstrap_std": round(sd, 6),
                    "significant_at_95": sig,
                    "n_bootstrap": cfg.n_bootstrap,
                }
            )

    for label, scenarios in SCALE_GROUPS.items():
        _add_subgroup("DC_scale", label, scenarios)
    for label, scenarios in TBAND_GROUPS.items():
        _add_subgroup("T_band", label, scenarios)

    per_subgroup_df = pd.DataFrame(per_subgroup)

    per_scenario_df.to_csv(out_path, index=False)
    per_subgroup_df.to_csv(sub_path, index=False)

    LOG.info("Wrote %d per-scenario rows to %s", len(per_scenario_df), out_path)
    LOG.info("Wrote %d per-subgroup rows to %s", len(per_subgroup_df), sub_path)

    _print_summary(per_scenario_df, per_subgroup_df)
    return per_scenario_df


def _print_summary(per_scenario_df: pd.DataFrame, per_subgroup_df: pd.DataFrame) -> None:
    LOG.info("=" * 78)
    LOG.info("PER-SCENARIO GAP CI - significant findings only (CI does not cross 0)")
    LOG.info("=" * 78)
    sig = per_scenario_df[per_scenario_df["significant_at_95"]]
    if sig.empty:
        LOG.info("(none - no per-scenario gap reaches significance with n=10)")
    else:
        for _, row in sig.iterrows():
            LOG.info(
                "%s | %s | gap=%+.4f CI [%+.4f, %+.4f] sig",
                row["scenario"],
                row["metric"],
                row["gap_D_minus_A0_mean"],
                row["ci95_lo"],
                row["ci95_hi"],
            )

    LOG.info("=" * 78)
    LOG.info("PER-SUBGROUP GAP CI - all findings")
    LOG.info("=" * 78)
    for partition in ["DC_scale", "T_band"]:
        for _, row in per_subgroup_df[per_subgroup_df["partition"] == partition].iterrows():
            flag = "SIG" if row["significant_at_95"] else "ns"
            LOG.info(
                "%-8s | %-12s | %-18s | gap=%+.4f CI [%+.4f, %+.4f] %s",
                partition,
                row["subgroup"],
                row["metric"],
                row["gap_D_minus_A0_mean"],
                row["ci95_lo"],
                row["ci95_hi"],
                flag,
            )


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--ci-level", type=float, default=0.95)
    a = p.parse_args()
    return Config(
        seed=a.seed,
        skip_existing=a.skip_existing,
        n_bootstrap=a.n_bootstrap,
        ci_level=a.ci_level,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
