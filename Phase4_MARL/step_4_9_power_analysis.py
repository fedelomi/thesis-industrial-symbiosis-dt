"""
step_4_9_power_analysis.py — Effect size + ex-post power analysis on PoA / Shapley gap.

Inputs:  results/ablation_results.csv (200 rows: 2 configs x 10 scenarios x 10 seeds)
Outputs: results/robustness/step_4_9_power_analysis.csv
         results/robustness/step_4_9_required_n.csv
Gate:    none (informational)

Goal: convert the non-significant PoA gap (+0.030, CI95 [-0.018, 0.078]) from a
weakness into a methodologically-anchored finding. We compute Cohen's d on the
observed paired differences and the ex-post statistical power for the observed
effect size at the current sample size. We also compute the n that would be
required to reach 80 percent power for the observed effect, demonstrating that
the test was underpowered for an effect of this magnitude. This is the standard
defense for a non-significant finding when the null hypothesis is itself the
desired outcome (non-destructive shielding).

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "robustness"


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    seed: int = 42
    skip_existing: bool = False
    alpha: float = 0.05
    target_power: float = 0.80
    metrics: tuple[str, ...] = ("price_of_anarchy", "shapley_fairness", "convergence_rate")
    scenario_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("9LC", ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")),
        ("S_AIR_M", ("S_AIR_M",)),
    )


def _cohens_d_paired(diffs: np.ndarray) -> tuple[float, float]:
    """Cohen's d on paired differences. Returns (d, std_of_diffs)."""
    if diffs.size < 2:
        return float("nan"), float("nan")
    sd = float(diffs.std(ddof=1))
    d = float(diffs.mean() / sd) if sd > 0 else float("inf") if diffs.mean() != 0 else 0.0
    return d, sd


def _power_paired_t(d_effect: float, n: int, alpha: float = 0.05) -> float:
    """
    Power for a paired t-test (two-sided), non-central t formulation.

    H0: mean_diff = 0
    H1: mean_diff != 0
    Effect size: Cohen's d on paired differences.
    """
    if not np.isfinite(d_effect) or n < 2:
        return float("nan")
    nc = d_effect * np.sqrt(n)             # non-centrality
    df = n - 1
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    # P(reject H0 | H1) = P(|T| > crit) under noncentral t
    pwr = (
        1.0
        - stats.nct.cdf(crit, df, nc)
        + stats.nct.cdf(-crit, df, nc)
    )
    return float(np.clip(pwr, 0.0, 1.0))


def _required_n_paired_t(
    d_effect: float, target_power: float = 0.80, alpha: float = 0.05, n_max: int = 5000
) -> int:
    """Smallest n such that paired t-test power >= target_power for the given d."""
    if not np.isfinite(d_effect) or d_effect == 0:
        return -1  # effect is zero, required n is infinite
    for n in range(2, n_max + 1):
        if _power_paired_t(abs(d_effect), n, alpha) >= target_power:
            return n
    return n_max + 1  # exceeded budget


def _paired_diffs(df: pd.DataFrame, metric: str, scenarios: tuple[str, ...]) -> np.ndarray:
    """
    Build per-(scenario, seed) paired differences D - A0 over the provided scenarios.
    """
    sub = df[df["scenario"].isin(scenarios)].copy()
    pivot = sub.pivot_table(
        index=["scenario", "seed"], columns="config", values=metric, aggfunc="first"
    )
    pivot = pivot.dropna(subset=["A0", "D"])
    return (pivot["D"] - pivot["A0"]).to_numpy()


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the power analysis dataframe."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "step_4_9_power_analysis.csv"
    n_path = OUT_DIR / "step_4_9_required_n.csv"

    if cfg.skip_existing and out_path.exists() and n_path.exists():
        LOG.info("Skipping (outputs exist): %s, %s", out_path, n_path)
        return pd.read_csv(out_path)

    src = RESULTS / "ablation_results.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing input: {src}")
    df = pd.read_csv(src)
    LOG.info("Loaded %d rows from %s", len(df), src)

    rows: list[dict] = []
    n_rows: list[dict] = []

    for group_name, scen_tuple in cfg.scenario_groups:
        for metric in cfg.metrics:
            diffs = _paired_diffs(df, metric, scen_tuple)
            n = int(diffs.size)

            if n < 2:
                LOG.warning("Skipping group=%s metric=%s (n=%d)", group_name, metric, n)
                continue

            mean_diff = float(diffs.mean())
            d, sd = _cohens_d_paired(diffs)

            # observed power for the observed effect at the observed n
            power_obs = _power_paired_t(d, n, cfg.alpha)

            # required n to reach 80 percent power for the observed effect
            n_req_80 = _required_n_paired_t(d, cfg.target_power, cfg.alpha)

            # sanity reference: power if the effect were Cohen's d = 0.5 (medium)
            power_medium = _power_paired_t(0.5, n, cfg.alpha)
            n_req_medium = _required_n_paired_t(0.5, cfg.target_power, cfg.alpha)

            interp = _interpret_d(d)

            rows.append(
                {
                    "scenario_group": group_name,
                    "metric": metric,
                    "n_paired_obs": n,
                    "mean_gap_D_minus_A0": round(mean_diff, 6),
                    "std_paired_diffs": round(sd, 6),
                    "cohens_d": round(d, 4),
                    "cohens_d_interpretation": interp,
                    "alpha": cfg.alpha,
                    "observed_power_at_observed_d": round(power_obs, 4),
                    "observed_power_at_d_medium_0p5": round(power_medium, 4),
                    "target_power": cfg.target_power,
                    "n_required_for_observed_d": n_req_80,
                    "n_required_for_d_medium_0p5": n_req_medium,
                    "underpowered_for_observed_effect": power_obs < cfg.target_power,
                }
            )

            n_rows.append(
                {
                    "scenario_group": group_name,
                    "metric": metric,
                    "current_n": n,
                    "observed_cohens_d": round(d, 4),
                    "current_power": round(power_obs, 4),
                    "n_for_power_0p50": _required_n_paired_t(d, 0.50, cfg.alpha),
                    "n_for_power_0p80": _required_n_paired_t(d, 0.80, cfg.alpha),
                    "n_for_power_0p90": _required_n_paired_t(d, 0.90, cfg.alpha),
                    "extra_seeds_to_reach_0p80_per_scenario": _seeds_required(
                        _required_n_paired_t(d, 0.80, cfg.alpha), len(scen_tuple), n
                    ),
                }
            )

    out_df = pd.DataFrame(rows)
    n_df = pd.DataFrame(n_rows)

    out_df.to_csv(out_path, index=False)
    n_df.to_csv(n_path, index=False)

    LOG.info("Wrote %d power rows to %s", len(out_df), out_path)
    LOG.info("Wrote %d required-n rows to %s", len(n_df), n_path)

    _print_summary(out_df)

    return out_df


def _interpret_d(d: float) -> str:
    """Cohen 1988 conventions for paired-difference d."""
    if not np.isfinite(d):
        return "undefined"
    a = abs(d)
    if a < 0.20:
        return "negligible (<0.20)"
    if a < 0.50:
        return "small (0.20-0.50)"
    if a < 0.80:
        return "medium (0.50-0.80)"
    return "large (>=0.80)"


def _seeds_required(n_required: int, n_scenarios: int, current_n: int) -> int:
    """Translate required n (paired obs) into extra seeds per scenario."""
    if n_required < 0 or n_required > 10000:
        return -1
    extra_obs = max(0, n_required - current_n)
    if n_scenarios <= 0:
        return -1
    # ceil division
    return int((extra_obs + n_scenarios - 1) // n_scenarios)


def _print_summary(df: pd.DataFrame) -> None:
    LOG.info("=" * 70)
    LOG.info("POWER ANALYSIS SUMMARY")
    LOG.info("=" * 70)
    for _, row in df.iterrows():
        LOG.info(
            "%s | %s | n=%d | d=%.4f (%s) | gap=%+.4f | power=%.3f | "
            "n_req=%s",
            row["scenario_group"],
            row["metric"],
            row["n_paired_obs"],
            row["cohens_d"],
            row["cohens_d_interpretation"],
            row["mean_gap_D_minus_A0"],
            row["observed_power_at_observed_d"],
            row["n_required_for_observed_d"],
        )


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--target-power", type=float, default=0.80)
    a = p.parse_args()
    return Config(
        seed=a.seed,
        skip_existing=a.skip_existing,
        alpha=a.alpha,
        target_power=a.target_power,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
