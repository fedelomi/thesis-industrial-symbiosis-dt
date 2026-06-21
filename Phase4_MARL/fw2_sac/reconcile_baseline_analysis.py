"""reconcile_baseline_analysis.py - re-analysis of the FW2 2x2 ablation against
the CANONICAL S_AIR_M baseline (no Modal run, reads existing CSVs only).

The canonical shielded D baseline on S_AIR_M is conv 0.4067, CI95 [0.255, 0.558],
n=10 seeds x 100k (ablation_results.csv per-seed, ablation_results_ci95.csv
aggregated, thesis Tab 5.8). The bare "0.41" cited in the FW2 docstrings IS that
10-seed mean, so it is a valid canonical figure. (An earlier pass mistakenly
compared against a superseded 5-seed x 50k run, ablation_results_5seeds_50k_*,
which gives 0.533; this script uses the canonical 10-seed file.)

The FW2 rich harness (n_steps=2048, n_epochs=10, 50k, n_envs=1) lifts even plain
PPO_nocurr to 0.853, far above the canonical 0.41. This script quantifies, with
Welch t-tests, CI overlap and Cohen's d, how much of the gain is harness vs
algorithm vs curriculum, plus a power analysis on the observed effects.

It reads only existing result files and writes nothing.

Sources:
  rich FW2 run   results/modal_fw2_all_5seed_50k.csv          (n=5, 50k, n_steps=2048, n_epochs=10, eval=30, n_envs=1)
  canonical D/A0 results/ablation_results.csv                 (n=10 x 100k, S_AIR_M; CI in ablation_results_ci95.csv, Tab 5.8)
  step_4_6 LOO   results/robustness/step_4_6_shielding_loo.csv (n=3, light config 512/5, 50k, raw env, n_envs=2)

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger("fw2.reconcile")

HERE = Path(__file__).resolve().parent
PHASE4_ROOT = HERE.parent
RESULTS = PHASE4_ROOT / "results"

RICH_CSV = RESULTS / "modal_fw2_all_5seed_50k.csv"
CANON_CSV = RESULTS / "ablation_results.csv"  # canonical 10-seed x 100k (Tab 5.8)
LOO_CSV = RESULTS / "robustness" / "step_4_6_shielding_loo.csv"

TARGET = "S_AIR_M"
CANON_MEAN = 0.41  # canonical D 10-seed mean (0.4067), thesis Tab 5.8
ALPHA = 0.05
TARGET_POWER = 0.80


@dataclass(frozen=True)
class Summary:
    """Descriptive stats for one group of per-seed convergence rates."""

    label: str
    n: int
    mean: float
    std: float
    ci_half: float

    @property
    def lo(self) -> float:
        return self.mean - self.ci_half

    @property
    def hi(self) -> float:
        return self.mean + self.ci_half


@dataclass(frozen=True)
class Welch:
    """Welch (unequal variance) two-sample comparison with effect size."""

    a_label: str
    b_label: str
    diff: float
    t_stat: float
    df: float
    p_val: float
    cohen_d: float
    ci_overlap: bool


def summarise(label: str, values: np.ndarray) -> Summary:
    """Mean, std (ddof=1) and 95% CI half-width via Student t for one group."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return Summary(label, 0, float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(v))
    if n < 2:
        return Summary(label, n, mean, 0.0, 0.0)
    std = float(np.std(v, ddof=1))
    ci_half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return Summary(label, n, mean, std, ci_half)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d for two independent groups."""
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp_sq = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    sp = float(np.sqrt(sp_sq))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def welch(a_sum: Summary, a: np.ndarray, b_sum: Summary, b: np.ndarray) -> Welch:
    """Welch t-test between two groups plus Cohen's d and a CI-overlap flag."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite df for reporting.
    va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    overlap = not (a_sum.lo > b_sum.hi or b_sum.lo > a_sum.hi)
    return Welch(a_sum.label, b_sum.label, float(a_sum.mean - b_sum.mean),
                 float(t_stat), float(df), float(p_val), _cohen_d(a, b), overlap)


def required_n(effect_size: float, *, alpha: float = ALPHA, power: float = TARGET_POWER) -> float:
    """Per-group n for a two-sided two-sample t-test at the given power.

    Uses statsmodels TTestIndPower (Cohen's d convention, equal-n groups).
    """
    from statsmodels.stats.power import TTestIndPower

    return float(TTestIndPower().solve_power(
        effect_size=abs(effect_size), alpha=alpha, power=power, alternative="two-sided"))


def load_groups() -> dict[str, np.ndarray]:
    """Read the three result files and return label -> per-seed conv-rate arrays."""
    rich = pd.read_csv(RICH_CSV)
    canon = pd.read_csv(CANON_CSV)
    loo = pd.read_csv(LOO_CSV)

    groups: dict[str, np.ndarray] = {}
    for cfg in ("SAC_curr", "SAC_nocurr", "PPO_curr", "PPO_nocurr"):
        groups[f"FW2_{cfg}"] = rich.loc[rich["config"] == cfg, "convergence_rate"].to_numpy()

    canon_air = canon[canon["scenario"] == TARGET]
    groups["canon_D"] = canon_air.loc[canon_air["config"] == "D", "convergence_rate"].to_numpy()
    groups["canon_A0"] = canon_air.loc[canon_air["config"] == "A0", "convergence_rate"].to_numpy()

    base = loo[(loo["rule_disabled"] == "NONE_baseline") & (loo["scenario"] == TARGET)]
    groups["step_4_6_base"] = base["conv_rate"].to_numpy()
    return groups


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    groups = load_groups()
    summ = {k: summarise(k, v) for k, v in groups.items()}

    print("=" * 78)
    print("FW2 ablation re-analysis vs CANONICAL S_AIR_M baseline (no Modal run)")
    print("=" * 78)
    print("\nCanonical shielded D baseline (Tab 5.8): conv "
          f"{summ['canon_D'].mean:.4f}, CI95 [{summ['canon_D'].lo:.3f}, {summ['canon_D'].hi:.3f}], "
          f"n={summ['canon_D'].n} x 100k.")
    print(f"step_4_6 light-config 50k (n={summ['step_4_6_base'].n}) = "
          f"{summ['step_4_6_base'].mean:.3f}. The rich FW2 harness lifts PPO_nocurr to "
          f"{summ['FW2_PPO_nocurr'].mean:.3f}.")

    order = ["FW2_SAC_curr", "FW2_PPO_nocurr", "FW2_SAC_nocurr", "FW2_PPO_curr",
             "canon_D", "canon_A0", "step_4_6_base"]
    print(f"\n{'group':>16} {'n':>3} {'mean':>7} {'std':>7}   {'CI95':>20}")
    print("-" * 60)
    for k in order:
        s = summ[k]
        print(f"{k:>16} {s.n:>3} {s.mean:>7.3f} {s.std:>7.3f}   "
              f"[{s.lo:>6.3f}, {s.hi:>6.3f}]")

    print("\n" + "=" * 78)
    print("Key comparisons (Welch t-test, unequal variance)")
    print("=" * 78)
    pairs = [
        ("FW2_PPO_nocurr", "canon_D",
         "harness effect: rich PPO_nocurr vs canonical D (same algo, no curriculum)"),
        ("FW2_SAC_curr", "canon_D",
         "FW2 SAC+curr vs canonical shielded baseline (the headline claim)"),
        ("FW2_SAC_curr", "FW2_PPO_nocurr",
         "apples-to-apples inside the rich harness: SAC+curr vs PPO-no-curr"),
        ("FW2_PPO_curr", "FW2_PPO_nocurr",
         "curriculum effect under PPO inside the rich harness"),
        ("FW2_SAC_nocurr", "FW2_PPO_nocurr",
         "algorithm effect with no curriculum inside the rich harness"),
    ]
    welch_results: dict[tuple[str, str], Welch] = {}
    for a_key, b_key, desc in pairs:
        w = welch(summ[a_key], groups[a_key], summ[b_key], groups[b_key])
        welch_results[(a_key, b_key)] = w
        sig = "SIGNIFICANT (p<0.05)" if w.p_val < 0.05 else "not significant"
        ov = "CI OVERLAP" if w.ci_overlap else "CI separated"
        print(f"\n  {desc}")
        print(f"    {w.a_label} ({summ[a_key].mean:.3f}) vs {w.b_label} ({summ[b_key].mean:.3f})"
              f"  diff={w.diff:+.3f}")
        print(f"    t={w.t_stat:+.2f}  df={w.df:.1f}  p={w.p_val:.4f}  d={w.cohen_d:+.2f}"
              f"   -> {sig}, {ov}")

    print("\n" + "=" * 78)
    print(f"Power analysis: required n per cell ({TARGET_POWER:.0%} power, alpha {ALPHA}, two-sided)")
    print("=" * 78)
    print("  Effect sizes are the Cohen's d observed above; n is rounded UP (power is a floor).")
    power_pairs = [
        ("FW2_SAC_curr", "FW2_PPO_nocurr", "algorithm + curriculum inside the harness"),
        ("FW2_SAC_curr", "canon_D", "FW2 vs canonical shielded baseline"),
        ("FW2_PPO_nocurr", "canon_D", "PPO harness effect (what the data really shows)"),
    ]
    for a_key, b_key, desc in power_pairs:
        w = welch_results[(a_key, b_key)]
        n_exact = required_n(w.cohen_d)
        print(f"\n  {desc}")
        print(f"    {a_key} vs {b_key}  d={w.cohen_d:+.2f}"
              f"  -> n per cell = {math.ceil(n_exact)} (exact {n_exact:.1f}); current n=5")

    print("\n" + "=" * 78)
    print(f"One-sample check vs canonical mean {CANON_MEAN} (rigorous version is the Welch above)")
    print("=" * 78)
    a = groups["FW2_SAC_curr"]
    t1, p1 = stats.ttest_1samp(a, CANON_MEAN)
    print(f"  ttest_1samp(SAC_curr, {CANON_MEAN}): t={t1:.2f}, p={p1:.4f}")
    print(f"  ^ 0.41 is the canonical 10-seed mean; the Welch vs the 10 canonical seeds "
          f"(CI95 [{summ['canon_D'].lo:.3f}, {summ['canon_D'].hi:.3f}]) is the sound test.")


if __name__ == "__main__":
    main()
