"""Smoke test for Phase 4 - Agentic Negotiation Layer (MARL).

Loads the canonical ablation CSV (ablation_results.csv) and re-runs a
lightweight BCa bootstrap (1000 resamples) on the paired D-minus-A0 gap for
the 9 LC scenarios. Asserts that the recomputed PoA gap mean and its 95% CI
are consistent with the values published in thesis Tables 5.8 / 5.9.

Offline-only: no PPO training, no Gymnasium env, no Modal cloud runs.
Runtime: <5 seconds (1000 bootstrap resamples on n=90 paired observations).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PHASE4_RESULTS = Path(__file__).resolve().parents[1] / "results"
ABLATION_CSV = PHASE4_RESULTS / "ablation_results.csv"
ROBUSTNESS_DIR = PHASE4_RESULTS / "robustness"
GAP_BCA_CSV = ROBUSTNESS_DIR / "step_4_5_gap_ci_bca.csv"

# Published thesis values (Table 5.8 / 5.9, canonical BCa 10000-resample)
PUBLISHED_POA_GAP_MEAN = 0.030
PUBLISHED_POA_GAP_CI_LO = -0.0179
PUBLISHED_POA_GAP_CI_HI = 0.0779

PUBLISHED_SHAPLEY_GAP_MEAN = -0.2078
PUBLISHED_SHAPLEY_GAP_CI_LO = -0.2877
PUBLISHED_SHAPLEY_GAP_CI_HI = -0.1386

N_BOOTSTRAP_SMOKE = 1000
BOOTSTRAP_SEED = 42
LC_SCENARIOS = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"}


def _load_ablation() -> list[dict[str, str]]:
    with ABLATION_CSV.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _paired_gap_vector(rows: list[dict[str, str]], metric: str, scenarios: set[str]) -> np.ndarray:
    """Return per-(seed, scenario) paired gap D - A0 for the given scenario set."""
    d_vals: dict[tuple[str, str], float] = {}
    a0_vals: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["scenario"] not in scenarios:
            continue
        key = (row["seed"], row["scenario"])
        val = float(row[metric])
        if row["config"] == "D":
            d_vals[key] = val
        elif row["config"] == "A0":
            a0_vals[key] = val
    common = sorted(set(d_vals) & set(a0_vals))
    return np.array([d_vals[k] - a0_vals[k] for k in common])


def _bca_ci(theta_hat: float, boot: np.ndarray, jackknife: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if np.allclose(boot, boot[0]):
        return float(boot[0]), float(boot[0])
    prop = float(np.sum(boot < theta_hat)) / boot.size
    prop = min(max(prop, 1.0 / boot.size), 1.0 - 1.0 / boot.size)
    z0 = stats.norm.ppf(prop)
    diff = jackknife.mean() - jackknife
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    accel = float(np.sum(diff ** 3) / denom) if denom != 0.0 else 0.0
    z_lo, z_hi = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    return float(np.percentile(boot, 100.0 * a1)), float(np.percentile(boot, 100.0 * a2))


def _run_bootstrap_gap(diffs: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via BCa bootstrap."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = diffs.size
    theta_hat = float(diffs.mean())
    boot = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(N_BOOTSTRAP_SMOKE)])
    jk = (diffs.sum() - diffs) / (n - 1)
    lo, hi = _bca_ci(theta_hat, boot, jk)
    return theta_hat, lo, hi


# ---- tests ----

def test_ablation_csv_exists() -> None:
    assert ABLATION_CSV.exists(), f"Canonical ablation CSV not found: {ABLATION_CSV}"


def test_ablation_row_count() -> None:
    rows = _load_ablation()
    assert len(rows) == 200, f"Expected 200 rows (2 configs x 10 seeds x 10 scenarios), got {len(rows)}"


def test_required_columns() -> None:
    rows = _load_ablation()
    required = {"config", "seed", "scenario", "price_of_anarchy", "shapley_fairness", "convergence_rate"}
    actual = set(rows[0].keys())
    missing = required - actual
    assert not missing, f"Missing columns in ablation CSV: {missing}"


def test_poa_gap_mean_matches_published() -> None:
    rows = _load_ablation()
    diffs = _paired_gap_vector(rows, "price_of_anarchy", LC_SCENARIOS)
    assert len(diffs) == 90, f"Expected 90 paired observations on 9LC arm, got {len(diffs)}"
    observed_mean = float(diffs.mean())
    assert abs(observed_mean - PUBLISHED_POA_GAP_MEAN) < 0.002, (
        f"PoA gap mean {observed_mean:.4f} differs from published {PUBLISHED_POA_GAP_MEAN:.4f}"
    )


def test_poa_gap_bca_ci_overlaps_published() -> None:
    """BCa CI from 1000 resamples must overlap with the thesis canonical CI95 BCa (10k resamples)."""
    rows = _load_ablation()
    diffs = _paired_gap_vector(rows, "price_of_anarchy", LC_SCENARIOS)
    mean, lo, hi = _run_bootstrap_gap(diffs)
    # Overlap condition: smoke CI must intersect [PUBLISHED_CI_LO, PUBLISHED_CI_HI]
    assert lo <= PUBLISHED_POA_GAP_CI_HI, (
        f"Smoke BCa CI lower {lo:.4f} exceeds published CI upper {PUBLISHED_POA_GAP_CI_HI}"
    )
    assert hi >= PUBLISHED_POA_GAP_CI_LO, (
        f"Smoke BCa CI upper {hi:.4f} below published CI lower {PUBLISHED_POA_GAP_CI_LO}"
    )


def test_shapley_gap_significant() -> None:
    """Shapley fairness gap on 9LC must be negative and significant (thesis Table 5.9)."""
    rows = _load_ablation()
    diffs = _paired_gap_vector(rows, "shapley_fairness", LC_SCENARIOS)
    mean, lo, hi = _run_bootstrap_gap(diffs)
    assert mean < 0, f"Shapley gap mean {mean:.4f} should be negative"
    assert abs(mean - PUBLISHED_SHAPLEY_GAP_MEAN) < 0.01, (
        f"Shapley gap mean {mean:.4f} differs from published {PUBLISHED_SHAPLEY_GAP_MEAN:.4f}"
    )
    assert hi < 0, (
        f"Shapley gap BCa CI upper {hi:.4f} not below zero; expected significant negative gap"
    )


def test_gap_bca_csv_exists() -> None:
    assert GAP_BCA_CSV.exists(), (
        f"Canonical gap BCa CSV not found: {GAP_BCA_CSV}\n"
        "Run step_4_12_bca_holm_cohen.py to generate it."
    )


def test_gap_bca_csv_poa_row() -> None:
    if not GAP_BCA_CSV.exists():
        pytest.skip("Gap BCa CSV not present; run step_4_12 first")
    with GAP_BCA_CSV.open(encoding="utf-8") as fh:
        rows = {(r["scenario_group"], r["metric"]): r for r in csv.DictReader(fh)}
    key = ("9LC", "price_of_anarchy")
    assert key in rows, f"Row ('9LC', 'price_of_anarchy') missing from {GAP_BCA_CSV.name}"
    lo = float(rows[key]["ci95_lo_bca"])
    hi = float(rows[key]["ci95_hi_bca"])
    assert abs(lo - PUBLISHED_POA_GAP_CI_LO) < 0.005, (
        f"Committed CI95 lo {lo:.4f} differs from published {PUBLISHED_POA_GAP_CI_LO}"
    )
    assert abs(hi - PUBLISHED_POA_GAP_CI_HI) < 0.005, (
        f"Committed CI95 hi {hi:.4f} differs from published {PUBLISHED_POA_GAP_CI_HI}"
    )
