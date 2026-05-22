"""
Unit tests for Phase 2 robustness layer (step_2_5 to step_2_8).

Each test reads the canonical CSV produced by the corresponding module
and asserts the sanity bounds documented in the module docstrings.
The tests assume the modules have already run; in CI the orchestrator
or a `make_results` fixture should produce the CSVs first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SOBOL_CSV = RESULTS_DIR / "step_2_5_sobol_weights.csv"
STRESS_CSV = RESULTS_DIR / "step_2_6_stress_test.csv"
CARBON_CSV = RESULTS_DIR / "step_2_7_carbon_emissions_avoided.csv"
PROMETHEE_CORR_CSV = RESULTS_DIR / "step_2_8_promethee_correlation.csv"

PROMETHEE_SPEARMAN_MIN = 0.60


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Required CSV not produced yet: {path.name}")
    return pd.read_csv(path)


def test_sobol_indices_additivity() -> None:
    """Sum of first-order Sobol indices for Y_aggregate should be ~ 1.0
    (additive model with no large interaction effects)."""
    df = _require(SOBOL_CSV)
    agg = df[df["metric"] == "Y_aggregate"]
    assert len(agg) == 3, f"Expected 3 parameters for Y_aggregate, got {len(agg)}"
    s1_sum = float(agg["S1"].sum())
    assert 0.85 <= s1_sum <= 1.15, (
        f"Sobol additivity off: sum(S1)={s1_sum:.3f} not in [0.85, 1.15]"
    )


def test_stress_top3_stability() -> None:
    """At least 6/8 perturbation scenarios should preserve the top-3 set."""
    df = _require(STRESS_CSV)
    assert len(df) == 8, f"Expected 8 stress scenarios, got {len(df)}"
    n_pass = int(df["top3_unchanged"].sum())
    assert n_pass >= 6, (
        f"Top-3 stability broken: only {n_pass}/8 scenarios preserved the set"
    )


def test_carbon_magnitude_mid_lowt() -> None:
    """Mid_LC + LowT_60C CO2 avoided should be in the [1000, 3000] tCO2/yr range
    (sanity bound from 2shift hours x ri_temporal x EU EEA 2024 factor)."""
    df = _require(CARBON_CSV)
    row = df[(df["dc_name"] == "Mid_LC") & (df["process_name"] == "LowT_60C")]
    assert not row.empty, "Mid_LC + LowT_60C pair missing from carbon CSV"
    co2 = float(row.iloc[0]["CO2_avoided_tons_yr"])
    assert 1000.0 <= co2 <= 3000.0, (
        f"Mid_LC + LowT_60C CO2 avoided = {co2:.0f} tCO2/yr outside [1000, 3000]"
    )


def test_promethee_correlation() -> None:
    """Spearman rho between IS-Match and PROMETHEE II should be >= 0.60.
    Tolerant lower bound: the 4th CO2 criterion shifts ranking toward
    Hyperscale_LC (documented behaviour, see step_2_8 docstring)."""
    df = _require(PROMETHEE_CORR_CSV)
    rho_row = df[df["metric"] == "spearman_rho"]
    assert not rho_row.empty, "spearman_rho row missing from correlation CSV"
    rho = float(rho_row.iloc[0]["value"])
    assert rho >= PROMETHEE_SPEARMAN_MIN, (
        f"Spearman rho = {rho:.3f} below sanity bound {PROMETHEE_SPEARMAN_MIN}"
    )
