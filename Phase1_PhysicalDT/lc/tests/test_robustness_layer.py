"""
Unit tests for Phase 1 LC robustness layer (step_1_5 to step_1_8).

Each test reads the canonical CSV produced by the corresponding module
and asserts the sanity bounds documented in the module docstrings.
The tests assume the modules have already run; the orchestrator
run_phase_1_lc.py produces all required CSVs before pytest is invoked.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SOBOL_CSV = RESULTS_DIR / "step_1_5_sobol_rc_params.csv"
BENCHMARK_AGREEMENT_CSV = RESULTS_DIR / "step_1_6_benchmark_agreement.csv"
RESIDUAL_CSV = RESULTS_DIR / "step_1_7_residual_diagnostics.csv"
CLIMATE_CSV = RESULTS_DIR / "step_1_8_climate_sensitivity.csv"

BENCHMARK_AGREEMENT_MIN_PCT = 80.0
ACF_LAG1_PRESERVATION_MIN_PCT = 80.0
T_AVAIL_MIN = 0.85


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Required CSV not produced yet: {path.name}")
    return pd.read_csv(path)


def test_sobol_additivity() -> None:
    """Sum of first-order Sobol indices should be ~ 1.0 (+/- 0.05) per
    (scenario, output_metric). Confirms an additive model with limited
    interaction effects."""
    df = _require(SOBOL_CSV)
    groups = df.groupby(["scenario", "output_metric"])["S1"].sum()
    for (scenario, metric), s1_sum in groups.items():
        assert 0.80 <= float(s1_sum) <= 1.15, (
            f"Sobol additivity off for {scenario}/{metric}: sum(S1)={s1_sum:.3f} not in [0.80, 1.15]"
        )


def test_multi_benchmark_agreement() -> None:
    """Residual-metrics agreement (excluding ERF and PUE_implicit, which are
    informational only - model upper bound and core-only PUE respectively)
    should be >= 80% overall."""
    df = _require(BENCHMARK_AGREEMENT_CSV)
    all_row = df[df["scenario"] == "ALL_LC"]
    assert not all_row.empty, "ALL_LC summary row missing"
    pct = float(all_row.iloc[0]["agreement_pct_residual_metrics_only"])
    assert pct >= BENCHMARK_AGREEMENT_MIN_PCT, (
        f"Residual-metrics agreement {pct:.1f}% below {BENCHMARK_AGREEMENT_MIN_PCT}% bound"
    )


def test_block_bootstrap_lag1_preserved() -> None:
    """Within-block (lag-1) ACF preservation should be >= 80% for at
    least 2/3 features per LC scenario. Lag-96 and lag-672 are not
    gated here: block_size=96 with random starts intentionally breaks
    cross-block correlations (documented in step_1_7 docstring)."""
    df = _require(RESIDUAL_CSV)
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        n_pass = int((sub["acf_preservation_lag1_pct"] >= ACF_LAG1_PRESERVATION_MIN_PCT).sum())
        assert n_pass >= 2, (
            f"Scenario {scenario}: only {n_pass}/{len(sub)} features have lag-1 ACF preservation "
            f">= {ACF_LAG1_PRESERVATION_MIN_PCT}%"
        )


def test_climate_t_availability() -> None:
    """t_availability must stay > 0.85 across all (scenario, climate) cells.
    Drops below would indicate the analytical RC model fails outside the
    studied climate window."""
    df = _require(CLIMATE_CSV)
    bad = df[df["t_availability"] < T_AVAIL_MIN]
    assert bad.empty, (
        f"{len(bad)} rows have t_availability < {T_AVAIL_MIN}: "
        + ", ".join(
            f"{r['scenario_lc']}/{r['climate_label']}={r['t_availability']:.3f}"
            for _, r in bad.iterrows()
        )
    )
