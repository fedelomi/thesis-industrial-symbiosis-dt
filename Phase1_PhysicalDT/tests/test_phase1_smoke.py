"""Smoke test for Phase 1 - Physical Digital Twin.

Loads the canonical benchmark-agreement CSV (step_1_6_benchmark_agreement.csv)
and asserts that the headline multi-benchmark cross-check metrics match the
values published in thesis Table 5.1.

Offline-only: no RC simulation, no CoolProp calls, no file writes.
Total runtime: <1 second.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

PHASE1_RESULTS = Path(__file__).resolve().parents[1] / "lc" / "results"
BENCHMARK_CSV = PHASE1_RESULTS / "step_1_6_benchmark_agreement.csv"

# Headline published values from thesis Table 5.1 (canonical, 2026-05-17 freeze)
PUBLISHED = {
    "Edge_LC": {"agreement_all": 80.0, "agreement_residual": 100.0},
    "Mid_LC": {"agreement_all": 70.0, "agreement_residual": 100.0},
    "Hyperscale_LC": {"agreement_all": 50.0, "agreement_residual": 83.3},
    "ALL_LC": {"agreement_all": 66.7, "agreement_residual": 94.4},
}
TOLERANCE = 0.11  # floating-point rounding tolerance in percentage points


def _load_benchmark_csv() -> dict[str, dict[str, float]]:
    rows = {}
    with BENCHMARK_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["scenario"]] = {
                "agreement_all": float(row["agreement_pct_all_metrics"]),
                "agreement_residual": float(row["agreement_pct_residual_metrics_only"]),
            }
    return rows


def test_benchmark_csv_exists() -> None:
    assert BENCHMARK_CSV.exists(), (
        f"Canonical Phase 1 benchmark CSV not found: {BENCHMARK_CSV}"
    )


def test_headline_agreement_all_lc() -> None:
    data = _load_benchmark_csv()
    assert "ALL_LC" in data, "Row 'ALL_LC' missing from benchmark CSV"
    observed = data["ALL_LC"]["agreement_residual"]
    expected = PUBLISHED["ALL_LC"]["agreement_residual"]
    assert abs(observed - expected) < TOLERANCE, (
        f"ALL_LC residual agreement: expected {expected}%, got {observed}%"
    )


@pytest.mark.parametrize("scenario", ["Edge_LC", "Mid_LC", "Hyperscale_LC", "ALL_LC"])
def test_per_scenario_agreement_residual(scenario: str) -> None:
    data = _load_benchmark_csv()
    assert scenario in data, f"Scenario '{scenario}' missing from benchmark CSV"
    observed = data[scenario]["agreement_residual"]
    expected = PUBLISHED[scenario]["agreement_residual"]
    assert abs(observed - expected) < TOLERANCE, (
        f"{scenario} residual agreement: expected {expected}%, got {observed}%"
    )


def test_all_scenarios_present() -> None:
    data = _load_benchmark_csv()
    for s in PUBLISHED:
        assert s in data, f"Expected scenario '{s}' not in CSV"


def test_overall_residual_above_90pct() -> None:
    """Lower-bound guard: CI95 lower on 17/18 binomial is ~77%. Published 94.4% must hold."""
    data = _load_benchmark_csv()
    observed = data["ALL_LC"]["agreement_residual"]
    assert observed >= 90.0, (
        f"ALL_LC residual agreement {observed}% is below 90% floor"
    )
