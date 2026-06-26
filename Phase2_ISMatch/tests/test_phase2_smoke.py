"""Smoke test for Phase 2 - IS-Match Score.

Loads the canonical IS-Match score CSV (step_2_1_is_match_scores_lc.csv)
and asserts that the headline metrics match the values published in
thesis Table 5.11 / Section 5.4 (IS-Match).

Offline-only: no Sobol re-run, no cvxpy optimisation. Runtime: <1 second.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

PHASE2_RESULTS = Path(__file__).resolve().parents[1] / "results"
IS_MATCH_CSV = PHASE2_RESULTS / "step_2_1_is_match_scores_lc.csv"

# Published score bounds from thesis §5.4 / Table 5.11 (canonical freeze 2026-05-17)
# IS-Match scores range over 9 LC scenarios.
PUBLISHED_MIN_SCORE = 0.39   # lowest published score (HighT_130C tier, not-feasible arm)
PUBLISHED_MAX_SCORE = 0.57   # highest published score (LowT_60C arm)
EXPECTED_N_SCENARIOS = 9


def _load_is_match_csv() -> list[dict[str, str]]:
    with IS_MATCH_CSV.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_is_match_csv_exists() -> None:
    assert IS_MATCH_CSV.exists(), (
        f"Canonical Phase 2 IS-Match CSV not found: {IS_MATCH_CSV}"
    )


def test_expected_row_count() -> None:
    rows = _load_is_match_csv()
    assert len(rows) == EXPECTED_N_SCENARIOS, (
        f"Expected {EXPECTED_N_SCENARIOS} IS-Match rows, got {len(rows)}"
    )


def test_required_columns_present() -> None:
    rows = _load_is_match_csv()
    required = {"dc_name", "process_name", "is_match_score", "tier", "rank"}
    actual = set(rows[0].keys())
    missing = required - actual
    assert not missing, f"Columns missing from IS-Match CSV: {missing}"


def test_all_scores_in_published_range() -> None:
    rows = _load_is_match_csv()
    for row in rows:
        score = float(row["is_match_score"])
        assert PUBLISHED_MIN_SCORE - 0.01 <= score <= PUBLISHED_MAX_SCORE + 0.01, (
            f"IS-Match score {score:.4f} for {row['dc_name']}+{row['process_name']} "
            f"outside published range [{PUBLISHED_MIN_SCORE}, {PUBLISHED_MAX_SCORE}]"
        )


def test_sobol_weights_csv_exists() -> None:
    sobol_csv = PHASE2_RESULTS / "step_2_5_sobol_weights.csv"
    assert sobol_csv.exists(), (
        f"Sobol sensitivity CSV not found: {sobol_csv}"
    )


def test_sobol_delta_dominance() -> None:
    """Asserts delta coefficient is the dominant Sobol first-order index (thesis §5.7.1)."""
    sobol_csv = PHASE2_RESULTS / "step_2_5_sobol_weights.csv"
    if not sobol_csv.exists():
        pytest.skip("Sobol CSV not available")
    with sobol_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        pytest.skip("Sobol CSV is empty")
    # Published: S1_delta = 0.870, S1_gamma = 0.095, S1_beta = 0.014
    # Assert delta is the largest first-order index
    delta_rows = [r for r in rows if r.get("parameter") == "delta" and r.get("metric") == "Y_aggregate"]
    if not delta_rows:
        pytest.skip("Delta row (parameter=delta, metric=Y_aggregate) not found in Sobol CSV")
    s1_delta = float(delta_rows[0]["S1"])
    assert s1_delta >= 0.80, (
        f"Sobol S1 for delta ({s1_delta:.3f}) should be >= 0.80 per thesis §5.7.1"
    )
