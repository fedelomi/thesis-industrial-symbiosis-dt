"""
Unit tests for Phase 4 robustness layer (step_4_5 to step_4_8).

Each test reads the canonical CSV produced by the corresponding module
and asserts the sanity bounds documented in the module docstrings.
Tests skip (rather than fail) when their CSV is not produced yet, so
the layer ships gracefully when `--skip-heavy` is used in the
orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PHASE4_DIR = Path(__file__).resolve().parents[1]
ROBUSTNESS_DIR = PHASE4_DIR / "results" / "robustness"

BOOTSTRAP_GAP_CSV = ROBUSTNESS_DIR / "step_4_5_gap_ci.csv"
STATE_RANKING_CSV = ROBUSTNESS_DIR / "step_4_7_dimension_ranking.csv"
SHIELDING_RANKING_CSV = ROBUSTNESS_DIR / "step_4_6_shielding_criticality_ranking.csv"
DELTATC_SUMMARY_CSV = ROBUSTNESS_DIR / "step_4_8_deltatc_robustness_summary.csv"

BOOTSTRAP_9LC_PoA_TARGET = 0.0300
BOOTSTRAP_TOLERANCE = 0.010
TOP3_SENSITIVITY_DIM_CANDIDATES = {
    "Q_avail_over_Q_inst",
    "Q_proc_over_Q_avail_ref",
}
DELTATC_MAX_DELTA_POA = 0.10


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Required CSV not produced yet: {path.name}")
    return pd.read_csv(path)


def test_bootstrap_gap_sanity() -> None:
    """9LC PoA gap mean should match the canonical 0.0303 within +-0.010."""
    df = _require(BOOTSTRAP_GAP_CSV)
    sub = df[(df["scenario_group"] == "9LC") & (df["metric"] == "PoA")]
    assert not sub.empty, "9LC/PoA row missing in gap CSV"
    gap = float(sub.iloc[0]["gap_D_minus_A0_mean"])
    assert abs(gap - BOOTSTRAP_9LC_PoA_TARGET) <= BOOTSTRAP_TOLERANCE, (
        f"9LC PoA gap = {gap:+.4f}, expected {BOOTSTRAP_9LC_PoA_TARGET:+.4f} "
        f"+-{BOOTSTRAP_TOLERANCE:.3f}"
    )


def test_shielding_loo_rule_critical() -> None:
    """At least one shielding rule must show a non-trivial impact when
    disabled - if every rule has zero delta then the shielding layer
    is inactive during eval (red flag)."""
    df = _require(SHIELDING_RANKING_CSV)
    max_score = float(df["criticality_score"].abs().max())
    assert max_score > 0.0, (
        "All shielding rules show zero criticality score; shielding may be inactive"
    )


def test_state_sensitivity_Q_in_top3() -> None:
    """Q-related observation dims (Q_avail/Q_inst or Q_proc/Q_avail_ref)
    should appear in the top-3 most sensitive dimensions. If neither
    is, the policy ignores the physical-capacity signal (red flag)."""
    df = _require(STATE_RANKING_CSV)
    top3 = set(df.head(3)["state_dim_name"].astype(str).tolist())
    assert top3 & TOP3_SENSITIVITY_DIM_CANDIDATES, (
        f"Top-3 sensitive dims = {top3}; none of {TOP3_SENSITIVITY_DIM_CANDIDATES} present"
    )


def test_deltatc_robustness_within_threshold() -> None:
    """Maximum observed |delta PoA| under delta_TC perturbations must
    be below the documented threshold (0.10)."""
    df = _require(DELTATC_SUMMARY_CSV)
    by_metric = dict(zip(df["metric"], df["value"]))
    max_delta = abs(float(by_metric["max_delta_PoA_observed"]))
    assert max_delta <= DELTATC_MAX_DELTA_POA, (
        f"max |delta PoA| = {max_delta:.4f} above bound {DELTATC_MAX_DELTA_POA}"
    )
