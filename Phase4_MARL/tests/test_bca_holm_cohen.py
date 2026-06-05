"""TDD tests for the deterministic BCa + Holm + Cohen regenerator (CRITICAL #1b, Phase 4).

The committed results/robustness/step_4_5_bootstrap_ci_bca.csv, step_4_5_gap_ci_bca.csv,
step_4_5_holm_bonferroni.csv and step_4_5_cohens_d.csv had no producing script. This
regenerator recomputes them deterministically from the canonical 10-seed
ablation_results.csv, reusing step_4_5's exact paired-bootstrap machinery so the
canonical PoA / Shapley numbers and their BCa intervals are regenerable.
"""
from __future__ import annotations

import importlib

import pytest

reg = importlib.import_module("step_4_12_bca_holm_cohen")


def test_poa_a0_9lc_mean_and_bca_reproduce_canonical() -> None:
    """A0 9LC PoA reproduces mean 0.7525 with BCa CI [0.7294, 0.7775]."""
    r = reg.regenerate()
    poa = r["per_group"][("A0", "9LC", "price_of_anarchy")]
    assert poa["mean"] == pytest.approx(0.7525, abs=1e-3)
    assert poa["ci95_lo_bca"] == pytest.approx(0.7294, abs=2e-3)
    assert poa["ci95_hi_bca"] == pytest.approx(0.7775, abs=2e-3)


def test_gap_poa_9lc_not_significant() -> None:
    """The 9LC PoA gap (+0.030) BCa CI crosses zero (not significant)."""
    g = reg.regenerate()["gap"][("9LC", "price_of_anarchy")]
    assert g["gap_mean"] == pytest.approx(0.0304, abs=1e-3)
    assert g["ci95_lo_bca"] < 0.0 < g["ci95_hi_bca"]
    assert g["significant_at_95_bca"] is False


def test_gap_shapley_9lc_significant_negative() -> None:
    """The 9LC Shapley gap (-0.207) BCa CI lies entirely below zero."""
    g = reg.regenerate()["gap"][("9LC", "shapley_fairness")]
    assert g["gap_mean"] == pytest.approx(-0.2071, abs=1e-3)
    assert g["ci95_hi_bca"] < 0.0
    assert g["significant_at_95_bca"] is True


def test_holm_shapley_significant_poa_not() -> None:
    """Holm keeps the Shapley gaps significant and PoA non-significant."""
    holm = reg.regenerate()["holm"]
    assert holm[("9LC", "shapley_fairness")]["p_holm"] < 0.05
    assert holm[("9LC", "price_of_anarchy")]["p_holm"] >= 0.05


def test_cohens_d_shapley_9lc_reproduces() -> None:
    """Paired Cohen's d for the 9LC Shapley gap reproduces -0.583 (large)."""
    d = reg.regenerate()["cohens_d"][("9LC", "shapley_fairness")]
    assert d == pytest.approx(-0.583, abs=5e-3)


def test_determinism_identical_across_runs() -> None:
    """Two regenerations produce identical gap BCa endpoints (fixed seed)."""
    a = reg.regenerate()["gap"][("9LC", "shapley_fairness")]
    b = reg.regenerate()["gap"][("9LC", "shapley_fairness")]
    assert a["ci95_lo_bca"] == b["ci95_lo_bca"]
    assert a["ci95_hi_bca"] == b["ci95_hi_bca"]
