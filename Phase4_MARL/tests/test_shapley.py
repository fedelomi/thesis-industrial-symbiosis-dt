"""Unit tests for the closed-form bilateral Shapley value (B5)."""

from __future__ import annotations

import math

import pytest

from agents.shapley import shapley_fairness_ratio, shapley_value_bilateral


def test_shapley_efficiency_symmetric() -> None:
    """Symmetric case: equal stand-alone values -> equal allocations summing to v(N)."""
    phi_dc, phi_mfg = shapley_value_bilateral(v_dc=10.0, v_mfg=10.0, v_coalition=30.0)
    assert math.isclose(phi_dc, phi_mfg)
    assert math.isclose(phi_dc + phi_mfg, 30.0)
    # Each gets stand-alone + half the synergy: 10 + (30 - 20)/2 = 15
    assert math.isclose(phi_dc, 15.0)


def test_shapley_dc_dominant() -> None:
    """DC stand-alone >> MFG stand-alone -> DC gets larger share."""
    phi_dc, phi_mfg = shapley_value_bilateral(v_dc=100.0, v_mfg=10.0, v_coalition=150.0)
    assert phi_dc > phi_mfg
    assert math.isclose(phi_dc + phi_mfg, 150.0)


def test_shapley_zero_standalone() -> None:
    """Zero stand-alone for both -> synergy split 50/50."""
    phi_dc, phi_mfg = shapley_value_bilateral(v_dc=0.0, v_mfg=0.0, v_coalition=42.0)
    assert math.isclose(phi_dc, 21.0)
    assert math.isclose(phi_mfg, 21.0)
    assert math.isclose(shapley_fairness_ratio(phi_dc, phi_mfg), 1.0)


def test_shapley_efficiency_axiom_random() -> None:
    """Efficiency axiom: phi_dc + phi_mfg == v(N) for arbitrary inputs."""
    cases = [(7.5, -3.0, 25.0), (-2.0, -1.0, 5.0), (0.1, 0.2, 0.5)]
    for v_dc, v_mfg, v_n in cases:
        phi_dc, phi_mfg = shapley_value_bilateral(v_dc, v_mfg, v_n)
        assert math.isclose(phi_dc + phi_mfg, v_n)


def test_fairness_ratio_bounds() -> None:
    assert math.isclose(shapley_fairness_ratio(5.0, 5.0), 1.0)
    assert shapley_fairness_ratio(1.0, 10.0) == pytest.approx(0.1)
    assert shapley_fairness_ratio(0.0, 0.0) == 0.0


def test_fairness_ratio_negative_allocation() -> None:
    """Negative values: absolute-value semantics so the ratio stays in [0, 1]."""
    assert shapley_fairness_ratio(-4.0, 8.0) == pytest.approx(0.5)
