"""TDD test for CRITICAL #2: the stress test must score with the canonical model.

step_2_6_stress_test.py re-declared ALPHA_d = 0.15 while the canonical scorer
step_2_1_is_match_score_lc.py uses ALPHA_d = 0.0 (D-decision 2026-05-31: distance
no longer affects the score). The stress test therefore perturbed and validated a
superseded model. The fix aligns the constant so the robustness layer validates
exactly the scorer that produces the canonical IS-Match numbers.
"""
from __future__ import annotations

import importlib

import pytest

stress = importlib.import_module("step_2_6_stress_test")
scorer = importlib.import_module("step_2_1_is_match_score_lc")


def test_stress_alpha_d_is_canonical_zero() -> None:
    """The stress test distance weight ALPHA_d is 0.0 (canonical, distance inert)."""
    assert stress.ALPHA_d == pytest.approx(0.0, abs=1e-12)


def test_stress_alpha_d_matches_scorer() -> None:
    """The stress test ALPHA_d equals the canonical scorer ALPHA_d (no drift)."""
    assert stress.ALPHA_d == scorer.ALPHA_d
