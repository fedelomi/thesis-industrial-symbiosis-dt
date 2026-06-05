"""TDD tests for the judge-summary regeneration utility (CRITICAL #1a).

The canonical EM-semantic accuracy reported in thesis Table 5.6 is 0.41 (41.0%)
with EM-strict 0.59 (59.0%). The committed step_3_9_llm_judge_summary.csv was
overwritten by a later glob-latest run to 63/66. This regenerator must recompute
the canonical 59/41 split deterministically from the frozen per-question judge
verdicts, so the number is regenerable from committed code rather than surviving
only in git history.
"""
from __future__ import annotations

import importlib

import pytest

regen = importlib.import_module("step_3_12_judge_regenerate")


def test_regenerate_reproduces_canonical_semantic_41() -> None:
    """The regenerated summary reproduces EM-semantic 41.0 exactly."""
    summary = regen.regenerate_judge_summary()
    assert summary["em_semantic_pct"] == pytest.approx(41.0, abs=1e-9)


def test_regenerate_reproduces_canonical_strict_59() -> None:
    """The regenerated summary reproduces EM-strict 59.0 exactly."""
    summary = regen.regenerate_judge_summary()
    assert summary["em_strict_pct"] == pytest.approx(59.0, abs=1e-9)
