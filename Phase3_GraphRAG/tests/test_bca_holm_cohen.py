"""TDD tests for the deterministic BCa + Holm + Cohen regenerator (CRITICAL #1b, Phase 3).

The thesis cites a BCa 95% CI on the EM gap (Table 5.5) but the committed
results/step_3_7_bootstrap_em_ci_bca.csv had no producing code. This regenerator
recomputes the canonical BCa intervals, the Holm-Bonferroni family correction and
Cohen's h deterministically (fixed seed, pinned canonical eval files) so the
intervals are regenerable from committed code.
"""
from __future__ import annotations

import importlib

import pytest

bhc = importlib.import_module("step_3_13_bca_holm_cohen")


def test_point_estimates_reproduce_canonical_em() -> None:
    """Observed EM reproduces the canonical 0.59 / 0.28 / 0.26 split exactly."""
    res = bhc.regenerate()
    assert res["graph-rag"]["em_observed"] == pytest.approx(0.59, abs=1e-9)
    assert res["no-rag"]["em_observed"] == pytest.approx(0.28, abs=1e-9)
    assert res["llm-cypher"]["em_observed"] == pytest.approx(0.26, abs=1e-9)


def test_gap_bootstrap_mean_reproduces_canonical() -> None:
    """The paired-bootstrap gap mean reproduces the canonical 0.3108 / 0.3308."""
    res = bhc.regenerate()
    assert res["graph-rag"]["gap_vs_no_rag_mean"] == pytest.approx(0.3108, abs=5e-4)
    assert res["graph-rag"]["gap_vs_llm_cypher_mean"] == pytest.approx(0.3308, abs=5e-4)


def test_gap_bca_ci_significant() -> None:
    """Both BCa gap intervals lie entirely above zero (significant)."""
    res = bhc.regenerate()
    assert res["graph-rag"]["gap_vs_no_rag_ci95_lo_bca"] > 0.0
    assert res["graph-rag"]["gap_vs_llm_cypher_ci95_lo_bca"] > 0.0


def test_holm_keeps_both_gaps_significant() -> None:
    """Holm-Bonferroni adjusted p-values keep both gaps significant at 0.05."""
    res = bhc.regenerate()
    assert res["holm"]["graph_vs_no_rag"]["p_holm"] < 0.05
    assert res["holm"]["graph_vs_llm_cypher"]["p_holm"] < 0.05
    # Holm is a step-down: adjusted p >= raw p for each test.
    assert res["holm"]["graph_vs_no_rag"]["p_holm"] >= res["holm"]["graph_vs_no_rag"]["p_raw"]


def test_cohens_h_graph_vs_no_rag_is_large() -> None:
    """Cohen's h on the graph-rag vs no-rag proportions is a large effect (>= 0.5)."""
    res = bhc.regenerate()
    assert res["cohens_h"]["graph_vs_no_rag"] >= 0.5


def test_determinism_identical_across_runs() -> None:
    """Two regenerations produce identical BCa endpoints (fixed seed)."""
    a = bhc.regenerate()
    b = bhc.regenerate()
    assert a["graph-rag"]["gap_vs_no_rag_ci95_lo_bca"] == b["graph-rag"]["gap_vs_no_rag_ci95_lo_bca"]
    assert a["graph-rag"]["gap_vs_no_rag_ci95_hi_bca"] == b["graph-rag"]["gap_vs_no_rag_ci95_hi_bca"]
