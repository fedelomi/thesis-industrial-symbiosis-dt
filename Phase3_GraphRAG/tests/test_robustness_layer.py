"""
Unit tests for Phase 3 robustness layer (step_3_7 to step_3_10).

Each test reads the canonical CSV produced by the corresponding module
and asserts the sanity bounds documented in the module docstrings.
The tests assume the modules have already run; run_phase_3.py produces
all required CSVs before pytest is invoked.

step_3_9 (LLM judge) and step_3_10 (paraphrase routing) can be skipped
via run_phase_3.py --skip-llm-judge / --skip-paraphrase. The
corresponding tests pytest.skip() if their CSVs are absent rather than
fail; this lets the layer ship without API spend when needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BOOTSTRAP_CSV = RESULTS_DIR / "step_3_7_bootstrap_em_ci.csv"
TEMPLATE_LOO_CSV = RESULTS_DIR / "step_3_8_template_leave_one_out.csv"
LLM_JUDGE_SUMMARY_CSV = RESULTS_DIR / "step_3_9_llm_judge_summary.csv"
PARAPHRASE_SUMMARY_CSV = RESULTS_DIR / "step_3_10_routing_stability_summary.csv"

CRITICAL_TEMPLATE_DROP_PP_MIN = 5.0
MIN_CRITICAL_TEMPLATES = 3
LLM_JUDGE_FP_RATE_MAX_PCT = 30.0
LLM_JUDGE_AGREEMENT_MIN_PCT = 70.0
ROUTING_STABILITY_MIN_PCT = 60.0


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Required CSV not produced yet: {path.name}")
    return pd.read_csv(path)


def test_bootstrap_em_ci_significant() -> None:
    """CI95 of the gap Graph RAG vs no-rag must exclude 0 (lower bound > 0)."""
    df = _require(BOOTSTRAP_CSV)
    gr = df[df["baseline"] == "graph-rag"]
    assert not gr.empty, "graph-rag row missing in bootstrap CSV"
    lo = float(gr.iloc[0]["gap_vs_no_rag_ci95_lo"])
    hi = float(gr.iloc[0]["gap_vs_no_rag_ci95_hi"])
    assert lo > 0, (
        f"Gap Graph RAG vs no-rag CI95 = [{lo:.3f}, {hi:.3f}] includes 0; "
        "improvement is NOT statistically significant"
    )


def test_template_loo_criticality_ranked() -> None:
    """At least MIN_CRITICAL_TEMPLATES templates must drop EM by >= 5 pp
    when removed; this is what makes the LOO ranking informative."""
    df = _require(TEMPLATE_LOO_CSV)
    n_critical = int((df["drop_em_if_removed_pp"] >= CRITICAL_TEMPLATE_DROP_PP_MIN).sum())
    assert n_critical >= MIN_CRITICAL_TEMPLATES, (
        f"Only {n_critical} templates have drop_em_if_removed_pp >= "
        f"{CRITICAL_TEMPLATE_DROP_PP_MIN}; expected >= {MIN_CRITICAL_TEMPLATES}"
    )


def test_llm_judge_false_positive_rate() -> None:
    """The keyword matcher must not be massively too permissive: at
    most LLM_JUDGE_FP_RATE_MAX_PCT % of strict EM=True answers may be
    judged semantically wrong by the cross-model judge. The threshold
    is intentionally tolerant - the 2026-05-22 finding measured 22 %
    FP, well above naive 5 % expectations, so the goal here is to
    catch a catastrophic regression rather than to gate the
    acceptable level."""
    df = _require(LLM_JUDGE_SUMMARY_CSV)
    by_metric = dict(zip(df["metric"], df["value"]))
    fp = float(by_metric["false_positive_rate_pct"])
    assert fp < LLM_JUDGE_FP_RATE_MAX_PCT, (
        f"Keyword matcher false-positive rate {fp:.1f}% above "
        f"{LLM_JUDGE_FP_RATE_MAX_PCT}% bound; strict EM overstates accuracy"
    )


def test_llm_judge_agreement() -> None:
    """Non-directional sanity: cross-model judge and keyword matcher
    must agree on at least LLM_JUDGE_AGREEMENT_MIN_PCT % of queries.
    Below would mean the keyword matcher and the judge are looking at
    different signals altogether."""
    df = _require(LLM_JUDGE_SUMMARY_CSV)
    by_metric = dict(zip(df["metric"], df["value"]))
    agreement = float(by_metric["agreement_pct"])
    assert agreement >= LLM_JUDGE_AGREEMENT_MIN_PCT, (
        f"Judge / matcher agreement {agreement:.1f}% below "
        f"{LLM_JUDGE_AGREEMENT_MIN_PCT}% bound"
    )


def test_paraphrase_routing_stability_threshold() -> None:
    """Routing stability under paraphrasing must clear the tolerant
    lower bound (>= 60%); below means the keyword matcher is too fragile."""
    df = _require(PARAPHRASE_SUMMARY_CSV)
    by_metric = dict(zip(df["metric"], df["value"]))
    pct = float(by_metric["routing_stability_pct"])
    assert pct >= ROUTING_STABILITY_MIN_PCT, (
        f"Routing stability {pct:.1f}% below {ROUTING_STABILITY_MIN_PCT}% bound; "
        "keyword router too fragile to paraphrasing"
    )
