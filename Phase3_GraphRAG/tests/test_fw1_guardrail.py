"""
tests/test_fw1_guardrail.py — offline tests for the FW1 tool chain
(guardrail_common, step_3_14, step_3_16 logic). No API, no Neo4j, no SDK.

Run: python -m pytest tests/test_fw1_guardrail.py -q

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from guardrail_common import (  # noqa: E402
    CANONICAL_EVAL_JSON, EvalRecord, em_strict, extract_json_verdict,
    load_eval_records,
)
from step_3_16_runtime_guardrail import ABSTAIN_TEXT, _guardrail_one  # noqa: E402


# ---------------------------------------------------------------- em_strict

def test_em_strict_basic_match() -> None:
    assert em_strict("The threshold is 1.0 MW under EED.", "1.0 MW") is True


def test_em_strict_below_half_keywords() -> None:
    assert em_strict("completely unrelated text", "1.0 MW threshold") is False


def test_em_strict_unscorable() -> None:
    assert em_strict("", "1.0 MW") is None
    assert em_strict("answer", "") is None


def test_em_strict_parity_with_canonical_artefact() -> None:
    """The replica must reproduce every stored exact_match flag bit-exactly."""
    if not CANONICAL_EVAL_JSON.exists():
        pytest.skip("canonical artefact not present")
    records = load_eval_records(CANONICAL_EVAL_JSON)
    scored = [r for r in records if r.stored_exact_match is not None]
    assert scored, "no scorable records in canonical artefact"
    mismatches = [
        r.question_id for r in scored
        if em_strict(r.stored_answer, r.ground_truth) != r.stored_exact_match
    ]
    assert mismatches == []


def test_canonical_artefact_is_context_only() -> None:
    """Regression lock for the step_3_14 finding (answer == context, 100/100)."""
    if not CANONICAL_EVAL_JSON.exists():
        pytest.skip("canonical artefact not present")
    records = load_eval_records(CANONICAL_EVAL_JSON)
    same = sum(1 for r in records
               if (r.stored_answer or "").strip() == (r.context or "").strip())
    assert same == len(records)


# ------------------------------------------------------- verifier JSON parse

def test_extract_json_verdict_pass() -> None:
    ok, fb = extract_json_verdict('{"pass": true, "feedback": ""}')
    assert ok is True and fb == ""


def test_extract_json_verdict_fail_open_is_false() -> None:
    ok, fb = extract_json_verdict("no json here")
    assert ok is False and "unparseable" in fb


def test_extract_json_verdict_with_prose_wrapper() -> None:
    ok, _ = extract_json_verdict('Sure: {"pass": false, "feedback": "missing count"}')
    assert ok is False


# ------------------------------------------------------- guardrail decision

def _record() -> EvalRecord:
    return EvalRecord(question_id="T01", nl_question="What is the threshold?",
                      ground_truth="1.0 MW", context="threshold_mw: 1.0")


def test_guardrail_pass_first_try() -> None:
    row = _guardrail_one(
        _record(),
        call_gen=lambda p: "The threshold is 1.0 MW.",
        call_gate=lambda p: '{"pass": true, "feedback": ""}',
    )
    assert row["status"] == "pass"
    assert row["em_final"] is True
    assert row["abstained"] is False


def test_guardrail_retry_then_pass() -> None:
    gate_verdicts = iter(['{"pass": false, "feedback": "state the number"}',
                          '{"pass": true, "feedback": ""}'])
    gen_answers = iter(["It is mandatory.", "The threshold is 1.0 MW."])
    row = _guardrail_one(
        _record(),
        call_gen=lambda p: next(gen_answers),
        call_gate=lambda p: next(gate_verdicts),
    )
    assert row["status"] == "retry_pass"
    assert row["em_final"] is True
    assert "state the number" in row["gate_feedback_v1"]


def test_guardrail_abstain_after_two_fails() -> None:
    row = _guardrail_one(
        _record(),
        call_gen=lambda p: "Irrelevant words.",
        call_gate=lambda p: '{"pass": false, "feedback": "not grounded"}',
    )
    assert row["status"] == "abstain"
    assert row["final_answer"] == ABSTAIN_TEXT
    assert row["abstained"] is True


def test_guardrail_ground_truth_never_in_prompts() -> None:
    """Runtime separation: the gate and generator must never see ground truth."""
    record = EvalRecord(question_id="T02", nl_question="What is the threshold?",
                        ground_truth="ZZUNIQUE99 sentinel value",
                        context="threshold_mw: 1.0")
    seen: list[str] = []

    def spy_gen(prompt: str) -> str:
        seen.append(prompt)
        return "Some answer."

    def spy_gate(prompt: str) -> str:
        seen.append(prompt)
        return '{"pass": true, "feedback": ""}'

    _guardrail_one(record, call_gen=spy_gen, call_gate=spy_gate)
    assert seen, "no prompts captured"
    assert all("ZZUNIQUE99" not in p for p in seen)
