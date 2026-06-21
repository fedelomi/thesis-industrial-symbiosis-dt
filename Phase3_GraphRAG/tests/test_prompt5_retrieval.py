"""TDD tests for PROMPT 5 multi-template retrieval + densification (v7 candidate).

Covers the four new units in prompt5_retrieval.py:
  C2  route_question_multi   - top-N gated candidate pool
  C2  union_template_rows    - dedup + cap union of template rows
  C3  densify_context        - declarative formatter, verbatim anchors preserved
  fix refine_error_class     - separates under_retrieval from synthesis_fail

No Neo4j, no API: the router and the Cypher runner are injected as stubs.
"""
from __future__ import annotations

import pytest

from prompt5_retrieval import (
    route_question_multi,
    union_template_rows,
    densify_context,
    refine_error_class,
)


class _StubRouter:
    """Minimal router exposing .score(query) -> [(template_id, confidence), ...]."""

    def __init__(self, ranked):
        self._ranked = ranked

    def score(self, query, candidates=None):  # noqa: ARG002 - signature parity
        return list(self._ranked)


# --------------------------------------------------------------------------- C2
def test_route_multi_keyword_only_returns_keyword_template():
    # "penetration" is a deterministic keyword for GENERIC_COUNTRY.
    pool = route_question_multi("what is the dh penetration", use_semantic=False)
    assert "GENERIC_COUNTRY" in pool


def test_route_multi_falls_back_to_entity_when_empty():
    pool = route_question_multi("zzz qqq", use_semantic=False)
    assert pool == ["P6_full_is_path"]


def test_route_multi_extends_with_semantic_up_to_n():
    router = _StubRouter([("A", 0.9), ("B", 0.5), ("C", 0.1)])
    pool = route_question_multi(
        "zzz qqq", n=5, use_semantic=True, router=router, conf_floor=0.2
    )
    # C is filtered (below conf_floor); A and B kept in rank order.
    assert pool == ["A", "B"]


def test_route_multi_caps_at_n():
    router = _StubRouter([("A", 0.9), ("B", 0.8), ("C", 0.7)])
    pool = route_question_multi(
        "zzz qqq", n=1, use_semantic=True, router=router, conf_floor=0.0
    )
    assert pool == ["A"]


def test_route_multi_dedups():
    router = _StubRouter([("A", 0.9), ("A", 0.8), ("B", 0.5)])
    pool = route_question_multi(
        "zzz qqq", n=5, use_semantic=True, router=router, conf_floor=0.0
    )
    assert pool == ["A", "B"]


# --------------------------------------------------------------------------- C2
def test_union_collects_and_dedups_rows():
    data = {"T1": [{"a": 1}], "T2": [{"a": 1}, {"b": 2}]}
    rows = union_template_rows(["T1", "T2"], data.get, row_cap=40)
    assert rows == [{"a": 1}, {"b": 2}]


def test_union_respects_row_cap():
    data = {"T1": [{"a": 1}, {"b": 2}, {"c": 3}]}
    rows = union_template_rows(["T1"], data.get, row_cap=2)
    assert len(rows) == 2


def test_union_empty_when_no_rows():
    rows = union_template_rows(["T1"], lambda _t: [], row_cap=40)
    assert rows == []


# --------------------------------------------------------------------------- C3
def test_densify_preserves_verbatim_anchor_tokens():
    rows = [{"article_id": "EED-ART-23", "supply_c": 47.6, "id": "DC-L", "eur_per_mwh": 25.8}]
    out = densify_context(rows)
    for anchor in ("EED-ART-23", "47.6", "DC-L", "25.8"):
        assert anchor in out


def test_densify_drops_pipe_noise():
    rows = [{"a": 1, "b": 2}]
    out = densify_context(rows)
    assert " | " not in out


def test_densify_one_line_per_row():
    rows = [{"a": 1}, {"b": 2}, {"c": 3}]
    out = densify_context(rows)
    assert len(out.splitlines()) == 3


def test_densify_empty_matches_sentinel():
    assert densify_context([]) == "No data found in knowledge graph."


# ---------------------------------------------------------------------- metric
def test_refine_under_retrieval_distinct_from_synthesis():
    # routed ok, lcr ok, not semantically correct, but provenance NOT fully retrieved.
    cls = refine_error_class(
        is_gap=False, sem_correct=False, routing_ok=True, lcr=1.0,
        retrieval_complete=False, min_lcr=0.5,
    )
    assert cls == "under_retrieval"


def test_refine_synthesis_when_retrieval_complete():
    cls = refine_error_class(
        is_gap=False, sem_correct=False, routing_ok=True, lcr=1.0,
        retrieval_complete=True, min_lcr=0.5,
    )
    assert cls == "synthesis_fail"


def test_refine_gap_and_correct_and_routing_unchanged():
    assert refine_error_class(True, False, True, 1.0, True, 0.5) == "coverage_gap"
    assert refine_error_class(False, True, True, 1.0, False, 0.5) == "correct"
    assert refine_error_class(False, False, False, 1.0, True, 0.5) == "routing_fail"
    assert refine_error_class(False, False, True, 0.2, True, 0.5) == "routing_fail"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
