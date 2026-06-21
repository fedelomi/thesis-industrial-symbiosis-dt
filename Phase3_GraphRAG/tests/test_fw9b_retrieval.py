"""FW9b Task 5: union, densifier and pruner unit tests. No API, no Neo4j.

Pure-function tests on synthetic rows: union dedup + cap + order, densifier anchor
preservation + controlled standard-constant injection, and query-type-aware pruning.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from prompt5_retrieval import (  # noqa: E402
    NO_DATA_SENTINEL,
    classify_query_type,
    densify_context,
    densify_context_v2,
    prune_rows,
    union_template_rows,
)


# ----------------------------------------------------------------- union/dedup
def test_union_dedups_identical_rows_and_preserves_rank_order() -> None:
    data = {
        "A": [{"id": "X1", "v": 1}, {"id": "X2", "v": 2}],
        "B": [{"id": "X2", "v": 2}, {"id": "X3", "v": 3}],  # X2 is a duplicate
    }
    rows = union_template_rows(["A", "B"], lambda tid: data[tid])
    assert rows == [{"id": "X1", "v": 1}, {"id": "X2", "v": 2}, {"id": "X3", "v": 3}]


def test_union_respects_row_cap() -> None:
    data = {"A": [{"id": f"X{i}"} for i in range(100)]}
    rows = union_template_rows(["A"], lambda tid: data[tid], row_cap=40)
    assert len(rows) == 40


def test_union_is_deterministic() -> None:
    data = {"A": [{"id": "X1"}, {"id": "X2"}], "B": [{"id": "X3"}]}
    a = union_template_rows(["A", "B"], lambda tid: data[tid])
    b = union_template_rows(["A", "B"], lambda tid: data[tid])
    assert a == b


# ----------------------------------------------------------------- densifier
def test_densifier_preserves_anchor_tokens_verbatim() -> None:
    rows = [{"article": "Article 26", "threshold_mw": "1 MW", "temp": "60 degC",
             "value": "EUR 0.061/MWh", "pct": "53%", "id": "DC-L-01"}]
    out = densify_context(rows)
    for anchor in ("Article 26", "1 MW", "60 degC", "EUR 0.061/MWh", "53%", "DC-L-01"):
        assert anchor in out, f"anchor lost: {anchor}"


def test_densifier_injects_iso_constant_only_for_iso_templates() -> None:
    rows = [{"clause": "4.4", "topic": "energy review"}]
    # An ISO50001 template fired: the standard constant is injected once.
    out = densify_context_v2(rows, template_ids=["ISO50001_ARTICLES"])
    assert "ISO 50001:2018" in out
    assert out.count("ISO 50001:2018") == 1
    assert "energy review" in out  # body anchors preserved


def test_densifier_does_not_inject_for_unrelated_templates() -> None:
    rows = [{"clause": "4.4", "topic": "energy review"}]
    out = densify_context_v2(rows, template_ids=["P1_eed_art26_threshold"])
    assert "ISO 50001:2018" not in out


def test_densifier_no_injection_when_constant_already_present() -> None:
    rows = [{"standard": "ISO 50001:2018", "clause": "4.4"}]
    out = densify_context_v2(rows, template_ids=["ISO50001_ARTICLES"])
    assert out.count("ISO 50001:2018") == 1  # not duplicated


def test_densifier_empty_rows_returns_sentinel() -> None:
    assert densify_context_v2([], template_ids=["ISO50001_ARTICLES"]) == NO_DATA_SENTINEL


# ----------------------------------------------------------------- pruner
def test_query_type_classification() -> None:
    assert classify_query_type("How many scenarios target the food sector?") == "count"
    assert classify_query_type("Compare the 3GDH and 4GDH networks") == "compare"
    assert classify_query_type("List all regulatory articles") == "aggregate"
    assert classify_query_type("Which agency enforces the heat supply act?") == "lookup"


def test_pruner_never_prunes_count_or_compare() -> None:
    rows = [{"id": f"X{i}", "sector": "food"} for i in range(30)]
    assert prune_rows(rows, "How many scenarios involve the food sector?") == rows
    assert prune_rows(rows, "Compare food and paper sectors") == rows


def test_pruner_keeps_all_under_threshold_even_for_lookup() -> None:
    rows = [{"id": f"X{i}"} for i in range(5)]
    assert prune_rows(rows, "what is the supply temperature", prune_threshold=12) == rows


def test_pruner_mechanism_fires_only_when_threshold_is_forced_low() -> None:
    """The prune mechanism works when its threshold is forced below the union size.

    This forces prune_threshold=12 to exercise the code path. At the DEFAULT
    threshold (the row cap) and the realistic union sizes observed in the eval
    (10 to 35 rows) the pruner is DORMANT (see the test below), so pruning is a
    last-resort guard, not a contributing lever of the v8 result.
    """
    rows = [{"id": "REL", "topic": "danish heat supply act enforcement agency"}]
    rows += [{"id": f"NOISE{i}", "topic": "unrelated thermal band"} for i in range(20)]
    out = prune_rows(rows, "which agency enforces the danish heat supply act", prune_threshold=12)
    assert len(out) < len(rows)
    assert {"id": "REL", "topic": "danish heat supply act enforcement agency"} in out


def test_pruner_is_dormant_at_default_threshold_on_realistic_unions() -> None:
    """At the default threshold (row cap) a 16-row or 35-row lookup union is intact."""
    rows16 = [{"id": f"R{i}", "topic": "regulatory article"} for i in range(16)]
    assert prune_rows(rows16, "which agency enforces the heat supply act") == rows16
    rows35 = [{"id": f"R{i}"} for i in range(35)]
    assert prune_rows(rows35, "what is the supply temperature") == rows35


def test_pruner_is_deterministic() -> None:
    rows = [{"id": "REL", "topic": "agency enforcement"}]
    rows += [{"id": f"N{i}", "topic": "noise"} for i in range(20)]
    a = prune_rows(rows, "which agency does enforcement", prune_threshold=12)
    b = prune_rows(rows, "which agency does enforcement", prune_threshold=12)
    assert a == b
