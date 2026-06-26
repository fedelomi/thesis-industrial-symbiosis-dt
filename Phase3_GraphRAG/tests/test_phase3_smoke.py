"""Smoke test for Phase 3 - Graph-RAG (Institutional-LLM Layer).

Loads the canonical evaluation JSON (evaluation_results_graph-rag_20260517_155731.json)
and recomputes the EM strict metric offline. Asserts that the recomputed value
falls within the published 95% CI (BCa) of Table 5.5.

Offline-only: no Neo4j connection, no LangChain pipeline, no Anthropic API calls.
Runtime: <2 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

PHASE3_DATA = Path(__file__).resolve().parents[1] / "data"
CANONICAL_JSON = PHASE3_DATA / "evaluation_results_graph-rag_20260517_155731.json"

# Published thesis CI95 BCa (Table 5.5, canonical 2026-05-17 freeze)
PUBLISHED_EM_OBSERVED = 0.59
PUBLISHED_CI95_LO = 0.48
PUBLISHED_CI95_HI = 0.67
PUBLISHED_N_QUERIES = 100


def _load_canonical_eval() -> list[dict]:
    with CANONICAL_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


def _compute_em(records: list[dict]) -> float:
    hits = sum(1 for r in records if r.get("exact_match") is True)
    return hits / len(records) if records else 0.0


def test_canonical_json_exists() -> None:
    assert CANONICAL_JSON.exists(), (
        f"Canonical Phase 3 evaluation JSON not found: {CANONICAL_JSON}"
    )


def test_query_count() -> None:
    records = _load_canonical_eval()
    assert len(records) == PUBLISHED_N_QUERIES, (
        f"Expected {PUBLISHED_N_QUERIES} queries in canonical JSON, got {len(records)}"
    )


def test_em_strict_matches_published() -> None:
    records = _load_canonical_eval()
    em = _compute_em(records)
    assert abs(em - PUBLISHED_EM_OBSERVED) < 0.005, (
        f"Recomputed EM strict {em:.3f} differs from published {PUBLISHED_EM_OBSERVED}"
    )


def test_em_strict_within_published_ci95() -> None:
    """Guard: recomputed EM must be within the published BCa CI95 [0.48, 0.67]."""
    records = _load_canonical_eval()
    em = _compute_em(records)
    assert PUBLISHED_CI95_LO <= em <= PUBLISHED_CI95_HI, (
        f"EM strict {em:.3f} outside published CI95 BCa "
        f"[{PUBLISHED_CI95_LO}, {PUBLISHED_CI95_HI}]"
    )


def test_required_fields_present() -> None:
    records = _load_canonical_eval()
    required = {"question_id", "exact_match", "config", "answer", "ground_truth"}
    for rec in records[:5]:
        missing = required - set(rec.keys())
        assert not missing, f"Fields missing from canonical JSON: {missing}"


def test_config_is_graph_rag() -> None:
    records = _load_canonical_eval()
    configs = {r.get("config") for r in records}
    assert "graph-rag" in configs, (
        f"Expected 'graph-rag' config in canonical JSON, found: {configs}"
    )


def test_no_fatal_errors_in_canonical() -> None:
    records = _load_canonical_eval()
    fatal = [r for r in records if r.get("error") and r["error"] is not None]
    assert len(fatal) == 0, (
        f"{len(fatal)} queries have non-null error in canonical JSON: "
        f"{[r['question_id'] for r in fatal[:5]]}"
    )
