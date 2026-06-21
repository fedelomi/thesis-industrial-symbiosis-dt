"""FW9b Task 1 + Task 5: the BGE router is deterministic across repeated calls.

No API, no Neo4j. Loads the local BGE backend (skipped if sentence-transformers is
unavailable) and asserts that routing all OOD questions five times yields bit-
identical ordered output, for both the ranked score() and the multi pool.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers (BGE backend) not installed",
)

OOD_FILE = BASE / "data" / "benchmarks" / "benchmark_ood_v1.jsonl"


def _ood_questions() -> list[str]:
    if not OOD_FILE.is_file():
        pytest.skip(f"OOD benchmark not found: {OOD_FILE}")
    return [
        json.loads(line)["nl_question"]
        for line in OOD_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def bge_router():
    from semantic_router import SemanticRouter, build_backend

    return SemanticRouter(backend=build_backend("st"))


def test_score_is_identical_across_five_calls(bge_router) -> None:
    """router.score(q) returns the same ordered ranking on five repeated calls."""
    for q in _ood_questions():
        runs = [tuple(bge_router.score(q)) for _ in range(5)]
        assert len(set(runs)) == 1, f"non-deterministic score for: {q!r}"


def test_multi_pool_is_identical_across_five_calls(bge_router) -> None:
    """route_question_multi returns the same ordered pool on five repeated calls."""
    from prompt5_retrieval import route_question_multi

    for q in _ood_questions():
        runs = [
            tuple(route_question_multi(q, n=5, use_semantic=True, router=bge_router))
            for _ in range(5)
        ]
        assert len(set(runs)) == 1, f"non-deterministic pool for: {q!r}"


def test_score_tie_break_is_template_id_ascending() -> None:
    """On an exact score tie the order is template_id ascending (explicit stable sort)."""
    from semantic_router import SemanticRouter

    class _ConstBackend:
        name = "const"
        default_conf_threshold = 0.0
        default_margin = 0.0
        use_disk_cache = False

        def prepare(self, corpus):
            del corpus

        def encode(self, texts):
            import numpy as np

            return np.ones((len(list(texts)), 4), dtype="float32") / 2.0

    r = SemanticRouter(backend=_ConstBackend())
    ranked = r.score("anything")
    tids = [t for t, _ in ranked]
    assert tids == sorted(tids), "tie-break must be template_id ascending"
