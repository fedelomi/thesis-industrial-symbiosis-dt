"""
prompt5_retrieval.py
====================
Phase 3 - Graph RAG, PROMPT 5 ablation (v7 candidate system).

New retrieval/synthesis units for the 4-arm ablation (A0 keyword-top1 -> A1 BGE-top1
-> A2 BGE multi-template union -> A3 + densification). Kept in a SEPARATE module so the
canonical `step_3_4_evaluation.py` (route_question, cypher_rows_to_context) stays byte-for-byte
unchanged and the canonical v2 (59/41) numbers are not at risk.

Units
-----
  route_question_multi  : top-N gated candidate pool (keyword candidates + semantic ranking).
  union_template_rows   : run + dedup + cap the rows of several templates into one context set.
  densify_context       : declarative formatter; verbatim anchor tokens (ids, article numbers,
                          temperatures, euro values) are preserved so strict EM does not break.
  refine_error_class    : metric refinement that separates under_retrieval (single/partial
                          template missed provenance facts) from a true synthesis_fail.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional, Sequence

from step_3_4_evaluation import (
    keyword_candidates,
    entity_fallback,
    ENABLE_SEMANTIC_FALLBACK,
)

# Drop-in sentinel identical to step_3_4_evaluation.cypher_rows_to_context.
NO_DATA_SENTINEL = "No data found in knowledge graph."


# --------------------------------------------------------------------------- C2
def route_question_multi(
    question: str,
    n: int = 5,
    use_semantic: Optional[bool] = None,
    router: Optional[object] = None,
    conf_floor: float = 0.0,
) -> list[str]:
    """Return up to `n` template ids to fire for multi-template retrieval.

    Keyword candidates come first (priority order, deterministic). The semantic
    ranking then fills the remaining slots with templates whose confidence clears
    `conf_floor`. Falls back to the legacy entity heuristic when nothing fires.
    The canonical single-template `route_question` is left untouched.
    """
    if use_semantic is None:
        use_semantic = ENABLE_SEMANTIC_FALLBACK
    q_lower = question.lower()
    pool: list[str] = list(keyword_candidates(q_lower))

    if use_semantic and len(pool) < n:
        r = router
        if r is None:
            from semantic_router import get_default_router
            r = get_default_router()
        for tid, conf in r.score(question):
            if len(pool) >= n:
                break
            if conf < conf_floor:
                continue
            if tid not in pool:
                pool.append(tid)

    if not pool:
        pool = [entity_fallback(q_lower)]
    return pool[:n]


# --------------------------------------------------------------------------- C2
def union_template_rows(
    template_ids: Sequence[str],
    run_template: Callable[[str], list[dict]],
    row_cap: int = 40,
) -> list[dict]:
    """Union the rows returned by each template, deduped by content, capped.

    `run_template(template_id) -> list[dict]` executes one template's Cypher. Kept
    as an injected callable so the union logic is pure and testable without Neo4j.
    """
    seen: set[str] = set()
    rows: list[dict] = []
    for tid in template_ids:
        for row in run_template(tid):
            key = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= row_cap:
                return rows
    return rows


# --------------------------------------------------------------------------- C3
def densify_context(rows: list[dict], max_rows: int = 40) -> str:
    """Flatten Neo4j rows into one declarative line per row.

    Drops the `key: value |` pipe noise of the legacy formatter to lower Haiku's
    parsing load, while keeping every value verbatim so anchor tokens (ids, article
    numbers, temperatures, euro figures) survive for the strict EM matcher.
    """
    if not rows:
        return NO_DATA_SENTINEL
    lines: list[str] = []
    for row in rows[:max_rows]:
        clause = ", ".join(f"{k} {v}" for k, v in row.items())
        lines.append(f"{clause}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- C3
# Controlled standard-constant injection (the only fact allowed beyond the rows).
# A template that is intrinsically ABOUT a named standard may state that standard's
# canonical name even when the rows carry only its clauses (the OOD03 fix). The
# constant is injected once, only when such a template fired, only if absent.
STANDARD_CONSTANTS: dict[str, str] = {
    "ISO50001_ARTICLES": "ISO 50001:2018",
    "ISO50001_PROCESS_COMPLIANCE": "ISO 50001:2018",
}


def densify_context_v2(
    rows: list[dict],
    template_ids: Sequence[str] = (),
    max_rows: int = 40,
) -> str:
    """Densify rows, with a controlled standard-constant header where applicable.

    The body is the deterministic per-row verbalizer of `densify_context` (every
    value verbatim, so strict-EM anchors survive). When one of the fired templates
    is intrinsically about a named standard, that standard's constant name is added
    as a single header line if it is not already present. No other fact is injected.

    Args:
        rows: Unioned Neo4j rows.
        template_ids: The templates that produced the rows (single or multi).
        max_rows: Row cap passed to the body verbalizer.

    Returns:
        The densified context string.
    """
    body = densify_context(rows, max_rows=max_rows)
    if body == NO_DATA_SENTINEL:
        return body
    consts: list[str] = []
    for tid in template_ids:
        const = STANDARD_CONSTANTS.get(tid)
        if const and const not in body and const not in consts:
            consts.append(const)
    if not consts:
        return body
    header = " ".join(f"The applicable standard is {c}." for c in consts)
    return f"{header}\n{body}"


# --------------------------------------------------------------------------- C4
_COUNT_CUES = ("how many", "number of", "count of", "count the")
_COMPARE_CUES = ("compare", "versus", " vs ", "difference between", "higher", "lower",
                 "better", "stronger", "weaker", "more than", "less than")
_AGGREGATE_CUES = ("total", "sum of", "average", "maximum", "minimum", "highest",
                   "lowest", "across all", "for each", "every", "list all", "all the")

_PRUNE_STOPWORDS = {
    "the", "and", "or", "is", "are", "in", "of", "to", "a", "an", "for", "with",
    "as", "at", "by", "from", "that", "this", "be", "not", "which", "what", "who",
    "would", "could", "does", "do", "it", "its", "on", "they",
}


def classify_query_type(question: str) -> str:
    """Coarse query intent: count, compare, aggregate or lookup (deterministic)."""
    q = f" {question.lower()} "
    if any(cue in q for cue in _COUNT_CUES):
        return "count"
    if any(cue in q for cue in _COMPARE_CUES):
        return "compare"
    if any(cue in q for cue in _AGGREGATE_CUES):
        return "aggregate"
    return "lookup"


def _question_tokens(question: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", question.lower())
            if t not in _PRUNE_STOPWORDS and len(t) >= 2}


def prune_rows(
    rows: list[dict],
    question: str,
    prune_threshold: int = 40,
    min_keep: int = 3,
) -> list[dict]:
    """Drop rows irrelevant to a lookup question, never for count/compare/aggregate.

    Count, compare and aggregate queries need every row (a dropped row changes the
    count or hides one side of a comparison), so they are returned unchanged. Lookup
    queries are pruned only when the union exceeds `prune_threshold` rows (the oracle
    showed unions stay small, so this is a noise guard, not the common path). Pruning
    keeps the rows whose values share a token with the question, in original order;
    if too few match it falls back to the highest-overlap rows by a stable key.

    Args:
        rows: Unioned rows.
        question: The natural-language question.
        prune_threshold: Only prune a lookup union larger than this.
        min_keep: Never return fewer than this many rows.

    Returns:
        The (possibly pruned) rows, deterministic order preserved.
    """
    if classify_query_type(question) != "lookup":
        return rows
    if len(rows) <= prune_threshold:
        return rows
    q_tokens = _question_tokens(question)
    scored = []
    for i, row in enumerate(rows):
        text = " ".join(str(v) for v in row.values()).lower()
        overlap = len(q_tokens & set(re.findall(r"[a-z0-9]+", text)))
        scored.append((i, overlap))
    relevant = [i for i, s in scored if s > 0]
    if len(relevant) >= min_keep:
        keep = set(relevant)
    else:
        top = sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:max(min_keep, prune_threshold)]
        keep = {i for i, _ in top}
    return [row for i, row in enumerate(rows) if i in keep]


# ---------------------------------------------------------------------- metric
def refine_error_class(
    is_gap: bool,
    sem_correct: bool,
    routing_ok: bool,
    lcr: float,
    retrieval_complete: bool,
    min_lcr: float,
) -> str:
    """Error taxonomy that splits under_retrieval out of synthesis_fail.

    The legacy step_3_11 classifier called every routed-but-wrong answer a
    synthesis_fail even when the single template did not carry the provenance facts
    (under-retrieval masked as synthesis). With multi-template retrieval we can tell
    them apart: synthesis_fail now means the facts WERE retrieved yet not composed.
    """
    if is_gap:
        return "coverage_gap"
    if sem_correct:
        return "correct"
    if not routing_ok:
        return "routing_fail"
    if lcr < min_lcr:
        return "routing_fail"
    if not retrieval_complete:
        return "under_retrieval"
    return "synthesis_fail"
