"""
guardrail_common.py — shared helpers for the FW1 runtime-guardrail tool chain
(step_3_14 context-mode audit, step_3_15 LLM answer re-run, step_3_16 runtime
guardrail, step_3_17 third-judge evaluation).

Inputs:  data/evaluation_results_*.json artefacts (stored per-query context,
         answer, ground_truth), no Neo4j access required.
Outputs: none (library module).
Gate:    em_strict() must reproduce the stored `exact_match` flags of the
         canonical artefact bit-exactly (verified by tests/test_fw1_guardrail.py).

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

# Canonical in-distribution artefact (EM strict 0.59, context-only mode).
CANONICAL_EVAL_JSON = DATA / "evaluation_results_graph-rag_20260517_155731.json"
# OOD artefact with stored contexts (38 queries, LLM-in-loop answers).
OOD_EVAL_JSON = DATA / "evaluation_results_graph-rag-ood_20260603_102245.json"

# Indicative API prices (USD per million tokens), overridable via env.
DEFAULT_PRICES = {
    "haiku_in": 1.00, "haiku_out": 5.00,
    "sonnet_in": 3.00, "sonnet_out": 15.00,
    "opus_in": 15.00, "opus_out": 75.00,
}

STOPWORDS = {
    "the", "and", "or", "is", "are", "in", "of", "to", "a", "an",
    "for", "with", "as", "at", "by", "from", "that", "this", "be",
    "not", "no", "yes", "both", "also", "via", "per", "if",
}


def em_strict(answer: str | None, ground_truth: str | None) -> bool | None:
    """Replicate step_3_4_evaluation.compute_exact_match() for a single pair.

    Token-level keyword match: >= 50% of the significant ground-truth tokens
    must appear (substring, lowercase) in the answer. Returns None when the
    pair is not scorable (empty answer/ground truth or no keywords).
    """
    if not answer or not ground_truth:
        return None
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:[.,][0-9]+)?|%", ground_truth)
    keywords = [t.lower() for t in raw_tokens
                if t.lower() not in STOPWORDS and len(t) >= 1]
    if not keywords:
        return None
    ans_lower = answer.lower()
    matches = sum(1 for kw in keywords if kw in ans_lower)
    return (matches / len(keywords)) >= 0.5


def answer_generation_prompt(context: str, question: str) -> str:
    """The exact answer-generation prompt of step_3_4_evaluation.run_graph_rag()."""
    return (
        "You answer questions about an industrial-symbiosis knowledge graph "
        "using ONLY the rows provided. Rules:\n"
        "1. Answer the SPECIFIC question directly in one or two sentences, "
        "stating the exact value, id, article, name or count it asks for.\n"
        "2. If the question asks 'how many', COUNT the matching rows, give the "
        "number first, then list the matching ids.\n"
        "3. If it asks to compare or says 'vs', state BOTH sides and the "
        "difference explicitly.\n"
        "4. Do NOT paste raw rows or unrelated fields; use only what answers "
        "the question.\n"
        "5. If the rows do not contain the answer, reply exactly: "
        "'Not available in the knowledge graph.'\n\n"
        f"Rows:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def verifier_prompt(context: str, question: str, candidate: str) -> str:
    """Runtime-guardrail verification prompt (Sonnet-class gate, FW1).

    The gate checks grounding and directness, NOT agreement with any ground
    truth: ground truth must never reach the runtime path.
    """
    return (
        "You are a verification gate for a knowledge-graph QA system. "
        "Judge the CANDIDATE ANSWER against the ROWS only.\n"
        "PASS criteria (all must hold):\n"
        "1. The answer directly addresses the specific question (value, id, "
        "article, name or count it asks for).\n"
        "2. Every factual claim in the answer is supported by the rows.\n"
        "3. The answer is natural language, not a paste of raw rows.\n"
        "If the rows do not contain the information and the answer says so, "
        "that is a PASS.\n\n"
        f"Rows:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Candidate answer: {candidate}\n\n"
        "Reply with a single JSON object, no prose: "
        '{"pass": true|false, "feedback": "<one sentence: what is missing or '
        'wrong, empty if pass>"}'
    )


def retry_prompt(context: str, question: str, feedback: str) -> str:
    """Second-attempt generation prompt with structured verifier feedback."""
    base = answer_generation_prompt(context, question)
    return base.replace(
        "Answer:",
        f"A previous attempt was rejected by a verification gate with this "
        f"feedback: {feedback}\nFix exactly that and answer again.\n\nAnswer:",
    )


@dataclass
class EvalRecord:
    """One benchmark query with its stored retrieval context."""

    question_id: str
    nl_question: str
    ground_truth: str
    context: str
    stored_answer: str | None = None
    stored_exact_match: bool | None = None
    category: str = ""


def load_eval_records(path: Path) -> list[EvalRecord]:
    """Load per-query records from an evaluation_results_*.json artefact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    records: list[EvalRecord] = []
    for r in items:
        records.append(EvalRecord(
            question_id=str(r.get("question_id") or r.get("id") or ""),
            nl_question=r.get("nl_question") or r.get("question") or "",
            ground_truth=r.get("ground_truth") or "",
            context=r.get("context") or "",
            stored_answer=r.get("answer"),
            stored_exact_match=r.get("exact_match"),
            category=r.get("category") or "",
        ))
    LOG.info("Loaded %d records from %s", len(records), path)
    return records


@dataclass
class CostMeter:
    """Token and cost accounting across API calls (indicative prices)."""

    prices: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    tokens: dict[str, int] = field(default_factory=dict)

    def add(self, family: str, tokens_in: int, tokens_out: int) -> None:
        self.tokens[f"{family}_in"] = self.tokens.get(f"{family}_in", 0) + tokens_in
        self.tokens[f"{family}_out"] = self.tokens.get(f"{family}_out", 0) + tokens_out

    def usd(self) -> float:
        total = 0.0
        for key, n_tok in self.tokens.items():
            total += (n_tok / 1e6) * self.prices.get(key, 0.0)
        return total


def extract_json_verdict(text: str) -> tuple[bool, str]:
    """Parse the verifier JSON; fail-open=False (treat unparseable as FAIL)."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return False, "unparseable verifier output"
    try:
        obj = json.loads(match.group(0))
        return bool(obj.get("pass", False)), str(obj.get("feedback", ""))[:300]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return False, f"verifier JSON error: {exc}"


def _resolve_api_key() -> str:
    """Resolve ANTHROPIC_API_KEY via the canonical config.py (.env loader),
    falling back to a direct dotenv load and then to the bare environment.
    The key is never logged or written to any artefact."""
    import os
    try:
        from config import ANTHROPIC_API_KEY  # canonical .env loading path
        if ANTHROPIC_API_KEY:
            return ANTHROPIC_API_KEY
    except Exception:  # noqa: BLE001 — config may require unrelated vars
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except ImportError:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found: set it in Phase3_GraphRAG/.env "
            "or in the environment before running the API steps.")
    return key


def make_anthropic_caller(model: str, family: str, meter: CostMeter,
                          temperature: float | None = 0.0,
                          max_tokens: int = 300) -> Callable[[str], str]:
    """Return a prompt->text callable bound to one Anthropic model.

    The anthropic import is lazy so that offline tools and tests never need
    the SDK or an API key. Pass temperature=None for models that deprecate
    the temperature parameter (e.g. claude-opus-4-8): the provider default
    sampling is used and this is recorded by the caller in its summary.
    """
    from anthropic import Anthropic  # lazy import by design
    client = Anthropic(api_key=_resolve_api_key())

    def _call(prompt: str) -> str:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        message = client.messages.create(**kwargs)
        usage = getattr(message, "usage", None)
        if usage is not None:
            meter.add(family, int(usage.input_tokens), int(usage.output_tokens))
        return "".join(b.text for b in message.content if hasattr(b, "text"))

    return _call
