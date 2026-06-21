"""
step_3_11_ood_eval.py
=====================
Phase 3 - Out-of-distribution benchmark evaluation.

Runs the production Graph RAG v4 pipeline (keyword routing plus the merged
semantic fallback, honouring ENABLE_SEMANTIC_FALLBACK) on the held-out OOD
benchmark and compares accuracy against the in-distribution canonical numbers.

Metrics:
  - EM strict   : the keyword token matcher (same definition as step_3_4).
  - EM semantic : Sonnet 4.6 cross-model judge of the Haiku-produced answer
                  against the ground truth (same judge as step_3_9; Sonnet here
                  scores Haiku answers, so no self-enhancement bias).
  - LCR         : logical consistency rate of the production context (deterministic
                  neuro-symbolic check, reused from step_3_4_bis).

Error classification (the methodological finding): each non-gap failure is
labelled
  - routing_fail   : the production template is not among the ground-truth
                     provenance templates, or it returned no rows.
  - synthesis_fail : routing was correct and the context carried the entities,
                     yet the answer is semantically wrong (an LLM synthesis miss).
  - coverage_gap   : the query was flagged KG_COVERAGE_GAP at ground-truth time;
                     reported separately and excluded from the accuracy denominator.

Inputs:
    data/benchmarks/benchmark_ood_v1.jsonl  (status may be UNCURATED)
Outputs:
    results/step_3_11_ood_eval_per_query.csv
    results/step_3_11_ood_eval_summary.csv
    data/evaluation_results_graph-rag-ood_<ts>.json

Usage:
    python step_3_11_ood_eval.py --limit 5     # dry-run
    python step_3_11_ood_eval.py

Cost: Haiku answers (cheap) plus one Sonnet judge call per scored query
(~0.5 USD for 40).

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase, exceptions as neo4j_exc

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY, DATA_DIR,
)
from templates import CYPHER_TEMPLATES
from step_3_4_evaluation import route_question, cypher_rows_to_context, ENABLE_SEMANTIC_FALLBACK
from step_3_4_bis_neuro_symbolic import extract_entity_ids, verify_nodes
from step_3_9_llm_judge import _build_judge_prompt, _parse_judge_output, JUDGE_MODEL

BENCH_DIR = BASE_DIR / "data" / "benchmarks"
OOD_FILE = BENCH_DIR / "benchmark_ood_v1.jsonl"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical reference points (strict / semantic) for the delta report.
CANONICAL_V2 = (59.0, 41.0)
CANONICAL_V4 = (66.0, 63.0)

JUDGE_TEMPERATURE = 0.0
MAX_JUDGE_TOKENS = 512
MIN_LCR = 0.5
STOPWORDS = {
    "the", "and", "or", "is", "are", "in", "of", "to", "a", "an", "for", "with",
    "as", "at", "by", "from", "that", "this", "be", "not", "no", "yes", "both",
    "also", "via", "per", "if",
}

PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def em_strict(answer: str, ground_truth: str) -> bool:
    """Keyword token matcher, identical in spirit to step_3_4.compute_exact_match."""
    if not answer or not ground_truth:
        return False
    raw = re.findall(r"[A-Za-z0-9]+(?:[.,][0-9]+)?|%", ground_truth)
    kws = [t.lower() for t in raw if t.lower() not in STOPWORDS and len(t) >= 1]
    if not kws:
        return False
    low = answer.lower()
    return (sum(1 for k in kws if k in low) / len(kws)) >= 0.5


GRAPH_RAG_PROMPT = (
    "You answer questions about an industrial-symbiosis knowledge graph using "
    "ONLY the rows provided. Answer the specific question directly in one or two "
    "sentences, stating the exact value, id, article, name or count. If it asks "
    "'how many', count the rows. If it asks to compare, state both sides and the "
    "difference. If the rows do not contain the answer, reply exactly: "
    "'Not available in the knowledge graph.'\n\n"
    "Rows:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD benchmark evaluation")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N (dry-run)")
    parser.add_argument("--ood", type=Path, default=OOD_FILE)
    parser.add_argument("--version", default="v6", help="label recorded in outputs")
    args = parser.parse_args()

    if not args.ood.exists():
        logger.error("OOD benchmark not found: %s (run build_ood_ground_truth.py)", args.ood)
        return
    queries = [json.loads(line) for line in args.ood.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        queries = queries[: args.limit]

    statuses = {q.get("status") for q in queries}
    if statuses == {"UNCURATED"}:
        logger.warning("Benchmark is UNCURATED: results are PRELIMINARY pending manual approval.")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        logger.error("anthropic SDK not installed: %s", exc)
        return
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except (neo4j_exc.ServiceUnavailable, neo4j_exc.AuthError) as exc:
        logger.error("Neo4j unavailable: %s", exc)
        return

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    rows: list[dict] = []
    tok_in = tok_out = 0
    t0 = time.time()

    with driver.session(database=NEO4J_DATABASE) as session:
        for i, q in enumerate(queries, start=1):
            question, gt = q["nl_question"], q.get("ground_truth", "")
            is_gap = q.get("coverage") == "KG_COVERAGE_GAP"

            # Production routing + retrieval (single template, as in run_graph_rag).
            template_id = route_question(question)
            try:
                rows_db = session.execute_read(
                    lambda tx, c=CYPHER_TEMPLATES[template_id]: tx.run(c).data()
                )
            except Exception as exc:
                rows_db = []
                logger.debug("cypher failed for %s: %s", q["id"], exc)
            context = cypher_rows_to_context(rows_db)

            # Production answer (Haiku).
            answer = ""
            if context and not context.startswith("Cypher error"):
                try:
                    msg = client.messages.create(
                        model=LLM_MODEL, max_tokens=320, temperature=LLM_TEMPERATURE,
                        messages=[{"role": "user", "content":
                                   GRAPH_RAG_PROMPT.format(context=context, question=question)}],
                    )
                    answer = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
                    tok_in += msg.usage.input_tokens
                    tok_out += msg.usage.output_tokens
                except Exception as exc:
                    logger.warning("answer call failed for %s: %s", q["id"], exc)

            # Deterministic LCR on the production context.
            verified, missing = verify_nodes(session, extract_entity_ids(context))
            denom = len(verified) + len(missing)
            lcr = (len(verified) / denom) if denom else 0.0

            # Strict + semantic scoring (skip gap queries for accuracy).
            strict = sem = None
            if not is_gap:
                strict = em_strict(answer, gt)
                try:
                    jmsg = client.messages.create(
                        model=JUDGE_MODEL, max_tokens=MAX_JUDGE_TOKENS, temperature=JUDGE_TEMPERATURE,
                        messages=[{"role": "user", "content": _build_judge_prompt(question, gt, answer)}],
                    )
                    raw = "".join(b.text for b in jmsg.content if hasattr(b, "text"))
                    sem, _ = _parse_judge_output(raw)
                    tok_in += jmsg.usage.input_tokens
                    tok_out += jmsg.usage.output_tokens
                except Exception as exc:
                    logger.warning("judge failed for %s: %s", q["id"], exc)
                    sem = strict

            # Error classification.
            provenance = q.get("provenance_templates", [])
            routing_ok = (template_id in provenance) and bool(rows_db)
            if is_gap:
                err = "coverage_gap"
            elif sem:
                err = "correct"
            elif not routing_ok:
                err = "routing_fail"
            elif lcr < MIN_LCR:
                err = "routing_fail"   # routed in-set but context lacked real entities
            else:
                err = "synthesis_fail"

            rows.append({
                "id": q["id"], "category": q["category"], "difficulty": q["difficulty"],
                "coverage": q.get("coverage"), "production_template": template_id,
                "routing_ok": routing_ok, "lcr": round(lcr, 3),
                "em_strict": None if strict is None else int(strict),
                "em_semantic": None if sem is None else int(sem),
                "error_class": err, "answer": answer[:300],
            })
            if i % 5 == 0 or i == len(queries):
                logger.info("  scored %d/%d (%.1fs)", i, len(queries), time.time() - t0)

    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "step_3_11_ood_eval_per_query.csv", index=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (DATA_DIR / f"evaluation_results_graph-rag-ood_{ts}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    scored = df[df["em_semantic"].notna()]
    n_scored = len(scored)
    strict_pct = round(100.0 * scored["em_strict"].mean(), 1) if n_scored else 0.0
    sem_pct = round(100.0 * scored["em_semantic"].mean(), 1) if n_scored else 0.0
    lcr_mean = round(float(df["lcr"].mean()), 3)
    err_counts = df["error_class"].value_counts().to_dict()
    cost = tok_in / 1e6 * PRICE_IN_PER_MTOK + tok_out / 1e6 * PRICE_OUT_PER_MTOK

    summary = [
        {"metric": "n_total", "value": len(df)},
        {"metric": "n_scored", "value": n_scored},
        {"metric": "n_coverage_gap", "value": int((df["error_class"] == "coverage_gap").sum())},
        {"metric": "ood_em_strict_pct", "value": strict_pct},
        {"metric": "ood_em_semantic_pct", "value": sem_pct},
        {"metric": "ood_lcr_mean", "value": lcr_mean},
        {"metric": "delta_strict_vs_v2", "value": round(strict_pct - CANONICAL_V2[0], 1)},
        {"metric": "delta_semantic_vs_v2", "value": round(sem_pct - CANONICAL_V2[1], 1)},
        {"metric": "delta_strict_vs_v4", "value": round(strict_pct - CANONICAL_V4[0], 1)},
        {"metric": "delta_semantic_vs_v4", "value": round(sem_pct - CANONICAL_V4[1], 1)},
        {"metric": "routing_fail", "value": err_counts.get("routing_fail", 0)},
        {"metric": "synthesis_fail", "value": err_counts.get("synthesis_fail", 0)},
        {"metric": "judge_model", "value": JUDGE_MODEL},
        {"metric": "semantic_fallback", "value": ENABLE_SEMANTIC_FALLBACK},
        {"metric": "version", "value": args.version},
        {"metric": "ood_file", "value": args.ood.name},
        {"metric": "est_cost_usd", "value": round(cost, 4)},
    ]
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "step_3_11_ood_eval_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("  OOD EVALUATION SUMMARY  (semantic_fallback=%s)" % ENABLE_SEMANTIC_FALLBACK)
    print("=" * 60)
    print(f"  scored {n_scored} / {len(df)}  (coverage gaps excluded: "
          f"{err_counts.get('coverage_gap', 0)})")
    print(f"  OOD  EM strict   = {strict_pct:.1f}%   (v2 {CANONICAL_V2[0]}, v4 {CANONICAL_V4[0]})")
    print(f"  OOD  EM semantic = {sem_pct:.1f}%   (v2 {CANONICAL_V2[1]}, v4 {CANONICAL_V4[1]})")
    print(f"  OOD  LCR (mean)  = {lcr_mean:.3f}")
    print(f"  error classes    : {err_counts}")
    print(f"  est cost         : ${cost:.4f}")


if __name__ == "__main__":
    main()
