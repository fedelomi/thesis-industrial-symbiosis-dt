"""
build_ood_ground_truth.py
=========================
Phase 3 - Ground truth construction for the OOD benchmark.

For each curated OOD query this builds a provisional ground truth that is
grounded in the actual knowledge graph (not invented by an LLM from nothing):

  1. Retrieve the top-3 candidate templates by semantic match (semantic_router).
  2. Execute each on Neo4j and union the returned rows (FREE).
  3. Deterministic neuro-symbolic validation (reuses step_3_4_bis): extract the
     entity ids in the union context and verify they exist as nodes; report a
     consistency rate. This layer is NON-LLM by design.
  4. A single Sonnet 4.6 pass writes the answer FROM the union context, or
     returns COVERAGE_GAP when the rows do not contain an answer.
  5. A query is flagged KG_COVERAGE_GAP when Sonnet reports a gap OR the
     deterministic layer finds no real supporting entity. That flag is a
     finding, not noise: it localises where the KG (or the template set) cannot
     answer an independently posed question.

Every record is written with status="UNCURATED": the file is provisional until
the researcher approves it manually.

Inputs:
    data/benchmarks/ood_candidates_curated.json
Outputs:
    data/benchmarks/benchmark_ood_v1.jsonl   (one JSON object per line)

Usage:
    python build_ood_ground_truth.py --limit 5     # dry-run
    python build_ood_ground_truth.py               # full set

Cost: one Sonnet 4.6 call per query (~400 in, ~120 out). 40 queries < 0.60 USD.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase, exceptions as neo4j_exc

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, ANTHROPIC_API_KEY,
)
from templates import CYPHER_TEMPLATES
from step_3_4_bis_neuro_symbolic import extract_entity_ids, verify_nodes
from step_3_4_evaluation import route_question

BENCH_DIR = BASE_DIR / "data" / "benchmarks"
CURATED = BENCH_DIR / "ood_candidates_curated.json"
OUT_JSONL = BENCH_DIR / "benchmark_ood_v1.jsonl"

GT_MODEL = "claude-sonnet-4-6"
GT_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 320
TOP_K = 3
MAX_ROWS_PER_TEMPLATE = 15
MIN_CONSISTENCY = 0.5  # below this the union has no real supporting entity

PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def rows_to_text(rows: list[dict], cap: int = MAX_ROWS_PER_TEMPLATE) -> str:
    return "\n".join(" | ".join(f"{k}: {v}" for k, v in row.items()) for row in rows[:cap])


def execute_template(session, template_id: str) -> list[dict]:
    cypher = CYPHER_TEMPLATES[template_id]
    try:
        return session.execute_read(lambda tx, c=cypher: tx.run(c).data())
    except Exception as exc:  # a template may not match an OOD question's intent
        logger.debug("template %s failed: %s", template_id, exc)
        return []


def build_gt_prompt(question: str, context: str) -> str:
    return (
        "You build an evaluation ground truth from a knowledge graph. Using ONLY "
        "the rows below, write the single most precise correct answer to the "
        "question (the exact value, id, article, name, count or comparison). "
        "Be terse: at most one or two sentences, state the facts only, NO preamble "
        "(do not write 'Based on the rows'), NO markdown, NO bullet lists. "
        "If the rows do NOT contain enough to answer, reply with exactly the token "
        "COVERAGE_GAP and nothing else.\n\n"
        f"Rows:\n{context if context else '(no rows returned)'}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOD ground truth")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N (dry-run)")
    parser.add_argument("--curated", type=Path, default=CURATED)
    parser.add_argument("--out", type=Path, default=OUT_JSONL,
                        help="output jsonl (use a v2 path to preserve v1)")
    args = parser.parse_args()

    if not args.curated.exists():
        logger.error("Curated set not found: %s (run curate_ood_benchmark.py)", args.curated)
        return
    queries = json.loads(args.curated.read_text(encoding="utf-8"))
    if args.limit > 0:
        queries = queries[: args.limit]

    try:
        from anthropic import Anthropic
        from semantic_router import get_default_router
    except ImportError as exc:
        logger.error("dependency missing: %s", exc)
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except (neo4j_exc.ServiceUnavailable, neo4j_exc.AuthError) as exc:
        logger.error("Neo4j unavailable: %s", exc)
        return

    router = get_default_router()
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    records: list[dict] = []
    tok_in = tok_out = 0
    n_gap = 0
    t0 = time.time()

    with driver.session(database=NEO4J_DATABASE) as session:
        for i, q in enumerate(queries, start=1):
            question = q["nl_question"]
            top = router.route(question, top_k=TOP_K)
            # Union the semantic top-3 WITH the production route (route_question).
            # The original builder used only the raw semantic top-3, which could
            # miss the template production actually routes to, producing false
            # coverage gaps. Including route_question fixes that.
            prod_route = route_question(question)
            union_ids = list(dict.fromkeys([tid for tid, _ in top] + [prod_route]))
            provenance = union_ids

            union_rows: list[dict] = []
            for tid in union_ids:
                union_rows.extend(execute_template(session, tid))
            context = rows_to_text(union_rows, cap=3 * MAX_ROWS_PER_TEMPLATE)

            entities = extract_entity_ids(context)
            verified, missing = verify_nodes(session, entities)
            denom = len(verified) + len(missing)
            consistency = (len(verified) / denom) if denom else 0.0

            prompt = build_gt_prompt(question, context)
            try:
                msg = client.messages.create(
                    model=GT_MODEL, max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=GT_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
                tok_in += msg.usage.input_tokens
                tok_out += msg.usage.output_tokens
            except Exception as exc:
                logger.warning("GT call failed for %s: %s", q["id"], exc)
                answer = "COVERAGE_GAP"

            gap = answer.strip().upper().startswith("COVERAGE_GAP") or consistency < MIN_CONSISTENCY
            coverage = "KG_COVERAGE_GAP" if gap else "covered"
            if gap:
                n_gap += 1

            records.append({
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "nl_question": question,
                "ground_truth": "" if gap else answer,
                "coverage": coverage,
                "provenance_templates": provenance,
                "production_route": prod_route,
                "neuro_symbolic": {
                    "consistency_rate": round(consistency, 3),
                    "verified_entities": verified,
                    "missing_entities": missing,
                },
                "generator_model": q.get("generator_model", ""),
                "gt_model": GT_MODEL,
                "status": "UNCURATED",
            })
            if i % 5 == 0 or i == len(queries):
                logger.info("  GT built %d/%d (%.1fs)", i, len(queries), time.time() - t0)

    driver.close()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    cost = tok_in / 1e6 * PRICE_IN_PER_MTOK + tok_out / 1e6 * PRICE_OUT_PER_MTOK
    logger.info("wrote %d records (%d coverage gaps) -> %s", len(records), n_gap, args.out.name)
    logger.info("tokens in=%d out=%d  est cost=$%.4f", tok_in, tok_out, cost)
    print(f"\nBuilt ground truth for {len(records)} queries "
          f"({n_gap} KG_COVERAGE_GAP), est cost ${cost:.4f}. "
          f"File flagged UNCURATED. Next: python step_3_11_ood_eval.py")


if __name__ == "__main__":
    main()
