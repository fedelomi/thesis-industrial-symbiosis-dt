"""
step_3_21_multitemplate_cot_indist.py - v7 multi-template retrieval path
combined with the structured chain-of-thought variant (Lever 2 of the FW11
roadmap of Section 6.3.2). Measured on the CANONICAL 100-query in-distribution
benchmark at constant Haiku generator and constant Opus third-judge for direct
comparability with step_3_20 (flat) and step_3_4 (canonical).

Motivation: step_3_20 measured the multi-template retrieval lever at EM
semantic 0.65 (Opus third-judge, +24 pp vs canonical 0.41). The residual
compositional-synthesis floor on derived and comparative queries is the
FW9-ter target, but a simpler intervention is to expose the synthesis chain
explicitly inside the same single forward pass via structured CoT (the LLM
decomposes the question into atomic fact extraction + reasoning chain +
composed answer). This step measures whether the structured-CoT lever
provides a marginal lift over the flat prompt of step_3_20 at the same
retrieval (multi-template top-5 union + prune + densify).

Pre-registered predictions (2026-06-23, before the run):
  P1. Overall EM semantic rises from 0.65 (step_3_20) toward 0.70-0.75.
  P2. The lift is concentrated on multi-hop (Cat B) and comparative (Cat C)
      where single-pass synthesis is the bottleneck; factual (Cat A) stays
      near 0.94 (already saturated by retrieval).
  P3. JSON parse failure rate < 5% (Haiku 4.5 is robust to structured output).

Pipeline per query: route_question_multi(n=5) -> union_template_rows ->
prune_rows -> densify_context_v2 -> Haiku structured-CoT answer (JSON with
extracted_facts, reasoning_chain, final_answer, confidence) -> parse JSON
-> EM strict on final_answer field only. Canonical files untouched.

Inputs:  Neo4j live (config NEO4J_*), ANTHROPIC_API_KEY, canonical benchmark
         artefact (questions + ground truths + canonical contexts),
         results/step_3_19_kg_coverage_audit.csv (routing flags)
Outputs: results/step_3_21_multitemplate_cot_indist_per_query.csv
         (judge-ready schema for step_3_17 --guardrail-csv),
         results/step_3_21_multitemplate_cot_indist_summary.csv

Cost:    ~100 Haiku calls with longer output (~0.15-0.20 USD). Requires the
         local Neo4j DBMS to be running. Optional judging: step_3_17
         (+~0.94 USD with Opus).

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from guardrail_common import (
    CANONICAL_EVAL_JSON, RESULTS, CostMeter,
    em_strict, load_eval_records, make_anthropic_caller,
)

LOG = logging.getLogger(__name__)
PER_QUERY_PATH = RESULTS / "step_3_21_multitemplate_cot_indist_per_query.csv"
SUMMARY_PATH = RESULTS / "step_3_21_multitemplate_cot_indist_summary.csv"
AUDIT_PATH = RESULTS / "step_3_19_kg_coverage_audit.csv"


# --- Structured CoT prompt + parser (mirror of step_3_2 helpers) ---

def cot_prompt(context: str, question: str) -> str:
    """Structured CoT generation prompt. Mirror of _v7_answer_prompt_structured
    in step_3_2_graph_rag_pipeline.py so that the two entry points stay
    behaviourally aligned."""
    return (
        "You are answering a question about data centre waste heat recovery "
        "and European energy regulation. Use ONLY the facts in the context "
        "block. Preserve verbatim numeric values, article numbers, "
        "identifiers and unit labels.\n\n"
        "Output ONE JSON object with these four fields and nothing else:\n"
        "  - \"extracted_facts\": list of the verbatim facts from the context "
        "that are relevant to the question.\n"
        "  - \"reasoning_chain\": list of 1-5 short reasoning steps that "
        "compose the extracted facts into the answer.\n"
        "  - \"final_answer\": the final answer string. If the context does "
        "not contain enough information, set this to exactly \"Not available "
        "in the knowledge graph.\".\n"
        "  - \"confidence\": one of \"high\", \"medium\", \"low\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "JSON:"
    )


def parse_cot_answer(raw: str) -> tuple[str, dict]:
    """Extract final_answer and structured payload from JSON output.
    Falls back to raw string if JSON parsing fails (rare on Haiku 4.5)."""
    s = raw.strip()
    if s.lower().startswith("json:"):
        s = s[5:].strip()
    fence = re.match(r"^```(?:json)?\s*(.+?)\s*```\s*$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        payload = json.loads(s)
    except Exception:  # noqa: BLE001
        block = re.search(r"\{.*\}", s, re.DOTALL)
        if block:
            try:
                payload = json.loads(block.group(0))
            except Exception:  # noqa: BLE001
                return raw.strip(), {}
        else:
            return raw.strip(), {}
    final = str(payload.get("final_answer", "")).strip()
    if not final:
        return raw.strip(), payload
    return final, payload


# --- Config ---

@dataclass
class Config:
    eval_json: Path
    gen_model: str = "claude-haiku-4-5-20251001"
    top_n: int = 5
    row_cap: int = 40
    sleep_s: float = 0.05
    dry_run: bool = False
    yes: bool = False
    limit: int | None = None
    skip_existing: bool = False


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the per-query dataframe."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and PER_QUERY_PATH.exists():
        LOG.info("Skipping (output exists): %s", PER_QUERY_PATH)
        return pd.read_csv(PER_QUERY_PATH)

    from neo4j import GraphDatabase
    import config
    from templates import CYPHER_TEMPLATES
    from prompt5_retrieval import (
        densify_context_v2, prune_rows, route_question_multi,
        union_template_rows,
    )

    records = load_eval_records(cfg.eval_json)
    records_to_run = records[:3] if cfg.dry_run else records[: cfg.limit or None]
    LOG.info("Indicative full-run cost: ~0.15-0.20 USD (%d Haiku calls with "
             "longer JSON output). Neo4j must be running at %s.",
             len(records), config.NEO4J_URI)
    if cfg.dry_run:
        LOG.info("DRY RUN: executing only %d queries. Use --full --yes for "
                 "the complete benchmark.", len(records_to_run))
    elif not cfg.yes:
        LOG.error("Refusing the full paid run without --yes (cost gate).")
        sys.exit(2)

    routing_flags: dict[str, str] = {}
    if AUDIT_PATH.exists():
        audit = pd.read_csv(AUDIT_PATH)
        routing_flags = dict(zip(audit["question_id"], audit["class"]))
    else:
        LOG.warning("step_3_19 audit CSV not found: bucket column will be empty.")

    meter = CostMeter()
    call_gen = make_anthropic_caller(cfg.gen_model, "haiku", meter)
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    rows: list[dict] = []
    parse_failures = 0
    with driver.session(database=config.NEO4J_DATABASE) as session:

        def run_template(tid: str) -> list[dict]:
            cypher = CYPHER_TEMPLATES.get(tid)
            if not cypher:
                return []
            try:
                return session.execute_read(lambda tx: tx.run(cypher).data())
            except Exception:  # noqa: BLE001
                return []

        for i, r in enumerate(records_to_run, 1):
            try:
                pool = route_question_multi(r.nl_question, n=cfg.top_n,
                                            use_semantic=True)
                union = prune_rows(
                    union_template_rows(pool, run_template,
                                        row_cap=cfg.row_cap),
                    r.nl_question)
                context = densify_context_v2(union, template_ids=pool)
                raw = call_gen(cot_prompt(context, r.nl_question))
                answer, payload = parse_cot_answer(raw)
                if not payload:
                    parse_failures += 1
            except Exception as exc:  # noqa: BLE001
                LOG.error("Query %s failed: %s", r.question_id, exc)
                pool, union, context, answer, payload = [], [], "", "", {}
            is_abstain = answer.strip().lower().startswith("not available")
            em_new = em_strict(answer, r.ground_truth)
            ctx_has_gt = em_strict(context, r.ground_truth)
            confidence = str(payload.get("confidence", "")) if payload else ""
            n_facts = len(payload.get("extracted_facts", [])) if payload else 0
            n_steps = len(payload.get("reasoning_chain", [])) if payload else 0
            rows.append({
                "question_id": r.question_id,
                "category": r.category,
                "status": "abstain" if is_abstain else "pass",
                "final_answer": answer,
                "em_final": em_new,
                "abstained": is_abstain,
                "em_context_only_canonical": r.stored_exact_match,
                "ctx_contains_gt_keywords": ctx_has_gt,
                "pool": "|".join(pool),
                "union_rows": len(union),
                "step_3_19_bucket": routing_flags.get(r.question_id, ""),
                "gen_model": cfg.gen_model,
                "cot_confidence": confidence,
                "cot_n_facts": n_facts,
                "cot_n_steps": n_steps,
                "cot_parse_ok": bool(payload),
            })
            LOG.info("%d/%d %s pool=%d rows=%d em=%s ctx_gt=%s conf=%s "
                     "facts=%d steps=%d [%s]",
                     i, len(records_to_run), r.question_id, len(pool),
                     len(union), em_new, ctx_has_gt, confidence,
                     n_facts, n_steps,
                     routing_flags.get(r.question_id, "-"))
            time.sleep(cfg.sleep_s)
    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(PER_QUERY_PATH, index=False)

    routing = df[df["step_3_19_bucket"] == "routing_under_retrieval"]
    summary = pd.DataFrame([{
        "n_queries": len(df),
        "em_strict_multitemplate_cot": round(df["em_final"].astype(float).mean(), 4),
        "em_strict_singletemplate_nl_ref": 0.45,
        "em_strict_multitemplate_flat_ref": 0.69,
        "ctx_contains_gt_overall": round(
            df["ctx_contains_gt_keywords"].astype(float).mean(), 4),
        "routing29_n": len(routing),
        "routing29_ctx_contains_gt": (round(routing["ctx_contains_gt_keywords"]
                                            .astype(float).mean(), 4)
                                      if len(routing) else None),
        "routing29_em_strict": (round(routing["em_final"].astype(float).mean(), 4)
                                if len(routing) else None),
        "abstain_rate": round(df["abstained"].mean(), 4),
        "cot_parse_ok_rate": round(df["cot_parse_ok"].mean(), 4),
        "cot_parse_failures": parse_failures,
        "top_n": cfg.top_n,
        "gen_model": cfg.gen_model,
        "tokens": dict(meter.tokens).__repr__(),
        "est_cost_usd": round(meter.usd(), 4),
        "dry_run": cfg.dry_run,
    }])
    summary.to_csv(SUMMARY_PATH, index=False)
    LOG.info("Wrote %s and %s | actual cost %.4f USD | parse_failures=%d",
             PER_QUERY_PATH.name, SUMMARY_PATH.name, meter.usd(),
             parse_failures)
    return df


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true",
                   help="Run the full benchmark (vs --dry-run for 3 queries).")
    p.add_argument("--dry-run", action="store_true",
                   help="Execute only the first 3 queries.")
    p.add_argument("--yes", action="store_true",
                   help="Confirm full paid run cost gate.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N queries.")
    p.add_argument("--eval-json", type=Path, default=CANONICAL_EVAL_JSON)
    p.add_argument("--gen-model", type=str,
                   default="claude-haiku-4-5-20251001")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--row-cap", type=int, default=40)
    p.add_argument("--sleep-s", type=float, default=0.05)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip if output CSV already exists.")
    a = p.parse_args()
    return Config(
        eval_json=a.eval_json, gen_model=a.gen_model, top_n=a.top_n,
        row_cap=a.row_cap, sleep_s=a.sleep_s, dry_run=a.dry_run,
        yes=a.yes, limit=a.limit, skip_existing=a.skip_existing,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = _parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
