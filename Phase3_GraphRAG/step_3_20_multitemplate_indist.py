"""
step_3_20_multitemplate_indist.py — v7 multi-template retrieval path measured
on the CANONICAL 100-query in-distribution benchmark (never done before:
the v7 path was evaluated only on the OOD arm).

Motivation: the step_3_19 coverage audit classifies 29 of the 57 structural
semantic failures (failed by Haiku, Sonnet and Fable alike) as routing /
under-retrieval: the fact is IN the graph but the canonical single-template
top-1 routing never reaches it. The v7 multi-template union (route top-5,
union + prune + densify) is the already-built fix (+55 pp on the OOD
synthesis_fail set). This step measures it in-distribution at constant
generator (Haiku), closing the chain: graph healthy -> routing was the
bottleneck -> fix already built.

Ex-ante predictions (formulated before running the experiment; documented in
the experiment design notes on 2026-06-13; committed alongside outcomes in the
audit trail -- git commit timestamps reflect the documentation event, not the
moment the predictions were formed):
  P1. On the 29 routing-classified queries, at least 40% flip to a context
      that contains the ground-truth facts (em of context vs GT).
  P2. Overall EM strict (NL answers) rises above the single-template NL
      0.45 of step_3_15, toward 0.55-0.70.
  P3. The 19 synthesis-classified and 8 derived-classified queries move
      little (the lever for those is FW9-ter, not retrieval).

Pipeline per query: route_question_multi(n=5) -> union_template_rows ->
prune_rows -> densify_context_v2 -> Haiku answer (exact step_3_4 prompt) ->
EM strict. Canonical files untouched; prompt5_retrieval is the isolated v7
module by design.

Inputs:  Neo4j live (config NEO4J_*), ANTHROPIC_API_KEY, canonical benchmark
         artefact (questions + ground truths + canonical contexts),
         results/step_3_19_kg_coverage_audit.csv (routing flags)
Outputs: results/step_3_20_multitemplate_indist_per_query.csv (judge-ready
         schema for step_3_17 --guardrail-csv),
         results/step_3_20_multitemplate_indist_summary.csv
Gate:    none (exploratory addendum); cost gate via --dry-run / --yes

Cost:    ~100 Haiku calls (~0.07-0.10 USD). Requires the local Neo4j DBMS
         to be running. Optional judging: step_3_17 (+~0.94 USD).

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from guardrail_common import (
    CANONICAL_EVAL_JSON, RESULTS, CostMeter, answer_generation_prompt,
    em_strict, load_eval_records, make_anthropic_caller,
)

LOG = logging.getLogger(__name__)
PER_QUERY_PATH = RESULTS / "step_3_20_multitemplate_indist_per_query.csv"
SUMMARY_PATH = RESULTS / "step_3_20_multitemplate_indist_summary.csv"
AUDIT_PATH = RESULTS / "step_3_19_kg_coverage_audit.csv"


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    eval_json: Path = CANONICAL_EVAL_JSON
    gen_model: str = "claude-haiku-4-5-20251001"
    top_n: int = 5
    row_cap: int = 40
    limit: int | None = None
    dry_run: bool = True
    yes: bool = False
    skip_existing: bool = False
    sleep_s: float = 0.2


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the per-query dataframe."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and PER_QUERY_PATH.exists():
        LOG.info("Skipping (output exists): %s", PER_QUERY_PATH)
        return pd.read_csv(PER_QUERY_PATH)

    # Lazy heavy imports so --help and offline tests never need Neo4j.
    from neo4j import GraphDatabase
    import config
    from templates import CYPHER_TEMPLATES
    from prompt5_retrieval import (
        densify_context_v2, prune_rows, route_question_multi,
        union_template_rows,
    )

    records = load_eval_records(cfg.eval_json)
    records_to_run = records[:3] if cfg.dry_run else records[: cfg.limit or None]
    LOG.info("Indicative full-run cost: ~0.07-0.10 USD (%d Haiku calls). "
             "Neo4j must be running at %s.", len(records), config.NEO4J_URI)
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
    with driver.session(database=config.NEO4J_DATABASE) as session:

        def run_template(tid: str) -> list[dict]:
            cypher = CYPHER_TEMPLATES.get(tid)
            if not cypher:
                return []
            try:
                return session.execute_read(lambda tx: tx.run(cypher).data())
            except Exception:  # noqa: BLE001 — template errors yield no rows
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
                answer = call_gen(answer_generation_prompt(context,
                                                           r.nl_question))
            except Exception as exc:  # noqa: BLE001 — keep the run alive
                LOG.error("Query %s failed: %s", r.question_id, exc)
                pool, union, context, answer = [], [], "", ""
            is_abstain = answer.strip().lower().startswith("not available")
            em_new = em_strict(answer, r.ground_truth)
            ctx_has_gt = em_strict(context, r.ground_truth)
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
            })
            LOG.info("%d/%d %s pool=%d rows=%d em=%s ctx_gt=%s [%s]",
                     i, len(records_to_run), r.question_id, len(pool),
                     len(union), em_new, ctx_has_gt,
                     routing_flags.get(r.question_id, "-"))
            time.sleep(cfg.sleep_s)
    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(PER_QUERY_PATH, index=False)

    routing = df[df["step_3_19_bucket"] == "routing_under_retrieval"]
    summary = pd.DataFrame([{
        "n_queries": len(df),
        "em_strict_multitemplate": round(df["em_final"].astype(float).mean(), 4),
        "em_strict_singletemplate_nl_ref": 0.45,
        "ctx_contains_gt_overall": round(
            df["ctx_contains_gt_keywords"].astype(float).mean(), 4),
        "routing29_n": len(routing),
        "routing29_ctx_contains_gt": (round(routing["ctx_contains_gt_keywords"]
                                            .astype(float).mean(), 4)
                                      if len(routing) else None),
        "routing29_em_strict": (round(routing["em_final"].astype(float).mean(), 4)
                                if len(routing) else None),
        "abstain_rate": round(df["abstained"].mean(), 4),
        "top_n": cfg.top_n,
        "gen_model": cfg.gen_model,
        "tokens": dict(meter.tokens).__repr__(),
        "est_cost_usd": round(meter.usd(), 4),
        "dry_run": cfg.dry_run,
    }])
    summary.to_csv(SUMMARY_PATH, index=False)
    LOG.info("Wrote %s and %s | actual cost %.4f USD",
             PER_QUERY_PATH.name, SUMMARY_PATH.name, meter.usd())
    return df


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--gen-model", default="claude-haiku-4-5-20251001")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()
    return Config(gen_model=a.gen_model, top_n=a.top_n, limit=a.limit,
                  dry_run=not a.full, yes=a.yes, skip_existing=a.skip_existing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
