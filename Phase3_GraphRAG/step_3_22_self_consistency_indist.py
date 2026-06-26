"""
step_3_22_self_consistency_indist.py - v7 multi-template retrieval + structured
chain-of-thought + self-consistency sampling (Lever 5 of the FW11 roadmap of
Section 6.3.2). Measured on the CANONICAL 100-query in-distribution benchmark
at constant Haiku generator and constant Opus third-judge for direct
comparability with step_3_20 (flat) and step_3_21 (CoT single-pass).

Motivation: step_3_21 measured the structured-CoT lever at EM semantic 0.73
(+8 pp vs flat 0.65). The residual ceiling on comparative queries
(Cat C, EM semantic 0.48) and the JSON parse failure rate of 0.41 both
suggest that single-sample synthesis is still the bottleneck: when the LLM
falls into a wrong reasoning chain it has no way out. Self-consistency
sampling (Wang et al. 2023) addresses this by drawing N independent samples
at temperature > 0 and taking the majority vote on the final_answer field,
trading API cost for variance reduction.

Ex-ante predictions (formulated before running the experiment; documented in
the experiment design notes on 2026-06-23; committed alongside outcomes in the
audit trail -- git commit timestamps reflect the documentation event, not the
moment the predictions were formed):
  P1. Overall EM semantic rises from 0.73 (step_3_21) toward 0.75-0.80.
  P2. The lift is concentrated on Cat C (comparative): 0.48 -> 0.55-0.65.
  P3. The JSON parse failure rate of step_3_21 (0.41) drops to <0.20
      because at least one of the N samples gets the format right.
  P4. Cat A (factual) stays near 0.97 (saturated).

Pipeline per query: route_question_multi(n=5) -> union_template_rows ->
prune_rows -> densify_context_v2 -> Haiku structured-CoT answer with
temperature=0.5, N_SAMPLES=3 independent calls -> majority vote on
final_answer (string normalized lowercase strip) -> EM strict.

Inputs:  Neo4j live, ANTHROPIC_API_KEY, canonical benchmark, audit CSV.
Outputs: results/step_3_22_self_consistency_indist_per_query.csv,
         results/step_3_22_self_consistency_indist_summary.csv

Cost:    ~3 x step_3_21 ~ 0.40-0.60 USD per Haiku at temp 0.5 with longer
         output. Optional third-judge eval (+~1.12 USD with Opus).

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from guardrail_common import (
    CANONICAL_EVAL_JSON, RESULTS, CostMeter,
    em_strict, load_eval_records, make_anthropic_caller,
)

LOG = logging.getLogger(__name__)
PER_QUERY_PATH = RESULTS / "step_3_22_self_consistency_indist_per_query.csv"
SUMMARY_PATH = RESULTS / "step_3_22_self_consistency_indist_summary.csv"
AUDIT_PATH = RESULTS / "step_3_19_kg_coverage_audit.csv"

N_SAMPLES = 3
SAMPLING_TEMPERATURE = 0.5


def cot_prompt(context: str, question: str) -> str:
    """Mirror of step_3_21.cot_prompt."""
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
    """Mirror of step_3_21.parse_cot_answer."""
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


def _normalize_for_vote(answer: str) -> str:
    """Soft normalization for majority vote: lowercase, strip, collapse
    whitespace, remove trailing punctuation that does not change semantics."""
    s = answer.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:")
    return s


def majority_vote(answers: list[str]) -> tuple[str, int]:
    """Return the majority answer and the number of votes it received.
    Ties are broken by first-seen order (stable).
    """
    if not answers:
        return "", 0
    normalized = [_normalize_for_vote(a) for a in answers]
    counts = Counter(normalized)
    top_norm, top_count = counts.most_common(1)[0]
    # Return the first original answer whose normalization matches.
    for original in answers:
        if _normalize_for_vote(original) == top_norm:
            return original, top_count
    return answers[0], 1


@dataclass
class Config:
    eval_json: Path
    gen_model: str = "claude-haiku-4-5-20251001"
    top_n: int = 5
    row_cap: int = 40
    n_samples: int = N_SAMPLES
    temperature: float = SAMPLING_TEMPERATURE
    sleep_s: float = 0.05
    dry_run: bool = False
    yes: bool = False
    limit: int | None = None
    skip_existing: bool = False


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and PER_QUERY_PATH.exists():
        LOG.info("Skipping (output exists): %s", PER_QUERY_PATH)
        return pd.read_csv(PER_QUERY_PATH)

    from neo4j import GraphDatabase
    import config as cfg_mod
    from templates import CYPHER_TEMPLATES
    from prompt5_retrieval import (
        densify_context_v2, prune_rows, route_question_multi,
        union_template_rows,
    )
    try:
        from anthropic import Anthropic
    except ImportError:
        LOG.error("anthropic SDK missing. pip install anthropic")
        sys.exit(1)
    client = Anthropic(api_key=cfg_mod.ANTHROPIC_API_KEY)

    records = load_eval_records(cfg.eval_json)
    records_to_run = records[:3] if cfg.dry_run else records[: cfg.limit or None]
    LOG.info("Indicative full-run cost: ~0.40-0.60 USD (%d queries x %d "
             "Haiku samples each at temp %s). Neo4j must be running at %s.",
             len(records_to_run), cfg.n_samples, cfg.temperature,
             cfg_mod.NEO4J_URI)
    if cfg.dry_run:
        LOG.info("DRY RUN: only %d queries.", len(records_to_run))
    elif not cfg.yes:
        LOG.error("Refusing the full paid run without --yes (cost gate).")
        sys.exit(2)

    routing_flags: dict[str, str] = {}
    if AUDIT_PATH.exists():
        audit = pd.read_csv(AUDIT_PATH)
        routing_flags = dict(zip(audit["question_id"], audit["class"]))
    else:
        LOG.warning("step_3_19 audit CSV not found.")

    meter = CostMeter()
    driver = GraphDatabase.driver(
        cfg_mod.NEO4J_URI, auth=(cfg_mod.NEO4J_USER, cfg_mod.NEO4J_PASSWORD))

    rows: list[dict] = []
    parse_failures = 0
    unanimous_count = 0
    with driver.session(database=cfg_mod.NEO4J_DATABASE) as session:

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
                prompt = cot_prompt(context, r.nl_question)
                # N independent samples at temperature > 0
                sample_answers: list[str] = []
                sample_parsed: list[bool] = []
                for _ in range(cfg.n_samples):
                    resp = client.messages.create(
                        model=cfg.gen_model,
                        max_tokens=1024,
                        temperature=cfg.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    # Update cost meter
                    if resp.usage:
                        meter.tokens["haiku_in"] = (
                            meter.tokens.get("haiku_in", 0)
                            + resp.usage.input_tokens)
                        meter.tokens["haiku_out"] = (
                            meter.tokens.get("haiku_out", 0)
                            + resp.usage.output_tokens)
                    raw = resp.content[0].text.strip()
                    ans, payload = parse_cot_answer(raw)
                    sample_answers.append(ans)
                    sample_parsed.append(bool(payload))
                final_answer, votes = majority_vote(sample_answers)
                if votes == cfg.n_samples:
                    unanimous_count += 1
                if not any(sample_parsed):
                    parse_failures += 1
            except Exception as exc:  # noqa: BLE001
                LOG.error("Query %s failed: %s", r.question_id, exc)
                pool, union, context = [], [], ""
                sample_answers = []
                sample_parsed = []
                final_answer, votes = "", 0
            is_abstain = final_answer.strip().lower().startswith("not available")
            em_new = em_strict(final_answer, r.ground_truth)
            ctx_has_gt = em_strict(context, r.ground_truth)
            parse_ok_n = sum(1 for p in sample_parsed if p)
            rows.append({
                "question_id": r.question_id,
                "category": r.category,
                "status": "abstain" if is_abstain else "pass",
                "final_answer": final_answer,
                "em_final": em_new,
                "abstained": is_abstain,
                "em_context_only_canonical": r.stored_exact_match,
                "ctx_contains_gt_keywords": ctx_has_gt,
                "pool": "|".join(pool),
                "union_rows": len(union),
                "step_3_19_bucket": routing_flags.get(r.question_id, ""),
                "gen_model": cfg.gen_model,
                "n_samples": cfg.n_samples,
                "vote_count": votes,
                "unanimous": votes == cfg.n_samples,
                "parse_ok_n": parse_ok_n,
            })
            LOG.info("%d/%d %s pool=%d rows=%d em=%s ctx_gt=%s vote=%d/%d "
                     "parse=%d/%d [%s]",
                     i, len(records_to_run), r.question_id, len(pool),
                     len(union), em_new, ctx_has_gt, votes, cfg.n_samples,
                     parse_ok_n, cfg.n_samples,
                     routing_flags.get(r.question_id, "-"))
            time.sleep(cfg.sleep_s)
    driver.close()

    df = pd.DataFrame(rows)
    df.to_csv(PER_QUERY_PATH, index=False)

    routing = df[df["step_3_19_bucket"] == "routing_under_retrieval"]
    summary = pd.DataFrame([{
        "n_queries": len(df),
        "n_samples": cfg.n_samples,
        "temperature": cfg.temperature,
        "em_strict_self_consistency": round(df["em_final"].astype(float).mean(), 4),
        "em_strict_multitemplate_cot_ref": 0.72,
        "em_strict_multitemplate_flat_ref": 0.69,
        "em_strict_singletemplate_nl_ref": 0.45,
        "ctx_contains_gt_overall": round(
            df["ctx_contains_gt_keywords"].astype(float).mean(), 4),
        "routing29_n": len(routing),
        "routing29_em_strict": (round(routing["em_final"].astype(float).mean(), 4)
                                if len(routing) else None),
        "abstain_rate": round(df["abstained"].mean(), 4),
        "unanimous_rate": round(df["unanimous"].mean(), 4),
        "avg_parse_ok_per_query": round(
            df["parse_ok_n"].astype(float).mean() / cfg.n_samples, 4),
        "all_failed_parse_n": parse_failures,
        "top_n": cfg.top_n,
        "gen_model": cfg.gen_model,
        "tokens": dict(meter.tokens).__repr__(),
        "est_cost_usd": round(meter.usd(), 4),
        "dry_run": cfg.dry_run,
    }])
    summary.to_csv(SUMMARY_PATH, index=False)
    LOG.info("Wrote %s and %s | cost %.4f USD | unanimous=%d/%d | "
             "all_failed_parse=%d",
             PER_QUERY_PATH.name, SUMMARY_PATH.name, meter.usd(),
             unanimous_count, len(df), parse_failures)
    return df


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-json", type=Path, default=CANONICAL_EVAL_JSON)
    p.add_argument("--gen-model", type=str,
                   default="claude-haiku-4-5-20251001")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--row-cap", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=N_SAMPLES,
                   help="Number of self-consistency samples per query "
                        "(default 3).")
    p.add_argument("--temperature", type=float, default=SAMPLING_TEMPERATURE)
    p.add_argument("--sleep-s", type=float, default=0.05)
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()
    return Config(
        eval_json=a.eval_json, gen_model=a.gen_model, top_n=a.top_n,
        row_cap=a.row_cap, n_samples=a.n_samples, temperature=a.temperature,
        sleep_s=a.sleep_s, dry_run=a.dry_run, yes=a.yes, limit=a.limit,
        skip_existing=a.skip_existing,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = _parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
