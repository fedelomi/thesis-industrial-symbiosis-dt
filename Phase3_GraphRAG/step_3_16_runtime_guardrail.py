"""
step_3_16_runtime_guardrail.py — FW1 runtime guardrail, measured.

Pipeline per query (over the SAME stored retrieval contexts, no Neo4j):
    1. Haiku generates the answer (production model, step_3_4 prompt).
    2. A Sonnet-class verification gate checks the answer against the rows
       only (grounding + directness; the ground truth NEVER enters the
       runtime path).
    3. On FAIL: one retry with the gate's structured feedback.
    4. On second FAIL: abstain ("Not available in the knowledge graph."),
       flagged as abstention.

Reported metrics (selective-prediction style):
    - em_effective: EM strict with abstentions counted as incorrect
    - coverage: fraction of queries answered (not abstained)
    - em_on_answered: EM strict on the answered subset
    - retry_rate, abstain_rate, api calls and indicative cost

Methodological separation (see step_3_17): the runtime gate (Sonnet) is part
of the SYSTEM under test; it must not be reused as the evaluator. Final
answers are evaluated by the EM-strict matcher here and by an independent
third judge in step_3_17.

Inputs:  stored contexts from data/evaluation_results_*.json,
         ANTHROPIC_API_KEY in the environment (.env supported)
Outputs: results/step_3_16_guardrail_per_query.csv,
         results/step_3_16_guardrail_summary.csv
Gate:    none (exploratory addendum); cost gate via --dry-run / --yes

Cost:    100 queries x (1 Haiku + 1 Sonnet) + retry on a fraction of queries:
         indicatively 0.30-0.80 USD for the in-distribution benchmark.

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
    CANONICAL_EVAL_JSON, OOD_EVAL_JSON, RESULTS, CostMeter, EvalRecord,
    answer_generation_prompt, em_strict, extract_json_verdict,
    load_eval_records, make_anthropic_caller, retry_prompt, verifier_prompt,
)

LOG = logging.getLogger(__name__)
ABSTAIN_TEXT = "Not available in the knowledge graph."


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    eval_json: Path = CANONICAL_EVAL_JSON
    arm: str = "canonical"               # canonical | ood (label only)
    gen_model: str = "claude-haiku-4-5-20251001"
    gate_model: str = "claude-sonnet-4-6"
    limit: int | None = None
    dry_run: bool = True
    yes: bool = False
    skip_existing: bool = False
    sleep_s: float = 0.2


def _paths(cfg: Config) -> tuple[Path, Path]:
    suffix = "" if cfg.arm == "canonical" else f"_{cfg.arm}"
    return (RESULTS / f"step_3_16_guardrail_per_query{suffix}.csv",
            RESULTS / f"step_3_16_guardrail_summary{suffix}.csv")


def _guardrail_one(r: EvalRecord, call_gen, call_gate) -> dict:
    """Run generate -> verify -> retry -> abstain for one query."""
    answer_v1 = call_gen(answer_generation_prompt(r.context, r.nl_question))
    ok_v1, feedback = extract_json_verdict(
        call_gate(verifier_prompt(r.context, r.nl_question, answer_v1)))
    answer_v2, ok_v2 = None, None
    if not ok_v1:
        answer_v2 = call_gen(retry_prompt(r.context, r.nl_question, feedback))
        ok_v2, _ = extract_json_verdict(
            call_gate(verifier_prompt(r.context, r.nl_question, answer_v2)))
    if ok_v1:
        final, status = answer_v1, "pass"
    elif ok_v2:
        final, status = answer_v2, "retry_pass"
    else:
        final, status = ABSTAIN_TEXT, "abstain"
    return {
        "question_id": r.question_id,
        "category": r.category,
        "status": status,
        "final_answer": final,
        "answer_v1": answer_v1,
        "gate_feedback_v1": feedback,
        "answer_v2": answer_v2,
        "em_context_only": r.stored_exact_match,
        "em_final": em_strict(final, r.ground_truth),
        "abstained": status == "abstain",
    }


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the per-query dataframe."""
    per_query_path, summary_path = _paths(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and per_query_path.exists():
        LOG.info("Skipping (output exists): %s", per_query_path)
        return pd.read_csv(per_query_path)

    records = load_eval_records(cfg.eval_json)
    records_to_run = records[:3] if cfg.dry_run else records[: cfg.limit or None]

    n_full = len(records)
    LOG.info("Indicative full-run cost: %.2f-%.2f USD for %d queries "
             "(1 gen + 1 gate per query, plus retries).",
             0.003 * n_full, 0.008 * n_full, n_full)
    if cfg.dry_run:
        LOG.info("DRY RUN: executing only %d queries. Use --full --yes for "
                 "the complete benchmark.", len(records_to_run))
    elif not cfg.yes:
        LOG.error("Refusing the full paid run without --yes (cost gate).")
        sys.exit(2)

    meter = CostMeter()
    call_gen = make_anthropic_caller(cfg.gen_model, "haiku", meter)
    call_gate = make_anthropic_caller(cfg.gate_model, "sonnet", meter,
                                      max_tokens=200)
    rows: list[dict] = []
    for i, r in enumerate(records_to_run, 1):
        try:
            row = _guardrail_one(r, call_gen, call_gate)
        except Exception as exc:  # noqa: BLE001 — log and keep the run alive
            LOG.error("Query %s failed: %s", r.question_id, exc)
            row = {"question_id": r.question_id, "category": r.category,
                   "status": "error", "final_answer": "",
                   "em_context_only": r.stored_exact_match,
                   "em_final": None, "abstained": False}
        row["arm"] = cfg.arm
        rows.append(row)
        LOG.info("%d/%d %s -> %s (em=%s)", i, len(records_to_run),
                 r.question_id, row["status"], row["em_final"])
        time.sleep(cfg.sleep_s)

    df = pd.DataFrame(rows)
    df.to_csv(per_query_path, index=False)

    answered = df[~df["abstained"] & df["em_final"].notna()]
    effective = df[df["em_final"].notna() | df["abstained"]]
    em_effective = (
        sum(1 for _, x in effective.iterrows()
            if (x["em_final"] is True or x["em_final"] == 1.0) and not x["abstained"])
        / len(effective)) if len(effective) else None
    summary = pd.DataFrame([{
        "arm": cfg.arm,
        "n_queries": len(df),
        "em_context_only": round(df["em_context_only"].astype(float).mean(), 4),
        "em_effective": round(em_effective, 4) if em_effective is not None else None,
        "coverage": round(1 - df["abstained"].mean(), 4),
        "em_on_answered": (round(answered["em_final"].astype(float).mean(), 4)
                           if len(answered) else None),
        "retry_rate": round((df["status"] == "retry_pass").mean()
                            + (df["status"] == "abstain").mean(), 4),
        "abstain_rate": round(df["abstained"].mean(), 4),
        "gen_model": cfg.gen_model,
        "gate_model": cfg.gate_model,
        "source_artefact": Path(cfg.eval_json).name,
        "tokens": dict(meter.tokens).__repr__(),
        "est_cost_usd": round(meter.usd(), 4),
        "dry_run": cfg.dry_run,
    }])
    summary.to_csv(summary_path, index=False)
    LOG.info("Wrote %s and %s | actual cost %.4f USD",
             per_query_path.name, summary_path.name, meter.usd())
    return df


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--arm", choices=["canonical", "ood"], default="canonical")
    p.add_argument("--gen-model", default="claude-haiku-4-5-20251001")
    p.add_argument("--gate-model", default="claude-sonnet-4-6")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()
    eval_json = CANONICAL_EVAL_JSON if a.arm == "canonical" else OOD_EVAL_JSON
    return Config(eval_json=eval_json, arm=a.arm, gen_model=a.gen_model,
                  gate_model=a.gate_model, limit=a.limit, dry_run=not a.full,
                  yes=a.yes, skip_existing=a.skip_existing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
