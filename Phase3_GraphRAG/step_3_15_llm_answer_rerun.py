"""
step_3_15_llm_answer_rerun.py — re-run the canonical 100-query benchmark with
the Haiku answer-generation step ON TOP of the stored retrieval contexts.

Purpose: step_3_14 documents that the canonical EM strict 0.59 was measured in
context-only mode (llm=None). This step isolates the marginal contribution of
the LLM rendering layer by generating natural-language answers from the SAME
stored contexts (no Neo4j needed) and re-scoring EM strict with the identical
matcher. Canonical numbers are not modified; the output is an addendum
artefact in the spirit of Section 6.2.2-bis.

Inputs:  data/evaluation_results_graph-rag_20260517_155731.json (stored
         contexts), ANTHROPIC_API_KEY in the environment (.env supported)
Outputs: results/step_3_15_llm_rerun_per_query.csv,
         results/step_3_15_llm_rerun_summary.csv
Gate:    none (exploratory addendum); cost gate via --dry-run / --yes

Cost:    ~100 Haiku calls, input ~600 tok / output ~80 tok each:
         well under 0.10 USD at indicative prices. Printed before running.

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
    CANONICAL_EVAL_JSON, RESULTS, CostMeter, EvalRecord,
    answer_generation_prompt, em_strict, load_eval_records,
    make_anthropic_caller,
)

LOG = logging.getLogger(__name__)
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _paths_for(model: str) -> tuple[Path, Path]:
    """Model-suffixed output paths so that re-runs with a different generator
    never overwrite the canonical Haiku artefacts."""
    if model == DEFAULT_MODEL:
        suffix = ""
    else:
        tag = model.replace("claude-", "").replace(".", "").replace("/", "-")
        suffix = f"_{tag}"
    return (RESULTS / f"step_3_15_llm_rerun_per_query{suffix}.csv",
            RESULTS / f"step_3_15_llm_rerun_summary{suffix}.csv")


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    eval_json: Path = CANONICAL_EVAL_JSON
    model: str = "claude-haiku-4-5-20251001"
    limit: int | None = None
    dry_run: bool = True          # safe default: 3 queries + cost estimate
    yes: bool = False             # explicit consent for the full paid run
    skip_existing: bool = False
    sleep_s: float = 0.2


def _estimate_cost(records: list[EvalRecord], meter: CostMeter) -> float:
    """Rough pre-run estimate from prompt sizes (4 chars per token)."""
    tok_in = sum(len(answer_generation_prompt(r.context, r.nl_question)) // 4
                 for r in records)
    tok_out = 80 * len(records)
    return (tok_in / 1e6) * meter.prices["haiku_in"] + \
           (tok_out / 1e6) * meter.prices["haiku_out"]


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the per-query dataframe."""
    per_query_path, summary_path = _paths_for(cfg.model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and per_query_path.exists():
        LOG.info("Skipping (output exists): %s", per_query_path)
        return pd.read_csv(per_query_path)

    records = load_eval_records(cfg.eval_json)
    if cfg.dry_run:
        records_to_run = records[:3]
    elif cfg.limit:
        records_to_run = records[: cfg.limit]
    else:
        records_to_run = records

    meter = CostMeter()
    estimate = _estimate_cost(records, meter)
    LOG.info("Full-run cost estimate (indicative): %.3f USD for %d queries",
             estimate, len(records))
    if cfg.dry_run:
        LOG.info("DRY RUN: executing only %d queries. Re-run with --yes for "
                 "the full benchmark.", len(records_to_run))
    elif not cfg.yes:
        LOG.error("Refusing the full paid run without --yes (cost gate).")
        sys.exit(2)

    call_haiku = make_anthropic_caller(cfg.model, "haiku", meter)
    rows: list[dict] = []
    for i, r in enumerate(records_to_run, 1):
        prompt = answer_generation_prompt(r.context, r.nl_question)
        try:
            answer = call_haiku(prompt)
        except Exception as exc:  # noqa: BLE001 — log and keep the run alive
            LOG.error("Query %s failed: %s", r.question_id, exc)
            answer = ""
        em_new = em_strict(answer, r.ground_truth)
        is_abstain = answer.strip().lower().startswith("not available")
        rows.append({
            "question_id": r.question_id,
            "category": r.category,
            "em_context_only": r.stored_exact_match,
            "em_llm_rerun": em_new,
            "answer_llm": answer,
            # Aliases so the third judge (step_3_17 --guardrail-csv) can
            # consume this artefact directly:
            "final_answer": answer,
            "em_final": em_new,
            "status": "abstain" if is_abstain else "pass",
            "abstained": is_abstain,
            "model": cfg.model,
        })
        LOG.info("%d/%d %s em_ctx=%s em_llm=%s",
                 i, len(records_to_run), r.question_id,
                 r.stored_exact_match, em_new)
        time.sleep(cfg.sleep_s)

    df = pd.DataFrame(rows)
    df.to_csv(per_query_path, index=False)

    scored = df[df["em_llm_rerun"].notna()]
    summary = pd.DataFrame([{
        "n_queries": len(df),
        "em_context_only": round(
            scored["em_context_only"].astype(float).mean(), 4) if len(scored) else None,
        "em_llm_rerun": round(
            scored["em_llm_rerun"].astype(float).mean(), 4) if len(scored) else None,
        "delta_pp": round(100 * (scored["em_llm_rerun"].astype(float).mean()
                                 - scored["em_context_only"].astype(float).mean()), 1)
        if len(scored) else None,
        "model": cfg.model,
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
    p.add_argument("--full", action="store_true",
                   help="disable dry-run (requires --yes)")
    p.add_argument("--yes", action="store_true",
                   help="consent to the paid full run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--eval-json", type=Path, default=CANONICAL_EVAL_JSON)
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()
    return Config(eval_json=a.eval_json, model=a.model, limit=a.limit,
                  dry_run=not a.full, yes=a.yes, skip_existing=a.skip_existing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
