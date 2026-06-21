"""
step_3_17_third_judge_eval.py — independent semantic evaluation of the
guardrail outputs (methodological separation for FW1).

Why this exists: in step_3_16 the Sonnet gate becomes a COMPONENT of the
system under test. Reusing the same model (and the same prompt family) as the
semantic evaluator would introduce a self-confirmation bias analogous to the
self-enhancement bias documented by Zheng et al. (judge != production model).
This step therefore evaluates the FINAL guardrail answers with a third model
that is neither the generator (Haiku) nor the runtime gate (Sonnet), using
the same binary rubric as the canonical step_3_9 judge for comparability,
and additionally emits a stratified manual spot-check sample.

Inputs:  results/step_3_16_guardrail_per_query.csv,
         data/benchmark ground truths (already inside the per-query CSV via
         join on the source artefact), ANTHROPIC_API_KEY in the environment
Outputs: results/step_3_17_third_judge_per_query.csv,
         results/step_3_17_third_judge_summary.csv,
         results/step_3_17_spot_check_sample.csv (human review, 15 rows)
Gate:    none (exploratory addendum); cost gate via --dry-run / --yes

Default third judge: claude-opus-4-8 (override with --judge-model or the
THIRD_JUDGE_MODEL env var). Keep the triple {generator, gate, evaluator}
pairwise distinct and DECLARE it in the manuscript addendum.

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from guardrail_common import (
    CANONICAL_EVAL_JSON, RESULTS, CostMeter, load_eval_records,
    make_anthropic_caller,
)

LOG = logging.getLogger(__name__)
DEFAULT_INPUT_STEM = "step_3_16_guardrail_per_query"


def _paths_for(guardrail_csv: Path) -> tuple[Path, Path, Path]:
    """Input-suffixed output paths so that judging different answer sets
    (guardrail, sonnet rerun, fable rerun) never overwrites prior verdicts."""
    stem = Path(guardrail_csv).stem
    if stem == DEFAULT_INPUT_STEM:
        suffix = ""
    else:
        suffix = "_" + stem.replace("step_3_15_llm_rerun_per_query", "rerun") \
                            .replace("step_3_18_", "")
    return (RESULTS / f"step_3_17_third_judge_per_query{suffix}.csv",
            RESULTS / f"step_3_17_third_judge_summary{suffix}.csv",
            RESULTS / f"step_3_17_spot_check_sample{suffix}.csv")


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    guardrail_csv: Path = RESULTS / "step_3_16_guardrail_per_query.csv"
    eval_json: Path = CANONICAL_EVAL_JSON
    judge_model: str = os.environ.get("THIRD_JUDGE_MODEL", "claude-opus-4-8")
    gen_model_name: str = "claude-haiku-4-5-20251001"
    gate_model_name: str = "claude-sonnet-4-6"
    limit: int | None = None
    dry_run: bool = True
    yes: bool = False
    seed: int = 42
    sleep_s: float = 0.3


def judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    """Binary semantic rubric, same shape as the canonical step_3_9 judge."""
    return (
        "You are an impartial judge for a knowledge-graph QA benchmark. "
        "Decide whether the MODEL ANSWER is substantively correct with "
        "respect to the REFERENCE. Judge meaning, not wording: numbers, "
        "thresholds, article ids and entity names must match the reference; "
        "extra correct context is fine; missing or contradicted core facts "
        "mean incorrect. An explicit abstention is incorrect.\n\n"
        f"Question: {question}\n"
        f"Reference: {ground_truth}\n"
        f"Model answer: {answer}\n\n"
        'Reply with a single JSON object, no prose: {"correct": true|false, '
        '"rationale": "<one sentence>"}'
    )


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the per-query judged dataframe."""
    per_query_path, summary_path, spot_check_path = _paths_for(cfg.guardrail_csv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not cfg.guardrail_csv.exists():
        LOG.error("Guardrail output not found: %s (run step_3_16 first)",
                  cfg.guardrail_csv)
        sys.exit(1)
    if cfg.judge_model in {cfg.gen_model_name, cfg.gate_model_name}:
        LOG.error("Third judge (%s) must differ from generator and gate.",
                  cfg.judge_model)
        sys.exit(1)

    guard = pd.read_csv(cfg.guardrail_csv)
    truth = {r.question_id: r for r in load_eval_records(cfg.eval_json)}
    guard["ground_truth"] = guard["question_id"].map(
        lambda q: truth[q].ground_truth if q in truth else "")
    guard["nl_question"] = guard["question_id"].map(
        lambda q: truth[q].nl_question if q in truth else "")

    rows_to_run = guard.head(3) if cfg.dry_run else guard.head(cfg.limit or len(guard))
    LOG.info("Indicative full-run cost with %s: depends on judge pricing; "
             "%d short calls (~700 in / ~80 out tokens each).",
             cfg.judge_model, len(guard))
    if cfg.dry_run:
        LOG.info("DRY RUN: judging only %d answers. Use --full --yes for all.",
                 len(rows_to_run))
    elif not cfg.yes:
        LOG.error("Refusing the full paid run without --yes (cost gate).")
        sys.exit(2)

    meter = CostMeter()
    family = "opus" if "opus" in cfg.judge_model else "sonnet"
    # temperature=None: claude-opus-4-8 deprecates the temperature parameter;
    # the judge runs at provider-default sampling (binary rubric, low variance).
    call_judge = make_anthropic_caller(cfg.judge_model, family, meter,
                                       temperature=None, max_tokens=160)
    out: list[dict] = []
    for i, row in enumerate(rows_to_run.itertuples(index=False), 1):
        import json as _json
        import re as _re
        verdict, rationale = None, ""
        try:
            text = call_judge(judge_prompt(row.nl_question, row.ground_truth,
                                           str(row.final_answer)))
            m = _re.search(r"\{.*\}", text, flags=_re.S)
            obj = _json.loads(m.group(0)) if m else {}
            verdict = bool(obj.get("correct", False))
            rationale = str(obj.get("rationale", ""))[:300]
        except Exception as exc:  # noqa: BLE001 — log and keep the run alive
            LOG.error("Judge failed on %s: %s", row.question_id, exc)
        out.append({
            "question_id": row.question_id,
            "status": row.status,
            "em_final": row.em_final,
            "third_judge_correct": verdict,
            "third_judge_rationale": rationale,
            "judge_model": cfg.judge_model,
        })
        LOG.info("%d/%d %s -> judge=%s", i, len(rows_to_run),
                 row.question_id, verdict)
        time.sleep(cfg.sleep_s)

    df = pd.DataFrame(out)
    df.to_csv(per_query_path, index=False)

    judged = df[df["third_judge_correct"].notna()]
    summary = pd.DataFrame([{
        "n_judged": len(judged),
        "em_semantic_third_judge": (round(judged["third_judge_correct"]
                                          .astype(float).mean(), 4)
                                    if len(judged) else None),
        "agreement_with_em_strict": (round((judged["third_judge_correct"]
                                            == judged["em_final"]).mean(), 4)
                                     if len(judged) else None),
        "judge_model": cfg.judge_model,
        "generator": cfg.gen_model_name,
        "runtime_gate": cfg.gate_model_name,
        "separation_ok": True,
        "tokens": dict(meter.tokens).__repr__(),
        "est_cost_usd": round(meter.usd(), 4),
        "dry_run": cfg.dry_run,
    }])
    summary.to_csv(summary_path, index=False)

    # Stratified manual spot-check sample: 5 agreements, 5 disagreements,
    # 5 abstentions (or as many as available), with an empty human column.
    merged = df.merge(guard[["question_id", "final_answer", "ground_truth",
                             "nl_question"]], on="question_id", how="left")
    agree = merged[merged["third_judge_correct"] == merged["em_final"]]
    disagree = merged[(merged["third_judge_correct"].notna())
                      & (merged["third_judge_correct"] != merged["em_final"])]
    abstain = merged[merged["status"] == "abstain"]
    sample = pd.concat([
        agree.sample(min(5, len(agree)), random_state=cfg.seed),
        disagree.sample(min(5, len(disagree)), random_state=cfg.seed),
        abstain.sample(min(5, len(abstain)), random_state=cfg.seed),
    ]).drop_duplicates("question_id")
    sample = sample.assign(human_verdict="", human_notes="")
    sample.to_csv(spot_check_path, index=False)
    LOG.info("Wrote %s (%d rows), %s and %s | actual cost %.4f USD",
             per_query_path.name, len(df), summary_path.name,
             spot_check_path.name, meter.usd())
    return df


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-model",
                   default=os.environ.get("THIRD_JUDGE_MODEL", "claude-opus-4-8"))
    p.add_argument("--guardrail-csv", type=Path,
                   default=RESULTS / "step_3_16_guardrail_per_query.csv")
    a = p.parse_args()
    return Config(guardrail_csv=a.guardrail_csv, judge_model=a.judge_model,
                  limit=a.limit, dry_run=not a.full, yes=a.yes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
