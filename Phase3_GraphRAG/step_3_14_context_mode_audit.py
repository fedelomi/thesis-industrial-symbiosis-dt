"""
step_3_14_context_mode_audit.py — classify every stored evaluation artefact by
answer-generation mode (context-only vs LLM-in-loop) and recompute EM strict.

Background: the canonical 100-query Graph-RAG artefact (EM strict 0.59) stores
answer == context on 100/100 queries, i.e. it was produced in the llm=None
context-only mode of step_3_4_evaluation.run_graph_rag(). This step makes the
mode of every artefact explicit and auditable, so that the manuscript can
state precisely which configuration each reported number measures.

Inputs:  data/evaluation_results_*.json (committed artefacts; no API, no Neo4j)
Outputs: results/step_3_14_context_mode_audit.csv (one row per artefact)
Gate:    em_recomputed must equal em_stored on every artefact (matcher parity)

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from guardrail_common import DATA, RESULTS, em_strict, load_eval_records

LOG = logging.getLogger(__name__)
OUT_PATH = RESULTS / "step_3_14_context_mode_audit.csv"


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    pattern: str = "evaluation_results_*.json"
    skip_existing: bool = False


def _classify(answer_eq_context_frac: float) -> str:
    if answer_eq_context_frac >= 0.99:
        return "context-only"
    if answer_eq_context_frac <= 0.10:
        return "llm-in-loop"
    return "mixed"


def run(cfg: Config) -> pd.DataFrame:
    """Top-level entry point. Returns the audit table."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.skip_existing and OUT_PATH.exists():
        LOG.info("Skipping (output exists): %s", OUT_PATH)
        return pd.read_csv(OUT_PATH)

    rows: list[dict] = []
    for path in sorted(DATA.glob(cfg.pattern)):
        records = load_eval_records(path)
        if not records:
            continue
        n = len(records)
        same = sum(
            1 for r in records
            if (r.stored_answer or "").strip() == (r.context or "").strip()
        )
        stored = [r.stored_exact_match for r in records
                  if r.stored_exact_match is not None]
        recomputed = [em_strict(r.stored_answer, r.ground_truth) for r in records]
        recomputed_valid = [x for x in recomputed if x is not None]
        parity = sum(
            1 for r, x in zip(records, recomputed)
            if r.stored_exact_match is not None and x == r.stored_exact_match
        )
        rows.append({
            "artefact": path.name,
            "n_queries": n,
            "answer_eq_context_frac": round(same / n, 4),
            "mode": _classify(same / n),
            "em_stored": round(sum(stored) / len(stored), 4) if stored else None,
            "em_recomputed": (round(sum(recomputed_valid) / len(recomputed_valid), 4)
                              if recomputed_valid else None),
            "matcher_parity_frac": (round(parity / len(stored), 4)
                                    if stored else None),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    LOG.info("Wrote %d rows to %s", len(df), OUT_PATH)

    gate_path = RESULTS / "gates" / "gate_step_3_14.csv"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    worst = df["matcher_parity_frac"].dropna().min() if len(df) else 0.0
    pd.DataFrame([{
        "metric": "em_matcher_parity_min",
        "value": worst,
        "threshold": 1.0,
        "status": "PASS" if worst >= 1.0 else "FAIL",
    }]).to_csv(gate_path, index=False)
    LOG.info("Gate em_matcher_parity_min = %s", worst)
    return df


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--pattern", default="evaluation_results_*.json")
    a = p.parse_args()
    return Config(pattern=a.pattern, skip_existing=a.skip_existing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(run(_parse_args()).to_string(index=False))
