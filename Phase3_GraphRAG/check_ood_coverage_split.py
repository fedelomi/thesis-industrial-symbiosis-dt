"""
check_ood_coverage_split.py
==========================
Phase 3 - OOD coverage-gap split gate (ZERO API, deterministic).

Reads the manual review backing (benchmark_ood_v1_review.jsonl) and, once the
researcher has filled the `decision` field on each gap, produces:

  1. the final split count {kg_missing, retrieval_miss, ambiguous};
  2. a revised step_3_11 error classification, where coverage-gap queries are
     re-labelled by their decision (retrieval_miss -> routing_fail, since the
     fact was actually present; kg_missing -> the true coverage gap; ambiguous
     -> needs_composition);
  3. the canonical post-curation number, e.g.
     "true KG coverage gap = 4/38 OOD queries = 10.5% (lower bound)".

Until decisions are filled it runs in PROVISIONAL mode on the gap_type_hint, so
the numbers can be previewed; the banner makes clear these are hints, not
researcher decisions. It never re-runs step_3_11 and never edits the source
benchmark (stays UNCURATED).

Inputs:
    data/benchmarks/benchmark_ood_v1_review.jsonl   (decision field)
    results/step_3_11_ood_eval_per_query.csv        (original eval, optional)
Outputs:
    results/step_3_11_ood_eval_revised_summary.csv

Usage:
    python check_ood_coverage_split.py
    python check_ood_coverage_split.py --use-hints   # force provisional

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BENCH_DIR = BASE_DIR / "data" / "benchmarks"
REVIEW = BENCH_DIR / "benchmark_ood_v1_review.jsonl"
EVAL_PER_QUERY = BASE_DIR / "results" / "step_3_11_ood_eval_per_query.csv"
OUT_SUMMARY = BASE_DIR / "results" / "step_3_11_ood_eval_revised_summary.csv"

N_OOD_TOTAL = 38
VALID_DECISIONS = {"kg_missing", "retrieval_miss", "ambiguous"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD coverage-gap split gate")
    parser.add_argument("--use-hints", action="store_true",
                        help="ignore decisions and report the provisional hint split")
    args = parser.parse_args()

    if not REVIEW.exists():
        logger.error("Review file not found: %s (run build_ood_coverage_review.py)", REVIEW)
        return
    rows = [json.loads(l) for l in REVIEW.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)

    filled = [r for r in rows if str(r.get("decision", "")).strip()]
    provisional = args.use_hints or not filled
    source = "gap_type_hint (PROVISIONAL)" if provisional else "decision (researcher)"

    invalid = []
    split: dict[str, int] = {k: 0 for k in VALID_DECISIONS}
    resolved: dict[str, str] = {}
    for r in rows:
        if provisional:
            label = r.get("gap_type_hint", "ambiguous")
        else:
            label = str(r.get("decision", "")).strip() or r.get("gap_type_hint", "ambiguous")
        if label not in VALID_DECISIONS:
            invalid.append((r["query_id"], label))
            label = "ambiguous"
        split[label] += 1
        resolved[r["query_id"]] = label

    # Revised classification merged with the original step_3_11 eval.
    revised_counts = {}
    if EVAL_PER_QUERY.exists():
        df = pd.read_csv(EVAL_PER_QUERY)
        relabel = {"kg_missing": "kg_coverage_gap_true",
                   "retrieval_miss": "routing_fail",
                   "ambiguous": "needs_composition"}
        def _new_class(row):
            if row["error_class"] == "coverage_gap":
                return relabel.get(resolved.get(row["id"], "ambiguous"), "needs_composition")
            return row["error_class"]
        df["revised_class"] = df.apply(_new_class, axis=1)
        revised_counts = df["revised_class"].value_counts().to_dict()

    true_gap = split["kg_missing"]
    true_gap_pct = round(100.0 * true_gap / N_OOD_TOTAL, 1)
    orig_gap_pct = round(100.0 * n / N_OOD_TOTAL, 1)

    summary = [
        {"metric": "decisions_filled", "value": f"{len(filled)}/{n}"},
        {"metric": "source", "value": source},
        {"metric": "kg_missing", "value": split["kg_missing"]},
        {"metric": "retrieval_miss", "value": split["retrieval_miss"]},
        {"metric": "ambiguous", "value": split["ambiguous"]},
        {"metric": "orig_coverage_gap_pct", "value": orig_gap_pct},
        {"metric": "true_kg_coverage_gap_pct_lower_bound", "value": true_gap_pct},
    ]
    for k, v in revised_counts.items():
        summary.append({"metric": f"revised_class.{k}", "value": int(v)})
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 62)
    print("  OOD COVERAGE-GAP SPLIT")
    print("=" * 62)
    print(f"  source           : {source}")
    print(f"  decisions filled : {len(filled)}/{n}")
    if invalid:
        print(f"  INVALID decisions (treated as ambiguous): {invalid}")
    print(f"  split            : kg_missing={split['kg_missing']}  "
          f"retrieval_miss={split['retrieval_miss']}  ambiguous={split['ambiguous']}")
    print(f"\n  original coverage-gap headline : {n}/{N_OOD_TOTAL} = {orig_gap_pct}% (upper bound)")
    print(f"  true KG coverage gap           : {true_gap}/{N_OOD_TOTAL} = {true_gap_pct}% (lower bound)")
    if revised_counts:
        print(f"\n  revised step_3_11 error classes: {revised_counts}")
        print("  (retrieval_miss gaps reclassified as routing_fail: the fact was "
              "present but unrouted)")
    if provisional:
        print("\n  NOTE: PROVISIONAL (gap_type_hint). Fill 'decision' in "
              f"{REVIEW.name} then re-run for the researcher-confirmed split.")
    else:
        ready = len(filled) == n
        print(f"\n  curation status: {'COMPLETE - ready to mark CURATED' if ready else 'PARTIAL - ' + str(n - len(filled)) + ' decisions still empty'}")
    print(f"\n  wrote {OUT_SUMMARY.name}")


if __name__ == "__main__":
    main()
