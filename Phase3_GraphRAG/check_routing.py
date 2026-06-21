"""check_routing.py - zero-API pre-flight for the graph-rag routing.

Validates, WITHOUT touching Neo4j or the LLM (no API cost), that:
  1. every benchmark question routes to a template that EXISTS in CYPHER_TEMPLATES
     (a missing template would crash step_3_4 mid-run and waste API calls);
  2. the questions targeted by the 2026-06 fixes now route where intended.

Run this for free before the paid benchmark:
    python check_routing.py

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from step_3_4_evaluation import route_question
from templates import CYPHER_TEMPLATES

HERE = Path(__file__).resolve().parent
DATASET = HERE / "data" / "benchmark_qa_dataset.json"

# Questions targeted by the routing / template fixes, and where they should land.
EXPECTED = {
    "A14": "TEMPERATURE_BAND_DEF",
    "A19": "GENERIC_DC",
    "A24": "GENERIC_ACTOR",
    "A30": "DK_NECP_TARGET",
    "A31": "GENERIC_ACTOR",
    "A32": "GENERIC_COUNTRY",
    "B02": "P2_thermal_compatibility_all",
    "B07": "P2_thermal_compatibility_all",
    "B19": "P2_thermal_compatibility_all",
    "B28": "SCENARIO_COUNT_BY_SECTOR",
    "B32": "GENERIC_COUNTRY",
    "B33": "GENERIC_COUNTRY",
    "C03": "GENERIC_DC",
    "C10": "GENERIC_COUNTRY",
    "C13": "IT_DK_SUPPORT_COMPARE",
    "C16": "IT_DK_SUPPORT_COMPARE",
    "C17": "GENERIC_COUNTRY",
    "C31": "IT_DK_SUPPORT_COMPARE",
    # 2026-06-02 regression recoveries (were correct, mis-routed by the 06-01 edits)
    "B17": "P3_regulatory_screening_dk",
    "C06": "P2_thermal_compatibility_all",
    "C23": "P2_thermal_compatibility_all",
}


def main() -> None:
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    missing: list[tuple[str, str]] = []
    dist: Counter = Counter()
    hits: dict[str, str] = {}

    for q in questions:
        tid = route_question(q["nl_question"])
        dist[tid] += 1
        if tid not in CYPHER_TEMPLATES:
            missing.append((q["id"], tid))
        if q["id"] in EXPECTED:
            hits[q["id"]] = tid

    print("=== missing templates (these WOULD crash the paid run) ===")
    print("  none" if not missing else "\n".join(f"  {qid} -> {tid}" for qid, tid in missing))

    print("\n=== target questions: routed -> expected ===")
    ok = 0
    for qid, want in sorted(EXPECTED.items()):
        got = hits.get(qid, "<not in dataset>")
        if got == want:
            ok += 1
            print(f"  [OK ] {qid}: {got}")
        else:
            print(f"  [MISS] {qid}: {got}   (want {want})")
    print(f"\n  {ok}/{len(EXPECTED)} target questions route as intended.")

    print("\n=== routing distribution (all questions) ===")
    for tid, n in dist.most_common():
        flag = "" if tid in CYPHER_TEMPLATES else "  <-- MISSING TEMPLATE"
        print(f"  {n:>3}  {tid}{flag}")


if __name__ == "__main__":
    main()
