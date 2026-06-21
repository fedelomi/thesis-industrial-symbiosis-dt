"""check_routing_diff.py - free routing-level regression gate.

No API, no Neo4j. Compares the template each benchmark question routes to NOW
(current code, honouring ENABLE_SEMANTIC_FALLBACK and the chosen backend)
against a frozen v4 routing snapshot in data/_baseline_routing_v4.json.

Why a routing diff and not check_regressions.py?
  check_regressions.py compares the per-question strict exact_match of two
  stored answer JSONs. Those JSONs are only produced by a (paid) eval run, so it
  cannot reflect a routing change until after such a run. This gate is the FREE
  proxy: a routing change is a necessary precondition for any strict-EM change,
  so "only the intended questions changed route" bounds the strict-EM blast
  radius to zero on the untouched questions.

Snapshot the v4 baseline once (before editing the router):
    python -c "import json; from step_3_4_evaluation import route_question; \
        d=json.load(open('data/benchmark_qa_dataset.json',encoding='utf-8')); \
        json.dump({q['id']:route_question(q['nl_question']) for q in d}, \
        open('data/_baseline_routing_v4.json','w'), indent=2)"

Then, after changes:
    python check_routing_diff.py

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

from step_3_4_evaluation import ENABLE_SEMANTIC_FALLBACK, route_question
from semantic_router import DEFAULT_BACKEND

HERE = Path(__file__).resolve().parent
DATASET = HERE / "data" / "benchmark_qa_dataset.json"
BASELINE = HERE / "data" / "_baseline_routing_v4.json"

# Verified routing changes vs the v4 baseline introduced by the semantic
# fallback. Each was checked against the question ground truth and the RETURN
# fields of the old and new templates: a change is allowed only if the new
# template's context contains the asked fact at least as well as the old one
# (IMPROVEMENT) or both expose it equally (NEUTRAL). Format: id -> (old, new).
# The targeted collisions B17/C23/C06 are NOT here: the semantic stage lands
# them on the same template the removed priority rules did, so their final route
# is unchanged and they must not appear in the diff at all.
VERIFIED_CHANGES: dict[str, tuple[str, str]] = {
    # NEUTRAL: ALL_REGULATORY_ARTICLES returns r.id 'EED-2023-1791' (carries 2023).
    "A04": ("GENERIC_REGULATION", "ALL_REGULATORY_ARTICLES"),
    # IMPROVEMENT: full IS pathway to sector; P5 is DC-L-only and cannot trace it.
    "B09": ("P5_scenario_comparison_L", "P6_full_is_path"),
    # IMPROVEMENT: incentive eligibility lives in P4, not the compatibility table.
    "B13": ("P2_thermal_compatibility_all", "P4_incentives_it_whr"),
    # IMPROVEMENT: needs the 25.8 EUR/MWh value, only P4 returns it.
    "B30": ("P2_thermal_compatibility_all", "P4_incentives_it_whr"),
    # IMPROVEMENT: exact per-sector scenario counts vs counting P6 rows.
    "C07": ("P6_full_is_path", "SCENARIO_COUNT_BY_SECTOR"),
    # IMPROVEMENT: needs supply_temp_c; P6 returns no temperatures.
    "C20": ("P6_full_is_path", "P5_scenario_comparison_L"),
    # IMPROVEMENT: needs supply vs required temp for the 130C process; P6 has none.
    "C22": ("P6_full_is_path", "P2_thermal_compatibility_all"),
    # IMPROVEMENT: absorption-chiller eligibility is in P4.eligible_tech.
    "C24": ("P2_thermal_compatibility_all", "P4_incentives_it_whr"),
    # IMPROVEMENT: a 3GDH-vs-4GDH comparison needs both rows (DK_DH_COMPARE).
    "C29": ("DK_4GDH_PARAMS", "DK_DH_COMPARE"),
}


def main() -> None:
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    if not BASELINE.exists():
        print(f"Baseline snapshot missing: {BASELINE.name}")
        print("Create it from the pre-change code (see module docstring).")
        return
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    by_id = {q["id"]: q["nl_question"] for q in questions}
    changes: list[tuple[str, str, str]] = []
    for qid, nl in by_id.items():
        new = route_question(nl)
        old = baseline.get(qid, "<absent>")
        if new != old:
            changes.append((qid, old, new))

    print(f"backend={DEFAULT_BACKEND}  ENABLE_SEMANTIC_FALLBACK={ENABLE_SEMANTIC_FALLBACK}")
    print(f"baseline={BASELINE.name}  questions={len(by_id)}")
    print(f"\nROUTING CHANGES vs v4 baseline: {len(changes)}")
    for qid, old, new in sorted(changes):
        print(f"  {qid}: {old}  ->  {new}")
        print(f"        Q: {by_id[qid]}")

    unexpected = []
    for qid, old, new in changes:
        verified = VERIFIED_CHANGES.get(qid)
        if verified != (old, new):
            unexpected.append((qid, old, new))

    print(f"\nUnexpected route changes (NOT in the verified set): {len(unexpected)}")
    for qid, old, new in sorted(unexpected):
        print(f"  - {qid}: {old} -> {new}")
    print(
        "\nGATE: PASS (all route changes are pre-verified improvements/neutral)"
        if not unexpected
        else "\nGATE: REVIEW (a change is unverified or differs from the audited set)"
    )
    print("NOTE: routing-level proxy; strict-EM check_regressions needs a paid run.")


if __name__ == "__main__":
    main()
