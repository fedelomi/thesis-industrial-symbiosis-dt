"""
run_phase_3.py
==============
Phase 3 — Graph RAG IS (OS3, Strato 2)

Full Phase 3 pipeline orchestrator (audit 2026-05-05, FIX-3).

Sequence (post-reorg):
  step_3_0   -> Neo4j schema (constraints + indexes)
  step_3_1a  -> Tier A ingest (EED, ASHRAE, TemperatureBand, DC, Scenarios)
  step_3_1b  -> Tier B ingest IT (TEE/CB decrees, GSE)
  step_3_1c  -> Tier B ingest DK (NECP, DH networks, DEA)
  step_3_1d  -> HeatSources × upgrade tech, Scenario relationships
  step_3_1e  -> ISO 50001 ingest (D4 — Prof. Simeone integral text)
  step_3_2   -> Graph RAG retrieval pipeline
  step_3_3   -> Benchmark QA design (D3 — document-grounded protocol)
  step_3_4   -> Evaluation (RAGAS faithfulness, answer relevancy, context precision)
  step_3_4_bis -> Neuro-symbolic consistency check (D10 NREL pattern)
  step_3_5   -> Phase 1 LC integration (KG context layer + ΔTC mapping)
  step_3_6   -> Privacy preservation gate (extracted from step_3_5 — FIX-6)

Usage:
    python run_phase_3.py                  # full run (graph reset + all steps)
    python run_phase_3.py --from-step 3    # resume from step index N
    python run_phase_3.py --verify-only    # only verify graph counts
    python run_phase_3.py --ingest-only    # stop after step_3_1e (no RAG/QA)
    python run_phase_3.py --skip-rag       # skip steps 3.2-3.4-bis (only ingest + 3.5/3.6)

Wiki references:
- [[phase-1-2-3-roadmap]] FASE 3
- [[implementation-decisions]] D3, D4
- [[concepts/graph-rag]], [[concepts/is-match-score]]
"""

# Decisions active: D3 — document-grounded benchmark.
#                   D4 — ISO 50001 integral text.

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase, exceptions as neo4j_exc


# --- Configurazione ---

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

# Logical step index → script. Indices are stable so --from-step is meaningful.
STEPS: list[tuple[int, str]] = [
    (0,  "step_3_0_neo4j_schema.py"),
    (1,  "step_3_1a_ingest_tier_a.py"),
    (2,  "step_3_1b_ingest_tier_b_it.py"),
    (3,  "step_3_1c_ingest_tier_b_dk.py"),
    (4,  "step_3_1d_ingest_scenarios_heatsources.py"),
    (5,  "step_3_1e_ingest_iso50001.py"),
    (6,  "step_3_2_graph_rag_pipeline.py"),
    (7,  "step_3_3_benchmark_qa_design.py"),
    (8,  "step_3_4_evaluation.py"),
    (9,  "step_3_4_bis_neuro_symbolic.py"),
    (10, "step_3_5_phase1_integration.py"),
    (11, "step_3_6_privacy_gate.py"),
]

LAST_INGEST_STEP = 5
RAG_STEPS        = {6, 7, 8, 9}

EXPECTED_NODES: dict[str, int] = {
    "Actor":                2,
    "Country":              3,
    "DataCenter":           3,
    "DHNetwork":            2,
    "Document":             2,
    "HeatSource":           9,
    "Incentive":            1,
    "IndustrialSector":     3,
    "ManufacturingProcess": 9,
    "PolicyFramework":      2,
    "Regulation":           7,
    "RegulatoryArticle":    6,
    "Scenario":             9,
    "Standard":             2,
    "TemperatureBand":      3,
}

EXPECTED_RELS: dict[str, int] = {
    "IN_BAND":         9,
    "USES_DC":         9,
    "TARGETS_PROCESS": 9,
    "HAS_HEATSOURCE":  9,
    "PRODUCES_HEAT":   9,
    "HAS_FRAMEWORK":   2,
    "GOVERNED_BY":     4,
    "PART_OF":         9,
}


def run_step(step_n: int, script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP {step_n}: {script}")
    print(f"{'='*60}")
    here = Path(__file__).resolve().parent
    script_path = here / script
    if not script_path.exists():
        print(f"  [ERROR] script not found: {script_path}")
        return False
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(here))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"\n  --> Step {step_n} {'OK' if ok else 'FAILED'} in {elapsed:.1f}s")
    return ok


def verify_graph() -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as exc:
        return False, [f"Cannot connect: {exc}"]

    with driver.session(database=NEO4J_DATABASE) as session:
        node_rows = session.execute_read(
            lambda tx: tx.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label"
            ).data()
        )
        rel_rows = session.execute_read(
            lambda tx: tx.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY rel"
            ).data()
        )

    driver.close()

    actual_nodes = {r["label"]: r["count"] for r in node_rows}
    actual_rels  = {r["rel"]:   r["count"] for r in rel_rows}

    print("\n-- Node counts -----------------------------------------------")
    for label, expected in sorted(EXPECTED_NODES.items()):
        actual = actual_nodes.get(label, 0)
        sym = "OK  " if actual >= expected else "WARN"
        print(f"  [{sym}]  {label:<25} actual={actual}  expected>={expected}")
        if actual < expected:
            issues.append(f"Node {label}: got {actual}, expected>={expected}")

    print("\n-- Relationship counts ----------------------------------------")
    for rel, expected in sorted(EXPECTED_RELS.items()):
        actual = actual_rels.get(rel, 0)
        sym = "OK  " if actual >= expected else "WARN"
        print(f"  [{sym}]  {rel:<25} actual={actual}  expected>={expected}")
        if actual < expected:
            issues.append(f"Rel {rel}: got {actual}, expected>={expected}")

    total_nodes = sum(actual_nodes.values())
    total_rels  = sum(actual_rels.values())
    print(f"\n  Total nodes         : {total_nodes}")
    print(f"  Total relationships : {total_rels}")

    return len(issues) == 0, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 full pipeline orchestrator")
    parser.add_argument("--from-step", type=int, default=0, metavar="N",
                        help="Skip steps before N (0-11). Default: 0")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip all steps, only verify graph")
    parser.add_argument("--ingest-only", action="store_true",
                        help=f"Run only ingestion steps 0..{LAST_INGEST_STEP}")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip RAG/QA/eval steps (3.2 .. 3.4-bis)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Phase 3 - run_phase_3.py")
    print("  Full Phase 3 orchestrator (ingest + RAG + integration)")
    print("=" * 60)

    if not args.verify_only:
        steps_to_run = []
        for n, s in STEPS:
            if n < args.from_step:
                continue
            if args.ingest_only and n > LAST_INGEST_STEP:
                continue
            if args.skip_rag and n in RAG_STEPS:
                continue
            steps_to_run.append((n, s))

        t0 = time.time()
        for step_n, script in steps_to_run:
            ok = run_step(step_n, script)
            if not ok:
                print(f"\nStep {step_n} failed -- aborting.")
                print(f"Fix the error and re-run with: --from-step {step_n}")
                sys.exit(1)

        print(f"\nAll {len(steps_to_run)} steps completed in {time.time()-t0:.1f}s")

    print("\n-- Final graph verification -----------------------------------")
    ok, issues = verify_graph()

    if ok:
        print("\nGraph verification PASSED.")
    else:
        print(f"\nGraph verification: {len(issues)} issue(s):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)


if __name__ == "__main__":
    main()
