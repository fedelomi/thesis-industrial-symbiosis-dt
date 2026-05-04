"""
run_phase_3_ingest.py
=====================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Orchestratore completo dell'ingestione Neo4j.
Esegue in sequenza tutti gli script step_3_* e verifica i conteggi finali.

Sequenza:
  step_3_0  -> schema (constraint + index)
  step_3_1a -> Tier A (EED, ASHRAE, TemperatureBand, DC, Scenari)
  step_3_1b -> Tier B IT (decreti TEE/CB, GSE)
  step_3_1c -> Tier B DK (NECP, DH networks, DEA)
  step_3_1d -> HeatSources x upgrade tech, relazioni Scenario

Uso:
    python run_phase_3_ingest.py
    python run_phase_3_ingest.py --from-step 3
    python run_phase_3_ingest.py --verify-only

Riferimento wiki: [[roadmap-fasi-1-2-3]] Passo 3.1
"""

from __future__ import annotations
import os

import argparse
import subprocess
import sys
import time

from neo4j import GraphDatabase, exceptions as neo4j_exc


# --- Configurazione ---

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = "neo4j"

STEPS: list[tuple[int, str]] = [
    (0, "step_3_0_neo4j_schema.py"),
    (1, "step_3_1a_ingest_tier_a.py"),
    (2, "step_3_1b_ingest_tier_b_it.py"),
    (3, "step_3_1c_ingest_tier_b_dk.py"),
    (4, "step_3_1d_ingest_scenarios_heatsources.py"),
]

# Conteggi minimi attesi dopo ingestione completa
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


# --- Helpers ---

def run_step(step_n: int, script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP {step_n}: {script}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
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


# --- Entry point ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 ingest orchestrator")
    parser.add_argument("--from-step", type=int, default=0, metavar="N",
                        help="Skip steps before N (0-4). Default: 0")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip all steps, only verify graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Phase 3 - run_phase_3_ingest.py")
    print("  Full Neo4j ingestion orchestrator")
    print("=" * 60)

    if not args.verify_only:
        steps_to_run = [(n, s) for n, s in STEPS if n >= args.from_step]
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
        print("Next: python step_3_2_graph_rag_pipeline.py")
    else:
        print(f"\nGraph verification: {len(issues)} issue(s):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)


if __name__ == "__main__":
    main()