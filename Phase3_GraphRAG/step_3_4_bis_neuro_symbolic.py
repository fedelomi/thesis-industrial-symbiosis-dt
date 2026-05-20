"""
step_3_4_bis_neuro_symbolic.py
================================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Neuro-symbolic consistency check (Passo 3.4-bis della roadmap).

Per ogni domanda del benchmark, verifica che le entita' e le relazioni
citate nella risposta (o nel contesto recuperato) esistano realmente
nel grafo Neo4j. Questo e' il "symbolic layer" che controlla il
"neural layer" (LLM o template RAG).

Logica:
  Per ogni EvalResult (da step_3_4):
    1. Estrae entity_ids menzionati nel contesto (es. "DC-S", "EED-ART-26")
    2. Verifica che ogni entity_id esista nel grafo come nodo
    3. Verifica che le relazioni attese per quel template esistano
    4. Calcola logical_consistency_rate = n_coerenti / n_totali

Metrica tesi (da [[phase-1-2-3-roadmap]] Passo 3.4-bis):
    logical_consistency_rate >= 0.90

Uso:
    python step_3_4_bis_neuro_symbolic.py

    # Su un file di risultati specifico
    python step_3_4_bis_neuro_symbolic.py --results data/evaluation_results_graph-rag_<ts>.json

Output:
    data/neuro_symbolic_check_<timestamp>.json
    Stampa: logical_consistency_rate e breakdown per categoria

Riferimento wiki: [[phase-1-2-3-roadmap]] Passo 3.4-bis,
                  [[graph-rag-entity-schema]] relationship types
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase, exceptions as neo4j_exc

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, DATA_DIR


# --- Pattern per estrarre entity IDs dal contesto ---
#
# I contesti sono stringhe tipo:
#   "scenario: S1 | datacenter: DC-S | ..."
#   "article: Data Centers | obligation: mandatory | threshold_mw: 1.0 ..."
#   "regulation: EED-2023-1791 | reg_name: ..."

ENTITY_PATTERNS: list[tuple[str, str]] = [
    # (pattern_regex, node_label)
    (r'\bS[1-9]\b',                          "Scenario"),
    (r'\bDC-[SML]\b',                        "DataCenter"),
    (r'\bHS-[SML]-\w+',                      "HeatSource"),
    (r'\bT[123]\b',                          "TemperatureBand"),
    (r'\bEED-ART-\d+\b',                     "RegulatoryArticle"),
    (r'\bDEL-REG-ART-\d+\b',                 "RegulatoryArticle"),
    (r'\bDK-DH-ACT-[\w-]+',                  "RegulatoryArticle"),
    (r'\bDK-NECP-DC-[\w-]+',                 "RegulatoryArticle"),
    (r'\bEED-\d{4}-\d+\b',                   "Regulation"),
    (r'\bDEL-REG-\d{4}-\d+\b',              "Regulation"),
    (r'\bIT-D[MD]-[\w-]+',                   "Regulation"),
    (r'\bDK-(?:NECP|DH)-[\w-]+',            "PolicyFramework"),
    (r'\bDK-[34]GDH-\w+',                   "DHNetwork"),
    (r'\bIT-TEE-CB\b',                       "Incentive"),
    (r'\bGSE\b',                             "Actor"),
    (r'\bDK-DEA\b',                          "Actor"),
    (r'\bASHRAE-[\d.]+\b',                   "Standard"),
    (r'\bISO-\d+(?:-\d+)?\b',               "Standard"),
    (r'\bISO50001-[\d-]+\b',                 "RegulatoryArticle"),
    (r'\bPROC-[\w-]+',                       "ManufacturingProcess"),
    (r'\b(?:food-drink-milk|paper-pulp|rubber-chemical)\b', "IndustrialSector"),
    (r'\b(?:IT|DK|EU)\b',                    "Country"),
]

# Relazioni attese per ogni cypher template
TEMPLATE_RELATIONS: dict[str, list[tuple[str, str, str]]] = {
    # (from_label, rel_type, to_label)
    "P1_eed_art26_threshold": [
        ("Regulation", "CONTAINS", "RegulatoryArticle"),
    ],
    "P2_thermal_compatibility_S1": [
        ("Scenario", "USES_DC", "DataCenter"),
        ("Scenario", "TARGETS_PROCESS", "ManufacturingProcess"),
        ("Scenario", "HAS_HEATSOURCE", "HeatSource"),
        ("HeatSource", "IN_BAND", "TemperatureBand"),
    ],
    "P2_thermal_compatibility_all": [
        ("Scenario", "USES_DC", "DataCenter"),
        ("Scenario", "TARGETS_PROCESS", "ManufacturingProcess"),
        ("Scenario", "HAS_HEATSOURCE", "HeatSource"),
    ],
    "P3_regulatory_screening_dk": [
        ("Country", "HAS_FRAMEWORK", "PolicyFramework"),
        ("PolicyFramework", "CONTAINS", "RegulatoryArticle"),
    ],
    "P4_incentives_it_whr": [
        ("Incentive", "GOVERNED_BY", "Regulation"),
    ],
    "P5_scenario_comparison_L": [
        ("Scenario", "USES_DC", "DataCenter"),
        ("HeatSource", "IN_BAND", "TemperatureBand"),
    ],
    "P6_full_is_path": [
        ("DataCenter", "PRODUCES_HEAT", "HeatSource"),
        ("HeatSource", "IN_BAND", "TemperatureBand"),
        ("ManufacturingProcess", "PART_OF", "IndustrialSector"),
        ("Scenario", "USES_DC", "DataCenter"),
        ("Scenario", "TARGETS_PROCESS", "ManufacturingProcess"),
    ],
    "ALL_REGULATORY_ARTICLES": [
        ("Regulation", "CONTAINS", "RegulatoryArticle"),
    ],
    "ISO50001_ARTICLES": [
        ("Standard", "CONTAINS", "RegulatoryArticle"),
    ],
    "ISO50001_PROCESS_COMPLIANCE": [
        ("ManufacturingProcess", "REQUIRES_COMPLIANCE", "Standard"),
        ("Standard", "CONTAINS", "RegulatoryArticle"),
    ],
    "GENERIC_DC": [],
    "GENERIC_STANDARD": [],
    "GENERIC_REGULATION": [],
    "GENERIC_COUNTRY": [],
}


# --- Struttura risultato check ---

@dataclass
class SymbolicCheckResult:
    question_id:           str
    category:              str
    cypher_template:       str
    entities_found:        list[str]
    entities_verified:     int
    entities_missing:      int
    missing_entities:      list[str]
    relations_verified:    int
    relations_missing:     int
    missing_relations:     list[str]
    logical_consistent:    bool
    consistency_score:     float   # 0.0 - 1.0


# --- Verifica nodi ---

def extract_entity_ids(context: str) -> list[tuple[str, str]]:
    """Estrae (entity_id, label) dal contesto."""
    found = []
    for pattern, label in ENTITY_PATTERNS:
        matches = re.findall(pattern, context, re.IGNORECASE)
        for m in matches:
            found.append((m.upper() if len(m) <= 3 else m, label))
    # Deduplicazione
    return list(dict.fromkeys(found))


def verify_nodes(session, entities: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """
    Verifica che i nodi esistano nel grafo.
    Ritorna (verified_ids, missing_ids).
    """
    if not entities:
        return [], []

    verified = []
    missing = []

    for entity_id, label in entities:
        # Prova con proprieta' 'id' prima, poi 'iso' (Country), poi 'slug' (Document)
        prop = "iso" if label == "Country" else ("slug" if label == "Document" else "id")
        query = f"MATCH (n:{label} {{{prop}: $eid}}) RETURN count(n) AS cnt"
        try:
            result = session.execute_read(
                lambda tx, q=query, e=entity_id: tx.run(q, eid=e).data()
            )
            if result and result[0]["cnt"] > 0:
                verified.append(entity_id)
            else:
                missing.append(entity_id)
        except Exception:
            missing.append(entity_id)

    return verified, missing


def verify_relations(session, template: str) -> tuple[int, list[str]]:
    """
    Verifica che le relazioni attese per il template esistano nel grafo.
    Ritorna (n_verified, missing_rel_descriptions).
    """
    expected = TEMPLATE_RELATIONS.get(template, [])
    if not expected:
        return 0, []

    missing = []
    verified = 0

    for from_label, rel_type, to_label in expected:
        query = (
            f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
            f"RETURN count(r) AS cnt"
        )
        try:
            result = session.execute_read(lambda tx, q=query: tx.run(q).data())
            if result and result[0]["cnt"] > 0:
                verified += 1
            else:
                missing.append(f"({from_label})-[:{rel_type}]->({to_label})")
        except Exception as exc:
            missing.append(f"({from_label})-[:{rel_type}]->({to_label}) [err: {exc}]")

    return verified, missing


# --- Check principale ---

def run_symbolic_check(
    eval_results: list[dict],
    driver,
) -> list[SymbolicCheckResult]:
    """Esegue il check su tutti i risultati di valutazione."""

    check_results = []
    n = len(eval_results)

    with driver.session(database=NEO4J_DATABASE) as session:
        for i, r in enumerate(eval_results):
            print(f"    {i+1}/{n}: {r['question_id']}", end=" ", flush=True)

            context = r.get("context", "") or ""
            template = r.get("cypher_used", "") or "UNKNOWN"

            # 1. Estrai e verifica entita'
            entities = extract_entity_ids(context)
            verified_ids, missing_ids = verify_nodes(session, entities)

            # 2. Verifica relazioni del template
            n_rel_ok, missing_rels = verify_relations(session, template)
            n_rel_expected = len(TEMPLATE_RELATIONS.get(template, []))

            # 3. Calcola score
            total_checks = len(entities) + n_rel_expected
            passed_checks = len(verified_ids) + n_rel_ok

            if total_checks == 0:
                score = 1.0   # nessun check da fare = consistente per definizione
                consistent = True
            else:
                score = passed_checks / total_checks
                consistent = score >= 0.80   # soglia per singola domanda

            print(f"score={score:.2f} entities={len(verified_ids)}/{len(entities)} rels={n_rel_ok}/{n_rel_expected}")

            check_results.append(SymbolicCheckResult(
                question_id=r["question_id"],
                category=r["category"],
                cypher_template=template,
                entities_found=[e for e, _ in entities],
                entities_verified=len(verified_ids),
                entities_missing=len(missing_ids),
                missing_entities=missing_ids,
                relations_verified=n_rel_ok,
                relations_missing=len(missing_rels),
                missing_relations=missing_rels,
                logical_consistent=consistent,
                consistency_score=score,
            ))

    return check_results


# --- Report ---

def print_report(results: list[SymbolicCheckResult]) -> float:
    n = len(results)
    n_consistent = sum(1 for r in results if r.logical_consistent)
    lcr = n_consistent / n if n > 0 else 0.0

    print(f"\n-- Neuro-Symbolic Check Report --------------------------------")
    print(f"  Total questions        : {n}")
    print(f"  Logically consistent   : {n_consistent}")
    print(f"  logical_consistency_rate: {lcr:.1%}")
    print(f"  Target (>= 90%)        : {'PASS' if lcr >= 0.90 else 'FAIL'}")

    # Per categoria
    cats = sorted(set(r.category for r in results))
    print(f"\n  By category:")
    for cat in cats:
        cat_r = [r for r in results if r.category == cat]
        cat_lcr = sum(1 for r in cat_r if r.logical_consistent) / len(cat_r)
        print(f"    {cat:<25} LCR={cat_lcr:.1%} ({len(cat_r)} samples)")

    # Per template
    print(f"\n  By cypher template:")
    templates = sorted(set(r.cypher_template for r in results))
    for tpl in templates:
        tpl_r = [r for r in results if r.cypher_template == tpl]
        tpl_lcr = sum(1 for r in tpl_r if r.logical_consistent) / len(tpl_r)
        print(f"    {tpl:<35} LCR={tpl_lcr:.1%} ({len(tpl_r)})")

    # Entita' mancanti piu' frequenti
    all_missing = []
    for r in results:
        all_missing.extend(r.missing_entities)
    if all_missing:
        from collections import Counter
        top_missing = Counter(all_missing).most_common(5)
        print(f"\n  Top missing entities:")
        for eid, cnt in top_missing:
            print(f"    {eid:<20} missing in {cnt} checks")

    return lcr


def save_results(results: list[SymbolicCheckResult], lcr: float) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"neuro_symbolic_check_{ts}.json"

    output = {
        "logical_consistency_rate": lcr,
        "n_questions": len(results),
        "n_consistent": sum(1 for r in results if r.logical_consistent),
        "target_passed": lcr >= 0.90,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path


# --- Entry point ---

def find_latest_eval_results() -> Path | None:
    """Trova il file evaluation_results piu' recente in data/."""
    files = sorted(DATA_DIR.glob("evaluation_results_graph-rag_*.json"), reverse=True)
    return files[0] if files else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuro-symbolic consistency check")
    parser.add_argument(
        "--results", type=str, default="",
        help="Path al file evaluation_results JSON (default: piu' recente in data/)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Phase 3 - step_3_4_bis_neuro_symbolic.py")
    print("  Logical consistency check on Graph RAG results")
    print("=" * 60)

    # Trova il file dei risultati
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = find_latest_eval_results()

    if not results_path or not results_path.exists():
        print(f"Nessun file evaluation_results trovato in {DATA_DIR}/")
        print("Esegui prima: python step_3_4_evaluation.py --config graph-rag")
        sys.exit(1)

    print(f"  Input: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        eval_results = json.load(f)
    print(f"  Loaded: {len(eval_results)} evaluation results")

    # Connessione Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"  Neo4j connected: {NEO4J_URI}\n")
    except (neo4j_exc.ServiceUnavailable, neo4j_exc.AuthError) as exc:
        print(f"  Neo4j error: {exc}")
        sys.exit(1)

    try:
        print("-- Running symbolic checks ----------------------------------")
        check_results = run_symbolic_check(eval_results, driver)
        lcr = print_report(check_results)
        out_path = save_results(check_results, lcr)
        print(f"\n  Saved: {out_path}")
        print("\nNext: step_3_5_phase1_integration.py  (ΔTC update + IS-Match rerun)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
