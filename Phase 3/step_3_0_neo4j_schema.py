"""
step_3_0_neo4j_schema.py
========================
Phase 3 — Graph RAG IS (OS3, Strato 2)

Inizializza il database Neo4j con:
  - Unique constraints su tutti i nodi primari
  - Index su proprietà frequentemente interrogate

Da eseguire UNA SOLA VOLTA prima di qualsiasi ingestione.
Se lo esegui di nuovo su un DB già inizializzato non causa errori
(le constraint esistenti vengono skippate).

Requisiti:
    pip install neo4j
    Neo4j Desktop in esecuzione su bolt://localhost:7687

Uso:
    python step_3_0_neo4j_schema.py

Riferimento wiki: [[graph-rag-entity-schema]], [[roadmap-fasi-1-2-3]] §Passo 3.0
"""

from __future__ import annotations
import os

import sys
import time
from dataclasses import dataclass

from neo4j import GraphDatabase, exceptions as neo4j_exc


# ─── Configurazione ──────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")   # <-- cambia con la tua password Neo4j Desktop


# ─── Schema: constraint e index ──────────────────────────────────────────────
#
# Struttura a 4 layer (da [[graph-rag-entity-schema]]):
#   Layer 0  — Meta (Document)
#   Layer 1  — Physical-DT
#   Layer 2  — Institutional-LLM
#   Layer 2b — Contract Models
#   Layer 3  — Agentic Negotiation

CONSTRAINTS: list[tuple[str, str, str]] = [
    # (label, property, constraint_name)
    # Layer 0
    ("Document",            "slug",         "constraint_document_slug"),
    # Layer 1
    ("Standard",            "id",           "constraint_standard_id"),
    ("DataCenter",          "id",           "constraint_datacenter_id"),
    ("HeatSource",          "id",           "constraint_heatsource_id"),
    ("TemperatureBand",     "id",           "constraint_tempband_id"),
    ("ManufacturingProcess","id",           "constraint_mfgprocess_id"),
    ("IndustrialSector",    "id",           "constraint_indsector_id"),
    ("Scenario",            "id",           "constraint_scenario_id"),
    # Layer 2
    ("Regulation",          "id",           "constraint_regulation_id"),
    ("RegulatoryArticle",   "id",           "constraint_regarticle_id"),
    ("PolicyFramework",     "id",           "constraint_policyfw_id"),
    ("Incentive",           "id",           "constraint_incentive_id"),
    ("Country",             "iso",          "constraint_country_iso"),
    # Layer 2b
    ("ContractModel",       "id",           "constraint_contractmodel_id"),
    # Layer 3
    ("Actor",               "id",           "constraint_actor_id"),
    ("ISProject",           "id",           "constraint_isproject_id"),
    ("DHNetwork",           "id",           "constraint_dhnetwork_id"),
]

INDEXES: list[tuple[str, str, str]] = [
    # (label, property, index_name)
    # Query frequenti su temperature
    ("ManufacturingProcess", "temp_required_c",  "idx_mfgprocess_temp"),
    ("HeatSource",           "temp_supply_c",     "idx_heatsource_temp"),
    # Query frequenti su obblighi
    ("RegulatoryArticle",    "obligation_type",   "idx_regarticle_obligation"),
    ("RegulatoryArticle",    "threshold_value",   "idx_regarticle_threshold"),
    # Query frequenti su paese
    ("Actor",                "country",           "idx_actor_country"),
    ("ISProject",            "country",           "idx_isproject_country"),
    ("Regulation",           "jurisdiction",      "idx_regulation_jurisdiction"),
    # Query frequenti su scala DC e fascia termica (9 scenari)
    ("Scenario",             "dc_scale",          "idx_scenario_scale"),
    ("Scenario",             "temp_band",         "idx_scenario_tempband"),
    ("DataCenter",           "scale",             "idx_datacenter_scale"),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

@dataclass
class SchemaResult:
    constraints_created: int
    constraints_skipped: int
    indexes_created: int
    indexes_skipped: int
    errors: list[str]


def _run_constraint(tx, label: str, prop: str, name: str) -> bool:
    """Crea constraint di unicità. Restituisce True se creata, False se già esiste."""
    cypher = (
        f"CREATE CONSTRAINT {name} IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
    )
    tx.run(cypher)
    return True


def _run_index(tx, label: str, prop: str, name: str) -> bool:
    """Crea index btree. Restituisce True se creato, False se già esiste."""
    cypher = (
        f"CREATE INDEX {name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{prop})"
    )
    tx.run(cypher)
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def initialize_schema(uri: str, user: str, password: str) -> SchemaResult:
    result = SchemaResult(0, 0, 0, 0, [])

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"✓ Connesso a Neo4j: {uri}")
    except neo4j_exc.ServiceUnavailable as exc:
        print(f"✗ Neo4j non raggiungibile: {exc}")
        print("  Verifica che Neo4j Desktop sia avviato e il DB 'thesis-is-graph' sia attivo.")
        sys.exit(1)
    except neo4j_exc.AuthError as exc:
        print(f"✗ Credenziali errate: {exc}")
        print(f"  Controlla NEO4J_USER='{user}' e NEO4J_PASSWORD in cima allo script.")
        sys.exit(1)

    with driver.session() as session:
        # ── Constraints ──────────────────────────────────────────────────────
        print("\n── Constraints ──────────────────────────────────────────────")
        for label, prop, name in CONSTRAINTS:
            try:
                session.execute_write(_run_constraint, label, prop, name)
                print(f"  ✓ {name}  ({label}.{prop})")
                result.constraints_created += 1
            except neo4j_exc.ClientError as exc:
                if "already exists" in str(exc).lower() or "equivalent" in str(exc).lower():
                    print(f"  · {name}  (già esistente)")
                    result.constraints_skipped += 1
                else:
                    msg = f"ERRORE constraint {name}: {exc}"
                    print(f"  ✗ {msg}")
                    result.errors.append(msg)

        # Attendi che Neo4j propaghi i constraint prima di creare gli index
        time.sleep(0.5)

        # ── Indexes ──────────────────────────────────────────────────────────
        print("\n── Indexes ──────────────────────────────────────────────────")
        for label, prop, name in INDEXES:
            try:
                session.execute_write(_run_index, label, prop, name)
                print(f"  ✓ {name}  ({label}.{prop})")
                result.indexes_created += 1
            except neo4j_exc.ClientError as exc:
                if "already exists" in str(exc).lower() or "equivalent" in str(exc).lower():
                    print(f"  · {name}  (già esistente)")
                    result.indexes_skipped += 1
                else:
                    msg = f"ERRORE index {name}: {exc}"
                    print(f"  ✗ {msg}")
                    result.errors.append(msg)

    driver.close()
    return result


def verify_schema(uri: str, user: str, password: str) -> None:
    """Mostra un riepilogo di constraint e index presenti nel DB."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        constraints = session.run("SHOW CONSTRAINTS").data()
        indexes     = session.run("SHOW INDEXES").data()

    n_constraints = len(constraints)
    n_indexes     = len([i for i in indexes if i.get("type") != "LOOKUP"])  # esclude LOOKUP built-in

    print(f"\n── Verifica schema ──────────────────────────────────────────")
    print(f"  Constraint nel DB : {n_constraints}")
    print(f"  Index nel DB      : {n_indexes}  (esclusi LOOKUP built-in)")

    if n_constraints < len(CONSTRAINTS):
        print(f"  ⚠ Attesi {len(CONSTRAINTS)} constraint — verifica errori sopra")
    else:
        print(f"  ✓ Tutti i {len(CONSTRAINTS)} constraint presenti")

    driver.close()


def main() -> None:
    print("=" * 60)
    print("  Phase 3 — step_3_0_neo4j_schema.py")
    print("  Inizializzazione schema grafo IS")
    print("=" * 60)

    result = initialize_schema(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("\n── Riepilogo ────────────────────────────────────────────────")
    print(f"  Constraint create  : {result.constraints_created}")
    print(f"  Constraint skipped : {result.constraints_skipped}")
    print(f"  Index creati       : {result.indexes_created}")
    print(f"  Index skipped      : {result.indexes_skipped}")

    if result.errors:
        print(f"\n  ✗ {len(result.errors)} errori:")
        for e in result.errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        verify_schema(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print("\n✓ Schema inizializzato correttamente.")
        print("  Prossimo passo: python step_3_1a_ingest_tier_a.py")


if __name__ == "__main__":
    main()
