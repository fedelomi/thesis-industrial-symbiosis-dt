"""
step_3_1b_ingest_tier_b_it.py
==============================
Phase 3 — Graph RAG IS (OS3, Strato 2)

Popola il cluster regolatorio italiano (Tier B):
  - 5 Regulation  (DM fondante 2021 + 3 DD attuativi + DM MASE 2025)
  - 1 Incentive   (TEE/CB con eligible_tech e doppia GOVERNED_BY)
  - 1 Actor       (GSE — gestore operativo CB)

Prerequisiti:
  - step_3_0_neo4j_schema.py già eseguito
  - step_3_1a_ingest_tier_a.py già eseguito (Country IT deve esistere)

Uso:
    python step_3_1b_ingest_tier_b_it.py

Riferimento wiki: [[sources/decreto-ministeriale-21-maggio-2021-tee]],
                  [[sources/decreto-mase-certificati-bianchi-2025]],
                  [[entities/gse-gestore-servizi-energetici]],
                  [[concepts/certificati-bianchi-tee]]
"""

from __future__ import annotations
import os

import sys
from neo4j import GraphDatabase, exceptions as neo4j_exc


# ─── Configurazione ───────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = "neo4j"


# ─── Population functions ─────────────────────────────────────────────────────

def populate_it_regulations(session) -> None:
    """5 decreti IT: DM fondante + 3 DD attuativi + DM MASE 2025."""
    print("  > Populating Italian Regulations (Tier B)...")
    it_regs = [
        {
            "id": "IT-DM-2021-05-21",
            "name": "Decreto Ministeriale 21 maggio 2021",
            "short_name": "DM 2021 (fondante)",
            "jurisdiction": "IT",
            "type": "ministerial-decree",
            "year": 2021,
            "in_force": True,
            "notes": "Decreto-madre sistema TEE/CB; soglia 25 tep; durata 5/10 anni; M&V obbligatorio",
        },
        {
            "id": "IT-DD-2022-05-03",
            "name": "Decreto Direttoriale 3 maggio 2022",
            "short_name": "DD Maggio 2022",
            "jurisdiction": "IT",
            "type": "directorial-decree",
            "year": 2022,
            "in_force": True,
            "notes": "Schede tecniche TEE 2022 — IND/RET per WHR",
        },
        {
            "id": "IT-DD-2023-05-04",
            "name": "Decreto Direttoriale 4 maggio 2023",
            "short_name": "DD Maggio 2023",
            "jurisdiction": "IT",
            "type": "directorial-decree",
            "year": 2023,
            "in_force": True,
            "notes": "Schede tecniche TEE 2023 — aggiornamento",
        },
        {
            "id": "IT-DD-2023-10-13",
            "name": "Decreto Direttoriale 13 ottobre 2023",
            "short_name": "DD Ottobre 2023",
            "jurisdiction": "IT",
            "type": "directorial-decree",
            "year": 2023,
            "in_force": True,
            "notes": "Schede tecniche TEE aggiornamento autunno 2023",
        },
        {
            "id": "IT-DM-MASE-2025-07-21",
            "name": "Decreto MASE 21 luglio 2025 Certificati Bianchi",
            "short_name": "DM MASE 2025",
            "jurisdiction": "IT",
            "type": "ministerial-decree",
            "year": 2025,
            "in_force": True,
            "notes": "Aggiornamento CB post-EED; possibile prima scheda DC WHR esplicita",
        },
    ]

    query = """
    UNWIND $data AS row
    MERGE (r:Regulation {id: row.id})
    SET r += row
    WITH r
    MATCH (c:Country {iso: 'IT'})
    MERGE (c)-[:IMPLEMENTS]->(r)
    """
    session.execute_write(lambda tx: tx.run(query, data=it_regs))
    print(f"    {len(it_regs)} regulations merged.")


def populate_it_incentives(session) -> None:
    """TEE/CB — collegato sia al decreto fondante che all'aggiornamento 2025."""
    print("  > Populating Italian Incentives (TEE/CB)...")

    incentives = [
        {
            "id": "IT-TEE-CB",
            "name": "Titoli di Efficienza Energetica / Certificati Bianchi",
            "short_name": "TEE/CB",
            "country": "IT",
            "type": "white-certificate",
            "value_eur_toe": 300.0,
            "value_eur_mwh": 25.8,        # 300 / 11.63 MWh/toe
            "eligible_tech": ["waste-heat-recovery", "heat-pump", "absorption-chiller"],
            "governing_regs": ["IT-DM-2021-05-21", "IT-DM-MASE-2025-07-21"],
        }
    ]

    # SET proprietà scalari esplicitamente (eligible_tech è lista — ok in Neo4j)
    # UNWIND governing_regs per creare relazioni multiple GOVERNED_BY
    query = """
    UNWIND $data AS row
    MERGE (i:Incentive {id: row.id})
    SET i.name            = row.name,
        i.short_name      = row.short_name,
        i.country         = row.country,
        i.type            = row.type,
        i.value_eur_toe   = row.value_eur_toe,
        i.value_eur_mwh   = row.value_eur_mwh,
        i.eligible_tech   = row.eligible_tech
    WITH i, row
    UNWIND row.governing_regs AS reg_id
    MATCH (r:Regulation {id: reg_id})
    MERGE (i)-[:GOVERNED_BY]->(r)
    """
    session.execute_write(lambda tx: tx.run(query, data=incentives))
    print("    1 incentive merged with 2 GOVERNED_BY relationships.")


def populate_it_actors(session) -> None:
    """GSE — gestore operativo del sistema TEE/CB."""
    print("  > Populating Italian Actors (GSE)...")
    actors = [
        {
            "id": "GSE",
            "name": "Gestore Servizi Energetici S.p.A.",
            "type": "regulator",
            "country": "IT",
            "notes": "Emette e verifica i Titoli di Efficienza Energetica; Actor:regulator nel graph RAG",
        }
    ]
    query = """
    UNWIND $data AS row
    MERGE (a:Actor {id: row.id})
    SET a += row
    """
    session.execute_write(lambda tx: tx.run(query, data=actors))
    print("    1 actor merged.")


def print_summary(session) -> None:
    """Conta nodi IT nel grafo."""
    print("\n── Summary (IT Cluster) ──────────────────────────────────")
    query = """
    MATCH (n)
    WHERE (n:Regulation  AND n.jurisdiction = 'IT')
       OR (n:Incentive   AND n.country = 'IT')
       OR (n:Actor       AND n.country = 'IT')
    RETURN labels(n)[0] AS label, count(n) AS count
    ORDER BY label
    """
    rows = session.execute_read(lambda tx: tx.run(query).data())
    for row in rows:
        print(f"  {row['label']}: {row['count']}")

    # Verifica relazioni GOVERNED_BY
    rel_query = """
    MATCH (i:Incentive)-[:GOVERNED_BY]->(r:Regulation)
    RETURN i.id AS incentive, collect(r.id) AS governed_by
    """
    rels = session.execute_read(lambda tx: tx.run(rel_query).data())
    if rels:
        for row in rels:
            print(f"  GOVERNED_BY: {row['incentive']} → {row['governed_by']}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Phase 3 — step_3_1b_ingest_tier_b_it.py")
    print("  Italian Regulatory Cluster (Tier B)")
    print("=" * 60)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"✓ Connected to {NEO4J_URI}\n")
    except neo4j_exc.ServiceUnavailable:
        print("✗ Neo4j not reachable — is the database Running in Neo4j Desktop?")
        sys.exit(1)
    except neo4j_exc.AuthError:
        print("✗ Auth error — check NEO4J_PASSWORD in script.")
        sys.exit(1)

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            populate_it_regulations(session)
            populate_it_incentives(session)
            populate_it_actors(session)
            print_summary(session)
    except Exception as exc:
        print(f"\n✗ Unexpected error: {exc}")
        sys.exit(1)
    finally:
        driver.close()
        print("\n✓ Connection closed.")
        print("  Next: python step_3_1c_ingest_tier_b_dk.py")


if __name__ == "__main__":
    main()