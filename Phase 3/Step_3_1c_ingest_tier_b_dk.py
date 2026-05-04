"""
step_3_1c_ingest_tier_b_dk.py
==============================
Phase 3 — Graph RAG IS (OS3, Strato 2)

Popola il cluster regolatorio danese (Tier B):
  - 2 PolicyFramework  (NECP 2024 + DH Planning Act)
  - 1 RegulatoryArticle (obbligo connessione DH >1MW — Varmeforsyningsloven)
  - 2 DHNetwork         (3GDH generico + 4GDH generico — abilitante per LC DC)
  - 1 Actor             (Danish Energy Agency — regolatore DH)

Prerequisiti:
  - step_3_0_neo4j_schema.py già eseguito
  - step_3_1a_ingest_tier_a.py già eseguito (Country DK deve esistere)

Uso:
    python step_3_1c_ingest_tier_b_dk.py

Riferimento wiki: [[sources/dk-final-necp-2024]],
                  [[sources/regulation-and-planning-district-heating-denmark]],
                  [[concepts/district-heating-generations]],
                  [[entities/stockholm-exergi]]
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

def populate_dk_policy_frameworks(session) -> None:
    """NECP 2024 finale + Varmeforsyningsloven come PolicyFramework."""
    print("  > Populating Danish PolicyFrameworks...")
    frameworks = [
        {
            "id": "DK-NECP-2024",
            "name": "Danish National Energy and Climate Plan 2024",
            "country": "DK",
            "type": "necp",
            "horizon_year": 2030,
            "status": "final",
            "notes": "Target 70% GHG reduction by 2030; DC waste heat in DH strategy; 4GDH transition",
        },
        {
            "id": "DK-DH-PLANNING-ACT",
            "name": "Varmeforsyningsloven — Heat Supply Act",
            "country": "DK",
            "type": "national-dh-plan",
            "horizon_year": 2035,
            "status": "final",
            "notes": "Municipal DH planning obligation; connection mandate >1MW; 3GDH→4GDH transition compatibility",
        },
    ]
    query = """
    UNWIND $data AS row
    MERGE (pf:PolicyFramework {id: row.id})
    SET pf += row
    WITH pf
    MATCH (c:Country {iso: 'DK'})
    MERGE (c)-[:HAS_FRAMEWORK]->(pf)
    """
    session.execute_write(lambda tx: tx.run(query, data=frameworks))
    print(f"    {len(frameworks)} policy frameworks merged.")


def populate_dk_regulatory_articles(session) -> None:
    """Obbligo connessione DH per impianti >1MW — Varmeforsyningsloven."""
    print("  > Populating Danish RegulatoryArticles...")
    articles = [
        {
            "id": "DK-DH-ACT-CONNECTION-MANDATE",
            "regulation_id": "DK-DH-PLANNING-ACT",
            "article_number": "§11",
            "title": "DH Connection Mandate",
            "obligation_type": "mandatory",
            "threshold_value": 1.0,
            "threshold_unit": "MW",
            "summary": "Heat sources >1MW in municipal DH planning areas must connect to DH network if economically feasible",
        },
        {
            "id": "DK-NECP-DC-WHR-TARGET",
            "regulation_id": "DK-NECP-2024",
            "article_number": "Ch.4",
            "title": "Data Center Waste Heat Integration",
            "obligation_type": "voluntary",
            "threshold_value": None,
            "threshold_unit": None,
            "summary": "DC waste heat identified as strategic DH supply source in 4GDH transition plan",
        },
    ]
    # threshold_value può essere None — gestito con proprietà condizionale
    query = """
    UNWIND $data AS row
    MERGE (a:RegulatoryArticle {id: row.id})
    SET a.regulation_id    = row.regulation_id,
        a.article_number   = row.article_number,
        a.title            = row.title,
        a.obligation_type  = row.obligation_type,
        a.summary          = row.summary
    FOREACH (v IN CASE WHEN row.threshold_value IS NOT NULL THEN [1] ELSE [] END |
        SET a.threshold_value = row.threshold_value,
            a.threshold_unit  = row.threshold_unit
    )
    WITH a, row
    MATCH (pf:PolicyFramework {id: row.regulation_id})
    MERGE (pf)-[:CONTAINS]->(a)
    """
    session.execute_write(lambda tx: tx.run(query, data=articles))
    print(f"    {len(articles)} regulatory articles merged.")


def populate_dk_dh_networks(session) -> None:
    """2 DHNetwork generici DK: 3GDH (legacy) e 4GDH (abilitante per LC DC).

    Thesis relevance: 4GDH supply 55-65°C è la condizione abilitante per
    connessione DC liquid-cooled (T_supply ~47°C) senza heat pump aggiuntiva.
    Questo è il motivo del clustering di hyperscaler in Jutland (Danimarca).
    """
    print("  > Populating Danish DHNetworks (3GDH + 4GDH)...")
    networks = [
        {
            "id": "DK-3GDH-GENERIC",
            "name": "Danish 3rd Generation DH (Generic)",
            "generation": "3GDH",
            "supply_temp_c": 80.0,
            "return_temp_c": 40.0,
            "capacity_mw": 500.0,
            "country": "DK",
            "notes": "Legacy network; DC LC WHR requires HP upgrade to reach 80°C supply",
        },
        {
            "id": "DK-4GDH-GENERIC",
            "name": "Danish 4th Generation DH (Generic)",
            "generation": "4GDH",
            "supply_temp_c": 60.0,
            "return_temp_c": 35.0,
            "capacity_mw": 300.0,
            "country": "DK",
            "notes": "Enabling condition: DC LC T_supply ~47°C directly compatible without HP; reason for hyperscaler cluster in Jutland",
        },
    ]
    query = """
    UNWIND $data AS row
    MERGE (dh:DHNetwork {id: row.id})
    SET dh += row
    WITH dh
    MATCH (pf:PolicyFramework {id: 'DK-DH-PLANNING-ACT'})
    MERGE (dh)-[:GOVERNED_BY]->(pf)
    """
    session.execute_write(lambda tx: tx.run(query, data=networks))
    print(f"    {len(networks)} DH networks merged.")


def populate_dk_actors(session) -> None:
    """Danish Energy Agency — regolatore DH e DC WHR reporting."""
    print("  > Populating Danish Actors...")
    actors = [
        {
            "id": "DK-DEA",
            "name": "Danish Energy Agency",
            "type": "regulator",
            "country": "DK",
            "notes": "Enforces Varmeforsyningsloven; coordinates NECP DC WHR targets; manages DH planning assessments",
        },
    ]
    query = """
    UNWIND $data AS row
    MERGE (a:Actor {id: row.id})
    SET a += row
    """
    session.execute_write(lambda tx: tx.run(query, data=actors))
    print(f"    {len(actors)} actor merged.")


def print_summary(session) -> None:
    """Conta nodi DK e verifica relazioni chiave."""
    print("\n── Summary (DK Cluster) ──────────────────────────────────")

    count_query = """
    MATCH (n)
    WHERE (n:PolicyFramework  AND n.country = 'DK')
       OR (n:DHNetwork        AND n.country = 'DK')
       OR (n:Actor            AND n.country = 'DK')
       OR (n:RegulatoryArticle AND (n.regulation_id STARTS WITH 'DK-'))
    RETURN labels(n)[0] AS label, count(n) AS count
    ORDER BY label
    """
    rows = session.execute_read(lambda tx: tx.run(count_query).data())
    for row in rows:
        print(f"  {row['label']}: {row['count']}")

    # Verifica GOVERNED_BY DH → PolicyFramework
    gov_query = """
    MATCH (dh:DHNetwork)-[:GOVERNED_BY]->(pf:PolicyFramework)
    RETURN dh.id AS network, pf.id AS framework
    ORDER BY network
    """
    rels = session.execute_read(lambda tx: tx.run(gov_query).data())
    for row in rels:
        print(f"  GOVERNED_BY: {row['network']} → {row['framework']}")

    # Verifica HAS_FRAMEWORK Country → PolicyFramework
    fw_query = """
    MATCH (c:Country {iso:'DK'})-[:HAS_FRAMEWORK]->(pf:PolicyFramework)
    RETURN pf.id AS framework
    ORDER BY framework
    """
    fws = session.execute_read(lambda tx: tx.run(fw_query).data())
    for row in fws:
        print(f"  HAS_FRAMEWORK: DK → {row['framework']}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Phase 3 — step_3_1c_ingest_tier_b_dk.py")
    print("  Danish Regulatory Cluster (Tier B)")
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
            populate_dk_policy_frameworks(session)
            populate_dk_regulatory_articles(session)
            populate_dk_dh_networks(session)
            populate_dk_actors(session)
            print_summary(session)
    except Exception as exc:
        print(f"\n✗ Unexpected error: {exc}")
        sys.exit(1)
    finally:
        driver.close()
        print("\n✓ Connection closed.")
        print("  Next: python step_3_1d_ingest_scenarios_heatsources.py")


if __name__ == "__main__":
    main()