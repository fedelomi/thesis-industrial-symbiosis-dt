"""
step_3_1e_ingest_iso50001.py
============================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Ingesta ISO 50001:2018+A1:2024 nel Knowledge Graph IS.

Nodi creati:
  1 Standard          (ISO-50001-2018)
  5 RegulatoryArticle (§4.1, §6.3, §6.4, §9.1, §9.1.2)
  1 Document          (slug: bs-en-iso-50001-2018-a1-2024)

Relazioni:
  Standard -[:CONTAINS]-> RegulatoryArticle (x5)
  Standard -[:REQUIRES_COMPLIANCE]-> ManufacturingProcess (x9)
  Standard -[:DEFINES]-> RegulatoryArticle (EnPI framework)

Rilevanza tesi (Strato 2 — Institutional-LLM):
  - §6.3 Energy review: obbligo di identificare SEU e opportunita' WHR
    -> manufacturing plants certificati ISO 50001 DEVONO documentare WHR come opportunita'
  - §6.4 EnPI: allineamento con KPI TWH/ERF/EREUSE di Del.Reg 2024/1364
  - §9.1.1: obbligo monitoraggio SEU -> vincola il DC a misure oggettive
  - §4.1 (A1:2024): "climate change shall be determined as a relevant issue"
    -> rafforza il business case IS-WHR anche per gli impianti manifatturieri

Fonte: BS EN ISO 50001:2018+A1:2024 (incorpora ISO amendment 1:2024)
       Documento disponibile in raw/F - DC WHR Cases & Policy/

Uso:
    python step_3_1e_ingest_iso50001.py

Prerequisiti:
  - step_3_0_neo4j_schema.py eseguito
  - step_3_1a_ingest_tier_a.py eseguito (ManufacturingProcess x9 devono esistere)

Riferimento wiki: [[roadmap-fasi-1-2-3]] Passo 3.1 (D4),
                  [[graph-rag-entity-schema]] Layer 2
"""

from __future__ import annotations
import os

import sys
from neo4j import GraphDatabase, exceptions as neo4j_exc


# --- Configurazione ---

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = "neo4j"


# ============================================================
# Dati estratti da BS EN ISO 50001:2018+A1:2024
# ============================================================

ISO50001_STANDARD = {
    "id":    "ISO-50001-2018",
    "name":  "Energy Management Systems — Requirements with guidance for use",
    "issuer": "ISO/CEN",
    "year":  2018,
    "amendment": "A1:2024",
    "scope": "energy-management-system",
    "notes": "BS EN ISO 50001:2018+A1:2024. Incorpora amendment 1:2024 (climate change §4.1). "
             "Applicabile a qualsiasi organizzazione indipendentemente da tipo, dimensione, "
             "settore geografico. Voluntary standard ma spesso richiesto da supply chain. "
             "Sostituisce ISO 50001:2011.",
}

ISO50001_ARTICLES = [
    {
        "id": "ISO50001-4-1",
        "regulation_id": "ISO-50001-2018",
        "article_number": "4.1",
        "title": "Understanding the organization and its context",
        "obligation_type": "mandatory",
        "threshold_value": 0.0,
        "threshold_unit": None,
        "summary": (
            "Organization shall determine external and internal issues relevant to its purpose "
            "affecting EnMS and energy performance. "
            "A1:2024 adds: 'The organization shall determine whether climate change is a relevant issue.' "
            "For DC operators: climate targets create strategic driver for WHR partnerships."
        ),
    },
    {
        "id": "ISO50001-6-3",
        "regulation_id": "ISO-50001-2018",
        "article_number": "6.3",
        "title": "Energy review",
        "obligation_type": "mandatory",
        "threshold_value": 0.0,
        "threshold_unit": None,
        "summary": (
            "Organization shall develop and conduct an energy review: "
            "(a) analyse energy use and consumption; "
            "(b) identify SEUs (Significant Energy Uses); "
            "(c) for each SEU determine relevant variables, current performance, persons responsible; "
            "(d) determine and prioritize opportunities for improving energy performance; "
            "(e) estimate future energy use. "
            "WHR explicitly qualifies as an energy performance improvement opportunity per §A.6.3: "
            "'organizations should consider the extent to which energy is recoverable'."
        ),
    },
    {
        "id": "ISO50001-6-4",
        "regulation_id": "ISO-50001-2018",
        "article_number": "6.4",
        "title": "Energy performance indicators (EnPI)",
        "obligation_type": "mandatory",
        "threshold_value": 0.0,
        "threshold_unit": None,
        "summary": (
            "Organization shall determine EnPIs appropriate for measuring and monitoring energy performance "
            "and enabling demonstration of energy performance improvement. "
            "EnPI can be expressed as simple metric, ratio, or model. "
            "Aligns with Del.Reg 2024/1364 KPIs: TWH (Temperature of Waste Heat), ERF, EREUSE. "
            "DC operators subject to both ISO 50001 §6.4 and Del.Reg 2024/1364 must report WHR metrics."
        ),
    },
    {
        "id": "ISO50001-9-1",
        "regulation_id": "ISO-50001-2018",
        "article_number": "9.1.1",
        "title": "Monitoring, measurement, analysis and evaluation",
        "obligation_type": "mandatory",
        "threshold_value": 0.0,
        "threshold_unit": None,
        "summary": (
            "Organization shall monitor and measure key characteristics at planned intervals, including: "
            "(1) effectiveness of action plans vs objectives and energy targets; "
            "(2) EnPIs; "
            "(3) operation of SEUs; "
            "(4) actual vs expected energy consumption. "
            "For IS-WHR: manufacturing plant certified ISO 50001 must measure WHR flow as SEU "
            "if it constitutes substantial energy consumption or offers considerable improvement potential."
        ),
    },
    {
        "id": "ISO50001-9-1-2",
        "regulation_id": "ISO-50001-2018",
        "article_number": "9.1.2",
        "title": "Evaluation of compliance with legal requirements",
        "obligation_type": "mandatory",
        "threshold_value": 0.0,
        "threshold_unit": None,
        "summary": (
            "At planned intervals, organization shall evaluate compliance with legal and other requirements "
            "related to energy efficiency, energy use, energy consumption and the EnMS. "
            "For IS-WHR actors: this includes EED 2023/1791 (EU), national transpositions (IT/DK) "
            "and Del.Reg 2024/1364 reporting obligations. ISO 50001 §9.1.2 creates the internal "
            "compliance verification loop that aligns with EED Art.26 external reporting."
        ),
    },
]

ISO50001_DOCUMENT = {
    "slug":         "bs-en-iso-50001-2018-a1-2024",
    "filename":     "BS_EN_ISO_50001-2018_A1-2024.pdf",
    "tier":         "A",
    "doc_type":     "standard",
    "jurisdiction": "EU",
    "year":         2024,
    "language":     "EN",
}


# ============================================================
# Population functions
# ============================================================

def populate_iso50001_standard(session) -> None:
    print("  > Merging ISO 50001:2018 Standard node...")
    query = """
    MERGE (s:Standard {id: $id})
    SET s.name      = $name,
        s.issuer    = $issuer,
        s.year      = $year,
        s.amendment = $amendment,
        s.scope     = $scope,
        s.notes     = $notes
    """
    session.execute_write(lambda tx: tx.run(query, **ISO50001_STANDARD))
    print("    Standard ISO-50001-2018 merged.")


def populate_iso50001_articles(session) -> None:
    print("  > Merging ISO 50001 RegulatoryArticles (5 clauses)...")
    query = """
    UNWIND $data AS row
    MERGE (a:RegulatoryArticle {id: row.id})
    SET a.regulation_id   = row.regulation_id,
        a.article_number  = row.article_number,
        a.title           = row.title,
        a.obligation_type = row.obligation_type,
        a.threshold_value = row.threshold_value,
        a.summary         = row.summary
    FOREACH (u IN CASE WHEN row.threshold_unit IS NOT NULL THEN [1] ELSE [] END |
        SET a.threshold_unit = row.threshold_unit
    )
    WITH a, row
    MATCH (s:Standard {id: row.regulation_id})
    MERGE (s)-[:CONTAINS]->(a)
    """
    session.execute_write(lambda tx: tx.run(query, data=ISO50001_ARTICLES))
    print(f"    {len(ISO50001_ARTICLES)} articles merged with CONTAINS relationships.")


def link_iso50001_to_processes(session) -> None:
    """
    Collega ISO-50001-2018 a tutti i ManufacturingProcess via REQUIRES_COMPLIANCE.
    Questo riflette il fatto che un impianto manifatturiero certificato ISO 50001
    deve documentare WHR come opportunita' di miglioramento energetico (§6.3).
    """
    print("  > Linking ISO 50001 to ManufacturingProcesses (REQUIRES_COMPLIANCE)...")
    query = """
    MATCH (s:Standard {id: 'ISO-50001-2018'})
    MATCH (mp:ManufacturingProcess)
    MERGE (mp)-[:REQUIRES_COMPLIANCE]->(s)
    RETURN count(mp) AS linked
    """
    result = session.execute_write(lambda tx: tx.run(query).data())
    n = result[0]["linked"] if result else 0
    print(f"    {n} ManufacturingProcess nodes linked via REQUIRES_COMPLIANCE.")


def populate_iso50001_document(session) -> None:
    print("  > Merging Document metadata...")
    query = """
    MERGE (d:Document {slug: $slug})
    SET d.filename     = $filename,
        d.tier         = $tier,
        d.doc_type     = $doc_type,
        d.jurisdiction = $jurisdiction,
        d.year         = $year,
        d.language     = $language
    WITH d
    MATCH (s:Standard {id: 'ISO-50001-2018'})
    MERGE (d)-[:DOCUMENTS]->(s)
    """
    session.execute_write(lambda tx: tx.run(query, **ISO50001_DOCUMENT))
    print("    Document node merged with DOCUMENTS relationship.")


def print_summary(session) -> None:
    print("\n-- Summary (ISO 50001) ----------------------------------------")

    # Standard
    std = session.execute_read(
        lambda tx: tx.run(
            "MATCH (s:Standard {id: 'ISO-50001-2018'}) RETURN s.name AS name, s.amendment AS amend"
        ).data()
    )
    if std:
        print(f"  Standard  : {std[0]['name']} [{std[0]['amend']}]")

    # Articles
    arts = session.execute_read(
        lambda tx: tx.run(
            "MATCH (s:Standard {id:'ISO-50001-2018'})-[:CONTAINS]->(a:RegulatoryArticle) "
            "RETURN a.article_number AS art, a.title AS title ORDER BY art"
        ).data()
    )
    print(f"  Articles  : {len(arts)}")
    for a in arts:
        print(f"    §{a['art']}: {a['title']}")

    # REQUIRES_COMPLIANCE
    n_rc = session.execute_read(
        lambda tx: tx.run(
            "MATCH ()-[:REQUIRES_COMPLIANCE]->(s:Standard {id:'ISO-50001-2018'}) RETURN count(*) AS n"
        ).data()
    )[0]["n"]
    print(f"  REQUIRES_COMPLIANCE: {n_rc} ManufacturingProcess nodes")

    # Query multi-hop di verifica
    path_check = session.execute_read(
        lambda tx: tx.run(
            "MATCH (mp:ManufacturingProcess)-[:REQUIRES_COMPLIANCE]->(s:Standard {id:'ISO-50001-2018'}) "
            "-[:CONTAINS]->(a:RegulatoryArticle {id:'ISO50001-6-3'}) "
            "RETURN mp.name AS process, a.title AS clause LIMIT 3"
        ).data()
    )
    if path_check:
        print(f"  Multi-hop check (MP -> ISO50001 -> §6.3): {len(path_check)} paths verified")
        for row in path_check:
            print(f"    {row['process']} -> {row['clause']}")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    print("=" * 60)
    print("  Phase 3 - step_3_1e_ingest_iso50001.py")
    print("  ISO 50001:2018+A1:2024 — Energy Management Systems")
    print("=" * 60)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to {NEO4J_URI}\n")
    except neo4j_exc.ServiceUnavailable:
        print("Neo4j not reachable.")
        sys.exit(1)
    except neo4j_exc.AuthError:
        print("Auth error.")
        sys.exit(1)

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            populate_iso50001_standard(session)
            populate_iso50001_articles(session)
            link_iso50001_to_processes(session)
            populate_iso50001_document(session)
            print_summary(session)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        raise
    finally:
        driver.close()
        print("\nConnection closed.")
        print("Next: aggiornare benchmark_qa_dataset con domande ISO 50001 (step_3_3)")
        print("      e aggiungere GENERIC_STANDARD template per ISO-50001 in step_3_4")


if __name__ == "__main__":
    main()
