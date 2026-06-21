"""
step_3_1f_ingest_eed_audit.py
=============================
Phase 3 - Graph RAG IS (OS3, Strato 2). KG enrichment (PROMPT 4, Fase 4a).

Closes the high-confidence kg_missing coverage gaps on EED energy audits
(OOD16, OOD21, OOD34) by adding the EED 2023/1791 energy-audit / energy-
management articles to the Knowledge Graph. Before this, the KG only modelled
EED Art. 23/24/26 plus the delegated regulation, so any independently posed
question about energy audits hit a coverage gap.

Nodes created:
  RegulatoryArticle EED-ART-11 (Energy management systems and energy audits)
  RegulatoryArticle EED-ART-12 (Energy management systems)

Relationship:
  (Regulation {id:'EED-2023-1791'}) -[:CONTAINS]-> (RegulatoryArticle)   x2

Schema note (spec reconciliation): the prompt referenced
`(:RegulatoryArticle)-[:PART_OF]->(:Regulation {celex_id:'EED 2023/1791'})`.
The live KG instead links Regulation -[:CONTAINS]-> RegulatoryArticle and keys
the regulation by `id='EED-2023-1791'`. This ingest follows the LIVE schema so
the new nodes are reachable by the existing ALL_REGULATORY_ARTICLES template
(which the OOD16/21/34 routes use). A `celex_id` property is also set for
forward compatibility.

LEGAL-ACCURACY FLAG (read before citing): in EED 2023/1791 the energy-audit /
energy-management-system provisions are Article 11. The OOD16 benchmark query
refers to "Article 12 ... energy audit", which appears to be a generation-side
numbering error. Art. 11 below carries the substantive, legally correct content;
Art. 12 is added per the Fase 4a spec but flagged status='PROVISIONAL' and
needs_verification=True. Verify the exact CELEX text of Art. 12 before any
thesis citation.

Idempotent: uses MERGE on id, safe to re-run. Rollback in the module footer.

Usage:
    python step_3_1f_ingest_eed_audit.py

Prerequisiti:
  - step_3_0_neo4j_schema.py eseguito (constraint_regarticle_eed_article)
  - EED-2023-1791 Regulation node esistente (step_3_1a)

Riferimento wiki: [[phase-1-2-3-roadmap]] Passo 3.1, [[graph-rag-entity-schema]] Layer 2.
Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import sys

from neo4j import GraphDatabase, exceptions as neo4j_exc

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

EED_REGULATION_ID = "EED-2023-1791"
EED_CELEX = "EED 2023/1791"

# Provisional pending researcher verification of the exact CELEX article text.
EED_AUDIT_ARTICLES = [
    {
        "id": "EED-ART-11",
        "regulation_id": EED_REGULATION_ID,
        "celex_id": EED_CELEX,
        "celex_path": "Directive (EU) 2023/1791 Art. 11",
        "article_number": "11",
        "eed_article": 11,
        "title": "Energy management systems and energy audits",
        "obligation_type": "mandatory",
        "scope": "audit",
        "summary": (
            "Enterprises with high energy consumption shall implement an energy management "
            "system; enterprises above the consumption thresholds that do not implement an "
            "energy management system shall be subject to an energy audit carried out at "
            "least every four years by qualified or accredited experts. Audits shall be "
            "proportionate, representative, and lead to identified, costed and prioritised "
            "recommendations; Member States shall ensure recommendations are followed up. "
            "For a data center this is the entry obligation that surfaces waste heat reuse as "
            "a recommended energy-efficiency measure."
        ),
        "status": "PROVISIONAL",
        "needs_verification": True,
    },
    {
        "id": "EED-ART-12",
        "regulation_id": EED_REGULATION_ID,
        "celex_id": EED_CELEX,
        "celex_path": "Directive (EU) 2023/1791 Art. 12",
        "article_number": "12",
        "eed_article": 12,
        "title": "Energy management systems",
        "obligation_type": "mandatory",
        "scope": "management",
        "summary": (
            "Energy management system provisions for enterprises. NOTE: added per the Fase 4a "
            "specification to resolve benchmark queries that cite 'Article 12' for energy "
            "audits; in EED 2023/1791 the audit obligation is in Art. 11. Verify the exact "
            "CELEX content of Art. 12 before citation."
        ),
        "status": "PROVISIONAL",
        "needs_verification": True,
    },
]


def ingest_articles(session) -> int:
    query = """
    MATCH (r:Regulation {id: $reg_id})
    UNWIND $data AS row
    MERGE (a:RegulatoryArticle {id: row.id})
    SET a.regulation_id     = row.regulation_id,
        a.celex_id          = row.celex_id,
        a.celex_path        = row.celex_path,
        a.article_number    = row.article_number,
        a.eed_article       = row.eed_article,
        a.title             = row.title,
        a.obligation_type   = row.obligation_type,
        a.scope             = row.scope,
        a.summary           = row.summary,
        a.status            = row.status,
        a.needs_verification = row.needs_verification
    MERGE (r)-[:CONTAINS]->(a)
    RETURN count(a) AS n
    """
    res = session.execute_write(
        lambda tx: tx.run(query, reg_id=EED_REGULATION_ID, data=EED_AUDIT_ARTICLES).data()
    )
    return res[0]["n"] if res else 0


def validate(session) -> list[dict]:
    """The Fase 4a validation query (adapted to the live CONTAINS/id schema)."""
    query = """
    MATCH (r:Regulation {id: $reg_id})-[:CONTAINS]->(a:RegulatoryArticle)
    WHERE a.eed_article IN [11, 12]
    RETURN a.id AS id, a.article_number AS art, a.title AS title,
           a.scope AS scope, a.status AS status
    ORDER BY a.eed_article
    """
    return session.execute_read(lambda tx: tx.run(query, reg_id=EED_REGULATION_ID).data())


def main() -> None:
    print("=" * 60)
    print("  Phase 3 - step_3_1f_ingest_eed_audit.py")
    print("  EED 2023/1791 Art. 11 / 12 (energy audits and management)")
    print("=" * 60)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to {NEO4J_URI}\n")
    except (neo4j_exc.ServiceUnavailable, neo4j_exc.AuthError) as exc:
        print(f"Neo4j not reachable: {exc}")
        sys.exit(1)

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # Guard: the parent regulation must exist.
            exists = session.execute_read(
                lambda tx: tx.run(
                    "MATCH (r:Regulation {id:$id}) RETURN count(r) AS n", id=EED_REGULATION_ID
                ).single()["n"]
            )
            if not exists:
                print(f"ERROR: Regulation {EED_REGULATION_ID} not found; run step_3_1a first.")
                sys.exit(1)

            n = ingest_articles(session)
            print(f"  > Merged {n} EED audit articles (CONTAINS -> {EED_REGULATION_ID}).")

            paths = validate(session)
            print(f"\n-- Validation: {len(paths)} path(s) (expected 2) --")
            for p in paths:
                print(f"  {p['id']} (Art. {p['art']}, {p['scope']}, {p['status']}): {p['title']}")
            if len(paths) != 2:
                print("  WARNING: expected exactly 2 paths.")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        raise
    finally:
        driver.close()
        print("\nConnection closed.")
        print("Rollback (if needed): "
              "MATCH (a:RegulatoryArticle) WHERE a.eed_article IN [11,12] DETACH DELETE a")


if __name__ == "__main__":
    main()
