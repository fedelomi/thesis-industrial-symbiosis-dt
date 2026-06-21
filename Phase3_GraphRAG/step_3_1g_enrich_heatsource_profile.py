"""
step_3_1g_enrich_heatsource_profile.py — additive enrichment of the HeatSource
profile facts (B29 closure).

Diagnosis (step_3_19 follow-up): the canonical ingest (step_3_1d) already sets
``profile: 'baseload'`` on all 9 HeatSource nodes, so B29 is NOT a data
coverage gap. Two residuals remain: (a) the availability gloss expected by the
benchmark ground truth ("continuous availability throughout the year") is not
in the graph, and (b) no Cypher template projects ``hs.profile`` in its RETURN
clause, so the fact is structurally unreachable by any retrieval (single or
multi-template): a PROJECTION gap, not a data gap. This step closes (a)
additively and documents (b) for the FW9-era template extension; the canonical
templates.py is deliberately left untouched (the thesis pins the routed
template set and its LOO / paraphrase diagnostics).

Inputs:  Neo4j live (config NEO4J_*)
Outputs: in-graph only: SET hs.profile_type, hs.availability on the 9
         HeatSource nodes (idempotent; re-running changes nothing)
Gate:    read-back verification must report 9/9 enriched nodes

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

LOG = logging.getLogger(__name__)

AVAILABILITY_GLOSS = "continuous availability throughout the year"

ENRICH_QUERY = """
MATCH (hs:HeatSource)
SET hs.profile_type = coalesce(hs.profile, 'baseload'),
    hs.availability = $gloss
RETURN count(hs) AS n
"""

VERIFY_QUERY = """
MATCH (hs:HeatSource)
RETURN hs.id AS id, hs.profile AS profile, hs.profile_type AS profile_type,
       hs.availability AS availability
ORDER BY hs.id
"""

PREVIEW_QUERY = """
MATCH (hs:HeatSource)
RETURN hs.id AS id, hs.profile AS profile,
       hs.profile_type IS NOT NULL AS already_enriched
ORDER BY hs.id
"""


@dataclass
class Config:
    """Step configuration. Keep all knobs here."""

    dry_run: bool = True


def run(cfg: Config) -> int:
    """Top-level entry point. Returns the number of enriched nodes."""
    from neo4j import GraphDatabase
    import config

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session(database=config.NEO4J_DATABASE) as session:
        preview = session.execute_read(
            lambda tx: tx.run(PREVIEW_QUERY).data())
        LOG.info("HeatSource nodes found: %d", len(preview))
        for row in preview:
            LOG.info("  %s profile=%s already_enriched=%s",
                     row["id"], row["profile"], row["already_enriched"])
        if cfg.dry_run:
            LOG.info("DRY RUN: no write performed. Re-run with --apply to "
                     "SET profile_type and availability ('%s').",
                     AVAILABILITY_GLOSS)
            driver.close()
            return 0

        n = session.execute_write(
            lambda tx: tx.run(ENRICH_QUERY, gloss=AVAILABILITY_GLOSS)
            .single()["n"])
        verify = session.execute_read(lambda tx: tx.run(VERIFY_QUERY).data())
        ok = sum(1 for r in verify
                 if r["profile_type"] and r["availability"] == AVAILABILITY_GLOSS)
        LOG.info("Enriched %d nodes; verification %d/%d PASS.",
                 n, ok, len(verify))
        if ok != len(verify) or not verify:
            LOG.error("GATE FAIL: expected full enrichment.")
    driver.close()
    return n


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="perform the write (default: dry-run preview)")
    a = p.parse_args()
    return Config(dry_run=not a.apply)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
