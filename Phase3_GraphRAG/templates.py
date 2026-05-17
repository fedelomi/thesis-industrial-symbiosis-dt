"""
templates.py
============
Phase 3 - Cypher templates condivisi (unica fonte di verita').

Importato da:
  - step_3_2_graph_rag_pipeline.py  (QUERY_TEMPLATES usa cypher da qui)
  - step_3_4_evaluation.py          (CYPHER_TEMPLATES importato direttamente)

I template coprono i 6 pattern multi-hop definiti in [[graph-rag-entity-schema]]
piu' i template generici per lookup su singola label.

Pattern:
  P1 - Lookup fattuale (regolatorio)        2-hop
  P2 - Compatibilita termica DC -> processo 4-hop
  P3 - Screening normativo DK               3-hop
  P4 - Incentivi IT per WHR                 3-hop
  P5 - Confronto scenari DC-L               5-hop
  P6 - Path IS completo                     5-hop
  GENERIC_* - Lookup su singola label       1-hop
"""

from __future__ import annotations

CYPHER_TEMPLATES: dict[str, str] = {

    # ------------------------------------------------------------------
    # P1 — Lookup: soglia obbligo disclosure EED Art.26
    # ------------------------------------------------------------------
    "P1_eed_art26_threshold": """
        MATCH (r:Regulation {id: 'EED-2023-1791'})-[:CONTAINS]->(a:RegulatoryArticle {id: 'EED-ART-26'})
        RETURN a.title AS article, a.obligation_type AS obligation,
               a.threshold_value AS threshold_mw, a.threshold_unit AS unit,
               a.summary AS summary
    """,

    # ------------------------------------------------------------------
    # P2a — Compatibilita termica: Scenario S1 (DC-S x T1)
    # ------------------------------------------------------------------
    "P2_thermal_compatibility_S1": """
        MATCH (s:Scenario {id: 'S1'})
        MATCH (s)-[:USES_DC]->(dc:DataCenter)
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
        MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource)
        MATCH (hs)-[:IN_BAND]->(tb:TemperatureBand)
        RETURN s.id AS scenario,
               dc.id AS datacenter, dc.it_capacity_kw AS dc_kw,
               mp.name AS target_process, mp.temp_required_c AS process_temp_c,
               hs.upgrade_tech AS upgrade_tech, hs.temp_supply_c AS supply_temp_c,
               tb.id AS band,
               CASE WHEN hs.temp_supply_c >= mp.temp_required_c
                    THEN 'COMPATIBLE' ELSE 'REQUIRES_UPGRADE' END AS compatibility
    """,

    # ------------------------------------------------------------------
    # P2b — Compatibilita termica: tutti i 9 scenari
    # ------------------------------------------------------------------
    "P2_thermal_compatibility_all": """
        MATCH (s:Scenario)
        MATCH (s)-[:USES_DC]->(dc:DataCenter)
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
        MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource)
        RETURN s.id AS scenario,
               dc.scale AS dc_scale,
               mp.name AS process,
               hs.upgrade_tech AS upgrade_tech,
               hs.temp_supply_c AS supply_c,
               mp.temp_required_c AS required_c,
               CASE WHEN hs.temp_supply_c >= mp.temp_required_c
                    THEN 'OK' ELSE 'GAP' END AS status
        ORDER BY s.id
    """,

    # ------------------------------------------------------------------
    # P3 — Screening normativo: obblighi DK per DC > 1 MW
    # ------------------------------------------------------------------
    "P3_regulatory_screening_dk": """
        MATCH (c:Country {iso: 'DK'})-[:HAS_FRAMEWORK]->(pf:PolicyFramework)
        MATCH (pf)-[:CONTAINS]->(a:RegulatoryArticle)
        WHERE a.obligation_type = 'mandatory'
        MATCH (dc:DataCenter) WHERE dc.it_capacity_kw >= 1000
        RETURN c.name AS country,
               pf.name AS framework,
               a.title AS obligation,
               a.threshold_value AS threshold_mw,
               a.summary AS summary,
               collect(dc.id) AS compliant_dcs
    """,

    # ------------------------------------------------------------------
    # P4 — Incentivi IT per waste heat recovery
    # ------------------------------------------------------------------
    "P4_incentives_it_whr": """
        MATCH (i:Incentive)
        WHERE 'waste-heat-recovery' IN i.eligible_tech
        MATCH (i)-[:GOVERNED_BY]->(r:Regulation)
        RETURN i.name AS incentive,
               i.value_eur_toe AS eur_per_toe,
               i.value_eur_mwh AS eur_per_mwh,
               i.eligible_tech AS eligible_technologies,
               collect(r.short_name) AS governing_decrees
    """,

    # ------------------------------------------------------------------
    # P5 — Confronto scenari DC-L (hyperscale) per fascia termica
    # ------------------------------------------------------------------
    "P5_scenario_comparison_L": """
        MATCH (s:Scenario)-[:USES_DC]->(dc:DataCenter {id: 'DC-L'})
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
        MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource)
        MATCH (hs)-[:IN_BAND]->(tb:TemperatureBand)
        RETURN s.id AS scenario,
               tb.id AS temp_band,
               mp.name AS process,
               mp.temp_required_c AS process_temp_c,
               hs.upgrade_tech AS upgrade_tech,
               hs.capacity_kw AS available_kw,
               hs.temp_supply_c AS supply_temp_c
        ORDER BY tb.id
    """,

    # ------------------------------------------------------------------
    # P6 — Path IS completo: DC -> HeatSource -> Band -> Process -> Sector
    # ------------------------------------------------------------------
    "P6_full_is_path": """
        MATCH path = (dc:DataCenter)-[:PRODUCES_HEAT]->(hs:HeatSource)-[:IN_BAND]->(tb:TemperatureBand)
        MATCH (s:Scenario)-[:USES_DC]->(dc)
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)-[:PART_OF]->(sector:IndustrialSector)
        RETURN dc.id AS datacenter,
               dc.scale AS dc_scale,
               hs.upgrade_tech AS upgrade,
               tb.id AS band,
               mp.name AS process,
               sector.name AS sector,
               s.id AS scenario
        ORDER BY dc.id, tb.id
    """,

    # ------------------------------------------------------------------
    # GENERIC — Tutti gli articoli regolatori con summary
    # ------------------------------------------------------------------
    "ALL_REGULATORY_ARTICLES": """
        MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle)
        RETURN r.id AS regulation, r.short_name AS reg_name,
               a.id AS article_id, a.article_number AS art_no,
               a.title AS title, a.obligation_type AS obligation,
               a.threshold_value AS threshold_mw,
               a.summary AS summary
        ORDER BY r.id, a.article_number
    """,

    # ------------------------------------------------------------------
    # GENERIC — ISO 50001: articoli e relazioni con processi manifatturieri
    # ------------------------------------------------------------------
    "ISO50001_ARTICLES": """
        MATCH (s:Standard {id: 'ISO-50001-2018'})-[:CONTAINS]->(a:RegulatoryArticle)
        RETURN s.id AS standard, s.name AS standard_name,
               s.amendment AS amendment,
               a.article_number AS article, a.title AS title,
               a.obligation_type AS obligation, a.summary AS summary
        ORDER BY a.article_number
    """,

    # ------------------------------------------------------------------
    # GENERIC — ISO 50001 -> ManufacturingProcess (multi-hop)
    # ------------------------------------------------------------------
    "ISO50001_PROCESS_COMPLIANCE": """
        MATCH (mp:ManufacturingProcess)-[:REQUIRES_COMPLIANCE]->(s:Standard {id: 'ISO-50001-2018'})
        MATCH (s)-[:CONTAINS]->(a:RegulatoryArticle)
        OPTIONAL MATCH (mp)-[:PART_OF]->(sector:IndustrialSector)
        RETURN mp.id AS process_id, mp.name AS process,
               mp.temp_required_c AS temp_c,
               sector.name AS sector,
               collect(a.article_number) AS iso_clauses
        ORDER BY mp.temp_required_c
    """,

    # ------------------------------------------------------------------
    # GENERIC — Lookup su singola label
    # ------------------------------------------------------------------
    "GENERIC_REGULATION": """
        MATCH (r:Regulation)
        OPTIONAL MATCH (r)-[:CONTAINS]->(a:RegulatoryArticle)
        RETURN r.id AS reg_id, r.name AS name, r.jurisdiction AS jurisdiction,
               r.year AS year, collect(a.title) AS articles
        ORDER BY r.jurisdiction, r.year
    """,

    "GENERIC_COUNTRY": """
        MATCH (c:Country)
        RETURN c.iso AS iso, c.name AS name,
               c.dh_penetration_pct AS dh_pct, c.eed_transposed AS eed_transposed
        ORDER BY c.iso
    """,

    "GENERIC_STANDARD": """
        MATCH (s:Standard)
        RETURN s.id AS id, s.name AS name, s.issuer AS issuer,
               s.year AS year, s.scope AS scope
        ORDER BY s.issuer
    """,

    "GENERIC_DC": """
        MATCH (dc:DataCenter)
        RETURN dc.id AS id, dc.scale AS scale, dc.it_capacity_kw AS it_kw,
               dc.cooling_type AS cooling, dc.pue_nominal AS pue,
               dc.waste_heat_kw AS waste_heat_kw
        ORDER BY dc.it_capacity_kw
    """,
}
