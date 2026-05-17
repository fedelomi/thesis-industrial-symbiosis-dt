"""
step_3_1d_ingest_scenarios_heatsources.py
==========================================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Crea i nodi HeatSource differenziati per upgrade tech e i 9 Scenario nodes
con le relazioni complete verso DataCenter e ManufacturingProcess.

Nodi creati:
  - 9 HeatSource  (3 DC x 3 upgrade tech: direct_HX / HP / CO2_HTHP)
  - Relazioni HeatSource -[:IN_BAND]-> TemperatureBand
  - Relazioni Scenario   -[:USES_DC]-> DataCenter
  - Relazioni Scenario   -[:TARGETS_PROCESS]-> ManufacturingProcess

Upgrade tech e temperature di supply (da Phase 1 LC + wiki [[graph-rag-entity-schema]]):
  direct_HX : T_supply = 47.6 C  (T_CDU_return medio Phase 1 LC)
  HP        : T_supply = 70.0 C  (heat pump upgrade)
  CO2_HTHP  : T_supply = 110.0 C (transcritical CO2 HTHP, max industriale)

Scenario matrix (3 DC scale x 3 temp band):
  S1 = DC-S x T1  |  S2 = DC-S x T2  |  S3 = DC-S x T3
  S4 = DC-M x T1  |  S5 = DC-M x T2  |  S6 = DC-M x T3
  S7 = DC-L x T1  |  S8 = DC-L x T2  |  S9 = DC-L x T3

Prerequisiti:
  - step_3_1a_ingest_tier_a.py gia eseguito
    (DataCenter, ManufacturingProcess, TemperatureBand, Scenario devono esistere)

Uso:
    python step_3_1d_ingest_scenarios_heatsources.py

Riferimento wiki: [[roadmap-fasi-1-2-3]] Passo 3.1,
                  [[graph-rag-entity-schema]], [[concepts/is-match-score]]
"""

from __future__ import annotations

import sys
from neo4j import GraphDatabase, exceptions as neo4j_exc

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE


# --- Mapping dati ---

# DC scale -> (dc_id, it_capacity_kw)
DC_MAP = {
    "S": ("DC-S", 500),
    "M": ("DC-M", 3200),
    "L": ("DC-L", 25000),
}

# Temp band -> (band_id, representative process id, temp_required_c)
BAND_MAP = {
    "T1": ("T1", "PROC-PASTEUR",    60.0),
    "T2": ("T2", "PROC-PAPER-DRY",  90.0),
    "T3": ("T3", "PROC-STERIL",    130.0),
}

# Upgrade tech -> T_supply_c
UPGRADE_TECH = {
    "direct_HX": 47.6,
    "HP":        70.0,
    "CO2_HTHP": 110.0,
}

# Quale upgrade tech e' rilevante per ogni banda termica
# (usato per la relazione IN_BAND - collega la HS piu appropriata alla banda)
BAND_TO_UPGRADE = {
    "T1": "direct_HX",   # 47.6 C > 40 C minimo T1 - compatibile diretto
    "T2": "HP",          # 70 C dentro T2 (60-90 C)
    "T3": "CO2_HTHP",    # 110 C dentro T3 (90-130 C)
}


# --- Population functions ---

def populate_heatsources_by_upgrade(session) -> None:
    """9 HeatSource: una per ogni (DC scale x upgrade tech).
    
    Ogni HS rappresenta la capacita termica recuperabile dal DC
    con una specifica tecnologia di upgrade.
    origin segue la nomenclatura dello schema: dc-liquid-cooling-<upgrade>
    """
    print("  > Populating HeatSources (3 DC x 3 upgrade tech = 9 nodes)...")

    heatsources = []
    for dc_key, (dc_id, it_kw) in DC_MAP.items():
        for tech, t_supply in UPGRADE_TECH.items():
            # Q_available stimato da Phase 1 LC: ~118% di IT capacity per liquid cooling
            q_available_kw = round(it_kw * 1.18, 1)
            hs = {
                "id":            f"HS-{dc_key}-{tech}",
                "dc_id":         dc_id,
                "origin":        f"dc-liquid-cooling-{tech}",
                "temp_supply_c": t_supply,
                "temp_return_c": round(t_supply - 10.0, 1),  # delta T tipico = 10 C
                "capacity_kw":   q_available_kw,
                "profile":       "baseload",
                "upgrade_tech":  tech,
                "notes":         f"DC-{dc_key} LC with {tech} upgrade; T_supply={t_supply}C",
            }
            heatsources.append(hs)

    query = """
    UNWIND $data AS row
    MERGE (hs:HeatSource {id: row.id})
    SET hs += row
    WITH hs, row
    MATCH (dc:DataCenter {id: row.dc_id})
    MERGE (dc)-[:PRODUCES_HEAT]->(hs)
    """
    session.execute_write(lambda tx: tx.run(query, data=heatsources))
    print(f"    {len(heatsources)} heat source nodes merged.")


def link_heatsources_to_bands(session) -> None:
    """Collega ogni HeatSource alla TemperatureBand compatibile.
    
    Logica: HS e' IN_BAND se temp_supply_c >= band.range_min_c.
    Per semplicita' usiamo il mapping diretto upgrade -> banda primaria.
    """
    print("  > Linking HeatSources to TemperatureBands (IN_BAND)...")

    links = []
    for dc_key in DC_MAP:
        for band_id, upgrade_tech in BAND_TO_UPGRADE.items():
            links.append({
                "hs_id":   f"HS-{dc_key}-{upgrade_tech}",
                "band_id": band_id,
            })

    query = """
    UNWIND $data AS row
    MATCH (hs:HeatSource {id: row.hs_id})
    MATCH (tb:TemperatureBand {id: row.band_id})
    MERGE (hs)-[:IN_BAND]->(tb)
    """
    session.execute_write(lambda tx: tx.run(query, data=links))
    print(f"    {len(links)} IN_BAND relationships merged.")


def link_scenarios_to_dc_and_processes(session) -> None:
    """Collega i 9 Scenario nodes a DataCenter e ManufacturingProcess.
    
    I nodi Scenario S1-S9 sono gia stati creati da step_3_1a.
    Qui aggiungiamo le relazioni USES_DC e TARGETS_PROCESS che li
    rendono navigabili nel grafo per le query multi-hop.
    """
    print("  > Linking Scenarios to DataCenters and ManufacturingProcesses...")

    scenario_n = 1
    links = []
    for dc_key, (dc_id, _) in DC_MAP.items():
        for band_key, (band_id, proc_id, _) in BAND_MAP.items():
            links.append({
                "scenario_id": f"S{scenario_n}",
                "dc_id":       dc_id,
                "proc_id":     proc_id,
                "band_id":     band_id,
            })
            scenario_n += 1

    query = """
    UNWIND $data AS row
    MATCH (s:Scenario {id: row.scenario_id})
    MATCH (dc:DataCenter {id: row.dc_id})
    MATCH (mp:ManufacturingProcess {id: row.proc_id})
    MERGE (s)-[:USES_DC]->(dc)
    MERGE (s)-[:TARGETS_PROCESS]->(mp)
    """
    session.execute_write(lambda tx: tx.run(query, data=links))
    print(f"    {len(links)} scenarios linked (USES_DC + TARGETS_PROCESS).")


def link_scenarios_to_heatsources(session) -> None:
    """Collega ogni Scenario alla HeatSource con upgrade appropriato per la banda."""
    print("  > Linking Scenarios to HeatSources (USES_DC chain)...")

    scenario_n = 1
    links = []
    for dc_key in DC_MAP:
        for band_key, (band_id, _, _) in BAND_MAP.items():
            upgrade = BAND_TO_UPGRADE[band_key]
            links.append({
                "scenario_id": f"S{scenario_n}",
                "hs_id":       f"HS-{dc_key}-{upgrade}",
            })
            scenario_n += 1

    query = """
    UNWIND $data AS row
    MATCH (s:Scenario {id: row.scenario_id})
    MATCH (hs:HeatSource {id: row.hs_id})
    MERGE (s)-[:HAS_HEATSOURCE]->(hs)
    """
    session.execute_write(lambda tx: tx.run(query, data=links))
    print(f"    {len(links)} HAS_HEATSOURCE relationships merged.")


def print_summary(session) -> None:
    print("\n-- Summary (Scenarios + HeatSources) ----------------------------")

    # Conta HeatSource per upgrade tech
    hs_query = """
    MATCH (hs:HeatSource)
    RETURN hs.upgrade_tech AS tech, count(hs) AS count,
           avg(hs.temp_supply_c) AS avg_t_supply
    ORDER BY tech
    """
    rows = session.execute_read(lambda tx: tx.run(hs_query).data())
    for row in rows:
        print(f"  HeatSource [{row['tech']}]: {row['count']} nodes, "
              f"T_supply={row['avg_t_supply']:.1f}C")

    # Verifica relazioni Scenario
    sc_query = """
    MATCH (s:Scenario)-[:USES_DC]->(dc:DataCenter)
    MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
    RETURN s.id AS scenario, dc.id AS dc, mp.id AS process
    ORDER BY s.id
    """
    rows = session.execute_read(lambda tx: tx.run(sc_query).data())
    print(f"\n  Scenarios with USES_DC + TARGETS_PROCESS: {len(rows)}/9")
    for row in rows:
        print(f"    {row['scenario']}: {row['dc']} x {row['process']}")

    # Verifica IN_BAND
    ib_query = """
    MATCH (hs:HeatSource)-[:IN_BAND]->(tb:TemperatureBand)
    RETURN count(*) AS n_in_band
    """
    n = session.execute_read(lambda tx: tx.run(ib_query).data())[0]["n_in_band"]
    print(f"\n  IN_BAND relationships: {n} (expected 9)")


# --- Entry point ---

def main() -> None:
    print("=" * 60)
    print("  Phase 3 - step_3_1d_ingest_scenarios_heatsources.py")
    print("  Scenarios S1-S9 + HeatSources with upgrade tech")
    print("=" * 60)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to {NEO4J_URI}\n")
    except neo4j_exc.ServiceUnavailable:
        print("Neo4j not reachable - is the database Running in Neo4j Desktop?")
        sys.exit(1)
    except neo4j_exc.AuthError:
        print("Auth error - check NEO4J_PASSWORD in script.")
        sys.exit(1)

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            populate_heatsources_by_upgrade(session)
            link_heatsources_to_bands(session)
            link_scenarios_to_dc_and_processes(session)
            link_scenarios_to_heatsources(session)
            print_summary(session)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        raise
    finally:
        driver.close()
        print("\nConnection closed.")
        print("Next: python run_phase_3_ingest.py  (orchestratore completo)")


if __name__ == "__main__":
    main()
