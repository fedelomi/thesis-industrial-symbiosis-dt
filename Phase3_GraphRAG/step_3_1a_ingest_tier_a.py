
# Decisions active: D3 — Tier A normative ingest (EU EED, ASHRAE).

from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from neo4j import GraphDatabase, Session, exceptions as neo4j_exc

# --- Configuration ---
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

# --- Population Functions ---

def populate_countries(session: Session):
    print("  > Populating Countries...")
    countries = [
        {"iso": "IT", "name": "Italy", "dh_penetration_pct": 5.0, "eed_transposed": True},
        {"iso": "DK", "name": "Denmark", "dh_penetration_pct": 65.0, "eed_transposed": True},
        {"iso": "EU", "name": "European Union", "dh_penetration_pct": 12.0, "eed_transposed": True}
    ]
    query = """
    UNWIND $data AS row
    MERGE (c:Country {iso: row.iso})
    SET c += row
    """
    session.execute_write(lambda tx: tx.run(query, data=countries))

def populate_regulations(session: Session):
    print("  > Populating Regulations (Tier A)...")
    regs = [
        {"id": "EED-2023-1791", "name": "Energy Efficiency Directive 2023/1791", "short_name": "EED 2023", "jurisdiction": "EU", "type": "Directive", "year": 2023, "in_force": True},
        {"id": "DEL-REG-2024-1364", "name": "Delegated Regulation (EU) 2024/1364", "short_name": "Del.Reg 2024", "jurisdiction": "EU", "type": "Regulation", "year": 2024, "in_force": True}
    ]
    query = """
    UNWIND $data AS row
    MERGE (r:Regulation {id: row.id})
    SET r += row
    WITH r, row
    MATCH (c:Country {iso: row.jurisdiction})
    MERGE (c)-[:IMPLEMENTS]->(r)
    """
    session.execute_write(lambda tx: tx.run(query, data=regs))

def populate_regulatory_articles(session: Session):
    print("  > Populating Regulatory Articles...")
    articles = [
        {"id": "EED-ART-26", "regulation_id": "EED-2023-1791", "article_number": "26", "title": "Data Centers", "obligation_type": "mandatory", "threshold_value": 1.0, "threshold_unit": "MW", "summary": "DC with IT power >= 1MW must disclose waste heat capacity annually"},
        {"id": "EED-ART-24", "regulation_id": "EED-2023-1791", "article_number": "24", "title": "Heating and Cooling", "obligation_type": "mandatory", "summary": "Efficient district heating and cooling criteria"},
        {"id": "EED-ART-23", "regulation_id": "EED-2023-1791", "article_number": "23", "title": "Comprehensive Assessment", "obligation_type": "mandatory", "summary": "Mandatory assessment of DH every 5 years"},
        {"id": "DEL-REG-ART-2", "regulation_id": "DEL-REG-2024-1364", "article_number": "2", "title": "Definitions and KPIs", "obligation_type": "mandatory", "summary": "Defines KPI TWH (T_CDU_return), ERF, EREUSE"}
    ]
    query = """
    UNWIND $data AS row
    MERGE (a:RegulatoryArticle {id: row.id})
    SET a += row
    WITH a, row
    MATCH (r:Regulation {id: row.regulation_id})
    MERGE (r)-[:CONTAINS]->(a)
    """
    session.execute_write(lambda tx: tx.run(query, data=articles))

def populate_standards(session: Session):
    print("  > Populating Standards...")
    standards = [
        {"id": "ASHRAE-90.4", "name": "Energy Standard for Data Centers", "issuer": "ASHRAE", "year": 2022, "scope": "data-center-energy-efficiency"},
        {"id": "ISO-23247", "name": "Digital Twin manufacturing framework", "issuer": "ISO", "year": 2021, "scope": "digital-twin-manufacturing"}
    ]
    query = "UNWIND $data AS row MERGE (s:Standard {id: row.id}) SET s += row"
    session.execute_write(lambda tx: tx.run(query, data=standards))

def populate_temperature_bands(session: Session):
    print("  > Populating Temperature Bands...")
    bands = [
        {"id": "T1", "label": "Low Grade", "range_min_c": 40.0, "range_max_c": 60.0},
        {"id": "T2", "label": "Medium Grade", "range_min_c": 60.0, "range_max_c": 90.0},
        {"id": "T3", "label": "High Grade", "range_min_c": 90.0, "range_max_c": 130.0}
    ]
    query = "UNWIND $data AS row MERGE (tb:TemperatureBand {id: row.id}) SET tb += row"
    session.execute_write(lambda tx: tx.run(query, data=bands))

def populate_heatsources(session: Session):
    print("  > Populating HeatSources (linked to T1)...")
    hs_data = [
        {"id": "HS-S", "origin": "dc-liquid-cooling", "temp_supply_c": 47.6, "profile": "baseload"},
        {"id": "HS-M", "origin": "dc-liquid-cooling", "temp_supply_c": 47.6, "profile": "baseload"},
        {"id": "HS-L", "origin": "dc-liquid-cooling", "temp_supply_c": 47.6, "profile": "baseload"}
    ]
    query = """
    UNWIND $data AS row
    MERGE (h:HeatSource {id: row.id})
    SET h += row
    WITH h
    MATCH (tb:TemperatureBand {id: 'T1'})
    MERGE (h)-[:IN_BAND]->(tb)
    """
    session.execute_write(lambda tx: tx.run(query, data=hs_data))

def populate_datacenter_nodes(session: Session):
    print("  > Populating Data Center archetypes...")
    dcs = [
        {"id": "DC-S", "scale": "Edge", "it_capacity_kw": 500, "cooling_type": "liquid", "pue_nominal": 1.1, "waste_heat_kw": 590},
        {"id": "DC-M", "scale": "Mid-size", "it_capacity_kw": 3200, "cooling_type": "liquid", "pue_nominal": 1.15, "waste_heat_kw": 3840},
        {"id": "DC-L", "scale": "Hyperscale", "it_capacity_kw": 25000, "cooling_type": "liquid", "pue_nominal": 1.12, "waste_heat_kw": 30000}
    ]
    query = "UNWIND $data AS row MERGE (d:DataCenter {id: row.id}) SET d += row"
    session.execute_write(lambda tx: tx.run(query, data=dcs))

def populate_industrial_sectors(session: Session):
    print("  > Populating Industrial Sectors...")
    sectors = [
        {"id": "food-drink-milk", "name": "Food, Drink and Milk", "bref_ref": "FDM_BREF"},
        {"id": "paper-pulp", "name": "Pulp and Paper Industry", "bref_ref": "PP_BREF"},
        {"id": "rubber-chemical", "name": "Rubber and Chemical Industry", "bref_ref": "CWW_BREF"}
    ]
    query = "UNWIND $data AS row MERGE (i:IndustrialSector {id: row.id}) SET i += row"
    session.execute_write(lambda tx: tx.run(query, data=sectors))

def populate_manufacturing_processes(session: Session):
    print("  > Populating 9 Manufacturing Processes...")
    processes = [
        # LowT (~60°C)
        {"id": "PROC-PASTEUR", "name": "Pasteurization", "sector": "food-drink-milk", "temp_required_c": 60.0},
        {"id": "PROC-DAIRY-WASH", "name": "Dairy Washing", "sector": "food-drink-milk", "temp_required_c": 60.0},
        {"id": "PROC-FOOD-DRY", "name": "Food Drying", "sector": "food-drink-milk", "temp_required_c": 60.0},
        # MidT (~90°C)
        {"id": "PROC-PAPER-DRY", "name": "Paper Drying", "sector": "paper-pulp", "temp_required_c": 90.0},
        {"id": "PROC-PULP-COOK", "name": "Pulp Cooking", "sector": "paper-pulp", "temp_required_c": 90.0},
        {"id": "PROC-CIP-CLEAN", "name": "CIP Cleaning", "sector": "food-drink-milk", "temp_required_c": 90.0},
        # HighT (~130°C)
        {"id": "PROC-STERIL", "name": "Sterilization", "sector": "food-drink-milk", "temp_required_c": 130.0},
        {"id": "PROC-RUBBER-VULC", "name": "Rubber Vulcanization", "sector": "rubber-chemical", "temp_required_c": 130.0},
        {"id": "PROC-CHEM-STEAM", "name": "Chemical Steam Generation", "sector": "rubber-chemical", "temp_required_c": 130.0}
    ]
    query = """
    UNWIND $data AS row
    MERGE (p:ManufacturingProcess {id: row.id})
    SET p += row
    WITH p, row
    MATCH (s:IndustrialSector {id: row.sector})
    MERGE (p)-[:PART_OF]->(s)
    """
    session.execute_write(lambda tx: tx.run(query, data=processes))

def populate_scenarios(session: Session):
    print("  > Populating Scenarios S1..S9...")
    # Mapping logic: 3 scales * 3 temp bands
    scales = [("DC-S", "Edge"), ("DC-M", "Mid-size"), ("DC-L", "Hyperscale")]
    bands = ["T1", "T2", "T3"]
    
    # Representative processes for each band to link S1..S9
    process_mapping = {"T1": "PROC-PASTEUR", "T2": "PROC-PAPER-DRY", "T3": "PROC-STERIL"}
    
    scenarios = []
    idx = 1
    for dc_id, scale in scales:
        for band in bands:
            sc_id = f"S{idx}"
            scenarios.append({
                "id": sc_id,
                "dc_id": dc_id,
                "dc_scale": scale,
                "temp_band": band,
                "label": f"Scenario {sc_id}: {scale} DC to {band} Process",
                "process_id": process_mapping[band]
            })
            idx += 1

    query = """
    UNWIND $data AS row
    MERGE (sc:Scenario {id: row.id})
    SET sc.dc_scale = row.dc_scale, sc.temp_band = row.temp_band, sc.label = row.label
    WITH sc, row
    MATCH (dc:DataCenter {id: row.dc_id})
    MATCH (p:ManufacturingProcess {id: row.process_id})
    MERGE (sc)-[:USES_DC]->(dc)
    MERGE (sc)-[:TARGETS_PROCESS]->(p)
    """
    session.execute_write(lambda tx: tx.run(query, data=scenarios))

def populate_documents(session: Session):
    print("  > Populating Document metadata...")
    docs = [
        {"slug": "eed-2023-1791-official", "filename": "EED_2023_1791.pdf", "tier": "A", "doc_type": "Regulation", "jurisdiction": "EU"},
        {"slug": "ashrae-90-4-2022-summary", "filename": "ASHRAE_90_4.pdf", "tier": "A", "doc_type": "Standard", "jurisdiction": "EU"}
    ]
    query = "UNWIND $data AS row MERGE (d:Document {slug: row.slug}) SET d += row"
    session.execute_write(lambda tx: tx.run(query, data=docs))

def print_summary(session: Session):
    print("\n--- Summary: Node Counts ---")
    query = "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label"
    data = session.execute_read(lambda tx: tx.run(query).data())
    for record in data:
        print(f"  {record['label']}: {record['count']}")

def main():
    print("="*60)
    print("  PHASE 3: Task 3.1a - Tier A Ingestion")
    print("="*60)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            populate_countries(session)
            populate_regulations(session)
            populate_regulatory_articles(session)
            populate_standards(session)
            populate_temperature_bands(session)
            populate_heatsources(session)
            populate_datacenter_nodes(session)
            populate_industrial_sectors(session)
            populate_manufacturing_processes(session)
            populate_scenarios(session)
            populate_documents(session)
            print_summary(session)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
    finally:
        driver.close()
        print("\n✓ Connection closed.")

if __name__ == "__main__":
    main()