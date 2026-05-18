"""
step_3_3_benchmark_qa_design.py
================================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Genera il dataset benchmark QA per la valutazione del Graph RAG IS.
Salva il dataset come JSON in data/benchmark_qa_dataset.json.

Struttura dataset (110 domande, 3 categorie + copertura ISO 50001):
  Cat A - factual-lookup    : 38 domande, 1-2 hop, ground truth clause-grounded
  Cat B - multi-hop-is      : 37 domande, 3-5 hop, reasoning su path IS completo
  Cat C - comparative       : 35 domande, 3-5 hop, confronto scenari/paesi/tecnologie
  (A35-A38, B34-B37, C34-C35 coprono ISO 50001:2018+A1:2024 ingestito in step_3_1e)

Ogni domanda ha:
  id, category, hop_count, nl_question,
  expected_cypher_template (riferimento a P1-P6),
  ground_truth, source_clause, difficulty (easy/medium/hard)

Uso:
    python step_3_3_benchmark_qa_design.py

Output:
    data/benchmark_qa_dataset.json
    data/benchmark_qa_dataset_summary.txt

Riferimento wiki: [[roadmap-fasi-1-2-3]] Passo 3.3,
                  [[graph-rag-entity-schema]] Multi-hop Query Patterns
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# --- Struttura domanda ---

@dataclass
class QAItem:
    id: str
    category: str            # "factual-lookup" | "multi-hop-is" | "comparative"
    hop_count: int
    difficulty: str          # "easy" | "medium" | "hard"
    nl_question: str
    ground_truth: str
    source_clause: str       # es. "EED-ART-26", "ASHRAE-90.4", "DK-DH-ACT-§11"
    cypher_template: str     # riferimento a P1-P6 o "custom"
    notes: str = ""


# --- Dataset ---
#
# Principi di costruzione:
# 1. Ogni domanda ha una risposta verificabile nel grafo Neo4j attuale
# 2. source_clause punta a un nodo RegulatoryArticle, Standard o PolicyFramework
# 3. Le domande "hard" richiedono join su 4+ label diverse
# 4. Le domande comparative forzano il LLM a aggregare su piu scenari

DATASET: list[QAItem] = [

    # =========================================================
    # CATEGORIA A — FACTUAL LOOKUP (34 domande, 1-2 hop)
    # =========================================================

    # A1-A8: EED 2023/1791
    QAItem("A01", "factual-lookup", 2, "easy",
        "What is the minimum IT power threshold for mandatory waste heat disclosure under EED Article 26?",
        "1.0 MW",
        "EED-ART-26",
        "P1_eed_art26_threshold"),

    QAItem("A02", "factual-lookup", 2, "easy",
        "Is EED Article 26 compliance mandatory or voluntary for data centers above 1 MW?",
        "mandatory",
        "EED-ART-26",
        "P1_eed_art26_threshold"),

    QAItem("A03", "factual-lookup", 2, "easy",
        "What does EED Article 23 require from member states regarding district heating?",
        "Mandatory comprehensive assessment of DH every 5 years",
        "EED-ART-23",
        "custom"),

    QAItem("A04", "factual-lookup", 2, "easy",
        "What year was the Energy Efficiency Directive 2023/1791 published?",
        "2023",
        "EED-2023-1791",
        "custom"),

    QAItem("A05", "factual-lookup", 2, "easy",
        "What KPIs does Delegated Regulation 2024/1364 define for data center waste heat?",
        "TWH (Temperature of Waste Heat), ERF (Energy Reuse Factor) and EREUSE",
        "DEL-REG-ART-2",
        "custom"),

    QAItem("A06", "factual-lookup", 2, "medium",
        "What is the obligation type of EED Article 24 regarding district heating criteria?",
        "mandatory",
        "EED-ART-24",
        "custom"),

    QAItem("A07", "factual-lookup", 2, "medium",
        "How many RegulatoryArticles are defined under EED 2023/1791 in the knowledge graph?",
        "3 (Art.23, Art.24, Art.26)",
        "EED-2023-1791",
        "custom"),

    QAItem("A08", "factual-lookup", 2, "medium",
        "Which EU regulation defines the KPI methodology for data center thermal waste heat reporting?",
        "Delegated Regulation EU 2024/1364",
        "DEL-REG-ART-2",
        "custom"),

    # A9-A14: Standards
    QAItem("A09", "factual-lookup", 1, "easy",
        "What is the scope of the ASHRAE 90.4 standard?",
        "data-center-energy-efficiency",
        "ASHRAE-90.4",
        "custom"),

    QAItem("A10", "factual-lookup", 1, "easy",
        "Which organization publishes the ASHRAE 90.4 standard?",
        "ASHRAE",
        "ASHRAE-90.4",
        "custom"),

    QAItem("A11", "factual-lookup", 1, "easy",
        "What does ISO 23247 cover?",
        "Digital twin manufacturing framework",
        "ISO-23247",
        "custom"),

    QAItem("A12", "factual-lookup", 1, "easy",
        "In which year was ISO 23247 published?",
        "2021",
        "ISO-23247",
        "custom"),

    # A13-A18: Temperature Bands
    QAItem("A13", "factual-lookup", 1, "easy",
        "What temperature range defines the T1 low-grade heat band?",
        "40 to 60 degrees Celsius",
        "TemperatureBand-T1",
        "custom"),

    QAItem("A14", "factual-lookup", 1, "easy",
        "What temperature range defines the T2 medium-grade heat band?",
        "60 to 90 degrees Celsius",
        "TemperatureBand-T2",
        "custom"),

    QAItem("A15", "factual-lookup", 1, "easy",
        "What temperature range defines the T3 high-grade heat band?",
        "90 to 130 degrees Celsius",
        "TemperatureBand-T3",
        "custom"),

    QAItem("A16", "factual-lookup", 1, "easy",
        "How many temperature bands are defined in the IS framework?",
        "3 (T1, T2, T3)",
        "TemperatureBand-T1",
        "custom"),

    # A17-A22: Data Centers
    QAItem("A17", "factual-lookup", 1, "easy",
        "What is the IT capacity of the Edge data center archetype in kilowatts?",
        "500 kW",
        "DataCenter-DC-S",
        "custom"),

    QAItem("A18", "factual-lookup", 1, "easy",
        "What cooling type is used by all three data center archetypes in the framework?",
        "liquid cooling",
        "DataCenter-DC-S",
        "custom"),

    QAItem("A19", "factual-lookup", 1, "easy",
        "What is the waste heat capacity of the hyperscale data center archetype?",
        "30000 kW",
        "DataCenter-DC-L",
        "custom"),

    QAItem("A20", "factual-lookup", 1, "medium",
        "What is the nominal PUE of the mid-size data center?",
        "1.15",
        "DataCenter-DC-M",
        "custom"),

    # A21-A26: Italian regulatory cluster
    QAItem("A21", "factual-lookup", 2, "easy",
        "What is the value of Italian white certificates (TEE/CB) in EUR per toe?",
        "300 EUR/toe",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    QAItem("A22", "factual-lookup", 2, "easy",
        "What is the EUR per MWh value of Italian TEE/CB incentives?",
        "25.8 EUR/MWh",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    QAItem("A23", "factual-lookup", 2, "easy",
        "Which technologies are eligible for Italian TEE/CB white certificates?",
        "waste-heat-recovery, heat-pump and absorption-chiller",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    QAItem("A24", "factual-lookup", 2, "medium",
        "Which Italian body administers the white certificate (TEE/CB) system?",
        "GSE - Gestore Servizi Energetici",
        "IT-TEE-CB",
        "custom"),

    QAItem("A25", "factual-lookup", 2, "medium",
        "How many Italian regulatory decrees on energy efficiency are in the knowledge graph?",
        "5 (DM 2021, DD May 2022, DD May 2023, DD Oct 2023, DM MASE 2025)",
        "IT-DM-2021-05-21",
        "custom"),

    QAItem("A26", "factual-lookup", 2, "medium",
        "Which Italian decree is considered the founding act of the TEE/CB system?",
        "Decreto Ministeriale 21 maggio 2021",
        "IT-DM-2021-05-21",
        "custom"),

    # A27-A34: Danish regulatory cluster
    QAItem("A27", "factual-lookup", 2, "easy",
        "What is the mandatory DH connection threshold under Danish Varmeforsyningsloven?",
        "1.0 MW",
        "DK-DH-ACT-CONNECTION-MANDATE",
        "P3_regulatory_screening_dk"),

    QAItem("A28", "factual-lookup", 2, "easy",
        "What is the supply temperature of the Danish 4th generation district heating network?",
        "60 degrees Celsius",
        "DK-4GDH-GENERIC",
        "custom"),

    QAItem("A29", "factual-lookup", 2, "easy",
        "What is the return temperature of the Danish 4GDH network?",
        "35 degrees Celsius",
        "DK-4GDH-GENERIC",
        "custom"),

    QAItem("A30", "factual-lookup", 2, "easy",
        "What is Denmark's GHG reduction target horizon year in the NECP 2024?",
        "2030",
        "DK-NECP-2024",
        "custom"),

    QAItem("A31", "factual-lookup", 2, "medium",
        "Which Danish agency enforces the Heat Supply Act obligations?",
        "Danish Energy Agency (DK-DEA)",
        "DK-NECP-2024",
        "custom"),

    QAItem("A32", "factual-lookup", 2, "medium",
        "What is the DH penetration rate in Denmark according to the knowledge graph?",
        "65%",
        "Country-DK",
        "custom"),

    QAItem("A33", "factual-lookup", 2, "medium",
        "What generation is the legacy Danish district heating network in the knowledge graph?",
        "3GDH",
        "DK-3GDH-GENERIC",
        "custom"),

    QAItem("A34", "factual-lookup", 2, "medium",
        "What is the supply temperature of the Danish 3GDH legacy network?",
        "80 degrees Celsius",
        "DK-3GDH-GENERIC",
        "custom"),


    # =========================================================
    # CATEGORIA B — MULTI-HOP IS (33 domande, 3-5 hop)
    # =========================================================

    # B1-B9: Scenario paths
    QAItem("B01", "multi-hop-is", 4, "medium",
        "What is the thermal compatibility status of Scenario S1 using direct heat exchange?",
        "REQUIRES_UPGRADE: supply 47.6C < pasteurization requirement 60C",
        "EED-ART-26",
        "P2_thermal_compatibility_S1"),

    QAItem("B02", "multi-hop-is", 4, "medium",
        "Which data center and process are involved in Scenario S5?",
        "DC-M (mid-size, 3200 kW) and Paper Drying process (90C required)",
        "DataCenter-DC-M",
        "P2_thermal_compatibility_all"),

    QAItem("B03", "multi-hop-is", 4, "medium",
        "What upgrade technology is associated with Scenario S9?",
        "CO2_HTHP with supply temperature 110C",
        "DataCenter-DC-L",
        "P2_thermal_compatibility_all"),

    QAItem("B04", "multi-hop-is", 4, "medium",
        "How many of the 9 IS scenarios use liquid cooling with direct heat exchange?",
        "3 scenarios (S1, S4, S7 - one per DC scale, all targeting T1 band)",
        "TemperatureBand-T1",
        "P2_thermal_compatibility_all"),

    QAItem("B05", "multi-hop-is", 4, "medium",
        "What is the available heat capacity in Scenario S7 (hyperscale DC, T1 band)?",
        "29500 kW from HS-L-direct_HX",
        "DataCenter-DC-L",
        "P5_scenario_comparison_L"),

    QAItem("B06", "multi-hop-is", 4, "hard",
        "For Scenario S3 (Edge DC, T3 band), what is the gap between supply temperature and process requirement?",
        "20C gap: CO2_HTHP supplies 110C, sterilization requires 130C",
        "TemperatureBand-T3",
        "P2_thermal_compatibility_all"),

    QAItem("B07", "multi-hop-is", 4, "medium",
        "What industrial sector does the target process in Scenario S2 belong to?",
        "Pulp and Paper Industry (Paper Drying process)",
        "DataCenter-DC-S",
        "P6_full_is_path"),

    QAItem("B08", "multi-hop-is", 4, "medium",
        "How many scenarios target processes in the Food, Drink and Milk sector?",
        "6 scenarios (S1, S3, S4, S6, S7, S9 - pasteurization and sterilization)",
        "IndustrialSector-food-drink-milk",
        "P6_full_is_path"),

    QAItem("B09", "multi-hop-is", 5, "hard",
        "Trace the full IS pathway for Scenario S8: from data center to industrial sector.",
        "DC-L (hyperscale) -> HS-L-HP (70C, 29500kW) -> T2 band -> Paper Drying (90C) -> Pulp and Paper Industry",
        "DataCenter-DC-L",
        "P6_full_is_path"),

    # B10-B18: Regulatory paths
    QAItem("B10", "multi-hop-is", 3, "medium",
        "Which data centers in the framework must comply with EED Article 26 disclosure?",
        "DC-M (3200 kW = 3.2 MW) and DC-L (25000 kW = 25 MW) - both above 1 MW threshold",
        "EED-ART-26",
        "P3_regulatory_screening_dk"),

    QAItem("B11", "multi-hop-is", 3, "medium",
        "Does the Edge data center (DC-S) fall under EED Article 26 mandatory disclosure?",
        "No - DC-S has 500 kW IT capacity, below the 1 MW threshold",
        "EED-ART-26",
        "custom"),

    QAItem("B12", "multi-hop-is", 3, "medium",
        "Which data centers must connect to DH under Danish Varmeforsyningsloven?",
        "DC-M (3.2 MW) and DC-L (25 MW) - both above 1 MW connection mandate",
        "DK-DH-ACT-CONNECTION-MANDATE",
        "P3_regulatory_screening_dk"),

    QAItem("B13", "multi-hop-is", 3, "medium",
        "Can Italian TEE/CB incentives be claimed for a heat pump upgrade in an IS project?",
        "Yes - heat-pump is in the eligible_tech list of IT-TEE-CB",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    QAItem("B14", "multi-hop-is", 3, "medium",
        "Is waste heat recovery directly eligible for Italian white certificates?",
        "Yes - waste-heat-recovery is in eligible_tech of IT-TEE-CB (300 EUR/toe)",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    QAItem("B15", "multi-hop-is", 3, "hard",
        "What is the financial value in EUR/MWh of recovering waste heat from a mid-size DC in Italy under TEE/CB?",
        "25.8 EUR/MWh from TEE/CB, applicable to DC-M waste heat recovery projects",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    QAItem("B16", "multi-hop-is", 4, "hard",
        "Which EU regulation mandates KPI reporting for a hyperscale DC and what specific KPIs are required?",
        "Del.Reg 2024/1364 Art.2 mandates TWH, ERF and EREUSE reporting for DC-L",
        "DEL-REG-ART-2",
        "custom"),

    QAItem("B17", "multi-hop-is", 3, "medium",
        "What policy framework governs the Danish 4GDH network in the knowledge graph?",
        "DK-DH-PLANNING-ACT (Varmeforsyningsloven) via GOVERNED_BY relationship",
        "DK-DH-PLANNING-ACT",
        "custom"),

    QAItem("B18", "multi-hop-is", 3, "medium",
        "How many mandatory regulatory obligations apply to a 5 MW data center operating in Denmark?",
        "2: EED Art.26 disclosure (EU) and DK DH connection mandate (national)",
        "EED-ART-26",
        "custom"),

    # B19-B27: HeatSource and thermal paths
    QAItem("B19", "multi-hop-is", 3, "medium",
        "What is the supply temperature of the heat source associated with Scenario S4?",
        "47.6C (HS-M-direct_HX, direct heat exchange from mid-size LC data center)",
        "DataCenter-DC-M",
        "custom"),

    QAItem("B20", "multi-hop-is", 3, "medium",
        "How many heat source nodes are linked to the hyperscale data center?",
        "3 (HS-L-direct_HX, HS-L-HP, HS-L-CO2_HTHP)",
        "DataCenter-DC-L",
        "custom"),

    QAItem("B21", "multi-hop-is", 4, "medium",
        "What temperature band is the heat source HS-M-HP linked to?",
        "T2 (60-90C medium-grade band) via IN_BAND relationship",
        "TemperatureBand-T2",
        "custom"),

    QAItem("B22", "multi-hop-is", 4, "hard",
        "For a mid-size LC data center with HP upgrade, which manufacturing processes are reachable?",
        "Processes in T1 and T2 bands: Pasteurization (60C) is reachable; Paper Drying (90C) is borderline GAP at 70C supply",
        "DataCenter-DC-M",
        "custom"),

    QAItem("B23", "multi-hop-is", 3, "medium",
        "What is the delta-T between HS-L-CO2_HTHP supply and sterilization process requirement?",
        "20C gap (110C supply vs 130C required)",
        "TemperatureBand-T3",
        "custom"),

    QAItem("B24", "multi-hop-is", 4, "hard",
        "Is a direct connection between DC-S and Danish 4GDH thermally feasible without upgrade?",
        "Partially feasible: DC-S direct_HX supply 47.6C is below 4GDH supply 60C but above return 35C - pre-heating role possible",
        "DK-4GDH-GENERIC",
        "custom"),

    QAItem("B25", "multi-hop-is", 4, "hard",
        "What upgrade technology would allow DC-M waste heat to reach T2 band compatibility?",
        "HP upgrade (HS-M-HP supply 70C) approaches but does not fully reach T2 upper bound 90C; borderline case",
        "TemperatureBand-T2",
        "custom"),

    QAItem("B26", "multi-hop-is", 3, "medium",
        "How many PRODUCES_HEAT relationships exist between DataCenter and HeatSource nodes?",
        "9 (3 DC x 3 upgrade technologies)",
        "DataCenter-DC-S",
        "custom"),

    QAItem("B27", "multi-hop-is", 4, "medium",
        "Which industrial sector has the highest temperature requirement in the framework?",
        "Food, Drink and Milk (sterilization at 130C) and Pulp and Paper (paper drying at 90C); sterilization is highest",
        "IndustrialSector-food-drink-milk",
        "custom"),

    # B28-B33: IS project context
    QAItem("B28", "multi-hop-is", 3, "medium",
        "How many IS scenarios involve the Pulp and Paper sector?",
        "3 scenarios (S2, S5, S8 - Paper Drying process across DC-S, DC-M, DC-L)",
        "IndustrialSector-paper-pulp",
        "P6_full_is_path"),

    QAItem("B29", "multi-hop-is", 3, "medium",
        "What is the profile type of all heat sources derived from liquid-cooled data centers?",
        "baseload - continuous availability throughout the year",
        "DataCenter-DC-S",
        "custom"),

    QAItem("B30", "multi-hop-is", 4, "hard",
        "For an Italian IS project using DC-M and a heat pump, what is the maximum incentive revenue at 25.8 EUR/MWh assuming 7000 operating hours?",
        "DC-M waste heat: 3840 kW x 7000h = 26880 MWh/yr x 25.8 EUR/MWh = approx 693,504 EUR/yr",
        "IT-TEE-CB",
        "custom"),

    QAItem("B31", "multi-hop-is", 4, "hard",
        "Which regulatory framework would a Danish hyperscale DC need to comply with for DH connection?",
        "DK-DH-PLANNING-ACT (Varmeforsyningsloven §11): DC-L at 25 MW >> 1 MW threshold, mandatory connection if economically feasible",
        "DK-DH-ACT-CONNECTION-MANDATE",
        "P3_regulatory_screening_dk"),

    QAItem("B32", "multi-hop-is", 3, "medium",
        "Does Denmark have a transposed EED according to the knowledge graph?",
        "Yes - Country DK has eed_transposed = True",
        "Country-DK",
        "custom"),

    QAItem("B33", "multi-hop-is", 3, "medium",
        "What is the DH penetration rate in Italy according to the knowledge graph?",
        "5% - significantly lower than Denmark at 65%",
        "Country-IT",
        "custom"),


    # =========================================================
    # CATEGORIA C — COMPARATIVE (33 domande, 3-5 hop)
    # =========================================================

    # C1-C9: Scenario comparisons
    QAItem("C01", "comparative", 4, "medium",
        "Compare the available heat capacity across the three hyperscale scenarios S7, S8 and S9.",
        "All three have 29500 kW available; difference is in supply temperature: 47.6C (S7), 70C (S8), 110C (S9)",
        "DataCenter-DC-L",
        "P5_scenario_comparison_L"),

    QAItem("C02", "comparative", 4, "medium",
        "Which scenario provides the highest supply temperature for industrial use?",
        "S3, S6, S9 (CO2_HTHP upgrade, T_supply=110C) - all three DC scales with T3 band",
        "TemperatureBand-T3",
        "P2_thermal_compatibility_all"),

    QAItem("C03", "comparative", 4, "medium",
        "Compare Edge (DC-S) and Hyperscale (DC-L) data centers in terms of recoverable heat capacity.",
        "DC-S: 590 kW waste heat / HS capacity 590 kW; DC-L: 30000 kW waste heat / HS capacity 29500 kW - ratio ~50x",
        "DataCenter-DC-S",
        "custom"),

    QAItem("C04", "comparative", 3, "easy",
        "Which data center scale is exempt from EED Article 26 mandatory disclosure?",
        "DC-S (Edge, 500 kW) - below 1 MW threshold; DC-M and DC-L must disclose",
        "EED-ART-26",
        "custom"),

    QAItem("C05", "comparative", 4, "medium",
        "Compare the thermal gap for direct heat exchange across the three temperature bands.",
        "T1 gap: 12.4C (47.6 vs 60C); T2 gap: 42.4C if direct_HX but 20C with HP (70 vs 90C); T3 gap: 20C with CO2_HTHP (110 vs 130C)",
        "TemperatureBand-T1",
        "P2_thermal_compatibility_all"),

    QAItem("C06", "comparative", 4, "hard",
        "Which upgrade technology minimizes the thermal gap for T2 band processes?",
        "HP upgrade (70C supply) gives 20C gap vs Paper Drying 90C; CO2_HTHP would be overcapacity at 110C",
        "TemperatureBand-T2",
        "custom"),

    QAItem("C07", "comparative", 4, "medium",
        "Compare the number of scenarios that target Food sector vs Paper sector processes.",
        "Food sector: 6 scenarios (pasteurization S1,S4,S7 + sterilization S3,S6,S9); Paper sector: 3 scenarios (S2,S5,S8)",
        "IndustrialSector-food-drink-milk",
        "P6_full_is_path"),

    QAItem("C08", "comparative", 5, "hard",
        "For each data center scale, which scenario offers the best thermal match considering upgrade cost?",
        "S1/S4/S7 (direct_HX, T1): lowest upgrade cost but 12.4C gap; S2/S5/S8 (HP, T2): moderate cost, 20C gap; S3/S6/S9 (CO2_HTHP, T3): highest cost, 20C gap",
        "DataCenter-DC-S",
        "P2_thermal_compatibility_all"),

    QAItem("C09", "comparative", 4, "medium",
        "How does the available heat capacity scale from Edge to Hyperscale with HP upgrade?",
        "HS-S-HP: 590 kW; HS-M-HP: 3776 kW; HS-L-HP: 29500 kW - roughly proportional to IT capacity",
        "DataCenter-DC-S",
        "custom"),

    # C10-C18: Country/regulatory comparisons
    QAItem("C10", "comparative", 3, "medium",
        "Compare DH penetration rates between Italy and Denmark in the knowledge graph.",
        "Italy: 5%, Denmark: 65% - Denmark has 13x higher DH penetration",
        "Country-IT",
        "custom"),

    QAItem("C11", "comparative", 3, "medium",
        "Which country has stronger mandatory IS obligations for data centers in the knowledge graph?",
        "Denmark: mandatory DH connection >1MW (Varmeforsyningsloven) plus NECP DC WHR target; Italy has incentive-based (TEE/CB) not mandatory connection",
        "DK-DH-ACT-CONNECTION-MANDATE",
        "custom"),

    QAItem("C12", "comparative", 3, "medium",
        "Compare the policy framework types for Italy vs Denmark in the knowledge graph.",
        "Italy: 5 ministerial/directorial decrees (incentive-based TEE/CB); Denmark: NECP + Heat Supply Act (obligation-based)",
        "Country-IT",
        "custom"),

    QAItem("C13", "comparative", 4, "hard",
        "Which jurisdiction offers better financial support for IS projects - Italy or Denmark based on knowledge graph?",
        "Italy: explicit TEE/CB at 25.8 EUR/MWh for WHR; Denmark: mandatory connection ensures revenue but no direct subsidy in graph - Italy has more explicit financial incentive",
        "IT-TEE-CB",
        "custom"),

    QAItem("C14", "comparative", 3, "medium",
        "Compare 3GDH and 4GDH supply temperatures and their compatibility with LC data center waste heat.",
        "3GDH supply 80C: DC LC direct_HX (47.6C) incompatible without HP; 4GDH supply 60C: DC LC compatible as pre-heat source",
        "DK-3GDH-GENERIC",
        "custom"),

    QAItem("C15", "comparative", 3, "medium",
        "How many regulatory articles define mandatory obligations vs voluntary in the knowledge graph?",
        "Mandatory: EED-ART-26, EED-ART-23, EED-ART-24, DK-DH-ACT-CONNECTION-MANDATE = 4; Voluntary: DK-NECP-DC-WHR-TARGET = 1",
        "EED-ART-26",
        "custom"),

    QAItem("C16", "comparative", 4, "hard",
        "Compare the regulatory burden for a 5 MW data center in Italy vs Denmark.",
        "Italy: EED Art.26 disclosure mandatory, TEE/CB available; Denmark: EED Art.26 + Varmeforsyningsloven mandatory DH connection + NECP target - Denmark has higher regulatory burden",
        "EED-ART-26",
        "custom"),

    QAItem("C17", "comparative", 3, "medium",
        "Which country in the knowledge graph has completed EED transposition?",
        "Both Italy and Denmark have eed_transposed = True; EU entry also marked True",
        "Country-IT",
        "custom"),

    QAItem("C18", "comparative", 4, "hard",
        "For an IS project in Denmark using a 3GDH network, what upgrade is needed vs a 4GDH network?",
        "3GDH (80C supply): DC LC WHR at 47.6C requires HP upgrade to 80C; 4GDH (60C supply): direct_HX compatible as low-temperature pre-heat source",
        "DK-3GDH-GENERIC",
        "custom"),

    # C19-C27: Technology comparisons
    QAItem("C19", "comparative", 3, "easy",
        "Compare the three upgrade technologies by supply temperature: direct_HX, HP and CO2_HTHP.",
        "direct_HX: 47.6C (no upgrade), HP: 70.0C (+22.4C), CO2_HTHP: 110.0C (+62.4C from base)",
        "TemperatureBand-T1",
        "P2_thermal_compatibility_all"),

    QAItem("C20", "comparative", 3, "easy",
        "Which upgrade technology achieves the highest supply temperature in the framework?",
        "CO2_HTHP at 110C supply temperature",
        "TemperatureBand-T3",
        "custom"),

    QAItem("C21", "comparative", 4, "medium",
        "Compare the temperature delta-T (supply minus return) across the three upgrade technologies.",
        "All three have 10C delta-T by design (direct_HX: 47.6/37.6; HP: 70/60; CO2_HTHP: 110/100)",
        "DataCenter-DC-S",
        "custom"),

    QAItem("C22", "comparative", 4, "medium",
        "Which upgrade technology is most appropriate for a rubber vulcanization process requiring 130C?",
        "CO2_HTHP (110C supply) is closest but still has 20C gap; no current technology in graph achieves direct compatibility",
        "TemperatureBand-T3",
        "custom"),

    QAItem("C23", "comparative", 4, "medium",
        "Compare the IS scenarios for the mid-size DC across all three temperature bands.",
        "S4 (DC-M, T1, direct_HX, 47.6C, 3776kW); S5 (DC-M, T2, HP, 70C, 3776kW); S6 (DC-M, T3, CO2_HTHP, 110C, 3776kW)",
        "DataCenter-DC-M",
        "P5_scenario_comparison_L",
        notes="Analogo a P5 ma per DC-M"),

    QAItem("C24", "comparative", 4, "hard",
        "For which scenarios is absorption chiller a potentially eligible TEE/CB technology in Italy?",
        "Scenarios with sufficient WHR for absorption cooling (typically T2/T3): S2,S3,S5,S6,S8,S9 - absorption-chiller is in eligible_tech list",
        "IT-TEE-CB",
        "custom"),

    QAItem("C25", "comparative", 3, "medium",
        "Compare the number of industrial sectors targeted by T1 vs T2 vs T3 band scenarios.",
        "T1: Food (pasteurization); T2: Paper (paper drying); T3: Food (sterilization) - 2 sectors for T1+T3, 1 for T2",
        "IndustrialSector-food-drink-milk",
        "custom"),

    QAItem("C26", "comparative", 5, "hard",
        "Which combination of data center scale and upgrade technology maximizes both heat capacity and temperature for IS?",
        "DC-L + CO2_HTHP: 29500 kW at 110C (S9) - highest capacity and highest temperature in the framework",
        "DataCenter-DC-L",
        "P5_scenario_comparison_L"),

    QAItem("C27", "comparative", 4, "medium",
        "Compare TEE/CB incentive applicability across the 9 IS scenarios in Italy.",
        "All 9 scenarios qualify (waste-heat-recovery eligible); HP scenarios additionally qualify under heat-pump category; total revenue scales with DC capacity",
        "IT-TEE-CB",
        "P4_incentives_it_whr"),

    # C28-C33: Framework-level comparisons
    QAItem("C28", "comparative", 3, "medium",
        "How many nodes in total are of type Regulation vs RegulatoryArticle in the knowledge graph?",
        "Regulation: 7 nodes; RegulatoryArticle: 6 nodes",
        "EED-2023-1791",
        "custom"),

    QAItem("C29", "comparative", 3, "easy",
        "Compare the DH network capacity between Danish 3GDH and 4GDH in the knowledge graph.",
        "3GDH: 500 MW capacity; 4GDH: 300 MW capacity (newer, more efficient network)",
        "DK-3GDH-GENERIC",
        "custom"),

    QAItem("C30", "comparative", 4, "hard",
        "Which IS scenarios are directly eligible for Danish DH connection mandate without additional regulatory steps?",
        "Scenarios with DC-M (3.2 MW) and DC-L (25 MW): S4,S5,S6,S7,S8,S9 - all above 1 MW threshold",
        "DK-DH-ACT-CONNECTION-MANDATE",
        "P3_regulatory_screening_dk"),

    QAItem("C31", "comparative", 4, "hard",
        "Compare the transaction cost reduction potential for IT vs DK IS projects based on regulatory incentives.",
        "Italy: TEE/CB reduces financial risk (positive revenue 25.8 EUR/MWh); Denmark: mandatory connection reduces market access risk but no direct subsidy - different TC reduction mechanisms",
        "IT-TEE-CB",
        "custom"),

    QAItem("C32", "comparative", 3, "medium",
        "Which country has more regulatory instruments for IS in the knowledge graph?",
        "Italy: 5 decrees + 1 incentive + 1 actor = 7 nodes; Denmark: 2 frameworks + 2 articles + 2 networks + 1 actor = 7 nodes - equal coverage, different types",
        "Country-IT",
        "custom"),

    QAItem("C33", "comparative", 5, "hard",
        "For a new IS project developer, which scenario offers the best combination of thermal feasibility, regulatory support and financial incentive?",
        "S4 or S7 (DC-M/DC-L, T1, direct_HX) in Italy: lowest upgrade cost (no HP), TEE/CB eligible at 25.8 EUR/MWh, EED Art.26 compliance via disclosure only; OR S5/S8 in Denmark (4GDH compatible with HP, mandatory connection provides stable off-taker)",
        "IT-TEE-CB",
        "custom"),


    # =========================================================
    # ISO 50001:2018+A1:2024 — CATEGORIA A (factual-lookup)
    # =========================================================

    QAItem("A35", "factual-lookup", 1, "easy",
        "What does ISO 50001 Section 6.3 require organizations to do regarding waste heat recovery?",
        "Organizations shall identify and prioritize energy performance improvement opportunities; "
        "Annex A.6.3 explicitly states they should consider the extent to which energy is recoverable",
        "ISO50001-6-3",
        "ISO50001_ARTICLES"),

    QAItem("A36", "factual-lookup", 1, "easy",
        "What amendment does the ISO 50001:2018 standard used in the knowledge graph include?",
        "A1:2024 — adds climate change as a mandatory contextual issue under Section 4.1",
        "ISO50001-4-1",
        "ISO50001_ARTICLES"),

    QAItem("A37", "factual-lookup", 1, "medium",
        "What is the obligation type of all ISO 50001 clauses in the knowledge graph?",
        "mandatory — all 5 clauses (4.1, 6.3, 6.4, 9.1.1, 9.1.2) are mandatory requirements",
        "ISO50001-4-1",
        "ISO50001_ARTICLES"),

    QAItem("A38", "factual-lookup", 2, "medium",
        "Which ISO 50001 clause aligns with the KPIs defined in Delegated Regulation 2024/1364?",
        "Section 6.4 (Energy Performance Indicators / EnPI) aligns with TWH, ERF and EREUSE KPIs from Del.Reg 2024/1364",
        "ISO50001-6-4",
        "ISO50001_ARTICLES"),


    # =========================================================
    # ISO 50001:2018+A1:2024 — CATEGORIA B (multi-hop-is)
    # =========================================================

    QAItem("B34", "multi-hop-is", 3, "medium",
        "How many manufacturing processes in the knowledge graph must comply with ISO 50001?",
        "All 9 ManufacturingProcess nodes have a REQUIRES_COMPLIANCE relationship to ISO-50001-2018",
        "ISO50001-6-3",
        "ISO50001_PROCESS_COMPLIANCE"),

    QAItem("B35", "multi-hop-is", 3, "medium",
        "Under ISO 50001 Section 6.3, must a pasteurization plant document waste heat recovery as an energy improvement opportunity?",
        "Yes — PROC-PASTEUR has REQUIRES_COMPLIANCE to ISO-50001-2018; §6.3 mandates identifying WHR as improvement opportunity per Annex A.6.3",
        "ISO50001-6-3",
        "ISO50001_PROCESS_COMPLIANCE"),

    QAItem("B36", "multi-hop-is", 3, "hard",
        "What internal compliance loop does ISO 50001 Section 9.1.2 create in relation to EED Article 26?",
        "ISO 50001 §9.1.2 requires periodic evaluation of compliance with legal energy requirements; "
        "for DC operators subject to EED Art.26, this creates an internal verification cycle aligned with the external reporting obligation",
        "ISO50001-9-1-2",
        "ISO50001_ARTICLES"),

    QAItem("B37", "multi-hop-is", 4, "hard",
        "For a manufacturing plant certified ISO 50001 receiving waste heat from DC-M via HP upgrade, which ISO clause most directly mandates monitoring the WHR flow?",
        "Section 9.1.1: WHR input qualifies as a Significant Energy Use (SEU) and must be monitored at planned intervals; "
        "Section 6.3 mandates its identification as improvement opportunity",
        "ISO50001-9-1",
        "ISO50001_PROCESS_COMPLIANCE"),


    # =========================================================
    # ISO 50001:2018+A1:2024 — CATEGORIA C (comparative)
    # =========================================================

    QAItem("C34", "comparative", 3, "medium",
        "Compare the scope of ISO 50001 with ASHRAE 90.4 in the knowledge graph.",
        "ISO 50001 scope: energy-management-system (applies to any organization, voluntary but widely required); "
        "ASHRAE 90.4 scope: data-center-energy-efficiency (sector-specific, prescriptive). "
        "ISO 50001 is broader and process-oriented; ASHRAE 90.4 is technical and DC-specific",
        "ISO50001-6-3",
        "GENERIC_STANDARD"),

    QAItem("C35", "comparative", 4, "hard",
        "For a DC operator subject to both EED Article 26 and ISO 50001, how do the two frameworks complement each other for WHR?",
        "EED Art.26 creates external mandatory disclosure obligation (threshold 1 MW); "
        "ISO 50001 §6.3 creates internal obligation to identify and prioritize WHR as improvement opportunity; "
        "ISO 50001 §9.1.2 creates internal compliance verification loop for EED; "
        "together they form a push-pull mechanism: external regulation + internal management system",
        "ISO50001-9-1-2",
        "custom"),
]


# --- Output functions ---

def save_dataset(dataset: list[QAItem], output_dir: str = "data") -> tuple[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_path / "benchmark_qa_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in dataset], f, indent=2, ensure_ascii=False)

    # Summary
    summary_path = output_path / "benchmark_qa_dataset_summary.txt"
    cat_counts = {}
    diff_counts = {}
    hop_dist: dict[int, int] = {}
    for q in dataset:
        cat_counts[q.category] = cat_counts.get(q.category, 0) + 1
        diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1
        hop_dist[q.hop_count] = hop_dist.get(q.hop_count, 0) + 1

    lines = [
        "Benchmark QA Dataset — Phase 3 Graph RAG IS",
        f"Total questions : {len(dataset)}",
        "",
        "By category:",
    ]
    for cat, n in sorted(cat_counts.items()):
        lines.append(f"  {cat:<25} {n}")
    lines += ["", "By difficulty:"]
    for diff, n in sorted(diff_counts.items()):
        lines.append(f"  {diff:<10} {n}")
    lines += ["", "By hop count:"]
    for hop, n in sorted(hop_dist.items()):
        lines.append(f"  {hop}-hop       {n}")
    lines += ["", "Source clauses covered:"]
    clauses = sorted(set(q.source_clause for q in dataset))
    for c in clauses:
        lines.append(f"  {c}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(json_path), str(summary_path)


def print_summary(dataset: list[QAItem]) -> None:
    cat_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}
    for q in dataset:
        cat_counts[q.category] = cat_counts.get(q.category, 0) + 1
        diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1

    print("\n-- Dataset summary -------------------------------------------")
    print(f"  Total questions : {len(dataset)}")
    print("\n  By category:")
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat:<25} {n}")
    print("\n  By difficulty:")
    for diff, n in sorted(diff_counts.items()):
        print(f"    {diff:<10} {n}")


# --- Entry point ---

def main() -> None:
    print("=" * 60)
    print("  Phase 3 - step_3_3_benchmark_qa_design.py")
    print("  Benchmark QA dataset generation")
    print("=" * 60)

    print_summary(DATASET)

    json_path, summary_path = save_dataset(DATASET)
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {summary_path}")
    print("\nNext: step_3_4_evaluation.py  (RAGAs evaluation pipeline)")


if __name__ == "__main__":
    main()
