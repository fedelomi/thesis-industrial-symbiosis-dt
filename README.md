# Industrial Symbiosis via Digital Twins and Agentic AI
### Master's Thesis — Data Center Waste Heat Recovery

> **Politecnico di Milano / University of Florence** · MSc Energy Engineering · 2025-2026  
> Author: Federico Lomi

---

## Overview

This repository contains the implementation code for a three-layer framework that enables **data center waste heat recovery** through industrial symbiosis (IS), modelled with digital twin (DT) technology and coordinated by agentic AI.

The framework addresses the following research question: *Can an agentic AI system — grounded in a regulatory knowledge graph — autonomously negotiate and validate waste heat reuse agreements between data centers and industrial partners, replacing today's costly manual IS brokerage?*

The three layers are:

| Layer | Role | Technology |
|---|---|---|
| **Physical-DT** (Layer 1) | Thermal and workload simulation of the data center | RC model, CoolProp, SustainGym |
| **Institutional-LLM** (Layer 2) | Regulatory knowledge retrieval and IS feasibility scoring | Neo4j, Graph-RAG, LangChain |
| **Agentic Negotiation** (Layer 3) | Multi-agent IS brokerage with RL policy | Stable-Baselines3, Gymnasium |

The implementation covers **9 scenarios** (3 DC scales × 3 manufacturing temperature bands) and is scoped as a proof-of-concept for the magistrale thesis.

---

## Repository Structure

```
.
├── Phase1/                      # Layer 1 — Physical Digital Twin
│   ├── DatacenterGym/           # RC thermal model + synthetic workload generator
│   │   ├── run_phase_1.py       # Pipeline orchestrator (Steps 1.1 → 1.4b)
│   │   ├── step_1_1_2_rc_model.py
│   │   ├── step_1_3_synthetic.py
│   │   ├── step_1_4_validation.py
│   │   └── step_1_4b_sensitivity.py
│   └── LC-Opt/                  # Liquid-cooling variant + EED compliance
│       ├── run_phase_1_lc.py
│       ├── step_0_eed_compliance.py
│       └── step_1_4c_erf_sensitivity.py
│
├── Phase2/                      # IS-Match Scoring Engine
│   ├── run_phase_2_lc.py        # Pipeline orchestrator (Steps 2.0 → 2.4)
│   ├── step_2_0_epsilon_filter_lc.py
│   ├── step_2_1_is_match_score_lc.py
│   ├── step_2_2_dataset_builder_lc.py
│   ├── step_2_3_ranking_validation_lc.py
│   └── step_2_4_delta_tc_calibration_lc.py
│
├── Phase 3/                     # Layer 2 — Institutional LLM + Graph-RAG
│   ├── run_phase_3_ingest.py    # Pipeline orchestrator (Steps 3.0 → 3.1e)
│   ├── step_3_0_neo4j_schema.py
│   ├── step_3_1a_ingest_tier_a.py
│   ├── step_3_1b_ingest_tier_b_it.py
│   ├── step_3_1c_ingest_tier_b_dk.py
│   ├── step_3_1d_ingest_scenarios_heatsources.py
│   ├── step_3_1e_ingest_iso50001.py
│   ├── step_3_2_graph_rag_pipeline.py
│   ├── step_3_3_benchmark_qa_design.py
│   ├── step_3_4_evaluation.py
│   └── step_3_4_bis_neuro_symbolic.py
│
├── .gitignore
└── README.md
```

---

## Phase 1 — Physical Digital Twin

**Goal:** Build a privacy-preserving digital twin of the data center that exposes heat availability as a signal for IS matching.

**Steps:**
- **1.1/1.2** — RC thermal model with exergy analysis (3 DC scales: Edge 100 kW, Mid 1 MW, Hyperscale 10 MW)
- **1.3** — Privacy-preserving synthetic workload generator (Wasserstein distance ≤ 0.05 gate)
- **1.4** — Validation via W1 distance and Normalised Deviation Error (NDE)
- **1.4b** — Sensitivity analysis: privacy-fidelity trade-off
- **LC-Opt** — Liquid cooling variant with EED Art. 26 compliance check and ERF sensitivity

**Key outputs:** `datacenter_dt_results_annual.csv`, `synthetic_profile_annual.csv`, `phase1_results.xlsx`

**Run:**
```bash
cd Phase1/DatacenterGym
python run_phase_1.py
# or, to skip steps with existing outputs:
python run_phase_1.py --skip-existing
```

---

## Phase 2 — IS-Match Scoring

**Goal:** Compute a multi-criteria IS-Match Score between data center heat output and candidate industrial heat sinks, using the CE-HEAT / Hotmaps dataset.

**Steps:**
- **2.0** — ε-parameter pre-screening gate (temperature band compatibility)
- **2.2** — CE-HEAT-inspired dataset builder (Hotmaps industrial database)
- **2.1** — IS-Match Score: weighted combination of thermal, geographic and regulatory criteria
- **2.3** — Ranking validation via NDCG and Precision@3
- **2.4** — ΔTc calibration loop (approach temperature optimisation)

**Key outputs:** `step_2_1_is_match_scores_lc.csv`, `step_2_3_ranking_metrics_lc.csv`, `phase2_lc_results.xlsx`

**Run:**
```bash
cd Phase2
python run_phase_2_lc.py
```

---

## Phase 3 — Institutional Knowledge Graph + Graph-RAG

**Goal:** Encode the regulatory and institutional context (EED, ISO 50001, Italian TEE/CB incentives, Danish NECP and DH networks) as a Neo4j knowledge graph, queryable via a Graph-RAG pipeline evaluated with RAGAS metrics.

**Steps:**
- **3.0** — Neo4j schema (constraints and indexes)
- **3.1a** — Tier A ingest: EED Art. 26, ASHRAE standards, temperature bands, DC nodes, scenarios
- **3.1b/c** — Tier B ingest: Italian regulatory instruments (TEE, CB, GSE) and Danish instruments (NECP, DEA, DH networks)
- **3.1d** — HeatSource nodes with upgrade technologies and Scenario relationships
- **3.1e** — ISO 50001 energy management framework
- **3.2** — Graph-RAG pipeline (retrieval + LLM generation)
- **3.3** — Benchmark QA dataset design (domain-expert questions)
- **3.4** — RAGAS evaluation (faithfulness, answer relevancy, context precision)
- **3.4-bis** — Neuro-symbolic consistency check

**Key outputs:** Neo4j graph, `evaluation_results_graph-rag_*.json`, `benchmark_qa_dataset.json`

**Requirements:** Neo4j instance running at `bolt://127.0.0.1:7687`. Set credentials via environment variable before running:

```bash
export NEO4J_PASSWORD=your_password
cd "Phase 3"
python run_phase_3_ingest.py
# selective re-run from a specific step:
python run_phase_3_ingest.py --from-step 3
# verification only (no writes):
python run_phase_3_ingest.py --verify-only
```

---

## Setup

**Python version:** 3.10+

Install dependencies (per phase):

```bash
# Phase 1
pip install numpy pandas scipy openpyxl coolprop

# Phase 2
pip install numpy pandas scipy openpyxl

# Phase 3
pip install neo4j langchain langchain-openai langchain-community \
            sentence-transformers ragas openai python-dotenv
```

> **Note on credentials:** Phase 3 scripts require a Neo4j password and an OpenAI API key. Store them as environment variables (`NEO4J_PASSWORD`, `OPENAI_API_KEY`) or in a `.env` file — never commit credentials to the repository.

---

## Methodology Reference

The framework maps directly onto the thesis gap analysis chain:

```
Gap → Research Question → Objective → Phase → Metric
```

| Phase | Objective | Primary Metric |
|---|---|---|
| 1 | Privacy-preserving DT | W1 ≤ 0.05, NDE ≤ 10% |
| 2 | IS feasibility scoring | NDCG, Precision@3 |
| 3 | Regulatory RAG retrieval | RAGAS faithfulness ≥ 0.80 |

Data sources: Google Cluster Data (workloads), Hotmaps Industrial Database (heat sinks), EED / ISO 50001 / Italian and Danish regulatory documents (knowledge graph).

---

## License

Code released for academic reproducibility. Cite as:

> F. Lomi, *Industrial Symbiosis via Digital Twins and Agentic AI for Data Center Waste Heat Recovery*, MSc Thesis, 2026.
