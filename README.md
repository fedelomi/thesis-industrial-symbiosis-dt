# Industrial Symbiosis via Digital Twins and Agentic AI
### MSc Thesis — Data Center Waste Heat Recovery for Manufacturing Processes

> **Politecnico di Torino — Laurea Magistrale in Ingegneria della Produzione Industriale e dell'Innovazione Tecnologica**
> A.A. 2025-2026 · Author: Federico Lomi
> Supervisors: Prof. Alessandro Simeone (Loughborough University), Prof. Li Yi, Prof. Gongzhuang Peng

---

## Overview

This repository contains the implementation code for a three-layer framework that enables **data center waste heat recovery** through industrial symbiosis (IS), modelled with digital twin (DT) technology and coordinated by agentic AI.

The framework addresses the research question: *Can an agentic AI system, grounded in a regulatory knowledge graph, autonomously negotiate and validate waste heat reuse agreements between data centers and industrial partners, replacing today's costly manual IS brokerage?*

The three layers map directly onto the three implementation phases in this repo:

| Layer (thesis) | Implementation phase | Role | Tech stack |
|---|---|---|---|
| **Physical-DT** (Strato 1) | `Phase1_PhysicalDT/` | Thermal + workload simulation of the DC | RC model, CoolProp, SustainGym |
| **IS-Match Score** (OS2) | `Phase2_ISMatch/` | Multi-criteria scoring DC ↔ manufacturing pair | numpy, pandas |
| **Institutional-LLM** (Strato 2) | `Phase3_GraphRAG/` | Regulatory KG + Graph RAG retrieval + RAGAS eval | Neo4j, LangChain, RAGAS |
| Agentic Negotiation (Strato 3 — future Phase 4) | not in this repo yet | Multi-agent IS brokerage with RL policy | Stable-Baselines3, Gymnasium |

The implementation covers **9 scenarios** (3 DC scales × 3 manufacturing temperature bands) and is scoped as a proof-of-concept for the magistrale thesis.

---

## Repository structure (post-reorg, audit 2026-05-05)

```
Github/
├── README.md                       — this file
├── .gitignore
├── common/
│   ├── __init__.py
│   └── dc_id_mapping.py            — single source of truth for DC identifiers
│                                     across phases (Edge_LC↔DC-S etc.)
│
├── Phase1_PhysicalDT/              — Physical Digital Twin (Layer 1, Strato 1)
│   ├── airside/
│   │   ├── run_phase_1.py          — pipeline orchestrator (1.1 → 1.4b)
│   │   ├── step_1_1_2_rc_model.py
│   │   ├── step_1_3_synthetic.py
│   │   ├── step_1_4_validation.py
│   │   ├── step_1_4b_sensitivity.py
│   │   ├── examples/, sustaingym/  — local SustainGym fork
│   │   └── results/                — generated CSVs + phase1_results.xlsx
│   └── lc/                         — liquid-cooling variant (D5, primary branch)
│       ├── run_phase_1_lc.py       — orchestrator (steps 1.1 → 1.4d + step_0)
│       ├── step_0_eed_compliance.py
│       ├── step_1_1_2_rc_model_lc.py
│       ├── step_1_3_synthetic_lc.py
│       ├── step_1_4_validation_lc.py
│       ├── step_1_4b_sensitivity_lc.py
│       ├── step_1_4c_erf_sensitivity.py
│       ├── step_1_4d_benchmark_comparison.py
│       ├── data/                   — committed reference inputs (LC profiles)
│       └── results/                — generated CSVs + phase1_lc_results.xlsx
│
├── Phase2_ISMatch/                 — IS-Match Score engine (LC-only — D6)
│   ├── run_phase_2_lc.py           — orchestrator (2.0 → 2.4 + Regulatory_KPIs)
│   ├── step_2_0_epsilon_filter_lc.py
│   ├── step_2_1_is_match_score_lc.py
│   ├── step_2_2_dataset_builder_lc.py
│   ├── step_2_3_ranking_validation_lc.py
│   ├── step_2_4_delta_tc_calibration_lc.py
│   ├── data/                       — Hotmaps EU industrial DB + CE-HEAT snapshot
│   └── results/                    — generated CSVs + phase2_lc_results.xlsx
│
└── Phase3_GraphRAG/                — Graph RAG IS (Layer 2, Strato 2)
    ├── run_phase_3.py              — full orchestrator (3.0 → 3.6)
    ├── requirements.txt            — pinned dependencies (Neo4j, LangChain, RAGAS)
    ├── .env.example                — NEO4J_PASSWORD, OPENAI_API_KEY template
    ├── step_3_0_neo4j_schema.py
    ├── step_3_1a_ingest_tier_a.py
    ├── step_3_1b_ingest_tier_b_it.py
    ├── step_3_1c_ingest_tier_b_dk.py
    ├── step_3_1d_ingest_scenarios_heatsources.py
    ├── step_3_1e_ingest_iso50001.py
    ├── step_3_2_graph_rag_pipeline.py
    ├── step_3_3_benchmark_qa_design.py
    ├── step_3_4_evaluation.py
    ├── step_3_4_bis_neuro_symbolic.py
    ├── step_3_5_phase1_integration.py
    ├── step_3_6_privacy_gate.py
    └── data/                       — JSON outputs (benchmark, eval, gate, stats)
```

> **Reorg notes (audit 2026-05-05).** Phase folders were renamed from `Phase1` / `Phase2` / `Phase 3` to `Phase1_PhysicalDT` / `Phase2_ISMatch` / `Phase3_GraphRAG` for clarity. Each phase now uses `data/` for committed inputs and `results/` for generated outputs. Scripts auto-create `results/` and write into it; orchestrators look there first. See the `# Decision active:` header at the top of every script for the implementation decision (D1–D6) it implements.

---

## Phase 1 — Physical Digital Twin

**Goal.** Build a privacy-preserving digital twin of the data center that exposes heat availability as a signal for IS matching.

**DC scales modelled** (Phase 1 wiki: [[roadmap-fasi-1-2-3]] FASE 1):

| Scenario | Airside (`airside/`) | Liquid Cooling (`lc/`) |
|---|---|---|
| Edge | 100 kW IT | 500 kW IT |
| Mid  | 5 MW IT  | 3.2 MW IT (anchored on Frontier LC-Opt) |
| Hyperscale | 100 MW IT | 25 MW IT |

**Steps (both branches).** RC thermal model with exact analytical solution; per-step exergy via vectorised CoolProp; SIDED-style block bootstrap with σ = 0.05 × IQR; Wasserstein W₁ + adapted NDE gate; sensitivity over σ ∈ {0.025, 0.05, 0.10, 0.25}.

**LC-only post-processing.** `step_0_eed_compliance.py` (EED Art. 12 / Art. 26 + TEE estimate), `step_1_4c_erf_sensitivity.py` (capture-fraction sweep, ERF = cf²), `step_1_4d_benchmark_comparison.py` (TWH vs. real LC DC cases).

**Active decisions.** D1 — DT calibrated on KPIs published in C1 (Frontier ORNL data not available). D5 — LC parameters from grey-box identification on the LC-Opt FMU (HPE/ORNL, NeurIPS 2025).

**Run.**
```bash
cd Phase1_PhysicalDT/airside && python run_phase_1.py            # airside
cd Phase1_PhysicalDT/lc      && python run_phase_1_lc.py         # LC (primary, D6)
```

---

## Phase 2 — IS-Match Score (LC-only, D6)

**Goal.** Compute a composite multi-criteria IS-Match Score for each (DC × manufacturing plant) pair and validate the ranking against a static RI baseline.

**Pipeline.** ε-parameter pre-screening gate → sector-parametric dataset builder (D2) → IS-Match Score (β·RI_temporal + γ·Exergy_DT_norm − δ·ΔTC_norm) → NDCG@9 + precision@3 ranking validation → ΔTC calibration loop (Phase 3 placeholder until step 3.5-bis fills the actual reduction factors).

**Active decisions.** D2 — geography-agnostic, sector-parametric dataset (no real sites; 9 plants × 3 proximity bands; ENEA / Hotmaps statistics / CE-HEAT taxonomy). D6 — Phase 2 is LC-only because the airside IS-Match Scores (max 0.28) are uniformly *not_viable* for all manufacturing tiers; the LC scores reach *marginal* (0.36–0.53) and are the meaningful workload for downstream phases.

**Run.**
```bash
cd Phase2_ISMatch && python run_phase_2_lc.py
```

---

## Phase 3 — Graph RAG IS

**Goal.** Encode the regulatory and institutional context (EU EED Art. 14 / Art. 26, ISO 50001, ASHRAE 90.4, IT TEE/CB, Danish NECP / DH networks / DEA) as a Neo4j knowledge graph; query it via a Graph RAG pipeline; evaluate with RAGAS metrics; check neuro-symbolic consistency; integrate Phase 1 LC stats as a context layer; verify a privacy gate on real-vs-synthetic profiles.

**Pipeline (12 steps, orchestrated by `run_phase_3.py`).**

| # | Script | Purpose |
|---|---|---|
| 0 | `step_3_0_neo4j_schema.py` | Constraints + indexes |
| 1 | `step_3_1a_ingest_tier_a.py` | EED, ASHRAE, TemperatureBand, DC, Scenarios |
| 2 | `step_3_1b_ingest_tier_b_it.py` | TEE / CB decrees, GSE |
| 3 | `step_3_1c_ingest_tier_b_dk.py` | NECP, DH networks, DEA |
| 4 | `step_3_1d_ingest_scenarios_heatsources.py` | HeatSource × upgrade tech |
| 5 | `step_3_1e_ingest_iso50001.py` | ISO 50001 integral text (D4) |
| 6 | `step_3_2_graph_rag_pipeline.py` | Retrieval (Cypher + LLM generation) |
| 7 | `step_3_3_benchmark_qa_design.py` | Document-grounded QA dataset (D3) |
| 8 | `step_3_4_evaluation.py` | RAGAS faithfulness / answer relevancy / context precision |
| 9 | `step_3_4_bis_neuro_symbolic.py` | Logical consistency vs. KG triples |
| 10 | `step_3_5_phase1_integration.py` | Phase 1 LC stats into KG + ΔTC mapping (3.5-bis) |
| 11 | `step_3_6_privacy_gate.py` | Privacy preservation gate (Δ IS-Match proxy < 5%) |

**Active decisions.** D3 — document-grounded QA protocol (no inter-rater Cohen's κ; ground truth = clause text). D4 — ISO 50001 integral text obtained via Prof. Simeone.

**Setup.**
```bash
cd Phase3_GraphRAG
python -m venv venv && source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                # then fill NEO4J_PASSWORD + OPENAI_API_KEY
```

**Run.**
```bash
python run_phase_3.py                  # full pipeline (0 → 11)
python run_phase_3.py --from-step 6    # resume from a specific step
python run_phase_3.py --ingest-only    # stop after step 5 (3.1e ISO 50001)
python run_phase_3.py --skip-rag       # skip RAG/QA/eval (steps 6–9)
python run_phase_3.py --verify-only    # only verify Neo4j node/relationship counts
```

---

## Active implementation decisions (D1–D6)

These are the methodological decisions driving the code. Each is documented in detail in the Obsidian wiki ([[decisioni-implementative]]) and re-stated in the docstring header of every script that implements it.

| ID | Topic | Phase | Status | Thesis chapter |
|---|---|---|---|---|
| D1 | DT calibration on published KPIs (C1) instead of proprietary Frontier ORNL data | Phase 1 (airside) | Workaround | Cap. 4.2 |
| D2 | Geography-agnostic, sector-parametric IS dataset | Phase 2 step 2.2 | Confirmed | Cap. 4.5, 6.2 |
| D3 | Document-grounded benchmark QA (no inter-rater Cohen's κ) | Phase 3 step 3.3 | Confirmed | Cap. 4.4, 6.2 |
| D4 | ISO 50001 integral text available (Prof. Simeone) | Phase 3 step 3.1e | Resolved | n/a |
| D5 | Grey-box LC parametrization from LC-Opt FMU (HPE/ORNL) | Phase 1 (LC) | Confirmed | Cap. 4.2 |
| D6 | Phase 2 LC-only (evidence-based exclusion of airside) | Phase 2 entire pipeline | Confirmed | Cap. 1.4, 4.1, 5.2 |

---

## Methodology reference

The framework maps onto the thesis gap-analysis chain:

```
Gap → Research Question → Objective → Phase → Metric
```

| Phase | Objective | Primary metric |
|---|---|---|
| 1 | Privacy-preserving DT | W₁_norm ≤ 0.05, NDE ≤ 0.20 |
| 2 | IS feasibility scoring | NDCG@9, precision@3 |
| 3 | Regulatory RAG retrieval | RAGAS faithfulness ≥ 0.80, hallucination < 15% |
| 4 (future) | Multi-agent negotiation | Nash convergence, IS-Match lift > 0.60 |

Data sources: synthetic DC workloads (block bootstrap from RC model output), Hotmaps Industrial Database (manufacturing heat sinks, statistical reference only), EED / ISO 50001 / Italian and Danish regulatory documents (knowledge graph corpus).

---

## License

Code released for academic reproducibility. Cite as:

> F. Lomi, *Industrial Symbiosis via Digital Twins and Agentic AI for Data Center Waste Heat Recovery*, MSc Thesis, Politecnico di Torino, 2026. the LC scores reach *marginal* (0.36–0.53) and are the meaningful workload for downstream phases.

**Run.**
```bash
cd Phase2_ISMatch && python run_phase_2_lc.py
```

---

## Phase 3 — Graph RAG IS

**Goal.** Encode the regulatory and institutional context (EU EED Art. 14 / Art. 26, ISO 50001, ASHRAE 90.4, IT TEE/CB, Danish NECP / DH networks / DEA) as a Neo4j knowledge graph; query it via a Graph RAG pipeline; evaluate with RAGAS metrics; check neuro-symbolic consistency; integrate Phase 1 LC stats as a context layer; verify a privacy gate on real-vs-synthetic profiles.

**Pipeline (12 steps, orchestrated by `run_phase_3.py`).**

| # | Script | Purpose |
|---|---|---|
| 0 | `step_3_0_neo4j_schema.py` | Constraints + indexes |
| 1 | `step_3_1a_ingest_tier_a.py` | EED, ASHRAE, TemperatureBand, DC, Scenarios |
| 2 | `step_3_1b_ingest_tier_b_it.py` | TEE / CB decrees, GSE |
| 3 | `step_3_1c_ingest_tier_b_dk.py` | NECP, DH networks, DEA |
| 4 | `step_3_1d_ingest_scenarios_heatsources.py` | HeatSource × upgrade tech |
| 5 | `step_3_1e_ingest_iso50001.py` | ISO 50001 integral text (D4) |
| 6 | `step_3_2_graph_rag_pipeline.py` | Retrieval (Cypher + LLM generation) |
| 7 | `step_3_3_benchmark_qa_design.py` | Document-grounded QA dataset (D3) |
| 8 | `step_3_4_evaluation.py` | RAGAS faithfulness / answer relevancy / context precision |
| 9 | `step_3_4_bis_neuro_symbolic.py` | Logical consistency vs. KG triples |
| 10 | `step_3_5_phase1_integration.py` | Phase 1 LC stats into KG + ΔTC mapping (3.5-bis) |
| 11 | `step_3_6_privacy_gate.py` | Privacy preservation gate (Δ IS-Match proxy < 5%) |

**Active decisions.** D3 — document-grounded QA protocol (no inter-rater Cohen's κ; ground truth = clause text). D4 — ISO 50001 integral text obtained via Prof. Simeone.

**Setup.**
```bash
cd Phase3_GraphRAG
python -m venv venv && source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                # then fill NEO4J_PASSWORD + OPENAI_API_KEY
```

**Run.**
```bash
python run_phase_3.py                  # full pipeline (0 → 11)
python run_phase_3.py --from-step 6    # resume from a specific step
python run_phase_3.py --ingest-only    # stop after step 5 (3.1e ISO 50001)
python run_phase_3.py --skip-rag       # skip RAG/QA/eval (steps 6–9)
python run_phase_3.py --verify-only    # only verify Neo4j node/relationship counts
```

---

## Active implementation decisions (D1–D6)

These are the methodological decisions driving the code. Each is documented in detail in the Obsidian wiki ([[decisioni-implementative]]) and re-stated in the docstring header of every script that implements it.

| ID | Topic | Phase | Status | Thesis chapter |
|---|---|---|---|---|
| D1 | DT calibration on published KPIs (C1) instead of proprietary Frontier ORNL data | Phase 1 (airside) | Workaround | Cap. 4.2 |
| D2 | Geography-agnostic, sector-parametric IS dataset | Phase 2 step 2.2 | Confirmed | Cap. 4.5, 6.2 |
| D3 | Document-grounded benchmark QA (no inter-rater Cohen's κ) | Phase 3 step 3.3 | Confirmed | Cap. 4.4, 6.2 |
| D4 | ISO 50001 integral text available (Prof. Simeone) | Phase 3 step 3.1e | Resolved | n/a |
| D5 | Grey-box LC parametrization from LC-Opt FMU (HPE/ORNL) | Phase 1 (LC) | Confirmed | Cap. 4.2 |
| D6 | Phase 2 LC-only (evidence-based exclusion of airside) | Phase 2 entire pipeline | Confirmed | Cap. 1.4, 4.1, 5.2 |

---

## Methodology reference

The framework maps onto the thesis gap-analysis chain:

```
Gap → Research Question → Objective → Phase → Metric
```

| Phase | Objective | Primary metric |
|---|---|---|
| 1 | Privacy-preserving DT | W₁_norm ≤ 0.05, NDE ≤ 0.20 |
| 2 | IS feasibility scoring | NDCG@9, precision@3 |
| 3 | Regulatory RAG retrieval | RAGAS faithfulness ≥ 0.80, hallucination < 15% |
| 4 (future) | Multi-agent negotiation | Nash convergence, IS-Match lift > 0.60 |

Data sources: synthetic DC workloads (block bootstrap from RC model output), Hotmaps Industrial Database (manufacturing heat sinks, statistical reference only), EED / ISO 50001 / Italian and Danish regulatory documents (knowledge graph corpus).

---

## License

Code released for academic reproducibility. Cite as:

> F. Lomi, *Industrial Symbiosis via Digital Twins and Agentic AI for Data Center Waste Heat Recovery*, MSc Thesis, Politecnico di Torino, 2026.
