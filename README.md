# Industrial Symbiosis via Digital Twins and Agentic AI
### MSc Thesis — Data Centre Waste Heat Recovery for Manufacturing Processes

> **Politecnico di Torino — Laurea Magistrale in Ingegneria della Produzione Industriale e dell'Innovazione Tecnologica**
> A.A. 2025-2026 · Author: Federico Lomi
> Supervisors: Prof. Alessandro Simeone (Loughborough University), Prof. Li Yi, Prof. Gongzhuang Peng

---

## Overview

This repository contains the implementation code for a three-layer framework that enables **data centre waste heat recovery** through industrial symbiosis (IS), modelled with digital twin (DT) technology and coordinated by agentic AI.

The framework addresses the research question: *Can an agentic AI system, grounded in a regulatory knowledge graph and a calibrated physical digital twin, autonomously characterise, match and negotiate waste heat reuse agreements between data centres and manufacturing partners, replacing today's costly manual IS brokerage?*

European data centres consume on the order of 240-340 TWh of electricity per year and convert almost all of it into low- and mid-grade waste heat. A growing share of this heat is recoverable but the literature documents that the bottleneck is rarely physical: it is institutional, regulatory and informational. The framework integrates engineering, regulatory and behavioural dimensions of industrial symbiosis into a single computational pipeline rather than treating them in isolation.

## Architecture

```
+----------------------------------------------------+
|  Layer 1: Physical-DT                              |
|  Closed-form RC thermal model (Eq. 4.1)            |
|  Block-bootstrap privacy-preserving export         |
|  Multi-benchmark calibration vs Frontier KPIs      |
+----------------------------------------------------+
                       |
                       v
+----------------------------------------------------+
|  Layer 2: Institutional-LLM                        |
|  Neo4j Knowledge Graph (29 regulatory documents)   |
|  23 Cypher templates, 14 routed (deterministic)    |
|  Cross-model judge (Haiku produces, Sonnet rates)  |
|  Produces Delta_T_c regulatory friction term       |
+----------------------------------------------------+
                       |
                       v
+----------------------------------------------------+
|  Layer 3: Negotiation-Agent                        |
|  Bilateral N=2 Gymnasium environment               |
|  PPO learning DC agent vs rule-based responder     |
|  Four normative shielding rules (R1-R4)            |
|  LP welfare baseline + closed-form Shapley         |
+----------------------------------------------------+
                       |
                       v
+----------------------------------------------------+
|  Methodological Robustness Layer (orthogonal)      |
|  17 post-hoc diagnostics in 5 families:            |
|  Sobol | Bootstrap CI | Power | LOO | Perturbation |
+----------------------------------------------------+
```

The four implementation phases map directly onto these layers:

| Layer (thesis) | Implementation phase | Role | Tech stack |
|---|---|---|---|
| **Physical-DT** (Strato 1) | `Phase1_PhysicalDT/` | Thermal + workload simulation of the DC | RC model, CoolProp, block-bootstrap export |
| **IS-Match Score** (OS2) | `Phase2_ISMatch/` | Multi-criteria scoring DC vs manufacturing pair | numpy, pandas, SALib, pyDecision |
| **Institutional-LLM** (Strato 2) | `Phase3_GraphRAG/` | Regulatory KG + Graph-RAG retrieval + cross-model judge | Neo4j, LangChain, Anthropic SDK |
| **Agentic Negotiation** (Strato 3) | `Phase4_MARL/` | Bilateral RL (PPO) + Yazdanpanah shielding + Shapley allocation | Stable-Baselines3, Gymnasium, cvxpy |

The framework is evaluated across a **nine-scenario matrix** (3 DC scales x 3 manufacturing temperature bands) plus one out-of-distribution airside arm (`S_AIR_M`), with 10 seeds per scenario and a methodological robustness layer of 17 post-hoc diagnostics.

## Headline results

| Layer | Metric | Result |
|-------|--------|--------|
| Layer 1 | Multi-benchmark agreement on residual scope-DT metrics | **94.4%** vs 80% threshold |
| Layer 1 | Lag-1 autocorrelation preservation of synthetic export | **98%** across 3 DC scales |
| Layer 1 | Wasserstein-1 marginal fidelity vs target 0.05 | passes by > 1 order of magnitude |
| Layer 2 | Graph-RAG strict EM gap vs no-RAG | **+31.1 pp** (CI95 [+20.3, +33.7]) |
| Layer 2 | Graph-RAG strict EM gap vs LLM-Cypher | **+10 pp** (CI95 [+3.8, +16.2]) |
| Layer 2 | Routing stability under paraphrasing | **96%** at routing target 90% |
| Layer 3 | PoA gap shielded vs unshielded, aggregate 9 LC arm (n=90) | **+0.030** ns (CI95 BCa [-0.018, +0.078]) |
| Layer 3 | PoA gap, Mid_LC subgroup (n=30, exploratory uncorrected) | +0.187 (CI95 [+0.04, +0.37]) — *exploratory, no multiplicity correction* |
| Layer 3 | Convergence rate, shielded configuration | **100%** vs 32.8% baseline |
| Layer 3 | Edge welfare under FW3 IR-floor (cloud validation) | **-7.6 to +202 kEUR/yr** |

Every numerical result in the thesis corresponds to a committed CSV under `results/` inside the relevant phase, so the manuscript and the codebase are traceable end to end.

---

## Repository structure

```
Fasi Applicative/
├── README.md                     this file
├── .gitignore
├── common/                       single source of truth for DC identifiers across phases
│
├── Phase1_PhysicalDT/            Physical Digital Twin (Layer 1)
│   ├── lc/                       9 canonical liquid-cooled scenarios (primary, D6)
│   │   ├── step_1_1_2_rc_model_lc.py
│   │   ├── step_1_3_synthetic_lc.py        block-bootstrap synthetic export
│   │   ├── step_1_4_validation_lc.py       W1 and NDE gates
│   │   ├── step_1_5_sobol_rc_params.py     Saltelli-Sobol sensitivity sweep
│   │   ├── step_1_6_multi_benchmark_calibration.py
│   │   ├── step_1_7_residual_diagnostics.py
│   │   ├── step_1_8_climate_sensitivity.py
│   │   └── results/
│   └── airside/                  S_AIR_M out-of-distribution arm
│
├── Phase2_ISMatch/               IS-Match composite scoring + top-K ranker (LC-only, D6)
│   ├── step_2_1_is_match_score_lc.py
│   ├── step_2_2_top_k_ranker.py
│   ├── step_2_4_delta_tc_calibration_lc.py
│   ├── step_2_5_sobol_weights.py
│   ├── step_2_6_stress_test.py             +-20% perturbation in 8 directions
│   ├── step_2_7_carbon_emissions_avoided.py
│   ├── step_2_8_promethee_crosscheck.py
│   └── results/
│
├── Phase3_GraphRAG/              Knowledge Graph + Cypher routing + LLM judge (Layer 2)
│   ├── step_3_0_setup_schema.py
│   ├── step_3_1{a,b,c,d,e}_ingest_*.py
│   ├── step_3_2_graph_rag_pipeline.py
│   ├── step_3_4_evaluation.py              canonical 100-query benchmark
│   ├── step_3_7_bootstrap_em_ci.py
│   ├── step_3_8_template_loo.py
│   ├── step_3_9_llm_judge.py               Sonnet-class cross-model judge
│   ├── step_3_10_paraphrase_routing_stability.py
│   ├── templates.py                        23 Cypher templates, 14 in canonical pipeline
│   ├── data/                               100-query benchmark + 38-query OOD held-out
│   └── results/
│
└── Phase4_MARL/                  Bilateral PPO negotiation environment (Layer 3)
    ├── env/
    │   ├── is_negotiation_env.py           Gymnasium env, 8-dim observation
    │   └── shielding.py                    R1-R4 normative action masking
    ├── agents/
    │   ├── baseline_lp.py                  LP welfare baseline (cvxpy + fallback chain)
    │   └── shapley.py                      closed-form Shapley N=2
    ├── train/
    │   ├── train_ppo.py                    Stable-Baselines3 PPO training
    │   └── run_ablation.py                 A0 vs D ablation runner
    ├── step_4_5_bootstrap_poa_ci.py        bootstrap CI on PoA and Shapley
    ├── step_4_6_shielding_loo_evaluation.py
    ├── step_4_7_state_observation_sensitivity.py
    ├── step_4_8_deltatc_perturbation.py
    ├── step_4_9_power_analysis.py
    ├── step_4_10_per_scenario_gap_ci.py
    └── results/
```

Scripts follow the convention `step_X_Y[_Z]_<descriptor>.py` where X encodes the phase and Y, Z the substep order. The orchestrator `run_phase_X.py` for each phase chains the canonical sequence. Robustness modules live at higher numeric suffixes and write to `results/robustness/`.

---

## Quick start

```bash
git clone https://github.com/fedelomi/thesis-industrial-symbiosis-dt.git
cd thesis-industrial-symbiosis-dt
python -m venv .venv && source .venv/bin/activate           # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp Phase3_GraphRAG/.env.example Phase3_GraphRAG/.env        # add Neo4j and Anthropic credentials
```

Run a phase end to end:

```bash
python Phase1_PhysicalDT/lc/run_phase_1_lc.py
python Phase2_ISMatch/run_phase_2_lc.py
python Phase3_GraphRAG/run_phase_3.py                       # full pipeline (0 to 11)
python Phase4_MARL/train/run_ablation.py --seeds 10 --timesteps 100000
```

The Phase 3 benchmark exercises the pipeline in deterministic context-only mode (`llm=None`) and is fully reproducible without API access. The generative rendering step is measured separately in the post-freeze instrumentation campaign of Section 6.2.2-ter.

## Software stack

Python 3.11, Gymnasium 0.29, Stable-Baselines3 2.1, CoolProp 6.4, cvxpy 1.4 (CLARABEL, SCS, ECOS fallback chain), Neo4j 5.x Python driver, Anthropic SDK 0.34, SALib 1.5, scipy 1.11, pyDecision 4.x, pandas 2.1, matplotlib 3.8.

## Reproducibility protocol

The canonical evaluation uses 10 PPO training seeds across the 9 LC scenarios (n=90 paired observations per metric), 10 seeds for the S_AIR_M out-of-distribution arm and 10,000 bootstrap resamples for each confidence interval. Seed 42 is canonical for the Saltelli-Sobol sweep of Phase 2. Detailed seed and configuration tables are in Appendix B of the thesis.

---

## Active implementation decisions

These are the methodological decisions driving the code. Each is documented in the docstring header of every script that implements it.

| ID | Topic | Phase | Status |
|---|---|---|---|
| D1 | DT calibration on aggregate Frontier KPIs (grey-box) | Phase 1 | Confirmed |
| D2 | Geography-agnostic, sector-parametric IS dataset | Phase 2 | Confirmed |
| D3 | Regulatory corpus restricted to Italy and Denmark | Phase 3 | Confirmed |
| D6 | Phase 2 LC-only (airside excluded on IS-Match evidence) | Phase 2-4 | Confirmed |
| D9 | Action-denormalisation contract on Q_negotiated | Phase 4 | Confirmed |
| D11 | Negotiation topology bilateral N=2 (multilateral as FW4) | Phase 4 | Confirmed |

## Methodological notes

A few non-obvious choices worth flagging for anyone reading this code:

- **Grey-box calibration (D1).** The Phase 1 RC model is calibrated against aggregate Frontier KPIs published by Jadhav and Liu, not against the underlying 47186-record dataset which is not public. The artefact is, strictly, a literature-calibrated digital model with DT-ready interfaces.
- **Privacy-utility trade-off (Section 3.3.3 of the thesis).** The block-bootstrap export preserves lag-1 autocorrelation at 98% while breaking long-range autocorrelation at lag-96 and lag-672 by design, against an explicit threat model focused on operator-schedule reconstruction.
- **Cross-model judge (Section 4.3.3).** A Haiku-class model serves the production hot path; a Sonnet-class judge evaluates it. The 18 pp gap between strict and semantic EM is interpreted as the false-positive rate of keyword matching rather than a defect of the retrieval pipeline.
- **Bilateral framing (D11).** The negotiation is bilateral N=2 with closed-form Shapley. The multilateral N>=3 extension via Monte Carlo Shapley is declared as Future Work FW4.

## Methodology reference

The framework maps onto the thesis gap-analysis chain:

```
Gap -> Research Question -> Objective -> Phase -> Metric
```

| Phase | Objective | Primary metric |
|---|---|---|
| 1 | Privacy-preserving DT | W1_norm <= 0.05, NDE <= 0.20, multi-benchmark agreement 94.4% |
| 2 | IS feasibility scoring | Top-3 invariance under +-20%, Sobol delta-dominant |
| 3 | Regulatory Graph-RAG retrieval | EM strict +31.1 pp vs no-RAG, routing stability 96% |
| 4 | Bilateral negotiation (RL + shielding) | PoA gap +0.030 ns (aggregate); +0.187 Mid_LC subgroup (exploratory, uncorrected); Shapley gap -0.207 sig |

---

## Citation

If you use any part of this framework, please cite:

```bibtex
@mastersthesis{lomi2026is,
  author  = {Federico Lomi},
  title   = {An Integrated Three-Layer Framework for Data Centre to Manufacturing Industrial Symbiosis:
             Physical Digital Twin, Regulatory Graph-RAG and Agentic Negotiation},
  school  = {Politecnico di Torino},
  year    = {2026},
  type    = {Master's Thesis}
}
```

## License

The code in this repository is released under the MIT license. The regulatory corpus ingested into the Knowledge Graph references public European, Italian and Danish documents whose own licensing terms apply.

## Contact

Federico Lomi, federico.lomi@gmail.com
