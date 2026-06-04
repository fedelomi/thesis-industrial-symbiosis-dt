# READING LOG — Phase 3 Blind Reconstruction

Chronological audit trail of every source read during the blind design of Phase 3
(Institutional-LLM / Strato 2). One line per file with what was extracted. Skips
required by Rule Zero (anti-anchor) are logged with `# SKIPPED per anti-anchor rule`.

---

## ⚠️ PRE-EXISTING ANTI-ANCHOR LEAK (disclosed up front, NOT read during design)

Before any file was read, the session's persistent auto-memory index (`MEMORY.md`,
injected by the harness into the system context) already contained pointers that
**partially reveal the existing Phase 3 architecture**. These are background context
I did not choose to read; they arrived in the system reminder. I am disclosing them
here per the mission's "route around the leak" instruction:

- A memory pointer names the existing Phase 3 as **"GraphRAG"** and references
  "BGE + top-5 union" retrieval, a "multi-template" variant (v7), and metrics
  ("EM 59% vs semantic 41%", "OOD 35.3→61.8%").
- This tells me the incumbent is a **graph-based retrieval-augmented-generation**
  design using a knowledge graph + dense (BGE) embeddings.

### ⚠️ SECOND PRE-EXISTING LEAK — project `CLAUDE.md` (harness auto-injected)

While reading the Phase 4 contract code, the harness auto-injected the project's
`Fasi Applicative/CLAUDE.md` as a system reminder (I did not choose to open it).
It contains a **substantial Phase 3 architecture leak**:
- "Layer 2 — Institutional-LLM: **Neo4j Knowledge Graph, Graph RAG, LangChain**"
- Folder note: "Phase3_GraphRAG/ ← Neo4j Graph RAG (KG ingest, **Cypher queries**, **RAGAs** benchmark)"
- Tech stack lists `neo4j`, `langchain` (GraphCypherQAChain), `ragas`.

This confirms and extends the memory leak: the incumbent is a **Neo4j + Graph-RAG +
LangChain GraphCypherQAChain** pipeline, benchmarked with **RAGAs**.

**Route-around decision:** I treat this as background contamination, NOT as a design
input. I will design from the *contract + corpus* on first principles. I still do
NOT know: the KG entity/relation schema, the Cypher templates, the benchmark
questions, the ΔT_c derivation method, or the performance internals. Notably, the
mission's own stack constraints ("the gate must be **deterministic and reproducible
across runs**"; "probabilistic LLM calls must be wrapped in a verification layer
that can be audited") point me toward a design decision I can defend INDEPENDENTLY of
the leak — see ARCHITECTURE.md. Full disclosure in FINAL_REPORT anti-anchor self-audit.

**Mitigation actions taken:**
1. I did NOT open the detailed memory files (`project_phase3_*.md`,
   `project_code_review_*.md`) — only the one-line index entries were visible.
2. I will design from first principles against the framework spec + the corpus,
   and in `FINAL_REPORT.md` "Anti-anchor self-audit" I will state explicitly that
   I knew the incumbent is GraphRAG, and how I prevented that knowledge from either
   (a) making me copy a graph design, or (b) making me reflexively avoid graphs if
   they are genuinely the best fit.
3. The convergence/divergence experiment is therefore **partially compromised on
   the headline label** (I know it's a graph). It is NOT compromised on the
   internals: I have not seen the KG schema, the Cypher templates, the exact
   retrieval pipeline, the benchmark questions, or the ΔT_c derivation method.

---

## Reading sessions (chronological)

### Session 0 — Territory mapping (2026-06-03)
- `Glob Fasi Applicative/Phase1_PhysicalDT/**` — confirmed Phase 1 = calibrated RC thermal model (airside + lc variants), step_1_* + step_0_eed_compliance.
- `Get-ChildItem "Fasi Applicative" -Directory` — confirmed real folder names: Phase1_PhysicalDT, **Phase2_ISMatch** (spec called it Phase2_AgenticDT), **Phase3_GraphRAG** (DENIED), **Phase4_MARL** (spec called it Phase4_AgenticNegotiation), common, Documenti. I observed the name `Phase3_GraphRAG` in the parent listing but did NOT descend into it. # Phase3_GraphRAG NOT ENTERED per anti-anchor rule.
- `Glob Phase2_ISMatch/**/*.py` — 13 scripts incl. `step_2_4_delta_tc_calibration_lc.py` (ΔT_c is calibrated/consumed in Phase 2 too) and `step_2_1_is_match_score_lc.py` (IS-Match Score; the normalization unit for ΔT_c).
- `Glob Phase4_MARL/**/*.py` — downstream consumer. Contract-critical files: config/reward_params.py, config/scenarios.py, env/is_negotiation_env.py, env/models.py, env/shielding.py, step_4_8_deltatc_perturbation.py, step_4_6_shielding_loo_evaluation.py.
- `Glob common/**` — common/dc_id_mapping.py (shared DC identity mapping).
- Located corpus: `thesis-obsidian-vault/Test Second Brain/raw/F - DC WHR Cases & Policy/` — 29 policy PDFs (F01–F29) + 1 xlsx. ALLOWED (corpus is given by framework).

### Session 1 — Phase 4 contract (downstream consumer) (2026-06-03)
- `Phase4_MARL/config/reward_params.py` — RewardParams dataclass: lambda_is_match_uplift=1.0, convergence_delta=0.02, gas 45 EUR/MWh, emission 0.20 tCO2/MWh, transport 0.04 EUR/MWh/km, amort 10 yr, 8000 h/yr. No ΔT_c field here directly.
- `Phase4_MARL/config/scenarios.py` — **SCENARIO CONTRACT**: 3 DC scales {Edge_LC 475kW, Mid_LC 3104kW, Hyperscale_LC 24500kW} x 3 tiers {LowT 60C, MidT 90C, HighT 130C} = S1..S9 (+S_AIR_M). Tier reps: LowT_02_Agro 60C/HP, MidT_04_PaperPulp 90C/HP, HighT_07_Rubber 130C/CO2_HTHP. IS-Match scores loaded from `Phase2_ISMatch/results/step_2_4_delta_tc_calibration_lc.csv` iteration=4, range 0.47-0.5767. upgrade_required: HP for LowT/MidT, CO2_HTHP for HighT (dT>50C).
- `Phase4_MARL/env/models.py` — DCProfile.is_match_score float; exergy_dt in [0,1]; Offer=[Q,T,upgrade,price,duration]. IS-Match target: marginal 0.36-0.53 -> high-priority >0.60.
- `Phase4_MARL/env/shielding.py` — ShieldingLayer enforces 4 hard rules (Q<=Q_avail BLOCKED; T>=T_req-5 BLOCKED; price>=amortised floor WARNING; upgrade==required BLOCKED). NOTE: these are physical/thermal/economic/normative-upgrade rules, NOT a jurisdictional regulatory compliance gate. The Phase 3 compliance gate is NOT yet wired into shielding -> my gate must be a standalone callable that does not require editing Phase 4.
- `Phase4_MARL/env/is_negotiation_env.py` — **KEY**: obs is 8-dim Box[0,1] = [Q_avail/Q_inst, T_supply/100, exergy, **is_match**, round_norm, Q_proc/Q_avail, T_req/200, budget]. is_match_score is clipped to [0,1] in _observe (line 334). Reward = econ/8e5 + lambda*(uplift/0.10) + convergence/blocked terms. IS-Match formula referenced (line 491): **IS-Match = beta*RI_temp + gamma*Exergy - delta*DeltaTC** -> ΔT_c is a SUBTRACTED penalty in [0,1] units.
- `Phase4_MARL/step_4_8_deltatc_perturbation.py` — ΔT_c robustness: perturbs is_match_score by x1.2/x0.8/uniform-0.5; robust if |ΔPoA|<0.05 under +-20%. is_match clipped to [0,1] (`np.clip(is_match_override, 0.0, 1.0)`). Treats ΔT_c handle as scalar multiplier on is_match_score.

### ⚠️ THIRD LEAK encountered (in allowed Phase 4 file) — routed around
- `step_4_8_deltatc_perturbation.py` docstring names a Phase 3 internal artifact:
  `Phase3_GraphRAG/data/step_3_5_bis_delta_tc.json` ("baseline factors"). This reveals
  the incumbent's ΔT_c output **filename** and that step numbering reaches `step_3_5_bis`.
  **Action:** I did NOT open that JSON. # SKIPPED per anti-anchor rule. I will derive my
  own ΔT_c independently and use my own clean step naming. The only thing I take from this
  is the CONTRACT (ΔT_c is per-tier, in IS-Match [0,1] units, folded into is_match_score) —
  which is independently visible in the allowed Phase 2/4 code, not a Phase 3 secret.

### Session 2 — Phase 2 ΔT_c producer/consumer contract (2026-06-03)
- `Phase2_ISMatch/step_2_1_is_match_score_lc.py` — **IS-Match formula**: `β·RI_temporal + γ·Exergy_DT_norm − δ·ΔTC_norm`, clipped [0,1]. Weights β=0.40, γ=0.40, δ=0.20 (simplex). RI = α_Q·Q + α_T·T_compat + α_t·Avail (α=0.25/0.35/0.25, α_d=0). Tiers high≥0.60 / marginal≥0.30 / not_viable<0.30. **ΔTC_baseline inline anchors: LowT_60C=0.25, MidT_90C=0.42, HighT_130C=0.55** (monotone increasing with T = higher friction for HighT). CSV schema: dc_name, process_name∈{LowT_60C,MidT_90C,HighT_130C}, ri_temporal, exergy_dt_norm, delta_tc_norm, is_match_score, tier.
- `Phase2_ISMatch/step_2_4_delta_tc_calibration_lc.py` — **DOWNSTREAM CONTRACT** (Phase 3 -> Phase 2 wiring, legitimately visible in allowed Phase 2 file): loads `{"reduction_factors": {"LowT_60C":r,"MidT_90C":r,"HighT_130C":r}}`, r∈[0.1,0.3]; applies `ΔTC_post = ΔTC_baseline·(1−r)`; recomputes is_match; iterates max 5, converge max|Δis_match|<0.01; writes `step_2_4_delta_tc_calibration_lc.csv` (cols incl. iteration, is_match_score) consumed by Phase 4 `build_scenarios()` at iteration=4. Fallback r=0.10 all tiers. **My Phase 3 must emit this JSON schema as a drop-in.** The "barrier reduction" framing = institutional-LLM layer reduces the cross-firm information-asymmetry component of regulatory transaction cost (Gap 3).
- `Phase4_MARL/tests/conftest.py` + `tests/test_env_step.py` — **TEST HARNESS**: Phase 4 imported with PHASE4_ROOT on sys.path (`from config.scenarios import SCENARIOS`, `from env.is_negotiation_env import ISNegotiationEnv`). Existing tests assert obs.shape==(8,), float32, action Box(4), reward is float. My integration test must replicate the sys.path setup and NOT modify these.

**ΔT_c CONTRACT SUMMARY (what I must satisfy, derived independently):**
- Output A (mission literal): vector {LowT,MidT,HighT} of absolute ΔT_c_norm ∈ [0,1] (IS-Match units), corpus-derived.
- Output B (real wiring): `reduction_factors` JSON {tier_60/90/130 keys} ∈ [0.1,0.3] consumed by step_2_4.
- Output C (gate): deterministic callable scenario->verdict{compliant/non_compliant/conditional}+articles+thresholds.
- I will write artifacts under `Phase3_Blind/data/` (NEVER into Phase3_GraphRAG/).

### Session 3 — Framework chapters (2026-06-03)
- `chapters/02-literature-review.md` (full, 338 lines) — **Gap 3** = "apply LLM+RAG to the IS information barrier for DC-manufacturing using an IS-specific corpus (EED Art 26, Del.Reg 2024/1364, DK framework, IT White Certificates, ISO 50001, contract templates)." Transaction-cost theory (Yazdanpanah: transport/treatment/transaction; transaction=institutional, hardest). Three institutional barriers (Lind&Rundgren): information asymmetry, contractual tension (20yr hold-up), trust deficit. Manufacturing temp tiers (JRC FDM BREF): LowT<60, MidT 60-120, HighT>120C. **§2.4.5 NREL neuro-symbolic (Buster et al.): LLM extracts structured facts + symbolic system (decision tree) validates/reasons -> explainable, auditable, 85-90%** = literature-grounded alternative to Graph-RAG. §2.4.2 standard vector RAG vs Graph-RAG (Polimi/Voltiva +13pp). Regulatory facts §2.5: EED Art 12 (DC>=500kW IT report Annex VII), Art 23 (5yr comprehensive assessment), Art 24 (efficient DHC >=50% RES/WH thresholds; >45MW feasibility), Art 25 (industrial WH assessment), **Art 26 (DC>=1MW: reuse heat OR supply DH OR document negative CBA)**. Country targets Tab 2.2 (DE 10%/2026..30%/2028; SE/DK 25-35%/2025-30; etc). IT=incentive/TEE vs DK=planning/Heat Supply Act zoning (D3 two-jurisdiction corpus).
- ⚠️ LEAK in lit review (allowed): Fig 2.9 caption says incumbent uses "deterministic-template specialization (Section 4.3.2)"; Gap 3 title + Table 2.3 say "Graph RAG"; §2.5 says relations "inform the Knowledge Graph schema (Ch4)". Confirms known GraphRAG incumbent. Routed around: did NOT read §4.3.2 / KG schema.
- `chapters/00-abstract.md` — framework summary. ⚠️ PERFORMANCE-METRIC LEAK: incumbent Graph-RAG "+31.1pp over no-RAG on 100-Q benchmark; cross-model Sonnet judge surfaces 18pp keyword overstatement." **Logged & routed around**: I will NOT target +31.1pp or the 18pp figure; my ~100-Q benchmark + LLM-judge + cross-model are independently MISSION-REQUIRED (deliverables 3 + stack constraints), not copied. Other (non-Phase3) facts used freely: 9-scenario matrix, scale-dependent shielding (Edge PoA -0.199, Mid +0.187, Hyper +0.103), Hyperscale+MidT optimum (125.8 GWh, 34712 tCO2/yr).
- `chapters/03-system-methodology.md` §3.4.1 "IS-Specific KG Construction", §3.4.2 "Graph-RAG", §3.4.3 "Neuro-Symbolic Validation". # SKIPPED §3.4.1-3.4.3 per anti-anchor rule (meta-instruction: route around KG schema). Read §3.1-3.3, §3.4 intro + §3.4.4 (Layer2->Layer3 interface=contract), §3.5, §3.6 (incl 3.6.4 robustness), §3.7.

- `chapters/03-system-methodology.md` §3.1-3.3, §3.4 intro+§3.4.4, §3.5, §3.6, §3.7 (read; §3.4.1-3.4.3 skipped). **Eq 3.3** IS-Match = β·RI + γ·Exergy − δ·ΔTC_estimated (β=γ=0.40, δ=0.20); **δ is Sobol-dominant** (S_δ=0.870) -> ΔT_c drives the ranking. Table 3.1: Layer 2 OUT = "ΔT_c transaction cost per tier" + IS-Match/top-K; Layer 3 IN = "ΔT_c cost parameter". §3.6.1 matrix: Edge 500kW (Art 12 only), Mid 3.2MW + Hyper 25MW (Art 12 + **Art 26 ≥1MW**); temp Low<60/Med 60-120/High>120 (JRC FDM BREF). §3.6.2 Layer-2 targets: **accuracy >85%, hallucination <15%, 100-Q benchmark, neuro-symbolic logical-consistency rate, privacy Δ<5%**. §3.6.4 robustness families: Sobol / Bootstrap-BCa-CI(10k) / Power / LOO / Perturbation (generic, framework-level; I adopt the FAMILIES for my eval).
- ⚠️ LEAK in Ch03 Table 3.6 (allowed): names incumbent Phase 3 internals - "step_3_7 (EM CI)", "step_3_8 (template LOO)", "step_3_10 (paraphrase routing)", "three Cypher templates whose removal degrades EM accuracy by 6-8 pp". # Logged & routed around: I will NOT design exactly-3 Cypher templates, will NOT copy EM/paraphrase-routing specifics; my components + metrics are independently designed. Generic robustness families are framework-level (Gap 6) and legitimately shared.

### Session 4 — Concept pages + impl-decisions D1-D5 + Ch06 (2026-06-03)
- `concepts/transaction-cost.md` — 3 TC types (transport/treatment physical; transaction=institutional: search/negotiation/monitoring/enforcement). Reduced by data-sharing protocols, DTs (anonymised), regulatory mandates. Shapley allocation.
- `concepts/eed-art-dc-waste-heat.md` — **GATE DECISION TREE**: DC≥500kW IT -> Art 12 report KPIs; total-rated >1MW -> Art 26 WHR obligation -> [DH efficient+nearby+CBA+ ⇒ MANDATORY_CONNECT; CBA− documented ⇒ EXCEPTION (still report); else ⇒ explore other WHR]. "efficient DHC" RES threshold scaled 50%(2025)->100%(2045). (Page references denied [[graph-rag]]/[[graph-rag-entity-schema]] but I did not follow those links.)
- `concepts/comprehensive-assessment-dhc.md` — Art 23 5-step CA: demand-map -> supply-inventory -> spatial-match (10-15km DH main, 3-5km branch) -> techno-econ feasibility -> recommendation. DK Bilag 13: mandatory-connection zones for WH sources **>5 MW**.
- `concepts/certificati-bianchi-tee.md` — IT TEE: ≥25 tep/yr access, 5yr (10 grandi progetti), 250-350 €/TEE, 1 TEE≈11.63 MWh, ~21-30 €/MWh. TEE makes CBA positive => friction reducer (Italy).
- `concepts/decision-support-system.md` — RI>0.5 feasible /<0.3 not; tech menu (HX, HP, CO2-HP≤110C, absorption, ORC, TES). Recovery Index = Simeone WHER.
- `implementation-decisions.md` D1-D5 (stopped at D6/line141). **D3 = document-grounded benchmark protocol**: ground truth = exact corpus clause (no inter-rater); 3 categories (factual lookup / multi-hop IS reasoning / comparative screening); metrics EM + ROUGE-L + faithfulness/context-precision; supervisor reviews coverage. (Aligns with mission's "deterministic ground truth" requirement; I adopt document-grounded but design my own taxonomy/metrics.) D1/D5 grey-box DT calib; D2 sector-parametric mfg (LowT 60-65C agro/dairy, MidT 90C paper/pulp, HighT 130-135C rubber/steam); D4 ISO 50001 available.
- `chapters/06-conclusions.md` §6.1 summary + RQ1/RQ2/RQ4 (read), §6.2.1 engineering scope, §6.2.3 design trade-offs (read). # SKIPPED RQ3-via-GraphRAG subsection (47-54), §6.2.2 + §6.2.2-bis (75-106) per anti-anchor. ⚠️ §6.1 Layer-2 paragraph re-confirmed known leaks (deterministic Cypher templates + Haiku gen, +31.1pp, 18pp, Sonnet judge) — already disclosed; routed around (won't target those numbers). Useful (non-Phase3): design trade-offs (Edge scale-localized fairness/efficiency coupling, bilateral-DT asymmetry, trajectory-vs-privacy, conv-vs-OOD, simulated-vs-human) -> inform my limitations/future-work.

### Deprioritized allowed reads (redundant, NOT anti-anchor skips)
- `chapters/01-introduction.md`, `chapters/04` §4.1/4.2/4.4/4.5, `chapters/05` §5.1/5.2/5.3/5.5/5.6: allowed, but mirror the Phase 1/2/4 CODE I already read directly + §3.x methodology. RQ3 statement already supplied verbatim in the mission brief. Deprioritized to avoid low-value reading; not anti-anchor skips.

### Session 5 — Corpus extraction + design + implementation (2026-06-03)
- Parallel extraction workflow over the 29 corpus PDFs (29 read-only Explore subagents, anti-anchor preamble forbidding any Phase3_GraphRAG access, 518 s) -> `data/corpus_facts.json` (204 articles, 381 thresholds, 163 triggers). Dumped F10 (EED) + F25 (Del.Reg 2024/1364) + F24 (FDM BREF) to ground the gate rules and tier temps with corpus provenance.
- Designed + built the Neuro-Symbolic Regulatory Reasoner: step_3_0..step_3_7 + build_benchmark + run_phase_3_blind + integration_test + tests. No Phase3_GraphRAG file was ever opened; all design decisions trace to the contract (Phase 2/4 code), the allowed chapters/concepts, and the corpus facts.

### READING-LOG STATISTICS (final)
- Directly read allowed files: ~20 (8 Phase4 code, 2 Phase2 code, 4 chapters, 5 concept pages, impl-decisions D1-D5).
- Corpus PDFs read via extraction workflow: 29.
- Skipped per Rule Zero (anti-anchor): ~13 targets (Phase3_GraphRAG folder; Ch04 4.3-4.4; Ch05 5.4; Ch06 RQ3-subsec/6.2.2/6.2.2-bis; Ch03 3.4.1-3.4.3; log.md; lesson-3; lesson-5 Phase3 paras; concepts graph-rag/graph-rag-entity-schema/neuro-symbolic-ai; impl D6-D19; step_3_5_bis_delta_tc.json).
- Leaks encountered + routed around: 7 (memory index; project CLAUDE.md; step_4_8 filename; Fig 2.9 caption; abstract +31.1pp/18pp; Table 3.6 three-Cypher-templates; 6.1 Layer-2 paragraph).
- Effort split (relative, no wall clock): reading/extraction ~45%, design ~10%, implementation+validation ~45%.

