# Phase 3 Blind Reconstruction: Final Report

**Layer 2 (Institutional-LLM / Strato 2) reconstructed blind from the framework
specification and the Phase 1/2/4 interfaces.** Date: 2026-06-03.

---

## 1. The architectural choice, in one paragraph

I built a **Neuro-Symbolic Regulatory Reasoner (NSRR)**: an offline, schema-constrained
LLM extraction pass turns the 29-document regulatory corpus into a typed
Regulatory Knowledge Base (articles, numeric thresholds, IF-THEN triggers, each with
provenance), which is then loaded deterministically at runtime and consumed by a
forward-chaining symbolic rule engine. The two contractual outputs to Phase 4, the
compliance gate verdict and the per-tier `ΔT_c` regulatory-friction vector, are
produced entirely by the symbolic engine with no runtime model call, so they are
bit-for-bit reproducible and every output traces to specific rules and corpus
clauses. A structured retriever serves the open-ended cross-firm question answering,
evaluated against a vector-only RAG baseline. The decisive reason for putting the
symbolic engine first, rather than a graph-or-vector RAG that generates the answer,
is the framework's hard requirement that the gate be deterministic, reproducible and
auditable, which an LLM-generated verdict cannot satisfy; the neuro-symbolic split
confines the irreducible model uncertainty to a single audited offline step.

## 2. Reading-log statistics (anti-anchor audit)

- **Directly read allowed sources: ~20 files.** Phase 4 code (8: reward_params,
  scenarios, models, shielding, is_negotiation_env, step_4_8, tests/conftest,
  tests/test_env_step); Phase 2 code (2: step_2_1 IS-Match, step_2_4 ΔTC
  calibration); chapters (4: 00 full, 02 full, 03 partial, 06 partial); concept
  pages (5: transaction-cost, eed-art-dc-waste-heat, certificati-bianchi-tee,
  comprehensive-assessment-dhc, decision-support-system); implementation-decisions
  (D1-D5).
- **Corpus: 29 policy PDFs** read by a parallel extraction workflow (29 read-only
  Explore subagents, 518 s wall-clock), distilled into `data/corpus_facts.json`
  (204 articles, 381 thresholds, 163 triggers).
- **Skipped per Rule Zero (anti-anchor): ~13 targets.** The entire
  `Phase3_GraphRAG/` folder; Ch04 4.3-4.4 (Phase 3 impl/eval); Ch05 5.4 (Phase 3
  results); Ch06 RQ3-via-GraphRAG subsection, 6.2.2, 6.2.2-bis; Ch03 3.4.1-3.4.3 (KG
  construction, Graph-RAG, neuro-symbolic-validation internals); `log.md`,
  `lesson-3`, the Phase-3 paragraphs of `lesson-5`; the concept pages `graph-rag`,
  `graph-rag-entity-schema`, `neuro-symbolic-ai`; implementation-decisions D6-D19;
  and the Phase 3 artefact `step_3_5_bis_delta_tc.json` (encountered by reference in
  an allowed Phase 4 file, not opened).
- **Leaks encountered and routed around: 7 events**, all logged in `READING_LOG.md`:
  the session memory index, the project `CLAUDE.md`, the `step_4_8` filename
  reference, Figure 2.9's "deterministic-template" caption, the abstract's
  +31.1 pp / 18 pp metrics, Table 3.6's "three Cypher templates", and the Ch06 6.1
  Layer-2 paragraph.
- **Effort split (no wall clock; relative).** Reading and corpus extraction ~45%
  (dominated by the 29-document parallel extraction plus the chapter/code reads),
  architecture design ~10%, implementation plus validation ~45%. The isolation
  discipline (logging every read, routing around every leak) was the single largest
  reading-phase overhead, by intent.

## 3. Benchmark design summary

102 document-grounded questions (`data/benchmark.jsonl`) in four categories:
55 `threshold_lookup` (numeric, gold = corpus clause value), 15
`compliance_verdict` (gold = independently authored EED Art. 12/26 verdict), 20
`multi_hop` (gold = required fact-token set), 12 `comparative` (gold = keyword set).
Ground truth is the corpus clause itself (the strongest objective protocol), with
deterministic scorers and a symmetric expected-unit hint for both systems. Full
protocol and bias analysis in `benchmark_design.md`.

## 4. Evaluation results

| | NSRR | Vector-only baseline | Gap |
|--|------|----------------------|-----|
| **Overall (102 Q)** | **85.3%** | **67.6%** | **+17.6 pp** |
| compliance_verdict (15) | 100.0% | 33.3% | +66.7 |
| multi_hop (20) | 65.0% | 45.0% | +20.0 |
| threshold_lookup (55) | 92.7% | 85.5% | +7.3 |
| comparative (12) | 66.7% | 66.7% | +0.0 |
| groundedness | 94.1% | 91.2% | |

Bootstrap 95% CI on the gap: **[+10.8, +25.5] pp**, significant (10,000 resamples).
Leave-one-out: removing the symbolic gate drops NSRR to 72.5%, so the gate is
**load-bearing (+12.7 pp)**. Paraphrase perturbation leaves accuracy unchanged
(85.3% -> 85.3%, stable).

**What it means.** The structured, symbolic layer decisively beats vector-only RAG
exactly where RQ3 predicts: compliance verdicts (vector retrieval cannot assemble a
multi-clause decision and asserts no verdict), multi-hop chains, and numeric
thresholds (where vector similarity fabricates or mis-selects numbers). On fuzzy
conceptual `comparative` recall the two tie, because vector similarity is a genuinely
adequate tool for that sub-task. NSRR clears the framework's >85% accuracy target
(85.3%) while the baseline sits at 67.6%. The fact that the system does NOT sweep
every category (it ties on comparative) is reported as-is and makes the headline gap
more credible, not less.

## 5. Phase 4 integration test result

`integration_test.py` passes **3/3 checks against the unmodified Phase4_MARL code**
(also green under pytest):
1. the corpus-derived `ΔT_c` reduction factors (all in [0.1, 0.3]) applied to the
   Phase 2 IS-Match baseline yield an `is_match_score` in [0,1] that the real
   `ISNegotiationEnv` accepts (observation in bounds, reward a finite float);
2. the compliance gate is callable per episode and composes with the real
   `ShieldingLayer` as an admissibility mask without editing Phase 4;
3. all contractual data types and ranges match (is_match float in [0,1],
   reduction_factor float in [0.1, 0.3], verdict JSON-serialisable).
The **existing 20 Phase 4 tests still pass** (no Phase 4 file was modified), and the
8 Phase 3 Blind unit tests pass.

## 6. Convergence vs divergence against the framework

This is the experiment's payload. **Convergence:** my blind design lands on a
neuro-symbolic core (LLM-extract then symbolic-reason), on the per-tier `ΔT_c`
ordering LowT < MidT < HighT (0.53 < 0.72 < 0.98 vs the Phase 2 anchors
0.25 < 0.42 < 0.55), and on a document-grounded ~100-question benchmark with a
cross-model judge and a >85% target. **Divergence:** I made the symbolic engine the
PRIMARY decision-maker (a deterministic rule engine produces the verdict and `ΔT_c`),
whereas the framework, per the leaked labels, makes Graph-RAG primary with
neuro-symbolic only as a validation step; I used a typed in-memory KB plus
forward-chaining and TF-IDF retrieval rather than a Neo4j knowledge graph, Cypher
templates and LangChain; and my corpus-derived `ΔT_c` magnitudes run roughly double
the hand-set Phase 2 baseline (the model explicitly prices the CO2-HTHP physical
complexity and high-temperature standard exposure that the baseline compresses).
Because Phase 2 consumes the reduction factor, not the absolute, the magnitude
divergence does not perturb the downstream pipeline.

## 7. Three Future Work items

1. **Out-of-distribution jurisdiction generalisation.** The corpus is EU + Italy +
   Denmark (Decision D3), so the benchmark and `ΔT_c` are in-distribution. Re-extract
   a held-out jurisdiction (e.g. Germany or France) and measure cross-jurisdiction
   gate accuracy and `ΔT_c` transfer; this is the honest test the current scope
   defers (bias B4).
2. **Retrieval-substrate ablation (dense / graph arm).** I deliberately used TF-IDF
   for the deterministic baseline and symbolic reasoning for the decision. A dense
   sentence-transformer arm and a graph-traversal arm would quantify whether a
   stronger retrieval substrate narrows the multi-hop gap, isolating how much of
   NSRR's advantage is the symbolic engine versus the retriever.
3. **Learned `ΔT_c` weights and closed-loop calibration.** The burden weights are
   fixed and documented. Calibrating them against realised transaction-cost data and
   closing the Phase 3 -> Phase 2 -> Phase 4 feedback loop (the framework's deferred
   iterative cycle) would replace the fixed-weight assumption with an empirically
   fitted friction model.

## 8. Anti-anchor self-audit

I was not operating blind on the headline label, and I will not pretend otherwise.
Before reading a single file, the session memory index and the auto-injected project
`CLAUDE.md` already told me the incumbent Layer 2 is a Neo4j knowledge graph with
Graph-RAG, LangChain and a RAGAs benchmark, and later in-text leaks added
"deterministic Cypher templates", a Haiku generator, a Sonnet judge and the +31.1 pp
headline. So the convergence on the *neuro-symbolic* family is partly contaminated:
I also read the NREL neuro-symbolic paper in the allowed literature review (Section
2.4.5), which independently points at that family, but I cannot honestly claim a
clean blind rediscovery of the label. Where I do claim independence is the specific
specialisation: making the symbolic engine the primary decision-maker, refusing a
graph database, and deriving `ΔT_c` from a transparent burden score. I caught myself
anchoring once concretely: when designing the multi-hop path I was tempted to add a
property graph "because the incumbent has one and graphs are good at multi-hop". I
stopped and checked the constraint rather than the precedent: the gate must be
deterministic, which forbids an LLM that writes graph queries; and a 29-document,
two-jurisdiction corpus with a shallow article structure does not justify a graph
server. Both reasons hold regardless of what the incumbent did, so I rejected the
graph on its merits and kept the typed-KB-plus-rules design. The internals that
matter for the experiment, the KB schema, the rule set, the `ΔT_c` derivation method
and the benchmark questions, were built without ever seeing the incumbent's KG
schema, Cypher templates, benchmark or metrics, all of which I logged as skipped or
routed-around. Net assessment: a contaminated convergence on the architectural
family, a genuine and constraint-driven divergence on the decision substrate, and an
uncontaminated independent implementation of the layer's internals.
