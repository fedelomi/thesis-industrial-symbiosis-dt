# Phase 3 Blind Reconstruction: Institutional-LLM Layer (Strato 2)

**Architecture design document. Independent blind reconstruction.**
Author: Coding Agent (blind-design protocol). Date: 2026-06-03.
Scope: redesign Layer 2 (the regulatory-reasoning layer) of the thesis framework
from the framework specification and the Phase 1/2/4 interfaces only, without
reading the existing Phase 3 implementation. See `READING_LOG.md` for the
anti-anchor audit trail and the leaks that were encountered and routed around.

---

## 1. The problem, restated architecture-neutrally

Layer 2 sits between the Physical-DT layer (Phase 1, which exports a thermal
profile) and the IS-Match scorer (Phase 2) and the bilateral negotiation (Phase 4).
The framework specification (Chapter 3, Table 3.1) assigns Layer 2 two contractual
outputs and one research claim:

1. **A per-tier regulatory-friction vector `ΔT_c`** (one value for each manufacturing
   temperature band: LowT 60 C, MidT 90 C, HighT 130 C), expressed in the normalized
   `[0,1]` unit of the IS-Match Score, that enters Eq. (3.3)
   `IS-Match = β·RI + γ·Exergy − δ·ΔTC` as the `δ`-weighted penalty. The Sobol
   analysis of Section 3.3.4 shows `δ` is the dominant variance driver (`S_δ=0.870`),
   so this vector drives the downstream ranking and the Phase 4 reward.
2. **A compliance gate**: a function from a scenario tuple
   `(DC scale, manufacturing process, waste-heat supply temperature, country)` to a
   verdict `{compliant, non_compliant, conditional}` plus the list of triggered
   regulatory articles and the applicable threshold values. The specification states
   the gate must be **deterministic and reproducible across runs**.
3. **RQ3**: a structured retrieval-and-reasoning layer over the regulatory corpus can
   answer cross-firm compliance queries with measurable accuracy and groundedness
   where a vector-only RAG baseline fails.

The corpus is fixed by the framework: 29 documents in `raw/F` spanning the EU EED
2023/1791, the Delegated Regulation 2024/1364 KPI rules, the Italian White
Certificates (TEE) decrees, the Danish Heat Supply Act and Bilag 13, ISO 50001,
ASHRAE 90.4 and TC9.9, and the JRC FDM BREF process-temperature catalogue. The
architecture of the layer is the open design choice.

## 2. The decisive constraint and the resulting choice

The single constraint that dominates the design is the requirement that the gate
and the `ΔT_c` derivation be **deterministic, reproducible and auditable**, with any
probabilistic LLM call "wrapped in a verification layer that can be audited". A
compliance verdict that changes between two runs on the same input, or a `ΔT_c`
value that cannot be traced to a specific clause, is unusable as a contractual input
to a reinforcement-learning reward (Phase 4 trains for tens of thousands of steps
against `ΔT_c` and must see a stable signal) and indefensible in a thesis whose
entire Chapter 6 narrative rests on reproducibility.

A decision produced directly by an LLM (free-text generation, or an LLM that emits a
database query) is intrinsically non-deterministic: sampling temperature, model
updates, retrieval drift and prompt sensitivity all move the output. Therefore the
**verdict and the `ΔT_c` value must be produced by a deterministic symbolic engine**,
not by a language model. The language model is confined to the one task it is
uniquely good at and where errors are recoverable: turning unstructured regulatory
prose into structured, typed facts, performed **offline once**, cached, and validated
against the source clause before it is trusted.

This is the **neuro-symbolic** pattern that the literature review documents
independently of any retrieval architecture (NREL, Buster et al., Section 2.4.5):
"the LLM extracts structured facts and a symbolic system validates and reasons,
providing explainable, auditable logic". I adopt it as the spine of the layer.

### Chosen architecture: Neuro-Symbolic Regulatory Reasoner (NSRR)

```
            OFFLINE (once, audited)                RUNTIME (deterministic)
  ┌───────────────────────────────┐        ┌──────────────────────────────────┐
  │ 29 policy PDFs (raw/F)         │        │ Scenario tuple                    │
  │   │ LLM-assisted extraction    │        │ (scale, process, T_supply, ctry)  │
  │   ▼  (schema-constrained,      │        │            │                      │
  │ Typed Regulatory Facts         │──load─▶│            ▼                      │
  │  {article, threshold, trigger, │        │  Deterministic forward-chaining   │
  │   applicability, provenance}   │        │  RULE ENGINE  (compliance_gate)   │
  │   │ clause-grounding check     │        │     │  verdict + articles + thr   │
  │   ▼                            │        │     ▼                             │
  │ Regulatory Knowledge Base (KB) │        │  Symbolic ΔT_c burden estimator   │
  └───────────────────────────────┘        │     → reduction_factors[tier]     │
            │                               └──────────────────────────────────┘
            │ (open-ended cross-firm QA, RQ3 benchmark)
            ▼
   Structured retriever over KB  ──►  grounded answerer  ──►  symbolic verifier
                                       (numeric claims checked vs KB)
                          vs. baseline: vector-only RAG over raw chunks
```

The KB is a typed store of `RegFact` objects (article reference, topic, requirement
text, numeric thresholds with units and the obligation each gates, IF-THEN
compliance triggers, jurisdiction, and a provenance pointer back to the source
document). It is produced by an offline LLM extraction pass (executed for this
reconstruction as a 29-way parallel extraction with a JSON-schema constraint, see
`data/corpus_facts.json`), then loaded deterministically at runtime. Nothing in the
gate or the `ΔT_c` path calls a model at runtime, so both are bit-for-bit
reproducible.

## 3. Why this over the alternatives

### Alternatives considered and rejected

**(A) Vector-only RAG** (sentence-transformer or TF-IDF embeddings, FAISS index, LLM
answer generation). Rejected as the *decision* mechanism: the verdict would be a
generated string, non-deterministic and prone to fabricating numeric thresholds; it
loses the relational structure (an Article-26 obligation depends on a scale
threshold, an exemption clause and a district-heating-efficiency definition that
live in different parts of the text). It is, however, exactly the baseline that RQ3
requires me to beat, so I keep it as the **evaluation baseline** rather than
discarding it.

**(B) Knowledge graph plus LLM-generated graph queries** (e.g. a property graph with
an LLM translating questions into graph-query language). This is the most serious
contender and I treated it as such. It captures multi-hop structure well. I rejected
it for the *gate* for three reasons. First, an LLM that writes the query reintroduces
non-determinism at exactly the point the constraint forbids it. Second, a graph
database is heavyweight infrastructure for a corpus of 29 documents and two
jurisdictions whose article structure is shallow and enumerable; the multi-hop value
of a graph is recoverable by forward-chaining over a typed fact base without the
query-generation nondeterminism or the operational cost of a graph server. Third, it
optimizes for open-ended traversal, whereas the contractual outputs are a small,
fixed set of decisions (scale-vs-Article-26, efficiency thresholds, upgrade
feasibility) that a rule engine expresses more transparently and audits more easily.

**(C) Hand-coded rule table** (literal if-then thresholds typed in by the developer).
Rejected because the specification requires `ΔT_c` to be "derived from the corpus,
not hardcoded", and a hand table has no provenance to clauses, does not survive a
corpus update, and cannot be cross-checked against the source text. The NSRR keeps
the *engine* deterministic but feeds it facts that are extracted from and traced to
the corpus.

**(D) Fine-tuned LLM compliance classifier.** Rejected: no labelled training set
exists, the output is non-deterministic and opaque, and a black-box classifier
cannot list the triggered articles or expose its reasoning, failing both the
auditability constraint and the explanatory requirement.

The NSRR is the only option that simultaneously satisfies determinism (symbolic
engine), corpus-derivation (LLM extraction with provenance), auditability (every
verdict traces to rules and clauses) and the "LLM wrapped in verification" mandate
(extraction is validated against the source clause; runtime is model-free).

## 4. Contractual outputs to Phase 4

The interfaces were read from the allowed Phase 2 and Phase 4 code, not from Phase 3.

**Output 1, `ΔT_c` per tier.** I emit `data/step_3_4_delta_tc.json` with the schema
that the Phase 2 consumer `step_2_4_delta_tc_calibration_lc.py` expects, namely a
top-level `"reduction_factors"` mapping keyed by `"LowT_60C"`, `"MidT_90C"`,
`"HighT_130C"`, each a float in `[0.1, 0.3]`. Phase 2 applies
`ΔTC_post = ΔTC_baseline·(1 − reduction_factor)` and re-runs the IS-Match calibration
loop to convergence; Phase 4 then loads the resulting `is_match_score` (clipped to
`[0,1]`). The same JSON also carries `"delta_tc_norm"`, the absolute corpus-derived
friction per tier in IS-Match units (the mission's literal "vector of three
coefficients"), plus a `"provenance"` block listing the corpus triggers that drove
each number. `ΔT_c` is computed by a deterministic burden score: for each
`(tier, country)` the engine counts and weights the regulatory obligations that the
tier activates (upgrade-technology complexity, Article-26 reach, reporting and audit
duties, ISO 50001 EMS exposure) and subtracts the available mitigation (Italian TEE
incentive value, Danish municipal planning support). The `reduction_factor` is the
informational share of that friction, the fraction the institutional-LLM layer
removes by resolving the Gap-3 information asymmetry (knowing which articles apply
and how to comply), clipped to the `[0.1, 0.3]` band the framework reserves for
RAG-driven barrier reduction.

**Output 2, the compliance gate.** `compliance_gate(scenario) -> Verdict` is a pure
function. `Verdict` is a dataclass with `status` (`compliant|non_compliant|conditional`),
`triggered_articles` (list of `{ref, doc_id, requirement}`), `thresholds` (dict of
the numeric limits that fired, with units), and a human-readable `rationale`. It is
JSON-serializable and deterministic. The gate composes with the existing Phase 4
`ShieldingLayer` without modifying it: a `non_compliant` verdict can be used as an
additional admissibility mask on a scenario, exactly the pattern the shielding layer
already uses for its physical and normative rules, demonstrated in
`integration_test.py`.

**Value ranges (verified against Phase 4).** `is_match_score` and `delta_tc_norm` are
floats in `[0,1]`; `reduction_factor` in `[0.1,0.3]`; the DC scales map to IT power
{Edge 475 kW, Mid 3104 kW, Hyperscale 24500 kW} and tiers to `T_req` {60, 90, 130} C
exactly as `config/scenarios.py` defines them.

## 5. Evaluation methodology and baseline comparison

RQ3 is defended quantitatively on a **document-grounded benchmark of at least 100
questions** (`data/benchmark.jsonl`, protocol in `benchmark_design.md`). Following
the document-grounded principle (the ground truth is the corpus clause itself, not a
subjective annotator), questions fall into three categories: (i) **factual or
threshold lookup** (objective, deterministic ground truth, e.g. the Article-12
reporting threshold), (ii) **multi-hop compliance reasoning** (chains across scale,
upgrade and jurisdiction, e.g. whether a 3.2 MW DC supplying 90 C process heat in
Italy triggers Article 26 and what exemption applies), and (iii) **comparative
screening** (technology or jurisdiction comparisons). Category (i) and most of (ii)
have a deterministic gold answer (a number, an article reference, or a verdict);
genuinely interpretive items are scored with a cross-model LLM judge (Sonnet-class)
whose outputs are cached and whose numeric claims are re-verified by the symbolic
layer, with the keyword-exact-match-versus-semantic gap reported transparently to
expose any matcher overstatement.

The comparison is **NSRR versus the vector-only RAG baseline (alternative B-as-baseline,
option A above)** on the identical benchmark. Both run locally and deterministically:
the baseline uses TF-IDF or dense-embedding cosine retrieval over raw document chunks
and an extractive answer, mine uses structured retrieval over the typed KB plus the
symbolic reasoner. The headline result is the accuracy gap and its decomposition by
category. The expectation, consistent with the gap structure of the literature
review, is parity on simple factual lookup and a large NSRR advantage on the
multi-hop and threshold questions where vector similarity cannot assemble the
relational chain and tends to fabricate numbers. Robustness is reported with the
framework's own families (Section 3.6.4): bootstrap confidence intervals on the
accuracy gap, a component leave-one-out (ablating the symbolic engine to isolate its
contribution) and a paraphrase-perturbation stability check.

## 6. Module layout (naming follows the per-phase convention)

| File | Role |
|------|------|
| `step_3_0_config.py` | constants, dataclasses (`Scenario`, `Verdict`, `RegFact`, `DeltaTcResult`), scale/tier maps, paths, logging |
| `step_3_1_ingest.py` | load and validate `corpus_facts.json` into the typed Regulatory Knowledge Base |
| `step_3_2_retrieval.py` | structured KB retriever and the vector-only RAG baseline (TF-IDF default, optional dense) |
| `step_3_3_compliance_gate.py` | deterministic forward-chaining rule engine, `compliance_gate()` |
| `step_3_4_delta_tc.py` | symbolic `ΔT_c` burden estimator, writes the Phase 2 drop-in JSON |
| `step_3_5_answerer.py` | neuro-symbolic grounded answerer and the baseline answerer |
| `step_3_6_evaluate.py` | benchmark runner, NSRR vs baseline, mockable LLM judge |
| `step_3_7_robustness.py` | bootstrap CI, component LOO, paraphrase stability |
| `run_phase_3_blind.py` | orchestrator |
| `tests/` | unit tests (determinism, verdict correctness, range/monotonicity, schema, retrieval) |
| `integration_test.py` | end-to-end Phase 3 to Phase 4 contract test |

## 7. Reproducibility and cost

The gate and `ΔT_c` are model-free at runtime and therefore deterministic by
construction. The only optional API use is the Sonnet judge on the interpretive
benchmark subset; it is cached, mockable and off by default, so the test suite is
hermetic and the live API cost stays far below the 5 USD design budget (the offline
extraction was already performed). All randomness in the evaluation (bootstrap
resampling, any embedding fallback) is seeded.

## 8. Known limitations and honest scope

The corpus is restricted to two jurisdictions (Italy and Denmark, following the
framework's Decision D3), so `ΔT_c` does not generalize outside the EU without
re-extracting a local corpus. The offline extraction is an LLM step and can carry
extraction error; this is mitigated by the schema constraint and the clause-grounding
check, and it is the right place to spend the irreducible model-uncertainty budget
because errors there are caught once and audited, rather than at every runtime query.
The `ΔT_c` weights are fixed and documented rather than learned, consistent with the
framework's fixed-weight philosophy for Eq. (3.3); learning them would require a
labelled friction dataset that does not exist.
