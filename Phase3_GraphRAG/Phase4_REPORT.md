# PROMPT 4 - KG enrichment + routing fix + OOD v6 measurement

Date: 2026-06-03. No commit, no Modal. Total API spend this phase ~$0.41
(4c dry-run $0.034 + full GT $0.18 + full eval $0.21), budget cap $5.
`benchmark_ood_v1.jsonl` and `benchmark_ood_v1_review.jsonl` (Fede's decisions)
left UNCURATED and untouched; the v6 measurement uses a new
`benchmark_ood_v2.jsonl` (also UNCURATED).

## Fase 4a - KG enrichment (EED Art.11/12)

- `step_3_1f_ingest_eed_audit.py`: merged RegulatoryArticle EED-ART-11 and
  EED-ART-12, linked `(EED-2023-1791)-[:CONTAINS]->(article)`. Idempotent, rollback documented.
- `step_3_0_neo4j_schema.py`: added `constraint_regarticle_eed_article` (18 constraints, PASS).
- Gate green: validation 2 paths; OOD16/21/34 retrieve Art.11/12 via the top-3 union.
- Schema reconciliation: spec said `PART_OF`/`celex_id`; live KG uses
  `CONTAINS`/`id='EED-2023-1791'`. Followed the live schema (added `celex_id`
  for forward-compat) so the nodes are reachable by existing templates.
- LEGAL FLAG: in EED 2023/1791 the audit/EnMS provisions are Art. 11; the OOD16
  query's "Article 12" is likely a generator numbering error. Art. 11 carries
  correct substance; Art. 12 is `status=PROVISIONAL, needs_verification=true`.
  Verify the CELEX text of Art. 12 before citation.
- OOD37 decision: "sequence of legal steps" is procedural/temporal, a different
  paradigm from the entity-relationship KG. RECOMMEND out-of-scope (Cap 6.3
  known limitation), not a one-off ProcedureSequence node.

## Fase 4b - routing fixes (Standards + actor)

- `step_3_4_evaluation.py`: GENERIC_STANDARD keywords (+`ashrae 90.4`, `tc 9.9`,
  `digital twin framework`); GENERIC_ACTOR keywords (+`public body`,
  `responsible for issuing`, `managing certificati`). `semantic_router.py`:
  strengthened both anchors. v5 architecture (Stage 1 conflict-aware + Stage 2
  BGE + prior-aware 0.05) unchanged.
- Gate green: canonical **21/21**, routing-diff **0 unexpected** vs v5, latency
  **p50 14 / p95 20 ms** (invariant), pytest 5 passed.
- OOD routing under v6: OOD05/07/26 -> GENERIC_STANDARD, OOD28 -> GENERIC_ACTOR
  (6/8 retrieval_miss resolved); OOD13/OOD23 stay compound.

## Correction to the v5 provisional numbers (important)

The v5 provisional split (`retrieval_miss 8`, revised `routing_fail 9`) was
INFLATED: `build_ood_ground_truth.py` built the GT union from the raw semantic
top-3 (`router.route`), NOT the production `route_question`, so OOD05/07/19/14
were flagged coverage_gap as a GT-construction artifact even though production
routed them correctly. 4c fixes the builder (union now includes
`route_question`).

## Fase 4c - v6 measurement (benchmark_ood_v2.jsonl)

GT rebuilt with the fixed builder + enriched KG; coverage gaps fell **16 -> 4**.

| metric | v5 (benchmark_ood_v1) | v6 (benchmark_ood_v2) |
|--------|----------------------|-----------------------|
| scored / total | 22 / 38 | **34 / 38** |
| coverage_gap | 16 | **4** |
| routing_fail | 9 (revised) | **2** |
| synthesis_fail | 13 | **20** |
| correct | 8 | 12 |
| EM strict | 27.3% | 44.1% |
| EM semantic | 36.4% | **35.3%** |
| LCR (mean) | 0.842 | 0.868 |

Per-class delta vs the expected targets:
- coverage_gap 16 -> 4: **as expected** (enrichment + builder fix). Of the 4
  residual: OOD19 is a false gap from the `step_3_4_bis` entity regex (ns=0;
  not touched per constraint), OOD14 is attribute-granularity (Art. 24 criteria
  detail not a queryable field), OOD11/OOD36 are compound multi-hop.
- routing_fail 9 -> 2: **as expected** (Standards/GSE fix). The 2 residual are
  OOD16/OOD34: they route to `P4_incentives_it_whr` (strong "Italian incentive"
  keywords) and miss the now-present Art.11/12 in ALL_REGULATORY_ARTICLES. They
  are compound (audit obligation + incentive) and cannot be served by one template.
- synthesis_fail 13 -> 20: **grew, as expected when gaps are unmasked**. The 12
  newly-scored ex-gap queries are mostly compound; the single-template pipeline
  retrieves valid entities (LCR 0.87) but cannot compose the multi-part answer.

### The headline finding for Cap 6.2

The 4a+4b fixes worked at the mechanism level (coverage 16->4, routing 9->2), but
**OOD semantic accuracy did not improve (36.4% -> 35.3%)**. Reducing coverage and
routing failures simply exposed the real ceiling: **synthesis on compound
multi-hop queries** (20/34 failures), where one routed template returns correct
entities but cannot answer a question that spans regulation + scenario + standard
+ incentive. The OOD bottleneck is compositional reasoning, not retrieval. This
is the strongest argument for FW9-bis: multi-template composition / answer
synthesis, not more routing or KG nodes.

By category (semantic): parameter-lookup 0.71, regulatory-traversal 0.30,
cross-comparison 0.25, compliance-multi-hop 0.22 - accuracy collapses exactly
where compositional depth rises.

### Structural ablation (free attribution, no extra spend)

A paid fix-4a-only vs fix-4b-only ablation was skipped to respect budget; the
contributions are attributable structurally:
- 4a (EED enrichment) made Art. 11/12 retrievable; it helps only where routing
  reaches ALL_REGULATORY_ARTICLES (OOD21), and NOT OOD16/34 which route to P4
  (so 4a's measurable EM contribution is small without a composition step).
- 4b (Standards/GSE routing) + the builder union fix account for most of the
  16->4 coverage-gap reduction (OOD05/07/26/28 routed correctly; OOD13/14/19/23
  gained GT via the route_question union).
- Net: the gains are real for coverage/routing classes but do not move semantic
  EM, confirming the synthesis ceiling.

## Caveats / honesty

- v5 vs v6 EM is on different denominators (22 vs 34 scored) and different GT
  (v2 GT is concise, built post-enrichment), so the EM strict jump (27->44) is
  not a like-for-like accuracy gain; the trustworthy signal is semantic EM
  (flat) and the class shifts. EM strict > semantic again confirms the
  keyword matcher overstates (step_3_9 finding).
- benchmark_ood_v2.jsonl is UNCURATED (LLM-built GT, deterministic entity
  validation only). Numbers are PRELIMINARY.
- OOD19 gap is a measurement artifact in the untouched neuro-symbolic regex, not
  a real KG gap.
- Art. 12 node is PROVISIONAL pending legal verification.

## Out of scope for PROMPT 4 (candidate next PROMPT)

The 4 ambiguous compound gaps (OOD11/15/31/36) and, more broadly, the 20
synthesis failures point to FW9-bis: a multi-template composition / answer-merge
stage. Not touched here.

## Files

- new: `step_3_1f_ingest_eed_audit.py`, `data/benchmarks/benchmark_ood_v2.jsonl`
- edited: `step_3_0_neo4j_schema.py` (constraint), `step_3_4_evaluation.py`
  (keywords), `semantic_router.py` (anchors), `build_ood_ground_truth.py`
  (route_question union + `--out`), `step_3_11_ood_eval.py` (`--version`)
- results: `results/step_3_11_ood_eval_per_query.csv`,
  `results/step_3_11_ood_eval_summary.csv`
- untouched: `benchmark_ood_v1.jsonl`, `benchmark_ood_v1_review.jsonl`,
  `step_3_4_bis_neuro_symbolic.py`
