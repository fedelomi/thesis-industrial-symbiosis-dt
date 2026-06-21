# PROMPT 5 — v7 full OOD benchmark (multi-template union), citable for Cap 6.2

> **Scope.** Phase 3 GraphRAG, v7 candidate = A2 config (BGE backend + multi-template top-5
> union, legacy formatter, densification dropped). Canonical thesis v2 (59/41) FROZEN. These
> are post-submission-update numbers for Cap 6.2 / 6.3. No commit. Benchmark files unchanged.

Date: 2026-06-03. Run: A2 on all 38 OOD v2 queries (34 scored, 4 KG_COVERAGE_GAP), CURATED.
Cost ~$0.10. Gates PASS pre-spend (pytest 20/20, loader dry-run, warm route p95 27ms < 200ms,
canonical route_question untouched).

## Headline v7 (34 scored)

| Metric | v6 | **v7** | delta | attribution |
|--------|----|-------|-------|-------------|
| EM semantic (Sonnet judge) | 35.3% | **61.8%** | **+26.5 pp** | multi-template union |
| EM strict (keyword) | 44% | **76.5%** | +32.5 pp | multi-template union |
| LCR | 0.842 | **1.0** | +0.16 | neuro-symbolic intact |

Semantic is primary (strict overstates via keyword FP, §3.8.3). The +26.5 pp semantic jump is
attributed to multi-template union: the dominant lever confirmed by the 4-arm ablation
(backend-alone +0, multi-template +55 pp on the failure subset, densification dropped).

## v7 decomposition (refined metric) — Cowork prediction vs reality

| Class | Cowork prediction | **reality** | verdict |
|-------|------------------:|------------:|---------|
| coverage_gap | 4 | **4** | HIT |
| routing_fail | 2 | **0** | better (union absorbed routing misses) |
| under_retrieval | 3-5 | **4** | HIT |
| synthesis_fail (genuine) | 6 | **9** | MISS (+3 harder than expected) |
| correct | 17-19 | **21** | better (+2 over range) |

Well-calibrated on coverage_gap and under_retrieval; optimistic on routing (got 0, better);
the genuine synthesis floor is higher than projected (9/34 = **26.5%**, vs the 18% projected
from the 20-subset). Net the system beat the correct-count prediction (21 vs 17-19).

## Per-category EM semantic — the key finding

| Category | n | EM semantic | EM strict | reading |
|----------|--:|------------:|----------:|---------|
| parameter-lookup | 7 | **0.857** | 1.00 | multi-template lifts it |
| cross-comparison | 8 | **0.875** | 0.625 | multi-template lifts it |
| regulatory-traversal | 10 | **0.600** | 0.70 | lifted, 4 synthesis remain |
| compliance-multi-hop | 9 | **0.222** | 0.778 | **UNCHANGED from v6 0.22** |

**Answer to the open question: compliance-multi-hop did NOT rise** (0.222 vs v6 0.22). Multi-
template union lifted every category EXCEPT compliance-multi-hop. That category carries 4 of
the 9 genuine synthesis_fail plus 3 of the 4 under_retrieval — it is hard on both axes. This
is the **true compound-synthesis floor**: questions where the facts are retrieved (LCR 1.0)
but composing a correct multi-part compliance answer exceeds the single-pass Haiku synthesis.
This is the FW9-ter target, distinct from the retrieval-coverage problem multi-template solved.

Genuine synthesis_fail ids: OOD04, OOD08, OOD10, OOD12, OOD16, OOD29, OOD30, OOD32, OOD37
(4 compliance-multi-hop, 4 regulatory-traversal, 1 parameter-lookup).

## Honest caveat: 1 regression from union noise

Cross-check vs v6 per-query: **9 recovered** (v6 synthesis_fail -> v7 correct: OOD03/09/18/20/
21/24/25/27/35) and **1 REGRESSION: OOD08** (v6 correct -> v7 synthesis_fail). OOD08 broke
because the top-5 union added irrelevant rows that distracted Haiku where the single template
had sufficed. This is the precision cost of ungated union and is direct evidence that the
union-pruning lever (C4, deferred from v7) has a residual job. Net effect is strongly positive
(+9 / −1), but the regression is real and must be stated.

## Metric refinement shift (declared)

The v6 taxonomy (coverage 4 / routing 2 / synthesis 20 / correct 8 on 34) is re-measured under
`refine_error_class`, which splits **under_retrieval** out of synthesis_fail. v7 taxonomy:
coverage_gap 4 / routing_fail 0 / under_retrieval 4 / synthesis_fail 9 / correct 21. This is a
*metric refinement* (sharper measurement of the same answers), not a silent change. Cap 6.2
should cite the refined taxonomy and state the shift, consistent with the iterative
self-criticism narrative. benchmark_ood_v2 stays CURATED and researcher-confirmed; the
reclassification is on the error axis only, not on the ground truth.

## Cap 6.3 Future Work reframe

- **FW1 (semantic routing) — DELIVERED.** BGE multi-template union is implemented (v7); routing
  is no longer the OOD bottleneck.
- **FW9-bis (bespoke densification) — limited scope.** Generic densification was inert/negative
  in the ablation; bespoke would target only OOD10. Low priority.
- **FW9-ter (compound-synthesis floor) — NEW, High.** compliance-multi-hop 22% is a genuine
  single-pass synthesis floor (~26% of scored queries). Target: query decomposition or a
  compose-then-verify second pass, plus union-pruning to remove the OOD08-style noise regression.

## Caveats
- v7 numbers are not like-for-like with v6 on denominators only if GT changed; here GT is the
  same benchmark_ood_v2 (CURATED), so v6->v7 IS like-for-like. Reliable signal.
- EM strict 76.5 > semantic 61.8 confirms keyword overstatement (consistent with §3.8.3).
- n=34 scored; per-category counts 7-10 are small, category rates indicative not precise.
- Canonical v2 (59/41) unchanged and remains the thesis number; v7 is a post-submission update.
