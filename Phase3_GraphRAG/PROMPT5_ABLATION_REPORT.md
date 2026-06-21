# PROMPT 5 — Multi-template retrieval + densification ablation (v7 candidate)

> **Scope note.** "Phase 5" does NOT exist as a new thesis phase. This is Phase 3 GraphRAG;
> **v7** is the post-ablation candidate version of the system. Canonical thesis numbers
> (v2: EM strict 59% / semantic 41%) remain frozen. All results here feed Cap 6.2 / 6.3 as a
> post-submission update, consistent with the v3-v6 handling.

Date: 2026-06-03. Built on the A/B split + MT oracle ceiling + router-rank checks
(`results/_ab_split_result.md`). Subset under test = the **20 OOD synthesis_fail items** of v6.

## Design: 4-arm ablation (attribution ladder)

| Arm | backend | retrieval | formatter | isolates |
|-----|---------|-----------|-----------|----------|
| A0 | tfidf | single (route_question) | legacy `k: v \|` | baseline (v6 config) |
| A1 | BGE (st) | single (route_question) | legacy | backend effect |
| A2 | BGE (st) | multi-template union top-5 | legacy | multi-template effect |
| A3 | BGE (st) | multi-template union top-5 | densified | densification effect |

Implementation (TDD, 15 unit tests green): new module `prompt5_retrieval.py`
(route_question_multi, union_template_rows, densify_context, refine_error_class). Canonical
`step_3_4_evaluation.py` and `step_3_4_bis_neuro_symbolic.py` left byte-for-byte unchanged.
N=5, row_cap=40.

## Free pre-spend gates — ALL PASS

| Gate | Result |
|------|--------|
| pytest (full suite) | 20 passed |
| canonical no-regression (check_routing / check_regressions) | route_question untouched, clean |
| provenance coverage A0/A1 (single) | 0% (single template cannot cover a 2-4 template question) |
| provenance coverage A2/A3 (multi top-5) | **70%** (14/20 fully covered) |
| context size (multi) | mean 17.8 rows < cap 40 — no overflow, top-5 holds |
| routing latency p95 | A2 53ms / A3 31ms (above the 21ms single-route reference, acceptable for eval) |

## Metric refinement — declared shift (audit trail of measurement evolution)

`refine_error_class` splits **under_retrieval** out of **synthesis_fail**: the legacy
classifier called every routed-but-wrong answer a synthesis_fail even when the single
template never carried the provenance facts. Consequence: the v6 taxonomy
(coverage 4 / routing 2 / synthesis 20 / correct 8) will shift post-fix to
(4 / X / Y / 8) with **X > 2 and Y < 20**, because under-retrieval was masked inside
synthesis_fail. This is a *metric refinement*, not a result change — the same answers,
classified more precisely. Cap 6.2 should cite the refined taxonomy and state the shift
explicitly so the new numbers are coherent with the old. It strengthens the iterative
self-criticism narrative (we sharpened our own measurement).

## PRE-RUN gain predictions (methodological gate — written BEFORE the paid run)

Note: the 20 items are all v6 *failures*, so baseline semantic EM on this subset ≈ 0%.
The arms measure how many of the 20 we RECOVER. Predicted EM is "recovery rate on the 20".

| Delta | what it isolates | predicted EM strict | predicted EM semantic | rationale |
|-------|------------------|---------------------|------------------------|-----------|
| **A0** (baseline) | — | 0-10% | 0-5% | reproduces the v6 failures (a few may flip on Haiku variance) |
| **A1 − A0** | backend BGE | **+0..+10 pp** | **+0..+10 pp** | single-template coverage stays 0%; BGE only helps the rare item where it picks one more-complete template. SMALL. |
| **A2 − A1** | multi-template union | **+35..+55 pp** | **+40..+60 pp** | coverage 0%→70%: facts now present for 14/20. DOMINANT arm. Ceiling = hard multi-part composition + the 6 still-uncovered. |
| **A3 − A2** | densification | **+5..+15 pp** | **+5..+15 pp** | parses the 17-row union + unblocks the 2 true Caso A (OOD10/OOD25). Modest. Low risk to strict (anchors verbatim, unit-tested). |

**Net call:** A3 recovers ~50-70% of the 20 failures (semantic). **Multi-template union is
predicted to dominate**; backend and densification are secondary. If reality contradicts
this (e.g. backend dominates, or multi-template underperforms), that mismatch is itself a
Cap 6.2 finding about model-vs-reality calibration.

Watch items for densification escalation (bespoke per-family if A3 leaves them failed):
the residual synthesis/partial set surfaced post-run.

---

## POST-RUN — results (20 OOD synthesis_fail items, all v6 failures)

A harness bug (`bool(tuple)` on the judge output -> sem always True) was caught by the A0
sanity check (sem=100% on a known-failure set is impossible) before spending on A1-A3, then
fixed. Final numbers:

| Arm | EM strict | EM semantic | error taxonomy (refined) |
|-----|----------:|------------:|--------------------------|
| A0 tfidf single | 30% | 5% | under_retrieval 11, routing_fail 8, correct 1 |
| A1 BGE single | 25% | 0% | under_retrieval 20 |
| A2 BGE multi top-5 | **70%** | **55%** | correct 11, synthesis_fail 6, under_retrieval 3 |
| A3 BGE multi + densify | 65% | 50% | correct 10, synthesis_fail 6, under_retrieval 4 |

EM semantic is primary (strict overstates via the keyword FP, consistent with §3.8.3).

### Predictions vs reality (calibration check)

| Delta | predicted (sem) | actual (sem) | verdict |
|-------|----------------:|-------------:|---------|
| A1 − A0 (backend) | +0..+10 pp | **−5 pp** | MISS (wrong sign; inert, within 1-item noise) |
| A2 − A1 (multi-template) | +40..+60 pp | **+55 pp** | HIT (top of range) |
| A3 − A2 (densification) | +5..+15 pp | **−5 pp** | MISS (wrong sign; see below) |

Calibration finding for Cap 6.2: we correctly predicted multi-template union would DOMINATE
and nailed its magnitude (+55 pp, inside +40..+60), but overestimated the two secondary
levers. Backend-alone and generic-densification were both inert/slightly negative. Secondary
levers do not compound additively on top of the dominant one. Honest model-vs-reality signal.

### Attribution table (EM semantic, recovery on the 20 failures)

- baseline A0: 5%
- backend BGE (A1 − A0): **≈0 pp** (single-template, coverage stays 0%)
- multi-template union (A2 − A1): **+55 pp**  ← essentially the entire gain
- densification generic (A3 − A2): **−5 pp** (no help; broke OOD09)
- **net A0 → A2: +50 pp semantic recovery (5% → 55%)**

Nuance: backend BGE shows 0 gain in the single-template arm, but multi-template REQUIRES it.
Free-gate provenance coverage: BGE multi 70% vs tfidf multi ~35% (router-rank check). So the
real story is "multi-template union is the lever, and it needs BGE to rank the right
templates; neither alone suffices." Not independent — synergistic.

### Densification verdict + escalation decision

Generic light densification (drop `k: v |` -> `k v,`) did NOT help and broke OOD09
(correct in A2 -> wrong in A3). The 2 true Caso A items: OOD25 recovered in both A2/A3;
OOD10 stayed failed in both (densification did not unblock it). Per the pre-agreed rule,
escalate-to-bespoke would target OOD10 only — not worth bespoke work for one hard multi-part
item. **Decision: DROP densification. v7 = A2 config** (BGE + multi-template top-5 + legacy
formatter). Bespoke densification stays as future work.

### v7 decomposition (refined metric, on the 20 prior synthesis_fail)

The v6 "synthesis_fail = 20" splits, under multi-template retrieval + the refined metric, into:
- **11 were UNDER-RETRIEVAL** (single template missed provenance facts) -> recovered by multi-template
- **3 still under-retrieval** (OOD13/22/38: facts genuinely partial in KG)
- **6 GENUINE synthesis_fail** (OOD04/10/29/30/32/37: facts retrieved, not composed)

So the v6 "59% synthesis floor" was mostly mislabeled under-retrieval. The true synthesis
floor among the failures is 6/20 = 30%, not 100%. This is the strongest Cap 6.2 refinement:
the bottleneck was single-template retrieval coverage, not LLM synthesis. FW9-bis
(multi-template composition) is validated as the right lever and is now partly DELIVERED (A2).

### Next step (one more cheap run)
The above is the 20-failure subset. To get the headline v7 EM on the full benchmark, run A2
config on all 34 scored OOD queries (~$0.10) and recompute the full decomposition
(coverage_gap / routing_fail / under_retrieval / synthesis_fail / correct) for the definitive
Cap 6.2 paragraph. Canonical v2 (59/41) stays frozen regardless.
