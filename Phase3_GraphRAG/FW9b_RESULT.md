# FW9b RESULT (provisional, pending user verification)

Date 2026-06-06. One paid run only, cost USD 0.21 (hard cap USD 3, completed all 20,
not stopped at cap). All numbers below are provisional until the user verifies them.

## What passed (no spend)
- Task 1 routing determinism: 0/38 flips across 5 repeated calls, both the ranked
  score and the multi pool (BGE backend). Test green. Fix: explicit stable sort
  (score descending, template_id ascending) plus a fixed embedding seed and
  single-threaded encode.
- Task 3 densifier: per-row verbalizer preserves every anchor token verbatim, with
  the single controlled exception that an ISO 50001 template may state the constant
  "ISO 50001:2018" (the OOD03 fix). 6 tests green.
- Task 4 pruner: query-type aware, never prunes count, compare or aggregate queries.
  Default threshold raised to the row cap after the live dump showed an aggressive
  threshold dropped Article 23 from the OOD10 union (the union itself was correct).
  Unions stay 10 to 35 rows, so pruning is now effectively a last-resort guard.
  Tests green.
- Task 5 unit tests: 16 green (determinism, union dedup and cap and order, densifier,
  pruner).
- Free data-level check: the live RETRIEVAL_MODE=multi context for the five spot
  items (OOD10, OOD25, OOD18, OOD27, OOD30) contains the ground-truth facts by eye
  (Article 23 and 24 for OOD10, the scenario grid for OOD25, the Danish mandate and
  the Italian certificate values for OOD18, the Delegated Regulation KPIs for OOD27,
  the NECP-2024 and Heat Supply Act for OOD30). This proves the offline 9/20 was the
  broken generic anchor matcher, not the retrieval.

## The 20-item paid run (v7 frozen baseline vs v8 = BGE + union + densification)
- v7 semantic EM: 0/20 = 0.0% (the 20 are the v7 synthesis_fail subset by definition).
- v8 semantic EM: 10/20 = 50.0%. Delta +50.0 pp. Regressions: zero.
- Recovered (v7=0 to v8=1): OOD03, OOD09, OOD12, OOD18, OOD20, OOD21, OOD24, OOD25,
  OOD27, OOD35.
- Residual failures (10): OOD04, OOD10, OOD13, OOD15, OOD22, OOD29, OOD30, OOD32,
  OOD37, OOD38.
- Confound: v8 bundles two changes (the BGE router backend and the multi-template
  union plus densification). A BGE-single arm is needed to isolate the union effect.
  N is 20, so any confidence interval on the delta is wide.

## Residual analysis
The original "synthesis floor" was partly a measurement artifact (routing_ok was
true whenever the single routed template was anywhere in the 2 to 4 template
provenance set, so compound-query coverage misses were mislabeled synthesis). The
run splits the floor empirically:
- About half of the v7 synthesis failures were retrieval coverage, recovered by
  multi-template union (10 of 20).
- The residual is a mix of genuine synthesis misses (the live check confirmed the
  facts are present for OOD10 and OOD30 yet the judge still scored the composed
  multi-hop answer wrong) and harder coverage or KG-gap items.

## Flag against the evidence basis
The evidence basis listed five residual KG-gap items (OOD09, OOD13, OOD15, OOD22,
OOD38). In this run OOD09 RECOVERED under v8 (multi-template union surfaced its
facts), so the KG-gap set was slightly pessimistic. OOD13, OOD15, OOD22 and OOD38
stayed failing, consistent with the prediction.

## Caveat on the gt_anchor_coverage column
The per-query CSV carries a gt_anchor_coverage column from the same crude generic
regex that produced the misleading offline 9/20 (it scores OOD25 at 0.0 although the
facts are present and the judge marks it correct). Do not read it as ground truth;
the Sonnet judge verdict and the by-eye live dump are the reliable signals.

## Outputs (timestamped, frozen files untouched)
- results/_fw9b_v8_per_query_20260606_095849.csv
- data/evaluation_results_graph-rag-ood-v8_20260606_095849.json
- results/_fw9b_oracle_recheck.json (offline heuristic, kept with its caveat)

## Full-38 headline run (provisional, cost USD 0.46, cap USD 5)
Scored 22 non-gap items (16 coverage-gap items excluded, as in the incumbent harness),
same Sonnet judge:
- v7 frozen semantic EM: 27.3% (6/22).
- v8 (BGE + union + densify) semantic EM: 59.1% (13/22).
- HEADLINE delta: +31.8 pp, paired BCa 95% CI [+4.5, +50.0] (excludes zero, wide at n=22).
- Ablation (all BGE): union is the lever (A - B = +27.3 pp); BGE backend/route
  (C - v7 = +9.1 pp); densification is net negative on aggregate (B - C = -4.5 pp,
  one item, within LLM noise) and is kept only as the targeted OOD03 fix; the pruner
  is dormant at the observed union sizes.
- Regression: OOD08 (v7=1 to v8=0). Gained: OOD03, OOD09, OOD18, OOD20, OOD24, OOD27,
  OOD32, OOD35.
- Caveat: the synthesizer and judge are not bit-reproducible at temperature 0;
  3 to 4 items flip between the 20-run and the 38-run, so per-item verdicts and the
  exact aggregate carry run-to-run noise beyond the sampling CI.

## Recommended next step
Pending user verification of these numbers: finalise Cap 6.2 (drafted in
FW9b_cap6_2_draft.md, no chapter edited). Optional: repeat the full-38 a few times to
quantify the judge non-determinism band. No thesis file is edited until verification.
