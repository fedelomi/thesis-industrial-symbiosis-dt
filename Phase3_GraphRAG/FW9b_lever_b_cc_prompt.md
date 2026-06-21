# CC mega-prompt — FW9b: multi-template union retrieval (lever b)

Paste this whole file into Claude Code (Opus, MaxEffort). Run it from
`Fasi Applicative/Phase3_GraphRAG`. Read before acting; do not improvise beyond scope.

---

## Mission

Raise the rate of CORRECT answers (semantic EM, the Sonnet cross-model judge verdict)
of the Phase 3 Graph RAG on the OOD benchmark, by fixing the real bottleneck:
single-template retrieval coverage on compound queries. The fix is multi-template
UNION retrieval (top-N=5 candidate templates, union their rows) on the BGE router
backend, plus per-template densification of the context. This is NOT a synthesis-prompt
change and NOT a full query-decomposition engine. Keep it mostly deterministic and
behind a flag so the current pipeline (call it v7) stays runnable for a clean A/B.

## Evidence basis (already established, do not re-derive)

A prior session produced and I re-verified these. Trust them as the starting point:

- A/B split of the 20 OOD `synthesis_fail` items (`results/_ab_split_result.md`):
  2 Caso A (facts already in the single-template context: OOD10, OOD25), 1 densification
  special (OOD03), 17 Caso B (facts live on OTHER templates, context incomplete).
  So ~85% of the floor is retrieval coverage, not synthesis.
- The per-query answers (`results/step_3_11_ood_eval_per_query.csv`) corroborate:
  failing answers literally say "the provided rows contain X but do not include Y".
- Multi-template oracle (`results/_mt_retrieval_sim.json`): unioning the provenance
  templates' rows makes ~15/17 Caso B items answerable at the data level; union row
  counts stay 3-16, under the 40-row cap.
- Router top-N ranking (`results/_router_rank_check.json`): with the BGE backend,
  the top-5 candidate pool contains ALL needed provenance templates for 15/20 items
  (verified by recount: items where in_top5 == n_prov). The 5 residual misses are
  OOD09, OOD13, OOD15, OOD22, OOD38 and overlap genuine KG gaps; do not expect to
  fix those with retrieval.
- Measurement caveat to fix: the eval marks `routing_ok` true if the single routed
  template is anywhere in the 2-4 template provenance set, so compound-query misses
  were mislabeled `synthesis_fail` when the real cause is single-template coverage.

## Hard constraints (non-negotiable)

1. Secrets: never print, log, hardcode or commit API keys or the Neo4j password.
   Read them from `config.py` / env exactly as the existing scripts do. Do not rotate
   or modify any credential.
2. Budget: the ONLY paid step is one judged eval run on the 20-item subset, hard cap
   USD 3. Stop at the cap. Do not run the full 38 until the user has seen the 20-item
   numbers and explicitly approves (then cap USD 5).
3. Determinism: pin seeds and use stable sorts. Routing must be reproducible (see Task 1).
4. Reversibility: put the new behaviour behind a flag (`RETRIEVAL_MODE`, default
   `single` = current v7). v7 must still run unchanged. Never overwrite frozen result
   files; write new timestamped ones.
5. No invented numbers. Report whatever the run produces. Any number that will enter
   the thesis stays provisional until the user verifies it. Draft thesis text in a
   separate file, do NOT edit the thesis .tex/chapters directly.
6. Typography for any thesis-bound text: no spaced em-dash (use ":" "," or parentheses),
   no Oxford comma ("X, Y and Z"), English prose.
7. Use the existing file-write style of the repo. Prefer small surgical edits. After
   each edit, re-read the changed region to confirm it is intact.

## Task 0 — read and confirm signatures (no spend)

Read: `step_3_4_evaluation.py` (`route_question`, `cypher_rows_to_context`,
`KEYWORD_ROUTING`), `templates.py` (`CYPHER_TEMPLATES`), `prompt5_retrieval.py`
(semantic router + backend selection), `step_3_11_ood_eval.py` (OOD harness),
`step_3_9_llm_judge.py` (judge), `config.py`, and the script that generated
`results/_router_rank_check.json` (the multi-candidate pool builder = keyword
candidates + semantic top-K). Confirm the real function names and the BGE backend
switch (the split notes `SEMANTIC_ROUTER_BACKEND=st`). If a `route_question_multi`
already exists, reuse it; otherwise extract the pool logic from the rank-check script
into one deterministic helper. Write a 10-line note `FW9b_signatures.md` listing what
you found, then proceed.

## Task 1 — pin routing determinism (no spend, prerequisite)

The split observed that re-routing the same 20 questions yields a different template
for 11/20 (semantic-fallback non-determinism). This must be fixed first or the A/B
delta is noise. Find the source (likely tie handling or unseeded ops in the semantic
router), make candidate ranking deterministic: stable sort by (score desc, template_id
asc), fixed embedding seed, no set-iteration order leaking into output. Add a test that
calls the router 5 times over all 38 OOD questions and asserts identical ordered output
each time. Gate: test passes.

## Task 2 — multi-template union retrieval (no spend)

Add `RETRIEVAL_MODE` (`single` default, `multi` new). In `multi`:
1. Build the top-N=5 candidate pool (keyword candidates first, then BGE semantic
   top-K), deterministic order, on the BGE backend.
2. Run each candidate template's Cypher (cheap local reads). Union the rows, dedup by
   a stable row signature, preserve order by candidate rank, enforce the 40-row cap.
3. Pass the unioned rows to densification (Task 3) then to the existing synthesis +
   neuro-symbolic check unchanged. The checker must keep verifying against the KG
   canonical facts, NOT against the densified prose.
Keep `single` byte-for-byte equivalent to today so v7 is preserved.

## Task 3 — per-template densification (no spend)

Replace the `k: v | k: v` context with flat declarative sentences, one per row, via a
per-template (or generic-plus-special-case) verbalizer. Rules: keep every anchor token
verbatim (node/business ids, article numbers, degrees C, EUR/MWh, kW, percentages), do
NOT inject any fact not in the row, with ONE controlled exception: templates that are
intrinsically about a named standard may inject that standard's constant name (this is
the OOD03 fix, e.g. ISO 50001 templates may state "ISO 50001:2018"). Output must be
deterministic. Densification applies in both `single` and `multi` so the 2 Caso A items
(OOD10, OOD25) also benefit.

## Task 4 — cross-union pruning, query-type-aware (no spend)

Optional noise control across the unioned rows. Prune rows irrelevant to the question
ONLY for lookup-type queries; NEVER prune for count or aggregation or compare queries
(they need all rows). Since the oracle showed unions stay under the cap, default to
"prune only if union exceeds a noise threshold". Deterministic. Add a test: a count or
compare query keeps all rows; a single-fact lookup prunes; output stable.

## Task 5 — TDD (no spend, write tests alongside Tasks 1-4)

In `tests/`, no API, Neo4j read-only or mocked:
- router determinism (Task 1).
- union + dedup: no duplicate rows, cap respected, stable order.
- densifier: anchors preserved verbatim, no extra facts beyond the controlled standard
  constant, deterministic.
- pruning: count/compare keep all, lookup prunes, deterministic.
- free oracle integration: for OOD10 and OOD25 (Caso A) and for 3 Caso B items whose
  BGE top-5 covers provenance (e.g. OOD18, OOD27, OOD30), assert the unioned+densified
  context CONTAINS the ground-truth anchor strings. This is the free proof the fix works
  at the data level, before any spend.

## Task 6 — free gates (must ALL pass before any paid call)

- `python check_routing.py` stays green on existing targets (run it).
- All Task 5 tests pass.
- Re-confirm the data-level ceiling: with `multi` retrieval, the unioned+densified
  context surfaces the GT anchors for at least the ~13-15/20 the oracle predicted.
  Write `results/_fw9b_oracle_recheck.json`.
- Dry-run the full `multi` pipeline on 5 queries with the LLM call disabled
  (context-only / `llm=None`): no crashes, well-formed contexts.
If any gate fails, stop and report. Do not spend.

## Task 7 — one paid run, capped, staged (the ONLY spend)

Run the OOD eval on the 20 `synthesis_fail` subset, `RETRIEVAL_MODE=multi`, BGE backend,
Haiku synthesizer, Sonnet cross-model judge for semantic EM. Hard cap USD 3, stop at cap.
Write results to a NEW timestamped file (do not overwrite v6/v7). Then STOP and report to
the user:
- per-item: v7 em_semantic vs v8 em_semantic, and the new context's GT-anchor presence.
- aggregate: semantic EM v7 vs v8 on the 20, with the gains and any regressions named.
Do NOT run the full 38 and do NOT write thesis text until the user approves the 20-item
result.

## Task 8 — A/B rigor and honest framing (after user approves)

- Primary comparison: v7 (frozen baseline) vs v8 (BGE + union + densification) on the
  same items, same judge. Be explicit that v8 bundles two changes (backend and union);
  if budget allows one extra arm, add BGE-single to isolate the union effect, else state
  the confound honestly.
- Bootstrap BCa CI on per-arm semantic EM and on the delta. On N=20 the CI is wide:
  say so. Full 38 only with user OK (cap USD 5) for the headline.

## Task 9 — thesis narrative draft (no thesis-file edits)

Write `FW9b_cap6_2_draft.md` (do not touch the real chapter). Reframe honestly:
the "synthesis floor" was partly a measurement artifact (`routing_ok` true when the
single routed template was anywhere in the 2-4 provenance set, so compound-query
coverage misses were mislabeled synthesis); the true floor was single-template retrieval
coverage; multi-template union retrieval addresses it, with the measured before/after.
Apply the typography rules. Cite the evidence artifacts. Mark every number provisional
pending the user's verification.

## Deliverables

Code behind `RETRIEVAL_MODE`, deterministic router, densifier, pruner, tests, the free
gate artifacts, one capped 20-item judged run (timestamped), a v7-vs-v8 report, and the
Cap 6.2 draft. A short `FW9b_RESULT.md` summarizing what passed, the 20-item numbers,
the residual 5 KG-gap items, and the recommended next step.

## Anti-anchor on numbers

Canonical baselines for orientation only, do not target them: in-distribution semantic
EM about 41% frozen (higher in recent fixes), OOD semantic about 35-36%, compliance-
multi-hop about 0.22. The oracle suggests union makes ~15/17 Caso B answerable at the
data level and BGE top-5 covers 15/20, but realized lift depends on ranking and synthesis
and MUST be measured. Report the truth even if it disappoints, and flag anything that
contradicts the evidence basis above.
