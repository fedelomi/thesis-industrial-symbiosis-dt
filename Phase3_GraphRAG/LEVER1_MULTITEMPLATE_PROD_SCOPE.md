# Lever 1 — Multi-template Retrieval Promotion to Production Path

**Scope document** for promoting the FW9-ter multi-template retrieval probe (`prompt5_retrieval.py` + `step_3_20_multitemplate_indist.py`) from exploratory addendum to canonical production hot path in `step_3_2_graph_rag_pipeline.py`.

**Goal:** integrate the v7 multi-template union retrieval into the production Graph-RAG pipeline so that any downstream consumer (Phase 2 ΔT_c calibration, Phase 4 reward, future operator-interface FW1-bis) benefits from the +29 pp third-judge semantic gain measured on the OOD held-out benchmark, without breaking the canonical EM strict 0.59 / EM semantic 0.41 numbers that anchor Chapter 5.

**Constraint:** the canonical numbers of Chapter 5 are frozen. The refactor must therefore add the v7 path as an **opt-in** mode while keeping the single-template canonical path as the default for any reproducibility-bound consumer.

---

## Design

### Option A (recommended) — flag-based dual path

Modify `step_3_2_graph_rag_pipeline.py` to accept a `--pipeline-mode {canonical, v7}` CLI flag. When `canonical` (default), the pipeline behaves bit-identically to the current implementation. When `v7`, the routing engine uses `route_question_multi(n=5)` + `union_template_rows` + `densify_context_v2` from `prompt5_retrieval.py`.

Pro: backward compatible, canonical reproducibility preserved, both paths coexist for direct comparison.
Con: slightly more conditionals in the pipeline code.

### Option B — separate production script

Create `step_3_2_graph_rag_pipeline_v7.py` as a new script that imports the same Neo4j driver and benchmark loader but uses the v7 routing throughout. Canonical script untouched.

Pro: cleaner separation, easier to reason about each path independently.
Con: code duplication, two scripts to maintain, harder to flip the default later.

**Recommendation: Option A.** The conditional is small (one function call swap), and a single canonical entry point is easier for downstream consumers to import.

---

## Implementation steps

### Step 1 — Add CLI flag to `step_3_2_graph_rag_pipeline.py`

In `parse_args()` (line 269), add:

```python
parser.add_argument(
    "--pipeline-mode",
    choices=["canonical", "v7"],
    default="canonical",
    help="canonical: single-template top-1 routing (frozen Chapter 5 baseline). "
         "v7: multi-template top-5 union + densification + pruning (FW9-ter probe, "
         "+29 pp third-judge semantic on OOD held-out). Default canonical for "
         "reproducibility; pass v7 for production deployment.",
)
```

### Step 2 — Refactor `run_template_queries(driver)` to accept mode

Current signature (line 151):

```python
def run_template_queries(driver) -> list[QueryResult]:
```

New signature:

```python
def run_template_queries(driver, mode: str = "canonical") -> list[QueryResult]:
    if mode == "canonical":
        # existing behaviour unchanged
        ...
    elif mode == "v7":
        from prompt5_retrieval import (
            route_question_multi,
            union_template_rows,
            densify_context_v2,
            prune_rows,
        )
        # for each question:
        #   1. route_question_multi(question, n=5) -> candidate templates
        #   2. union_template_rows(candidates, driver) -> deduped rows
        #   3. prune_rows(rows, question) -> relevance-filtered rows
        #   4. densify_context_v2(rows) -> context string
        #   5. pass context to existing LLM answer step (build_llm_chain remains the same)
```

### Step 3 — Refactor `build_llm_chain(graph)` to accept mode

Currently the chain uses `GraphCypherQAChain` which auto-generates Cypher via LLM. In v7 mode, we bypass the auto-Cypher generation because the context is already pre-densified by `densify_context_v2`. The LLM in v7 mode receives the densified context directly and synthesizes the answer.

Two sub-options:
- **3a:** keep `GraphCypherQAChain` for canonical, use a custom chain (just an LLM prompt with the densified context) for v7.
- **3b:** always use a custom prompt-based chain, and just swap the context source. More uniform, less LangChain-specific.

Recommend **3b** for simplicity and predictability of the production path.

### Step 4 — Update `main()` to thread the mode through

```python
def main() -> None:
    args = parse_args()
    driver = ...  # existing
    graph = ...  # existing
    results = run_template_queries(driver, mode=args.pipeline_mode)
    # downstream printing/eval unchanged
```

### Step 5 — Add a smoke test

Create `tests/test_step_3_2_v7_smoke.py`:

```python
def test_v7_mode_produces_context_with_multi_template_union():
    """Smoke test: v7 mode for one representative query should produce a context
    that includes facts from at least 2 distinct templates (union evidence)."""
    # set up a Neo4j fixture or mock
    # run the pipeline in v7 mode for a known multi-hop query
    # assert that the context string contains tokens from >=2 templates
```

### Step 6 — Update the README of Phase3_GraphRAG

Add a section "Pipeline modes" documenting:
- `canonical` mode: the frozen Chapter 5 baseline
- `v7` mode: the multi-template production path with measured +29 pp third-judge semantic on OOD held-out (Section 6.2.2-ter), recommended for deployment

### Step 7 — Validate on the canonical 100-query benchmark

Run both modes on the canonical benchmark:

```bash
python step_3_2_graph_rag_pipeline.py --pipeline-mode canonical > out_canonical.txt
python step_3_2_graph_rag_pipeline.py --pipeline-mode v7         > out_v7.txt
diff out_canonical.txt out_v7.txt | head -50
```

Then run the eval (`step_3_4_evaluation.py`) on both outputs and confirm:
- canonical EM strict ≈ 0.59 (frozen baseline)
- v7 EM strict ≥ canonical (no regression on in-distribution)
- v7 third-judge semantic > canonical on the multi-hop subset (the expected lift)

Estimated cost: ~100 Haiku calls × 2 modes × $0.003 = **$0.60**.

---

## Acceptance criteria

1. `step_3_2_graph_rag_pipeline.py --pipeline-mode canonical` produces output byte-identical to the current (pre-refactor) script on the 100-query benchmark.
2. `step_3_2_graph_rag_pipeline.py --pipeline-mode v7` produces output where the context length is on average larger than canonical (union of top-5 templates instead of top-1) and the third-judge semantic accuracy on the OOD held-out 38-query benchmark exceeds the canonical baseline by at least +20 pp (target floor; the measured probe gain was +29 pp).
3. The smoke test passes on local Neo4j.
4. The README documents both modes and the recommended default for deployment vs reproducibility.

## Effort breakdown

| Step | Estimated time |
|---|---|
| 1 (CLI flag) | 15 min |
| 2 (refactor run_template_queries) | 2-3 hours |
| 3 (refactor build_llm_chain) | 1-2 hours |
| 4 (main thread) | 15 min |
| 5 (smoke test) | 1 hour |
| 6 (README) | 30 min |
| 7 (validation runs) | 1 hour (mostly waiting) + $0.60 API |
| **Total** | **~6-8 hours of focused work** + ~$1 API |

## Out of scope for this work item

- Re-running the full Chapter 5 baseline numbers (frozen)
- Modifying `step_3_4_evaluation.py` canonical (only consume its output)
- Modifying `templates.py` (the 23 templates stay)
- Production deployment infrastructure (auth, dashboard, monitoring) → see FW1-bis in Section 6.3.1

## Cross-references

- FW9-ter (Cap 6 §6.3.2): compound-synthesis floor lift via query decomposition. Complementary to Lever 1 but addresses a different failure mode (synthesis vs retrieval).
- FW11 (Cap 6 §6.3.2): comprehensive Graph-RAG accuracy roadmap. Lever 1 is Phase B (retrieval saturation) of the FW11 roadmap.
- FW1-bis (Cap 6 §6.3.1): operator-interface deployment tier. Consumer of the v7 production path.

---

**Author:** Federico Lomi, MSc thesis, Politecnico di Torino, 2026.
**Status:** scoped, ready for implementation.
