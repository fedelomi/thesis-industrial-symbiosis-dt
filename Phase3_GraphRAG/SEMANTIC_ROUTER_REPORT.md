# Phase 3 GraphRAG - Semantic routing fallback: free-gate report

Date: 2026-06-02. No commit, no API cost, local embedding model only.

## What changed

1. **New module `semantic_router.py`** - a pluggable cosine-similarity router over
   per-template intent anchors. Two backends:
   - `tfidf` (default, scikit-learn): zero download, fully offline, sub-millisecond.
   - `st` (`SEMANTIC_ROUTER_BACKEND=st`): local SentenceTransformer
     `BAAI/bge-small-en-v1.5` (one-time ~130 MB download, no API cost), stronger
     OOD generalisation. Anchor embeddings cached on disk
     (`data/semantic_router_cache/*.npz`, hash-keyed, auto-invalidated).
2. **`step_3_4_evaluation.py` - `route_question` made 2-stage** behind the
   `ENABLE_SEMANTIC_FALLBACK` feature flag (default on; set `=0` for the A/B
   baseline). The three hand-written priority patches added on 2026-06-02
   (B17 governance, C23 scenario comparison, C06 thermal gap) were **removed**.
3. **Two free gates** following the existing `check_*.py` pattern:
   `check_routing_diff.py` (routing-level regression proxy) and
   `check_routing_latency.py`.

Not touched: `step_3_4_bis_neuro_symbolic.py` and the downstream neuro-symbolic
check. `route_question` still returns a single template id; its interface and
all callers are unchanged.

## The mechanism (why this is not another patch)

The keyword router (`KEYWORD_ROUTING`) is first-match-wins on an ordered list,
so collisions were silently resolved by hand-placed priority rules. The new
Stage 1 instead collects **all** distinct keyword matches and exposes ambiguity:

- exactly 1 match  -> deterministic keyword route (41/100 questions; semantic
  never runs, zero risk).
- 0 matches or 2+ matches -> the question is ambiguous; Stage 2 runs (59/100).

Stage 2 ranks **all** templates by cosine to the anchors and overrides the
deterministic *prior* (the legacy route: first keyword match, else entity
fallback) only when its top template beats the prior template by a margin
(`SEMANTIC_ST_MARGIN=0.05`). For the zero-keyword case it must also beat the
global runner-up by the same margin (a flat distribution = undecided = keep the
safe prior). This gate is what makes the change a mechanism: it generalises to
the whole OOD collision class instead of matching specific phrasings, and it can
only override the legacy route on a clear semantic preference, which bounds
regressions.

Anti-leakage: anchors are authored from each template's documented data purpose,
not from the benchmark questions. Routing accuracy therefore stays an honest,
non-circular measurement.

## Free-gate results (bge backend)

| Gate | Result |
|------|--------|
| `check_routing.py` (flag on) | **21/21** targets, 0 missing templates |
| `check_routing.py` (flag off, patches removed) | 18/21 (B17, C23, C06 miss) |
| `check_routing_diff.py` vs v4 | 9 changes, **0 unexpected -> PASS** |
| `check_routing_latency.py` (bge) | p50 **14 ms**, p95 **21 ms** (target 50/100) |
| `check_routing_latency.py` (tfidf) | p50 **0.30 ms**, p95 **0.39 ms** |
| existing `pytest` suite | 5 passed |

**A/B delta**: removing the 3 manual patches drops routing to 18/21; the semantic
fallback recovers all 3 with no manual rules -> 21/21. B17 -> `P3` (governance
beats the `4gdh` thermal bucket), C23 -> `P2_thermal_compatibility_all` (the
correct mid-size scenario table, beating both keyword guesses `TEMPERATURE_BAND_DEF`
and `GENERIC_DC`), C06 -> `P2_thermal_compatibility_all`.

## Expected strict-EM gain vs v4

B17/C23/C06 were already correct in v4 (via the patches), so the **net strict
gain comes from 8 *additional* questions** the semantic stage re-routes to a
template that actually carries the asked fact, where v4's keyword router did not.
Per-item, judged by whether the new template's context contains the ground truth:

| Q | v4 -> new | Likely strict effect |
|---|-----------|----------------------|
| B09 | P5 -> P6 | gain (P5 lacks sector; GT needs the sector) |
| B13 | P2_all -> P4 | gain (eligible_tech is only in P4) |
| C20 | P6 -> P5 | gain (P6 has no supply temperature) |
| C22 | P6 -> P2_all | gain (needs supply vs required temp) |
| C29 | DK_4GDH -> DK_DH_COMPARE | gain (comparison needs both networks) |
| C07 | P6 -> SCENARIO_COUNT | likely gain (exact counts) |
| C24 | P2_all -> P4 | partial (eligibility yes, scenario list no) |
| B30 | P2_all -> P4 | neutral (a multiplication GT, neither answers) |
| A04 | GENERIC_REGULATION -> ALL_REG | neutral (both expose `2023` via `r.id`) |

Conservative estimate: **+4 to +6 strict-EM points vs v4** (roughly 72-74% from
the ~68% v4 baseline), to be confirmed by a paid `step_3_4_evaluation --config
graph-rag --ragas` run. The robustness value (the OOD collision class is now
closed structurally) is independent of that number.

## Residual error classes

1. **Jargon-entity, zero-signal questions** - e.g. B23 ("delta-T between
   HS-L-CO2_HTHP supply and sterilization process requirement"). All templates
   score ~0.67 (flat), so the discriminative guard keeps the safe prior (`P6`).
   It was failing in v4 too (no template returns both supply temp and the
   requirement together for that phrasing); the ideal home `P2_thermal_compatibility_all`
   is not reachable by either layer. Status: unchanged, EM-neutral.
2. **Calculation questions** - B30 needs an arithmetic product; no single
   template yields the computed figure. Routing cannot fix this; it is an
   answer-generation limitation.
3. **Small-margin gains left on the table** - C05, C18, C21, C11 would improve at
   a lower margin but are held at their v4 route to guarantee zero regression.
   They were already failing in v4, so this is a missed gain, not a regression.

## Alternatives considered

- **Embed Cypher templates instead of descriptions** (your prompt's suggestion):
  rejected. A sentence transformer is trained on natural language; raw Cypher
  (`MATCH`/`RETURN`/property names) is out of distribution and shares almost no
  surface with a user question. Description + paraphrase anchors win clearly.
- **TF-IDF as the semantic backend**: measured. It fixes C06 and the actor
  queries but **fails B17 and C23** - the lexical surface ("4gdh network",
  "mid-size dc") dominates the intent. Kept only as the zero-download default;
  the embedding backend is needed to close the flagship collisions.
- **`paraphrase-multilingual-MiniLM-L12-v2`** (your suggested model): viable, but
  the benchmark is English-only, so `bge-small-en-v1.5` (retrieval-tuned EN) is
  more discriminative on this corpus. Switch back via `SEMANTIC_ROUTER_MODEL` if
  native IT/DK queries are ever added.
- **Restricting Stage 2 to the conflicting candidates**: rejected. C23's correct
  template is neither candidate, so the conflict must only *trigger* a global
  ranking, not constrain it.
- **`step_3_10_paraphrase_routing_stability.py`** already exists in this phase and
  is the natural place to quantify the OOD-generalisation claim (route stability
  under paraphrase) once the paid run is done.

## How to run

```bash
# default (TF-IDF, no download)
python check_routing.py && python check_routing_diff.py && python check_routing_latency.py

# embedding backend (recommended; closes B17/C23)
pip install sentence-transformers
SEMANTIC_ROUTER_BACKEND=st python check_routing.py
SEMANTIC_ROUTER_BACKEND=st python check_routing_diff.py
SEMANTIC_ROUTER_BACKEND=st python check_routing_latency.py

# A/B baseline (semantic disabled)
ENABLE_SEMANTIC_FALLBACK=0 python check_routing.py
```
