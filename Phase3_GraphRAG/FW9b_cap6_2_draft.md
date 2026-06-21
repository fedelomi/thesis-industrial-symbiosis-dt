# Cap 6.2 draft: multi-template union retrieval on the OOD benchmark (FW9b)

Draft only. No thesis chapter is edited. Every number is provisional pending the
user's verification. Typography follows the thesis rules (no spaced em-dash, no
Oxford comma, English prose).

## The measurement artifact behind the apparent synthesis floor

The post-submission out-of-distribution (OOD) evaluation first attributed most of
its failures to synthesis: the routed template was in the ground-truth provenance
set, yet the composed answer was wrong. That attribution rested on a permissive
flag. The harness set routing_ok to true whenever the single routed template
appeared anywhere in the two-to-four template provenance set of a compound query,
so a compound query whose answer needed facts from a second or third provenance
template was scored as a synthesis failure even though its context was incomplete.
The label conflated under-retrieval with synthesis.

A controlled re-split of the failing items makes the distinction visible. Of the 20
items previously labelled synthesis failures, only a small minority had all needed
facts in the single-template context (the type-A items, for example OOD10 and
OOD25); the large majority (the type-B items) needed facts that lived on other
provenance templates, so the single-template context could not contain the answer.
The bottleneck was single-template retrieval coverage on compound queries, not the
language model's ability to compose.

## The fix: multi-template union retrieval

The fix retrieves a small pool of candidate templates instead of one. A question is
routed to a deterministic top-five candidate pool (keyword candidates first, then
the BGE semantic ranking), each candidate template's Cypher is executed against the
knowledge graph, the rows are unioned, deduplicated by a stable content signature
and capped at 40 rows, and the unioned rows are passed to the existing synthesis and
neuro-symbolic check unchanged. The check still verifies against the knowledge-graph
canonical facts, not against the formatted prose. The change sits behind a
RETRIEVAL_MODE flag (single is the unchanged baseline, multi is the new path), so the
incumbent pipeline stays runnable for a clean comparison. The router was first made
deterministic (an explicit stable ordering by score then template id, a fixed
embedding seed and single-threaded encoding), verified at zero flips across five
repeated routings of all 38 OOD questions, so the comparison is not routing noise.

## Measured before and after (provisional)

On the full 38-item OOD benchmark, scoring the 22 non-gap items (16 knowledge-graph
coverage-gap items are excluded from the accuracy denominator, as in the incumbent
harness) and judging with the same Sonnet cross-model judge:

- Incumbent semantic exact match: 27.3% (6 of 22).
- Multi-template union semantic exact match: 59.1% (13 of 22).
- Delta: +31.8 percentage points, paired BCa 95% confidence interval
  [+4.5, +50.0]. The interval excludes zero but is wide, as expected at this sample
  size.

An ablation isolates the levers (all arms on the BGE backend, same judge):

- The multi-template union is the dominant lever: the union over a single template,
  with densification held on for both, contributes +27.3 percentage points.
- The BGE routing backend over the incumbent routing contributes about +9 percentage
  points on its own.
- Densification is not a net lever: in single-template mode it scores -4.5 percentage
  points relative to the legacy row formatter (one item, within language-model
  noise). It is retained only as a targeted fix for the named-standard case (an ISO
  50001 template may state the constant ISO 50001:2018), which recovers OOD03, and it
  is reported as a targeted aid, not as a contributor to the headline.
- The query-type-aware pruner is dormant at the observed union sizes (10 to 35 rows,
  under the 40-row cap), so it does not contribute to the result and is kept only as
  a last-resort guard.

As a diagnostic on the 20-item subset of incumbent synthesis failures, the union
recovers 10 of 20 at the data level and, in a separate judged run, lifts that subset
from 0% to 50%. This subset figure is a diagnostic of where the gain comes from, not
the headline.

## Residual failures and honest framing

The union introduced one regression (OOD08, correct under the incumbent single
template, wrong once the union added distracting rows). After the fix, the residual
failures are of two kinds. The first is genuine multi-hop synthesis: for several
items (for example OOD10 and OOD30) a by-eye inspection of the live unioned context
confirms that every ground-truth fact is present, yet the model still composes the
multi-hop chain incorrectly. The second is harder coverage, including items the
knowledge graph does not fully support. The honest conclusion is that multi-template
union retrieval closes the retrieval-coverage half of the former synthesis floor,
and that the genuine residual is multi-hop composition, which a query-decomposition
or compose-then-verify step (a future-work item) would target, plus a smaller set of
knowledge-graph coverage gaps.

## Caveats (all numbers provisional)

- The sample is 22 scored items, so the confidence interval is wide.
- The synthesizer and the judge run at temperature zero but are not bit-reproducible:
  three to four items flip between repeated runs, so per-item verdicts and the exact
  aggregate carry run-to-run language-model noise on top of the sampling interval.
- The 20-item diagnostic subset and the 22 scored items of the full set are not
  identical: the frozen subset and the benchmark coverage flags disagree on a few
  items, which does not change the headline but is noted for traceability.
- The result bundles the BGE backend and the union (the ablation separates them but
  on the same small sample). The numbers stay provisional until the user verifies
  them.

Evidence artifacts: results/_fw9b_full38_per_query_<ts>.csv,
data/evaluation_results_graph-rag-ood-full38_<ts>.json,
results/_fw9b_v8_per_query_<ts>.csv (the 20-item diagnostic),
results/_mt_retrieval_sim.json (the data-level oracle), and the FW9b test suite.
