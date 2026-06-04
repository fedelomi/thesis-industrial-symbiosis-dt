# Phase 3 Blind: RQ3 Regulatory Benchmark Design

**Artefact:** `data/benchmark.jsonl` (102 questions). **Generator:** `build_benchmark.py`
(deterministic). **Runner:** `step_3_6_evaluate.py`. **Robustness:** `step_3_7_robustness.py`.

This benchmark defends RQ3: a structured retrieval-and-reasoning layer over the
regulatory corpus answers cross-firm compliance queries with measurable accuracy
and groundedness where a vector-only RAG baseline fails. It compares the
Neuro-Symbolic Regulatory Reasoner (NSRR) against a vector-only RAG baseline on the
identical question set.

## 1. Size and categories

| Category | N | Scoring | Ground-truth source |
|----------|---|---------|---------------------|
| `threshold_lookup` | 55 | numeric (1% tol + unit) | corpus numeric-threshold clause (data/corpus_facts.json) |
| `compliance_verdict` | 15 | exact verdict | EED Art. 12/26 decision logic, independently authored |
| `multi_hop` | 20 | token conjunction (synonym OR-groups) | chained clauses across scale/upgrade/jurisdiction |
| `comparative` | 12 | keyword (any) | jurisdiction/technology comparison clause |
| **Total** | **102** | | |

102 >= the 100-question minimum required by the deliverable.

## 2. Ground-truth protocol (document-grounded)

The benchmark is **document-grounded**: the ground truth is the corpus clause
itself, not a subjective annotation. This is the strongest form of objective ground
truth for a regulatory QA task and it makes the benchmark fully reproducible.

- **threshold_lookup**: each question is templated directly from a numeric threshold
  extracted from the corpus (e.g. "Data centre IT power threshold for reporting:
  500 kW", F10). The gold answer is the extracted `(value, unit)`. Only thresholds
  with a single unambiguous value and a recognised unit, from documents of DC-WHR
  relevance >= 3, are used. Each item carries its `clause` field for audit.
- **compliance_verdict**: 15 concrete scenarios whose gold verdict
  (`compliant`/`non_compliant`/`conditional`) is **authored independently** by
  applying the EED Art. 12 (>= 500 kW reporting) and Art. 26 (> 1 MW WHR
  obligation, efficient-DHC criterion, Art. 26(8) negative-CBA exemption) decision
  logic to the scenario. The gold is NOT produced by calling the gate, so the
  category is a genuine test of the gate rather than a tautology (see bias B2).
- **multi_hop**: gold is a set of required fact tokens (with synonym alternatives
  via `|`); a correct answer must surface every required fact (e.g. the mandated
  upgrade technology AND the triggered article).
- **comparative**: gold is a keyword set; a correct answer must surface at least
  one acceptable keyword (these are deliberately fuzzy conceptual questions).

## 3. Scoring (deterministic, symmetric)

All scorers are deterministic; no API is needed. Both systems receive the same
expected-unit hint (`prefer_unit`) and both ground their answer in their own top-3
retrieved context, so the comparison is symmetric.

- `numeric`: predicted value within 1% of gold AND same canonical unit family.
- `verdict`: the system's asserted verdict equals the gold verdict (the vector
  baseline asserts none, so it can only score when a verdict word coincidentally
  appears in retrieved context).
- `multi_token`: every gold token (each a `|`-separated OR-group) matches the
  answer text, with flexible matching (substring, article-abbreviation variants,
  all-words-present).
- `keyword`: at least one gold token matches.
- **groundedness**: the gold document id appears among the cited documents.

An optional cross-model semantic judge (`semantic_judge`, Sonnet-class) is provided
behind a mock; it defaults to a deterministic lexical-overlap proxy so the
evaluation stays hermetic, and exposes the keyword-vs-semantic gap if enabled.

## 4. Results (reproducible)

| Category | N | NSRR | Baseline | Gap |
|----------|---|------|----------|-----|
| comparative | 12 | 66.7% | 66.7% | +0.0 |
| compliance_verdict | 15 | 100.0% | 33.3% | +66.7 |
| multi_hop | 20 | 65.0% | 45.0% | +20.0 |
| threshold_lookup | 55 | 92.7% | 85.5% | +7.3 |
| **OVERALL** | **102** | **85.3%** | **67.6%** | **+17.6** |

Bootstrap 95% CI on the gap: **[+10.8, +25.5] pp** (10,000 resamples, significant).
Gate leave-one-out: removing the symbolic gate drops NSRR to 72.5% (gate contributes
+12.7 pp). Paraphrase perturbation: 85.3% -> 85.3% (stable within 5 pp).

## 5. Biases this methodology is exposed to, and mitigations

- **B1 Template-phrasing bias.** Threshold questions are templated from corpus
  clauses, which can favour systems that index those clauses. *Mitigation:* the
  paraphrase-perturbation diagnostic (step_3_7) rewrites every question with surface
  paraphrases and shows NSRR accuracy is unchanged (0.0 pp), and the `comparative`
  category uses natural conceptual phrasing where NSRR and the baseline tie.
- **B2 Near-circularity of compliance verdicts.** The gate and the gold both encode
  the EED logic, so a correct gate scores ~100% by design. *Mitigation:* the gold is
  authored independently from the regulation (not from the gate); per-category
  reporting makes the mechanism transparent; and the decisive comparison is that the
  vector baseline cannot assert a verdict at all (33.3% by coincidence vs 100%). A
  deterministic gate scoring 100% on objective compliance questions is the intended
  property, not an artefact.
- **B3 Extraction error propagation.** Gold thresholds come from an offline LLM
  extraction, which can mis-extract. *Mitigation:* the extraction was schema-
  constrained and each item exposes its source `clause` for spot-audit; only clean
  single-value thresholds are used.
- **B4 In-distribution corpus.** The corpus is restricted to EU + Italy + Denmark
  (framework Decision D3); the benchmark therefore measures in-distribution
  performance and does not test cross-jurisdiction generalisation. *Mitigation:*
  declared as a limitation; a held-out jurisdiction (e.g. Germany) is the natural
  out-of-distribution extension (Future Work).
- **B5 Lexical vs semantic scoring.** Deterministic scorers may under- or over-credit
  paraphrased answers. *Mitigation:* the optional cross-model LLM judge can re-score
  the interpretive subset; numeric and verdict items are unambiguous and need no
  judge.

## 6. Reproducibility

`python build_benchmark.py && python step_3_6_evaluate.py && python step_3_7_robustness.py`
regenerates the benchmark, the results (`results/eval_summary.json`,
`results/eval_results.csv`) and the robustness summary
(`results/robustness_summary.json`) bit-for-bit (all randomness seeded with
`GLOBAL_SEED=42`).
