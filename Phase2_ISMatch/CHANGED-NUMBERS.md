# Phase 2 (IS-Match) regenerated numbers - old vs new

Generated 2026-05-30 after recovering and regenerating the full Phase 2 LC pipeline.
Every value below was produced by `python run_phase_2_lc.py` (GATE PASS on all 9 steps)
and is bit-for-bit reproducible across reruns. "Old" = committed git HEAD CSV, "New" =
regenerated CSV in the working tree. Typography: no spaced em-dash, no Oxford comma.

Root cause of the changes: a Cowork Edit-tool bug truncated the long Phase 2 scripts; the
lost step_2_1 and step_2_3 were never migrated to the 2026-04-30 plant-level
geography-agnostic dataset schema. The recovered step_2_1 (fixed before this task, kept
as-is) groups the 27-plant dataset by tier and reads the `plant_` columns, which also
corrects the per-tier shift regime. step_2_3 is rebuilt here to rank at the plant level.

## 1. Core IS-Match Score (step_2_1, tier-level, pre-calibration)

Headline numbers: abstract, Cap. 5.6 Table 5.11bis, Figure 5.1, lesson-2, lesson-5.

| Quantity | Old (committed) | New | 3-dp for text |
|---|---|---|---|
| IS-Match LowT_60C (Edge, Mid) | 0.5298 | 0.5479 | 0.530 -> 0.548 |
| IS-Match LowT_60C (Hyperscale) | 0.5297 | 0.5476 | 0.530 -> 0.548 |
| IS-Match MidT_90C (Edge, Mid) | 0.4561 | 0.4532 | 0.456 -> 0.453 |
| IS-Match MidT_90C (Hyperscale) | 0.4558 | 0.4531 | 0.456 -> 0.453 |
| IS-Match HighT_130C (Edge, Mid) | 0.3628 | 0.3681 | 0.363 -> 0.368 |
| IS-Match HighT_130C (Hyperscale) | 0.3625 | 0.3678 | 0.363 -> 0.368 |
| Score range (min to max) | 0.36 to 0.53 | 0.37 to 0.55 | 0.3625-0.5298 -> 0.3678-0.5479 |

## 2. Score components (step_2_1)

| Quantity | Old | New |
|---|---|---|
| ri_temporal LowT_60C | 0.6966 | 0.7419 |
| ri_temporal MidT_90C | 0.5956 | 0.5916 |
| ri_temporal HighT_130C | 0.4406 | 0.4556 |
| delta_tc_norm LowT_60C | 0.2500 | 0.2500 (unchanged) |
| delta_tc_norm MidT_90C | 0.4167 | 0.4233 |
| delta_tc_norm HighT_130C | 0.5733 | 0.5767 |
| exergy_dt_norm (all LC) | 0.7529 / 0.7583 | 0.7529 / 0.7583 (unchanged) |
| n_weighted_steps LowT_60C | 35040 | 16704 |
| n_weighted_steps MidT_90C | 16704 | 35040 |
| n_weighted_steps HighT_130C | 16704 | 16704 (unchanged) |

Note on n_weighted_steps: the old values put LowT on the continuous (35040-step) weight and
MidT on the 2-shift (16704-step) weight, which contradicted Table 5.0 (LowT and HighT are
2-shift, MidT is continuous). The new values match Table 5.0. This shift-regime correction
is the main driver of the ri_temporal change and so of the LowT score moving 0.530 to 0.548.
No thesis text cites n_weighted_steps directly.

## 3. Ranking validation (step_2_3) - reconstructed, see RANKING-UNIT note

| Quantity | Old (committed) | New |
|---|---|---|
| Ranking unit | 3 tiers per DC (trivialising quick-fix) | 9 plants per DC, 27 pooled |
| precision@3 (headline) | 0.3333 | 0.3333 (preserved as the pooled metric) |
| precision@3 per DC | 0.3333 | 1.0000 (genuine: LowT tops both indices within a DC) |
| NDCG@9 per DC | 1.0000 | 0.9884 / 0.9881 / 0.9985 (Edge / Mid / Hyperscale) |
| NDCG@9 pooled | not reported | 0.9759 |
| NDCG@9 weight-grid mean | 1.0000 (flat) | 0.9915 to 0.9970 (still near-flat: ranking weight-robust) |
| Baseline label | RI_static_baseline | RI_temporal_baseline (ground truth stays RI_static) |
| CSV columns | method,dc_name,ndcg_at_k,precision_at_3,... | method,scope,ndcg_at_9,precision_at_3,n_marginal_plus,n_feasible_baseline,n_pairs |

If the thesis cites NDCG@9 = 1.000 it must become the new non-degenerate value (0.976
pooled, 0.99 per DC). If it cites precision@3 = 0.333 that value is preserved, now as the
pooled-over-27-pairs metric (the discriminating one). See the RANKING-UNIT note.

## 4. Delta-TC calibration (step_2_4)

| Quantity | Old | New |
|---|---|---|
| N_iter to converge | 5 | 5 (unchanged) |
| final LowT_60C score | 0.5723 | 0.5904 |
| final MidT_90C score | 0.5224 | 0.5206 |
| final HighT_130C score | 0.4483 | 0.4542 |
| final delta_max (iter 5) | 0.0092 | 0.0093 |
| convergence threshold | 0.01 | 0.01 (unchanged) |

## 5. Sobol on weights (step_2_5, seed 42)

Point estimates shift marginally; the interpretation (delta dominant, ranking
weight-invariant) is unchanged.

| Index | Old | New |
|---|---|---|
| S1 delta | 0.8699 | 0.8734 |
| ST delta | 0.8870 | 0.8905 |
| S1 gamma | 0.0947 | 0.0884 |
| S1 beta | 0.0141 | 0.0170 |

## 6. Stress test (step_2_6)

Headline conclusion unchanged and CSV byte-identical to HEAD: top-3 ranking stable under
all 8 perturbations (+/- 20% on Q, T, distance, delta_tc), max_rank_shift sequence
[0, 0, 2, 2, 0, 0, 0, 0]. The stress test re-derives scores internally, so it did not pick
up the score change.

## 7. CO2 avoided and heat recovery (step_2_7) - CHANGED (thesis headline)

These ARE thesis numbers (abstract and Cap. 5.6 Table 5.12). They change because the carbon
calc uses ri_temporal_utilization, which now reflects the corrected per-tier shift weighting
(section 2). The leader cell stays Hyperscale + MidT.

| Pair | CO2 old (tCO2/yr) | CO2 new | Q old (GWh/yr) | Q new |
|---|---|---|---|---|
| Hyperscale_LC + MidT_90C (leader) | 34,925 | 34,712 | 126.54 | 125.77 |
| Hyperscale_LC + LowT_60C | 11,681 | 12,439 | 42.32 | 45.07 |
| Hyperscale_LC + HighT_130C | 7,347 | 7,601 | 26.62 | 27.54 |
| Mid_LC + MidT_90C | 4,470 | 4,442 | 16.20 | 16.10 |
| Mid_LC + LowT_60C | 1,492 | 1,589 | 5.41 | 5.76 |
| Mid_LC + HighT_130C | 944 | 976 | 3.42 | 3.54 |
| Edge_LC + MidT_90C | 684 | 680 | 2.48 | 2.46 |
| Edge_LC + LowT_60C | 228 | 243 | 0.83 | 0.86 |
| Edge_LC + HighT_130C | 144 | 153 | 0.52 | 0.55 |

Thesis text to update: the abstract and Table 5.12 "126.5 GWh of annual heat recovery and
34,925 tCO2 of annual avoided emissions" become "125.8 GWh and 34,712 tCO2". The
equivalent-cars figure (7,592) scales with CO2 and shifts slightly too.

## 8. Unchanged (verified identical or equivalent)

| Quantity | Value (old = new) |
|---|---|
| Top pair | Edge_LC + LowT and Mid_LC + LowT tied at rank 1-2 (0.548), Hyperscale + LowT rank 3 |
| All 9 pairs in the marginal band (< 0.60) | yes |
| Scale invariance within tier (max spread) | < 0.0003 (0.5479 vs 0.5476 on LowT) |
| Feasibility summary (step_2_2, tier-level) | identical: Edge 2/9, Mid 3/9, Hyperscale 7/9 feasible plants |
| Epsilon gate (step_2_0) | 9/9 pass, epsilon HP 0.8648, CO2-HTHP 0.7820 (CSV byte-identical) |
| PROMETHEE II (step_2_8 ranking and correlation) | Spearman 0.6667, Kendall 0.4444, top-3 agreement 2/3 (CSVs byte-identical) |
| Dataset numeric columns (step_2_2, incl. RI_static) | identical; only plant_description and plant_data_source text changed (mojibake and em-dash replaced by hyphen) |
| Dataset summary (step_2_2_dataset_summary_lc.csv) | byte-identical |

## RANKING-UNIT note (step_2_3 design decision)

The ranking unit is the plant-level pair: each DC is a query of its 9 candidate plants
(3 plants per temperature tier), 27 DC-plant pairs in total. This is the finest unit the
dataset supports and the only one that keeps precision@3 meaningful.

- Not 3 tiers per DC: ranking 3 items makes the top-3 the whole set, so precision@3 is 1.000
  by construction. That was the trivialising quick-fix in the broken tree.
- Per-DC precision@3 = 1.0 is a true result here, not the old artefact: within any DC the
  three LowT plants top both IS-Match and RI_static, so the top-3-of-9 sets coincide.
- The discriminating metric is the pooled precision@3 over the 27 pairs = 0.3333, which
  recovers the committed canonical value. It falls below 1.0 because IS-Match and RI_static
  weight capacity differently: IS-Match uses capacity utilisation (Q_available / Q_nominal,
  near 0.8 for every DC scale) while RI_static uses capacity match to plant demand
  (RI_quantity), which penalises the small Edge DC. By RI_static the Mid and Hyperscale Dairy
  and Food plants top the pooled list, by IS-Match the three Food plants top it, so the
  pooled top-3 overlap by 1 of 3.

## Files regenerated

- All 15 CSVs under `Phase2_ISMatch/results/` plus `phase2_lc_results.xlsx`.
- `wiki/thesis/figures/fig-5-1-heatmap-is-match-9-scenarios.pdf` and `.png` (live scores,
  Times New Roman, 88 mm square, 300 DPI) via the new `gen_fig_5_1_heatmap.py`.

Scripts changed: `step_2_3_ranking_validation_lc.py` (rebuilt to plant-level),
`step_2_2_dataset_builder_lc.py` (mojibake and em-dash replaced by hyphen). `step_2_0` and
`step_2_1` carry their pre-existing working-tree truncation repair and plant-tier schema fix
(confirmed, not re-edited in this task).
