# Thesis codebase review — 2026-05-17

## 0. Executive summary

- The entire post-reorg codebase (Phase1_PhysicalDT, Phase2_ISMatch, Phase3_GraphRAG, Phase4, common) has never been committed. One stale initial commit holds the old folder structure. All working code is untracked.
- A hardcoded default Neo4j password (`thesis2026`) is present in Phase3_GraphRAG/config.py and will appear in version history once committed.
- Three Phase 1 LC post-processing scripts hardcode `/tmp/` Unix paths; these will fail at runtime on the Windows 11 development machine.
- Phase 4 train_ppo.py is missing the CheckpointCallback required by CLAUDE.md, and the `_effective_q_available()` method renders the DT-grounding dimension of the A0 vs D ablation a no-op for the current LC dataset.
- CLAUDE.md cluster descriptions for folders B, D and E are wrong relative to the authoritative codes-and-mapping.md. Code citations in docstrings are correct; only the CLAUDE.md legend is stale.
- Phase 1 airside scripts lack type hints on all public functions, and os.path is used pervasively in Phase 1 and Phase 2 scripts instead of pathlib.Path.

---

## 1. Codebase map

### File and line counts (thesis code only, SustainGym fork excluded)

| Phase | Files (.py) | Lines |
|---|---|---|
| Phase1_PhysicalDT (airside + lc) | 14 | 2992 |
| Phase2_ISMatch | 6 | 1770 |
| Phase3_GraphRAG | 15 | 4977 |
| Phase4 | 18 | 2020 |
| common | 2 | 105 |
| **Total** | **55** | **11 864** |

SustainGym fork (Phase1_PhysicalDT/airside/sustaingym/): 8 files, included for reference only, not in scope.

### Orphan / duplicate / stale files

| Path | Issue |
|---|---|
| `commit_audit_2026-05-05.sh` | Untracked shell script with stale references to `Phase 3/` (space) and `Phase1/`; contains `git add Phase1/ Phase2/ "Phase 3/"` at line 28. Not executable on Windows. Not harmful, but should be committed or removed. |
| README.md (lines 120-305) | Content block duplicated verbatim. The Phase 3 description and Active decisions table appear twice. |
| README.md (table row, line 23) | Phase 4 listed as "Agentic Negotiation (Strato 3 — future Phase 4) \| not in this repo yet". Phase 4 is complete and present. |
| Phase4/config/scenarios.py:L109 | Comment "S1..S9 follow roadmap Phase 5.1 order" — stale Phase 5 reference. Phase 5 does not exist per CLAUDE.md. |

No `Phase3_old/` folder or import of it was found anywhere in the codebase.

---

## 2. Confirmed issues (with severity)

### I-01 — Untracked codebase: no commit since initial (HIGH)

**Evidence:** `git status` shows `??` for Phase1_PhysicalDT/, Phase2_ISMatch/, Phase3_GraphRAG/, Phase4/, common/, CLAUDE.md and README.md. The only commit is `c23444e Initial commit: Phase 1, 2, 3 thesis implementation` which holds the old folder structure (deleted, marked `D`). Reproducibility requires that the current code be committed before the thesis defence.

```diff
# Not a code patch — requires a git commit of all untracked content.
# Suggested message:
# "chore: commit post-reorg codebase (Phase1_PhysicalDT..Phase4, common, CLAUDE.md)"
```

---

### I-02 — Hardcoded default Neo4j password (HIGH)

**Evidence:** `Phase3_GraphRAG/config.py:L33`

```python
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "thesis2026")
```

When committed, this literal will be visible in git history. If `NEO4J_PASSWORD` is not set in the environment the code silently uses the hardcoded value.

```diff
-NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "thesis2026")
+_raw = os.getenv("NEO4J_PASSWORD", "")
+if not _raw:
+    raise RuntimeError("NEO4J_PASSWORD env var is not set. Copy .env.example to .env.")
+NEO4J_PASSWORD: str = _raw
```

---

### I-03 — Windows-incompatible /tmp/ paths in Phase 1 LC (MEDIUM)

**Evidence:**
- `Phase1_PhysicalDT/lc/step_0_eed_compliance.py:L37`
- `Phase1_PhysicalDT/lc/step_1_4c_erf_sensitivity.py:L39`
- `Phase1_PhysicalDT/lc/step_1_4d_benchmark_comparison.py:L33`

All three define `TMP_XLSX = "/tmp/phase1_lc_results_*.xlsx"`. On Windows 11 the `/tmp/` directory does not exist and the open call will raise `FileNotFoundError`.

```diff
-TMP_XLSX   = "/tmp/phase1_lc_results_step0.xlsx"
+import tempfile
+TMP_XLSX   = Path(tempfile.gettempdir()) / "phase1_lc_results_step0.xlsx"
```

---

### I-04 — os.path instead of pathlib.Path (MEDIUM)

**Evidence:** Pervasive in Phase 1 (airside and lc) and Phase 2 orchestrators.

Representative locations:
- `Phase1_PhysicalDT/lc/run_phase_1_lc.py:L29-30, 65, 74, 82, 86`
- `Phase1_PhysicalDT/lc/step_0_eed_compliance.py:L31-36`
- `Phase1_PhysicalDT/lc/step_1_4c_erf_sensitivity.py:L32-39`
- `Phase2_ISMatch/run_phase_2_lc.py:L22, 26-28, 79, 88, 96-136`
- `Phase3_GraphRAG/run_phase_3.py:L39, 106-108`

Phase 2 step scripts (step_2_1 through step_2_4) already use pathlib.Path correctly and serve as the pattern to follow.

```diff
-BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
-RESULTS_DIR = os.path.join(BASE_DIR, "results")
-os.makedirs(RESULTS_DIR, exist_ok=True)
+BASE_DIR    = Path(__file__).parent
+RESULTS_DIR = BASE_DIR / "results"
+RESULTS_DIR.mkdir(exist_ok=True)
```

---

### I-05 — Missing type hints in airside RC model (MEDIUM)

**Evidence:** `Phase1_PhysicalDT/airside/step_1_1_2_rc_model.py:L80, 98, 103, 133`

```python
def it_load_profile(t, N_STEP, seed=42):          # no types
def compute_cv_rmse(sim_values, ref_values):       # no types
def compute_exergy(theta_K, Q_available, T0_K=288.15):  # no types
def run_simulation(scenario_name, params):         # no types
```

The LC counterpart (`step_1_1_2_rc_model_lc.py`) is mostly typed but `run_simulation` is missing a return type annotation.

```diff
-def run_simulation(scenario_name: str, params: dict) -> tuple:
+def run_simulation(scenario_name: str, params: dict) -> tuple[pd.DataFrame, bool]:
```

---

### I-06 — Bare except without logging in CoolProp calls (MEDIUM)

**Evidence:**
- `Phase1_PhysicalDT/lc/step_1_1_2_rc_model_lc.py:L175`
- `Phase1_PhysicalDT/airside/step_1_1_2_rc_model.py:L120`

Both catch the CoolProp exception silently and return zeros. A fluid-property failure produces zero exergy for all timesteps, which would silently corrupt the IS-Match Score without any warning.

```diff
-    except Exception:
-        return out   # fall back to zeros on error
+    except Exception as exc:
+        logger.warning("CoolProp vectorised call failed: %s; returning zero exergy array.", exc)
+        return out
```

---

### I-07 — print() in library-level code (LOW)

**Evidence:** Pervasive in Phase 1 and Phase 3 scripts. Representative:
- `Phase1_PhysicalDT/lc/step_1_1_2_rc_model_lc.py:L197-293` (multiple `print()`)
- `Phase1_PhysicalDT/lc/step_0_eed_compliance.py` (multiple `print()`)
- `Phase3_GraphRAG/step_3_6_privacy_gate.py:L74, 107-113`

Phase 2 step scripts and all Phase 4 code correctly use the `logging` module.

---

### I-08 — CheckpointCallback missing from train_ppo.py (MEDIUM)

**Evidence:** `Phase4/train/train_ppo.py` imports only `EvalCallback` (line 36) and instantiates it at line 81. No `CheckpointCallback` is present anywhere in Phase4.

CLAUDE.md requires: "CheckpointCallback every 50k steps".

```diff
-from stable_baselines3.common.callbacks import EvalCallback
+from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
 
 # (inside train())
+ckpt_cb = CheckpointCallback(
+    save_freq=max(1, 50_000 // n_envs),
+    save_path=str(out_dir / f"checkpoints_{scenario_id}"),
+    name_prefix="ppo",
+)
-model.learn(total_timesteps=total_timesteps, callback=eval_cb)
+model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb])
```

---

### I-09 — Stale Phase 5.1 reference in scenarios.py (LOW)

**Evidence:** `Phase4/config/scenarios.py:L109`

```python
# Scenario-matrix indexing (S1..S9 follow roadmap Phase 5.1 order)
```

CLAUDE.md states "There is no Phase5." The comment should reference the methodology section or roadmap step 4.6.

```diff
-# Scenario-matrix indexing (S1..S9 follow roadmap Phase 5.1 order)
+# Scenario-matrix indexing (S1..S9 follow methodology section 4.6)
```

---

### I-10 — CLAUDE.md cluster descriptions are wrong (MEDIUM — documentation)

**Evidence:** CLAUDE.md lists:
> A (IS theory), B (RL/MAS), C (Digital Twins), D (Thermodynamics), E (Policy/Reg), F (DC WHR Cases)

The authoritative `codes-and-mapping.md` defines:
> B = Industrial Symbiosis, C = Digital Twin & AI for DC, D = LLM & DSS for Energy Systems, E = LLM Multi-Agent Negotiation

Code citations in docstrings (B4 = Yazdanpanah, B5 = Shapley, C6 = LC-Opt, D5 = Graph RAG PoliMi) are all consistent with codes-and-mapping.md and are correct. Only the CLAUDE.md description table is stale.

Note: implementation decision codes D1-D6 (documented in CLAUDE.md) share the letter+number prefix with source cluster codes (D1 = CE-HEAT, D2 = Heriot-Watt, etc.). The two namespaces are disambiguated by context ("Decision active: D5" vs "calibrated on D5 ref.") but a reader new to the codebase may confuse them.

```diff
# In CLAUDE.md, Source clusters section:
-Source clusters: A (IS theory), B (RL/MAS), C (Digital Twins), D (Thermodynamics), E (Policy/Reg), F (DC WHR Cases)
+Source clusters: B (Industrial Symbiosis), C (Digital Twin & AI for DC), D (LLM & DSS for Energy Systems), E (LLM Multi-Agent Negotiation), F (DC WHR Cases & Policy)
+Note: implementation decisions D1-D6 are distinct from source cluster codes DX.
```

---

### I-11 — ΔTC reduction factors still placeholder (HIGH — methodology)

**Evidence:** `Phase2_ISMatch/step_2_4_delta_tc_calibration_lc.py:L66-70`

```python
_FALLBACK_REDUCTION: dict[str, float] = {
    "LowT_60C":   0.10,
    "MidT_90C":   0.10,
    "HighT_130C": 0.10,
}
```

The roadmap audit note (2026-05-09, roadmap-fasi-1-2-3.md:L143) shows the previously used placeholder values were `LowT=0.25, MidT=0.18, HighT=0.12` and the actual Phase 3 JSON had `0.5/0.5/0.5` (uniform). The current fallback 0.10 differs from both. Until `step_3_5_bis_delta_tc.json` is produced by a live Neo4j run, the IS-Match Scores driving Phase 4 agent initialisation depend on whichever placeholder is active. The thesis result in Cap. 5.3 must clearly state which reduction factors were used.

---

### I-12 — _effective_q_available() is a no-op ablation variable (MEDIUM — methodology)

**Evidence:** `Phase4/env/is_negotiation_env.py:L344-350`

```python
def _effective_q_available(self) -> float:
    if self.dt_grounded:
        return self.dc.q_available_kw
    return self.dc.q_available_kw   # same value both branches
```

The comment acknowledges that LC t_availability=1.0 makes static and dynamic Q identical. This means the A0 vs D ablation isolates *shielding only*, not DT-grounding, despite the ablation table labelling D as "RL + DT + governance" vs A0 as "RL without DT-grounding". The ablation finding (Δ PoA = +10.2 pp for HighT) correctly reflects the shielding benefit but cannot separately quantify DT-grounding's contribution.

---

### I-13 — 95% CI not computed in ablation results (LOW)

**Evidence:** `Phase4/train/run_ablation.py:L212-234`

The `_print_summary` function reports seed-averaged means but never calls `scipy.stats.t.interval()`. CLAUDE.md Rule 8 requires "5 seeds minimum, report 95% CI via scipy.stats.t." The CSV output (ablation_results.csv) contains per-seed rows, so the CI can be computed in post-processing, but it is not built into the pipeline.

---

### I-14 — README Phase 4 row is stale (LOW)

**Evidence:** README.md:L23

The table row for Phase 4 reads "not in this repo yet". Phase 4 is complete with 16/16 tests passing. The README was not updated after Phase 4 implementation.

---

## 3. Per-phase findings

### Phase 1 — Physical-DT

**Coding rule violations:**
- Type hints: all public functions in `step_1_1_2_rc_model.py` lack type annotations (I-05). LC counterpart is mostly compliant.
- pathlib: all Phase 1 scripts use `os.path` (I-04). Exception: step_2_x series in Phase 2 already converted.
- logging: `print()` used throughout instead of `logging` (I-07).
- `/tmp/` paths: three post-processing scripts will fail on Windows (I-03).
- No docstrings on public functions in `step_0_eed_compliance.py`, `step_1_4c_erf_sensitivity.py`, `step_1_4d_benchmark_comparison.py`.

**Domain checks:**
- RC model energy balance: the exact analytical solution `theta_ss + (theta - theta_ss) * exp(-dt/tau)` correctly conserves energy in the steady state. The vectorised exergy formula follows standard physical exergy for a liquid stream. Both are correct.
- Synthetic profile seed handling: `np.random.default_rng(seed=42)` is used consistently across step_1_3_synthetic.py and step_1_3_synthetic_lc.py. Modern API, reproducible.
- Gate file: `gate_1_results.csv` is not generated by step_1_4 or run_phase_1_lc.py; the gate is instead embedded in the PASS/FAIL printed output and in the validation CSV columns. Consider writing a dedicated `gate_1_results.csv` to satisfy the CLAUDE.md gate convention.
- CoolProp error handling: bare except without logging in vectorised call (I-06).
- EED compliance: step_0_eed_compliance.py correctly references EU Delegated Regulation 2024/1364 implementing EED 2023/1791 Art. 12 and Art. 26. CLAUDE.md spec mentions "EED Art. 25" which does not exist in EED 2023/1791; the code is correct. step_1_4c_erf_sensitivity.py has the regulation reference in the file header comment. Both pass the domain check.

### Phase 2 — IS-Match

**Coding rule violations:**
- pathlib: `run_phase_2_lc.py` uses `os.path` throughout (I-04). The step scripts are compliant.
- Bare except: `run_phase_2_lc.py:L90` catches `Exception` silently.
- Dataclass for config: `step_2_1_is_match_score_lc.py` defines an `ISMatchConfig` dataclass correctly; step_2_4 uses module-level constants. Compliant.

**Domain checks:**
- cvxpy solver fallback: No cvxpy optimisation in Phase 2 (IS-Match Score uses numpy directly). The cvxpy fallback chain CLARABEL → SCS → ECOS is present in Phase 4 `baseline_lp.py` where it is required. No issue.
- Epsilon filter: step_2_0 correctly formalises `epsilon = k * (1 - r)` and passes all 9 scenarios (k≈1, r≈0). Traceable to B6 citation in file header.
- IS-Match formula: `beta*RI_temporal + gamma*Exergy_DT_norm - delta*ΔTC_norm` matches the spec from 3-layer-framework-and-methodology.md Strato 1.
- Dataset builder reproducibility: step_2_2 uses a fixed seed for the sector-parametric profile generation. Compliant.
- Convergence log: step_2_4 writes `step_2_4_convergence_log_lc.csv` with per-iteration Δ_max. Compliant.
- Metric discrepancy: RQ1 in objectives-and-rqs.md states "NDCG@10" but Phase 2 implements and reports NDCG@9 (9 scenarios total). The roadmap uses NDCG@9. The thesis should use NDCG@9 and note the difference from the stated metric.

### Phase 3 — Graph RAG

**Coding rule violations:**
- Hardcoded password (I-02).
- pathlib: `run_phase_3.py` uses `os.path` (I-04). Internal step scripts use pathlib.
- logging vs print: several scripts use `print()` instead of `logging` (I-07). `step_3_6_privacy_gate.py` is print-only.
- Bare except: step_3_4_evaluation.py has 5 occurrences; step_3_2_graph_rag_pipeline.py has 2. Most log the exception but do not re-raise, allowing pipeline continuity despite errors.
- DBMS name: config.py defaults to `NEO4J_DATABASE = "neo4j"` while CLAUDE.md says `Graph_RAG_FL`. Functional mismatch unless the operator has renamed the default database.

**Domain checks:**
- Neo4j bolt URI: `bolt://127.0.0.1:7687` hardcoded correctly as default.
- Schema constraints: step_3_0_neo4j_schema.py defines uniqueness constraints on all primary node labels (DataCenter.id, Scenario.id, Regulation.id, RegulatoryArticle.id, etc.). All constraints use `IF NOT EXISTS` semantics (idempotent). Compliant.
- Ingest scripts: all five ingest scripts use MERGE on the primary key property, not CREATE. Re-runnable without duplicates. Compliant.
- Graph RAG pipeline: GraphCypherQAChain wired in step_3_2; 6 multi-hop patterns (P1-P6) plus GENERIC_* and ISO50001 templates in templates.py (single source of truth). Compliant.
- RAGAs benchmark: step_3_4_evaluation.py uses faithfulness, answer_relevancy, context_precision. RNG seeding uses LangChain/OpenAI defaults; explicit `random.seed` for the subset sampler is missing from step_3_4.
- Privacy gate: step_3_6 uses a Q_available_mean proxy rather than re-running the full Phase 2 pipeline on the synthetic profile. The docstring acknowledges this. The proxy is a valid approximation for the thesis scope but should be stated explicitly in Cap. 6.2 Limitations.

### Phase 4 — RL negotiation

**Coding rule violations:**
- CheckpointCallback missing (I-08).
- n_envs: train_ppo.py uses 8 (CLAUDE.md: 4) and SubprocVecEnv directly (CLAUDE.md: make_vec_env).
- 95% CI not computed (I-13).
- Stale Phase 5.1 comment (I-09).

**Domain checks:**
- Observation and action spaces: `spaces.Box` with explicit bounds `[0, 1]^8` and `[0, 1]^4`. Compliant.
- `reset(seed=...)` plumbs to `np.random.default_rng(seed)`. Compliant.
- `step()` returns 5-tuple `(obs, reward, terminated, truncated, info)`. Compliant.
- `info` dict includes `violations` (shielding flags), `converged`, `welfare_dc`, `welfare_mfg`, `is_match_post`. Compliant.
- Shielding implementation: action masking / projection (clip-and-report), not reward penalty. Compliant with spec.
- Four Yazdanpanah rules: Rule 1 (Q <= Q_available, BLOCKED, shielding.py:L137-143), Rule 2 (T_offered >= T_req-5C, BLOCKED, L151-157), Rule 3 (price >= CAPEX floor, WARNING clip, L160-170), Rule 4 (upgrade_tech == mfg.upgrade_required, BLOCKED, L173-180). All four present.
- Shapley closed-form N=2: `phi_i = 0.5*v(i) + 0.5*(v(N)-v(j))`. Correct.
- LP welfare baseline: social-planner LP with CLARABEL→SCS→ECOS fallback chain present. PoA denominator = LP welfare. Compliant.
- Ablation: 2 configs (A0, D) × 5 seeds × 9 scenarios = 90 runs. Results to ablation_results.csv with one row per (config, seed, scenario). Compliant with spec.
- DT-grounding ablation contrast: see I-12. A0 and D differ only in shielding; DT-grounding is not an independent variable for the LC dataset.

### Common / cross-cutting

- `common/dc_id_mapping.py`: uses `pathlib` (compliant), has type hints (compliant), uses `Final[str]` constants (compliant). Well-structured.
- `commit_audit_2026-05-05.sh`: stale script, references old paths, contains `git add`. Not dangerous (no destructive git commands) but should be committed or gitignored.
- No stale `Phase3_old` references found in any Python file or docstring.

---

## 3.bis Methodology — code traceability

### RQ and Objective coverage

| RQ | Objective | Phase(s) | Primary metric | File producing metric |
|---|---|---|---|---|
| RQ1 (Characterisation + IS-Match) | OS1 + OS2 | Phase 1 + Phase 2 | W1_norm (≤ 0.05), NDCG@9, precision@3 | lc_validation_1_4.csv, step_2_3_ranking_validation_lc.csv |
| RQ2 (LLM + Barriers) | OS3 | Phase 3 | RAGAS faithfulness, context_precision, privacy_gate delta (< 5%) | evaluation_results_graph-rag_*.json, step_3_6_privacy_gate.json |
| RQ3 (Agents + Nash convergence) | OS4 (part 1) | Phase 4 | PoA, convergence_rate | ablation_results.csv |
| RQ4 (9-scenario comparison) | OS4 (part 2) | Phase 4 + future | Scenario matrix PoA, LLM sensitivity (70/85/95%) | ablation_results.csv (9 scenarios); LLM sensitivity not yet implemented |

RQ4 LLM-accuracy sensitivity analysis (Cap. 5.4) is not covered by any existing phase script. It is currently future work.

### Objective coverage

| Objective | Phase | Deliverable |
|---|---|---|
| OS1 | Phase 1 | lc_dc_results_annual.csv (T_supply, Q_available, Exergy_DT, t_availability) |
| OS2 | Phase 2 | step_2_4_delta_tc_calibration_lc.csv (IS-Match Score per scenario) |
| OS3 | Phase 3 | evaluation_results_graph-rag_*.json, step_3_6_privacy_gate.json |
| OS4 | Phase 4 | ablation_results.csv, ppo_S*.zip models |

### Layer → Phase mapping

| Layer | Phase | Match |
|---|---|---|
| Layer 1 — Physical-DT (OS1) | Phase 1 | CORRECT |
| Layer 2 — Institutional-LLM (OS3) | Phase 3 | CORRECT |
| Layer 3 — Agentic Negotiation (OS4) | Phase 4 (with Phase 2 IS-Match as feeder) | CORRECT |
| Phase 2 (IS-Match, OS2) | Feeder to Layer 1 and Layer 3 | CORRECT per roadmap |

### Cluster citation audit

All B, C, D citations found in Python files are consistent with codes-and-mapping.md. Sample verified:

| Citation | File | Stated meaning | codes-and-mapping verdict |
|---|---|---|---|
| B4 | shielding.py:L6 | Yazdanpanah MAS governance | B4 = Engineering multiagent IS systems (Yazdanpanah) — OK |
| B5 | shapley.py:L6 | Shapley value IS | B5 = Dynamics and allocation of TC in multiagent IS — OK |
| B6 | step_2_0_epsilon_filter_lc.py header | epsilon-parameter | B6 = Industrial symbiosis: how to apply — OK |
| C1 | step_1_1_2_rc_model.py:L7 | Frontier DT calibration | C1 = DT-based cooling optimization Frontier DC — OK |
| C6 | step_1_1_2_rc_model_lc.py:L6 | LC-Opt FMU | C6 = LC-Opt RL benchmark liquid cooling — OK |
| D5 | step_3_2_graph_rag_pipeline.py header | Graph RAG PoliMi | D5 = Graph RAG energy efficiency QA — OK |
| D6 | step_2_0_epsilon_filter_lc.py header | Co-LLM HVAC baseline | D6 = Co-LLM chiller optimization — OK |

CLAUDE.md cluster descriptions for B, D, E are stale (see I-10) but the code citations themselves are correct.

No citations for non-existent cluster codes (A or F with numeric suffixes) were found in the Python files. The wiki gap-analysis text references A3 and A9 in prose (not in code comments), which is correct since Folder A raw sources have no assigned wiki codes.

### Wikilink integrity

All wikilinks found in Python docstrings and comments were verified against the wiki directory (163 pages):

| Wikilink | File | Status |
|---|---|---|
| [[roadmap-fasi-1-2-3]] | common/dc_id_mapping.py, Phase3 scripts | OK |
| [[decisioni-implementative]] | multiple files | OK |
| [[decisioni-implementative#D5]] | step_1_1_2_rc_model_lc.py | OK |
| [[decisioni-implementative#D2]] | step_2_2_dataset_builder_lc.py | OK |
| [[decisioni-implementative#D6]] | step_2_0_epsilon_filter_lc.py | OK |
| [[graph-rag-entity-schema]] | Phase3 scripts (6 refs) | OK |
| [[concepts/graph-rag]] | run_phase_3.py | OK |
| [[concepts/is-match-score]] | Phase3 scripts, step_3_6 | OK |
| [[sources/dk-final-necp-2024]] | step_3_1c_ingest_tier_b_dk.py | OK |
| [[sources/regulation-and-planning-district-heating-denmark]] | step_3_1c | OK |
| [[concepts/district-heating-generations]] | step_3_1c | OK |
| [[sources/decreto-ministeriale-21-maggio-2021-tee]] | step_3_1b | OK |
| [[sources/decreto-mase-certificati-bianchi-2025]] | step_3_1b | OK |
| [[entities/gse-gestore-servizi-energetici]] | step_3_1b | OK |
| [[concepts/certificati-bianchi-tee]] | step_3_1b | OK |

No broken wikilinks detected.

---

## 4. Test coverage gaps

### Phase 1

No pytest suite exists. Minimal recommended tests:

| Test | Rationale |
|---|---|
| `test_rc_model_energy_balance` | Verify `theta_ss` is reached within 5 time constants for Edge/Mid/Hyperscale LC; verify theta stays in physical range [T_CDU_supply, T_CDU_supply + 15K]. |
| `test_exergy_nonnegative` | Verify `compute_exergy_vectorized` returns non-negative values for all LC scenarios over a representative 96-step day. |
| `test_synthetic_gate_metrics` | Regression test: run bootstrap on a known LC annual CSV and assert W1_norm < 0.05 and NDE < 0.20. |
| `test_eed_size_categories` | Unit test for the EU size category logic in step_0: Edge=Very small, Mid=Large, Hyperscale=Very large. |

### Phase 2

No pytest suite exists. Minimal recommended tests:

| Test | Rationale |
|---|---|
| `test_epsilon_filter_all_pass` | Verify all 9 LC scenarios pass the epsilon gate (k=1, r=0 → epsilon=1.0 > epsilon_min). |
| `test_is_match_score_bounds` | Verify scores are in [0, 1] and that LowT > MidT > HighT for the same DC scale (expected ordering). |
| `test_delta_tc_convergence` | Verify the calibration loop converges within MAX_ITER=5 and Δ_max < 0.01. |

### Phase 3

No pytest suite exists. Minimal recommended tests:

| Test | Rationale |
|---|---|
| `test_schema_idempotent` | Call step_3_0 twice; verify no duplicate constraints or errors on second run (mock driver). |
| `test_privacy_gate_pass` | Verify step_3_6 returns `"PASS"` for the three LC scenarios (using the hardcoded PHASE1_LC_STATS). |
| `test_neuro_symbolic_consistency` | Verify `logical_consistency_rate` is within expected range for a mocked answer that cites only valid KG entities. |

### Phase 4

Suite exists at `Phase4/tests/` with 3 test files (16 tests confirmed passing). Identified gaps:

| Gap | Rationale |
|---|---|
| `test_ablation_dt_grounding_contrast` | Verify that A0 and D produce different Q_available values when the DC profile has t_availability < 1.0 (non-LC scenario). Currently the LC dataset makes A0 and D identical on this dimension (I-12). |
| `test_convergence_criterion` | Verify that Δw / max(|w|, 1) falls below convergence_delta within max_rounds for a cooperative scenario. |
| `test_poa_le_one` | Verify PoA = welfare_decentralised / welfare_LP ≤ 1.0 for at least one scenario (the LP is the upper bound by definition). |

---

## 5. STRATEGIC_QUESTIONS_FOR_FEDE

Q1. The `_effective_q_available()` method returns the same value for `dt_grounded=True` and `dt_grounded=False` because the LC annual profile has t_availability=1.0, so should the A0 configuration add a second differentiating variable (e.g. a stochastic noise on Q or a seasonal factor) to make the DT-grounding component of the ablation empirically separable from the shielding component?

Q2. The privacy gate in step_3_6 uses a mean-Q-delta proxy rather than re-running the full Phase 2 IS-Match pipeline on the synthetic profile — is this simplification sufficient for the thesis claim that the synthetic profile preserves decision quality, or should Passo 3.6 be upgraded to a full Phase 2 re-run before the thesis defence?

Q3. The ΔTC reduction factors currently used in the Phase 4 scenario initialisation are fallback constants (0.10 for all tiers) rather than the Phase 3 RAG-measured values — should the thesis present results based on the placeholder or should Phase 3 be re-run with a live Neo4j instance before generating the final Cap. 5 numbers?

Q4. CLAUDE.md lists "There is no Phase 5" and Phase 4 contains the multi-seed evaluation and scenario comparison, but the 3-layer-framework-and-methodology.md wiki page still describes Fase 5 (Valutazione degli Scenari) as "NON INIZIATO" with a separate scenario table — should Fase 5 be explicitly re-labeled as "integrated into Phase 4" in the wiki to avoid confusion with the CLAUDE.md statement?

Q5. The NDCG metric is implemented as NDCG@9 in the code but stated as NDCG@10 in objectives-and-rqs.md — which should be the canonical value in the thesis (NDCG@9 is more defensible since there are exactly 9 ranked pairs)?

Q6. The Neo4j database name in config.py defaults to `"neo4j"` while CLAUDE.md specifies `Graph_RAG_FL` — was the graph ever built in a named database `Graph_RAG_FL`, and if so does step_3_0 need to be re-run against the correct database before the thesis evaluation scripts are executed?

---

## 6. Suggested next 3 actions (after Fede answers Q1-Q6)

**Action 1 — Commit the current codebase (highest impact, lowest effort).**
All working code is untracked. Before any other work, stage and commit Phase1_PhysicalDT, Phase2_ISMatch, Phase3_GraphRAG, Phase4, common, CLAUDE.md, README.md (cleaned of the duplicate block) and .gitignore. This establishes a reproducibility baseline and unlocks `git diff` for future sessions.

**Action 2 — Fix the /tmp/ paths and os.path usage in Phase 1 (medium impact, medium effort).**
Replace the three `/tmp/` hardcoded paths with `tempfile.gettempdir()` (I-03), then migrate all Phase 1 `os.path` calls to `pathlib.Path` (I-04). This unblocks the Phase 1 pipeline on Windows 11 and brings the two highest-violation phases into compliance with CLAUDE.md Rule 3.

**Action 3 — Resolve the Phase 3 → Phase 4 ΔTC wiring before generating thesis figures (highest methodological impact).**
Answer Q3, then either: (a) run `step_3_5_phase1_integration.py` with a live Neo4j instance to produce a real `step_3_5_bis_delta_tc.json` and re-run Phase 2 step 2.4 and Phase 4 ablation; or (b) document explicitly in Cap. 5.3 that the 0.10 fallback reduction factors were used and that the result is a lower bound on the ΔTC benefit. This directly affects the PoA numbers in Table 5.3.
