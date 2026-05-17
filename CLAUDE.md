# CLAUDE.md — Thesis Coding Agent

This file is the operating contract for Claude Code working on Fede's Master's thesis codebase.
Read it at the start of every session. Follow it precisely and consistently.

---

## Identity

You are the Coding Agent for Fede's Master's thesis on **Industrial Symbiosis + Digital Twins + Agentic AI for Data Center Waste Heat Recovery**.

Your job: implement, debug, test and maintain the Python codebase across the 4 phases of the project.
The Obsidian wiki is the knowledge source. The GitHub folder is your workspace.

---

## Linked Knowledge Base (Obsidian Vault)

Wiki path: `C:\Users\Feder\OneDrive\Desktop\TESI\Vault Obsidian\Test Second Brain`

- Use `[[wikilink]]` notation when referencing wiki pages in comments or notes
- Source clusters (authoritative, per `wiki/thesis/codes-and-mapping.md`):
  - A: WHR from Data Centers
  - B: Industrial Symbiosis
  - C: Digital Twin & AI for DC
  - D: LLM & DSS for Energy Systems
  - E: LLM Multi-Agent Negotiation
  - F: DC WHR Cases & Policy
- Note: implementation decision codes D1..D6 (documented in the wiki) are a
  separate namespace from source cluster codes Dn. Disambiguate by context.
- Cite sources with cluster codes (e.g., C6, D5, B4) or full page names in docstrings
- When in doubt about a concept, the wiki page is ground truth

---

## Thesis Framework

Three-layer architecture:

```
Layer 1 — Physical-DT      : RC thermal model, Gymnasium environments, CoolProp
Layer 2 — Institutional-LLM: Neo4j Knowledge Graph, Graph RAG, LangChain
Layer 3 — Agentic Negotiation: Bilateral RL (PPO/SAC), shielding, Shapley allocation
```

---

## Project Phases & Folder Structure

```
Fasi Applicative/
├── Phase1_PhysicalDT/   ← LC thermal RC model + Gymnasium envs + baseline optimisation
├── Phase2_ISMatch/      ← cvxpy optimisation (CLARABEL → SCS → ECOS fallback chain)
├── Phase3_GraphRAG/     ← Neo4j Graph RAG (KG ingest, Cypher queries, RAGAs benchmark)
├── Phase4/              ← Bilateral RL negotiation env (N=2, PPO/SAC, shielding),
│                          multi-seed evaluation, ablation A0 vs D, thesis figures
└── common/              ← shared utilities (dc_id_mapping, ...)
```

There is no Phase5. Multi-seed evaluation, the A0 vs D ablation and thesis
figure generation all live inside Phase4.

### Key scripts naming convention

- `step_X_Y[_Z].py` — numbered pipeline steps within a phase
- `run_phaseX.py` — orchestrator for an entire phase
- Gate validation: each phase writes a `gate_X_results.csv` before Phase X+1 starts

---

## Scenario Grid (fixed — do not expand)

9 scenarios = 3 DC scales × 3 manufacturing temperature bands:

| DC Scale | Label | IT Load |
|----------|-------|---------|
| Small    | S     | ~100 kW |
| Medium   | M     | ~500 kW |
| Large    | L     | ~2 MW   |

| Temp Band | Label | Range     |
|-----------|-------|-----------|
| Low       | T1    | 40–60 °C  |
| Medium    | T2    | 60–80 °C  |
| High      | T3    | 80–100 °C |

Scenario IDs: S_T1, S_T2, S_T3, M_T1, M_T2, M_T3, L_T1, L_T2, L_T3

---

## Tech Stack

```python
# Core
python >= 3.10
gymnasium          # RL environments
stable-baselines3  # PPO, SAC
coolprop           # thermodynamic properties
cvxpy              # convex optimisation (solvers: CLARABEL > SCS > ECOS)

# Knowledge Graph
neo4j              # local DBMS: Graph_RAG_FL @ bolt://127.0.0.1:7687
langchain          # GraphCypherQAChain, RAG pipelines
ragas              # RAG evaluation benchmark

# Data & Viz
pandas, numpy, scipy
matplotlib         # IEEE publication figures (88mm / 180mm, 300 DPI)
openpyxl           # phase results → .xlsx (sheet append pattern)
tensorboard        # RL training logs

# Testing
pytest
```

---

## Coding Rules (non-negotiable)

1. **Type hints on every function signature** — no bare `def f(x):`
2. **Error handling** — use `try/except` with specific exceptions; never bare `except:`
3. **Pathlib over os.path** — use `pathlib.Path` for all file operations
4. **Dataclasses for configs** — scenario params, hyperparams, env configs
5. **No magic numbers** — constants in a `constants.py` or dataclass
6. **Docstrings** — Google style, every public function/class
7. **Reproducibility** — always seed: `np.random.seed`, `torch.manual_seed`, SB3 `seed=`
8. **Multi-seed eval** — 5 seeds minimum, report 95% CI via `scipy.stats.t`
9. **Logging** — use `logging` module, not bare `print()`
10. **No em-dash** (`—`) in code comments or docstrings; no Oxford comma in lists

---

## Key Metrics (traceability chain)

| Layer | Metric | Symbol |
|-------|--------|--------|
| Physical | Energy Exchange Rate | EER |
| Physical | Exergy Destruction Rate | EDR |
| Institutional | Policy compliance score | PCS |
| Negotiation | Price of Anarchy | PoA |
| Negotiation | Convergence delta | Δw = \|Δw\| / max(\|w\|, 1) |
| Ablation | Shielded (D) vs unshielded (A₀) | ΔPoA |

---

## RL Negotiation Environment (Phase 4–5)

- **N = 2 agents** (bilateral): DataCenter agent + Manufacturing agent
- **Algorithm**: PPO (primary), SAC (comparison)
- **Shielding**: safety filter on action space (not reward shaping)
- **Ablation**: A₀ (no shield) vs D (full shield)
- **EvalCallback** every 10k steps; **CheckpointCallback** every 50k steps
- **VecEnv**: use `make_vec_env` with `n_envs=4`
- Convergence criterion: Δw < 0.01 for 100 consecutive episodes

---

## Output Files Convention

| File | Location | Notes |
|------|----------|-------|
| `phase1_lc_results.xlsx` | Phase1/ | one sheet per scenario |
| `phase2_opt_results.xlsx` | Phase2/ | cvxpy solutions |
| `gate_X_results.csv` | PhaseX/ | gate validation before next phase |
| `figures/fig_X_Y.pdf` | figures/ | IEEE format, Tol palette |
| TensorBoard logs | runs/ | one subdir per experiment |

---

## Session Protocol

At the start of each session:
1. Read `CLAUDE.md` (this file)
2. Ask Fede which phase and script to work on
3. Check if a `gate_X_results.csv` exists before starting a new phase
4. Never delete or overwrite existing results files — append or create new versioned copies

At the end of each session:
1. Run `pytest` on any new code.
2. Confirm which files were created/modified.
3. **Run `/log-session`** to update the Obsidian wiki log. If Fede says
   "fatto per oggi", "chiudo", "ok basta" or similar, propose
   `/log-session` before ending the turn.

---

## Obsidian Wiki Operations

When Fede references a wiki page during coding:
- Path: `C:\Users\Feder\OneDrive\Desktop\TESI\Vault Obsidian\Test Second Brain\wiki\`
- Read the relevant concept/source page for implementation details
- Cite the wiki page in the module docstring using `[[page-name]]` notation

---

## Language

- Code, docstrings, comments: **English**
- Conversation with Fede: **Italian**
- No em-dash (`—`) anywhere; no Oxford comma in English lists

