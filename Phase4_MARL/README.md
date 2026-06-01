# Phase 4 — Agentic DT for IS Negotiation (OS4)

Implementation of the Phase 4 deliverables defined in
`wiki/thesis/3-layer-framework-and-methodology.md` (Passi 4.1–4.6) and the
roadmap in `wiki/roadmap-fasi-1-2-3.md`.

The package implements:

1. A Gymnasium environment for the bilateral DC–Manufacturing IS negotiation
   MDP (`env/is_negotiation_env.py`).
2. A Yazdanpanah-style shielding layer enforcing four hard normative rules
   (`env/shielding.py`).
3. A closed-form Shapley allocator for the N=2 coalition
   (`agents/shapley.py`, B5).
4. A centralised LP welfare baseline that yields the Price-of-Anarchy
   denominator (`agents/baseline_lp.py`, Passo 4.5).
5. A PPO trainer using Stable-Baselines3 with TensorBoard logging
   (`train/train_ppo.py`).
6. An ablation runner that compares configurations **A0** (RL only) vs **D**
   (full Agentic DT) over 9 scenarios x 5 seeds (`train/run_ablation.py`).

All numerical parameters trace back to the Phase 1 LC outputs (D5) and the
Phase 2 IS-Match results (D2, D6). IS-Match scores per scenario are read
dynamically from `Phase2_ISMatch/results/step_2_4_delta_tc_calibration_lc.csv`
so any rerun of Phase 2 propagates automatically.

## Scenario matrix (9 cells)

The matrix is row-major: DC scale outer, manufacturing tier inner.

| ID | DC          | Tier             | Plant (medium proximity)     |
|----|-------------|------------------|------------------------------|
| S1 | Edge_LC     | LowT 60 C        | LowT_02_Agro_Medium          |
| S2 | Edge_LC     | MidT 90 C        | MidT_04_PaperPulp_Medium     |
| S3 | Edge_LC     | HighT 130 C      | HighT_07_Rubber_Medium       |
| S4 | Mid_LC      | LowT 60 C        | LowT_02_Agro_Medium          |
| S5 | Mid_LC      | MidT 90 C        | MidT_04_PaperPulp_Medium     |
| S6 | Mid_LC      | HighT 130 C      | HighT_07_Rubber_Medium       |
| S7 | Hyperscale_LC | LowT 60 C      | LowT_02_Agro_Medium          |
| S8 | Hyperscale_LC | MidT 90 C      | MidT_04_PaperPulp_Medium     |
| S9 | Hyperscale_LC | HighT 130 C    | HighT_07_Rubber_Medium       |

## Installation

```bash
cd Phase4
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

### Train PPO on the pilot scenario (S5)

```bash
python train/train_ppo.py --scenario S5 --total-timesteps 200000
```

Outputs:

* `models/ppo_S5.zip` and `models/best_ppo_S5/best_model.zip`
* `models/ppo_S5_summary.json`
* TensorBoard logs in `logs/tb/S5/` (launch with `tensorboard --logdir logs/tb`)

### Run the full ablation (A0 vs D, 5 seeds, 9 scenarios)

```bash
python train/run_ablation.py
```

Outputs:

* `results/ablation_results.csv`
* A formatted summary table is printed at the end.

### Tests

```bash
pytest tests/
```

## Important note on HighT scenarios (S3, S6, S9)

HighT tier (T_req = 130–135 C) requires a CO2-HTHP upgrade with mid-range
CAPEX of **1500 EUR/kW** (Obsidian A9/D4). The centralised LP returns a
positive welfare upper bound for all 9 scenarios (HighT: ~826 EUR/kW/yr
marginal, vs LowT/MidT ~1166 EUR/kW/yr) so the IS contract is *economically*
viable everywhere under the default assumptions.

What HighT scenarios may *fail to achieve* is decentralised **negotiation
convergence**. Reasons documented in the thesis Cap. 5.4 / 6.2:

* The Phase 2 IS-Match Score for HighT is 0.46–0.47 ("marginal", below the
  0.60 high-priority threshold) even after ΔTC reduction.
* The shielding rule 4 (upgrade-tech mismatch -> BLOCKED) is more aggressive
  for HighT because CO2_HTHP is the only acceptable choice; agents that
  explore alternatives are penalised more often.
* Marginal pricing (price - amortised floor) is tighter for CO2_HTHP, so the
  feasibility region for both agents to accept positive welfare is smaller.

The ablation runner therefore executes all 9 scenarios but HighT rows may
exhibit lower convergence rates than LowT/MidT -- this is the expected and
scientifically valid result discussed in Cap. 5.4 (Tipping points) and
Cap. 6.2 (Limitations).

## Design decisions (recap)

* **MFG counter-offer**: rule-based; modifies *price + Q_negotiated*
  (price -5%, Q +5% per round) until welfare_mfg >= 0 or shielding floor.
* **Q imbalance** (e.g. Edge_LC × MidT, Q_dc << Q_demand): Q_partial supply
  is allowed; the manufacturer integrates the gap from its backup boiler.
* **Reward**: economic term + IS-Match-uplift bonus (lambda configurable in
  `config/reward_params.py`, default 1.0). Convergence bonus +0.5; timeout
  penalty -1.0; mild shaping -0.1 against shielding-blocked actions.
* **Ablation seeds**: 5 (vs the original 3) for robust 95% CIs.
* **CO2 savings**: tracked in `NegotiationOutcome.co2_saved_t_yr` using a
  default emission factor of 0.20 tCO2/MWh (IEA 2024, natural gas, 90%
  boiler efficiency). Useful for Cap. 5.4 narrative.

## Mapping to Obsidian sources

| Component                         | Obsidian reference                                    |
|-----------------------------------|-------------------------------------------------------|
| MDP formal definition             | `3-layer-framework-and-methodology.md` Passo 4.1     |
| Shielding layer (Yazdanpanah)     | B4 / `concepts/rl-shielding`                          |
| Shapley closed-form N=2           | B5 / `concepts/shapley-value` / Passo 4.3            |
| Centralised LP baseline           | Passo 4.5                                             |
| Ablation A0 vs D                  | Passo 4.6, Table 3.5                                  |
| DC LC parameters                  | `decisioni-implementative.md` D5                      |
| Manufacturing sector profiles     | `decisioni-implementative.md` D2                      |
| LC-only scope                     | `decisioni-implementative.md` D6                      |
| RL training time constraints      | `blocchi-implementativi.md` BA-7                      |
| NetworkX vs Neo4j                 | `blocchi-implementativi.md` BM-11                     |
