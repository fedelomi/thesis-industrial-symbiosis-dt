# FW2 — SAC + curriculum + shield-in-env (airside arm S_AIR_M)

Scaffold for Future Work item FW2: replace the on-policy PPO with off-policy,
entropy-regularised SAC plus a difficulty curriculum, to fix the S_AIR_M
convergence collapse (PPO config D converges on only 41 percent of seeds).

This package is **additive**. It does not modify `modal_train_phase4.py`, the
`env/` package or the canonical PPO results. It reuses the real interfaces
(`env.is_negotiation_env.ISNegotiationEnv`, `env.shielding.ShieldingLayer`,
`config.scenarios.build_scenarios` / `build_airside_scenario`).

## Design note (corrected 2026-06-01)

The first real run surfaced the true action architecture: the policy emits a
**4-dim** action, and the env augments it to a **5-slot** action internally before
the shield reads it (`action[4]`). The shield is therefore already inside the env
and the agent action sits upstream of it. Consequence: keeping the shield in the
env is both simpler and correct, and the off-policy replay buffer needs no
surgery, because `(state, agent_action)` already determines the transition. The
earlier "extract the shield into the wrapper plus a shielded replay buffer" design
was based on a wrong 5-dim agent-action assumption and was removed.

## Files

| File | Role |
|---|---|
| `curriculum_env.py` | Env that applies the shield once in `step` (exposing the executed action in `info["shielded_action"]`) and samples scenarios per a curriculum probability `p_target`. |
| `shielded_replay_buffer.py` | **Optional, reference only, not used by default.** SB3 `ReplayBuffer` subclass that stores the executed action. Unnecessary in this architecture (see Design note); keep only if you ever make the 5-slot shield input the agent action. |
| `callbacks.py` | `CurriculumCallback` (anneals `p_target`), `ConvergenceCallback` (`delta_w_rel`, same metric as the PPO curves), `NegotiationMetricsCallback`. |
| `train_sac_curriculum.py` | Multi-seed SAC trainer wiring all of the above, with EvalCallback + CheckpointCallback + TensorBoard, mirroring the modal PPO result columns. |
| `_stub_env.py` | Minimal stand-in for the real packages so the scaffold can be smoke tested offline. |

## Run

Real env (from the `Phase4_MARL` folder so `env` and `config` import):

```bash
python -m fw2_sac.train_sac_curriculum --n-seeds 5 --timesteps 2000000
```

Offline smoke test (uses `_stub_env`, no real packages needed):

```bash
python -m fw2_sac.train_sac_curriculum --smoke
```

Output: `results/sac_curriculum_S_AIR_M_<n>seed_<k>k.csv` with the same columns
as `modal_confirmatory_*.csv`, plus a TensorBoard run under `runs/`.

## The three FW2 pieces, and why

1. **SAC (off-policy, entropy-regularised).** PPO is on-policy and its mild
   entropy bonus lets the policy collapse into a local optimum on the
   out-of-distribution airside reward landscape. SAC reuses a replay buffer (it
   revisits the rare airside transitions many times) and the max-entropy
   objective with auto temperature keeps exploring until the value estimate is
   confident. That is the direct counter to the 41 percent collapse.

2. **Curriculum.** The agent starts on the easy in-distribution scenarios
   (`S1`, `S4`, `S7`, one per scale) and `p_target` is ramped so S_AIR_M is
   introduced gradually. The agent transfers a competent policy into the hard
   regime instead of facing it cold. The ramp stops below 1.0 so the easy
   scenarios keep being rehearsed (anti forgetting). **Evaluation is always on
   S_AIR_M only**, so the reported conv rate is honest.

3. **Shield stays in the env (corrected design).** The real `ISNegotiationEnv`
   already applies the shield inside `step`, and the agent action is 4-dim,
   upstream of the env's internal 4-to-5 slot augmentation. The whole chain
   `agent_action -> augment -> shield -> dynamics` is a deterministic function of
   `(state, agent_action)`, so the transition the off-policy buffer stores is
   already consistent: no phantom-Q problem at the agent-action level, and the
   default SB3 replay buffer is correct. The curriculum env therefore does NOT
   touch the shield, it only samples scenarios and delegates `step`.

## Honest caveats (read before trusting a number)

- **Not validated against the real env here.** In this checkout the `env/` and
  `config/` packages were empty (submodule not populated), so the scaffold was
  smoke tested only against `_stub_env`. Run `--smoke` first, then a short real
  run (`--timesteps 50000 --n-seeds 1`) on your populated copy before a full run.
- **SAC hyperparameters are starting points, not tuned.** Expect to sweep
  `learning_rate`, `batch_size`, `tau`, `learning_starts` and the curriculum
  schedule. SAC can trade the PPO local optimum for its own instability
  (Q divergence, temperature collapse). Success is not assumed.
- **Default replay buffer** is used (see Design note). The `ShieldedReplayBuffer`
  is left in the tree for reference only and is not on the training path.
- **PoA needs the real centralised welfare.** It is computed only in real mode
  (closed form replicated from `modal_train_phase4._centralised_welfare`); under
  the stub it is `nan`.

## Success criterion

FW2 works only if the **multi-seed** S_AIR_M convergence rate rises clearly above
the PPO baseline of 0.41 with a 95 percent CI that excludes it, without
degrading the 9-LC PoA. Report it the same way as the canonical results: mean and
t-distribution CI over the seed set `(0, 7, 17, 42, 123)`.

## Ablation (2x2: algorithm x curriculum) and cloud

To attribute the fix to SAC, the curriculum, or their interaction, run the full
2x2 on S_AIR_M. Each cell is 5 seeds (0, 7, 17, 42, 123).

Local, full ablation in one command:

```bash
python -m fw2_sac.train_sac_curriculum --ablation --n-seeds 5 --timesteps 200000
```

Single cells:

```bash
python -m fw2_sac.train_sac_curriculum --algo sac                  # SAC + curriculum
python -m fw2_sac.train_sac_curriculum --algo sac --no-curriculum  # SAC alone
python -m fw2_sac.train_sac_curriculum --algo ppo                  # PPO + curriculum
python -m fw2_sac.train_sac_curriculum --algo ppo --no-curriculum  # = canonical D (~0.41)
```

| label | algorithm | curriculum | isolates |
|---|---|---|---|
| SAC_curr | SAC | on | the full FW2 proposal |
| SAC_nocurr | SAC | off | the SAC contribution alone |
| PPO_curr | PPO | on | the curriculum contribution alone |
| PPO_nocurr | PPO | off | the canonical D baseline (reproduces ~0.41) |

The runner prints a per-cell table (conv_rate, PoA, welfare_dc with 95 percent CI)
and writes `results/fw2_ablation_S_AIR_M_5seed_200k.csv`.

Cloud (Modal, `../modal_fw2_ablation.py`):

```bash
modal run modal_fw2_ablation.py --seeds 5 --timesteps 200000
```

It reuses this package inside the container (single source of truth) and mirrors
the image and volume config of `modal_train_phase4.py`. Verify the `add_local_dir`
path matches `modal_train_phase4.py` before a full 20-run launch.

## Reading the result

FW2 is validated only if `SAC_curr` conv_rate (mean over 5 seeds, 95 percent CI)
clearly excludes the `PPO_nocurr` baseline (~0.41), AND `welfare_dc` stays positive
on every seed. A high conv_rate with negative welfare_dc is the hollow-convergence
Edge-D trap, not a win. The 2x2 then tells you whether SAC, the curriculum, or both
are responsible. The preliminary single-seed run (50k) gave SAC_curr conv_rate 0.97,
PoA 0.96, welfare_dc +1.87 M, fairness 0.76: healthy, but n=1 and untuned.
