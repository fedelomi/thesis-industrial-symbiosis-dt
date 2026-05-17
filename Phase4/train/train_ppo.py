"""PPO training script for the IS-negotiation env.

Default scenario: S5 (Mid_LC x MidT_90C). Override with ``--scenario S1..S9``.

Usage::

    python train/train_ppo.py --scenario S5 --total-timesteps 200000

The script:

    1. Builds the requested scenario.
    2. Instantiates an SB3 ``SubprocVecEnv`` with 8 parallel envs.
    3. Trains PPO with the hyperparameters specified in the Phase 4 task spec.
    4. Logs to TensorBoard (``Phase4/logs/tb/<scenario>``).
    5. Saves the model to ``Phase4/models/ppo_<scenario>.zip``.
    6. Runs a final eval rollout and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PHASE4_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from config.scenarios import SCENARIOS, build_scenarios
from env.is_negotiation_env import ISNegotiationEnv
from env.shielding import ShieldingLayer

logger = logging.getLogger("phase4.train")


def make_env_factory(scenario_id: str, seed: int):
    def _init():
        scenarios = SCENARIOS or build_scenarios()
        dc, mfg = scenarios[scenario_id]
        env = ISNegotiationEnv(
            dc=dc,
            mfg=mfg,
            shielding=ShieldingLayer(),
            max_rounds=20,
            dt_grounded=True,
            seed=seed,
        )
        return Monitor(env)
    return _init


def train(
    scenario_id: str,
    total_timesteps: int = 200_000,
    n_envs: int = 8,
    seed: int = 0,
    out_dir: Optional[Path] = None,
) -> Path:
    """Train a PPO agent on the given scenario and return the model path."""
    out_dir = out_dir or (PHASE4_ROOT / "models")
    log_dir = PHASE4_ROOT / "logs" / "tb" / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env_factory(scenario_id, seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    eval_env_fns = [make_env_factory(scenario_id, seed + 1000)]
    eval_env = SubprocVecEnv(eval_env_fns)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / f"best_ppo_{scenario_id}"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(1, 10_000 // n_envs),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        seed=seed,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model_path = out_dir / f"ppo_{scenario_id}.zip"
    model.save(str(model_path))
    vec_env.close()
    eval_env.close()

    # Quick post-training eval summary
    summary = _evaluate(scenario_id, model_path, seed=seed + 9999)
    summary_path = out_dir / f"ppo_{scenario_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved model to %s and summary to %s", model_path, summary_path)
    return model_path


def _evaluate(scenario_id: str, model_path: Path, seed: int = 0, n_episodes: int = 20) -> dict:
    scenarios = SCENARIOS or build_scenarios()
    dc, mfg = scenarios[scenario_id]
    env = ISNegotiationEnv(
        dc=dc,
        mfg=mfg,
        shielding=ShieldingLayer(),
        max_rounds=20,
        dt_grounded=True,
        seed=seed,
    )
    model = PPO.load(str(model_path))

    rounds, welfares_dc, welfares_mfg, convergence, uplifts = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
        outcome = env.outcome()
        rounds.append(outcome.n_rounds)
        welfares_dc.append(outcome.welfare_dc)
        welfares_mfg.append(outcome.welfare_mfg)
        convergence.append(int(outcome.converged))
        uplifts.append(outcome.is_match_uplift)

    return {
        "scenario": scenario_id,
        "n_episodes": n_episodes,
        "convergence_rate": float(np.mean(convergence)),
        "mean_rounds": float(np.mean(rounds)),
        "mean_welfare_dc_eur_yr": float(np.mean(welfares_dc)),
        "mean_welfare_mfg_eur_yr": float(np.mean(welfares_mfg)),
        "mean_is_match_uplift": float(np.mean(uplifts)),
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on IS-negotiation env")
    p.add_argument("--scenario", default="S5", choices=[f"S{i}" for i in range(1, 10)])
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    train(
        scenario_id=args.scenario,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
