"""Ablation runner: 2 configurations x N seeds x 9 scenarios.

Two configurations (Phase 4.6 spec, scoped to A and D per Obsidian):

    A0 -- RL without shielding and without DT-grounding (baseline).
    D  -- Full Agentic DT (RL + DT + shielding + Shapley + IS-Match bonus).

For each (config, seed, scenario) we record:

    ConvergenceRate, TimeoutRate, MeanRounds, WelfareDC, WelfareMFG,
    ShapleyFairness, PriceOfAnarchy (= welfare_D / welfare_LP_baseline).

Results are saved to ``Phase4/results/ablation_results.csv``.

By design decision 2026-05-14 the runner uses **5 seeds** (vs the original 3)
for better statistical robustness. Override with ``--seeds N``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PHASE4_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from agents.baseline_lp import compute_centralized_welfare
from config.scenarios import SCENARIOS, build_scenarios
from env.is_negotiation_env import ISNegotiationEnv
from env.shielding import ShieldingLayer
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger("phase4.ablation")

CONFIGS = ("A0", "D")
DEFAULT_TIMESTEPS = 50_000  # shorter than the pilot training for the ablation


@dataclass(slots=True)
class AblationRecord:
    config: str
    seed: int
    scenario: str
    convergence_rate: float
    timeout_rate: float
    mean_rounds: float
    welfare_dc: float
    welfare_mfg: float
    shapley_fairness: float
    price_of_anarchy: float


def _make_env(scenario_id: str, config: str, seed: int) -> ISNegotiationEnv:
    scenarios = SCENARIOS or build_scenarios()
    dc, mfg = scenarios[scenario_id]
    shielding = None if config == "A0" else ShieldingLayer()
    dt_grounded = config != "A0"
    return ISNegotiationEnv(
        dc=dc,
        mfg=mfg,
        shielding=shielding,
        max_rounds=20,
        dt_grounded=dt_grounded,
        seed=seed,
    )


def _train_quick(
    scenario_id: str,
    config: str,
    seed: int,
    total_timesteps: int,
) -> PPO:
    env_fns = [
        (lambda s=seed + i: Monitor(_make_env(scenario_id, config, s))) for i in range(2)
    ]
    vec_env = DummyVecEnv(env_fns)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        learning_rate=3e-4,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


def _evaluate_policy(
    scenario_id: str,
    config: str,
    seed: int,
    model: PPO,
    n_episodes: int = 30,
) -> Tuple[float, float, float, float, float, float]:
    env = _make_env(scenario_id, config, seed + 1234)
    convs, timeouts, rounds, w_dcs, w_mfgs, fairness = [], [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        outcome = env.outcome()
        convs.append(int(outcome.converged))
        timeouts.append(int(truncated and not outcome.converged))
        rounds.append(outcome.n_rounds)
        w_dcs.append(outcome.welfare_dc)
        w_mfgs.append(outcome.welfare_mfg)
        fairness.append(outcome.shapley_fairness_ratio)
    return (
        float(np.mean(convs)),
        float(np.mean(timeouts)),
        float(np.mean(rounds)),
        float(np.mean(w_dcs)),
        float(np.mean(w_mfgs)),
        float(np.mean(fairness)),
    )


def _run_one(
    config: str,
    seed: int,
    scenario_id: str,
    centralised: Dict[str, float],
    total_timesteps: int,
) -> AblationRecord:
    model = _train_quick(scenario_id, config, seed, total_timesteps)
    conv, timeout, rounds, w_dc, w_mfg, fair = _evaluate_policy(
        scenario_id, config, seed, model
    )
    welfare_total = w_dc + w_mfg
    welfare_lp = centralised.get(scenario_id, float("nan"))
    if welfare_lp and not np.isnan(welfare_lp) and abs(welfare_lp) > 1e-3:
        poa = welfare_total / welfare_lp
    else:
        poa = float("nan")
    return AblationRecord(
        config=config,
        seed=seed,
        scenario=scenario_id,
        convergence_rate=conv,
        timeout_rate=timeout,
        mean_rounds=rounds,
        welfare_dc=w_dc,
        welfare_mfg=w_mfg,
        shapley_fairness=fair,
        price_of_anarchy=poa,
    )


def run_ablation(
    seeds: int = 5,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    scenario_filter: Optional[List[str]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    scenarios = SCENARIOS or build_scenarios()
    scenario_ids = scenario_filter or list(scenarios.keys())
    centralised = compute_centralized_welfare({k: scenarios[k] for k in scenario_ids})

    out_path = out_path or (PHASE4_ROOT / "results" / "ablation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[AblationRecord] = []
    for cfg in CONFIGS:
        for seed in range(seeds):
            for s_id in scenario_ids:
                logger.info("Running cfg=%s seed=%d scenario=%s", cfg, seed, s_id)
                rec = _run_one(cfg, seed, s_id, centralised, total_timesteps)
                records.append(rec)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "config",
                "seed",
                "scenario",
                "convergence_rate",
                "timeout_rate",
                "mean_rounds",
                "welfare_dc",
                "welfare_mfg",
                "shapley_fairness",
                "price_of_anarchy",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(dataclasses.asdict(r))

    _print_summary(records)
    return out_path


def _print_summary(records: List[AblationRecord]) -> None:
    """Print a compact per-(config, scenario) summary table."""
    by_cs: Dict[Tuple[str, str], List[AblationRecord]] = {}
    for r in records:
        by_cs.setdefault((r.config, r.scenario), []).append(r)

    cols = ("conv", "timeout", "rounds", "W_dc", "W_mfg", "fair", "PoA")
    header = f"{'cfg':>3} {'scen':>4} | " + " ".join(f"{c:>8}" for c in cols)
    print(header)
    print("-" * len(header))
    for (cfg, scen), recs in sorted(by_cs.items()):
        means = {
            "conv": np.mean([r.convergence_rate for r in recs]),
            "timeout": np.mean([r.timeout_rate for r in recs]),
            "rounds": np.mean([r.mean_rounds for r in recs]),
            "W_dc": np.mean([r.welfare_dc for r in recs]),
            "W_mfg": np.mean([r.welfare_mfg for r in recs]),
            "fair": np.mean([r.shapley_fairness for r in recs]),
            "PoA": np.nanmean([r.price_of_anarchy for r in recs]),
        }
        row = f"{cfg:>3} {scen:>4} | " + " ".join(
            f"{means[c]:>8.3f}" if abs(means[c]) < 1e4 else f"{means[c]:>8.1e}"
            for c in cols
        )
        print(row)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation A0 vs D")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Subset of S1..S9 to run (default: all).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    run_ablation(
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        scenario_filter=args.scenarios,
    )


if __name__ == "__main__":
    main()
