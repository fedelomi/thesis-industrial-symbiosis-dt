"""train_sac_curriculum.py - FW2 trainer + 2x2 ablation (algo x curriculum) on S_AIR_M.

Attacks the S_AIR_M convergence collapse (D conv rate 0.41 under PPO). Supports the
full attribution ablation over algorithm {ppo, sac} and curriculum {on, off}:

    SAC_curr   = SAC + curriculum   (the FW2 proposal)
    SAC_nocurr = SAC, train on S_AIR_M from scratch (isolates SAC)
    PPO_curr   = PPO + curriculum    (isolates the curriculum under the canonical algo)
    PPO_nocurr = PPO, no curriculum  (= canonical D baseline, should reproduce ~0.41)

Run a single cell (real env, from the Phase4_MARL folder so env / config import):
    python -m fw2_sac.train_sac_curriculum --algo sac --n-seeds 5 --timesteps 200000
    python -m fw2_sac.train_sac_curriculum --algo sac --no-curriculum --n-seeds 5 --timesteps 200000
Run the full 2x2 ablation in one go:
    python -m fw2_sac.train_sac_curriculum --ablation --n-seeds 5 --timesteps 200000
Smoke (stub env, no heavy deps needed beyond sb3/torch):
    python -m fw2_sac.train_sac_curriculum --ablation --smoke

SAC hyperparameters are STARTING POINTS, not tuned. PPO mirrors the canonical
modal_train_phase4.py values (ent_coef=0.0) for a fair baseline.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

LOG = logging.getLogger("fw2.sac")
HERE = Path(__file__).resolve().parent
PHASE4_ROOT = HERE.parent
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

CANONICAL_SEEDS = (0, 7, 17, 42, 123)
ABLATION_CELLS = (("sac", True), ("sac", False), ("ppo", True), ("ppo", False))

SAC_KWARGS = dict(
    policy="MlpPolicy", learning_rate=3e-4, buffer_size=200_000, batch_size=256,
    tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, learning_starts=5_000,
    ent_coef="auto", target_entropy="auto",
)
# PPO mirrors modal_train_phase4.py (canonical baseline, ent_coef=0.0).
PPO_KWARGS = dict(
    policy="MlpPolicy", n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
)


def cell_label(algo: str, curriculum: bool) -> str:
    return f"{algo.upper()}_{'curr' if curriculum else 'nocurr'}"


@dataclass
class Config:
    target_id: str = "S_AIR_M"
    easy_ids: tuple = ("S1", "S4", "S7")
    algo: str = "sac"
    curriculum: bool = True
    ablation: bool = False
    n_seeds: int = 5
    timesteps: int = 200_000
    max_rounds: int = 20
    n_eval_episodes: int = 30
    eval_freq: int = 20_000
    checkpoint_freq: int = 100_000
    curriculum_warmup_frac: float = 0.10
    curriculum_ramp_end_frac: float = 0.70
    curriculum_p_final: float = 0.60
    smoke: bool = False
    out_dir: Path = field(default_factory=lambda: HERE / "results")
    runs_dir: Path = field(default_factory=lambda: HERE / "runs")
    models_dir: Path = field(default_factory=lambda: HERE / "models")


@dataclass
class World:
    scenarios: dict
    target_id: str
    easy_ids: list
    env_factory: Callable
    welfare_central_fn: Optional[Callable]


def build_world(cfg: Config) -> World:
    """Assemble scenarios and factories from the real packages or the stub."""
    if cfg.smoke:
        from fw2_sac import _stub_env as E

        scenarios = E.build_scenarios()
        scenarios[cfg.target_id] = E.build_airside_scenario()

        def env_factory(dc, mfg, seed):
            return E.ISNegotiationEnv(dc=dc, mfg=mfg, shielding=E.ShieldingLayer(),
                                      max_rounds=cfg.max_rounds, dt_grounded=True, seed=seed)

        return World(scenarios, cfg.target_id, list(cfg.easy_ids), env_factory, None)

    from config.reward_params import DEFAULT_REWARD_PARAMS
    from config.scenarios import (CAPEX_EUR_PER_KW, COP_BY_UPGRADE,
                                  build_airside_scenario, build_scenarios)
    from env.is_negotiation_env import ISNegotiationEnv
    from env.shielding import ShieldingLayer

    scenarios = dict(build_scenarios())
    scenarios[cfg.target_id] = build_airside_scenario()

    def env_factory(dc, mfg, seed):
        return ISNegotiationEnv(dc=dc, mfg=mfg, shielding=ShieldingLayer(),
                                max_rounds=cfg.max_rounds, dt_grounded=True, seed=seed)

    def welfare_central_fn(dc, mfg) -> float:
        rp = DEFAULT_REWARD_PARAMS
        up = mfg.upgrade_required
        transport = rp.transport_cost_eur_per_mwh_km * mfg.distance_km
        marginal = (
            (rp.gas_price_eur_per_mwh * COP_BY_UPGRADE[up] - transport)
            * rp.operating_hours_per_year / 1000.0
            - CAPEX_EUR_PER_KW[up] / max(1, rp.amortisation_years)
        )
        q_max = min(dc.q_available_kw, mfg.q_process_kw)
        return float(marginal * q_max) if marginal > 0 else 0.0

    return World(scenarios, cfg.target_id, list(cfg.easy_ids), env_factory, welfare_central_fn)


def _make_env(cfg: Config, world: World, seed: int, curriculum: bool, target_only: bool):
    from stable_baselines3.common.monitor import Monitor

    from fw2_sac.curriculum_env import CurriculumNegotiationEnv

    if target_only or not curriculum:
        easy = [world.target_id]   # train directly on the target (no curriculum)
    else:
        easy = world.easy_ids
    env = CurriculumNegotiationEnv(
        scenarios=world.scenarios, target_id=world.target_id, easy_ids=easy,
        env_factory=world.env_factory, max_rounds=cfg.max_rounds, seed=seed,
    )
    if target_only or not curriculum:
        env.set_curriculum(1.0)
    return Monitor(env)


def _build_model(cfg: Config, algo: str, vec_env, seed: int):
    from stable_baselines3 import PPO, SAC

    if algo == "sac":
        kw = dict(SAC_KWARGS)
        if cfg.smoke:
            kw.update(buffer_size=4_000, batch_size=64, learning_starts=200)
        return SAC(env=vec_env, seed=seed, verbose=0, tensorboard_log=str(cfg.runs_dir), **kw)
    if algo == "ppo":
        kw = dict(PPO_KWARGS)
        if cfg.smoke:
            kw.update(n_steps=256, batch_size=64)
        return PPO(env=vec_env, seed=seed, verbose=0, tensorboard_log=str(cfg.runs_dir), **kw)
    raise ValueError(f"Unknown algo {algo!r}; expected sac or ppo.")


def train_one_seed(cfg: Config, world: World, seed: int, algo: str, curriculum: bool) -> dict:
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from fw2_sac.callbacks import (ConvergenceCallback, CurriculumCallback,
                                   NegotiationMetricsCallback)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    label = cell_label(algo, curriculum)
    tag = f"{world.target_id}__{label}__seed{seed}"
    train_env = DummyVecEnv([lambda: _make_env(cfg, world, seed, curriculum, target_only=False)])
    eval_env = DummyVecEnv([lambda: _make_env(cfg, world, seed + 1234, curriculum, target_only=True)])

    model = _build_model(cfg, algo, train_env, seed)

    callbacks = [
        ConvergenceCallback(window=min(50_000, cfg.timesteps // 4 or 1)),
        NegotiationMetricsCallback(),
        EvalCallback(eval_env, eval_freq=cfg.eval_freq, n_eval_episodes=cfg.n_eval_episodes,
                     deterministic=True, best_model_save_path=str(cfg.models_dir / tag), verbose=0),
        CheckpointCallback(save_freq=cfg.checkpoint_freq, save_path=str(cfg.models_dir / tag),
                           name_prefix="ckpt"),
    ]
    if curriculum:
        callbacks.insert(0, CurriculumCallback(
            total_timesteps=cfg.timesteps, warmup_frac=cfg.curriculum_warmup_frac,
            ramp_end_frac=cfg.curriculum_ramp_end_frac, p_final=cfg.curriculum_p_final))

    LOG.info("%s start %s timesteps=%d", label, tag, cfg.timesteps)
    model.learn(total_timesteps=cfg.timesteps, tb_log_name=tag, callback=callbacks, progress_bar=False)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(cfg.models_dir / f"{tag}.zip"))
    train_env.close()
    eval_env.close()

    result = _final_eval(cfg, world, model, seed)
    result.update({"config": label, "scenario": world.target_id, "seed": seed})
    LOG.info("%s done %s conv_rate=%.2f poa=%.4f w_dc=%.0f", label, tag,
             result["convergence_rate"], result["final_poa"], result["welfare_dc"])
    return result


def _final_eval(cfg: Config, world: World, model, seed: int) -> dict:
    """Deterministic eval on the TARGET scenario only (run_ablation convention)."""
    from fw2_sac.curriculum_env import CurriculumNegotiationEnv

    env = CurriculumNegotiationEnv(
        scenarios=world.scenarios, target_id=world.target_id, easy_ids=[world.target_id],
        env_factory=world.env_factory, max_rounds=cfg.max_rounds, seed=seed + 4321,
    )
    env.set_curriculum(1.0)
    convs, rounds, w_dcs, w_mfgs, fair = [], [], [], [], []
    for ep in range(cfg.n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        oc = env.outcome()
        convs.append(int(oc.converged))
        rounds.append(oc.n_rounds)
        w_dcs.append(oc.welfare_dc)
        w_mfgs.append(oc.welfare_mfg)
        fair.append(oc.shapley_fairness_ratio)
    mean_w_dc, mean_w_mfg = float(np.mean(w_dcs)), float(np.mean(w_mfgs))
    poa = float("nan")
    if world.welfare_central_fn is not None:
        dc, mfg = world.scenarios[world.target_id]
        w_central = world.welfare_central_fn(dc, mfg)
        if abs(w_central) > 1e-3:
            poa = (mean_w_dc + mean_w_mfg) / w_central
    return {
        "final_poa": poa, "welfare_dc": mean_w_dc, "welfare_mfg": mean_w_mfg,
        "shapley_fairness": float(np.mean(fair)),
        "n_rounds": int(round(float(np.mean(rounds)))),
        "convergence_rate": float(np.mean(convs)),
    }


def _aggregate(rows: list) -> None:
    from scipy import stats

    def ic95(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 2:
            return (float(np.mean(v)) if len(v) else float("nan")), 0.0
        return float(np.mean(v)), float(stats.sem(v, ddof=1) * stats.t.ppf(0.975, len(v) - 1))

    cells = {}
    for r in rows:
        cells.setdefault(r["config"], []).append(r)
    LOG.info("=== ABLATION SUMMARY (S_AIR_M, baseline to beat: PPO_nocurr ~0.41) ===")
    LOG.info("%-12s %5s | %-18s %-18s %-12s", "cell", "n", "conv_rate", "PoA", "w_dc")
    for label, recs in sorted(cells.items()):
        c_m, c_h = ic95([r["convergence_rate"] for r in recs])
        p_m, p_h = ic95([r["final_poa"] for r in recs])
        w_m, _ = ic95([r["welfare_dc"] for r in recs])
        LOG.info("%-12s %5d | %.3f +/- %.3f     %.3f +/- %.3f     %+.0f",
                 label, len(recs), c_m, c_h, p_m, p_h, w_m)


def run(cfg: Config) -> list:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    world = build_world(cfg)
    seeds = (0,) if cfg.smoke else list(CANONICAL_SEEDS)[: cfg.n_seeds]
    cells = list(ABLATION_CELLS) if cfg.ablation else [(cfg.algo, cfg.curriculum)]

    rows = []
    for algo, curr in cells:
        for s in seeds:
            rows.append(train_one_seed(cfg, world, s, algo, curr))

    import csv

    name = "ablation" if cfg.ablation else cell_label(cfg.algo, cfg.curriculum)
    out_path = cfg.out_dir / f"fw2_{name}_{cfg.target_id}_{len(seeds)}seed_{cfg.timesteps // 1000}k.csv"
    fields = ["config", "scenario", "seed", "final_poa", "welfare_dc", "welfare_mfg",
              "shapley_fairness", "n_rounds", "convergence_rate"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["config"], r["seed"])):
            w.writerow({k: r[k] for k in fields})
    LOG.info("Wrote %d rows to %s", len(rows), out_path)
    if not cfg.smoke:
        _aggregate(rows)
    return rows


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--no-curriculum", action="store_true", help="train on the target from scratch")
    p.add_argument("--ablation", action="store_true", help="run the full 2x2 (algo x curriculum)")
    p.add_argument("--smoke", action="store_true", help="tiny run on the stub env")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--target", type=str, default="S_AIR_M")
    a = p.parse_args()
    cfg = Config(target_id=a.target, algo=a.algo, curriculum=not a.no_curriculum,
                 ablation=a.ablation, n_seeds=a.n_seeds, timesteps=a.timesteps, smoke=a.smoke)
    if a.smoke:
        cfg.timesteps = 1_500
        cfg.n_eval_episodes = 5
        cfg.eval_freq = 500
        cfg.checkpoint_freq = 1_000
    return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(_parse_args())
