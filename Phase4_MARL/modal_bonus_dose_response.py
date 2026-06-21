"""Modal launcher for the reward-bonus dose-response sweep (Edge scenarios).

Purpose. Section 5.5 of the thesis attributes the Edge welfare collapse under
configuration D to a reward-shaping mis-incentive: the IS-Match uplift bonus
plus the convergence bonus steer the DC agent to concede surplus to secure
agreement, and on thin Edge margins the concession turns ``welfare_dc``
negative. That mechanism was established forensically (cell-by-cell audit,
R1-binding falsification) and fixed by FW3. This experiment establishes it
CAUSALLY as a dose-response law: both shaping bonuses are scaled by a single
coefficient ``bonus_scale`` in {0.0, 0.25, 0.5, 0.75, 1.0, 1.5} and the Edge
welfare is measured at each dose. ``bonus_scale=1.0`` reproduces canonical D
(validation anchor); ``bonus_scale=0.0`` is the previously missing
"shield-only, no bonus" ablation arm.

Pre-registered predictions (2026-06-13, before the run):
    P1. ``welfare_dc`` on Edge decreases monotonically with ``bonus_scale``.
    P2. The zero crossing of ``welfare_dc`` lies between scale 0.5 and 1.0.
    P3. At scale 0.0 (shield-only) the Edge welfare is statistically close to
        the A0 confirmatory level (~+247 kEUR/yr): shielding alone is
        harmless, completing the per-component attribution that the two-arm
        A0-vs-D ablation could not provide.
    P4. The Shapley fairness collapse follows the same dose curve as the
        welfare (single coupled mechanism, Section 5.8).

What is scaled and what is not. Only the two POSITIVE shaping terms named by
the Section 5.5 mechanism are scaled: ``lambda_is_match_uplift`` (canonical
1.0) and the convergence bonus (canonical +0.5). The timeout penalty (-1.0)
and the blocked penalty (-0.1) are kept canonical, so convergence remains
preferred to timeout at every dose.

Canonical-preservation. No repository file is modified: the scaled reward is
implemented container-side as a subclass of ``ISNegotiationEnv`` overriding
``_compute_reward`` (body pinned to the canonical implementation of
``env/is_negotiation_env.py``). Shielding (canonical ShieldingLayer) and
``dt_grounded=True`` match configuration D exactly.

Running. From the ``Fasi Applicative`` folder (image paths are CWD-relative)::

    modal run Phase4_MARL/modal_bonus_dose_response.py                  # full sweep
    modal run Phase4_MARL/modal_bonus_dose_response.py --seeds 2 --timesteps 5000 --scales 0,1
                                                                        # smoke (~cents)

Budget. Full sweep = 6 scales x 3 Edge scenarios x 10 seeds = 180 jobs at
100k timesteps, cpu=8: comparable compute to the FW3 Edge run (90 jobs at
200k, ~1.5 USD of Modal credits), estimate 1.5-2.5 USD on Modal credits (not
the Anthropic API budget).

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("phase4.modal.dose")

# --------------------------------------------------------------------------- #
# Constants (mirroring modal_train_phase4.py)                                  #
# --------------------------------------------------------------------------- #
APP_NAME = "phase4-bonus-dose-response"
VOLUME_NAME = "phase4-models"
VOLUME_MOUNT = "/models"
REMOTE_CODE_DIR = "/root/Phase4"

# Rename-proof local package dir (the repo folder was renamed Phase4 ->
# Phase4_MARL after the FW3 runs; resolve it from this file's location).
_LOCAL_PKG_DIR = Path(__file__).resolve().parent.name

PHASE2_CSV_REL = "Phase2_ISMatch/results/step_2_4_delta_tc_calibration_lc.csv"
REMOTE_PHASE2_CSV = f"/root/{PHASE2_CSV_REL}"

EDGE_SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3")
DEFAULT_SCALES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
DEFAULT_TIMESTEPS = 100_000          # canonical ablation budget
DEFAULT_SEEDS = 10                   # canonical seed count
N_ENVS = 8
N_EVAL_EPISODES = 30
MAX_ROUNDS = 20

# PPO hyperparameters: identical to modal_train_phase4.py / train_ppo.py so the
# dose axis is the only difference vs the canonical D runs.
PPO_KWARGS = {
    "policy": "MlpPolicy",
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
}

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0+cpu",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "stable-baselines3==2.8.0",
        "gymnasium==1.2.2",
        "numpy==2.4.4",
        "pandas>=2.2",
        "scipy>=1.10",
    )
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(PHASE2_CSV_REL, remote_path=REMOTE_PHASE2_CSV)
    .add_local_dir(
        _LOCAL_PKG_DIR,
        remote_path=REMOTE_CODE_DIR,
        ignore=[
            "**/__pycache__",
            "**/*.zip",
            "results/**",
            "logs/**",
            "models/**",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Container-side helpers                                                       #
# --------------------------------------------------------------------------- #
def build_scaled_env_cls(bonus_scale: float):
    """Return an ISNegotiationEnv subclass with both shaping bonuses scaled.

    The override replicates the canonical ``_compute_reward`` body of
    ``env/is_negotiation_env.py`` (reward_econ + lambda*uplift + 0.5*converged
    - 0.1*blocked) and multiplies ONLY the two positive shaping terms by
    ``bonus_scale``. Negative penalties stay canonical. Defined lazily so the
    class is rebuilt inside each forked SubprocVecEnv worker.

    Args:
        bonus_scale: Multiplier applied to the IS-Match uplift bonus and to
            the convergence bonus (1.0 = canonical D, 0.0 = shield-only).

    Returns:
        The subclass, ready to instantiate with the usual env kwargs.
    """
    from env.is_negotiation_env import IS_MATCH_TARGET_GAP, ISNegotiationEnv

    class ScaledBonusEnv(ISNegotiationEnv):
        """Configuration D with dose-scaled shaping bonuses."""

        _bonus_scale: float = float(bonus_scale)

        def _compute_reward(
            self,
            w_dc: float,
            is_match_pre: float,
            is_match_post: float,
            converged: bool,
            timed_out: bool,
            blocked: bool,
        ) -> float:
            # Pinned to env/is_negotiation_env.py::_compute_reward (canonical),
            # with the two positive shaping terms multiplied by _bonus_scale.
            reward_econ = w_dc / 8.0e5
            uplift = is_match_post - is_match_pre
            reward_is_match = (
                self._bonus_scale
                * self.reward_params.lambda_is_match_uplift
                * (uplift / IS_MATCH_TARGET_GAP)
            )
            reward = reward_econ + reward_is_match
            if converged:
                reward += 0.5 * self._bonus_scale
            if blocked:
                reward -= 0.1
            return float(reward)

    return ScaledBonusEnv


def _centralised_welfare(dc, mfg) -> float:
    """Closed-form social-planner welfare (PoA denominator), as in
    modal_train_phase4.py::_centralised_welfare."""
    from config.reward_params import DEFAULT_REWARD_PARAMS
    from config.scenarios import CAPEX_EUR_PER_KW, COP_BY_UPGRADE

    rp = DEFAULT_REWARD_PARAMS
    upgrade = mfg.upgrade_required
    capex_per_kw = CAPEX_EUR_PER_KW[upgrade]
    cop = COP_BY_UPGRADE[upgrade]
    transport = rp.transport_cost_eur_per_mwh_km * mfg.distance_km
    marginal = (
        (rp.gas_price_eur_per_mwh * cop - transport)
        * rp.operating_hours_per_year
        / 1000.0
        - capex_per_kw / max(1, rp.amortisation_years)
    )
    q_max = min(dc.q_available_kw, mfg.q_process_kw)
    return float(marginal * q_max) if marginal > 0 else 0.0


# --------------------------------------------------------------------------- #
# Remote training function                                                     #
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=8.0,
    timeout=2 * 60 * 60,
)
def train_one_dose(
    scenario_id: str,
    seed: int,
    bonus_scale: float,
    total_timesteps: int = DEFAULT_TIMESTEPS,
) -> dict:
    """Train one PPO agent at one bonus dose and return evaluation metrics.

    Mirrors modal_train_phase4.py::train_one (training, deterministic eval,
    PoA vs centralised welfare) with configuration D held fixed (canonical
    ShieldingLayer, dt_grounded=True) and only the bonus dose varying.

    Args:
        scenario_id: One of S1..S3 (Edge).
        seed: Integer seed for PPO init, env reset and eval rollouts.
        bonus_scale: Multiplier on the two shaping bonuses (see module doc).
        total_timesteps: PPO training budget per run.

    Returns:
        dict with scenario, seed, bonus_scale and the standard metric set.
    """
    import sys

    if REMOTE_CODE_DIR not in sys.path:
        sys.path.insert(0, REMOTE_CODE_DIR)

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from config.scenarios import build_scenarios
    from env.shielding import ShieldingLayer

    torch.set_num_threads(1)
    scenarios = build_scenarios()
    dc, mfg = scenarios[scenario_id]

    def _env_factory(rank: int):
        def _init():
            env_cls = build_scaled_env_cls(bonus_scale)
            env = env_cls(
                dc=dc,
                mfg=mfg,
                shielding=ShieldingLayer(),
                max_rounds=MAX_ROUNDS,
                dt_grounded=True,
                seed=seed + rank,
            )
            return Monitor(env)

        return _init

    vec_env = SubprocVecEnv(
        [_env_factory(i) for i in range(N_ENVS)],
        start_method="fork",
    )
    model = PPO(env=vec_env, seed=seed, verbose=0, **PPO_KWARGS)
    logger.info(
        "dose start scenario=%s scale=%.2f seed=%d timesteps=%d",
        scenario_id, bonus_scale, seed, total_timesteps,
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()

    model_rel = f"D_BONUS{bonus_scale:g}/{scenario_id}_seed{seed}.zip"
    model_path = Path(VOLUME_MOUNT) / model_rel
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    volume.commit()

    env_cls = build_scaled_env_cls(bonus_scale)
    eval_env = env_cls(
        dc=dc,
        mfg=mfg,
        shielding=ShieldingLayer(),
        max_rounds=MAX_ROUNDS,
        dt_grounded=True,
        seed=seed + 1234,
    )
    convs, rounds, w_dcs, w_mfgs, fairness = [], [], [], [], []
    for ep in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=seed + ep)
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = eval_env.step(action)
        outcome = eval_env.outcome()
        convs.append(int(outcome.converged))
        rounds.append(outcome.n_rounds)
        w_dcs.append(outcome.welfare_dc)
        w_mfgs.append(outcome.welfare_mfg)
        fairness.append(outcome.shapley_fairness_ratio)

    mean_w_dc = float(np.mean(w_dcs))
    mean_w_mfg = float(np.mean(w_mfgs))
    w_central = _centralised_welfare(dc, mfg)
    welfare_total = mean_w_dc + mean_w_mfg
    final_poa = welfare_total / w_central if abs(w_central) > 1e-3 else float("nan")
    convergence_rate = float(np.mean(convs))

    result = {
        "scenario": scenario_id,
        "seed": seed,
        "bonus_scale": float(bonus_scale),
        "final_poa": final_poa,
        "welfare_dc": mean_w_dc,
        "welfare_mfg": mean_w_mfg,
        "shapley_fairness": float(np.mean(fairness)),
        "n_rounds": int(round(float(np.mean(rounds)))),
        "converged": bool(convergence_rate >= 0.5),
        "convergence_rate": convergence_rate,
        "model_path_in_volume": model_rel,
    }
    logger.info(
        "dose done scenario=%s scale=%.2f seed=%d poa=%.4f w_dc=%.1f",
        scenario_id, bonus_scale, seed, final_poa, mean_w_dc,
    )
    return result


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #
def _write_results(rows: List[dict], out_path: Path) -> None:
    """Write per-run results to CSV with a stable column order."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bonus_scale", "scenario", "seed", "final_poa", "welfare_dc",
        "welfare_mfg", "shapley_fairness", "n_rounds", "converged",
        "convergence_rate", "model_path_in_volume",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["bonus_scale"], r["scenario"],
                                               r["seed"])):
            writer.writerow({k: row[k] for k in fieldnames})


def _print_dose_curve(rows: List[dict]) -> None:
    """Print the Edge-aggregate dose curve and a naive zero-crossing estimate."""
    import statistics

    by_scale: dict = {}
    for row in rows:
        by_scale.setdefault(row["bonus_scale"], []).append(row)
    header = (f"{'scale':>6} | {'W_dc Edge':>12} {'Shapley':>8} {'PoA':>8} "
              f"{'conv':>5} {'rounds':>6}")
    print(header)
    print("-" * len(header))
    curve: List[Tuple[float, float]] = []
    for scale in sorted(by_scale):
        recs = by_scale[scale]
        w_dc = statistics.fmean(r["welfare_dc"] for r in recs)
        fair = statistics.fmean(r["shapley_fairness"] for r in recs)
        poas = [r["final_poa"] for r in recs if r["final_poa"] == r["final_poa"]]
        poa = statistics.fmean(poas) if poas else float("nan")
        conv = statistics.fmean(r["convergence_rate"] for r in recs)
        rnds = statistics.fmean(r["n_rounds"] for r in recs)
        curve.append((scale, w_dc))
        print(f"{scale:>6.2f} | {w_dc:>12,.0f} {fair:>8.3f} {poa:>8.3f} "
              f"{conv:>5.2f} {rnds:>6.1f}")
    for (s0, w0), (s1, w1) in zip(curve, curve[1:]):
        if w0 > 0 >= w1:
            cross = s0 + (s1 - s0) * (w0 / (w0 - w1))
            print(f"\nzero crossing of W_dc (linear interp): bonus_scale "
                  f"~ {cross:.2f}  [P2 band: 0.50-1.00]")
            break
    print("\nanchors: scale=1.0 should track canonical D Edge "
          "(~-7.4 kEUR/yr at 10x100k); scale=0.0 is the shield-only arm "
          "(P3: expected near the A0 confirmatory ~+247 kEUR/yr).")


@app.local_entrypoint()
def main(
    scales: str = "",
    seeds: int = DEFAULT_SEEDS,
    timesteps: int = DEFAULT_TIMESTEPS,
) -> None:
    """Dispatch the dose-response jobs to Modal and aggregate the results.

    Args:
        scales: Comma-separated bonus scales (default: the pre-registered
            six-point grid 0,0.25,0.5,0.75,1,1.5).
        seeds: Seeds per (scale, scenario) cell.
        timesteps: PPO budget per run.
    """
    scale_grid: Tuple[float, ...] = (
        tuple(float(s) for s in scales.split(",") if s.strip())
        if scales.strip() else DEFAULT_SCALES
    )
    jobs: List[Tuple[str, int, float, int]] = [
        (scenario, seed, scale, timesteps)
        for scale in scale_grid
        for scenario in EDGE_SCENARIOS
        for seed in range(seeds)
    ]
    logger.info(
        "Dispatching %d dose-response jobs (scales=%s, seeds=%d, "
        "timesteps=%d) to Modal", len(jobs), scale_grid, seeds, timesteps,
    )

    rows: List[dict] = list(train_one_dose.starmap(jobs))

    out_dir = Path(__file__).resolve().parent / "results"
    out_path = (out_dir /
                f"modal_bonus_dose_{seeds}seed_{timesteps // 1000}k.csv")
    _write_results(rows, out_path)
    logger.info("Wrote %d results to %s", len(rows), out_path)
    _print_dose_curve(rows)
