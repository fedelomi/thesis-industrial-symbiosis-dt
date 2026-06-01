"""Modal cloud launcher for Phase 4 PPO training (confirmatory + FW3 modes).

This module runs the bilateral DC-MFG IS-negotiation PPO training on Modal
cloud infrastructure. It mirrors the local pipeline (``train/train_ppo.py``,
``train/run_ablation.py``, ``agents/baseline_lp.py``) so the cloud results are
directly comparable to the committed canonical CSVs.

Two experiment modes are supported.

confirmatory:
    30 seeds x 200k timesteps, configs ``A0`` and ``D``, over the 9 LC
    scenarios S1..S9. This is the high-seed confirmation of the canonical
    A0-vs-D Price-of-Anarchy gap (the committed ablation used 10 seeds at 50k).

fw3:
    Adds a fifth shielding rule (the DC individual-rationality floor) as a new
    config ``D_FW3`` and retrains the 3 Edge scenarios (S1, S2, S3). The IR
    floor raises the minimum acceptable heat price from the CAPEX-only floor to
    the full DC break-even price (CAPEX amortisation plus the transport
    residual). Because ``welfare_dc = mwh * (price - price_ir)`` with a
    Q-independent ``price_ir``, clipping the price up to ``price_ir`` guarantees
    ``welfare_dc >= 0`` on every realised offer, which is the empirical test
    that FW3 removes the negative Edge welfare/PoA outcome observed under D.

Design note (FW3 mechanism).
    The framework note describes the IR rule as "BLOCKED -> counter-offer or
    no-op". A literal no-op would set Q=0 and zero BOTH welfares, which would
    drive PoA toward 0 rather than eliminate the negative gap (and so would not
    satisfy the stated goal of removing the negative Edge PoA). We therefore
    implement the IR floor as a price clip (a recoverable WARNING-level rule,
    like the existing economic Rule 3) propagated through
    ``_amortised_capex_floor`` so it also binds the MFG counter-offer. This is
    the only variant that produces a profitable Edge deal with
    ``welfare_dc >= 0``: at break-even the MFG still keeps a strictly positive
    margin on S1..S3, so the deal clears. The clip is surfaced as an explicit
    "Rule 5" entry in the violations log for traceability.

Running.
    The image upload paths are resolved relative to the current working
    directory, so run this from the ``Fasi Applicative`` folder::

        modal run Phase4/modal_train_phase4.py --mode confirmatory
        modal run Phase4/modal_train_phase4.py --mode fw3
        modal run Phase4/modal_train_phase4.py --mode confirmatory --seeds 5 --timesteps 50000

Only ``modal`` and the standard library are imported at module top level so the
file imports cleanly on the launching machine (where the Phase 4 packages and
the Phase 2 CSV are not on the path). Every Phase 4 / SB3 / numpy / torch
import lives inside the function bodies that execute in the container.
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
logger = logging.getLogger("phase4.modal")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
APP_NAME = "phase4-training"
VOLUME_NAME = "phase4-models"
VOLUME_MOUNT = "/models"

# Remote layout inside the container.
REMOTE_CODE_DIR = "/root/Phase4"
# config/scenarios.py resolves DEFAULT_PHASE2_CSV as
# Path(__file__).resolve().parents[2] / "Phase2_ISMatch" / "results" / <csv>.
# With config/scenarios.py at /root/Phase4/config/scenarios.py, parents[2] is
# /root, so build_scenarios() reads the CSV from /root/Phase2_ISMatch/...; we
# add the one required CSV at exactly that path (see image definition).
PHASE2_CSV_REL = "Phase2_ISMatch/results/step_2_4_delta_tc_calibration_lc.csv"
REMOTE_PHASE2_CSV = f"/root/{PHASE2_CSV_REL}"

# Scenario universe.
ALL_SCENARIOS: Tuple[str, ...] = tuple(f"S{i}" for i in range(1, 10))
EDGE_SCENARIOS: Tuple[str, ...] = ("S1", "S2", "S3")
MID_HYPER_SCENARIOS: Tuple[str, ...] = ("S4", "S5", "S6", "S7", "S8", "S9")
CONFIRMATORY_CONFIGS: Tuple[str, ...] = ("A0", "D")

# Training defaults (confirmatory). The spec pins the train_ppo production
# template: SubprocVecEnv n_envs=8, total_timesteps=200_000. ent_coef is kept at
# 0.0 to match the canonical baseline (train_ppo.py uses 0.0, run_ablation.py
# relies on the SB3 default 0.0), so the only difference confirmatory introduces
# is the higher seed count and budget, not the exploration regime.
DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEEDS = 30
N_ENVS = 8
N_EVAL_EPISODES = 30
MAX_ROUNDS = 20

# PPO hyperparameters (train_ppo.py production values). ent_coef=0.0 for
# bit-comparability with the ablation_results.csv canonical baseline
# (train_ppo.py, run_ablation.py). Holding ent_coef fixed keeps the D vs D_FW3
# gap attributable solely to the new IR shielding rule.
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

# --------------------------------------------------------------------------- #
# Modal app, image, volume                                                     #
# --------------------------------------------------------------------------- #
app = modal.App(APP_NAME)

# torch is pulled CPU-only (we request cpu=, not gpu=), which keeps the image
# small and the cold start fast. cvxpy and coolprop are intentionally excluded:
# the centralised welfare for the PoA denominator is replicated in closed form
# (see _centralised_welfare), and nothing in the import chain (config, env,
# agents.shapley) needs coolprop. pandas is required because SB3 Monitor imports
# it at module load.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # The "+cpu" local-version wheel exists only on the PyTorch CPU index,
        # so extra_index_url resolves it unambiguously (PyPI ships the CUDA
        # build under the same bare version). This matches the local 2.11.0+cpu.
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
        "Phase4",
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
# FW3: DC individual-rationality shielding (container-side)                    #
# --------------------------------------------------------------------------- #
def build_ir_shielding(distance_km: float):
    """Build a ``ShieldingLayer`` subclass instance enforcing the DC IR floor.

    Defined as a factory rather than a module-level class because it subclasses
    ``env.shielding.ShieldingLayer``, a container-only import. Calling this on
    the launching machine would fail (the Phase 4 packages are not importable
    there), so the import and the subclass definition happen lazily here, inside
    the container.

    The IR floor is the existing CAPEX amortisation floor plus the transport
    residual per MWh::

        price_ir = _amortised_capex_floor(tech, q) + transport_cost * distance_km

    Setting ``welfare_dc = mwh * (price - price_ir) = 0`` gives this break-even
    price, which is independent of Q (the CAPEX and transport terms both reduce
    to per-MWh constants). Overriding ``_amortised_capex_floor`` propagates the
    raised floor to the three places that call it on the shielding instance: the
    agent-offer economic rule (Rule 3 clip), the MFG counter-offer price clamp,
    and the IS-Match update. That single override is what guarantees
    ``welfare_dc >= 0`` on the realised offer, including the MFG counter-offer.

    Constructor deviation from the bare ``ShieldingLayerWithIRFloor()`` shown in
    the spec: the floor helper signature is fixed by the base class and does not
    receive ``mfg``, so the plant distance must be injected as instance state.
    The factory takes ``distance_km`` for that reason.

    Args:
        distance_km: DC-to-plant network distance (km) for the transport term.

    Returns:
        A ``ShieldingLayerWithIRFloor`` instance bound to ``distance_km``.
    """
    import numpy as np

    from config.reward_params import DEFAULT_REWARD_PARAMS, RewardParams
    from env.models import DCProfile, MfgParams
    from env.shielding import ShieldingDecision, ShieldingLayer

    class ShieldingLayerWithIRFloor(ShieldingLayer):
        """ShieldingLayer plus Rule 5 (DC individual-rationality price floor)."""

        def __init__(
            self,
            distance_km: float,
            reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
        ) -> None:
            """Store the plant distance used by the transport term of the floor.

            Args:
                distance_km: DC-to-plant network distance (km).
                reward_params: Economic parameters (defaults to the canonical set).
            """
            super().__init__(reward_params=reward_params)
            self.distance_km = float(distance_km)

        def _amortised_capex_floor(self, upgrade_tag: str, q_kw: float) -> float:
            """Return the DC break-even price (CAPEX floor plus transport term).

            Args:
                upgrade_tag: Upgrade technology key (e.g. ``"HP"``).
                q_kw: Negotiated thermal flow (kW); kept for signature parity, the
                    per-MWh floor is independent of it.

            Returns:
                Break-even heat price in EUR/MWh.
            """
            base = super()._amortised_capex_floor(upgrade_tag=upgrade_tag, q_kw=q_kw)
            transport_term = (
                self.reward_params.transport_cost_eur_per_mwh_km * self.distance_km
            )
            return base + transport_term

        def _evaluate(
            self,
            action: np.ndarray,
            dc: DCProfile,
            mfg: MfgParams,
        ) -> ShieldingDecision:
            """Run the base rules then record the Rule 5 IR-floor enforcement.

            ``super()._evaluate`` already clips the price up to the IR break-even
            via the overridden ``_amortised_capex_floor`` (Rule 3 uses
            ``self._amortised_capex_floor``). We capture the incoming price first
            and, if it was below the IR floor, append an explicit Rule 5 entry so
            the enforcement is visible in the violations log. The rule is
            WARNING-level (clipped, recoverable) by design, not BLOCKED, so the
            Edge deal still clears at break-even instead of collapsing to a
            zero-welfare no-op.

            Args:
                action: Candidate 5-slot action (mutated in place by the base).
                dc: DC profile.
                mfg: Manufacturing parameters.

            Returns:
                The base ``ShieldingDecision`` with the Rule 5 note appended.
            """
            incoming_price = float(action[2])
            decision = super()._evaluate(action, dc, mfg)
            ir_floor = self._amortised_capex_floor(
                upgrade_tag=mfg.upgrade_required,
                q_kw=float(decision.clipped_action[0]),
            )
            if incoming_price < ir_floor - 1e-9:
                decision.violations.append(
                    f"WARNING: Rule 5 (DC IR floor) enforced: price="
                    f"{incoming_price:.2f} below break-even={ir_floor:.2f} "
                    f"EUR/MWh (CAPEX + transport); clipped."
                )
            return decision

    return ShieldingLayerWithIRFloor(distance_km=distance_km)


def _resolve_config(config: str, mfg) -> Tuple[Optional[object], bool]:
    """Map a config label to its (shielding, dt_grounded) pair.

    Mirrors ``train/run_ablation.py::_make_env`` for A0 and D, and extends it
    with D_FW3 (D plus the Rule 5 IR floor). A0 is a double ablation: no shield
    AND no DT grounding (``dt_grounded = config != "A0"``). D and D_FW3 both use
    ``dt_grounded=True``.

    Args:
        config: One of ``"A0"``, ``"D"``, ``"D_FW3"``.
        mfg: ``MfgParams`` for the scenario (supplies ``distance_km`` for FW3).

    Returns:
        Tuple ``(shielding, dt_grounded)``: shielding is ``None`` for A0, a
        ``ShieldingLayer`` for D, and a ``ShieldingLayerWithIRFloor`` for D_FW3.

    Raises:
        ValueError: If ``config`` is not a recognised label.
    """
    from env.shielding import ShieldingLayer

    if config == "A0":
        return None, False
    if config == "D":
        return ShieldingLayer(), True
    if config == "D_FW3":
        return build_ir_shielding(distance_km=mfg.distance_km), True
    raise ValueError(f"Unknown config {config!r}; expected A0, D or D_FW3.")


def _centralised_welfare(dc, mfg) -> float:
    """Closed-form social-planner welfare for the PoA denominator (no cvxpy).

    Replicates the linear-program fallback in
    ``agents.baseline_lp._solve_one``. The program is linear in Q with a single
    upper bound, so the optimum is the closed form::

        marginal = (gas * cop - transport) * hours / 1000 - capex_per_kw / years
        q_max    = min(Q_available, Q_demand)
        welfare  = marginal * q_max  if marginal > 0 else 0.0

    where ``transport = transport_cost_eur_per_mwh_km * distance_km``. cvxpy is
    avoided so it stays out of the Modal image.

    Args:
        dc: DC profile (supplies ``q_available_kw``).
        mfg: Manufacturing parameters (tech, demand, distance).

    Returns:
        Optimal centralised welfare in EUR/yr.
    """
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
def train_one(
    scenario_id: str,
    seed: int,
    config: str,
    total_timesteps: int = 100_000,
) -> dict:
    """Train one PPO agent and return its evaluation metrics.

    Mirrors ``train/run_ablation.py::_run_one`` (training plus a deterministic
    eval rollout plus PoA against the centralised welfare) but runs inside a
    Modal container and saves the model to the shared Volume.

    Args:
        scenario_id: One of S1..S9 (``config.scenarios.SCENARIO_INDEX`` keys).
        seed: Integer seed controlling PPO init, env reset and eval rollouts.
        config: ``"A0"``, ``"D"`` or ``"D_FW3"`` (see ``_resolve_config``).
        total_timesteps: PPO training budget (200_000 for confirmatory).

    Returns:
        dict with keys: scenario, seed, config, final_poa, welfare_dc,
        welfare_mfg, shapley_fairness, n_rounds, converged, convergence_rate,
        model_path_in_volume.
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
    from env.is_negotiation_env import ISNegotiationEnv

    # One thread per env avoids BLAS oversubscription with 8 SubprocVecEnv workers.
    torch.set_num_threads(1)

    scenarios = build_scenarios()
    dc, mfg = scenarios[scenario_id]

    def _env_factory(rank: int):
        def _init():
            # Build a fresh shielding instance per worker so the forked child
            # owns its own object (the D_FW3 subclass is defined lazily inside
            # build_ir_shielding, which is re-run in each fork).
            local_shielding, local_dt = _resolve_config(config, mfg)
            env = ISNegotiationEnv(
                dc=dc,
                mfg=mfg,
                shielding=local_shielding,
                max_rounds=MAX_ROUNDS,
                dt_grounded=local_dt,
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
        "train_one start scenario=%s config=%s seed=%d timesteps=%d",
        scenario_id,
        config,
        seed,
        total_timesteps,
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()

    # Persist the model to the shared Volume.
    model_rel = f"{config}/{scenario_id}_seed{seed}.zip"
    model_path = Path(VOLUME_MOUNT) / model_rel
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    volume.commit()

    # Deterministic eval rollout (run_ablation convention: a fresh env at
    # seed+1234, reset each episode with seed+ep).
    eval_shielding, eval_dt = _resolve_config(config, mfg)
    eval_env = ISNegotiationEnv(
        dc=dc,
        mfg=mfg,
        shielding=eval_shielding,
        max_rounds=MAX_ROUNDS,
        dt_grounded=eval_dt,
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
    welfare_total = mean_w_dc + mean_w_mfg
    w_central = _centralised_welfare(dc, mfg)
    final_poa = (
        welfare_total / w_central if abs(w_central) > 1e-3 else float("nan")
    )
    convergence_rate = float(np.mean(convs))

    result = {
        "scenario": scenario_id,
        "seed": seed,
        "config": config,
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
        "train_one done scenario=%s config=%s seed=%d poa=%.4f w_dc=%.1f",
        scenario_id,
        config,
        seed,
        final_poa,
        mean_w_dc,
    )
    return result


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #
def _build_jobs(
    mode: str,
    seeds: int,
    timesteps: int,
) -> List[Tuple[str, int, str, int]]:
    """Build the ``(scenario, seed, config, timesteps)`` job list for a mode.

    Args:
        mode: ``"confirmatory"`` or ``"fw3"``.
        seeds: Number of seeds (0..seeds-1).
        timesteps: PPO training budget per run.

    Returns:
        List of starmap-ready argument tuples for ``train_one``.

    Raises:
        ValueError: If ``mode`` is not recognised.
    """
    jobs: List[Tuple[str, int, str, int]] = []
    if mode == "confirmatory":
        for cfg in CONFIRMATORY_CONFIGS:
            for scenario in ALL_SCENARIOS:
                for seed in range(seeds):
                    jobs.append((scenario, seed, cfg, timesteps))
    elif mode == "fw3":
        for scenario in EDGE_SCENARIOS:
            for seed in range(seeds):
                jobs.append((scenario, seed, "D_FW3", timesteps))
    elif mode == "fw3_validation":
        # Anti-attack defence: prove FW3 IR floor is welfare-Pareto improving
        # also on Mid and Hyperscale scenarios where the canonical welfare
        # collapse did not occur. Demonstrates the fix has no side effects
        # on the scales where it was not strictly needed.
        for scenario in MID_HYPER_SCENARIOS:
            for seed in range(seeds):
                jobs.append((scenario, seed, "D_FW3", timesteps))
    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected confirmatory, fw3 or fw3_validation."
        )
    return jobs


def _write_results(rows: List[dict], out_path: Path) -> None:
    """Write the aggregate results to a CSV with a stable column order.

    Args:
        rows: Per-run result dicts returned by ``train_one``.
        out_path: Destination CSV path (parents created if needed).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "scenario",
        "seed",
        "final_poa",
        "welfare_dc",
        "welfare_mfg",
        "shapley_fairness",
        "n_rounds",
        "converged",
        "convergence_rate",
        "model_path_in_volume",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["config"], r["scenario"], r["seed"])):
            writer.writerow({k: row[k] for k in fieldnames})


def _print_summary(rows: List[dict]) -> None:
    """Print a compact per-(config, scenario) mean table.

    Args:
        rows: Per-run result dicts returned by ``train_one``.
    """
    import statistics

    groups: dict = {}
    for row in rows:
        groups.setdefault((row["config"], row["scenario"]), []).append(row)
    header = (
        f"{'cfg':>5} {'scen':>4} | {'PoA':>8} {'W_dc':>12} {'W_mfg':>12} "
        f"{'fair':>6} {'conv':>5}"
    )
    print(header)
    print("-" * len(header))
    for (cfg, scen), recs in sorted(groups.items()):
        poas = [r["final_poa"] for r in recs if r["final_poa"] == r["final_poa"]]
        poa = statistics.fmean(poas) if poas else float("nan")
        w_dc = statistics.fmean(r["welfare_dc"] for r in recs)
        w_mfg = statistics.fmean(r["welfare_mfg"] for r in recs)
        fair = statistics.fmean(r["shapley_fairness"] for r in recs)
        conv = statistics.fmean(r["convergence_rate"] for r in recs)
        print(
            f"{cfg:>5} {scen:>4} | {poa:>8.3f} {w_dc:>12,.0f} {w_mfg:>12,.0f} "
            f"{fair:>6.3f} {conv:>5.2f}"
        )


@app.local_entrypoint()
def main(
    mode: str = "confirmatory",
    seeds: int = DEFAULT_SEEDS,
    timesteps: int = DEFAULT_TIMESTEPS,
) -> None:
    """Dispatch training jobs to Modal and write an aggregate results CSV.

    Args:
        mode: ``"confirmatory"`` (A0 + D over S1..S9) or ``"fw3"`` (D_FW3 over
            the 3 Edge scenarios).
        seeds: Number of seeds per (config, scenario) cell.
        timesteps: PPO training budget per run.
    """
    jobs = _build_jobs(mode=mode, seeds=seeds, timesteps=timesteps)
    logger.info(
        "Dispatching %d jobs (mode=%s, seeds=%d, timesteps=%d) to Modal",
        len(jobs),
        mode,
        seeds,
        timesteps,
    )

    rows: List[dict] = list(train_one.starmap(jobs))

    out_dir = Path(__file__).resolve().parent / "results"
    if mode == "confirmatory":
        out_path = out_dir / f"modal_confirmatory_{seeds}seed_{timesteps // 1000}k.csv"
    elif mode == "fw3_validation":
        out_path = out_dir / f"modal_fw3_validation_{seeds}seed_{timesteps // 1000}k.csv"
    else:
        out_path = out_dir / f"modal_fw3_edge_{seeds}seed_{timesteps // 1000}k.csv"
    _write_results(rows, out_path)
    logger.info("Wrote %d results to %s", len(rows), out_path)
    _print_summary(rows)
