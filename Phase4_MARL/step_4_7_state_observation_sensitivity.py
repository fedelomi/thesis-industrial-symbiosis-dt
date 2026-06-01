"""
Step 4.7 - State Observation Sensitivity
=========================================
Phase 4 robustness layer (additive, no retraining)

Measures the local sensitivity of the trained PPO policy to small
(+/-10%) perturbations of each of the 8 observation dimensions. For
each reference state and each (dim, direction) we compute the L2
distance between the baseline action and the action under the
perturbed observation. Mean L2 distance over the 50 reference states
acts as the per-dimension sensitivity.

Reference checkpoint: `models/ppo_S5.zip` (Mid_LC x MidT, the only
saved checkpoint at present). Observation layout (per
`env/is_negotiation_env.py`):

    [Q_avail/Q_inst, T_supply/100, exergy, is_match,
     round_norm, Q_proc/Q_avail_ref, T_req/200, budget_norm]

All entries are normalised to [0, 1] by construction, so perturbations
are clipped to that range before inference.

Outputs:
    results/robustness/step_4_7_state_sensitivity.csv      per-(dim,dir)
    results/robustness/step_4_7_dimension_ranking.csv      aggregated
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE4_ROOT = Path(__file__).resolve().parent
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from stable_baselines3 import PPO

ROBUSTNESS_DIR = PHASE4_ROOT / "results" / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = PHASE4_ROOT / "models" / "ppo_S5.zip"

OUT_PER_DIM = ROBUSTNESS_DIR / "step_4_7_state_sensitivity.csv"
OUT_RANKING = ROBUSTNESS_DIR / "step_4_7_dimension_ranking.csv"

OBS_DIM = 8
DIM_NAMES = (
    "Q_avail_over_Q_inst",
    "T_supply_over_100",
    "exergy",
    "is_match",
    "round_norm",
    "Q_proc_over_Q_avail_ref",
    "T_req_over_200",
    "budget_norm",
)
N_STATES = 50
PERTURBATION = 0.10
SAMPLING_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _sample_states(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=0.05, high=0.95, size=(n, OBS_DIM)).astype(np.float32)


def _predict_action(model: PPO, obs: np.ndarray) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=np.float64)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Reference PPO checkpoint missing: {MODEL_PATH}. "
            "Train it via `python train/train_ppo.py --scenario S5` first."
        )
    logger.info(
        "Step 4.7 starting: state-observation sensitivity on %s, %d states, +-%.0f%% perturbation",
        MODEL_PATH.name, N_STATES, 100 * PERTURBATION,
    )
    model = PPO.load(str(MODEL_PATH))

    states = _sample_states(N_STATES, SAMPLING_SEED)
    baseline_actions = np.array([_predict_action(model, s) for s in states])

    rows: list[dict] = []
    for dim_idx in range(OBS_DIM):
        for direction_label, factor in (("+10pct", 1.0 + PERTURBATION), ("-10pct", 1.0 - PERTURBATION)):
            distances = np.empty(N_STATES, dtype=np.float64)
            for i, base in enumerate(states):
                pert = base.copy()
                pert[dim_idx] = float(np.clip(pert[dim_idx] * factor, 0.0, 1.0))
                a_pert = _predict_action(model, pert)
                distances[i] = float(np.linalg.norm(a_pert - baseline_actions[i]))
            rows.append({
                "state_dim_idx": dim_idx,
                "state_dim_name": DIM_NAMES[dim_idx],
                "perturbation_direction": direction_label,
                "mean_action_distance": round(float(distances.mean()), 6),
                "std_action_distance": round(float(distances.std(ddof=1)) if N_STATES > 1 else 0.0, 6),
                "n_states_tested": N_STATES,
            })

    df_per = pd.DataFrame(rows)
    df_per.to_csv(OUT_PER_DIM, index=False)
    logger.info("Wrote %s (%d rows)", OUT_PER_DIM.name, len(df_per))

    agg = (
        df_per.groupby(["state_dim_idx", "state_dim_name"])["mean_action_distance"]
        .sum()
        .reset_index(name="total_sensitivity")
        .sort_values("total_sensitivity", ascending=False)
        .reset_index(drop=True)
    )
    agg["total_sensitivity"] = agg["total_sensitivity"].round(6)
    agg["sensitivity_rank"] = agg.index + 1
    agg.to_csv(OUT_RANKING, index=False)
    logger.info("Wrote %s (%d rows)", OUT_RANKING.name, len(agg))

    top3 = agg.head(3)
    logger.info("=" * 70)
    logger.info(
        "  State observation most sensitive dims (top-3 by total |action delta|): %s",
        ", ".join(f"{r['state_dim_name']} ({r['total_sensitivity']:.4f})" for _, r in top3.iterrows()),
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
