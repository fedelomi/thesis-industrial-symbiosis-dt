"""
Step 4.8 - delta_TC (IS-Match Score) Perturbation Robustness
==============================================================
Phase 4 robustness layer (additive, no retraining)

Tests how stable PoA / Shapley fairness / convergence rate of the
trained D policy are when the per-tier delta_TC reduction factors are
perturbed (without retraining). delta_TC enters the env through the
`is_match_score` field of `DCProfile` (Phase 2 calibrated value); a
scalar multiplier on that score is the operational handle for the
perturbation here.

Five perturbation modes:

    baseline                       no change to is_match_score
    plus_20pct                     is_match_score x 1.20
    minus_20pct                    is_match_score x 0.80
    stress_LowT_plus_50pct         is_match_score x 1.50 only when the
                                   manufacturing tier is LowT (t_req <= 60 C)
    fallback_uniform_05            is_match_score = 0.50 for all tiers

Caveats (documented in CSV header):

    Scoped to S5 representative scenario (Mid_LC x MidT, rank 4 in
    Phase 2 IS-Match ranking) because only ppo_S5.zip is saved and
    Phase 4 training is intentionally costly. delta_TC is a global
    reward parameter, so the perturbation impact is expected to be
    structurally similar across scenarios; cross-scenario validation
    is deferred to Future Work.

Inputs:
    models/ppo_S5.zip
    Phase3_GraphRAG/data/step_3_5_bis_delta_tc.json   (baseline factors)

Output:
    results/robustness/step_4_8_deltatc_perturbation.csv
    results/robustness/step_4_8_deltatc_robustness_summary.csv
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PHASE4_ROOT = Path(__file__).resolve().parent
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from stable_baselines3 import PPO

from agents.baseline_lp import compute_centralized_welfare
from config.scenarios import SCENARIOS, build_scenarios
from env.is_negotiation_env import ISNegotiationEnv
from env.shielding import ShieldingLayer

ROBUSTNESS_DIR = PHASE4_ROOT / "results" / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = PHASE4_ROOT / "models" / "ppo_S5.zip"

OUT_PER_ROW = ROBUSTNESS_DIR / "step_4_8_deltatc_perturbation.csv"
OUT_SUMMARY = ROBUSTNESS_DIR / "step_4_8_deltatc_robustness_summary.csv"

SCENARIO_REP = "S5"
N_EPISODES = 100
SEED = 0
ROBUSTNESS_THRESHOLD_PP = 0.05

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _perturb(label: str, baseline: float, t_req_c: float) -> float:
    if label == "baseline":
        return baseline
    if label == "plus_20pct":
        return baseline * 1.20
    if label == "minus_20pct":
        return baseline * 0.80
    if label == "stress_LowT_plus_50pct":
        return baseline * 1.50 if t_req_c <= 60.0 else baseline
    if label == "fallback_uniform_05":
        return 0.50
    raise ValueError(f"Unknown perturbation label: {label}")


def _make_env(scenario_id: str, is_match_override: float, seed: int) -> ISNegotiationEnv:
    scenarios = SCENARIOS or build_scenarios()
    dc, mfg = scenarios[scenario_id]
    dc_pert = replace(dc, is_match_score=float(np.clip(is_match_override, 0.0, 1.0)))
    return ISNegotiationEnv(
        dc=dc_pert, mfg=mfg, shielding=ShieldingLayer(),
        max_rounds=20, dt_grounded=True, seed=seed,
    )


def _eval(model: PPO, scenario_id: str, label: str) -> tuple[float, float, float]:
    scenarios = SCENARIOS or build_scenarios()
    dc, mfg = scenarios[scenario_id]
    perturbed = _perturb(label, dc.is_match_score, mfg.t_req_c)
    env = _make_env(scenario_id, perturbed, SEED + 9999)
    welfare_lp = compute_centralized_welfare({scenario_id: (dc, mfg)}).get(
        scenario_id, float("nan")
    )

    poas, fairs, convs = [], [], []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        out = env.outcome()
        convs.append(int(out.converged))
        if welfare_lp and not np.isnan(welfare_lp) and abs(welfare_lp) > 1e-3:
            poas.append((out.welfare_dc + out.welfare_mfg) / welfare_lp)
        fairs.append(out.shapley_fairness_ratio)
    return (
        float(np.mean(poas)) if poas else float("nan"),
        float(np.mean(fairs)),
        float(np.mean(convs)),
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Reference PPO checkpoint missing: {MODEL_PATH}. Train it via "
            "`python train/train_ppo.py --scenario S5` first."
        )
    logger.info(
        "Step 4.8 starting: delta_TC perturbation robustness scoped to %s (%d episodes per cell)",
        SCENARIO_REP, N_EPISODES,
    )
    model = PPO.load(str(MODEL_PATH))

    labels = (
        "baseline",
        "plus_20pct",
        "minus_20pct",
        "stress_LowT_plus_50pct",
        "fallback_uniform_05",
    )
    rows: list[dict] = []
    baseline_poa: float | None = None
    for label in labels:
        poa, fair, conv = _eval(model, SCENARIO_REP, label)
        if label == "baseline":
            baseline_poa = poa
        delta = poa - (baseline_poa if baseline_poa is not None else poa)
        rows.append({
            "perturbation_label": label,
            "scenario": SCENARIO_REP,
            "PoA_mean": round(poa, 4),
            "Shapley_fairness": round(fair, 4),
            "conv_rate": round(conv, 4),
            "delta_PoA_vs_baseline": round(delta, 4),
            "n_episodes": N_EPISODES,
        })
        logger.info(
            "  %s: PoA=%.4f (delta %+.4f), Shapley=%.4f, conv=%.4f",
            label, poa, delta, fair, conv,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PER_ROW, index=False)
    logger.info("Wrote %s (%d rows)", OUT_PER_ROW.name, len(df))

    perturbed = df[df["perturbation_label"] != "baseline"]
    max_idx = perturbed["delta_PoA_vs_baseline"].abs().idxmax()
    max_label = str(perturbed.loc[max_idx, "perturbation_label"])
    max_delta = float(perturbed.loc[max_idx, "delta_PoA_vs_baseline"])
    mean_abs_delta = float(perturbed["delta_PoA_vs_baseline"].abs().mean())
    robust_pm20 = bool(
        perturbed[perturbed["perturbation_label"].isin(["plus_20pct", "minus_20pct"])][
            "delta_PoA_vs_baseline"
        ].abs().max() < ROBUSTNESS_THRESHOLD_PP
    )

    summary_rows = [
        {"metric": "scenario", "value": SCENARIO_REP},
        {"metric": "max_delta_PoA_observed", "value": round(max_delta, 4)},
        {"metric": "max_delta_PoA_under_label", "value": max_label},
        {"metric": "mean_abs_delta_PoA_perturbations", "value": round(mean_abs_delta, 4)},
        {"metric": "poa_robust_to_pm20pct", "value": robust_pm20},
        {"metric": "robustness_threshold_pp", "value": ROBUSTNESS_THRESHOLD_PP},
        {"metric": "recommended_action_if_phase3_changes",
         "value": "evaluate-only, no retraining needed if observed delta < 0.05"},
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SUMMARY, index=False)
    logger.info("Wrote %s (%d rows)", OUT_SUMMARY.name, len(df_summary))

    logger.info("=" * 70)
    logger.info(
        "  delta_TC robustness on %s: max |delta PoA| = %.4f (under %s); "
        "mean |delta| = %.4f; robust to +-20%%: %s",
        SCENARIO_REP, abs(max_delta), max_label, mean_abs_delta, robust_pm20,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
