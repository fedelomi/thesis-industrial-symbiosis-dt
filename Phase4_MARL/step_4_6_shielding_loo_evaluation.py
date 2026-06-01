"""
Step 4.6 - Shielding Leave-One-Out Evaluation
==============================================
Phase 4 robustness layer (additive)

Quantifies the relative importance of each of the four shielding rules
in `env/shielding.py` by toggling them off ONE AT A TIME during
evaluation (the policy keeps the weights it learned with all four
rules active). Per (rule_disabled, scenario, seed) we collect PoA,
Shapley fairness ratio and convergence rate over 100 episodes and
report the delta vs the full-shielding baseline.

The four rules (see `env/shielding.py` docstring):

    R1 - Q_negotiated <= Q_available (physical, BLOCKED)
    R2 - T_supply_offered >= T_req - 5 C (thermal feasibility, BLOCKED)
    R3 - price_eur_mwh >= amortised CAPEX floor (economic, WARNING)
    R4 - upgrade_tech == mfg.upgrade_required (normative, BLOCKED)

Caveats (documented in CSV header):

    Indicative criticality ranking. Checkpoints trained 50k timesteps
    (lighter than the canonical 100k pilot for S5); intended for
    relative criticality across rules, not direct comparison against
    canonical PoA values. Full 100k re-train deferred to Future Work.

Policy / shielding mismatch evaluation: the policy was trained with
all four rules active, so evaluating with one rule off measures how
much each rule corrects the policy's residual unsafe behaviour, NOT
the counterfactual of a policy trained without that rule. This is a
legitimate diagnostic of which rules are LOAD-BEARING in the deployed
system; full counterfactual ablation would require retraining each
config from scratch.

Inputs:
    models/D_<scenario>_seed<i>.zip       checkpoints trained by this
                                          script if missing
    env/shielding.py                      ShieldingLayer with 4 rules
    env/is_negotiation_env.py             ISNegotiationEnv

Outputs:
    results/robustness/step_4_6_shielding_loo.csv          per-row
    results/robustness/step_4_6_shielding_criticality_ranking.csv
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PHASE4_ROOT = Path(__file__).resolve().parent
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config.scenarios import SCENARIOS, build_airside_scenario, build_scenarios
from env.is_negotiation_env import ISNegotiationEnv
from env.shielding import ShieldingDecision, ShieldingLayer, UPGRADE_INDEX_INVERSE
from env.models import DCProfile, MfgParams

ROBUSTNESS_DIR = PHASE4_ROOT / "results" / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PHASE4_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER_ROW = ROBUSTNESS_DIR / "step_4_6_shielding_loo.csv"
OUT_RANKING = ROBUSTNESS_DIR / "step_4_6_shielding_criticality_ranking.csv"

SCENARIOS_LOO = ("S1", "S4", "S7", "S_AIR_M")
SEEDS = (0, 1, 2)
N_EVAL_EPISODES = 100
TRAIN_TIMESTEPS = 50_000
RULES = ("R1_phys", "R2_thermal", "R3_price", "R4_upgrade")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_scenarios() -> dict[str, tuple[DCProfile, MfgParams]]:
    out = dict(SCENARIOS) if SCENARIOS else build_scenarios()
    if "S_AIR_M" not in out:
        try:
            out["S_AIR_M"] = build_airside_scenario()
        except Exception as exc:
            logger.warning("S_AIR_M unavailable: %s", exc)
    return out


class _LooShielding(ShieldingLayer):
    """ShieldingLayer with one of the four rules disabled (pass-through)."""

    def __init__(self, disabled: str | None) -> None:
        super().__init__()
        self._disabled = disabled

    def _evaluate(self, action: np.ndarray, dc: DCProfile, mfg: MfgParams) -> ShieldingDecision:
        if self._disabled is None:
            return super()._evaluate(action, dc, mfg)
        decision = super()._evaluate(action.copy(), dc, mfg)
        # Replay rules selectively. We re-evaluate from scratch with the
        # named rule skipped, by mutating the action array similarly to
        # the parent but omitting the relevant block.
        q_kw, t_offered, price, duration, upgrade_idx = (
            float(action[0]), float(action[1]), float(action[2]),
            float(action[3]), int(round(action[4])),
        )
        out = action.copy()
        violations: list[str] = []
        blocked = False
        if self._disabled != "R1_phys":
            if q_kw > dc.q_available_kw:
                out[0] = dc.q_available_kw; blocked = True
                violations.append("BLOCKED: Q>Q_available")
            elif q_kw < 0.0:
                out[0] = 0.0
        if self._disabled != "R2_thermal":
            t_min = mfg.t_req_c - 5.0
            if t_offered < t_min:
                out[1] = t_min; blocked = True
                violations.append("BLOCKED: T below pinch")
        if self._disabled != "R3_price":
            floor = self._amortised_capex_floor(
                upgrade_tag=mfg.upgrade_required, q_kw=float(out[0]),
            )
            if price < floor:
                out[2] = floor
                violations.append("WARNING: price below floor (clipped)")
        if self._disabled != "R4_upgrade":
            from env.shielding import UPGRADE_INDEX
            upgrade_tag = UPGRADE_INDEX.get(upgrade_idx)
            if upgrade_tag != mfg.upgrade_required:
                out[4] = UPGRADE_INDEX_INVERSE[mfg.upgrade_required]
                blocked = True
                violations.append("BLOCKED: upgrade mismatch")
        if duration < 1.0:
            out[3] = 1.0
        elif duration > 25.0:
            out[3] = 25.0
        return ShieldingDecision(clipped_action=out, violations=violations, blocked=blocked)


def _make_env(
    scenario_id: str, scenarios: dict, shielding: ShieldingLayer | None, seed: int
) -> ISNegotiationEnv:
    dc, mfg = scenarios[scenario_id]
    return ISNegotiationEnv(
        dc=dc, mfg=mfg, shielding=shielding,
        max_rounds=20, dt_grounded=True, seed=seed,
    )


def _train_d_checkpoint(
    scenario_id: str, seed: int, scenarios: dict, model_path: Path
) -> PPO:
    logger.info("  training D checkpoint for %s seed=%d (%d timesteps)...",
                scenario_id, seed, TRAIN_TIMESTEPS)
    env_fns = [
        (lambda s=seed + i: Monitor(_make_env(scenario_id, scenarios, ShieldingLayer(), s)))
        for i in range(2)
    ]
    vec_env = DummyVecEnv(env_fns)
    model = PPO(
        policy="MlpPolicy", env=vec_env,
        n_steps=512, batch_size=64, n_epochs=5, learning_rate=3e-4,
        seed=seed, verbose=0,
    )
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    model.save(str(model_path))
    vec_env.close()
    return model


def _get_or_train_checkpoint(scenario_id: str, seed: int, scenarios: dict) -> PPO:
    path = MODELS_DIR / f"D_{scenario_id}_seed{seed}.zip"
    if path.exists():
        return PPO.load(str(path))
    return _train_d_checkpoint(scenario_id, seed, scenarios, path)


def _eval_policy(
    model: PPO, scenario_id: str, scenarios: dict, disabled_rule: str | None, seed: int
) -> tuple[float, float, float]:
    shielding = _LooShielding(disabled_rule)
    env = _make_env(scenario_id, scenarios, shielding, seed + 9999)
    poas: list[float] = []
    fairs: list[float] = []
    convs: list[int] = []
    from agents.baseline_lp import compute_centralized_welfare
    centralised = compute_centralized_welfare({scenario_id: scenarios[scenario_id]})
    welfare_lp = centralised.get(scenario_id, float("nan"))
    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        out = env.outcome()
        convs.append(int(out.converged))
        if welfare_lp and not np.isnan(welfare_lp) and abs(welfare_lp) > 1e-3:
            poas.append((out.welfare_dc + out.welfare_mfg) / welfare_lp)
        fairs.append(out.shapley_fairness_ratio)
    return float(np.mean(poas)) if poas else float("nan"), float(np.mean(fairs)), float(np.mean(convs))


def main() -> None:
    logger.info(
        "Step 4.6 starting: shielding LOO over %d scenarios x %d seeds x %d rules + baseline",
        len(SCENARIOS_LOO), len(SEEDS), len(RULES),
    )
    scenarios = _load_scenarios()
    missing = [s for s in SCENARIOS_LOO if s not in scenarios]
    if missing:
        logger.error("Scenarios missing: %s; aborting", missing)
        return

    rows: list[dict] = []
    t0 = time.time()
    for scenario_id in SCENARIOS_LOO:
        for seed in SEEDS:
            model = _get_or_train_checkpoint(scenario_id, seed, scenarios)
            poa_full, fair_full, conv_full = _eval_policy(model, scenario_id, scenarios, None, seed)
            rows.append({
                "rule_disabled": "NONE_baseline",
                "scenario": scenario_id, "seed": seed,
                "PoA": round(poa_full, 4),
                "Shapley_fairness": round(fair_full, 4),
                "conv_rate": round(conv_full, 4),
                "n_episodes": N_EVAL_EPISODES,
                "delta_PoA_vs_full": 0.0,
                "delta_Shapley_vs_full": 0.0,
                "delta_conv_vs_full": 0.0,
            })
            for rule in RULES:
                poa, fair, conv = _eval_policy(model, scenario_id, scenarios, rule, seed)
                rows.append({
                    "rule_disabled": rule,
                    "scenario": scenario_id, "seed": seed,
                    "PoA": round(poa, 4),
                    "Shapley_fairness": round(fair, 4),
                    "conv_rate": round(conv, 4),
                    "n_episodes": N_EVAL_EPISODES,
                    "delta_PoA_vs_full": round(poa - poa_full, 4),
                    "delta_Shapley_vs_full": round(fair - fair_full, 4),
                    "delta_conv_vs_full": round(conv - conv_full, 4),
                })
            logger.info(
                "  done %s seed=%d (%.1fs elapsed)", scenario_id, seed, time.time() - t0
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PER_ROW, index=False)
    logger.info("Wrote %s (%d rows)", OUT_PER_ROW.name, len(df))

    perturbed = df[df["rule_disabled"] != "NONE_baseline"]
    agg = (
        perturbed.groupby("rule_disabled")[
            ["delta_PoA_vs_full", "delta_Shapley_vs_full", "delta_conv_vs_full"]
        ].mean().reset_index()
    )
    agg["criticality_score"] = (
        agg["delta_PoA_vs_full"].abs()
        + agg["delta_Shapley_vs_full"].abs()
        + agg["delta_conv_vs_full"].abs()
    )
    agg = agg.sort_values("criticality_score", ascending=False).reset_index(drop=True)
    agg["criticality_rank"] = agg.index + 1
    for c in ("delta_PoA_vs_full", "delta_Shapley_vs_full", "delta_conv_vs_full", "criticality_score"):
        agg[c] = agg[c].round(4)
    agg.to_csv(OUT_RANKING, index=False)
    logger.info("Wrote %s (%d rows)", OUT_RANKING.name, len(agg))

    logger.info("=" * 70)
    logger.info(
        "  Shielding rule criticality (most -> least): %s",
        " > ".join(agg["rule_disabled"].tolist()),
    )
    for _, r in agg.iterrows():
        logger.info(
            "  %s: dPoA=%+.3f  dShapley=%+.3f  dConv=%+.3f  (rank %d)",
            r["rule_disabled"],
            r["delta_PoA_vs_full"], r["delta_Shapley_vs_full"], r["delta_conv_vs_full"],
            int(r["criticality_rank"]),
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
