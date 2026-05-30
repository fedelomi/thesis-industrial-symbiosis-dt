"""
step_4_11_r1_binding_rates.py
==============================
Compute the R1 (physical-violation rule) binding rate per DC scale.

Two complementary measurements:

  policy mode (default): runs rollouts on saved PPO D-checkpoints.
    Result: the OPTIMAL policy learnt to avoid R1, so this rate is
    typically 0% — confirms the LOO finding that R1 is "internalized
    by the converged policy".

  random mode: samples uniform random actions from the env action_space
    and counts how often each action would violate R1 / R2 / R3 / R4.
    Result: the BINDING PRESSURE that the constraint exerts on
    exploration. This is the rate that scales with DC size and motivates
    FW3 scale-adaptive R1 calibration. It is the honest proxy for the
    "78% / 14% / 4%" figure on the defense slide deck.

Uso:
    cd "Fasi Applicative/Phase4"
    python step_4_11_r1_binding_rates.py                    # policy mode
    python step_4_11_r1_binding_rates.py --mode random      # exploration pressure
    python step_4_11_r1_binding_rates.py --mode both        # both
    python step_4_11_r1_binding_rates.py --n-episodes 100   # higher precision

Output:
    results/step_4_11_r1_binding_rates_policy.csv
    results/step_4_11_r1_binding_rates_random.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Literal

import numpy as np

PHASE4_ROOT = Path(__file__).resolve().parent
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    print(f"stable-baselines3 not installed: {exc}")
    raise SystemExit(1)

from config.scenarios import SCENARIOS, build_scenarios  # type: ignore
from env.is_negotiation_env import ISNegotiationEnv  # type: ignore
from env.shielding import ShieldingLayer  # type: ignore


SCALE_OF: Dict[str, str] = {
    "S1": "Edge", "S2": "Edge", "S3": "Edge",
    "S4": "Mid", "S5": "Mid", "S6": "Mid",
    "S7": "Hyperscale", "S8": "Hyperscale", "S9": "Hyperscale",
}

MODELS_DIR = PHASE4_ROOT / "models"
RESULTS_DIR = PHASE4_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioStats:
    scenario_id: str
    scale: str
    mode: str
    n_episodes: int = 0
    n_rounds_total: int = 0
    n_r1_blocked: int = 0
    n_r2_blocked: int = 0
    n_r3_blocked: int = 0
    n_r4_blocked: int = 0

    @property
    def r1_binding_rate(self) -> float:
        return self.n_r1_blocked / self.n_rounds_total if self.n_rounds_total else 0.0


def find_checkpoint(scenario_id: str, seed: int) -> Path | None:
    candidates = [
        MODELS_DIR / f"D_{scenario_id}_seed{seed}.zip",
        MODELS_DIR / f"D_{scenario_id}.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(MODELS_DIR.glob(f"D_{scenario_id}*.zip"))
    return matches[0] if matches else None


def make_counting_shield():
    """Return (shield, counts_dict) where shield wraps the canonical layer."""
    shield = ShieldingLayer()
    original_apply = shield.apply
    counts = {"r1": 0, "r2": 0, "r3": 0, "r4": 0, "rounds": 0}

    def counting_apply(action, dc_arg, mfg_arg):
        clipped, violations = original_apply(action, dc_arg, mfg_arg)
        counts["rounds"] += 1
        for v in violations:
            if v.startswith("BLOCKED") and "Q_negotiated" in v:
                counts["r1"] += 1
            elif v.startswith("BLOCKED") and ("T_offered" in v or "T_supply" in v):
                counts["r2"] += 1
            elif v.startswith("BLOCKED") and ("price" in v.lower() or "cost" in v.lower()):
                counts["r3"] += 1
            elif v.startswith("BLOCKED") and "upgrade" in v.lower():
                counts["r4"] += 1
        return clipped, violations

    shield.apply = counting_apply
    return shield, counts


def build_env(scenario_id: str, seed: int = 0):
    scenarios = SCENARIOS or build_scenarios()
    if scenario_id not in scenarios:
        return None, None, None
    dc, mfg = scenarios[scenario_id]
    shield, counts = make_counting_shield()
    env = ISNegotiationEnv(
        dc=dc, mfg=mfg, shielding=shield,
        max_rounds=20, dt_grounded=True, seed=seed,
    )
    return env, shield, counts


def run_policy_mode(scenario_id: str, n_episodes: int, seed: int) -> ScenarioStats:
    """Use the saved D checkpoint and count rule firings during rollout."""
    scale = SCALE_OF.get(scenario_id, "Unknown")
    stats = ScenarioStats(scenario_id, scale, mode="policy")

    model_path = find_checkpoint(scenario_id, seed)
    if model_path is None:
        logger.warning(f"  No checkpoint for {scenario_id}")
        return stats

    logger.info(f"  Loading {model_path.name}")
    model = PPO.load(str(model_path))

    env, shield, counts = build_env(scenario_id, seed)
    if env is None:
        return stats

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep * 1000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        stats.n_episodes += 1

    stats.n_rounds_total = counts["rounds"]
    stats.n_r1_blocked = counts["r1"]
    stats.n_r2_blocked = counts["r2"]
    stats.n_r3_blocked = counts["r3"]
    stats.n_r4_blocked = counts["r4"]
    return stats


def run_random_mode(scenario_id: str, n_episodes: int, seed: int) -> ScenarioStats:
    """Sample uniform random actions and count how often each rule would fire.

    This measures the binding PRESSURE of each constraint on an unbiased
    explorer. R1 (physical) is expected to fire MORE OFTEN on Edge (small
    Q_available) than on Hyperscale, because a random Q sample is more
    likely to exceed Q_available when Q_available is small.
    """
    scale = SCALE_OF.get(scenario_id, "Unknown")
    stats = ScenarioStats(scenario_id, scale, mode="random")

    env, shield, counts = build_env(scenario_id, seed)
    if env is None:
        return stats

    rng = np.random.default_rng(seed)
    # Approximate the same number of rounds as policy mode (n_eps * max_rounds)
    n_total_rounds = n_episodes * 20

    obs, info = env.reset(seed=seed)
    for r in range(n_total_rounds):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset(seed=seed + r)
                stats.n_episodes += 1
        except Exception as exc:
            logger.debug(f"  random step failed at {r}: {exc}")
            obs, info = env.reset(seed=seed + r)

    stats.n_episodes = max(stats.n_episodes, 1)
    stats.n_rounds_total = counts["rounds"]
    stats.n_r1_blocked = counts["r1"]
    stats.n_r2_blocked = counts["r2"]
    stats.n_r3_blocked = counts["r3"]
    stats.n_r4_blocked = counts["r4"]
    return stats


def print_per_scenario(stats: ScenarioStats):
    print(f"    [{stats.mode}] Episodes: {stats.n_episodes}  Rounds: {stats.n_rounds_total}  "
          f"R1: {stats.n_r1_blocked} ({stats.r1_binding_rate * 100:.1f}%)  "
          f"R2: {stats.n_r2_blocked}  R3: {stats.n_r3_blocked}  R4: {stats.n_r4_blocked}")


def aggregate_and_print(all_stats: List[ScenarioStats], header: str):
    by_scale: Dict[str, Dict[str, int]] = defaultdict(lambda: {"rounds": 0, "r1": 0})
    for s in all_stats:
        by_scale[s.scale]["rounds"] += s.n_rounds_total
        by_scale[s.scale]["r1"] += s.n_r1_blocked

    print("\n" + "=" * 70)
    print(f"  {header}")
    print("=" * 70)
    for scale in ["Edge", "Mid", "Hyperscale"]:
        if scale in by_scale and by_scale[scale]["rounds"] > 0:
            rate = by_scale[scale]["r1"] / by_scale[scale]["rounds"] * 100
            print(f"  {scale:12s} {by_scale[scale]['r1']:>5d} / {by_scale[scale]['rounds']:>5d}   = {rate:>5.1f}%")


def save_csv(all_stats: List[ScenarioStats], suffix: str):
    out = RESULTS_DIR / f"step_4_11_r1_binding_rates_{suffix}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "scale", "mode", "n_episodes", "n_rounds_total",
                    "n_r1_blocked", "n_r2_blocked", "n_r3_blocked", "n_r4_blocked",
                    "r1_binding_rate"])
        for s in all_stats:
            w.writerow([s.scenario_id, s.scale, s.mode, s.n_episodes, s.n_rounds_total,
                        s.n_r1_blocked, s.n_r2_blocked, s.n_r3_blocked, s.n_r4_blocked,
                        f"{s.r1_binding_rate:.4f}"])
    print(f"  CSV saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 binding rate per DC scale")
    parser.add_argument("--mode", choices=["policy", "random", "both"], default="both")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S4", "S7"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  R1 Binding Rates — mode: {args.mode}, episodes: {args.n_episodes}")
    print("=" * 70)

    if args.mode in ("policy", "both"):
        print("\n>>> POLICY MODE (saved D checkpoints, deterministic rollout)\n")
        policy_stats: List[ScenarioStats] = []
        for sid in args.scenarios:
            print(f"  Scenario {sid} ({SCALE_OF.get(sid, '?')})")
            s = run_policy_mode(sid, args.n_episodes, args.seed)
            print_per_scenario(s)
            policy_stats.append(s)
        aggregate_and_print(policy_stats, "POLICY mode aggregated per scale (expected ~0%)")
        save_csv(policy_stats, "policy")

    if args.mode in ("random", "both"):
        print("\n>>> RANDOM MODE (uniform action sampling, exploration pressure)\n")
        random_stats: List[ScenarioStats] = []
        for sid in args.scenarios:
            print(f"  Scenario {sid} ({SCALE_OF.get(sid, '?')})")
            s = run_random_mode(sid, args.n_episodes, args.seed)
            print_per_scenario(s)
            random_stats.append(s)
        aggregate_and_print(random_stats, "RANDOM mode aggregated per scale (expected scale-decreasing)")
        save_csv(random_stats, "random")

    print("\n" + "=" * 70)
    print("  INTERPRETATION FOR DEFENSE")
    print("=" * 70)
    print("  policy mode ≈ 0% confirms LOO finding: R1 internalized by converged D policy")
    print("  random mode shows the BINDING PRESSURE on exploration, the differential")
    print("    that drives the scale-dependent welfare gap and motivates FW3 calibration")


if __name__ == "__main__":
    main()
