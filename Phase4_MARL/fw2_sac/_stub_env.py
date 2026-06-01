"""_stub_env.py - minimal stand-in for the real Phase 4 negotiation packages.

Mirrors the public interface and the action-shape architecture of the real env /
config packages, so the FW2 scaffold can be smoke tested offline.

Architecture mirrored: the AGENT action is 4-dim. The env augments it internally
to a 5-slot action and applies the shield on that 5-slot form (the real shield
reads action[4]). So the policy and the replay buffer live in the 4-dim agent
space, while the shield operates downstream inside step. A 4-slot action passed
straight to the shield raises IndexError, exactly the real failure that caught the
earlier wrong design.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBS_DIM = 8
AGENT_ACT_DIM = 4
SHIELD_SLOTS = 5
IDX_Q, IDX_PRICE, IDX_UPGRADE = 0, 2, 4
_UPGRADE_IDX = {"HP": 0, "CO2HTHP": 1, "DIRECT": 2}


@dataclass
class DCProfile:
    q_available_kw: float = 475.0


@dataclass
class MfgParams:
    q_process_kw: float = 650.0
    distance_km: float = 30.0
    upgrade_required: str = "HP"


@dataclass
class ShieldingDecision:
    clipped_action: np.ndarray
    violations: list = field(default_factory=list)
    blocked: bool = False


@dataclass
class Outcome:
    converged: bool
    n_rounds: int
    welfare_dc: float
    welfare_mfg: float
    shapley_fairness_ratio: float


class ShieldingLayer:
    """Toy shield operating on the 5-slot augmented action (reads index 4)."""

    def __init__(self, reward_params: Optional[object] = None) -> None:
        self.reward_params = reward_params

    def _amortised_capex_floor(self, upgrade_tag: str, q_kw: float) -> float:
        return 0.35

    def _evaluate(self, action: np.ndarray, dc: "DCProfile", mfg: "MfgParams") -> ShieldingDecision:
        out = np.asarray(action, dtype=np.float32).copy()
        _ = int(round(out[IDX_UPGRADE]))  # requires a 5-slot action, like the real shield
        violations = []
        floor = self._amortised_capex_floor(mfg.upgrade_required, float(out[IDX_Q]))
        if out[IDX_PRICE] < floor:
            out[IDX_PRICE] = floor
            violations.append("WARNING: Rule 3 (economic floor) clip.")
        out[IDX_Q] = float(np.clip(out[IDX_Q], 0.0, 1.0))
        return ShieldingDecision(clipped_action=out, violations=violations, blocked=False)


class ISNegotiationEnv(gym.Env):
    """Toy negotiation: 8-dim obs, 4-dim AGENT action, shield applied in step."""

    metadata = {"render_modes": []}

    def __init__(self, dc, mfg, shielding=None, max_rounds=20, dt_grounded=True,
                 seed=None, accept_band=0.12):
        super().__init__()
        self.dc = dc
        self.mfg = mfg
        self.shielding = shielding
        self.max_rounds = int(max_rounds)
        self.dt_grounded = bool(dt_grounded)
        self.accept_band = float(accept_band)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(AGENT_ACT_DIM,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._round = 0
        self._converged = False
        self._w_dc = 0.0
        self._w_mfg = 0.0

    def _obs(self):
        base = self._rng.uniform(0.05, 0.95, size=OBS_DIM).astype(np.float32)
        base[3] = self._round / self.max_rounds
        return base

    def _augment(self, agent_action):
        a = np.asarray(agent_action, dtype=np.float32).ravel()
        upgrade = float(_UPGRADE_IDX.get(self.mfg.upgrade_required, 0))
        return np.array([a[0], a[1], a[2], a[3], upgrade], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._round = 0
        self._converged = False
        self._w_dc = 0.0
        self._w_mfg = 0.0
        return self._obs(), {}

    def step(self, action):
        self._round += 1
        full = self._augment(action)
        violations = []
        if self.shielding is not None:
            decision = self.shielding._evaluate(full.copy(), self.dc, self.mfg)
            full = np.asarray(decision.clipped_action, dtype=np.float32)
            violations = decision.violations
        price = float(full[IDX_PRICE])
        q = float(np.clip(full[IDX_Q], 0.0, 1.0))
        target = 0.5
        band = self.accept_band * (1.0 - 0.4 * min(self.mfg.distance_km / 200.0, 1.0))
        deal = abs(price - target) < band
        terminated = bool(deal)
        truncated = bool(self._round >= self.max_rounds)
        if terminated:
            self._converged = True
            margin = (price - 0.35) * q
            self._w_dc = float(max(margin, -0.2) * self.dc.q_available_kw)
            self._w_mfg = float((0.8 - price) * q * self.mfg.q_process_kw)
        reward = (1.0 if deal else 0.0) - 0.01 * self._round
        return self._obs(), float(reward), terminated, truncated, {"shield_violations": violations}

    def outcome(self):
        total = abs(self._w_dc) + abs(self._w_mfg) + 1e-9
        fairness = 1.0 - abs(self._w_dc - self._w_mfg) / total if self._converged else 0.0
        return Outcome(self._converged, self._round, self._w_dc, self._w_mfg, float(max(0.0, fairness)))


def build_scenarios():
    scales = {"Edge": 475.0, "Mid": 3104.0, "Hyper": 24500.0}
    tiers = {"LowT": (650.0, "HP", 30.0), "MidT": (1200.0, "HP", 45.0),
             "HighT": (2800.0, "CO2HTHP", 55.0)}
    out = {}
    i = 1
    for q_dc in scales.values():
        for q_p, up, dist in tiers.values():
            out[f"S{i}"] = (DCProfile(q_dc), MfgParams(q_p, dist, up))
            i += 1
    return out


def build_airside_scenario():
    return DCProfile(q_available_kw=900.0), MfgParams(q_process_kw=1500.0, distance_km=160.0, upgrade_required="CO2HTHP")
