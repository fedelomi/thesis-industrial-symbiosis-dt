"""Smoke tests for the ISNegotiationEnv on scenario S5."""

from __future__ import annotations

import numpy as np
import pytest

from config.scenarios import SCENARIOS
from env.is_negotiation_env import ISNegotiationEnv
from env.shielding import ShieldingLayer


@pytest.fixture
def env_s5() -> ISNegotiationEnv:
    if "S5" not in SCENARIOS:
        pytest.skip("Phase 2 IS-Match CSV not available; cannot build S5.")
    dc, mfg = SCENARIOS["S5"]
    return ISNegotiationEnv(dc=dc, mfg=mfg, shielding=ShieldingLayer(), max_rounds=5)


def test_reset_returns_correct_shapes(env_s5) -> None:
    obs, info = env_s5.reset(seed=0)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert "scenario" in info


def test_action_space_is_4dim(env_s5) -> None:
    """Action space must be Box(4): [Q, T, price, duration]."""
    assert env_s5.action_space.shape == (4,)


def test_five_random_steps(env_s5) -> None:
    env_s5.reset(seed=42)
    rewards = []
    for _ in range(5):
        action = env_s5.action_space.sample().astype(np.float32)
        obs, reward, terminated, truncated, info = env_s5.step(action)
        rewards.append(reward)
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(info["round_n"], int)
        # With upgrade fixed at construction, Rule 4 should never block.
        assert not any("upgrade_tech" in v for v in info["violations"])
        if terminated or truncated:
            break
    assert len(rewards) >= 1


def test_outcome_after_step(env_s5) -> None:
    env_s5.reset(seed=7)
    env_s5.step(env_s5.action_space.sample().astype(np.float32))
    outcome = env_s5.outcome()
    assert outcome.n_rounds >= 1
    assert outcome.offer_final is not None
    # Efficiency: phi_dc + phi_mfg ~= welfare_total
    assert abs(outcome.shapley_dc + outcome.shapley_mfg - outcome.welfare_total) < 1e-6
