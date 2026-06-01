"""curriculum_env.py - scenario-curriculum wrapper around the negotiation env.

Design correction (2026-06-01). The real ISNegotiationEnv already applies the
shield INSIDE step and the agent action is upstream of the env's internal 4-to-5
slot augmentation (the policy emits a 4-dim action, the env augments it and calls
the shield on the 5-slot form). Because the whole chain
agent_action -> augment -> shield -> dynamics is a deterministic function of
(state, agent_action), the transition stored by an off-policy algorithm is already
consistent: there is no phantom-Q problem at the agent-action level and no need to
substitute the executed action in the replay buffer.

So this wrapper does ONE thing the canonical env does not: it holds several
scenarios and samples one per episode under a curriculum probability p_target
(set by CurriculumCallback during training). The shield stays inside each inner
env. Evaluation uses a separate env pinned to the target scenario, so the reported
convergence rate is always measured on S_AIR_M.

The inner env is built by the injected env_factory (which configures the shield
itself), so this module is testable with _stub_env when the real env / config
packages are not on the path.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

LOG = logging.getLogger(__name__)

# EnvFactory(dc, mfg, seed) returns a ready inner env with its shield already
# configured inside (config D), or without it (A0). The wrapper stays agnostic.
EnvFactory = Callable[[object, object, int], gym.Env]


class CurriculumNegotiationEnv(gym.Env):
    """Single-learner negotiation env that samples scenarios under a curriculum."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: dict,
        target_id: str,
        easy_ids: Sequence[str],
        env_factory: EnvFactory,
        *,
        max_rounds: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if target_id not in scenarios:
            raise KeyError(f"target_id {target_id!r} not in scenarios")
        self.target_id = target_id
        self.easy_ids = [s for s in dict.fromkeys(easy_ids) if s in scenarios]
        if not self.easy_ids:
            raise ValueError("easy_ids is empty after filtering against scenarios")
        self._pool = list(dict.fromkeys([*self.easy_ids, target_id]))
        self._sampler_rng = np.random.default_rng(seed)
        self._p_target = 0.0

        # Pre-build one inner env per scenario (each owns its shield internally).
        self._envs: dict = {}
        for i, sid in enumerate(self._pool):
            dc, mfg = scenarios[sid]
            self._envs[sid] = env_factory(dc, mfg, seed + i)

        any_env = self._envs[self._pool[0]]
        self.observation_space = any_env.observation_space
        self.action_space = any_env.action_space
        self._active_id = self.target_id
        self._active = self._envs[self._active_id]

    # -- curriculum control (called by CurriculumCallback) ------------------- #
    def set_curriculum(self, p_target: float) -> None:
        """Set the probability of drawing the hard target scenario per episode."""
        self._p_target = float(np.clip(p_target, 0.0, 1.0))

    def get_curriculum(self) -> float:
        return self._p_target

    @property
    def active_scenario(self) -> str:
        return self._active_id

    # -- gym API ------------------------------------------------------------- #
    def _choose_scenario(self) -> str:
        if self._sampler_rng.random() < self._p_target:
            return self.target_id
        return str(self._sampler_rng.choice(self.easy_ids))

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._sampler_rng = np.random.default_rng(seed)
        self._active_id = self._choose_scenario()
        self._active = self._envs[self._active_id]
        obs, info = self._active.reset(seed=seed)
        info = dict(info)
        info["scenario_id"] = self._active_id
        return obs, info

    def step(self, action: np.ndarray):
        # Delegate straight to the active inner env. The inner env augments the
        # agent action and applies its shield internally (do NOT shield here).
        obs, reward, terminated, truncated, info = self._active.step(action)
        info = dict(info)
        info["scenario_id"] = self._active_id
        return obs, reward, terminated, truncated, info

    def outcome(self):
        """Delegate to the active inner env (valid right after an episode ends)."""
        return self._active.outcome()

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
