"""shielded_replay_buffer.py - SAC replay buffer that stores the executed action.

NOT USED BY DEFAULT (reference only). In this thesis the agent action is 4-dim and
sits upstream of the env's internal 4-to-5 augmentation and shield, so the default
SB3 buffer already stores a consistent (state, action) pair. This subclass is kept
only for the alternative design where the agent action IS the shield input (same
space), in which case storing the executed action removes the projection pre-image
ambiguity. The train script does not import it.

The executed action is read from info["shielded_action"] in ENV action space and
rescaled to [-1, 1] (SB3 SAC stores the scaled action), matching policy.scale_action.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

LOG = logging.getLogger(__name__)


class ShieldedReplayBuffer(ReplayBuffer):
    """ReplayBuffer that overwrites the stored action with the shielded one."""

    def __init__(self, *args: Any, store_shielded: bool = True,
                 scale_actions: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.store_shielded = bool(store_shielded)
        self.scale_actions = bool(scale_actions)
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        self._a_low = low
        self._a_span = np.where((high - low) > 0, high - low, 1.0).astype(np.float32)
        self._subst_count = 0

    def _scale_to_unit(self, a: np.ndarray) -> np.ndarray:
        """Env action space -> [-1, 1], matching SB3 policy.scale_action."""
        if not self.scale_actions:
            return a
        return (2.0 * (a - self._a_low) / self._a_span - 1.0).astype(np.float32)

    def add(self, obs, next_obs, action, reward, done, infos) -> None:  # type: ignore[override]
        if self.store_shielded:
            action = np.array(action, dtype=np.float32, copy=True)
            if action.ndim == 1:
                action = action[None, :]
            for i, info in enumerate(infos):
                shielded = info.get("shielded_action") if isinstance(info, dict) else None
                if shielded is not None:
                    action[i] = self._scale_to_unit(np.asarray(shielded, dtype=np.float32))
                    self._subst_count += 1
        super().add(obs, next_obs, action, reward, done, infos)

    @property
    def substitution_count(self) -> int:
        """How many transitions had their action replaced by the shielded one."""
        return self._subst_count
