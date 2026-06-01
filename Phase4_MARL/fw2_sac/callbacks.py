"""callbacks.py — curriculum scheduler, convergence proxy and rollout metrics.

Three Stable-Baselines3 callbacks for the FW2 SAC run:

- ``CurriculumCallback``: anneals the curriculum probability ``p_target`` (how
  often the hard S_AIR_M scenario is drawn) from 0 to ``p_final`` over a linear
  ramp, and pushes it into every vec-env worker via ``env_method``. A warmup
  keeps the agent on the easy pool first so it has a competent policy before the
  hard scenario is introduced; the ramp stops below 1.0 so the easy scenarios
  keep being rehearsed (anti forgetting).

- ``ConvergenceCallback``: logs the thesis convergence proxy
  ``delta_w_rel = |w(t) - w(t-window)| / max(|w(t)|, 1)`` with ``w`` the mean of
  ``|theta|`` over the SAC actor MLP, so the FW2 training curve uses the same
  metric as the canonical PPO curves.

- ``NegotiationMetricsCallback``: aggregates per-episode deal rate and shield
  interventions from the rollout ``infos`` and logs them to TensorBoard.

Author: Fede — Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

LOG = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """Linearly ramp the curriculum probability p_target and broadcast it."""

    def __init__(
        self,
        total_timesteps: int,
        *,
        warmup_frac: float = 0.10,
        ramp_end_frac: float = 0.70,
        p_final: float = 0.60,
        update_every: int = 2048,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.warmup_frac = float(warmup_frac)
        self.ramp_end_frac = float(ramp_end_frac)
        self.p_final = float(p_final)
        self.update_every = int(update_every)
        self._last_update = -1

    def _p_at(self, frac: float) -> float:
        span = max(self.ramp_end_frac - self.warmup_frac, 1e-9)
        ramp = (frac - self.warmup_frac) / span
        return float(self.p_final * np.clip(ramp, 0.0, 1.0))

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update < self.update_every:
            return True
        self._last_update = self.num_timesteps
        frac = self.num_timesteps / max(1, self.total_timesteps)
        p = self._p_at(frac)
        # Broadcast to every worker; the env exposes set_curriculum.
        self.training_env.env_method("set_curriculum", p)
        self.logger.record("curriculum/p_target", p)
        return True


class ConvergenceCallback(BaseCallback):
    """Log delta_w_rel over the SAC actor weights (thesis convergence metric)."""

    def __init__(self, *, window: int = 50_000, check_freq: int = 2048,
                 threshold: float = 0.01, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = int(window)
        self.check_freq = int(check_freq)
        self.threshold = float(threshold)
        self._history: list[tuple[int, float]] = []
        self._last_check = -1

    def _actor_weight_summary(self) -> float:
        import torch

        actor = getattr(self.model.policy, "actor", self.model.policy)
        with torch.no_grad():
            vals = [p.abs().mean().item() for p in actor.parameters() if p.requires_grad]
        return float(np.mean(vals)) if vals else 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps
        w = self._actor_weight_summary()
        self._history.append((self.num_timesteps, w))
        # Find the weight summary one window back.
        w_prev = None
        for t, val in reversed(self._history):
            if self.num_timesteps - t >= self.window:
                w_prev = val
                break
        if w_prev is not None:
            delta = abs(w - w_prev) / max(abs(w), 1.0)
            self.logger.record("train/delta_w_rel", delta)
            self.logger.record("train/converged", float(delta < self.threshold))
        self.logger.record("train/actor_weight_mean_abs", w)
        return True


class NegotiationMetricsCallback(BaseCallback):
    """Aggregate deal rate and shield interventions from rollout infos."""

    def __init__(self, *, log_every: int = 2048, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_every = int(log_every)
        self._deals: list[float] = []
        self._interv: list[float] = []
        self._last_log = -1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", self.locals.get("done", []))
        dones = np.atleast_1d(dones)
        for i, info in enumerate(infos):
            if isinstance(info, dict) and "n_shield_interventions" in info:
                self._interv.append(float(info["n_shield_interventions"]))
            if i < len(dones) and dones[i]:
                # terminated (deal) vs truncated (no deal): SB3 flags truncation
                # via info["TimeLimit.truncated"] or a Gymnasium truncated key.
                truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
                self._deals.append(0.0 if truncated else 1.0)
        if self.num_timesteps - self._last_log >= self.log_every:
            self._last_log = self.num_timesteps
            if self._deals:
                self.logger.record("train/deal_rate", float(np.mean(self._deals)))
                self._deals.clear()
            if self._interv:
                self.logger.record("train/shield_interventions_per_step", float(np.mean(self._interv)))
                self._interv.clear()
        return True
