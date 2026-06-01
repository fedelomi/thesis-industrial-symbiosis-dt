"""Reward function weights, thresholds, and economic parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardParams:
    """Hyperparameters for the negotiation reward function.

    convergence_delta: relative round-over-round utility change threshold,
    ``|w_dc(t) - w_dc(t-1)| / max(|w_dc|, 1) < delta``. Default 0.02 = 2 %.
    The relative formulation is invariant to welfare scale (10^4 to 10^7).
    """

    operating_hours_per_year: float = 8000.0
    amortisation_years: int = 10
    discount_rate: float = 0.05
    transport_cost_eur_per_mwh_km: float = 0.04
    gas_price_eur_per_mwh: float = 45.0
    emission_factor_gas_t_per_mwh: float = 0.20
    lambda_is_match_uplift: float = 1.0
    timeout_penalty: float = -1.0
    convergence_delta: float = 0.02


DEFAULT_REWARD_PARAMS = RewardParams()

__all__ = ["RewardParams", "DEFAULT_REWARD_PARAMS"]
