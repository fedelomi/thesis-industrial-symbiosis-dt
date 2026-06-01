"""Gymnasium environment for the bilateral DC-MFG IS negotiation MDP.

Based on the formal MDP definition in ``3-layer-framework-and-methodology.md``
Phase 4 (Passi 4.1-4.6). N=2 strict (Phase 4.4); topologies N>2 are explicit
future work.

State (observation): concatenated normalised features of both agents plus
round counter and budget remaining. Action: bounded continuous controls plus
a discrete upgrade-technology choice (one-hot expanded into the last 3 slots).

Reward (DC agent): economic term (revenue - amortised CAPEX - transport
residual TC) plus optional IS-Match uplift bonus (``lambda_is_match_uplift``).
A timeout penalty applies if the episode runs out of rounds.

The MFG counter-offer is a rule-based heuristic that modifies *price + Q*
(per design decision 2026-05-14):

    * If welfare_mfg(offer) >= 0 -> ACCEPT.
    * Else -> propose lower price (-5%) and/or higher Q (+5%) bounded by the
      shielding floor and Q_available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.reward_params import DEFAULT_REWARD_PARAMS, RewardParams
from config.scenarios import CAPEX_EUR_PER_KW, COP_BY_UPGRADE
from env.models import DCProfile, MfgParams, NegotiationOutcome, Offer
from env.shielding import (
    UPGRADE_INDEX_INVERSE,
    ShieldingLayer,
    T_PINCH_TOLERANCE,
)

logger = logging.getLogger("phase4.env")

# Phase 4 target: push IS-Match Score from "marginal" (0.36-0.53)
# to "high-priority" (> 0.60). The uplift bonus is normalised against
# this 0.10 gap so a full uplift contributes ``lambda`` units of reward.
IS_MATCH_TARGET_GAP = 0.10
HIGH_PRIORITY_THRESHOLD = 0.60


@dataclass(slots=True)
class _EnvState:
    """Internal mutable state of the env (kept off the public dataclass)."""

    round_n: int
    last_offer: Optional[Offer]
    last_dc_utility: float
    last_mfg_utility: float
    accepted: bool
    timed_out: bool
    is_match_pre: float
    is_match_post: float
    last_violations: Tuple[str, ...]
    q_available_this_step: float = 0.0


class ISNegotiationEnv(gym.Env):
    """Bilateral DC-Manufacturing IS negotiation environment.

    Parameters
    ----------
    dc:
        DC LC profile (Phase 1 LC output).
    mfg:
        Manufacturing surrogate parameters (Phase 2 dataset).
    shielding:
        Shielding layer instance. Pass ``None`` to bypass shielding entirely
        (used in ablation configuration A0).
    reward_params:
        Reward configuration.
    max_rounds:
        Maximum negotiation rounds before timeout (default 20).
    dt_grounded:
        If ``False``, use the **mean** Q_available across the year (static
        profile) instead of the time-varying value. Used in ablation A0 to
        isolate the DT-grounding contribution.
    seed:
        RNG seed for the env's local random number generator.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # Action layout: [Q_norm, T_norm, price_norm, duration_norm]
    # upgrade_tech is fixed at mfg.upgrade_required (normative hard constraint
    # by shielding Rule 4) and therefore not a learnable variable.
    ACTION_DIM = 4

    def __init__(
        self,
        dc: DCProfile,
        mfg: MfgParams,
        shielding: Optional[ShieldingLayer] = None,
        reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
        max_rounds: int = 20,
        dt_grounded: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dc = dc
        self.mfg = mfg
        self.shielding = shielding
        self.reward_params = reward_params
        self.max_rounds = int(max_rounds)
        self.dt_grounded = bool(dt_grounded)

        # Observation: [Q_avail/Q_inst, T_supply/100, exergy, is_match,
        #               round_norm, Q_proc/Q_avail_ref, T_req/200, budget_norm]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )

        # Continuous action: 4 normalised slots in [0, 1] for
        # [Q_norm, T_norm, price_norm, duration_norm]. The upgrade technology
        # is determined by mfg.upgrade_required (normative constraint).
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Precompute the canonical upgrade index (passed to the shielding layer
        # so Rule 4 is satisfied by construction).
        self._upgrade_idx = UPGRADE_INDEX_INVERSE[mfg.upgrade_required]

        # Price normalisation reference: 5 x amortised CAPEX floor for CO2_HTHP
        # ensures the agent can reach realistic price points (50-300 EUR/MWh).
        self._price_max_eur_mwh = (
            5.0
            * CAPEX_EUR_PER_KW["CO2_HTHP"]
            * 1000.0
            / (
                self.reward_params.amortisation_years
                * self.reward_params.operating_hours_per_year
            )
        )
        self._duration_max_yr = 15

        self._rng = np.random.default_rng(seed)
        self._state: Optional[_EnvState] = None

    # ------------------------------------------------------------------ #
    # Gymnasium API                                                       #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = _EnvState(
            round_n=0,
            last_offer=None,
            last_dc_utility=0.0,
            last_mfg_utility=0.0,
            accepted=False,
            timed_out=False,
            is_match_pre=self.dc.is_match_score,
            is_match_post=self.dc.is_match_score,
            last_violations=tuple(),
            q_available_this_step=self.dc.q_available_kw,
        )
        return self._observe(), {"scenario": f"{self.dc.dc_id}-{self.mfg.plant_id}"}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        assert self.action_space.contains(action) or True  # SB3 may pass slight overflows
        self._state.round_n += 1
        # Cache the (possibly stochastic) Q_available for this step so observe()
        # and the denormaliser see the same number.
        self._state.q_available_this_step = self._sample_q_available()

        # 1. Denormalise action
        raw_action = self._denormalise(np.asarray(action, dtype=np.float64))

        # 2. Apply shielding (if active)
        if self.shielding is not None:
            clipped, violations = self.shielding.apply(raw_action, self.dc, self.mfg)
        else:
            clipped, violations = raw_action, []
        self._state.last_violations = tuple(violations)
        is_blocked = any(v.startswith("BLOCKED") for v in violations)

        # 3. Build offer
        offer = self._action_to_offer(clipped, is_valid=not is_blocked)

        # 4. Manufacturer rule-based response
        mfg_accepts, counter_offer, w_mfg = self._mfg_response(offer)
        offer_after = counter_offer if not mfg_accepts else offer

        # 5. Utilities and IS-Match update
        w_dc = self._dc_welfare(offer_after)
        is_match_post = self._update_is_match(offer_after)

        # 6. Convergence check (round-over-round utility delta, relative scale)
        # We use a RELATIVE delta (delta / max(|w_dc|, 1)) because w_dc spans
        # 10^4-10^7 EUR/yr across scenarios. An absolute threshold of 0.02
        # would never fire for million-scale welfares; relative 2% is robust.
        prev = self._state.last_dc_utility
        delta_util = abs(w_dc - prev) / max(abs(w_dc), abs(prev), 1.0)
        converged = (
            mfg_accepts
            and delta_util < self.reward_params.convergence_delta
            and self._state.round_n >= 2
        )

        # 7. Reward
        reward = self._compute_reward(
            w_dc=w_dc,
            is_match_pre=self._state.is_match_pre,
            is_match_post=is_match_post,
            converged=converged,
            timed_out=False,
            blocked=is_blocked,
        )

        # 8. Update state
        self._state.last_offer = offer_after
        self._state.last_dc_utility = w_dc
        self._state.last_mfg_utility = w_mfg
        self._state.is_match_post = is_match_post
        self._state.accepted = mfg_accepts and converged

        terminated = converged
        truncated = self._state.round_n >= self.max_rounds and not terminated
        if truncated:
            self._state.timed_out = True
            reward += self.reward_params.timeout_penalty

        info: Dict[str, Any] = {
            "round_n": self._state.round_n,
            "mfg_accepts": mfg_accepts,
            "converged": converged,
            "is_match_pre": self._state.is_match_pre,
            "is_match_post": is_match_post,
            "is_match_uplift": is_match_post - self._state.is_match_pre,
            "welfare_dc": w_dc,
            "welfare_mfg": w_mfg,
            "violations": list(violations),
            "blocked": is_blocked,
            "offer": offer_after,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if self._state is None or self._state.last_offer is None:
            print("[ISNegotiationEnv] no offer yet (call reset+step first)")
            return
        o = self._state.last_offer
        rows = [
            ("scenario", f"{self.dc.dc_id} x {self.mfg.plant_id}"),
            ("round", f"{self._state.round_n}/{self.max_rounds}"),
            ("Q_negotiated_kW", f"{o.q_negotiated_kw:.1f}"),
            ("T_supply_offered_C", f"{o.t_supply_offered_c:.1f}"),
            ("upgrade_tech", o.upgrade_tech),
            ("price_EUR_per_MWh", f"{o.price_eur_mwh:.2f}"),
            ("contract_yr", f"{o.contract_duration_yr}"),
            ("welfare_dc_EUR_yr", f"{self._state.last_dc_utility:,.0f}"),
            ("welfare_mfg_EUR_yr", f"{self._state.last_mfg_utility:,.0f}"),
            ("is_match_pre/post", f"{self._state.is_match_pre:.4f} -> {self._state.is_match_post:.4f}"),
            ("accepted", str(self._state.accepted)),
        ]
        width = max(len(k) for k, _ in rows) + 2
        print("+" + "-" * (width + 30) + "+")
        for k, v in rows:
            print(f"| {k.ljust(width)} | {v}")
        if self._state.last_violations:
            print("| violations:")
            for msg in self._state.last_violations:
                print(f"|   - {msg}")
        print("+" + "-" * (width + 30) + "+")

    # ------------------------------------------------------------------ #
    # Public helpers used by outcome serialisation                        #
    # ------------------------------------------------------------------ #
    def outcome(self) -> NegotiationOutcome:
        """Return a :class:`NegotiationOutcome` snapshot of the current state."""
        if self._state is None or self._state.last_offer is None:
            raise RuntimeError("No outcome available; run at least one step.")
        from agents.shapley import shapley_fairness_ratio, shapley_value_bilateral

        v_dc = max(self._state.last_dc_utility, 0.0)  # disagreement payoff
        v_mfg = max(self._state.last_mfg_utility, 0.0)
        v_total = self._state.last_dc_utility + self._state.last_mfg_utility
        phi_dc, phi_mfg = shapley_value_bilateral(v_dc, v_mfg, v_total)

        co2 = self._co2_saved(self._state.last_offer)
        return NegotiationOutcome(
            converged=self._state.accepted,
            n_rounds=self._state.round_n,
            offer_final=self._state.last_offer,
            welfare_dc=self._state.last_dc_utility,
            welfare_mfg=self._state.last_mfg_utility,
            welfare_total=v_total,
            shapley_dc=phi_dc,
            shapley_mfg=phi_mfg,
            shapley_fairness_ratio=shapley_fairness_ratio(phi_dc, phi_mfg),
            co2_saved_t_yr=co2,
            is_match_uplift=self._state.is_match_post - self._state.is_match_pre,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _observe(self) -> np.ndarray:
        assert self._state is not None
        q_avail = self._effective_q_available()
        budget_remaining = (
            (self.max_rounds - self._state.round_n) / max(1, self.max_rounds)
        )
        obs = np.array(
            [
                np.clip(q_avail / max(1.0, self.dc.q_available_kw), 0.0, 1.0),
                np.clip(self.dc.t_supply_c / 100.0, 0.0, 1.0),
                np.clip(self.dc.exergy_dt, 0.0, 1.0),
                np.clip(self.dc.is_match_score, 0.0, 1.0),
                np.clip(self._state.round_n / max(1, self.max_rounds), 0.0, 1.0),
                np.clip(
                    self.mfg.q_process_kw
                    / max(1.0, self.dc.q_available_kw),
                    0.0,
                    1.0,
                ),
                np.clip(self.mfg.t_req_c / 200.0, 0.0, 1.0),
                np.clip(budget_remaining, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _sample_q_available(self) -> float:
        """Draw the per-step Q_available value. Stochastic only for airside.

        Behaviour matrix
        ----------------
        * ``q_available_profile is None`` (LC scenarios): both branches return
          the rated mean Q (deliberate Phase 4 design: LC has
          ``t_availability = 1.0``, so the ablation is shielding-only on LC).
        * ``q_available_profile is not None`` (airside sensitivity, S_AIR_M):

          - ``dt_grounded=True``  -> random sample from the annual profile,
          - ``dt_grounded=False`` -> profile mean (= rated Q by construction).

          Sampling uses the env's local RNG so multi-seed evaluation is
          reproducible.
        """
        profile = getattr(self.dc, "q_available_profile", None)
        if profile is None or not self.dt_grounded:
            return self.dc.q_available_kw
        idx = int(self._rng.integers(0, len(profile)))
        return float(profile[idx])

    def _effective_q_available(self) -> float:
        """Return the cached Q_available for the current step."""
        if self._state is None:
            return self.dc.q_available_kw
        return self._state.q_available_this_step

    def _denormalise(self, action: np.ndarray) -> np.ndarray:
        """Convert [0,1]-normalised continuous action to raw quantities.

        The fixed ``upgrade_idx`` is appended at position 4 so the shielding
        layer continues to receive a 5-slot action (preserving its API).

        Q is capped at ``min(Q_available, Q_demand)`` to enforce D9 (Q_partial
        supply): the plant cannot absorb more heat than its demand, and over-
        supply must not inflate welfare (would break PoA <= 1 by definition).
        """
        q_norm, t_norm, p_norm, d_norm = action[:4]
        # D9: Q cap at min(Q_available, Q_demand)
        q_max = min(self._effective_q_available(), self.mfg.q_process_kw)
        q_kw = float(q_norm) * q_max
        # T range: [T_req - pinch, T_req + 30] -> rich enough for HP/CO2_HTHP
        t_min = self.mfg.t_req_c - T_PINCH_TOLERANCE
        t_max = self.mfg.t_req_c + 30.0
        t_c = t_min + float(t_norm) * (t_max - t_min)
        price = float(p_norm) * self._price_max_eur_mwh
        duration = 1.0 + float(d_norm) * (self._duration_max_yr - 1.0)
        return np.array(
            [q_kw, t_c, price, duration, float(self._upgrade_idx)],
            dtype=np.float64,
        )

    def _action_to_offer(self, action: np.ndarray, is_valid: bool) -> Offer:
        return Offer(
            q_negotiated_kw=float(action[0]),
            t_supply_offered_c=float(action[1]),
            upgrade_tech=self.mfg.upgrade_required,
            price_eur_mwh=float(action[2]),
            contract_duration_yr=int(round(float(action[3]))),
            is_valid=is_valid,
        )

    def _mfg_response(self, offer: Offer) -> Tuple[bool, Offer, float]:
        """Rule-based manufacturer: accept if welfare positive, else counter.

        Counter strategy modifies *price* and *Q_negotiated* only (per
        design decision 2026-05-14). Price reduction -5%, Q increase +5%
        bounded above by Q_available and below by the shielding floor.
        """
        w_mfg = self._mfg_welfare(offer)
        if w_mfg >= 0.0:
            return True, offer, w_mfg

        # Counter-offer
        q_avail = self._effective_q_available()
        # Q_partial mode: cap at min(Q_avail, Q_demand)
        q_target = min(offer.q_negotiated_kw * 1.05, q_avail, self.mfg.q_process_kw)
        new_price = max(offer.price_eur_mwh * 0.95, 0.0)

        # Recompute floor against the *current* upgrade tech to avoid pushing
        # below the shielding warning level.
        if self.shielding is not None:
            floor = self.shielding._amortised_capex_floor(
                upgrade_tag=offer.upgrade_tech,
                q_kw=q_target,
            )
            new_price = max(new_price, floor)

        counter = Offer(
            q_negotiated_kw=q_target,
            t_supply_offered_c=offer.t_supply_offered_c,
            upgrade_tech=offer.upgrade_tech,
            price_eur_mwh=new_price,
            contract_duration_yr=offer.contract_duration_yr,
            is_valid=offer.is_valid,
        )
        return False, counter, self._mfg_welfare(counter)

    # ---- Welfare ---- #
    def _dc_welfare(self, offer: Offer) -> float:
        """Annual welfare of the DC agent (EUR/yr).

        revenue = price * Q * hours / 1000 (MWh)
        capex_amort = CAPEX * Q / amortisation
        tc_residual = transport_cost * Q * hours * distance / 1000
        """
        hours = self.reward_params.operating_hours_per_year
        mwh = offer.q_negotiated_kw * hours / 1000.0
        revenue = offer.price_eur_mwh * mwh
        capex_amort = (
            CAPEX_EUR_PER_KW.get(offer.upgrade_tech, 0.0)
            * offer.q_negotiated_kw
            / max(1, self.reward_params.amortisation_years)
        )
        tc_residual = (
            self.reward_params.transport_cost_eur_per_mwh_km
            * mwh
            * self.mfg.distance_km
        )
        return revenue - capex_amort - tc_residual

    def _mfg_welfare(self, offer: Offer) -> float:
        """Annual welfare of the manufacturing agent (EUR/yr).

        savings = gas_price * Q * hours * (1/COP_effective) / 1000
        cost = price * Q * hours / 1000
        We use the COP of the upgrade tech as a coarse proxy for the
        marginal thermal-to-electric ratio (full energy balance is in the LP).
        """
        hours = self.reward_params.operating_hours_per_year
        mwh = offer.q_negotiated_kw * hours / 1000.0
        cop = COP_BY_UPGRADE.get(offer.upgrade_tech, 1.0)
        # Manufacturer saves cop * mwh of gas equivalent for each mwh delivered.
        savings = self.reward_params.gas_price_eur_per_mwh * mwh * cop
        cost = offer.price_eur_mwh * mwh
        return savings - cost

    # ---- IS-Match update ---- #
    def _update_is_match(self, offer: Offer) -> float:
        """Heuristic IS-Match Score update following a successful upgrade.

        Phase 2 IS-Match Score = beta*RI_temp + gamma*Exergy - delta*DeltaTC.
        At Phase 4 the agent can effectively reduce DeltaTC further through
        contract terms (longer duration, fair price) -- we approximate this
        as a linear function of contract duration and price proximity to the
        amortised floor. This is a placeholder for the formal feedback loop
        defined in Phase 2 step 2.4 (max 5 iter, Δ < 1%).
        """
        if self.shielding is None:
            return self.dc.is_match_score
        floor = self.shielding._amortised_capex_floor(
            upgrade_tag=offer.upgrade_tech,
            q_kw=offer.q_negotiated_kw,
        )
        # Price uplift factor: 1.0 if price == floor, decreasing to 0.0 as price grows
        price_factor = max(0.0, min(1.0, floor / max(offer.price_eur_mwh, 1.0)))
        duration_factor = offer.contract_duration_yr / float(self._duration_max_yr)
        bonus = 0.05 * price_factor + 0.05 * duration_factor
        return float(self.dc.is_match_score + bonus)

    # ---- Reward ---- #
    def _compute_reward(
        self,
        w_dc: float,
        is_match_pre: float,
        is_match_post: float,
        converged: bool,
        timed_out: bool,
        blocked: bool,
    ) -> float:
        # Normalise economic reward by a reference scale so PPO gradients stay
        # well-behaved. Reference: 1 MWh * 100 EUR/MWh * 8000 h = 8e5 EUR/yr.
        reward_econ = w_dc / 8.0e5
        uplift = is_match_post - is_match_pre
        reward_is_match = self.reward_params.lambda_is_match_uplift * (
            uplift / IS_MATCH_TARGET_GAP
        )
        reward = reward_econ + reward_is_match
        if converged:
            reward += 0.5
        if blocked:
            reward -= 0.1
        return float(reward)

    def _co2_saved(self, offer: Offer) -> float:
        hours = self.reward_params.operating_hours_per_year
        mwh = offer.q_negotiated_kw * hours / 1000.0
        return float(mwh * self.reward_params.emission_factor_gas_t_per_mwh)


__all__ = ["ISNegotiationEnv"]
