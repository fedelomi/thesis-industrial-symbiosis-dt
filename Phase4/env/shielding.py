"""Yazdanpanah-style shielding layer for normative-safe RL actions.

The shielding layer (Obsidian B4 / ``concepts/rl-shielding``) intercepts agent
actions before execution and enforces hard constraints derived from the IS
governance protocol. This is preferred to reward-shaping to avoid reward
hacking, as discussed in ``3-layer-framework-and-methodology.md`` Phase 4.1.

Four hard rules are enforced (severity in parentheses):

    1. Q_negotiated <= Q_available(t)                            (BLOCKED)
       Physical constraint from Phase 1 step 1.2.
    2. T_supply_offered >= T_req - 5.0                           (BLOCKED)
       Thermal feasibility with 5 C heat-exchanger pinch tolerance.
    3. price_eur_mwh >= CAPEX_amortized_threshold                (WARNING)
       Economic floor (clipped, not blocked, so the agent can still explore
       sub-optimal pricing but cannot offer below DC break-even).
    4. upgrade_tech == mfg.upgrade_required                      (BLOCKED)
       Normative constraint: the chosen technology must match the upgrade
       mandated by the temperature gap (see Phase 2 Regulatory_KPIs note).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config.reward_params import DEFAULT_REWARD_PARAMS, RewardParams
from config.scenarios import CAPEX_EUR_PER_KW
from env.models import DCProfile, MfgParams

logger = logging.getLogger("phase4.shielding")

# Action layout (matches ISNegotiationEnv): continuous slots 0..3,
# discrete slot 4 = upgrade_tech index (0=direct_HX, 1=HP, 2=CO2_HTHP).
UPGRADE_INDEX = {0: "direct_HX", 1: "HP", 2: "CO2_HTHP"}
UPGRADE_INDEX_INVERSE = {v: k for k, v in UPGRADE_INDEX.items()}

# Pinch tolerance (C) for thermal feasibility (Task 3 spec).
T_PINCH_TOLERANCE = 5.0


@dataclass(slots=True)
class ShieldingDecision:
    """Container summarising a single shielding pass."""

    clipped_action: np.ndarray
    violations: List[str]
    blocked: bool


class ShieldingLayer:
    """Apply normative hard rules to a candidate action.

    Parameters
    ----------
    reward_params:
        Reward configuration (used for amortisation and operating hours).
    no_op_action:
        Action returned when a BLOCKED rule fires. Defaults to a status-quo
        offer (Q=0, T=0, price=0, duration=1, upgrade=correct) which yields
        zero welfare and is interpreted as "no agreement this round".
    """

    def __init__(
        self,
        reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
    ) -> None:
        self.reward_params = reward_params

    # --------------------------------------------------------------------- #
    # Public API                                                             #
    # --------------------------------------------------------------------- #
    def apply(
        self,
        action: np.ndarray,
        dc: DCProfile,
        mfg: MfgParams,
    ) -> Tuple[np.ndarray, List[str]]:
        """Run all four rules and return the clipped action plus violations log.

        Parameters
        ----------
        action:
            1-D numpy array with at least 5 entries::

                [Q_kw, T_offered_c, price_eur_mwh, duration_yr, upgrade_idx]

            The first four are continuous; ``upgrade_idx`` is an integer in
            ``{0, 1, 2}`` mapping to ``direct_HX`` / ``HP`` / ``CO2_HTHP``.
        dc:
            DC profile (provides ``q_available_kw``).
        mfg:
            Manufacturing parameters (provides ``t_req_c`` and
            ``upgrade_required``).

        Returns
        -------
        (clipped_action, violations)
            Clipped action (same shape) and a list of human-readable violation
            strings prefixed with ``WARNING:`` or ``BLOCKED:``.
        """
        decision = self._evaluate(action.copy(), dc, mfg)
        for msg in decision.violations:
            if msg.startswith("BLOCKED"):
                logger.warning(msg)
            else:
                # WARNING-level (clipped but recoverable) at DEBUG; otherwise
                # PPO training spams thousands of identical "price below floor"
                # messages per second when the policy samples low prices.
                logger.debug(msg)
        return decision.clipped_action, decision.violations

    # --------------------------------------------------------------------- #
    # Rule implementation                                                    #
    # --------------------------------------------------------------------- #
    def _evaluate(
        self,
        action: np.ndarray,
        dc: DCProfile,
        mfg: MfgParams,
    ) -> ShieldingDecision:
        violations: List[str] = []
        blocked = False

        q_kw, t_offered, price, duration, upgrade_idx = (
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
            int(round(action[4])),
        )

        # Rule 1 - physical constraint Q <= Q_available
        if q_kw > dc.q_available_kw:
            violations.append(
                f"BLOCKED: Q_negotiated={q_kw:.1f} kW exceeds Q_available="
                f"{dc.q_available_kw:.1f} kW (Phase 1 step 1.2 constraint)."
            )
            action[0] = dc.q_available_kw
            blocked = True
        elif q_kw < 0.0:
            violations.append(
                f"WARNING: Q_negotiated={q_kw:.1f} kW < 0; clipped to 0."
            )
            action[0] = 0.0

        # Rule 2 - thermal feasibility T_offered >= T_req - pinch
        t_min = mfg.t_req_c - T_PINCH_TOLERANCE
        if t_offered < t_min:
            violations.append(
                f"BLOCKED: T_supply_offered={t_offered:.1f} C below "
                f"T_req-pinch={t_min:.1f} C (5 C HX tolerance)."
            )
            action[1] = t_min
            blocked = True

        # Rule 3 - economic floor (warning, clipped only)
        upgrade_tag_for_price = UPGRADE_INDEX.get(upgrade_idx, mfg.upgrade_required)
        floor = self._amortised_capex_floor(
            upgrade_tag=upgrade_tag_for_price,
            q_kw=action[0],
        )
        if price < floor:
            violations.append(
                f"WARNING: price={price:.2f} EUR/MWh below amortised floor="
                f"{floor:.2f} EUR/MWh; clipped (not blocked)."
            )
            action[2] = floor

        # Rule 4 - normative upgrade match
        upgrade_tag = UPGRADE_INDEX.get(upgrade_idx)
        if upgrade_tag != mfg.upgrade_required:
            violations.append(
                f"BLOCKED: upgrade_tech={upgrade_tag!r} does not match "
                f"mfg.upgrade_required={mfg.upgrade_required!r} (normative)."
            )
            action[4] = UPGRADE_INDEX_INVERSE[mfg.upgrade_required]
            blocked = True

        # Duration sanity (non-normative, soft clip)
        if duration < 1.0:
            action[3] = 1.0
        elif duration > 25.0:
            action[3] = 25.0

        return ShieldingDecision(
            clipped_action=action,
            violations=violations,
            blocked=blocked,
        )

    # --------------------------------------------------------------------- #
    # Helpers                                                                #
    # --------------------------------------------------------------------- #
    def _amortised_capex_floor(self, upgrade_tag: str, q_kw: float) -> float:
        """Return the minimum acceptable heat price in EUR/MWh.

        ``floor = (CAPEX_per_kw * q_kw) / (amortisation_years * operating_hours
        * q_kw / 1000)`` simplifies to
        ``floor = CAPEX_per_kw * 1000 / (amortisation_years * operating_hours)``.
        The dependence on ``q_kw`` cancels: amortising fixed CAPEX over the
        same q_kw of thermal flow gives a per-MWh price independent of size.
        """
        capex = CAPEX_EUR_PER_KW.get(upgrade_tag, 0.0)
        denom = self.reward_params.amortisation_years * self.reward_params.operating_hours_per_year
        if denom == 0:
            return 0.0
        return capex * 1000.0 / denom


__all__ = ["ShieldingLayer", "ShieldingDecision", "UPGRADE_INDEX", "UPGRADE_INDEX_INVERSE"]
