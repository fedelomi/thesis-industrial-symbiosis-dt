"""Typed dataclasses for the Phase 4 IS-negotiation environment.

All numerical defaults reference the Phase 2 LC outputs documented in the
thesis Obsidian wiki:

    * DC parameters: ``implementation-decisions.md`` D5 (Edge_LC 500 kW,
      Mid_LC 3.2 MW, Hyperscale_LC 25 MW; T_CDU_return ~= 46-50 C)
    * Plant parameters: ``implementation-decisions.md`` D2 (sector-parametric,
      medium-proximity tier representatives)
    * IS-Match scores: ``Phase2_ISMatch/results/step_2_4_delta_tc_calibration_lc.csv``
      iteration=4 final values (range 0.4700-0.5767)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

UpgradeTech = Literal["direct_HX", "HP", "CO2_HTHP"]
ShiftPattern = Literal["continuous", "2shift", "3shift"]


@dataclass(slots=True)
class DCProfile:
    """Data centre liquid-cooling thermal profile.

    Values are derived from Phase 1 LC outputs and Phase 2 IS-Match calibration
    (see ``phase-1-2-3-roadmap.md`` Phase 1 LC section, Phase 2 step 2.4).

    Attributes
    ----------
    dc_id:
        Stable identifier (e.g. ``"Edge_LC"``, ``"Mid_LC"``, ``"Hyperscale_LC"``).
    q_available_kw:
        Negotiable thermal capacity (kW) after pump losses
        (``Q_negotiated <= 0.90 * Q_nominal`` per D5 capture fraction).
    t_supply_c:
        Mean LC supply temperature (°C). Edge/Mid ~= 47.6 C, Hyperscale ~= 46.0 C.
    exergy_dt:
        Normalised exergy index in [0, 1] from Phase 2 (per-scenario normalisation).
    t_availability:
        Annual availability fraction (Q > Q_min). Phase 1 LC = 1.0 for all DCs.
    is_match_score:
        IS-Match Score from Phase 2 step 2.4 iteration=4 (post DeltaTC calibration).
    q_available_profile:
        Optional annual time-series of Q_available (kW). When present, the
        environment can sample a random snapshot at each reset() to simulate
        a DT-grounded agent observing realistic intra-annual variability.
        Set to None (default) for LC scenarios where t_availability = 1.0
        and Q is taken as constant.
    """

    dc_id: str
    q_available_kw: float
    t_supply_c: float
    exergy_dt: float
    t_availability: float
    is_match_score: float
    q_available_profile: Optional[np.ndarray] = None


@dataclass(slots=True)
class MfgParams:
    """Manufacturing plant surrogate parameters (sector-parametric).

    The manufacturing side has no calibrated physical DT in this thesis scope
    (declared explicitly in Cap. 3.5.1). Parameters are drawn from the medium-
    proximity profile per tier in the Phase 2 LC dataset (D2 update 2026-04-30).

    Attributes
    ----------
    plant_id:
        Stable identifier (e.g. ``"LowT_02_Agro_Medium"``).
    t_req_c:
        Required process supply temperature (°C).
    q_process_kw:
        Plant thermal demand (kW). For Edge_LC × MidT/HighT, Q_demand >> Q_available
        and partial supply is handled in the environment (Q_partial mode).
    distance_km:
        Network distance DC -> plant. D_MAX_KM = 200 km (D2).
    upgrade_required:
        Mandatory upgrade technology to bridge T_supply -> T_req
        (per Phase 2 LC Regulatory_KPIs note: HP for LowT/MidT, CO2_HTHP for HighT).
    shift_pattern:
        Demand profile (continuous/2shift/3shift).
    """

    plant_id: str
    t_req_c: float
    q_process_kw: float
    distance_km: float
    upgrade_required: UpgradeTech
    shift_pattern: ShiftPattern


@dataclass(slots=True)
class Offer:
    """Single round offer in the bilateral IS negotiation.

    Format matches the protocol described in ``3-layer-framework-and-methodology.md``
    Strato 3: ``[Q_negotiated, T_supply, upgrade_tech, prezzo, durata]``.

    Attributes
    ----------
    q_negotiated_kw:
        Thermal flow proposed (kW). Hard rule: ``Q_negotiated <= Q_available``.
    t_supply_offered_c:
        Supply temperature offered (°C). Hard rule: ``T_offered >= T_req - 5``.
    upgrade_tech:
        Proposed upgrade technology (must match ``MfgParams.upgrade_required``).
    price_eur_mwh:
        Heat price (€/MWh). Hard rule: ``price >= CAPEX_amortized_threshold``.
    contract_duration_yr:
        Contract horizon (years). Typical IS contracts: 5-15 yr.
    is_valid:
        Set by ``ShieldingLayer.apply()`` (True if no BLOCKED rules violated).
    """

    q_negotiated_kw: float
    t_supply_offered_c: float
    upgrade_tech: UpgradeTech
    price_eur_mwh: float
    contract_duration_yr: int
    is_valid: bool = True


@dataclass(slots=True)
class NegotiationOutcome:
    """Final outcome of a single negotiation episode.

    Attributes
    ----------
    converged:
        True if both agents accepted before ``max_rounds``.
    n_rounds:
        Number of rounds executed.
    offer_final:
        Final offer (last counter-offer or accepted offer).
    welfare_dc, welfare_mfg, welfare_total:
        Per-agent and total welfare values (€/yr).
    shapley_dc, shapley_mfg:
        Closed-form Shapley allocations (see ``agents/shapley.py``, B5).
    shapley_fairness_ratio:
        ``min(phi_dc, phi_mfg) / max(phi_dc, phi_mfg)`` in [0, 1]. Target > 0.8.
    co2_saved_t_yr:
        Annual CO2 emissions avoided (t/yr) from natural gas displacement at the
        manufacturing plant. Default emission factor = 0.20 tCO2/MWh_thermal
        (IEA 2024 for natural gas with 90% boiler efficiency). Configurable in
        ``config/reward_params.py``.
    is_match_uplift:
        ``is_match_score_post - is_match_score_pre`` realised at episode end.
        Phase 4 objective per ``phase-1-2-3-roadmap.md`` is to push the score
        from "marginal" (0.36-0.53) to "high-priority" (> 0.60).
    """

    converged: bool
    n_rounds: int
    offer_final: Offer
    welfare_dc: float
    welfare_mfg: float
    welfare_total: float
    shapley_dc: float
    shapley_mfg: float
    shapley_fairness_ratio: float
    co2_saved_t_yr: float = 0.0
    is_match_uplift: float = 0.0


__all__ = [
    "DCProfile",
    "MfgParams",
    "Offer",
    "NegotiationOutcome",
    "UpgradeTech",
    "ShiftPattern",
]
