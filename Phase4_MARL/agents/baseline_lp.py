"""Centralised LP welfare baseline for Price-of-Anarchy upper bound.

Solves, per scenario, the social-planner problem::

    max  welfare_total = revenue_WHR(price*, Q) - transport_cost(Q) - CAPEX_amort(Q)
     Q
    s.t. 0 <= Q <= min(Q_available, Q_demand)
         upgrade_tech == mfg.upgrade_required   (continuous relaxation; MILP is future work)

The optimum returned is the **upper bound** of the negotiated welfare (since
in the centralised solution there is no negotiation friction). Price-of-
Anarchy is defined as ``PoA = welfare_decentralised / welfare_centralised``
and is computed in ``train/run_ablation.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cvxpy as cp

from config.reward_params import DEFAULT_REWARD_PARAMS, RewardParams
from config.scenarios import CAPEX_EUR_PER_KW, COP_BY_UPGRADE
from env.models import DCProfile, MfgParams


def _solve_one(
    dc: DCProfile,
    mfg: MfgParams,
    reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
) -> Tuple[float, float, float]:
    """Solve the social-planner LP for one (DC, plant) pair.

    Returns
    -------
    (welfare_optimal, q_optimal, price_optimal)
    """
    hours = reward_params.operating_hours_per_year
    q_max = min(dc.q_available_kw, mfg.q_process_kw)
    upgrade = mfg.upgrade_required
    capex_per_kw = CAPEX_EUR_PER_KW[upgrade]
    cop = COP_BY_UPGRADE[upgrade]
    transport = reward_params.transport_cost_eur_per_mwh_km * mfg.distance_km
    gas_price = reward_params.gas_price_eur_per_mwh

    # In the centralised problem the internal "price" is a transfer that
    # cancels between the two parties. Welfare reduces to::
    #     savings - capex_amort - transport
    #   = gas_price * cop * MWh - CAPEX*Q/years - transport_per_mwh * MWh
    # so the program is linear in Q.

    q = cp.Variable(nonneg=True)
    mwh = q * hours / 1000.0
    savings = gas_price * cop * mwh
    capex_amort = capex_per_kw * q / max(1, reward_params.amortisation_years)
    transport_cost = transport * mwh

    objective = cp.Maximize(savings - capex_amort - transport_cost)
    constraints = [q <= q_max]
    prob = cp.Problem(objective, constraints)

    # Try solvers in order of preference; cvxpy 1.4+ ships CLARABEL as default
    # but older installs may rely on ECOS / SCS. Fall through gracefully.
    last_error: Exception | None = None
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            if solver in cp.installed_solvers():
                prob.solve(solver=solver, verbose=False)
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    else:
        # No solver succeeded -> fall back to closed-form (LP is linear in Q
        # with single upper bound, so the optimum is trivial).
        marginal = (
            (gas_price * cop - transport) * hours / 1000.0
            - capex_per_kw / max(1, reward_params.amortisation_years)
        )
        if marginal > 0:
            q_opt = q_max
            welfare = marginal * q_max
        else:
            q_opt = 0.0
            welfare = 0.0
        return float(welfare), float(q_opt), float(gas_price * cop - transport)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return float("nan"), float("nan"), float("nan")

    q_opt = float(q.value)
    welfare = float(prob.value)

    # The implied break-even transfer price is the gas savings per MWh
    # net of transport on the MFG side, but since it is an internal transfer
    # we report the marginal value of Q (savings_per_MWh).
    price_opt = gas_price * cop - transport
    return welfare, q_opt, price_opt


def compute_centralized_welfare(
    scenarios: Dict[str, Tuple[DCProfile, MfgParams]],
    reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
) -> Dict[str, float]:
    """Compute the optimal welfare upper bound for each scenario.

    Parameters
    ----------
    scenarios:
        Mapping ``scenario_id -> (DCProfile, MfgParams)`` (use
        ``config.scenarios.build_scenarios()``).
    reward_params:
        Reward / economic parameters (default = ``DEFAULT_REWARD_PARAMS``).

    Returns
    -------
    dict
        ``{scenario_id: welfare_optimal_eur_per_year}``.

    Notes
    -----
    * The upgrade-technology variable is fixed to ``mfg.upgrade_required``
      (continuous relaxation of the MILP). A full MILP extension that allows
      the planner to also choose the upgrade is future work
      (see Cap. 6.3, "centralised MILP extension").
    * Negative welfare means even the social planner cannot produce a
      profitable IS contract -- this happens for HighT under the default
      gas price assumption and is the expected boundary case noted in the
      README.
    """
    results: Dict[str, float] = {}
    for s_id, (dc, mfg) in scenarios.items():
        welfare, _, _ = _solve_one(dc, mfg, reward_params)
        results[s_id] = welfare
    return results


def compute_centralized_detail(
    scenarios: Dict[str, Tuple[DCProfile, MfgParams]],
    reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
) -> Dict[str, Dict[str, float]]:
    """Same as :func:`compute_centralized_welfare` but returns Q* and price* too."""
    out: Dict[str, Dict[str, float]] = {}
    for s_id, (dc, mfg) in scenarios.items():
        welfare, q_opt, price_opt = _solve_one(dc, mfg, reward_params)
        out[s_id] = {
            "welfare_optimal_eur_yr": welfare,
            "q_optimal_kw": q_opt,
            "shadow_price_eur_per_mwh": price_opt,
        }
    return out


__all__ = ["compute_centralized_welfare", "compute_centralized_detail"]
