"""Closed-form Shapley value for bilateral (N=2) IS coalitions.

Reference (Obsidian): B5 -- Shapley value for fair allocation of cooperative
gains in industrial-symbiosis coalitions; in the bilateral case the canonical
formula reduces to a closed-form average of the marginal contributions
(Shapley 1953; see also ``concepts/shapley-value`` in the wiki).

For N=2 with players ``i, j`` and characteristic function ``v``::

    phi_i = (1/2) * v({i}) + (1/2) * (v({i, j}) - v({j}))

i.e. the Shapley value of player ``i`` is the average of its stand-alone
value and its marginal contribution to the grand coalition.
"""

from __future__ import annotations

from typing import Tuple


def shapley_value_bilateral(
    v_dc: float,
    v_mfg: float,
    v_coalition: float,
) -> Tuple[float, float]:
    """Closed-form Shapley split for a 2-agent coalition (B5).

    Parameters
    ----------
    v_dc:
        Stand-alone value of the data centre agent (`v({DC})`).
    v_mfg:
        Stand-alone value of the manufacturing agent (`v({MFG})`).
    v_coalition:
        Joint value of the grand coalition (`v({DC, MFG})`).

    Returns
    -------
    (phi_dc, phi_mfg)
        Shapley allocations summing (within floating-point precision) to
        ``v_coalition``.

    Notes
    -----
    The formula is symmetric and satisfies the four Shapley axioms
    (efficiency, symmetry, dummy, additivity). Negative stand-alone values
    are permitted -- they represent agents that would lose welfare absent the
    coalition (e.g. a DC paying for waste-heat dissipation it cannot avoid).
    """
    phi_dc = 0.5 * v_dc + 0.5 * (v_coalition - v_mfg)
    phi_mfg = 0.5 * v_mfg + 0.5 * (v_coalition - v_dc)
    return phi_dc, phi_mfg


def shapley_fairness_ratio(phi_dc: float, phi_mfg: float) -> float:
    """Fairness ratio ``min / max`` of the Shapley allocations.

    Returns a value in ``[0, 1]`` (or 0 if both are zero). Higher means more
    balanced. Phase 4 target (Cap. 5.3): ratio > 0.8.

    Negative allocations are treated by absolute value to keep the ratio
    interpretable as "balance"; an allocation pair (-10, +10) is *not*
    balanced from a participation-incentive standpoint.
    """
    a, b = abs(phi_dc), abs(phi_mfg)
    if a == 0.0 and b == 0.0:
        return 0.0
    lo, hi = (a, b) if a < b else (b, a)
    if hi == 0.0:
        return 0.0
    return lo / hi


__all__ = ["shapley_value_bilateral", "shapley_fairness_ratio"]
