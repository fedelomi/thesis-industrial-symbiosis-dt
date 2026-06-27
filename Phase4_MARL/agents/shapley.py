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


def shapley_fairness_ratio(phi_dc: float, phi_mfg: float,
                           sign_aware: bool = False) -> float:
    """Fairness ratio ``min / max`` of the Shapley allocations.

    Returns a value in ``[0, 1]``. Higher means more balanced.
    Phase 4 target (Cap. 5.3): ratio > 0.8.

    Two definitions are supported:

    * ``sign_aware=False`` (default, **canonical Chapter 5 behavior**): uses
      ``min(|phi_dc|, |phi_mfg|) / max(|phi_dc|, |phi_mfg|)``. Frozen for
      backward compatibility with the canonical PoA/Shapley CIs of Section 5.5.
      Caveat: this definition reports ``(-10, +10)`` as fairness 1.0, which
      is *not* balanced from a participation-incentive standpoint.

    * ``sign_aware=True`` (recommended for defense and follow-up): if the two
      allocations have opposite sign, the ratio collapses to 0 (one party
      pays for the other to gain, opposite of participation balance). If both
      are non-negative (or both non-positive), behaves as the canonical
      definition on absolute values. Use this for the FW3 IR-floor analysis
      and any sign-sensitive subgroup interpretation.

    The Opus 4.8 independent audit (A4) flagged the abs-based default as an
    edge-case distortion. The canonical -0.207 Shapley gap of Section 5.5 was
    measured under ``sign_aware=False``; re-evaluation under ``sign_aware=True``
    is a non-blocking confirmatory diagnostic and is recommended as part of
    FW3 cloud validation.
    """
    if sign_aware:
        # Opposite signs => one party gains, the other loses => fairness 0.
        if (phi_dc < 0.0 < phi_mfg) or (phi_mfg < 0.0 < phi_dc):
            return 0.0
    a, b = abs(phi_dc), abs(phi_mfg)
    if a == 0.0 and b == 0.0:
        return 0.0
    lo, hi = (a, b) if a < b else (b, a)
    if hi == 0.0:
        return 0.0
    return lo / hi


__all__ = ["shapley_value_bilateral", "shapley_fairness_ratio"]
