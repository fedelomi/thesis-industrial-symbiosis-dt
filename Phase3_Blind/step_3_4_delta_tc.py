"""
Step 3.4 Blind - Symbolic delta_TC (regulatory-friction) estimator
==================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Derives the per-tier regulatory transaction-cost vector delta_TC from the corpus,
deterministically and with full provenance. delta_TC is expressed in the [0,1]
IS-Match unit (Eq. 3.3 delta-weighted penalty term) and is decomposed into
corpus-grounded ordinal factors:

  f_upgrade    : temperature-lift upgrade complexity (direct/HP/CO2-HTHP), the
                 physically irreducible part. Grounded in F11/F22.
  f_temp       : temperature-band standard exposure (FDM BREF bands, F24).
  f_admin      : EMS/audit administrative burden from the tier's energy use vs the
                 85 TJ EED Art. 11 threshold (F10).
  base         : DC-side reporting baseline common to all tiers (Art. 12 / Del. Reg
                 2024/1364, F10/F25).
  relief       : jurisdiction relief (Italian TEE incentive, Danish planning
                 support) that lowers friction.

The Phase-2-facing output is the reduction_factors JSON (r in [0.1, 0.3]) consumed
by step_2_4_delta_tc_calibration_lc.py. r is the *informational* share of the
friction, the fraction the institutional-LLM layer removes by resolving the Gap-3
information asymmetry (the physical upgrade cost is not reducible by retrieval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

from step_3_0_config import (
    DELTA_TC_JSON,
    REDUCTION_MAX,
    REDUCTION_MIN,
    TIER_DEMAND_KW,
    TIER_PROCESS_KEY,
    Country,
    Tier,
    UpgradeTech,
    get_logger,
    required_upgrade,
)
from step_3_1_ingest import RegulatoryKB, load_kb

logger = get_logger(__name__)

# Fixed, documented combination weights (sum of positive structural weights = 1.0).
W_UPGRADE: float = 0.30   # physical (irreducible) friction
W_TEMP: float = 0.30      # temperature-band standard exposure (informational)
W_ADMIN: float = 0.15     # EMS/audit burden (informational)
W_BASE: float = 0.25      # DC-side reporting baseline (informational)

# Ordinal upgrade-complexity weights (direct < HP < CO2-HTHP), corpus F11/F22.
UPGRADE_COMPLEXITY: Dict[UpgradeTech, float] = {
    UpgradeTech.DIRECT_HX: 0.0,
    UpgradeTech.HP: 0.5,
    UpgradeTech.CO2_HTHP: 1.0,
}
# Ordinal temperature-band standard exposure, FDM BREF bands (F24):
# LowT cleaning/pasteurisation, MidT evaporation/sterilisation, HighT steam/BAT.
TEMP_STANDARD_EXPOSURE: Dict[Tier, float] = {Tier.LOW: 0.25, Tier.MID: 0.55, Tier.HIGH: 1.0}

# EED Art. 11 EMS obligation threshold (TJ/yr) and operating hours for energy sizing.
EMS_THRESHOLD_TJ: float = 85.0
OPERATING_HOURS_YR: float = 7000.0
MWH_TO_TJ: float = 0.0036

# Jurisdiction relief (subtracted from friction): IT TEE incentive, DK planning support.
COUNTRY_RELIEF: Dict[Country, float] = {Country.EU: 0.0, Country.ITALY: 0.05, Country.DENMARK: 0.07}

# Phase 2 baseline delta_TC anchors (step_2_1 inline defaults) for the convergence check.
PHASE2_BASELINE_ANCHORS: Dict[Tier, float] = {Tier.LOW: 0.25, Tier.MID: 0.42, Tier.HIGH: 0.55}


@dataclass(slots=True)
class TierBurden:
    """Decomposed regulatory burden for one tier."""

    tier: Tier
    upgrade: UpgradeTech
    f_upgrade: float
    f_temp: float
    f_admin: float
    delta_tc_norm_eu: float
    reduction_factor: float
    info_share: float
    energy_tj: float


def _admin_factor(tier: Tier) -> tuple[float, float]:
    """Return (f_admin, annual_energy_TJ) for a tier from its thermal demand.

    f_admin scales the manufacturing energy use against the 85 TJ EMS threshold,
    capped at 1.0.
    """
    annual_mwh = TIER_DEMAND_KW[tier] * OPERATING_HOURS_YR / 1000.0
    energy_tj = annual_mwh * MWH_TO_TJ
    return min(energy_tj / EMS_THRESHOLD_TJ, 1.0), energy_tj


def compute_tier_burden(tier: Tier, supply_temp_c: float, country: Country = Country.EU) -> TierBurden:
    """Compute the decomposed delta_TC burden for one tier (deterministic).

    Args:
        tier: Manufacturing temperature band.
        supply_temp_c: DC waste-heat supply temperature (C).
        country: Jurisdiction for the relief term (EU = no relief).

    Returns:
        A :class:`TierBurden` with the decomposed factors, the EU-baseline
        delta_TC_norm and the informational reduction factor.
    """
    upgrade = required_upgrade(TIER_T_REQ(tier), supply_temp_c)
    f_up = UPGRADE_COMPLEXITY[upgrade]
    f_temp = TEMP_STANDARD_EXPOSURE[tier]
    f_admin, energy_tj = _admin_factor(tier)

    info_component = W_TEMP * f_temp + W_ADMIN * f_admin + W_BASE
    phys_component = W_UPGRADE * f_up
    delta_eu = max(0.0, min(1.0, info_component + phys_component - COUNTRY_RELIEF[Country.EU]))

    total = info_component + phys_component
    info_share = info_component / total if total > 0 else 0.0
    reduction = max(REDUCTION_MIN, min(REDUCTION_MAX,
                                       REDUCTION_MIN + (REDUCTION_MAX - REDUCTION_MIN) * info_share))

    return TierBurden(
        tier=tier, upgrade=upgrade, f_upgrade=f_up, f_temp=f_temp, f_admin=f_admin,
        delta_tc_norm_eu=round(delta_eu, 4), reduction_factor=round(reduction, 4),
        info_share=round(info_share, 4), energy_tj=round(energy_tj, 1),
    )


def TIER_T_REQ(tier: Tier) -> float:
    """Required process temperature for a tier (C)."""
    from step_3_0_config import TIER_T_REQ_C
    return TIER_T_REQ_C[tier]


@dataclass(slots=True)
class DeltaTcResult:
    """Full delta_TC output for all tiers."""

    burdens: Dict[Tier, TierBurden] = field(default_factory=dict)

    def reduction_factors(self) -> Dict[str, float]:
        """Return the Phase-2-facing reduction-factor map keyed by process key."""
        return {TIER_PROCESS_KEY[t]: b.reduction_factor for t, b in self.burdens.items()}

    def delta_tc_norm(self) -> Dict[str, float]:
        """Return the absolute delta_TC_norm map keyed by tier name."""
        return {t.value: b.delta_tc_norm_eu for t, b in self.burdens.items()}


def estimate_delta_tc(kb: RegulatoryKB | None = None, supply_temp_c: float = 47.6) -> DeltaTcResult:
    """Compute delta_TC for all three tiers.

    Args:
        kb: Optional KB (used only to count corpus grounding evidence for provenance).
        supply_temp_c: DC supply temperature for the upgrade computation.

    Returns:
        A :class:`DeltaTcResult`.
    """
    burdens = {t: compute_tier_burden(t, supply_temp_c) for t in Tier}
    return DeltaTcResult(burdens=burdens)


def write_delta_tc_json(result: DeltaTcResult, path=DELTA_TC_JSON, kb: RegulatoryKB | None = None) -> Dict:
    """Write the Phase-2 drop-in delta_TC JSON and return the written payload.

    The schema matches step_2_4_delta_tc_calibration_lc.py
    (`_load_phase3_reductions`): a top-level ``reduction_factors`` map keyed by the
    Phase 2 process keys. Extra keys (delta_tc_norm, provenance) are additive and
    ignored by the consumer.
    """
    grounding = {}
    if kb is not None:
        grounding = {
            "eed_doc": "F10",
            "kpi_reg_doc": "F25",
            "fdm_bref_doc": "F24",
            "italy_tee_docs": [f.doc_id for f in kb.by_jurisdiction("Italy")],
            "denmark_docs": [f.doc_id for f in kb.by_jurisdiction("Denmark")],
            "corpus_documents": len(kb),
        }
    payload = {
        "reduction_factors": result.reduction_factors(),
        "delta_tc_norm": result.delta_tc_norm(),
        "method": "neuro-symbolic burden score (step_3_4_delta_tc)",
        "weights": {"upgrade": W_UPGRADE, "temp": W_TEMP, "admin": W_ADMIN, "base": W_BASE},
        "provenance": {
            t.value: {
                "upgrade": b.upgrade.value, "f_upgrade": b.f_upgrade, "f_temp": b.f_temp,
                "f_admin": b.f_admin, "energy_tj": b.energy_tj, "info_share": b.info_share,
                "delta_tc_norm": b.delta_tc_norm_eu, "reduction_factor": b.reduction_factor,
            }
            for t, b in result.burdens.items()
        },
        "corpus_grounding": grounding,
        "phase2_baseline_anchors": {t.value: v for t, v in PHASE2_BASELINE_ANCHORS.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote delta_TC JSON -> %s", path)
    return payload


if __name__ == "__main__":
    kb = load_kb()
    res = estimate_delta_tc(kb)
    payload = write_delta_tc_json(res, kb=kb)
    print("\ndelta_TC_norm (corpus-derived) vs Phase 2 baseline anchors:")
    for t in Tier:
        b = res.burdens[t]
        anchor = PHASE2_BASELINE_ANCHORS[t]
        print(f"  {t.value:6}: delta_tc_norm={b.delta_tc_norm_eu:.3f} (anchor {anchor:.2f})  "
              f"reduction_factor={b.reduction_factor:.3f}  upgrade={b.upgrade.value}  "
              f"info_share={b.info_share:.3f}")
    print(f"\nreduction_factors (Phase 2 drop-in): {res.reduction_factors()}")
    # Monotonicity / range checks
    dvals = [res.burdens[t].delta_tc_norm_eu for t in Tier]
    rvals = [res.burdens[t].reduction_factor for t in Tier]
    print(f"delta_tc monotone increasing: {dvals == sorted(dvals)}")
    print(f"all reductions in [0.1,0.3]: {all(REDUCTION_MIN <= r <= REDUCTION_MAX for r in rvals)}")
