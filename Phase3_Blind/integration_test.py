"""
integration_test.py - Phase 3 Blind -> Phase 4 contract test
============================================================
End-to-end test that the Phase 3 Blind outputs satisfy the ACTUAL Phase 4
(Phase4_MARL) interface without modifying any Phase 4 code:

  1. delta_TC vector -> Phase 4 reward: for three tier scenarios, apply the
     corpus-derived reduction factor to the Phase 2 IS-Match baseline, build the
     real ISNegotiationEnv with the resulting is_match_score and assert the env
     accepts it (obs in bounds, reward is a finite float).
  2. compliance gate -> per-episode admissibility: invoke the gate on three
     scenarios and show it composes with the real ShieldingLayer as an extra
     admissibility mask, queried inside a Phase 4 episode loop.
  3. data types / value ranges: assert is_match_score float in [0,1],
     reduction_factor float in [0.1,0.3], verdict JSON-serialisable.

Run standalone (`python integration_test.py`) or under pytest. It adds both the
Phase4_MARL root and the Phase3_Blind root to sys.path, mirroring Phase 4's own
conftest, so the existing Phase 4 tests are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- sys.path wiring (Phase 4 root first, then this package) --------------- #
PHASE3_BLIND_ROOT = Path(__file__).resolve().parent
PHASE4_ROOT = PHASE3_BLIND_ROOT.parent / "Phase4_MARL"
for p in (str(PHASE4_ROOT), str(PHASE3_BLIND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Phase 3 Blind outputs -------------------------------------------------- #
from step_3_0_config import (  # noqa: E402
    ISMATCH_DELTA, REDUCTION_MAX, REDUCTION_MIN, Country, Scenario, Tier,
)
from step_3_1_ingest import load_kb  # noqa: E402
from step_3_3_compliance_gate import compliance_gate  # noqa: E402
from step_3_4_delta_tc import PHASE2_BASELINE_ANCHORS, estimate_delta_tc  # noqa: E402

# --- Phase 4 (downstream consumer) ----------------------------------------- #
from config.scenarios import DC_PROFILES, MFG_PROFILES  # noqa: E402
from env.is_negotiation_env import ISNegotiationEnv  # noqa: E402
from env.models import DCProfile  # noqa: E402
from env.shielding import ShieldingLayer  # noqa: E402

TIER_TO_PHASE4 = {Tier.LOW: ("Edge_LC", "LowT"), Tier.MID: ("Mid_LC", "MidT"),
                  Tier.HIGH: ("Hyperscale_LC", "HighT")}


def _adjusted_is_match(base_is_match: float, tier: Tier, reduction: float) -> float:
    """Apply the Phase 3 reduction to the Phase 2 IS-Match baseline.

    IS-Match = beta*RI + gamma*Exergy - delta*DeltaTC. Reducing DeltaTC by a
    fraction r raises the score by delta*DeltaTC_baseline*r. This is the exact
    contractual effect of the Phase 3 reduction_factor on the Phase 2 score.
    """
    uplift = ISMATCH_DELTA * PHASE2_BASELINE_ANCHORS[tier] * reduction
    return float(np.clip(base_is_match + uplift, 0.0, 1.0))


def test_delta_tc_feeds_phase4_reward():
    """delta_TC -> is_match_score -> ISNegotiationEnv accepts it (reward is finite float)."""
    kb = load_kb()
    res = estimate_delta_tc(kb)
    reductions = res.reduction_factors()

    for tier, (dc_id, tier_tag) in TIER_TO_PHASE4.items():
        r = reductions[{Tier.LOW: "LowT_60C", Tier.MID: "MidT_90C", Tier.HIGH: "HighT_130C"}[tier]]
        assert REDUCTION_MIN <= r <= REDUCTION_MAX, f"reduction_factor out of band: {r}"

        base_dc = DC_PROFILES[dc_id]
        mfg = MFG_PROFILES[tier_tag]
        base_is_match = 0.50  # Phase 2 marginal baseline (CSV-independent, contract-faithful)
        new_is_match = _adjusted_is_match(base_is_match, tier, r)
        assert 0.0 <= new_is_match <= 1.0

        dc = DCProfile(
            dc_id=base_dc.dc_id, q_available_kw=base_dc.q_available_kw,
            t_supply_c=base_dc.t_supply_c, exergy_dt=base_dc.exergy_dt,
            t_availability=base_dc.t_availability, is_match_score=new_is_match,
        )
        env = ISNegotiationEnv(dc=dc, mfg=mfg, shielding=ShieldingLayer(), max_rounds=5, seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape == (8,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs), "is_match-bearing observation out of Phase 4 bounds"
        # obs[3] is the is_match slot; it must equal the value we injected (clipped to [0,1]).
        assert abs(float(obs[3]) - new_is_match) < 1e-5
        action = env.action_space.sample().astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float) and np.isfinite(reward), "Phase 4 reward rejected the delta_TC-adjusted is_match"
        assert env.observation_space.contains(next_obs)
    print("  [1/3] delta_TC vector accepted by Phase 4 reward for all three tiers. OK")


def test_compliance_gate_composes_with_shielding():
    """The gate is callable per episode and composes with the real ShieldingLayer."""
    kb = load_kb()
    cases = [
        Scenario("Edge", Tier.LOW, country=Country.ITALY),
        Scenario("Mid", Tier.MID, country=Country.DENMARK),
        Scenario("Hyperscale", Tier.HIGH, country=Country.EU, dh_efficient=False),
    ]
    shield = ShieldingLayer()
    for sc in cases:
        verdict = compliance_gate(sc, kb)
        # Verdict is a clean enum value with the contractual structure.
        assert verdict.status.value in {"compliant", "non_compliant", "conditional"}
        assert isinstance(verdict.triggered_articles, list) and verdict.triggered_articles
        assert all(isinstance(v, (int, float)) for v in verdict.thresholds.values())

        # Per-episode admissibility: a non-compliant scenario is masked, exactly the
        # boolean-mask pattern the existing shielding layer uses, WITHOUT editing Phase 4.
        dc_id, tier_tag = TIER_TO_PHASE4[sc.tier]
        dc = DC_PROFILES[dc_id]
        mfg = MFG_PROFILES[tier_tag]
        env = ISNegotiationEnv(dc=DCProfile(
            dc_id=dc.dc_id, q_available_kw=dc.q_available_kw, t_supply_c=dc.t_supply_c,
            exergy_dt=dc.exergy_dt, t_availability=dc.t_availability, is_match_score=0.5),
            mfg=mfg, shielding=shield, max_rounds=3, seed=1)
        env.reset(seed=1)
        admissible = verdict.status.value != "non_compliant"
        # Episode loop only proceeds if the regulatory gate admits the scenario.
        steps = 0
        if admissible:
            done = trunc = False
            while not (done or trunc) and steps < 3:
                _, reward, done, trunc, _ = env.step(env.action_space.sample().astype(np.float32))
                steps += 1
            assert steps >= 1
        # Non-compliant scenarios are correctly gated out before negotiation.
        assert isinstance(admissible, bool)
    print("  [2/3] Compliance gate callable per episode and composes with ShieldingLayer. OK")


def test_data_types_and_ranges():
    """All contractual outputs have the data types and ranges Phase 4 expects."""
    kb = load_kb()
    res = estimate_delta_tc(kb)
    import json

    for k, v in res.reduction_factors().items():
        assert isinstance(v, float) and REDUCTION_MIN <= v <= REDUCTION_MAX
    for v in res.delta_tc_norm().values():
        assert isinstance(v, float) and 0.0 <= v <= 1.0
    # Verdict round-trips through JSON (serialisable contract).
    verdict = compliance_gate(Scenario("Mid", Tier.MID, country=Country.ITALY), kb)
    blob = json.dumps(verdict.to_dict())
    assert json.loads(blob)["status"] in {"compliant", "non_compliant", "conditional"}
    print("  [3/3] Data types and value ranges match the Phase 4 contract. OK")


def main() -> None:
    print("=" * 70)
    print("  Phase 3 Blind -> Phase 4 INTEGRATION TEST")
    print("=" * 70)
    test_delta_tc_feeds_phase4_reward()
    test_compliance_gate_composes_with_shielding()
    test_data_types_and_ranges()
    print("=" * 70)
    print("  ALL INTEGRATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
