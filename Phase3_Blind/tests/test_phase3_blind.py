"""
Unit tests for the Phase 3 Blind Neuro-Symbolic Regulatory Reasoner.

Hermetic: no network, no API. Cover gate determinism and correctness, delta_TC
range and monotonicity, the Phase-2 JSON schema, KB ingestion, retrieval, and the
NSRR-over-baseline advantage.
"""

from __future__ import annotations

import json

import pytest

from step_3_0_config import (
    DELTA_TC_JSON,
    REDUCTION_MAX,
    REDUCTION_MIN,
    Country,
    Scenario,
    Tier,
    UpgradeTech,
    Verdict,
    required_upgrade,
)
from step_3_1_ingest import load_kb
from step_3_3_compliance_gate import compliance_gate
from step_3_4_delta_tc import estimate_delta_tc, write_delta_tc_json
from step_3_5_answerer import BaselineAnswerer, NSRRAnswerer


@pytest.fixture(scope="module")
def kb():
    try:
        return load_kb()
    except FileNotFoundError:
        pytest.skip("corpus_facts.json not available")


# --------------------------------------------------------------------------- #
# 1. Gate determinism                                                          #
# --------------------------------------------------------------------------- #
def test_gate_is_deterministic(kb):
    """The gate returns an identical verdict on repeated calls (reproducibility)."""
    s = Scenario("Mid", Tier.MID, country=Country.DENMARK)
    results = {compliance_gate(s, kb).status for _ in range(10)}
    assert len(results) == 1
    # And the full serialised result is stable.
    a = compliance_gate(s, kb).to_dict()
    b = compliance_gate(s, kb).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# 2. Gate verdict correctness on known regulatory cases                        #
# --------------------------------------------------------------------------- #
def test_gate_verdicts_match_regulation(kb):
    """Edge below 1 MW is compliant; Art. 26 + non-efficient DH is non-compliant;
    documented negative CBA is conditional (Art. 26(8) exemption)."""
    edge = compliance_gate(Scenario("Edge", Tier.LOW, country=Country.ITALY), kb)
    assert edge.status == Verdict.COMPLIANT

    non_eff = compliance_gate(Scenario("Mid", Tier.MID, country=Country.EU, dh_efficient=False), kb)
    assert non_eff.status == Verdict.NON_COMPLIANT

    exempt = compliance_gate(Scenario("Hyperscale", Tier.HIGH, country=Country.EU, cba_positive=False), kb)
    assert exempt.status == Verdict.CONDITIONAL

    # Art. 26 must be among the triggered articles when total rated > 1 MW.
    mid = compliance_gate(Scenario("Mid", Tier.MID, country=Country.ITALY), kb)
    assert any("26" in a.ref for a in mid.triggered_articles)
    # Edge (< 1 MW) must NOT trigger Art. 26.
    assert not any("26" in a.ref for a in edge.triggered_articles)


def test_required_upgrade_mapping():
    """Temperature-lift mapping: LowT/MidT -> HP, HighT -> CO2-HTHP."""
    assert required_upgrade(60.0) == UpgradeTech.HP
    assert required_upgrade(90.0) == UpgradeTech.HP
    assert required_upgrade(130.0) == UpgradeTech.CO2_HTHP


# --------------------------------------------------------------------------- #
# 3. delta_TC range and monotonicity                                           #
# --------------------------------------------------------------------------- #
def test_delta_tc_range_and_monotonicity(kb):
    """reduction_factors in [0.1,0.3]; delta_tc_norm in [0,1] and monotone increasing."""
    res = estimate_delta_tc(kb)
    rfs = res.reduction_factors()
    assert set(rfs.keys()) == {"LowT_60C", "MidT_90C", "HighT_130C"}
    for r in rfs.values():
        assert REDUCTION_MIN <= r <= REDUCTION_MAX

    dnorm = [res.burdens[t].delta_tc_norm_eu for t in Tier]
    assert all(0.0 <= d <= 1.0 for d in dnorm)
    assert dnorm == sorted(dnorm), "delta_tc_norm should be monotone increasing with tier"


# --------------------------------------------------------------------------- #
# 4. Phase-2 drop-in JSON schema                                               #
# --------------------------------------------------------------------------- #
def test_delta_tc_json_schema(kb, tmp_path):
    """The written JSON matches the step_2_4 consumer schema (reduction_factors)."""
    out = tmp_path / "delta_tc.json"
    payload = write_delta_tc_json(estimate_delta_tc(kb), path=out, kb=kb)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "reduction_factors" in data
    rf = data["reduction_factors"]
    assert set(rf) == {"LowT_60C", "MidT_90C", "HighT_130C"}
    assert all(isinstance(v, float) and REDUCTION_MIN <= v <= REDUCTION_MAX for v in rf.values())


# --------------------------------------------------------------------------- #
# 5. KB ingestion integrity                                                    #
# --------------------------------------------------------------------------- #
def test_kb_ingestion(kb):
    """The KB loads the full corpus with non-empty structured facts."""
    assert len(kb) >= 25
    assert sum(len(f.articles) for f in kb.facts) > 100
    assert sum(len(f.thresholds) for f in kb.facts) > 100
    assert kb.by_jurisdiction("Italy") and kb.by_jurisdiction("Denmark")
    assert len(list(kb.iter_chunks())) > 200


# --------------------------------------------------------------------------- #
# 6. Retrieval returns provenance-tagged hits                                  #
# --------------------------------------------------------------------------- #
def test_retrieval_returns_hits(kb):
    """Both retrievers return k provenance-tagged hits for a regulatory query."""
    nsrr = NSRRAnswerer(kb)
    base = BaselineAnswerer(kb)
    a = nsrr.answer("What installed IT power triggers EED Article 12 reporting?", prefer_unit="kW")
    assert a.grounding_docs
    assert a.value == 500.0 and a.unit == "kW"  # structural threshold extraction
    b = base.answer("What installed IT power triggers EED Article 12 reporting?", prefer_unit="kW")
    assert b.grounding_docs  # baseline retrieves but need not get the value right


# --------------------------------------------------------------------------- #
# 7. NSRR beats the vector-only baseline on a compliance query                 #
# --------------------------------------------------------------------------- #
def test_nsrr_beats_baseline_on_compliance(kb):
    """The symbolic gate yields a verdict where the vector baseline cannot."""
    q = ("A 3.2 MW data centre in Italy supplies 90 C process heat to an efficient "
         "district-heating network. Under EED 2023/1791 is this compliant, and which "
         "articles are triggered?")
    nsrr = NSRRAnswerer(kb)
    base = BaselineAnswerer(kb)
    a = nsrr.answer(q)
    b = base.answer(q)
    assert a.verdict in {"compliant", "non_compliant", "conditional"}
    assert a.method == "symbolic_gate"
    assert any("26" in art for art in a.articles)
    # The baseline returns retrieved text, not an asserted verdict.
    assert b.verdict == ""
