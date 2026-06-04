"""
Step 3.3 Blind - Deterministic compliance gate (symbolic rule engine)
=====================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

The compliance gate is a PURE, DETERMINISTIC function from a scenario tuple to a
verdict. No language model is called at runtime: the verdict is produced by
forward-chaining over the regulatory rules grounded in the Knowledge Base, so the
same scenario always yields the same verdict (the framework's reproducibility
requirement for the gate).

The rule set encodes the EED 2023/1791 decision logic (Art. 12 reporting, Art. 26
WHR obligation and cost-benefit exemption, Art. 26(1) efficient-DHC criteria), the
Delegated Regulation 2024/1364 reporting, the temperature-upgrade feasibility, and
the Italian (TEE) and Danish (Heat Supply Act / Bilag 13) national overlays. Each
fired rule attaches the triggering article and threshold for auditability.
"""

from __future__ import annotations

from typing import List, Optional

from step_3_0_config import (
    ART12_REPORTING_IT_KW,
    ART26_WHR_TOTAL_RATED_MW,
    DK_MANDATORY_CONNECTION_MW,
    TEE_ACCESS_TEP_YR,
    ArticleRef,
    ComplianceResult,
    Country,
    Scenario,
    UpgradeTech,
    Verdict,
    get_logger,
    required_upgrade,
)
from step_3_1_ingest import RegulatoryKB, load_kb

logger = get_logger(__name__)

# Spatial feasibility for DH connection (comprehensive-assessment-dhc: 10-15 km DH main).
DH_SPATIAL_KM: float = 15.0
# Maximum temperature reachable by the highest upgrade tech (CO2-HTHP industrial steam, F11/F22).
MAX_REACHABLE_TEMP_C: float = 150.0
# Operating hours and gas-displacement tep factor for TEE eligibility (certificati-bianchi-tee).
OPERATING_HOURS_YR: float = 7000.0
TEP_PER_MWH: float = 0.086  # gas primary-energy tep per MWh saved (F14/concept page)


def _article(kb: Optional[RegulatoryKB], doc_id: str, ref_prefix: str, fallback: str) -> ArticleRef:
    """Build an :class:`ArticleRef`, pulling the requirement text from the KB if present.

    Args:
        kb: The regulatory KB (or None to use the fallback text).
        doc_id: Source document id (e.g. ``"F10"``).
        ref_prefix: Article reference prefix to match (e.g. ``"Art. 26"``).
        fallback: Requirement text to use if no KB match is found.

    Returns:
        An :class:`ArticleRef` grounded in the corpus where possible.
    """
    if kb is not None:
        fact = kb.get(doc_id)
        if fact is not None:
            for ref, _topic, req in fact.articles:
                if ref.replace(" ", "").lower().startswith(ref_prefix.replace(" ", "").lower()):
                    return ArticleRef(ref=ref, doc_id=doc_id, requirement=req[:240])
    return ArticleRef(ref=ref_prefix, doc_id=doc_id, requirement=fallback)


def _tee_eligible(scenario: Scenario) -> bool:
    """Return True if the Italian White-Certificate access threshold is met.

    Eligibility requires a primary-energy saving of at least 25 tep/year. We size
    the saving from the manufacturing demand met by the recovered heat.
    """
    from step_3_0_config import TIER_DEMAND_KW

    annual_mwh = min(TIER_DEMAND_KW[scenario.tier], 30000.0) * OPERATING_HOURS_YR / 1000.0
    tep = annual_mwh * TEP_PER_MWH
    return tep >= TEE_ACCESS_TEP_YR


def compliance_gate(scenario: Scenario, kb: Optional[RegulatoryKB] = None) -> ComplianceResult:
    """Evaluate the regulatory compliance of an IS heat-supply scenario.

    Deterministic: the same scenario always returns the same verdict.

    Args:
        scenario: The DC-manufacturing compliance scenario.
        kb: Optional Regulatory KB used to attach grounded article text. The
            verdict itself does not depend on the KB (only the provenance text does),
            so the function is well-defined even when ``kb`` is None.

    Returns:
        A :class:`ComplianceResult` with status, triggered articles, thresholds,
        required upgrade technology and a human-readable rationale.
    """
    articles: List[ArticleRef] = []
    thresholds: dict[str, float] = {}
    reasons: List[str] = []

    upgrade = required_upgrade(scenario.t_req_c, scenario.wh_supply_temp_c)
    upgrade_feasible = scenario.t_req_c <= MAX_REACHABLE_TEMP_C

    # --- Rule 1: EED Art. 12 reporting (IT power >= 500 kW) -------------------- #
    art12 = scenario.it_power_kw >= ART12_REPORTING_IT_KW
    if art12:
        thresholds["art12_reporting_it_kw"] = ART12_REPORTING_IT_KW
        articles.append(_article(
            kb, "F10", "Art. 12",
            "Annual public reporting of Annex VII KPIs for data centres with IT power >= 500 kW."))
        articles.append(_article(
            kb, "F25", "Art. 3",
            "Communicate KPIs (incl. waste-heat reused and average waste-heat temperature) to the EU database."))
        reasons.append("IT power >= 500 kW triggers EED Art. 12 reporting and Del. Reg. 2024/1364 KPI disclosure.")

    # --- Rule 2: EED Art. 26 WHR obligation (total rated > 1 MW) --------------- #
    art26 = scenario.total_rated_kw > ART26_WHR_TOTAL_RATED_MW * 1000.0
    if not art26:
        thresholds["art26_total_rated_mw"] = ART26_WHR_TOTAL_RATED_MW
        status = Verdict.COMPLIANT
        reasons.append(
            f"Total rated input {scenario.total_rated_kw/1000:.2f} MW is below the 1 MW "
            "Art. 26 threshold; waste-heat reuse is voluntary, reporting-only obligation applies.")
        return ComplianceResult(
            scenario=scenario, status=status, triggered_articles=articles,
            thresholds=thresholds, required_upgrade=upgrade, rationale=" ".join(reasons))

    # Art. 26 is triggered.
    thresholds["art26_total_rated_mw"] = ART26_WHR_TOTAL_RATED_MW
    articles.append(_article(
        kb, "F10", "Art. 26",
        "Data centres with total rated input > 1 MW must utilise waste heat or document a negative CBA."))
    reasons.append(
        f"Total rated input {scenario.total_rated_kw/1000:.2f} MW exceeds 1 MW; EED Art. 26 WHR "
        f"obligation triggered. Required upgrade to reach {scenario.t_req_c:.0f} C: {upgrade.value}.")

    # --- National overlays (incentive / planning) ----------------------------- #
    incentive_support = False
    if scenario.country == Country.ITALY and _tee_eligible(scenario):
        incentive_support = True
        thresholds["tee_access_tep_yr"] = TEE_ACCESS_TEP_YR
        articles.append(_article(
            kb, "F14", "Art.",
            "Italian White Certificates (TEE): WHR projects >= 25 tep/yr eligible for tradable incentives, improving the CBA."))
        reasons.append("Italian TEE incentive available (>= 25 tep/yr), strengthening a positive cost-benefit position.")

    dk_mandatory = False
    if scenario.country == Country.DENMARK:
        # Source heat output proxied by total rated input; Bilag 13 mandatory zones for sources > 5 MW.
        if scenario.total_rated_kw > DK_MANDATORY_CONNECTION_MW * 1000.0:
            dk_mandatory = True
            incentive_support = True  # municipal planning support reduces search/feasibility friction
            thresholds["dk_mandatory_connection_mw"] = DK_MANDATORY_CONNECTION_MW
            articles.append(_article(
                kb, "F26", "",
                "Danish Heat Supply Act / Bilag 13: sources > 5 MW in designated zones face mandatory connection and cost-of-service pricing."))
            reasons.append("Danish municipal heat planning designates mandatory-connection zones for sources > 5 MW.")

    # --- Effective cost-benefit position -------------------------------------- #
    if scenario.cba_positive is None:
        effective_cba = (
            scenario.dh_efficient
            and scenario.dh_distance_km <= DH_SPATIAL_KM
            and upgrade_feasible
        ) or incentive_support
    else:
        effective_cba = scenario.cba_positive

    # --- Verdict decision tree ------------------------------------------------- #
    if not upgrade_feasible:
        status = Verdict.CONDITIONAL
        reasons.append(
            f"Process temperature {scenario.t_req_c:.0f} C exceeds the reachable bound "
            f"{MAX_REACHABLE_TEMP_C:.0f} C; a documented negative-CBA exemption (Art. 26(8)) is the compliant path.")
    elif not scenario.dh_efficient and scenario.cba_positive is not False:
        # Art. 26 discharges only via an EFFICIENT DH network (Art. 26(1) criteria).
        status = Verdict.NON_COMPLIANT
        thresholds["efficient_dhc_res_min"] = 0.50
        articles.append(_article(
            kb, "F10", "Art. 26(1)",
            "Efficient DHC requires >= 50% renewable or waste heat (escalating to 100% by 2050)."))
        reasons.append(
            "Target DH network does not meet the efficient-DHC criteria, so supplying it does not "
            "discharge the Art. 26 obligation and no negative-CBA exemption is documented.")
    elif effective_cba and scenario.dh_distance_km <= DH_SPATIAL_KM:
        status = Verdict.COMPLIANT
        if dk_mandatory:
            reasons.append("Mandatory connection applies and the network is efficient: arrangement is compliant.")
        else:
            reasons.append("Efficient DH reachable with a positive cost-benefit position: Art. 26 obligation satisfiable, compliant.")
    elif scenario.cba_positive is False:
        status = Verdict.CONDITIONAL
        articles.append(_article(
            kb, "F10", "Art. 26",
            "Exemption from the WHR requirement where a documented CBA shows technical/economic infeasibility (Art. 26(8))."))
        reasons.append("Documented negative CBA grants the Art. 26(8) exemption; reporting under Art. 12 still required (conditional).")
    else:
        status = Verdict.CONDITIONAL
        reasons.append(
            "Art. 26 obligation active but the efficient-DH off-take is not reachable within the spatial "
            "threshold or the CBA is not yet positive; the DC must document a CBA or explore alternative WHR applications.")

    # Deduplicate triggered articles (preserve first-seen order) for a clean audit trail.
    seen: set[tuple[str, str]] = set()
    unique_articles: List[ArticleRef] = []
    for a in articles:
        key = (a.doc_id, a.ref)
        if key not in seen:
            seen.add(key)
            unique_articles.append(a)

    return ComplianceResult(
        scenario=scenario, status=status, triggered_articles=unique_articles,
        thresholds=thresholds, required_upgrade=upgrade, rationale=" ".join(reasons))


if __name__ == "__main__":
    kb = load_kb()
    from step_3_0_config import Tier

    demos = [
        Scenario("Edge", Tier.LOW, country=Country.ITALY),
        Scenario("Mid", Tier.MID, country=Country.DENMARK),
        Scenario("Hyperscale", Tier.HIGH, country=Country.ITALY),
        Scenario("Mid", Tier.MID, country=Country.EU, dh_efficient=False),
        Scenario("Hyperscale", Tier.HIGH, country=Country.EU, cba_positive=False),
    ]
    for s in demos:
        r = compliance_gate(s, kb)
        print(f"\n{s.dc_scale:11} {s.tier.value:6} {s.country.value}  ->  {r.status.value.upper()}")
        print(f"   upgrade={r.required_upgrade.value}  articles={[a.ref for a in r.triggered_articles]}")
        print(f"   {r.rationale}")
