"""
Step 3.0 Blind - Configuration, constants and typed data model
==============================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Foundation module for the Neuro-Symbolic Regulatory Reasoner (NSRR). Defines the
scenario/verdict/fact dataclasses, the regulatory thresholds derived from the
corpus (EED 2023/1791, Delegated Regulation 2024/1364, national instruments) and
the deterministic scale/tier maps that the rule engine and the delta_TC estimator
consume.

All thresholds carry a provenance comment pointing at the corpus document and the
article that grounds them (see data/corpus_facts.json). Nothing here is invented;
the numbers are the ones extracted from the regulatory corpus.

Design rules followed (project CLAUDE.md): type hints on every signature,
dataclasses for configs, pathlib for paths, Google-style docstrings, structured
logging, no magic numbers in logic (constants live here), reproducible seeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RESULTS_DIR: Path = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_FACTS_JSON: Path = DATA_DIR / "corpus_facts.json"
BENCHMARK_JSONL: Path = DATA_DIR / "benchmark.jsonl"
DELTA_TC_JSON: Path = DATA_DIR / "step_3_4_delta_tc.json"

GLOBAL_SEED: int = 42

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Return a module logger with the project-standard format.

    Args:
        name: Logger name, usually ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# Domain enums                                                                 #
# --------------------------------------------------------------------------- #
class Tier(str, Enum):
    """Manufacturing process temperature band."""

    LOW = "LowT"
    MID = "MidT"
    HIGH = "HighT"


class Country(str, Enum):
    """Jurisdiction in scope (corpus restricted to IT and DK, plus generic EU)."""

    ITALY = "IT"
    DENMARK = "DK"
    EU = "EU"


class UpgradeTech(str, Enum):
    """Temperature-bridging technology required between DC supply and process T."""

    DIRECT_HX = "direct_HX"
    HP = "HP"
    CO2_HTHP = "CO2_HTHP"


class Verdict(str, Enum):
    """Compliance gate verdict."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    CONDITIONAL = "conditional"


# --------------------------------------------------------------------------- #
# Regulatory thresholds (corpus-grounded, see provenance comments)             #
# --------------------------------------------------------------------------- #
# EED 2023/1791 Art. 12(1) reporting threshold (F10).
ART12_REPORTING_IT_KW: float = 500.0
# EED 2023/1791 Art. 26(6)-(7) WHR obligation / CBA threshold on TOTAL rated input (F10).
ART26_WHR_TOTAL_RATED_MW: float = 1.0
# EED 2023/1791 Art. 26 DHC efficiency planning trigger on system heat output (F10).
DHC_PLANNING_MW: float = 5.0
# Danish Bilag 13 mandatory-connection-zone source threshold (F07 / comprehensive-assessment-dhc).
DK_MANDATORY_CONNECTION_MW: float = 5.0
# EED 2023/1791 Art. 11 EMS obligation threshold for enterprises (F10), all carriers.
EMS_OBLIGATION_TJ: float = 85.0
EMS_AUDIT_TJ: float = 10.0
# Italian TEE access threshold (certificati-bianchi-tee): >= 25 tep/year saving (F14/F15).
TEE_ACCESS_TEP_YR: float = 25.0
# Cooling-degree-day reference base temperature, Del. Reg. 2024/1364 (F25).
CDD_BASE_C: float = 21.0

# LC DC supply temperature reference (Phase 1 LC output, used to compute upgrade need).
LC_SUPPLY_TEMP_C: float = 47.6
T_PINCH_C: float = 5.0  # heat-exchanger pinch tolerance (Phase 4 shielding contract)

# Upgrade-technology selection by temperature lift dT = T_req - T_supply (corpus F11/F22 + scenarios).
DT_DIRECT_MAX_C: float = 5.0     # dT <= 5 C  -> direct heat exchange
DT_HP_MAX_C: float = 50.0        # 5 < dT <= 50 C -> vapor-compression heat pump
# dT > 50 C -> transcritical CO2 high-temperature heat pump

# Efficient-DHC renewable/waste-heat escalation schedule, EED Art. 26(1) (F10).
# year -> minimum renewable-or-waste-heat share (fraction).
EFFICIENT_DHC_RES_SCHEDULE: Dict[int, float] = {
    2027: 0.50,
    2035: 0.50,
    2040: 0.75,
    2045: 0.75,
    2050: 1.00,
}

# --------------------------------------------------------------------------- #
# DC scale and tier reference values (Phase 4 config/scenarios.py contract)    #
# --------------------------------------------------------------------------- #
# DC scale -> installed IT power (kW). Edge below the Art. 26 1 MW total-rated
# threshold (with PUE overhead it stays < 1 MW); Mid and Hyperscale above it.
DC_SCALE_IT_KW: Dict[str, float] = {
    "Edge": 500.0,
    "Mid": 3200.0,
    "Hyperscale": 25000.0,
}
# Liquid-cooling PUE used to convert IT power to total rated energy input.
PUE_LC: float = 1.2

# Tier -> (Phase 2 process key, required process temperature C). Matches
# config/scenarios.py (LowT 60 C, MidT 90 C, HighT 130 C) and FDM BREF bands (F24).
TIER_PROCESS_KEY: Dict[Tier, str] = {
    Tier.LOW: "LowT_60C",
    Tier.MID: "MidT_90C",
    Tier.HIGH: "HighT_130C",
}
TIER_T_REQ_C: Dict[Tier, float] = {Tier.LOW: 60.0, Tier.MID: 90.0, Tier.HIGH: 130.0}
# Typical manufacturing thermal demand per tier (kW), Phase 2 D2 medium-proximity
# representatives: agro 1200, paper/pulp 28128, rubber 2800.
TIER_DEMAND_KW: Dict[Tier, float] = {Tier.LOW: 1200.0, Tier.MID: 28127.9, Tier.HIGH: 2800.0}

# Phase 2 IS-Match weights (Eq. 3.3), needed only for the integration check.
ISMATCH_BETA: float = 0.40
ISMATCH_GAMMA: float = 0.40
ISMATCH_DELTA: float = 0.20

# delta_TC reduction factor band reserved by the framework for RAG-driven
# information-barrier reduction (Phase 2 step_2_4 contract: r in [0.1, 0.3]).
REDUCTION_MIN: float = 0.10
REDUCTION_MAX: float = 0.30


# --------------------------------------------------------------------------- #
# Typed data model                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Scenario:
    """A DC-manufacturing compliance scenario (the gate input tuple).

    Attributes:
        dc_scale: One of ``"Edge"``, ``"Mid"``, ``"Hyperscale"``.
        tier: Manufacturing temperature band.
        wh_supply_temp_c: Data-centre waste-heat supply temperature (C).
        country: Jurisdiction.
        dh_efficient: Whether a reachable district-heating network qualifies as
            ``efficient`` under EED Art. 26(1). Defaults to True (4GDH baseline).
        dh_distance_km: Distance to the nearest DH/off-taker (km).
        cba_positive: Whether the documented cost-benefit analysis is positive.
            ``None`` means "not yet assessed" (the gate then reasons conditionally).
    """

    dc_scale: str
    tier: Tier
    wh_supply_temp_c: float = LC_SUPPLY_TEMP_C
    country: Country = Country.EU
    dh_efficient: bool = True
    dh_distance_km: float = 10.0
    cba_positive: Optional[bool] = None

    @property
    def it_power_kw(self) -> float:
        """Installed IT power (kW) for the DC scale."""
        return DC_SCALE_IT_KW.get(self.dc_scale, 500.0)

    @property
    def total_rated_kw(self) -> float:
        """Total rated energy input (kW) = IT power x PUE (EED Art. 26 basis)."""
        return self.it_power_kw * PUE_LC

    @property
    def t_req_c(self) -> float:
        """Required process temperature (C) for the tier."""
        return TIER_T_REQ_C[self.tier]


@dataclass(frozen=True, slots=True)
class ArticleRef:
    """A reference to a triggered regulatory article."""

    ref: str
    doc_id: str
    requirement: str


@dataclass(slots=True)
class ComplianceResult:
    """Output of the compliance gate (deterministic, JSON-serialisable)."""

    scenario: Scenario
    status: Verdict
    triggered_articles: List[ArticleRef] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    required_upgrade: Optional[UpgradeTech] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return a plain-dict view suitable for JSON serialisation."""
        return {
            "scenario": {
                "dc_scale": self.scenario.dc_scale,
                "tier": self.scenario.tier.value,
                "wh_supply_temp_c": self.scenario.wh_supply_temp_c,
                "country": self.scenario.country.value,
                "it_power_kw": self.scenario.it_power_kw,
                "total_rated_kw": self.scenario.total_rated_kw,
            },
            "status": self.status.value,
            "triggered_articles": [
                {"ref": a.ref, "doc_id": a.doc_id, "requirement": a.requirement}
                for a in self.triggered_articles
            ],
            "thresholds": self.thresholds,
            "required_upgrade": self.required_upgrade.value if self.required_upgrade else None,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class NumericThreshold:
    """A numeric regulatory threshold extracted from a corpus document."""

    name: str
    value: str
    unit: str
    gates: str


@dataclass(slots=True)
class ComplianceTrigger:
    """An IF-THEN compliance trigger extracted from a corpus document."""

    condition: str
    consequence: str
    applies_to: str


@dataclass(slots=True)
class RegFact:
    """A typed regulatory-fact record for one corpus document.

    This is the unit of the Regulatory Knowledge Base. It is produced offline by
    the schema-constrained LLM extraction pass and loaded deterministically at
    runtime (see step_3_1_ingest).
    """

    doc_id: str
    title: str
    jurisdiction: str
    doc_type: str
    dc_whr_relevance: int
    summary: str
    articles: List[Tuple[str, str, str]] = field(default_factory=list)  # (ref, topic, requirement)
    thresholds: List[NumericThreshold] = field(default_factory=list)
    triggers: List[ComplianceTrigger] = field(default_factory=list)
    temperature_signals: str = ""
    transaction_cost_signals: str = ""
    notable_quotes: List[str] = field(default_factory=list)


def required_upgrade(t_req_c: float, t_supply_c: float = LC_SUPPLY_TEMP_C) -> UpgradeTech:
    """Return the upgrade technology mandated by the temperature lift.

    Deterministic mapping consistent with the Phase 2 Regulatory_KPIs note and the
    corpus (F11/F22): small lift -> direct exchange, moderate lift -> heat pump,
    large lift -> transcritical CO2 high-temperature heat pump.

    Args:
        t_req_c: Required process temperature (C).
        t_supply_c: DC waste-heat supply temperature (C).

    Returns:
        The required :class:`UpgradeTech`.
    """
    dt = t_req_c - t_supply_c
    if dt <= DT_DIRECT_MAX_C:
        return UpgradeTech.DIRECT_HX
    if dt <= DT_HP_MAX_C:
        return UpgradeTech.HP
    return UpgradeTech.CO2_HTHP


__all__ = [
    "Tier", "Country", "UpgradeTech", "Verdict",
    "Scenario", "ArticleRef", "ComplianceResult", "RegFact",
    "NumericThreshold", "ComplianceTrigger",
    "required_upgrade", "get_logger",
    "BASE_DIR", "DATA_DIR", "RESULTS_DIR", "CORPUS_FACTS_JSON",
    "BENCHMARK_JSONL", "DELTA_TC_JSON", "GLOBAL_SEED",
    "ART12_REPORTING_IT_KW", "ART26_WHR_TOTAL_RATED_MW", "DHC_PLANNING_MW",
    "DK_MANDATORY_CONNECTION_MW", "EMS_OBLIGATION_TJ", "TEE_ACCESS_TEP_YR",
    "EFFICIENT_DHC_RES_SCHEDULE", "DC_SCALE_IT_KW", "PUE_LC",
    "TIER_PROCESS_KEY", "TIER_T_REQ_C", "TIER_DEMAND_KW",
    "ISMATCH_BETA", "ISMATCH_GAMMA", "ISMATCH_DELTA",
    "REDUCTION_MIN", "REDUCTION_MAX", "LC_SUPPLY_TEMP_C",
]
