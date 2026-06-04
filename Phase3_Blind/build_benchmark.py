"""
build_benchmark.py - Deterministic generator for the RQ3 regulatory benchmark
=============================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Produces data/benchmark.jsonl with >= 100 questions in four categories, each with
a clause-grounded gold answer. The benchmark is document-grounded: the gold for a
threshold question IS the value extracted from the corpus clause (data/corpus_facts.json),
so the ground truth is objective and reproducible, not a subjective annotation.

Categories:
  threshold_lookup  : factual numeric thresholds (scoring="numeric")
  compliance_verdict: EED Art. 12/26 verdicts for concrete scenarios (scoring="verdict")
  multi_hop         : chained reasoning, gold = required token set (scoring="multi_token")
  comparative       : jurisdiction/technology comparisons (scoring="keyword")

The compliance gold verdicts are authored independently from the regulatory text
(EED Art. 12/26 + concepts/eed-art-dc-waste-heat), NOT by calling the gate, so the
benchmark is a genuine test of the gate rather than a tautology.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from step_3_0_config import BENCHMARK_JSONL, get_logger
from step_3_1_ingest import load_kb

logger = get_logger(__name__)

_NUM_ONLY = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*$")
_UNIT_KEEP = {"kw", "mw", "%", "tj", "tep", "km", "g/kwh", "years", "year", "kwh", "mwh"}


def _clean_value(value: str) -> float | None:
    """Return a single float if the threshold value is unambiguous, else None."""
    m = _NUM_ONLY.match(value.replace(",", "."))
    if m:
        return float(m.group(1))
    return None


def _canon_unit(u: str) -> str:
    u = u.lower().strip().strip("°")
    return {"c": "C", "degrees celsius": "C", "degree celsius": "C", "%": "%",
            "kw": "kW", "mw": "MW", "tj": "TJ", "tep": "tep", "km": "km",
            "g/kwh": "g/kWh", "years": "yr", "year": "yr", "kwh": "kWh", "mwh": "MWh"}.get(u, u)


def build_threshold_questions(kb) -> List[Dict]:
    """Generate factual threshold questions from clean corpus numeric thresholds."""
    out: List[Dict] = []
    seen: set = set()
    qid = 0
    for fact in sorted(kb.facts, key=lambda f: (-f.dc_whr_relevance, f.doc_id)):
        if fact.dc_whr_relevance < 3:
            continue
        for thr in fact.thresholds:
            val = _clean_value(thr.value)
            unit = _canon_unit(thr.unit)
            if val is None or unit.lower() not in _UNIT_KEEP and unit not in {"C", "kW", "MW", "%", "TJ", "tep", "g/kWh", "yr", "km"}:
                continue
            key = (round(val, 3), unit, fact.doc_id, thr.name[:40])
            if key in seen:
                continue
            seen.add(key)
            qid += 1
            out.append({
                "id": f"thr_{qid:03d}",
                "category": "threshold_lookup",
                "question": f"According to the regulatory corpus, what is the value of: {thr.name}?",
                "gold_answer": f"{val:g} {unit}".strip(),
                "gold_value": val,
                "gold_unit": unit,
                "grounding_doc": fact.doc_id,
                "scoring": "numeric",
                "clause": f"{fact.doc_id}: {thr.name} = {thr.value} {thr.unit} ({thr.gates[:80]})",
            })
            if len(out) >= 55:
                return out
    return out


def build_compliance_questions() -> List[Dict]:
    """Hand-authored compliance scenarios with independent gold verdicts (EED Art. 12/26)."""
    # (scale, tier_c, country, dh_efficient, cba, gold_verdict, note)
    scenarios = [
        ("Edge (500 kW IT)", 60, "Italy", True, None, "compliant", "below 1 MW, reporting-only"),
        ("Edge (500 kW IT)", 90, "EU", True, None, "compliant", "below 1 MW Art. 26 threshold"),
        ("Edge (500 kW IT)", 130, "Denmark", True, None, "compliant", "below 1 MW threshold"),
        ("Mid (3.2 MW IT)", 60, "Italy", True, None, "compliant", "Art. 26 + efficient DH + TEE"),
        ("Mid (3.2 MW IT)", 90, "Denmark", True, None, "compliant", "Art. 26 + efficient DH + planning"),
        ("Mid (3.2 MW IT)", 90, "EU", True, None, "compliant", "Art. 26 + efficient DH reachable"),
        ("Mid (3.2 MW IT)", 90, "EU", False, None, "non_compliant", "non-efficient DH does not discharge Art. 26"),
        ("Mid (3.2 MW IT)", 130, "EU", True, False, "conditional", "documented negative CBA -> Art. 26(8) exemption"),
        ("Hyperscale (25 MW IT)", 60, "Italy", True, None, "compliant", "Art. 26 + efficient DH + TEE"),
        ("Hyperscale (25 MW IT)", 90, "Denmark", True, None, "compliant", "Art. 26 + DK mandatory connection"),
        ("Hyperscale (25 MW IT)", 130, "Italy", True, None, "compliant", "Art. 26 + CO2-HTHP feasible + TEE"),
        ("Hyperscale (25 MW IT)", 130, "EU", True, False, "conditional", "negative CBA exemption"),
        ("Hyperscale (25 MW IT)", 130, "EU", False, None, "non_compliant", "non-efficient DH, no exemption"),
        ("Mid (3.2 MW IT)", 130, "Italy", True, None, "compliant", "Art. 26 + TEE + CO2-HTHP"),
        ("Hyperscale (25 MW IT)", 90, "EU", False, False, "conditional", "negative CBA exemption despite non-efficient DH"),
    ]
    out: List[Dict] = []
    for i, (scale, t, ctry, dh_eff, cba, gold, note) in enumerate(scenarios, 1):
        eff = "an efficient" if dh_eff else "a non-efficient"
        cba_s = ""
        if cba is False:
            cba_s = " The operator has documented a negative cost-benefit analysis."
        q = (f"A {scale} data centre in {ctry} supplies {t} C process heat to {eff} district-heating "
             f"network.{cba_s} Under EED 2023/1791, is this arrangement compliant, non-compliant or "
             f"conditional, and which articles are triggered?")
        out.append({
            "id": f"cmp_{i:03d}", "category": "compliance_verdict", "question": q,
            "gold_answer": gold, "gold_verdict": gold, "grounding_doc": "F10",
            "scoring": "verdict",
            "clause": f"EED Art. 12 (>=500 kW reporting) + Art. 26 (>1 MW WHR obligation/exemption); {note}",
        })
    return out


def build_multihop_questions() -> List[Dict]:
    """Hand-authored multi-hop questions; gold = required token set."""
    # gold_tokens are conjunctive (ALL must match); a token may list synonyms with "|".
    items = [
        ("A data centre supplies waste heat at 47 C to a process requiring 90 C in Italy. "
         "Which upgrade technology is required and does the EED Article 26 obligation apply to a 3.2 MW data centre?",
         ["hp|heat pump", "article 26"], "F10"),
        ("A 25 MW data centre must supply a 130 C industrial steam process. Which upgrade technology is mandated "
         "and what EED obligation is triggered?",
         ["co2|hthp", "article 26"], "F10"),
        ("If a 3.2 MW data centre documents a negative cost-benefit analysis, what is its position under EED Article 26?",
         ["exempt|exemption|conditional", "report|reporting"], "F10"),
        ("What temperature lift technology bridges a 47 C supply to a 60 C low-temperature process?",
         ["heat pump|hp"], "F11"),
        ("In Denmark, a 25 MW data centre in a designated heat-planning zone supplies an efficient network. "
         "What connection consequence follows and which instrument applies?",
         ["mandatory", "connection"], "F26"),
        ("Which Italian instrument can make the Article 26 cost-benefit analysis positive for a waste-heat project?",
         ["tee|white certificate|certificati"], "F14"),
        ("A district-heating network must reach what minimum renewable-or-waste-heat share by 2027 to be 'efficient' under the EED?",
         ["50"], "F10"),
        ("An enterprise consuming more than 85 TJ per year must implement what, per the EED?",
         ["energy management|ems|50001"], "F10"),
        ("What KPI does the Delegated Regulation 2024/1364 require a data centre to report regarding reused heat?",
         ["energy reuse|waste heat|reuse"], "F25"),
        ("A 500 kW edge data centre: which EED reporting article applies?",
         ["article 12"], "F10"),
        ("For a process at 90 C served by a 47 C supply, the temperature gap is about 43 C: heat pump or CO2-HTHP?",
         ["heat pump|hp"], "F11"),
        ("For a 130 C process from a 47 C supply (gap ~83 C), which upgrade is needed?",
         ["co2|hthp"], "F22"),
        ("Above what district-heating system size does the EED efficiency-planning requirement apply from 2025?",
         ["5"], "F10"),
        ("A 3.2 MW data centre in Italy at 90 C: name the reporting article, the obligation article and the incentive.",
         ["12", "26", "tee"], "F10"),
        ("What documented analysis exempts a >1 MW data centre from the waste-heat utilisation requirement?",
         ["cost-benefit|cba"], "F10"),
        ("Which standard governs the energy management system that large EU enterprises must certify?",
         ["50001"], "F08"),
        ("What is the waste-heat reuse measurement standard referenced by the Delegated Regulation 2024/1364?",
         ["50600"], "F25"),
        ("A hyperscale data centre supplying a non-efficient district-heating network without a CBA exemption is in what compliance state?",
         ["non-compliant|non_compliant|not compliant"], "F10"),
        ("Which two jurisdictions in the corpus represent the incentive-based and the planning-based regulatory models?",
         ["italy", "denmark"], "F26"),
        ("What is the EED reporting threshold in installed IT power and from which date does it apply?",
         ["500", "2024"], "F10"),
    ]
    out: List[Dict] = []
    for i, (q, tokens, doc) in enumerate(items, 1):
        out.append({
            "id": f"mh_{i:03d}", "category": "multi_hop", "question": q,
            "gold_answer": " / ".join(tokens), "gold_tokens": tokens,
            "grounding_doc": doc, "scoring": "multi_token",
            "clause": f"multi-hop chain grounded in {doc}",
        })
    return out


def build_comparative_questions() -> List[Dict]:
    """Hand-authored comparative questions; gold = required keyword(s)."""
    items = [
        ("Does Italy regulate data-centre waste heat primarily through incentives or through municipal planning?",
         ["incentive"], "F14"),
        ("Does Denmark regulate district heating primarily through incentives or through municipal planning and zoning?",
         ["planning"], "F26"),
        ("Which upgrade technology has the higher CAPEX, a vapour-compression heat pump or a CO2-HTHP?",
         ["co2"], "F11"),
        ("Is air-cooled or liquid-cooled data-centre waste heat more suitable for district-heating reuse?",
         ["liquid"], "F01"),
        ("Which EED article sets the data-centre reporting threshold, Article 12 or Article 26?",
         ["12"], "F10"),
        ("Which EED article sets the waste-heat utilisation obligation, Article 12 or Article 26?",
         ["26"], "F10"),
        ("Between a 500 kW and a 3.2 MW data centre, which one triggers the Article 26 obligation?",
         ["3.2", "mid", "1 mw", "exceed"], "F10"),
        ("For a 60 C versus a 130 C process, which requires the more complex upgrade technology?",
         ["130", "high"], "F11"),
        ("Is the Stockholm open-district-heating model a planning mandate or a market-based trading platform?",
         ["market", "trading"], "F04"),
        ("Which has the larger temperature lift requirement from a 47 C supply, a 90 C or a 130 C process?",
         ["130"], "F22"),
        ("Between reporting under Article 12 and the WHR obligation under Article 26, which applies to ALL data centres above 500 kW?",
         ["12", "report"], "F10"),
        ("Which jurisdiction uses White Certificates (TEE) as its efficiency instrument?",
         ["italy"], "F14"),
    ]
    out: List[Dict] = []
    for i, (q, tokens, doc) in enumerate(items, 1):
        out.append({
            "id": f"cmpv_{i:03d}", "category": "comparative", "question": q,
            "gold_answer": " / ".join(tokens), "gold_tokens": tokens,
            "grounding_doc": doc, "scoring": "keyword",
            "clause": f"comparative judgement grounded in {doc}",
        })
    return out


def build_benchmark() -> List[Dict]:
    """Assemble the full benchmark."""
    kb = load_kb()
    bench = (
        build_threshold_questions(kb)
        + build_compliance_questions()
        + build_multihop_questions()
        + build_comparative_questions()
    )
    return bench


def main() -> None:
    bench = build_benchmark()
    with BENCHMARK_JSONL.open("w", encoding="utf-8") as fh:
        for item in bench:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    cats: Dict[str, int] = {}
    for item in bench:
        cats[item["category"]] = cats.get(item["category"], 0) + 1
    logger.info("Wrote %d benchmark questions -> %s", len(bench), BENCHMARK_JSONL)
    for c, n in sorted(cats.items()):
        logger.info("  %-20s %d", c, n)


if __name__ == "__main__":
    main()
