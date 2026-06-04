"""
Step 3.5 Blind - Neuro-symbolic answerer and vector-only baseline answerer
==========================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Two answerers used by the RQ3 benchmark (step_3_6):

  * NSRRAnswerer       : routes a question to the deterministic compliance gate
                         (compliance / multi-hop questions) or to structured
                         retrieval with symbolic value extraction (threshold /
                         factual lookups). Answers are grounded in the typed KB.
  * BaselineAnswerer   : vector-only RAG. Retrieves top-k flat chunks and returns
                         an extractive answer with a regex value extraction. No
                         structure, no gate, no relational reasoning.

Both are deterministic. Each returns an :class:`AnswerResult` carrying the answer
text, a normalised answer key for scoring, the grounding document ids and the
routing method used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from step_3_0_config import Country, Scenario, Tier, get_logger
from step_3_1_ingest import RegulatoryKB
from step_3_2_retrieval import StructuredRetriever, VectorBaselineIndex
from step_3_3_compliance_gate import compliance_gate

logger = get_logger(__name__)

_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kw|mw|gw|°?c|degrees?|%|percent|tep|tj|g/kwh|gco2|years?|km|kwh|mwh)?", re.I)
_UNIT_CANON = {
    "kw": "kW", "mw": "MW", "gw": "GW", "c": "C", "°c": "C", "degree": "C", "degrees": "C",
    "%": "%", "percent": "%", "tep": "tep", "tj": "TJ", "g/kwh": "g/kWh", "gco2": "g/kWh",
    "year": "yr", "years": "yr", "km": "km", "kwh": "kWh", "mwh": "MWh",
}


def _canon_unit(u: Optional[str]) -> str:
    if not u:
        return ""
    return _UNIT_CANON.get(u.lower().strip("°"), u.lower())


def extract_value(text: str, prefer_unit: str = "") -> Tuple[Optional[float], str]:
    """Extract the most salient (value, unit) from a text span (deterministic).

    Args:
        text: Source text.
        prefer_unit: If given, prefer a number carrying this canonical unit.

    Returns:
        A (value, canonical_unit) tuple; value is None if nothing parses.
    """
    cands: List[Tuple[float, str]] = []
    for m in _NUM_RE.finditer(text):
        try:
            val = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        cands.append((val, _canon_unit(m.group(2))))
    if not cands:
        return None, ""
    if prefer_unit:
        for val, unit in cands:
            if unit == prefer_unit:
                return val, unit
    return cands[0]


@dataclass(slots=True)
class AnswerResult:
    """Output of an answerer for one question."""

    answer_text: str
    answer_key: str
    grounding_docs: List[str] = field(default_factory=list)
    method: str = ""
    value: Optional[float] = None
    unit: str = ""
    verdict: str = ""
    articles: List[str] = field(default_factory=list)


_COMPLIANCE_HINTS = ("compliant", "compliance", "obligation", "article 26", "art. 26",
                     "must connect", "exemption", "triggered", "obliged", "required to")
_SCALE_WORDS = {"edge": "Edge", "small": "Edge", "mid": "Mid", "medium": "Mid",
                "hyperscale": "Hyperscale", "large": "Hyperscale"}


def parse_scenario(question: str) -> Optional[Scenario]:
    """Heuristically parse a compliance scenario from a question (deterministic).

    Returns None if the question does not describe a concrete DC-manufacturing case.
    """
    low = question.lower()
    # DC scale or IT power.
    scale: Optional[str] = None
    for word, label in _SCALE_WORDS.items():
        if word in low:
            scale = label
            break
    m_mw = re.search(r"(\d+(?:\.\d+)?)\s*mw", low)
    m_kw = re.search(r"(\d+(?:\.\d+)?)\s*kw", low)
    if scale is None and (m_mw or m_kw):
        kw = float(m_mw.group(1)) * 1000 if m_mw else float(m_kw.group(1))
        scale = "Edge" if kw < 1000 else ("Mid" if kw < 10000 else "Hyperscale")
    # Process temperature / tier.
    tier: Optional[Tier] = None
    m_t = re.search(r"(\d{2,3})\s*°?\s*c\b", low)
    if m_t:
        t = float(m_t.group(1))
        tier = Tier.LOW if t <= 70 else (Tier.MID if t <= 110 else Tier.HIGH)
    elif "lowt" in low or "low-temperature" in low:
        tier = Tier.LOW
    elif "midt" in low or "medium-temperature" in low:
        tier = Tier.MID
    elif "hight" in low or "high-temperature" in low or "steam" in low:
        tier = Tier.HIGH
    if scale is None or tier is None:
        return None
    country = Country.EU
    if "ital" in low or "tee" in low or "white certificate" in low:
        country = Country.ITALY
    elif "denmar" in low or "danish" in low:
        country = Country.DENMARK
    cba = None
    if "negative cost-benefit" in low or "negative cba" in low or "cba is negative" in low:
        cba = False
    dh_eff = False if ("not efficient" in low or "non-efficient" in low or "inefficient" in low) else True
    return Scenario(dc_scale=scale, tier=tier, country=country, cba_positive=cba, dh_efficient=dh_eff)


class NSRRAnswerer:
    """Neuro-symbolic answerer: gate for compliance, structured retrieval for lookups.

    Args:
        kb: Regulatory knowledge base.
        use_gate: If False, the symbolic compliance gate is disabled and every
            question is answered by retrieval. Used by the leave-one-out ablation
            (step_3_7) to isolate the symbolic gate's contribution.
    """

    def __init__(self, kb: RegulatoryKB, use_gate: bool = True) -> None:
        self.kb = kb
        self.use_gate = use_gate
        self.retriever = StructuredRetriever(kb)

    def answer(self, question: str, prefer_unit: str = "") -> AnswerResult:
        low = question.lower()
        scenario = parse_scenario(question)
        is_compliance = self.use_gate and any(h in low for h in _COMPLIANCE_HINTS) and scenario is not None
        if is_compliance:
            res = compliance_gate(scenario, self.kb)
            arts = [a.ref for a in res.triggered_articles]
            return AnswerResult(
                answer_text=f"{res.status.value}: {res.rationale}",
                answer_key=res.status.value,
                grounding_docs=sorted({a.doc_id for a in res.triggered_articles}),
                method="symbolic_gate", verdict=res.status.value, articles=arts,
            )
        # Retrieval-grounded branch.
        if not prefer_unit:
            prefer_unit = self._infer_unit(low)
        hits = self.retriever.query(question, k=8)
        context = " | ".join(h.text for h in hits[:3])
        numeric_intent = bool(prefer_unit)
        if not numeric_intent:
            # Conceptual / comparative question: ground the answer in the top-3
            # structured hits (jurisdiction-boosted retrieval), not a forced number.
            return AnswerResult(
                answer_text=context,
                answer_key=(hits[0].text[:80] if hits else ""),
                grounding_docs=[h.doc_id for h in hits[:3]],
                method="structured_retrieval",
            )
        # Numeric lookup: exploit the typed threshold chunks: among the top hits,
        # prefer a threshold-typed chunk whose value carries the expected unit. This
        # structural routing is unavailable to the flat-text baseline.
        chosen = None
        chosen_val: Optional[float] = None
        chosen_unit = ""
        for h in hits:
            v, u = extract_value(h.text, prefer_unit=prefer_unit)
            if v is None:
                continue
            is_thr = "::thr" in h.chunk_id
            unit_ok = bool(prefer_unit) and u == prefer_unit
            # Priority: threshold chunk with the right unit > right unit > threshold chunk.
            if unit_ok and is_thr:
                chosen, chosen_val, chosen_unit = h, v, u
                break
            if chosen is None and (unit_ok or is_thr):
                chosen, chosen_val, chosen_unit = h, v, u
        if chosen is None and hits:
            chosen = hits[0]
            chosen_val, chosen_unit = extract_value(chosen.text, prefer_unit=prefer_unit)
        key = f"{chosen_val:g} {chosen_unit}".strip() if chosen_val is not None else (chosen.text[:60] if chosen else "")
        # answer_text carries the top-3 grounding context so multi-hop token scoring
        # sees the full retrieved evidence; value/unit drive the numeric scoring.
        answer_text = ((chosen.text + " | " if chosen else "") + context)
        return AnswerResult(
            answer_text=answer_text,
            answer_key=key,
            grounding_docs=[h.doc_id for h in hits[:3]],
            method="structured_retrieval", value=chosen_val, unit=chosen_unit,
        )

    @staticmethod
    def _infer_unit(low: str) -> str:
        """Infer the expected answer unit from question keywords (deterministic)."""
        if "it power" in low or " kw" in low or "kilowatt" in low:
            return "kW"
        if " mw" in low or "megawatt" in low or "rated input" in low or "rated energy" in low:
            return "MW"
        if "temperature" in low or "°c" in low or "degree" in low or "celsius" in low:
            return "C"
        if "percent" in low or "share" in low or "%" in low or "renewable energy %" in low:
            return "%"
        if "tep" in low:
            return "tep"
        if "tj" in low or "terajoule" in low:
            return "TJ"
        if "g/kwh" in low or "emission" in low:
            return "g/kWh"
        return ""


class BaselineAnswerer:
    """Vector-only RAG baseline answerer (no structure, no gate)."""

    def __init__(self, kb: RegulatoryKB, use_dense: bool = False) -> None:
        self.index = VectorBaselineIndex(list(kb.iter_chunks()), use_dense=use_dense)

    def answer(self, question: str, prefer_unit: str = "") -> AnswerResult:
        hits = self.index.query(question, k=5)
        top = hits[0] if hits else None
        val, unit = (None, "")
        if top is not None:
            val, unit = extract_value(top.text, prefer_unit=prefer_unit)
        key = f"{val:g} {unit}".strip() if val is not None else (top.text[:60] if top else "")
        # Standard RAG grounds its answer on the top-k retrieved context.
        context = " | ".join(h.text for h in hits[:3])
        return AnswerResult(
            answer_text=context,
            answer_key=key,
            grounding_docs=[h.doc_id for h in hits[:3]],
            method="vector_baseline", value=val, unit=unit,
        )


if __name__ == "__main__":
    from step_3_1_ingest import load_kb

    kb = load_kb()
    nsrr = NSRRAnswerer(kb)
    base = BaselineAnswerer(kb)
    qs = [
        ("What installed IT power triggers EED Article 12 reporting?", "kW"),
        ("Is a 3.2 MW data centre supplying 90 C process heat in Italy subject to the Article 26 obligation?", ""),
        ("Above what total rated input does the EED Article 26 waste-heat obligation apply?", "MW"),
    ]
    for q, u in qs:
        print("\nQ:", q)
        a = nsrr.answer(q, prefer_unit=u)
        b = base.answer(q, prefer_unit=u)
        print(f"  NSRR     [{a.method}] key='{a.answer_key}' docs={a.grounding_docs}")
        print(f"  Baseline [{b.method}] key='{b.answer_key}' docs={b.grounding_docs}")
