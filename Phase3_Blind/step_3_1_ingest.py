"""
Step 3.1 Blind - Corpus ingestion into the Regulatory Knowledge Base
====================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Loads the offline-extracted regulatory facts (data/corpus_facts.json, produced by
the schema-constrained LLM extraction pass over the 29-document corpus) into a
typed :class:`RegulatoryKB`. This is the deterministic runtime entry point: no
language model is called here, the facts are read from disk and validated.

The KB exposes structured query helpers (by jurisdiction, by keyword, by document)
that both the compliance gate (step_3_3), the delta_TC estimator (step_3_4) and the
structured retriever (step_3_2) consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from step_3_0_config import (
    CORPUS_FACTS_JSON,
    ComplianceTrigger,
    NumericThreshold,
    RegFact,
    get_logger,
)

logger = get_logger(__name__)


def _coerce_facts(raw: object) -> List[dict]:
    """Normalise the on-disk JSON into a list of per-document fact dicts.

    The extraction artefact may be a bare list, or a task-wrapper dict with the
    list under a ``result`` (or ``logs``) key. This helper accepts both.

    Args:
        raw: Parsed JSON object.

    Returns:
        A list of per-document dicts.

    Raises:
        ValueError: If no document list can be located.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("result", "results", "docs", "facts"):
            val = raw.get(key)
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    continue
            if isinstance(val, list):
                return val
    raise ValueError("Could not locate the document list in corpus facts JSON.")


def _to_regfact(d: dict) -> RegFact:
    """Build a :class:`RegFact` from one extracted-document dict."""
    return RegFact(
        doc_id=str(d.get("doc_id", "")),
        title=str(d.get("title", "")),
        jurisdiction=str(d.get("jurisdiction", "")),
        doc_type=str(d.get("doc_type", "")),
        dc_whr_relevance=int(d.get("dc_whr_relevance", 0) or 0),
        summary=str(d.get("summary", "")),
        articles=[
            (str(a.get("ref", "")), str(a.get("topic", "")), str(a.get("requirement", "")))
            for a in d.get("key_articles", []) or []
        ],
        thresholds=[
            NumericThreshold(
                name=str(t.get("name", "")),
                value=str(t.get("value", "")),
                unit=str(t.get("unit", "")),
                gates=str(t.get("gates", "")),
            )
            for t in d.get("numeric_thresholds", []) or []
        ],
        triggers=[
            ComplianceTrigger(
                condition=str(t.get("condition", "")),
                consequence=str(t.get("consequence", "")),
                applies_to=str(t.get("applies_to", "")),
            )
            for t in d.get("compliance_triggers", []) or []
        ],
        temperature_signals=str(d.get("temperature_signals", "") or ""),
        transaction_cost_signals=str(d.get("transaction_cost_signals", "") or ""),
        notable_quotes=[str(q) for q in d.get("notable_quotes", []) or []],
    )


class RegulatoryKB:
    """In-memory typed knowledge base over the regulatory corpus.

    Attributes:
        facts: List of :class:`RegFact`, one per corpus document.
    """

    def __init__(self, facts: List[RegFact]) -> None:
        self.facts: List[RegFact] = facts
        self._by_id: Dict[str, RegFact] = {f.doc_id: f for f in facts}

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Path = CORPUS_FACTS_JSON) -> "RegulatoryKB":
        """Load and validate the KB from the extracted facts JSON.

        Args:
            path: Path to ``corpus_facts.json``.

        Returns:
            A populated :class:`RegulatoryKB`.

        Raises:
            FileNotFoundError: If the facts file is missing.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Corpus facts not found at {path}. Run the offline extraction first "
                "(step_3_1 expects data/corpus_facts.json)."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        facts = [_to_regfact(d) for d in _coerce_facts(raw)]
        logger.info("Loaded KB: %d documents, %d articles, %d thresholds, %d triggers",
                    len(facts),
                    sum(len(f.articles) for f in facts),
                    sum(len(f.thresholds) for f in facts),
                    sum(len(f.triggers) for f in facts))
        return cls(facts)

    # ------------------------------------------------------------------ #
    # Query helpers                                                       #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.facts)

    def get(self, doc_id: str) -> Optional[RegFact]:
        """Return the fact record for a document id, or None."""
        return self._by_id.get(doc_id)

    def by_jurisdiction(self, *codes: str) -> List[RegFact]:
        """Return facts whose jurisdiction string contains any of the codes.

        Args:
            *codes: Jurisdiction substrings (e.g. ``"EU"``, ``"Italy"``, ``"Denmark"``).
        """
        wanted = tuple(c.lower() for c in codes)
        return [f for f in self.facts if any(w in f.jurisdiction.lower() for w in wanted)]

    def find_thresholds(self, keyword: str) -> List[tuple[str, NumericThreshold]]:
        """Return (doc_id, threshold) pairs whose name or gate matches a keyword."""
        kw = keyword.lower()
        out: List[tuple[str, NumericThreshold]] = []
        for f in self.facts:
            for t in f.thresholds:
                if kw in t.name.lower() or kw in t.gates.lower():
                    out.append((f.doc_id, t))
        return out

    def find_triggers(self, keyword: str) -> List[tuple[str, ComplianceTrigger]]:
        """Return (doc_id, trigger) pairs whose condition/consequence matches a keyword."""
        kw = keyword.lower()
        out: List[tuple[str, ComplianceTrigger]] = []
        for f in self.facts:
            for tr in f.triggers:
                blob = f"{tr.condition} {tr.consequence}".lower()
                if kw in blob:
                    out.append((f.doc_id, tr))
        return out

    def iter_chunks(self) -> Iterable[tuple[str, str]]:
        """Yield (chunk_id, text) over the corpus for the retrieval layer.

        One chunk per article, threshold and trigger, tagged with its document id.
        This is the deterministic text view that both the structured retriever and
        the vector-only baseline index.
        """
        for f in self.facts:
            yield (f"{f.doc_id}::summary", f"{f.title}. {f.summary}")
            for i, (ref, topic, req) in enumerate(f.articles):
                yield (f"{f.doc_id}::art{i}", f"{ref} {topic}: {req}")
            for i, t in enumerate(f.thresholds):
                yield (f"{f.doc_id}::thr{i}", f"{t.name}: {t.value} {t.unit}. {t.gates}")
            for i, tr in enumerate(f.triggers):
                yield (f"{f.doc_id}::trg{i}", f"If {tr.condition} then {tr.consequence} ({tr.applies_to})")


def load_kb(path: Path = CORPUS_FACTS_JSON) -> RegulatoryKB:
    """Convenience wrapper around :meth:`RegulatoryKB.load`."""
    return RegulatoryKB.load(path)


if __name__ == "__main__":
    kb = load_kb()
    print(f"KB documents: {len(kb)}")
    print(f"EU docs: {[f.doc_id for f in kb.by_jurisdiction('EU')]}")
    print(f"Italy docs: {[f.doc_id for f in kb.by_jurisdiction('Italy')]}")
    print(f"Denmark docs: {[f.doc_id for f in kb.by_jurisdiction('Denmark')]}")
    art26 = kb.find_triggers("waste heat")
    print(f"'waste heat' triggers: {len(art26)}")
    chunks = list(kb.iter_chunks())
    print(f"Total retrievable chunks: {len(chunks)}")
