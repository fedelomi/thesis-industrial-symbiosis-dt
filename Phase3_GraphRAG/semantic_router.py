"""
semantic_router.py
==================
Phase 3 - Semantic fallback router for the Graph RAG Cypher templates.

Purpose
-------
The primary router in `step_3_4_evaluation.py` is a deterministic keyword
matcher (`KEYWORD_ROUTING`, first match wins). It is fast and precise on
in-distribution phrasings, but greedy lexical buckets collide on
out-of-distribution (OOD) questions: e.g. "which policy framework governs the
Danish 4GDH network" matches both the `4gdh` bucket (thermal params) and the
`danish` bucket (regulatory screening). Historically these collisions were
patched with hand-written priority rules, one phrasing at a time.

This module replaces that patch-by-patch strategy with a *mechanism*: when the
keyword stage is ambiguous (zero matches, or two or more distinct templates),
the question is embedded and compared by cosine similarity against a small,
hand-authored set of intent anchors (one description plus a few paraphrases per
template). The closest template wins, subject to a confidence and margin gate.

Anti-leakage note
-----------------
The anchor texts are authored from each template's data *purpose*, paraphrased
independently. They are deliberately NOT copied from the benchmark questions in
`data/benchmark_qa_dataset.json`. Seeding the router with the evaluation
questions themselves would make routing accuracy circular and contaminate the
canonical thesis numbers.

Backends (pluggable)
--------------------
  - TfidfBackend            : scikit-learn TF-IDF cosine. Zero extra
                              dependencies, fully offline, sub-millisecond.
                              Lexical, not semantic.
  - SentenceTransformerBackend : local SentenceTransformer (default model
                              BAAI/bge-small-en-v1.5). Stronger paraphrase /
                              intent generalisation. Requires a one-time model
                              download (no API cost), then runs offline.

Public API
----------
    from semantic_router import semantic_route
    semantic_route("which agency enforces the heat supply act")
    # -> [("GENERIC_ACTOR", 0.41), ("DK_NECP_TARGET", 0.22), ...]

Riferimento wiki: [[graph-rag-entity-schema]], [[phase-1-2-3-roadmap]] Passo 3.2.
Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from templates import CYPHER_TEMPLATES

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data" / "semantic_router_cache"

# Default backend selection and embedding model. Overridable via env so the
# A/B test in step_3_4_evaluation can switch backends without code edits.
DEFAULT_BACKEND = os.getenv("SEMANTIC_ROUTER_BACKEND", "tfidf").lower()
DEFAULT_ST_MODEL = os.getenv("SEMANTIC_ROUTER_MODEL", "BAAI/bge-small-en-v1.5")

# Top-k returned by semantic_route by default.
DEFAULT_TOP_K = 3

# Fixed seed for the dense embedding backend so query encodings are bit-identical
# across runs (FW9b Task 1). Single-threaded encode pins the float reduction order.
EMBED_SEED = 42


# ---------------------------------------------------------------------------
# Intent anchors: template_id -> list of short intent strings.
# First string is the description; the rest are independent paraphrases.
# Authored from template PURPOSE, NOT from the benchmark questions.
# The deprecated alias DK_DHNETWORK_PARAMS is intentionally omitted so the
# semantic stage never routes to it (DK_DH_COMPARE supersedes it).
# ---------------------------------------------------------------------------
TEMPLATE_ANCHORS: dict[str, list[str]] = {
    "P1_eed_art26_threshold": [
        "the EED Article 26 mandatory waste heat disclosure obligation and its "
        "power threshold for data centers",
        "minimum megawatt threshold above which a data center must disclose waste heat",
        "is Article 26 disclosure mandatory or voluntary for large data centers",
    ],
    "P2_thermal_compatibility_S1": [
        "the thermal compatibility result for scenario S1 specifically",
        "details of scenario S1 alone, named S1, not any other scenario",
    ],
    "P2_thermal_compatibility_all": [
        "thermal compatibility detail for each of the nine symbiosis scenarios across "
        "every data center scale: target process, sector, supply versus required "
        "temperature and the resulting thermal gap",
        "the three scenarios of one data center scale, for example the mid-size or "
        "the edge data center, listed across the low, medium and high temperature bands",
        "which upgrade technology minimises the thermal gap for a process in a "
        "temperature band",
        "per-scenario lookup of process, sector or supply temperature for any scenario S1 to S9",
    ],
    "P3_regulatory_screening_dk": [
        "the Danish policy framework or legal act that governs district heating and "
        "the regulatory obligations of data centers",
        "which legal framework governs the Danish district heating network",
        "which data centers above one megawatt must connect to district heating in Denmark",
        "screening of mandatory Danish heat supply obligations for large data centers",
    ],
    "P4_incentives_it_whr": [
        "the monetary value in euro per MWh or per toe of the Italian white "
        "certificate incentive for waste heat recovery",
        "which technologies are eligible for the Italian TEE or CB white certificate incentive",
        "whether waste heat recovery or a heat pump can claim the Italian incentive",
    ],
    "P5_scenario_comparison_L": [
        "which of the hyperscale DC-L scenarios S7 S8 or S9 offers the highest "
        "available heat capacity or supply temperature",
        "ranking the largest hyperscale scenarios by available heat",
        "which combination of the hyperscale data center and upgrade technology "
        "maximises both capacity and supply temperature",
    ],
    "P6_full_is_path": [
        "the complete industrial symbiosis pathway from data center to heat source "
        "to temperature band to manufacturing process to industrial sector",
        "trace the full symbiosis path for a scenario from data center to sector",
        "how many scenarios target processes in a given industrial sector",
    ],
    "ALL_REGULATORY_ARTICLES": [
        "list every EU regulatory article with its summary, including EED Article 23 "
        "and Article 24 and the Delegated Regulation KPI requirements",
        "what does EED Article 23 or Article 24 require, or which KPIs the Delegated "
        "Regulation defines for data center waste heat",
        "how many regulatory articles are defined under the Energy Efficiency Directive",
    ],
    "ISO50001_ARTICLES": [
        "ISO 50001 energy management standard clauses, energy review, significant "
        "energy use and energy performance indicators",
        "what the ISO 50001 energy management system requires in its clauses",
    ],
    "ISO50001_PROCESS_COMPLIANCE": [
        "which manufacturing processes must comply with the ISO 50001 standard",
        "manufacturing processes linked to ISO 50001 energy management compliance",
    ],
    "GENERIC_REGULATION": [
        "general lookup of a regulation, directive, law or decree and its articles",
        "the publication year, jurisdiction or name of a regulation or directive",
        "details and jurisdiction of a regulatory instrument in the knowledge graph",
    ],
    "GENERIC_COUNTRY": [
        "country level facts such as district heating penetration rate and whether "
        "the EED has been transposed",
        "how many countries are in the knowledge graph or whether a country transposed the EED",
        "district heating penetration percentage for Italy or Denmark",
    ],
    "GENERIC_STANDARD": [
        "lookup of a technical standard such as ASHRAE 90.4 or ISO 23247: scope, "
        "issuing organisation and publication year",
        "which ASHRAE standard, for example ASHRAE 90.4 or TC 9.9, applies to data "
        "center energy efficiency",
        "the ISO standard number that defines a digital twin framework for "
        "manufacturing, ISO 23247",
        "what does a standard cover, who publishes it and in which year it was issued",
    ],
    "GENERIC_DC": [
        "specifications of a data center archetype: IT capacity in kilowatts, cooling "
        "type, nominal PUE and recoverable waste heat capacity",
        "recoverable waste heat capacity in kilowatts of the edge, mid-size or "
        "hyperscale archetype",
        "compare the recoverable waste heat capacity between the edge and hyperscale archetypes",
    ],
    "TEMPERATURE_BAND_DEF": [
        "the numeric Celsius range that defines each temperature band T1 T2 and T3, "
        "their lower and upper bounds",
        "what temperature range or numeric bounds define a low, medium or high grade heat band",
        "how many temperature bands are defined and what are their numeric ranges",
    ],
    "DK_4GDH_PARAMS": [
        "the supply temperature, return temperature and capacity in megawatts of the "
        "Danish 4th generation 4GDH district heating network",
        "the operating temperatures of the 4GDH network",
    ],
    "DK_3GDH_PARAMS": [
        "the supply temperature, return temperature and capacity in megawatts of the "
        "Danish 3rd generation legacy 3GDH district heating network",
        "the operating temperatures of the legacy 3GDH network",
    ],
    "DK_DH_COMPARE": [
        "side by side comparison of the Danish 3GDH and 4GDH district heating "
        "networks by capacity and supply temperature",
        "compare the capacity or temperatures of the two Danish district heating generations",
    ],
    "GENERIC_ACTOR": [
        "the administrative or enforcement body, agency or authority responsible for "
        "an energy programme",
        "the GSE is the Italian public body that issues, manages and administers the "
        "white certificate TEE and CB scheme",
        "the Danish Energy Agency is the authority that enforces the Heat Supply Act",
        "which public body, agency or authority issues, manages, administers or "
        "enforces an energy scheme or law",
    ],
    "DK_NECP_TARGET": [
        "the Danish national energy and climate plan greenhouse gas reduction target "
        "and its horizon year",
        "Denmark's emission reduction target year under the NECP",
    ],
    "IT_DK_SUPPORT_COMPARE": [
        "side by side comparison of support instruments for industrial symbiosis in "
        "Italy versus Denmark: financial incentive versus regulatory mandate",
        "which jurisdiction offers stronger or better support for symbiosis projects, "
        "Italy or Denmark",
        "compare the regulatory burden or transaction cost reduction between Italy and Denmark",
    ],
    "SCENARIO_COUNT_BY_SECTOR": [
        "count of how many industrial symbiosis scenarios involve a given industrial sector",
        "number of scenarios targeting the pulp and paper or the food sector",
    ],
}


# ---------------------------------------------------------------------------
# Backend protocol and implementations
# ---------------------------------------------------------------------------
@runtime_checkable
class SimilarityBackend(Protocol):
    """A pluggable text-to-vector backend producing L2-normalised rows."""

    name: str
    default_conf_threshold: float
    default_margin: float
    use_disk_cache: bool

    def prepare(self, corpus: list[str]) -> None:
        """Fit or load any state needed before encoding (e.g. TF-IDF vocab)."""
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalised row vectors."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left as zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class TfidfBackend:
    """Lexical TF-IDF cosine backend (scikit-learn). No extra dependency."""

    def __init__(self) -> None:
        self.name = "tfidf"
        # Sparse cosine values are small; thresholds tuned on the gate.
        self.default_conf_threshold = float(os.getenv("SEMANTIC_TFIDF_CONF", "0.08"))
        self.default_margin = float(os.getenv("SEMANTIC_TFIDF_MARGIN", "0.0"))
        # Re-fitting on ~60 short strings is microseconds; no disk cache needed.
        self.use_disk_cache = False
        self._vectorizer = None

    def prepare(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            stop_words="english",
            norm="l2",
        )
        self._vectorizer.fit(corpus)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TfidfBackend.prepare must be called before encode.")
        mat = self._vectorizer.transform(list(texts)).toarray().astype(np.float32)
        return _l2_normalize(mat)


class SentenceTransformerBackend:
    """Local SentenceTransformer backend (default BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = DEFAULT_ST_MODEL) -> None:
        self.name = f"st::{model_name}"
        self.model_name = model_name
        # Dense cosine on a retrieval-tuned model sits higher; tuned on the gate.
        self.default_conf_threshold = float(os.getenv("SEMANTIC_ST_CONF", "0.45"))
        # Override margin: the semantic top must beat the deterministic prior by
        # at least this cosine gap to override it. 0.05 keeps the targeted fixes
        # (B17/C23 clear their prior by ~0.05) and the high-confidence
        # improvements, while leaving small-delta near-ties on their v4 route so
        # the router cannot regress them. Tuned on check_routing + check_routing_diff.
        self.default_margin = float(os.getenv("SEMANTIC_ST_MARGIN", "0.05"))
        self.use_disk_cache = True
        self._model = None

    def prepare(self, corpus: list[str]) -> None:
        # Model load is deferred to first encode so a cache hit on the anchor
        # matrix can skip the (heavier) anchor encoding entirely.
        del corpus

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "'pip install sentence-transformers' or set "
                "SEMANTIC_ROUTER_BACKEND=tfidf."
            ) from exc
        logger.info("Loading SentenceTransformer model %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        # Pin determinism: fixed seed and single-threaded reduction so the same
        # query encodes to bit-identical vectors across runs (FW9b Task 1).
        try:
            import torch
            torch.manual_seed(EMBED_SEED)
            torch.set_num_threads(1)
        except ImportError:
            pass

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        emb = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


def build_backend(name: str) -> SimilarityBackend:
    """Factory: 'tfidf' or 'st'/'sentence-transformer' (optionally 'st::<model>')."""
    key = name.lower()
    if key == "tfidf":
        return TfidfBackend()
    if key in ("st", "sentence-transformer", "sentence-transformers"):
        return SentenceTransformerBackend()
    if key.startswith("st::"):
        return SentenceTransformerBackend(model_name=name.split("::", 1)[1])
    raise ValueError(f"Unknown semantic router backend: {name!r}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class SemanticRouter:
    """Cosine-similarity router over per-template intent anchors."""

    def __init__(
        self,
        backend: Optional[SimilarityBackend] = None,
        anchors: Optional[dict[str, list[str]]] = None,
        cache_dir: Path = CACHE_DIR,
        conf_threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> None:
        self.backend = backend if backend is not None else build_backend(DEFAULT_BACKEND)
        self.anchors = anchors if anchors is not None else TEMPLATE_ANCHORS
        self.cache_dir = cache_dir
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None
            else self.backend.default_conf_threshold
        )
        self.margin = margin if margin is not None else self.backend.default_margin

        unknown = set(self.anchors) - set(CYPHER_TEMPLATES)
        if unknown:
            raise ValueError(f"Anchors reference unknown templates: {sorted(unknown)}")

        # Flatten anchors into parallel arrays.
        self._anchor_texts: list[str] = []
        self._anchor_tids: list[str] = []
        for tid, texts in self.anchors.items():
            for text in texts:
                self._anchor_texts.append(text)
                self._anchor_tids.append(tid)
        self._tids_array = np.array(self._anchor_tids)
        self._template_ids = list(self.anchors.keys())

        self.backend.prepare(self._anchor_texts)
        self._anchor_matrix = self._load_or_build_matrix()

    # -- caching ----------------------------------------------------------
    def _cache_key(self) -> str:
        payload = self.backend.name + "|" + "".join(self._anchor_texts)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self) -> Path:
        safe = self.backend.name.replace("::", "_").replace("/", "_")
        return self.cache_dir / f"anchors_{safe}_{self._cache_key()}.npz"

    def _load_or_build_matrix(self) -> np.ndarray:
        if self.backend.use_disk_cache:
            path = self._cache_path()
            if path.exists():
                try:
                    data = np.load(path)
                    logger.info("Loaded anchor embeddings from cache %s", path.name)
                    return data["matrix"]
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning("Anchor cache unreadable (%s); rebuilding.", exc)
            matrix = self.backend.encode(self._anchor_texts)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(path, matrix=matrix)
                logger.info("Cached anchor embeddings to %s", path.name)
            except OSError as exc:
                logger.warning("Could not write anchor cache: %s", exc)
            return matrix
        return self.backend.encode(self._anchor_texts)

    # -- routing ----------------------------------------------------------
    def score(self, query: str, candidates: Optional[Sequence[str]] = None) -> list[tuple[str, float]]:
        """Return (template_id, confidence) sorted descending.

        Confidence of a template is the max cosine similarity over its anchor
        rows. If `candidates` is given, only those templates are scored.
        """
        q_vec = self.backend.encode([query])[0]
        sims = self._anchor_matrix @ q_vec  # rows are L2-normalised -> cosine

        allowed = set(candidates) if candidates else None
        per_template: dict[str, float] = {}
        for tid, sim in zip(self._anchor_tids, sims):
            if allowed is not None and tid not in allowed:
                continue
            sim_f = float(sim)
            if tid not in per_template or sim_f > per_template[tid]:
                per_template[tid] = sim_f
        # Deterministic order: score descending, template_id ascending on ties.
        # The explicit tie-break removes any dependence on dict insertion order or
        # on float reduction noise flipping near-ties across runs (FW9b Task 1).
        return sorted(per_template.items(), key=lambda kv: (-kv[1], kv[0]))

    def route(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidates: Optional[Sequence[str]] = None,
    ) -> list[tuple[str, float]]:
        """Top-k (template_id, confidence) by cosine similarity."""
        return self.score(query, candidates=candidates)[:top_k]

    def best(
        self,
        query: str,
        candidates: Optional[Sequence[str]] = None,
    ) -> Optional[tuple[str, float]]:
        """Return the top template only if it clears the confidence + margin gate.

        Returns None when the router is not confident enough, so the caller can
        fall back to the deterministic keyword result (no-regression guarantee).
        """
        ranked = self.score(query, candidates=candidates)
        if not ranked:
            return None
        top_id, top_conf = ranked[0]
        if top_conf < self.conf_threshold:
            return None
        if len(ranked) > 1 and (top_conf - ranked[1][1]) < self.margin:
            return None
        return top_id, top_conf


# ---------------------------------------------------------------------------
# Module-level singleton + convenience API
# ---------------------------------------------------------------------------
_DEFAULT_ROUTER: Optional[SemanticRouter] = None


def get_default_router() -> SemanticRouter:
    """Lazily build and cache the process-wide default router."""
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = SemanticRouter()
    return _DEFAULT_ROUTER


def semantic_route(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidates: Optional[Sequence[str]] = None,
) -> list[tuple[str, float]]:
    """Convenience wrapper around the default router's top-k ranking."""
    return get_default_router().route(query, top_k=top_k, candidates=candidates)


def semantic_best(
    query: str,
    candidates: Optional[Sequence[str]] = None,
) -> Optional[tuple[str, float]]:
    """Convenience wrapper: gated single best template or None."""
    return get_default_router().best(query, candidates=candidates)


if __name__ == "__main__":
    # Smoke test: print top-3 routes for a few OOD-style probes.
    logging.basicConfig(level=logging.INFO)
    router = get_default_router()
    print(f"backend={router.backend.name}  "
          f"conf>={router.conf_threshold}  margin>={router.margin}")
    probes = [
        "what policy framework governs the danish 4gdh network",
        "compare the is scenarios for the mid-size dc across all three temperature bands",
        "which upgrade technology minimizes the thermal gap for t2 band processes",
        "which agency enforces the heat supply act",
    ]
    for p in probes:
        print(f"\nQ: {p}")
        for tid, conf in router.route(p):
            print(f"   {conf:6.3f}  {tid}")
