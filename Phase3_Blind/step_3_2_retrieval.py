"""
Step 3.2 Blind - Retrieval layer (structured retriever + vector-only baseline)
==============================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Two retrievers over the same corpus chunk view (RegulatoryKB.iter_chunks):

  * StructuredRetriever  : field-aware lexical scoring over the typed KB with
                           jurisdiction and article boosts. Used by the NSRR
                           answerer (step_3_5).
  * VectorBaselineIndex  : a pure-Python TF-IDF cosine index, the architecture
                           BASELINE that RQ3 requires the NSRR to beat. Has no
                           access to the typed structure: it sees flat text only.
                           An optional dense-embedding backend (sentence-
                           transformers) is used if installed, otherwise TF-IDF.

Both are deterministic (no randomness, fixed tokenisation). The vector baseline is
deliberately dependency-light so the benchmark runs offline and reproducibly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from step_3_0_config import get_logger
from step_3_1_ingest import RegulatoryKB

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "by", "on", "with", "as", "at", "that", "this", "from", "must", "shall",
    "which", "what", "when", "does", "do", "if", "it", "its", "per", "than",
}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenisation with stopword removal (deterministic)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass(slots=True)
class Retrieved:
    """A retrieved chunk with its provenance and score."""

    chunk_id: str
    doc_id: str
    text: str
    score: float


class VectorBaselineIndex:
    """Vector-only retrieval baseline (TF-IDF cosine, optional dense backend).

    This is the alternative-architecture baseline of the RQ3 evaluation: it indexes
    flat text chunks and retrieves by vector similarity with NO structural or
    relational awareness.
    """

    def __init__(self, chunks: List[Tuple[str, str]], use_dense: bool = False) -> None:
        """Build the index.

        Args:
            chunks: List of (chunk_id, text).
            use_dense: If True and sentence-transformers is installed, use dense
                embeddings; otherwise fall back to TF-IDF.
        """
        self.chunk_ids = [c[0] for c in chunks]
        self.texts = [c[1] for c in chunks]
        self._dense_model = None
        self._dense_mat = None
        if use_dense:
            self._try_build_dense()
        if self._dense_model is None:
            self._build_tfidf()

    # -- dense (optional) -- #
    def _try_build_dense(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np  # noqa: F401

            self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
            mat = self._dense_model.encode(self.texts, normalize_embeddings=True)
            self._dense_mat = mat
            logger.info("VectorBaselineIndex: using dense embeddings (all-MiniLM-L6-v2)")
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.info("Dense backend unavailable (%s); falling back to TF-IDF.", type(exc).__name__)
            self._dense_model = None

    # -- TF-IDF (default) -- #
    def _build_tfidf(self) -> None:
        self._docs_tokens = [tokenize(t) for t in self.texts]
        df: Counter = Counter()
        for toks in self._docs_tokens:
            for tok in set(toks):
                df[tok] += 1
        n = max(1, len(self._docs_tokens))
        self._idf: Dict[str, float] = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self._vecs: List[Dict[str, float]] = [self._tfidf_vec(toks) for toks in self._docs_tokens]
        self._norms: List[float] = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in self._vecs]

    def _tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = max(1, len(tokens))
        return {t: (c / total) * self._idf.get(t, math.log(2.0)) for t, c in tf.items()}

    def query(self, text: str, k: int = 5) -> List[Retrieved]:
        """Return the top-k chunks by similarity to the query (deterministic)."""
        if self._dense_model is not None:  # pragma: no cover - environment dependent
            import numpy as np

            q = self._dense_model.encode([text], normalize_embeddings=True)[0]
            sims = self._dense_mat @ q
            order = list(np.argsort(-sims)[:k])
            return [Retrieved(self.chunk_ids[i], self.chunk_ids[i].split("::")[0], self.texts[i], float(sims[i]))
                    for i in order]
        qvec = self._tfidf_vec(tokenize(text))
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored: List[Tuple[float, int]] = []
        for i, vec in enumerate(self._vecs):
            dot = sum(qvec.get(t, 0.0) * w for t, w in vec.items())
            scored.append((dot / (qnorm * self._norms[i]), i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Retrieved(self.chunk_ids[i], self.chunk_ids[i].split("::")[0], self.texts[i], round(s, 4))
                for s, i in scored[:k]]


class StructuredRetriever:
    """Field-aware lexical retriever over the typed KB.

    Scores chunks by query-token overlap weighted by IDF, then applies structural
    boosts when the query mentions a jurisdiction (Italy/Denmark/EU) or a numeric
    intent (threshold/temperature/percentage). This is the NSRR retrieval branch.
    """

    JURIS_HINTS = {
        "ital": "italy", "tee": "italy", "certificat": "italy", "white certificate": "italy",
        "denmar": "denmark", "danish": "denmark", "bilag": "denmark",
        "eed": "eu", "directive": "eu", "european": "eu",
    }

    def __init__(self, kb: RegulatoryKB) -> None:
        self.kb = kb
        self.chunks: List[Tuple[str, str]] = list(kb.iter_chunks())
        self._index = VectorBaselineIndex(self.chunks, use_dense=False)

    def query(self, text: str, k: int = 5) -> List[Retrieved]:
        """Return top-k chunks with structural boosts (deterministic)."""
        base = self._index.query(text, k=max(k * 4, 12))
        wanted_juris = self._infer_jurisdiction(text)
        numeric_intent = bool(re.search(r"thresh|temperatur|kw|mw|percent|%|how much|minimum|maximum|degree", text.lower()))
        rescored: List[Retrieved] = []
        for r in base:
            score = r.score
            fact = self.kb.get(r.doc_id)
            if fact is not None and wanted_juris and wanted_juris in fact.jurisdiction.lower():
                score += 0.25
            if numeric_intent and "::thr" in r.chunk_id:
                score += 0.15
            rescored.append(Retrieved(r.chunk_id, r.doc_id, r.text, round(score, 4)))
        rescored.sort(key=lambda x: -x.score)
        return rescored[:k]

    def _infer_jurisdiction(self, text: str) -> Optional[str]:
        low = text.lower()
        for hint, juris in self.JURIS_HINTS.items():
            if hint in low:
                return juris
        return None


if __name__ == "__main__":
    from step_3_1_ingest import load_kb

    kb = load_kb()
    sr = StructuredRetriever(kb)
    vb = VectorBaselineIndex(list(kb.iter_chunks()))
    q = "What IT power threshold triggers EED data centre reporting obligations?"
    print("QUERY:", q)
    print("\n-- Structured retriever --")
    for r in sr.query(q, k=3):
        print(f"  [{r.doc_id}] {r.score:.3f} {r.text[:90]}")
    print("\n-- Vector-only baseline --")
    for r in vb.query(q, k=3):
        print(f"  [{r.doc_id}] {r.score:.3f} {r.text[:90]}")
