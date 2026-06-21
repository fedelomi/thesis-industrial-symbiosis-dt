"""
curate_ood_benchmark.py
=======================
Phase 3 - OOD benchmark curation gate (FREE, no API, no Neo4j).

Filters the blindly generated candidates (generate_ood_benchmark.py) down to a
clean held-out set before any paid ground-truth or evaluation call:

  1. Schema gate          : valid JSON and required fields present.
  2. Phrasing-similarity  : bge cosine of each candidate against all 100
                            canonical questions; drop > THRESHOLD (default 0.85)
                            so the OOD set is not a paraphrase of the in-
                            distribution benchmark.
  3. Structure-leakage    : drop candidates that reference internal structure
                            (a template id, or a Cypher/schema token). Domain
                            terms such as "Denmark" or "4GDH" are NOT leakage;
                            only references to the system internals are.

Inputs:
    data/benchmarks/ood_candidates_raw.json
    data/benchmark_qa_dataset.json        (canonical v2, for similarity)
Outputs:
    data/benchmarks/ood_candidates_curated.json
    data/benchmarks/ood_curation_report.json

Usage:
    python curate_ood_benchmark.py
    python curate_ood_benchmark.py --threshold 0.85

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from templates import CYPHER_TEMPLATES

BENCH_DIR = BASE_DIR / "data" / "benchmarks"
RAW = BENCH_DIR / "ood_candidates_raw.json"
CANONICAL = BASE_DIR / "data" / "benchmark_qa_dataset.json"
OUT_CURATED = BENCH_DIR / "ood_candidates_curated.json"
OUT_REPORT = BENCH_DIR / "ood_curation_report.json"

SIM_THRESHOLD = 0.85
REQUIRED_FIELDS = ("id", "category", "difficulty", "nl_question", "rationale")

# Tokens that betray knowledge of the system internals (not the domain).
STRUCTURE_TOKENS = [
    "cypher", "template", "match (", "return ", "optional match", "->",
    "neo4j", "node label", "relationship type", "schema",
    "temp_supply_c", "it_capacity_kw", "range_min_c", "waste_heat_kw",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _max_similarity(cand_texts: list[str], ref_texts: list[str]) -> np.ndarray:
    """Max cosine of each candidate against the reference set."""
    try:
        from semantic_router import SentenceTransformerBackend
        b = SentenceTransformerBackend()
        b.prepare([])
        cand = b.encode(cand_texts)
        ref = b.encode(ref_texts)
        backend_name = b.name
    except Exception as exc:
        logger.warning("bge unavailable (%s); TF-IDF cosine fallback.", exc)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        mat = vec.fit_transform(ref_texts + cand_texts)
        ref = mat[: len(ref_texts)].toarray().astype(np.float32)
        cand = mat[len(ref_texts):].toarray().astype(np.float32)
        ref /= np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9, None)
        cand /= np.clip(np.linalg.norm(cand, axis=1, keepdims=True), 1e-9, None)
        backend_name = "tfidf-fallback"
    sims = cand @ ref.T
    logger.info("similarity backend used: %s", backend_name)
    return sims.max(axis=1)


def _structure_leak(text: str) -> list[str]:
    """Return the internal-structure tokens found in the text (case-insensitive)."""
    low = text.lower()
    hits = [tok for tok in STRUCTURE_TOKENS if tok in low]
    hits += [tid for tid in CYPHER_TEMPLATES if tid.lower() in low]
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Free OOD curation gate")
    parser.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()

    if not args.raw.exists():
        logger.error("Candidates not found: %s (run generate_ood_benchmark.py)", args.raw)
        return
    candidates = json.loads(args.raw.read_text(encoding="utf-8"))
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    ref_texts = [q["nl_question"] for q in canonical]

    # 1. schema gate
    schema_ok, schema_rej = [], []
    for c in candidates:
        missing = [f for f in REQUIRED_FIELDS if not str(c.get(f, "")).strip()]
        (schema_rej if missing else schema_ok).append(
            {"id": c.get("id", "?"), "missing": missing} if missing else c
        )

    # 2. phrasing similarity (only on schema-valid candidates)
    cand_texts = [c["nl_question"] for c in schema_ok]
    max_sims = _max_similarity(cand_texts, ref_texts) if cand_texts else np.array([])

    kept, dropped = [], []
    for c, sim in zip(schema_ok, max_sims):
        leak = _structure_leak(c["nl_question"])
        sim = float(sim)
        if sim > args.threshold:
            dropped.append({**c, "drop_reason": "phrasing_similarity", "max_sim": round(sim, 3)})
        elif leak:
            dropped.append({**c, "drop_reason": "structure_leak", "leak_tokens": leak})
        else:
            kept.append({**c, "max_sim_to_canonical": round(sim, 3)})

    # re-id the curated set sequentially
    for i, c in enumerate(kept, start=1):
        c["id"] = f"OOD{i:02d}"

    report = {
        "n_candidates": len(candidates),
        "n_schema_rejected": len(schema_rej),
        "schema_rejected": schema_rej,
        "n_phrasing_dropped": sum(1 for d in dropped if d["drop_reason"] == "phrasing_similarity"),
        "n_leak_dropped": sum(1 for d in dropped if d["drop_reason"] == "structure_leak"),
        "n_kept": len(kept),
        "threshold": args.threshold,
        "dropped": dropped,
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CURATED.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("candidates=%d schema_rej=%d phrasing_drop=%d leak_drop=%d -> kept=%d",
                len(candidates), len(schema_rej), report["n_phrasing_dropped"],
                report["n_leak_dropped"], len(kept))
    if max_sims.size:
        logger.info("similarity to canonical: max=%.3f mean=%.3f",
                    float(max_sims.max()), float(max_sims.mean()))
    by_cat = {}
    for c in kept:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    logger.info("kept category spread: %s", by_cat)
    print(f"\nCurated {len(kept)} OOD queries -> {OUT_CURATED.name}. "
          f"Next: python build_ood_ground_truth.py")


if __name__ == "__main__":
    main()
