"""
Step 3.7 - Bootstrap CI on EM for Graph RAG and Baselines
==========================================================
Phase 3 robustness layer (additive, post-baseline)

Computes 95% bootstrap confidence intervals on the Exact Match (EM)
rate of the three benchmark configurations (graph-rag, no-rag,
llm-cypher) and on the EM gaps Graph RAG vs each baseline. Uses
PAIRED bootstrap: the same resample indices are reused across
baselines so the gap CIs are coherent (resample 100 question indices,
then read the EM outcome of each baseline at those same indices).

Inputs (read-only):
    data/evaluation_results_graph-rag_*.json   (most recent timestamp)
    data/evaluation_results_no-rag_*.json
    data/evaluation_results_llm-cypher_*.json

Output:
    results/step_3_7_bootstrap_em_ci.csv

llm-cypher entries with exact_match=None are coerced to False
(failure) so the headline EM matches the 26/100 = 26% figure already
reported in the canonical baseline numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RESULTS_DIR / "step_3_7_bootstrap_em_ci.csv"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5

BASELINES = ("graph-rag", "no-rag", "llm-cypher")
GRAPH_RAG = "graph-rag"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _latest_eval_file(config: str) -> Path:
    pattern = f"evaluation_results_{config}_*.json"
    candidates = sorted(DATA_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No evaluation results found for config={config} (pattern={pattern})")
    return candidates[-1]


def _load_em_vector(config: str) -> tuple[np.ndarray, list[str]]:
    path = _latest_eval_file(config)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    em = np.array([1 if e.get("exact_match") is True else 0 for e in data], dtype=np.int8)
    qids = [str(e.get("question_id", "")) for e in data]
    logger.info(
        "%s: loaded %d entries from %s (EM=%d/%d = %.2f%%)",
        config, len(em), path.name, int(em.sum()), len(em), 100.0 * em.mean(),
    )
    return em, qids


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _bootstrap_means(em: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return em[indices].mean(axis=1)


def main() -> None:
    logger.info(
        "Step 3.7 starting: paired bootstrap CI on EM, n_bootstrap=%d, seed=%d",
        N_BOOTSTRAP, RANDOM_SEED,
    )

    em_vectors: dict[str, np.ndarray] = {}
    qids_first: list[str] | None = None
    for cfg in BASELINES:
        em, qids = _load_em_vector(cfg)
        em_vectors[cfg] = em
        if qids_first is None:
            qids_first = qids
        elif qids != qids_first:
            logger.warning(
                "Question-id order differs between %s and %s; paired bootstrap may be biased",
                BASELINES[0], cfg,
            )

    n = len(em_vectors[GRAPH_RAG])
    if any(len(em) != n for em in em_vectors.values()):
        raise ValueError("Baselines have different number of questions; paired bootstrap aborted")

    indices = _bootstrap_indices(n, N_BOOTSTRAP, RANDOM_SEED)
    means = {cfg: _bootstrap_means(em_vectors[cfg], indices) for cfg in BASELINES}

    rows: list[dict] = []
    for cfg in BASELINES:
        m = means[cfg]
        em_obs = float(em_vectors[cfg].mean())
        row: dict = {
            "baseline": cfg,
            "em_observed": round(em_obs, 4),
            "em_bootstrap_mean": round(float(m.mean()), 4),
            "em_bootstrap_std": round(float(m.std(ddof=1)), 4),
            "ci95_lo": round(float(np.percentile(m, CI_LO_PCT)), 4),
            "ci95_hi": round(float(np.percentile(m, CI_HI_PCT)), 4),
            "n_bootstrap": N_BOOTSTRAP,
            "n_queries": n,
        }
        if cfg == GRAPH_RAG:
            row["gap_vs_no_rag_mean"] = ""
            row["gap_vs_no_rag_ci95_lo"] = ""
            row["gap_vs_no_rag_ci95_hi"] = ""
            row["gap_vs_llm_cypher_mean"] = ""
            row["gap_vs_llm_cypher_ci95_lo"] = ""
            row["gap_vs_llm_cypher_ci95_hi"] = ""
        else:
            gap = means[GRAPH_RAG] - m
            row["gap_vs_no_rag_mean"] = ""
            row["gap_vs_no_rag_ci95_lo"] = ""
            row["gap_vs_no_rag_ci95_hi"] = ""
            row["gap_vs_llm_cypher_mean"] = ""
            row["gap_vs_llm_cypher_ci95_lo"] = ""
            row["gap_vs_llm_cypher_ci95_hi"] = ""
            row[f"gap_graph_minus_{cfg.replace('-', '_')}_mean"] = round(float(gap.mean()), 4)
            row[f"gap_graph_minus_{cfg.replace('-', '_')}_ci95_lo"] = round(float(np.percentile(gap, CI_LO_PCT)), 4)
            row[f"gap_graph_minus_{cfg.replace('-', '_')}_ci95_hi"] = round(float(np.percentile(gap, CI_HI_PCT)), 4)
        rows.append(row)

    gap_no_rag = means[GRAPH_RAG] - means["no-rag"]
    gap_llm = means[GRAPH_RAG] - means["llm-cypher"]
    for row in rows:
        if row["baseline"] != GRAPH_RAG:
            continue
        row["gap_vs_no_rag_mean"] = round(float(gap_no_rag.mean()), 4)
        row["gap_vs_no_rag_ci95_lo"] = round(float(np.percentile(gap_no_rag, CI_LO_PCT)), 4)
        row["gap_vs_no_rag_ci95_hi"] = round(float(np.percentile(gap_no_rag, CI_HI_PCT)), 4)
        row["gap_vs_llm_cypher_mean"] = round(float(gap_llm.mean()), 4)
        row["gap_vs_llm_cypher_ci95_lo"] = round(float(np.percentile(gap_llm, CI_LO_PCT)), 4)
        row["gap_vs_llm_cypher_ci95_hi"] = round(float(np.percentile(gap_llm, CI_HI_PCT)), 4)

    cols = [
        "baseline", "em_observed", "em_bootstrap_mean", "em_bootstrap_std",
        "ci95_lo", "ci95_hi", "n_bootstrap", "n_queries",
        "gap_vs_no_rag_mean", "gap_vs_no_rag_ci95_lo", "gap_vs_no_rag_ci95_hi",
        "gap_vs_llm_cypher_mean", "gap_vs_llm_cypher_ci95_lo", "gap_vs_llm_cypher_ci95_hi",
    ]
    df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])
    df.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(df))

    gr = next(r for r in rows if r["baseline"] == GRAPH_RAG)
    logger.info("=" * 70)
    logger.info(
        "  EM Graph RAG: %.1f%% (95%% CI [%.1f%%, %.1f%%], n_bootstrap=%d)",
        100 * gr["em_observed"], 100 * gr["ci95_lo"], 100 * gr["ci95_hi"], N_BOOTSTRAP,
    )
    logger.info(
        "  Gap vs no-rag:    %+.1f pp (CI [%+.1f, %+.1f])",
        100 * gr["gap_vs_no_rag_mean"],
        100 * gr["gap_vs_no_rag_ci95_lo"],
        100 * gr["gap_vs_no_rag_ci95_hi"],
    )
    logger.info(
        "  Gap vs llm-cypher: %+.1f pp (CI [%+.1f, %+.1f])",
        100 * gr["gap_vs_llm_cypher_mean"],
        100 * gr["gap_vs_llm_cypher_ci95_lo"],
        100 * gr["gap_vs_llm_cypher_ci95_hi"],
    )
    sig_no_rag = gr["gap_vs_no_rag_ci95_lo"] > 0
    sig_llm = gr["gap_vs_llm_cypher_ci95_lo"] > 0
    logger.info(
        "  Statistical significance (CI95 of gap excludes 0): no-rag=%s, llm-cypher=%s",
        sig_no_rag, sig_llm,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
