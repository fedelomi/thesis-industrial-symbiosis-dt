"""Step 3.13 - Deterministic BCa CI + Holm + Cohen regenerator (CRITICAL #1b, Phase 3).

Phase 3 robustness layer (additive, post-audit 2026-06-05).

The thesis (Table 5.5) cites a BCa 95% confidence interval on the EM gap of
Graph-RAG over the no-RAG and llm-cypher baselines, but the committed
results/step_3_7_bootstrap_em_ci_bca.csv had no producing script and
step_3_7 itself only computes a percentile interval (and globs the newest
evaluation file, which now points at a post-submission 0.66 run). This module
makes the BCa intervals regenerable from committed code:

- it PINS the three canonical 20260517 evaluation files (graph-rag 0.59,
  no-rag 0.28, llm-cypher 0.26), never glob-latest;
- it reuses the exact paired-bootstrap machinery of step_3_7 (np.random
  default_rng(42), 10000 resamples, same index draw) so the bootstrap means
  reproduce the canonical 0.3108 / 0.3308 gaps;
- it adds the bias-correction (z0) and jackknife acceleration (a) of the BCa
  method on top of that distribution;
- it adds the Holm-Bonferroni step-down across the two gap comparisons and
  Cohen's h on the proportions.

No paid LLM call is made: the EM verdicts are read from the frozen evaluation
JSON files committed in data/.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OUT_BCA = RESULTS_DIR / "step_3_7_bootstrap_em_ci_bca.csv"
OUT_HOLM_COHEN = RESULTS_DIR / "step_3_13_holm_cohen.csv"

# Canonical frozen evaluation files (thesis Table 5.5 / 5.6), pinned by exact
# name so the regeneration is immune to the glob-latest drift documented in the
# 2026-06-05 code audit (the 20260602 runs at 0.63 / 0.66 are post-submission).
CANONICAL_EVAL = {
    "graph-rag": DATA_DIR / "evaluation_results_graph-rag_20260517_155731.json",
    "no-rag": DATA_DIR / "evaluation_results_no-rag_20260517_100514.json",
    "llm-cypher": DATA_DIR / "evaluation_results_llm-cypher_20260517_101352.json",
}
GRAPH_RAG = "graph-rag"
BASELINES = ("graph-rag", "no-rag", "llm-cypher")

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42
ALPHA = 0.05  # 95% CI

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_em_vector(config: str) -> np.ndarray:
    """Load the binary EM vector for a config from its pinned canonical file.

    Args:
        config: One of graph-rag, no-rag, llm-cypher.

    Returns:
        Int8 array of 0/1 exact-match outcomes, one per benchmark question.

    Raises:
        FileNotFoundError: If the pinned canonical evaluation file is missing.
    """
    path = CANONICAL_EVAL[config]
    if not path.is_file():
        raise FileNotFoundError(f"Canonical evaluation file missing for {config}: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return np.array([1 if e.get("exact_match") is True else 0 for e in data], dtype=np.int8)


def _bca_interval(
    theta_hat: float,
    boot: np.ndarray,
    jackknife: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) percentile interval.

    Args:
        theta_hat: Observed statistic on the full sample.
        boot: Bootstrap replicates of the statistic.
        jackknife: Leave-one-out replicates of the statistic.
        alpha: Two-sided significance level (0.05 gives a 95% interval).

    Returns:
        The (low, high) BCa interval endpoints.
    """
    n_below = float(np.sum(boot < theta_hat))
    prop = n_below / boot.size
    # Guard the degenerate all-on-one-side cases so z0 stays finite.
    prop = min(max(prop, 1.0 / boot.size), 1.0 - 1.0 / boot.size)
    z0 = stats.norm.ppf(prop)

    jack_mean = jackknife.mean()
    diff = jack_mean - jackknife
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    accel = float(np.sum(diff ** 3) / denom) if denom != 0.0 else 0.0

    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    lo = float(np.percentile(boot, 100.0 * a1))
    hi = float(np.percentile(boot, 100.0 * a2))
    return lo, hi


def _jackknife_proportion(em: np.ndarray) -> np.ndarray:
    """Leave-one-out proportions of a binary vector."""
    total = em.sum()
    n = em.size
    return (total - em) / (n - 1)


def _jackknife_gap(em_a: np.ndarray, em_b: np.ndarray) -> np.ndarray:
    """Leave-one-out paired gap (mean(a) - mean(b)) over the same dropped index."""
    n = em_a.size
    sum_a, sum_b = em_a.sum(), em_b.sum()
    return (sum_a - em_a) / (n - 1) - (sum_b - em_b) / (n - 1)


def _one_sided_gap_pvalue(gap_boot: np.ndarray) -> float:
    """One-sided bootstrap p-value for H0: gap <= 0, floored at 1/n_bootstrap."""
    frac_le_zero = float(np.mean(gap_boot <= 0.0))
    return max(frac_le_zero, 1.0 / gap_boot.size)


def _holm_bonferroni(pvalues: dict[str, float]) -> dict[str, dict[str, float]]:
    """Holm-Bonferroni step-down correction over a family of raw p-values.

    Args:
        pvalues: Mapping of test name to raw p-value.

    Returns:
        Mapping of test name to {p_raw, p_holm} with the step-down adjusted value.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float]] = {}
    running_max = 0.0
    for rank, (name, p_raw) in enumerate(items):
        p_adj = min((m - rank) * p_raw, 1.0)
        running_max = max(running_max, p_adj)  # enforce monotone non-decreasing
        out[name] = {"p_raw": float(p_raw), "p_holm": float(running_max)}
    return out


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2)))


def regenerate() -> dict:
    """Recompute the canonical BCa intervals, Holm correction and Cohen's h.

    Returns:
        Nested mapping with per-baseline observed EM, bootstrap mean, BCa CI and
        (for graph-rag) the paired gap statistics, plus top-level holm and
        cohens_h sub-mappings.
    """
    em = {cfg: _load_em_vector(cfg) for cfg in BASELINES}
    n = em[GRAPH_RAG].size
    if any(v.size != n for v in em.values()):
        raise ValueError("Canonical evaluation files have inconsistent question counts")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot_mean = {cfg: em[cfg][indices].mean(axis=1) for cfg in BASELINES}

    res: dict = {}
    for cfg in BASELINES:
        theta = float(em[cfg].mean())
        lo, hi = _bca_interval(theta, boot_mean[cfg], _jackknife_proportion(em[cfg]))
        res[cfg] = {
            "em_observed": round(theta, 4),
            "em_bootstrap_mean": round(float(boot_mean[cfg].mean()), 4),
            "em_bootstrap_std": round(float(boot_mean[cfg].std(ddof=1)), 4),
            "ci95_lo_bca": lo,
            "ci95_hi_bca": hi,
        }

    p_raw: dict[str, float] = {}
    for cfg, key in (("no-rag", "no_rag"), ("llm-cypher", "llm_cypher")):
        gap_boot = boot_mean[GRAPH_RAG] - boot_mean[cfg]
        gap_hat = float(em[GRAPH_RAG].mean() - em[cfg].mean())
        glo, ghi = _bca_interval(gap_hat, gap_boot, _jackknife_gap(em[GRAPH_RAG], em[cfg]))
        res[GRAPH_RAG][f"gap_vs_{key}_mean"] = round(float(gap_boot.mean()), 4)
        res[GRAPH_RAG][f"gap_vs_{key}_ci95_lo_bca"] = glo
        res[GRAPH_RAG][f"gap_vs_{key}_ci95_hi_bca"] = ghi
        p_raw[f"graph_vs_{key}"] = _one_sided_gap_pvalue(gap_boot)

    res["holm"] = _holm_bonferroni(p_raw)
    res["cohens_h"] = {
        "graph_vs_no_rag": _cohens_h(res[GRAPH_RAG]["em_observed"], res["no-rag"]["em_observed"]),
        "graph_vs_llm_cypher": _cohens_h(res[GRAPH_RAG]["em_observed"], res["llm-cypher"]["em_observed"]),
    }
    return res


def _write_csvs(res: dict) -> None:
    """Persist the BCa CSV (step_3_7 schema) and the Holm + Cohen CSV."""
    rows = []
    for cfg in BASELINES:
        r = res[cfg]
        row = {
            "baseline": cfg,
            "em_observed": r["em_observed"],
            "em_bootstrap_mean": r["em_bootstrap_mean"],
            "em_bootstrap_std": r["em_bootstrap_std"],
            "ci95_lo_bca": round(r["ci95_lo_bca"], 2),
            "ci95_hi_bca": round(r["ci95_hi_bca"], 2),
            "n_bootstrap": N_BOOTSTRAP,
            "n_queries": 100,
            "method": "BCa",
        }
        if cfg == GRAPH_RAG:
            row.update({
                "gap_vs_no_rag_mean": r["gap_vs_no_rag_mean"],
                "gap_vs_no_rag_ci95_lo_bca": round(r["gap_vs_no_rag_ci95_lo_bca"], 2),
                "gap_vs_no_rag_ci95_hi_bca": round(r["gap_vs_no_rag_ci95_hi_bca"], 2),
                "gap_vs_no_rag_sig_bca": r["gap_vs_no_rag_ci95_lo_bca"] > 0,
                "gap_vs_llm_cypher_mean": r["gap_vs_llm_cypher_mean"],
                "gap_vs_llm_cypher_ci95_lo_bca": round(r["gap_vs_llm_cypher_ci95_lo_bca"], 2),
                "gap_vs_llm_cypher_ci95_hi_bca": round(r["gap_vs_llm_cypher_ci95_hi_bca"], 2),
                "gap_vs_llm_cypher_sig_bca": r["gap_vs_llm_cypher_ci95_lo_bca"] > 0,
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_BCA, index=False)

    holm_rows = []
    for name, hv in res["holm"].items():
        holm_rows.append({
            "comparison": name,
            "p_raw": round(hv["p_raw"], 6),
            "p_holm": round(hv["p_holm"], 6),
            "significant_holm_005": hv["p_holm"] < 0.05,
            "cohens_h": round(res["cohens_h"][name], 4),
        })
    pd.DataFrame(holm_rows).to_csv(OUT_HOLM_COHEN, index=False)


def main() -> None:
    """Regenerate the BCa, Holm and Cohen artefacts and log a summary."""
    res = regenerate()
    _write_csvs(res)
    gr = res[GRAPH_RAG]
    logger.info("Regenerated BCa intervals -> %s", OUT_BCA.name)
    logger.info("Regenerated Holm + Cohen -> %s", OUT_HOLM_COHEN.name)
    logger.info("=" * 70)
    logger.info(
        "  EM Graph-RAG %.0f%% BCa [%.0f%%, %.0f%%]; gap vs no-RAG %+.1f pp BCa [%+.0f, %+.0f] (Holm p=%.4g).",
        100 * gr["em_observed"], 100 * gr["ci95_lo_bca"], 100 * gr["ci95_hi_bca"],
        100 * gr["gap_vs_no_rag_mean"], 100 * gr["gap_vs_no_rag_ci95_lo_bca"],
        100 * gr["gap_vs_no_rag_ci95_hi_bca"], res["holm"]["graph_vs_no_rag"]["p_holm"],
    )
    logger.info("  Cohen's h graph vs no-RAG = %.3f (large).", res["cohens_h"]["graph_vs_no_rag"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
