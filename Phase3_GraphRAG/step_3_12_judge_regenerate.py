"""Step 3.12 - Deterministic regeneration of the LLM-judge summary (CRITICAL #1a).

Phase 3 robustness layer (additive, post-audit 2026-06-05).

The canonical EM-semantic accuracy of thesis Table 5.6 is 0.41 (41.0%) with
EM-strict 0.59 (59.0%), produced by the Sonnet 4.6 cross-model judge of step_3_9
on the frozen 20260517 graph-rag evaluation. The committed
results/step_3_9_llm_judge_summary.csv was later overwritten to 66/63 because
step_3_9 picks the newest evaluation JSON by glob, so the canonical 59/41 split
survived only in git history (commit fa4f069).

This module restores reproducibility without any paid LLM call: it reads the
frozen per-question judge verdicts (data/step_3_9_judge_verdicts_canonical.csv,
extracted verbatim from fa4f069) and recomputes the exact same summary metrics
that step_3_9.main produces, so the 59/41 numbers are regenerable from committed
code and committed data. Run this, not step_3_9, to refresh the canonical judge
summary; step_3_9 remains the original (expensive, glob-latest) producer.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

CANONICAL_VERDICTS = DATA_DIR / "step_3_9_judge_verdicts_canonical.csv"
OUT_SUMMARY = RESULTS_DIR / "step_3_9_llm_judge_summary.csv"

# Recorded for traceability, identical to step_3_9 so the regenerated summary is
# byte-comparable with the original canonical run.
JUDGE_MODEL = "claude-sonnet-4-6"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def regenerate_judge_summary(verdicts_path: Path | None = None) -> dict[str, float | int | str]:
    """Recompute the canonical judge summary from frozen per-question verdicts.

    The aggregation matches step_3_9.main exactly (same rounding to 2 decimals and
    the same false-positive / false-negative definitions) so the returned values
    reproduce the canonical thesis Table 5.6 figures.

    Args:
        verdicts_path: Path to the frozen per-question verdicts CSV. Defaults to
            the committed canonical file extracted from commit fa4f069.

    Returns:
        Mapping of summary metric name to value, including em_strict_pct and
        em_semantic_pct.

    Raises:
        FileNotFoundError: If the frozen verdicts CSV is missing.
        ValueError: If the verdicts CSV lacks the required em_strict / em_semantic
            columns.
    """
    path = Path(verdicts_path) if verdicts_path is not None else CANONICAL_VERDICTS
    if not path.is_file():
        raise FileNotFoundError(f"Frozen judge verdicts not found: {path}")

    df_per = pd.read_csv(path)
    required = {"em_strict", "em_semantic"}
    missing = required - set(df_per.columns)
    if missing:
        raise ValueError(f"Verdicts CSV missing columns: {sorted(missing)}")

    em_strict_pct = round(100.0 * df_per["em_strict"].mean(), 2)
    em_semantic_pct = round(100.0 * df_per["em_semantic"].mean(), 2)
    if "agreement" in df_per.columns:
        agreement_pct = round(100.0 * df_per["agreement"].mean(), 2)
    else:
        agreement_pct = round(100.0 * (df_per["em_strict"] == df_per["em_semantic"]).mean(), 2)
    false_neg = df_per[(df_per["em_strict"] == 0) & (df_per["em_semantic"] == 1)]
    false_pos = df_per[(df_per["em_strict"] == 1) & (df_per["em_semantic"] == 0)]
    false_neg_rate = round(100.0 * len(false_neg) / len(df_per), 2)
    false_pos_rate = round(100.0 * len(false_pos) / len(df_per), 2)
    keyword_floor_pp = round(em_semantic_pct - em_strict_pct, 2)

    return {
        "em_strict_pct": em_strict_pct,
        "em_semantic_pct": em_semantic_pct,
        "agreement_pct": agreement_pct,
        "false_negative_rate_pct": false_neg_rate,
        "false_positive_rate_pct": false_pos_rate,
        "keyword_matcher_floor_pp": keyword_floor_pp,
        "n_queries": len(df_per),
        "judge_model": JUDGE_MODEL,
    }


def write_summary(summary: dict[str, float | int | str], out_path: Path | None = None) -> Path:
    """Write the summary mapping to the canonical summary CSV (metric, value).

    Args:
        summary: Mapping produced by regenerate_judge_summary.
        out_path: Destination CSV. Defaults to the canonical summary path.

    Returns:
        The path written.
    """
    dest = Path(out_path) if out_path is not None else OUT_SUMMARY
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


def main() -> None:
    """Regenerate and persist the canonical judge summary, then log the result."""
    summary = regenerate_judge_summary()
    dest = write_summary(summary)
    logger.info("Regenerated canonical judge summary from frozen verdicts -> %s", dest)
    logger.info(
        "  EM strict %.1f%%, EM semantic %.1f%% (judge %s); agreement %.1f%%, FP %.1f%%, FN %.1f%%.",
        summary["em_strict_pct"],
        summary["em_semantic_pct"],
        summary["judge_model"],
        summary["agreement_pct"],
        summary["false_positive_rate_pct"],
        summary["false_negative_rate_pct"],
    )


if __name__ == "__main__":
    main()
