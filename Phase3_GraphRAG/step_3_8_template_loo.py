"""
Step 3.8 - Cypher Template Leave-One-Out Criticality
=====================================================
Phase 3 robustness layer (additive, post-baseline)

Ranks each Cypher template by the pessimistic EM drop that would
result from removing it from the routing table. For each template T:

    n_queries_using       = number of graph-rag queries routed to T
    n_em_correct          = of those, how many had exact_match=True
    template_em_rate      = n_em_correct / n_queries_using
    drop_em_if_removed_pp = n_em_correct / 100 * 100   (in pp of the
                            overall EM, assuming pessimistic LOO:
                            without T, those queries fail).

criticality_rank is the dense rank by drop_em_if_removed_pp desc.

Input (read-only):
    data/evaluation_results_graph-rag_*.json (most recent timestamp)

Output:
    results/step_3_8_template_leave_one_out.csv
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RESULTS_DIR / "step_3_8_template_leave_one_out.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _latest_graph_rag_file() -> Path:
    candidates = sorted(DATA_DIR.glob("evaluation_results_graph-rag_*.json"))
    if not candidates:
        raise FileNotFoundError("No graph-rag evaluation result found in data/")
    return candidates[-1]


def main() -> None:
    path = _latest_graph_rag_file()
    logger.info("Step 3.8 starting: template LOO criticality from %s", path.name)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    n_total = len(data)
    em_total = sum(1 for e in data if e.get("exact_match") is True)
    logger.info("Loaded %d entries (EM baseline = %d/%d = %.1f%%)", n_total, em_total, n_total, 100*em_total/n_total)

    n_queries: Counter[str] = Counter()
    em_counts: Counter[str] = Counter()
    for entry in data:
        tpl = entry.get("cypher_used") or "UNROUTED"
        n_queries[tpl] += 1
        if entry.get("exact_match") is True:
            em_counts[tpl] += 1

    rows: list[dict] = []
    for tpl, n in n_queries.items():
        em_ok = em_counts.get(tpl, 0)
        rows.append({
            "template_name": tpl,
            "n_queries_using": n,
            "n_em_correct": em_ok,
            "template_em_rate": round(em_ok / n, 4) if n else 0.0,
            "drop_em_if_removed_pp": round(em_ok * 100.0 / n_total, 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("drop_em_if_removed_pp", ascending=False).reset_index(drop=True)
    df.insert(0, "criticality_rank", df.index + 1)
    df.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(df))

    sum_used = int(df["n_queries_using"].sum())
    if sum_used != n_total:
        logger.info(
            "Note: sum(n_queries_using)=%d != %d; this happens if some "
            "entries had no cypher_used field (treated as UNROUTED).",
            sum_used, n_total,
        )

    top3 = df.head(3)
    logger.info("=" * 70)
    logger.info(
        "  Top-3 templates by EM-loss-if-removed: %s",
        ", ".join(
            f"{r['template_name']} ({r['drop_em_if_removed_pp']:.1f} pp)"
            for _, r in top3.iterrows()
        ),
    )
    n_critical = int((df["drop_em_if_removed_pp"] > 5.0).sum())
    logger.info("  Templates with drop_em_if_removed_pp > 5 pp: %d", n_critical)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
