"""check_routing_latency.py - free per-query routing latency gate.

No API, no Neo4j. Measures the steady-state wall-clock cost of route_question
(keyword stage + semantic fallback when triggered), excluding the one-off
router/model build, over a fixed sample of benchmark questions.

Targets (thesis routing budget): p50 < 50 ms, p95 < 100 ms over 100 queries.

Usage:
    python check_routing_latency.py
    SEMANTIC_ROUTER_BACKEND=st python check_routing_latency.py

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from step_3_4_evaluation import ENABLE_SEMANTIC_FALLBACK, route_question
from semantic_router import DEFAULT_BACKEND, get_default_router

HERE = Path(__file__).resolve().parent
DATASET = HERE / "data" / "benchmark_qa_dataset.json"

P50_TARGET_MS = 50.0
P95_TARGET_MS = 100.0
N_SAMPLES = 100


def percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * len(sorted_vals) + 0.5)) - 1))
    return sorted_vals[k]


def main() -> None:
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    nl = [q["nl_question"] for q in questions]
    # Cycle to exactly N_SAMPLES queries.
    sample = [nl[i % len(nl)] for i in range(N_SAMPLES)]

    # Warm up: force the router (and any model load) to build, and JIT the
    # keyword path. This one-off cost is excluded from the measurement.
    if ENABLE_SEMANTIC_FALLBACK:
        get_default_router()
    for q in nl[:5]:
        route_question(q)

    timings_ms: list[float] = []
    for q in sample:
        t0 = time.perf_counter()
        route_question(q)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    p50 = percentile(timings_ms, 50)
    p95 = percentile(timings_ms, 95)
    mean = sum(timings_ms) / len(timings_ms)

    print(f"backend={DEFAULT_BACKEND}  ENABLE_SEMANTIC_FALLBACK={ENABLE_SEMANTIC_FALLBACK}")
    print(f"samples={len(timings_ms)} (steady-state, build excluded)")
    print(f"  mean : {mean:7.2f} ms")
    print(f"  p50  : {p50:7.2f} ms   target < {P50_TARGET_MS:.0f}   "
          f"{'PASS' if p50 < P50_TARGET_MS else 'FAIL'}")
    print(f"  p95  : {p95:7.2f} ms   target < {P95_TARGET_MS:.0f}   "
          f"{'PASS' if p95 < P95_TARGET_MS else 'FAIL'}")
    print(f"  max  : {timings_ms[-1]:7.2f} ms")


if __name__ == "__main__":
    main()
