"""
run_phase_3_blind.py - Phase 3 Blind orchestrator
=================================================
Runs the full Neuro-Symbolic Regulatory Reasoner pipeline end to end:

  1. Ingest the corpus facts into the Regulatory KB.
  2. Estimate delta_TC and write the Phase-2 drop-in JSON.
  3. Demonstrate the compliance gate on the canonical scenarios.
  4. Build the benchmark (if missing) and run the RQ3 evaluation.
  5. Run the robustness layer.

Everything is deterministic and runs offline (no API). Writes artefacts under
data/ and results/.
"""

from __future__ import annotations

from step_3_0_config import Country, Tier, Scenario, get_logger
from step_3_1_ingest import load_kb
from step_3_3_compliance_gate import compliance_gate
from step_3_4_delta_tc import estimate_delta_tc, write_delta_tc_json
from step_3_6_evaluate import main as run_eval
from step_3_7_robustness import main as run_robustness

logger = get_logger("phase3_blind")


def main() -> None:
    logger.info("=== Phase 3 Blind: Neuro-Symbolic Regulatory Reasoner ===")

    # 1. Ingest
    kb = load_kb()

    # 2. delta_TC
    res = estimate_delta_tc(kb)
    write_delta_tc_json(res, kb=kb)
    logger.info("delta_TC reduction_factors: %s", res.reduction_factors())

    # 3. Compliance gate demo on canonical scenarios
    logger.info("--- Compliance gate (canonical scenarios) ---")
    for scale in ("Edge", "Mid", "Hyperscale"):
        for tier in Tier:
            r = compliance_gate(Scenario(scale, tier, country=Country.ITALY), kb)
            logger.info("  %-11s %-6s IT -> %-13s [%s]",
                        scale, tier.value, r.status.value, r.required_upgrade.value)

    # 4. Benchmark + evaluation
    from step_3_0_config import BENCHMARK_JSONL
    if not BENCHMARK_JSONL.exists():
        from build_benchmark import main as build_bench
        build_bench()
    run_eval()

    # 5. Robustness
    run_robustness()

    logger.info("=== Phase 3 Blind pipeline complete ===")


if __name__ == "__main__":
    main()
