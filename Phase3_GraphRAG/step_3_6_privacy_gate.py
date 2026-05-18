"""
step_3_6_privacy_gate.py
========================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Passo 3.6: Privacy preservation gate (extracted from step_3_5_phase1_integration.py
during audit reorg 2026-05-05 — FIX-6).

Compares the IS-Match implication when using the real Phase 1 LC profile vs.
the synthetic profile (sigma=0.05*std bootstrap, Phase 1 step_1_3 calibration).

Gate criterion:
    Delta IS-Match proxy < 5%   ->   PASS
    Delta IS-Match proxy >= 5%  ->   FAIL

The proxy used here is the relative difference in Q_available_mean between real
and synthetic profiles (which propagates linearly to the RI_quantity component
of the IS-Match Score). A more rigorous version would re-run the entire Phase 2
pipeline on the synthetic profile and compare the resulting IS-Match Scores
directly — that is left to follow-up work.

Output:
    data/step_3_6_privacy_gate.json

Wiki references:
- [[roadmap-fasi-1-2-3]] Passo 3.6
- [[decisioni-implementative]] D5 (synthetic-via-bootstrap), D6 (LC-only)
- [[concepts/is-match-score]]
"""

# Decisions active: D5 — gate over the D5-parametrized synthetic LC profile.


from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Re-use the canonical Phase 1 LC statistics defined in step_3_5_phase1_integration.
# This avoids drift between the two scripts on the wiki-validated reference values.
from step_3_5_phase1_integration import (  # noqa: E402
    PHASE1_LC_STATS,
    DATA_DIR,
)

# Privacy gate threshold from roadmap Passo 3.6
DELTA_THRESHOLD: float = 0.05  # 5% relative difference

# Synthetic perturbation scale matches Phase 1 step_1_3 (SIDED-style bootstrap).
SIGMA_FRAC: float = 0.05

# Reproducible RNG seed
RNG_SEED: int = 42


def run_privacy_gate() -> dict:
    """Compute the privacy gate result for the three LC scenarios.

    Returns a dict ready to be JSON-serialised, with structure:
        {
            "gate": "PASS" | "FAIL",
            "details": {
                "DC-S": {"real_q_kw": ..., "synth_q_kw": ..., "delta_pct": ..., "gate": ...},
                "DC-M": {...},
                "DC-L": {...},
            },
            "threshold_pct": 5.0,
            "method": "Q_available_mean delta proxy with sigma=0.05*std bootstrap",
        }
    """
    logger.info("\n-- Passo 3.6: Privacy Gate ------------------------------------")

    rng = random.Random(RNG_SEED)
    real_stats: dict[str, float] = {
        k: v["q_available_mean_kw"] for k, v in PHASE1_LC_STATS.items()
    }
    synth_stats: dict[str, float] = {
        k: v["q_available_mean_kw"] * (
            1
            + rng.gauss(
                0,
                SIGMA_FRAC * v["q_available_std_kw"] / v["q_available_mean_kw"],
            )
        )
        for k, v in PHASE1_LC_STATS.items()
    }

    gate_results: dict[str, dict] = {}
    all_pass = True
    for dc_id in ["DC-S", "DC-M", "DC-L"]:
        real_q  = real_stats[dc_id]
        synth_q = synth_stats[dc_id]
        delta   = abs(real_q - synth_q) / real_q
        passed  = delta < DELTA_THRESHOLD
        if not passed:
            all_pass = False
        gate_results[dc_id] = {
            "real_q_kw":  real_q,
            "synth_q_kw": round(synth_q, 2),
            "delta_pct":  round(delta * 100, 3),
            "gate":       "PASS" if passed else "FAIL",
        }
        sym = "PASS" if passed else "FAIL"
        log_fn = logger.info if passed else logger.error
        log_fn(f"  {dc_id}: real={real_q:.0f} kW  synth={synth_q:.0f} kW  "
               f"delta={delta:.2%}  [{sym}]")

    overall = "PASS" if all_pass else "FAIL"
    if all_pass:
        logger.info(f"\n  Privacy Gate overall: {overall}")
    else:
        logger.error(f"\n  Privacy Gate overall: {overall}")
    logger.info(f"  (Target: Delta IS-Match proxy < {DELTA_THRESHOLD:.0%} -> PASS)")

    return {
        "gate":          overall,
        "details":       gate_results,
        "threshold_pct": DELTA_THRESHOLD * 100,
        "method":        f"Q_available_mean delta proxy with sigma={SIGMA_FRAC}*std bootstrap",
    }


def save_result(result: dict, out_path: Path | None = None) -> Path:
    """Persist the gate result as JSON. Returns the path written."""
    target = out_path or (DATA_DIR / "step_3_6_privacy_gate.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Passo 3.6 — Privacy gate")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional output path (default: data/step_3_6_privacy_gate.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  Phase 3 - step_3_6_privacy_gate.py")
    logger.info("  Passo 3.6 — Privacy preservation gate")
    logger.info("=" * 60)
    result = run_privacy_gate()
    save_result(result, out_path=args.out)
    if result["gate"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
