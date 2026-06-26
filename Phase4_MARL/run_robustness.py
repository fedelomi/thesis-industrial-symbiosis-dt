"""
run_robustness.py
==================
Phase 4 robustness layer orchestrator (additive, no retraining of
canonical checkpoints required).

Runs the four post-baseline robustness modules in sequence:

    step_4_5_bootstrap_poa_ci.py                bootstrap CI on PoA
    step_4_7_state_observation_sensitivity.py   observation sensitivity
    step_4_6_shielding_loo_evaluation.py        shielding LOO (heavy)
    step_4_8_deltatc_perturbation.py            delta_TC perturbation

Usage:
    python run_robustness.py                  # full run, ~10-15 min
    python run_robustness.py --skip-heavy     # skip step_4_6 (~8 min)

The fast modules (4_5, 4_7, 4_8) take ~3 min combined. step_4_6 is
the heavy one because it (re)trains 12 D-config checkpoints for the
S1/S4/S7/S_AIR_M x 3 seeds matrix when they are missing, then runs
the LOO evaluation.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PHASE4_ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Step 4.5 bootstrap PoA CI",                     "step_4_5_bootstrap_poa_ci.py",        False),
    ("Step 4.7 state observation sensitivity",         "step_4_7_state_observation_sensitivity.py", False),
    ("Step 4.6 shielding LOO evaluation (heavy)",     "step_4_6_shielding_loo_evaluation.py", True),
    ("Step 4.8 delta_TC perturbation (scoped S5)",    "step_4_8_deltatc_perturbation.py",    False),
    ("Step 4.12 BCa + Holm + Cohen regenerator",      "step_4_12_bca_holm_cohen.py",         False),
]


def _run(script: str) -> int:
    path = PHASE4_ROOT / script
    if not path.exists():
        logger.error("Script not found: %s", path)
        return 1
    res = subprocess.run([sys.executable, str(path)], cwd=str(PHASE4_ROOT))
    return int(res.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 robustness layer orchestrator")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip step_4_6 (shielding LOO; ~ 8 min retrain + eval)")
    args = parser.parse_args()

    t0 = time.time()
    failed: list[str] = []
    for name, script, heavy in STEPS:
        if heavy and args.skip_heavy:
            logger.info("[SKIP] %s (--skip-heavy)", name)
            continue
        logger.info("=" * 70)
        logger.info("[RUN]  %s -> %s", name, script)
        logger.info("=" * 70)
        rc = _run(script)
        tag = "PASS" if rc == 0 else "FAIL"
        logger.info("[%s] %s", tag, name)
        if rc != 0:
            failed.append(name)

    logger.info("=" * 70)
    logger.info("  Total time: %.1fs", time.time() - t0)
    if failed:
        logger.error("Failed steps: %s", failed)
        sys.exit(1)
    logger.info("  Phase 4 robustness layer: PASS")


if __name__ == "__main__":
    main()
