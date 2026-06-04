"""
Step 3.7 Blind - Robustness layer for the RQ3 evaluation
========================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Three post-hoc diagnostics over the RQ3 benchmark, mapping to the framework's
robustness families (Section 3.6.4): bootstrap confidence intervals, leave-one-out
component criticality and perturbation stability.

  bootstrap_ci   : paired percentile bootstrap CI on the NSRR-minus-baseline
                   accuracy gap (10,000 resamples, seeded -> reproducible).
  component_loo  : ablate the symbolic compliance gate (use_gate=False) and measure
                   the NSRR accuracy drop -> the gate's load-bearing contribution.
  paraphrase     : apply deterministic surface paraphrases to every question and
                   measure the NSRR accuracy delta -> routing/extraction stability.

All randomness is seeded with GLOBAL_SEED, so the outputs are reproducible.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import numpy as np

from step_3_0_config import GLOBAL_SEED, RESULTS_DIR, get_logger
from step_3_1_ingest import load_kb
from step_3_5_answerer import BaselineAnswerer, NSRRAnswerer
from step_3_6_evaluate import load_benchmark, score_item

logger = get_logger(__name__)

_PARAPHRASES = [
    (r"\bdata centre\b", "DC"),
    (r"\bwaste heat\b", "surplus heat"),
    (r"According to the regulatory corpus, what is the value of:", "State the corpus value for"),
    (r"\bWhich\b", "What"),
    (r"\bis required\b", "is mandated"),
    (r"\bapply\b", "hold"),
]


def _correctness_vectors(use_gate: bool = True, paraphrase: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return paired per-item correctness arrays (NSRR, baseline) and categories."""
    kb = load_kb()
    nsrr = NSRRAnswerer(kb, use_gate=use_gate)
    base = BaselineAnswerer(kb)
    bench = load_benchmark()
    n_ok, b_ok, cats = [], [], []
    for item in bench:
        q = item["question"]
        if paraphrase:
            for pat, repl in _PARAPHRASES:
                q = re.sub(pat, repl, q)
        unit = item.get("gold_unit", "")
        a_n = nsrr.answer(q, prefer_unit=unit)
        a_b = base.answer(q, prefer_unit=unit)
        n_ok.append(1 if score_item(item, a_n) else 0)
        b_ok.append(1 if score_item(item, a_b) else 0)
        cats.append(item["category"])
    return np.array(n_ok), np.array(b_ok), cats


def bootstrap_ci(n_ok: np.ndarray, b_ok: np.ndarray, n_boot: int = 10000,
                 seed: int = GLOBAL_SEED) -> Dict[str, float]:
    """Paired percentile bootstrap CI on the accuracy gap (NSRR - baseline)."""
    rng = np.random.default_rng(seed)
    n = len(n_ok)
    gaps = np.empty(n_boot)
    diff = n_ok - b_ok
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps[i] = diff[idx].mean()
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "gap_mean": float(diff.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "significant": bool(lo > 0.0),
        "n_boot": n_boot,
    }


def main() -> Dict:
    """Run all three diagnostics and persist a summary."""
    n_ok, b_ok, cats = _correctness_vectors(use_gate=True)
    ci = bootstrap_ci(n_ok, b_ok)

    # Component leave-one-out: disable the symbolic gate.
    n_ok_nogate, _, _ = _correctness_vectors(use_gate=False)
    full_acc = float(n_ok.mean())
    nogate_acc = float(n_ok_nogate.mean())

    # Paraphrase stability.
    n_ok_par, b_ok_par, _ = _correctness_vectors(use_gate=True, paraphrase=True)
    par_acc = float(n_ok_par.mean())

    summary = {
        "bootstrap_gap_ci": ci,
        "component_loo": {
            "nsrr_full_accuracy": round(full_acc, 4),
            "nsrr_no_gate_accuracy": round(nogate_acc, 4),
            "gate_contribution_pp": round((full_acc - nogate_acc) * 100, 2),
        },
        "paraphrase_stability": {
            "nsrr_accuracy_original": round(full_acc, 4),
            "nsrr_accuracy_paraphrased": round(par_acc, 4),
            "delta_pp": round((par_acc - full_acc) * 100, 2),
            "stable_within_5pp": bool(abs(par_acc - full_acc) <= 0.05),
        },
    }

    print("\n" + "=" * 70)
    print("  ROBUSTNESS LAYER (Phase 3 Blind)")
    print("=" * 70)
    print(f"  Bootstrap gap (NSRR-baseline): {ci['gap_mean']*100:+.1f} pp  "
          f"95% CI [{ci['ci95_low']*100:+.1f}, {ci['ci95_high']*100:+.1f}]  "
          f"significant={ci['significant']}")
    print(f"  Gate LOO: full={full_acc*100:.1f}%  no-gate={nogate_acc*100:.1f}%  "
          f"gate contributes {(full_acc-nogate_acc)*100:+.1f} pp")
    print(f"  Paraphrase: orig={full_acc*100:.1f}%  paraphrased={par_acc*100:.1f}%  "
          f"delta={(par_acc-full_acc)*100:+.1f} pp  stable={summary['paraphrase_stability']['stable_within_5pp']}")
    print("=" * 70)

    (RESULTS_DIR / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote results/robustness_summary.json")
    return summary


if __name__ == "__main__":
    main()
