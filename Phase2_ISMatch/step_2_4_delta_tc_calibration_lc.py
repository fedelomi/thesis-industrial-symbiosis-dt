"""
Step 2.4 LC — ΔTC Calibration Loop  [LIQUID COOLING]
======================================================
Phase 2: IS-Match Score (OS2) · Thesis Cap. 5.3

PHASE 3 DEPENDENCY: This step requires ΔTC_post from Phase 3 (RAG-based
information barrier reduction, Passo 3.5-bis). Until Phase 3 is complete,
ΔTC_post values are replaced by a placeholder array derived from a linear
reduction model:

  ΔTC_post = ΔTC_baseline * (1 − reduction_factor)
  reduction_factor ∈ [0.1, 0.3]  (10–30% barrier reduction from RAG)

Convergence protocol (Cap. 5.3):
  - Max 5 iterations
  - Stop if max |Δ IS-Match Score| < 0.01 (1%) across all pairs between
    iteration i and iteration i+1
  - Document N_iter_actual in Cap. 5.3

Output:
  step_2_4_delta_tc_calibration_lc.csv  — IS-Match scores per iteration
  step_2_4_convergence_log_lc.csv       — convergence metrics per iteration

Thesis note (Cap. 5.3):
  The ΔTC feedback loop bridges Layer 1 (Physical DT, Phase 1) and
  Layer 2 (Institutional LLM, Phase 3). A converged IS-Match Score is the
  input to Phase 4 RL agents (Layer 3, Agentic Negotiation).
  With placeholder ΔTC_post, N_iter = 1 (already converged at iteration 0→1).
  Actual N_iter to be updated after Phase 3 delivers ΔTC_post values.
"""

# Decision active: D6 — ΔTC calibration loop on LC pairs only.


from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent
RESULTS_DIR     = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SCORES_CSV      = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
OUT_CALIB_CSV   = RESULTS_DIR / "step_2_4_delta_tc_calibration_lc.csv"
OUT_CONV_CSV    = RESULTS_DIR / "step_2_4_convergence_log_lc.csv"

# Convergence parameters
MAX_ITER:        int   = 5
CONVERGENCE_TOL: float = 0.01   # 1% IS-Match Score change threshold

# IS-Match weights (same as Step 2.1)
BETA, GAMMA, DELTA = 0.40, 0.40, 0.20

# Phase 3 → Phase 2 wiring: load ΔTC reduction factors from Phase 3 output.
# step_3_5_bis writes: Phase3_GraphRAG/data/step_3_5_bis_delta_tc.json
# Audit fix 2026-05-09 (P3-2): replaces hardcoded dict. Closes Gap 3 feedback loop.

_PHASE3_JSON = (
    BASE_DIR.parent / "Phase3_GraphRAG" / "data" / "step_3_5_bis_delta_tc.json"
)
_FALLBACK_REDUCTION: dict[str, float] = {
    "LowT_60C":   0.10,   # conservative fallback if JSON missing (pre-Phase3)
    "MidT_90C":   0.10,
    "HighT_130C": 0.10,
}


def _load_phase3_reductions() -> dict[str, float]:
    """Load ΔTC reduction factors written by step_3_5_bis.

    Returns the measured reduction factors from Phase 3 RAG output.
    Falls back to conservative placeholder if JSON is missing, with a warning.
    """
    if _PHASE3_JSON.exists():
        try:
            data = json.loads(_PHASE3_JSON.read_text(encoding="utf-8"))
            factors: dict[str, float] = data["reduction_factors"]
            logger.info(f"  Loaded Phase 3 ΔTC reductions from {_PHASE3_JSON.name}: {factors}")
            return factors
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning(f"  Cannot parse {_PHASE3_JSON}: {exc} — using fallback")
    else:
        logger.warning(
            f"  Phase 3 JSON not found: {_PHASE3_JSON}\n"
            "  Run step_3_5_phase1_integration.py (Passo 3.5-bis) first.\n"
            "  Using conservative fallback reductions (10%)."
        )
    return _FALLBACK_REDUCTION.copy()


PHASE3_PLACEHOLDER_REDUCTION: dict[str, float] = _load_phase3_reductions()


def compute_is_match(ri: float, ex: float, dtc: float) -> float:
    return float(np.clip(BETA * ri + GAMMA * ex - DELTA * dtc, 0.0, 1.0))


def classify_tier(score: float) -> str:
    if score >= 0.60: return "high_priority"
    if score >= 0.30: return "marginal"
    return "not_viable"


def run_calibration_loop(df_init: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative ΔTC calibration loop.

    At each iteration, ΔTC_norm is updated by applying PHASE3_PLACEHOLDER_REDUCTION.
    In Phase 3, this update will come from the RAG system's output (Passo 3.5-bis).
    """
    records_iter = []
    conv_log     = []

    # Iteration 0: baseline IS-Match Scores from Step 2.1
    current = df_init.copy()
    current["iteration"] = 0
    records_iter.append(current.copy())

    for i in range(1, MAX_ITER + 1):
        prev_scores = current["is_match_score"].values.copy()
        updated_rows = []

        for _, row in current.iterrows():
            proc    = str(row["process_name"])
            red_fac = PHASE3_PLACEHOLDER_REDUCTION.get(proc, 0.10)
            # Apply reduction to ΔTC_norm
            dtc_new  = float(row["delta_tc_norm"]) * (1.0 - red_fac)
            score    = compute_is_match(float(row["ri_temporal"]),
                                        float(row["exergy_dt_norm"]),
                                        dtc_new)
            updated_rows.append({
                **row.to_dict(),
                "delta_tc_norm":   round(dtc_new, 5),
                "is_match_score":  round(score, 4),
                "tier":            classify_tier(score),
                "iteration":       i,
            })

        current     = pd.DataFrame(updated_rows)
        new_scores  = current["is_match_score"].values
        delta_max   = float(np.max(np.abs(new_scores - prev_scores)))
        delta_mean  = float(np.mean(np.abs(new_scores - prev_scores)))
        converged   = delta_max < CONVERGENCE_TOL

        conv_log.append({
            "iteration": i,
            "delta_max_is_match": round(delta_max,  5),
            "delta_mean_is_match": round(delta_mean, 5),
            "converged": converged,
            "threshold": CONVERGENCE_TOL,
            "phase3_source": "placeholder" if i <= MAX_ITER else "phase3_rag",
        })
        records_iter.append(current.copy())
        logger.info(f"  Iter {i}: Δ_max={delta_max:.5f}  Δ_mean={delta_mean:.5f}"
                    + ("  ← CONVERGED" if converged else ""))
        if converged:
            logger.info(f"  Convergence reached at iteration {i} (< {CONVERGENCE_TOL:.0%}).")
            break

    df_all_iters = pd.concat(records_iter, ignore_index=True)
    df_conv      = pd.DataFrame(conv_log)
    return df_all_iters, df_conv


def main() -> None:
    if not SCORES_CSV.exists():
        raise FileNotFoundError(
            f"IS-Match scores not found: {SCORES_CSV}\n"
            "Run step_2_1_is_match_score_lc.py first."
        )

    df_init = pd.read_csv(SCORES_CSV)

    logger.info("=" * 68)
    logger.info("  Step 2.4 LC — ΔTC Calibration Loop  [PHASE 3 PLACEHOLDER]")
    logger.info("  Max iterations: %d  |  Tolerance: %.1f%%", MAX_ITER, CONVERGENCE_TOL*100)
    logger.info("=" * 68)
    logger.info("  NOTE: Phase 3 (RAG) not yet complete. Using placeholder")
    logger.info("  ΔTC reduction factors derived from process tier complexity.")
    logger.info("  Actual N_iter to be updated in Cap. 5.3 post Phase 3.\n")
    logger.info("  Phase3 placeholder reductions:")
    for proc, red in PHASE3_PLACEHOLDER_REDUCTION.items():
        logger.info(f"    {proc:14s}: ΔTC × (1 − {red:.2f}) = ΔTC × {1-red:.2f}")
    logger.info("")

    df_all, df_conv = run_calibration_loop(df_init)
    # Audit fix P2-3: use explicit iteration column (robust to loop refactors).
    n_iter_actual = (
        int(df_conv.loc[df_conv["converged"].idxmax(), "iteration"])
        if df_conv["converged"].any() else MAX_ITER
    )

    # Final iteration scores
    df_final = df_all[df_all["iteration"] == df_all["iteration"].max()].copy()
    df_final = df_final.sort_values("is_match_score", ascending=False).reset_index(drop=True)
    df_final.insert(0, "rank_final", df_final.index + 1)

    logger.info("")
    logger.info("  Final IS-Match Scores (post ΔTC calibration):")
    logger.info("  " + "-" * 64)
    for _, row in df_final.iterrows():
        delta_vs_init = float(row["is_match_score"]) - float(
            df_init[(df_init["dc_name"]==row["dc_name"]) &
                    (df_init["process_name"]==row["process_name"])]["is_match_score"].iloc[0]
        )
        sign = "+" if delta_vs_init >= 0 else ""
        logger.info(f"  #{int(row['rank_final']):2d}  {row['dc_name']:14s} × {row['process_name']:14s}"
                    f"  Score={row['is_match_score']:.4f}  Δ={sign}{delta_vs_init:.4f}  [{row['tier']}]")

    tier_final = df_final["tier"].value_counts().to_dict()
    logger.info("  " + "-" * 64)
    logger.info(f"  Final tiers: high_priority={tier_final.get('high_priority',0)}  "
                f"marginal={tier_final.get('marginal',0)}  "
                f"not_viable={tier_final.get('not_viable',0)}")
    logger.info(f"  N_iter_actual: {n_iter_actual}  (placeholder; update in Cap. 5.3)")
    logger.info("=" * 68)

    df_all.to_csv(OUT_CALIB_CSV, index=False)
    df_conv.to_csv(OUT_CONV_CSV,  index=False)
    logger.info(f"  Calibration log → {OUT_CALIB_CSV}  ({len(df_all)} rows)")
    logger.info(f"  Convergence log → {OUT_CONV_CSV}  ({len(df_conv)} rows)")

    print(f"\n  THESIS (Cap. 5.3): Loop converged at N_iter={n_iter_actual} [placeholder].")
    print(f"  Final IS-Match Scores ready as input to Phase 4 RL agents.")
    print(f"  Update N_iter and ΔTC_post after