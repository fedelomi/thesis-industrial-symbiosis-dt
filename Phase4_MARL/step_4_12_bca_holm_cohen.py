"""Step 4.12 - Deterministic BCa CI + Holm + Cohen regenerator (CRITICAL #1b, Phase 4).

Phase 4 robustness layer (additive, post-audit 2026-06-05).

The committed BCa intervals (results/robustness/step_4_5_bootstrap_ci_bca.csv and
step_4_5_gap_ci_bca.csv), the Holm-Bonferroni family correction
(step_4_5_holm_bonferroni.csv) and the paired Cohen's d (step_4_5_cohens_d.csv)
had no producing script in the repository, so the canonical PoA / Shapley
intervals and adjusted p-values cited in Chapter 5 were not regenerable. This
module recomputes all four artefacts deterministically:

- it reuses the EXACT paired-bootstrap machinery of step_4_5 (np.random
  default_rng(42), 10000 resamples, per-group paired index draw) so the bootstrap
  means and the percentile distribution match the committed step_4_5 output;
- it adds the bias-correction (z0) and jackknife acceleration (a) of the BCa
  method on top of that distribution, for both the per-config means and the
  paired D-minus-A0 gaps;
- it adds the Holm-Bonferroni step-down across the six gap tests and the paired
  Cohen's d (mean of paired differences over the SD of paired differences).

No retraining: all statistics are read from the canonical 10-seed
results/ablation_results.csv.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from step_4_5_bootstrap_poa_ci import (  # noqa: E402  (path shim above)
    INPUT_CSV,
    METRICS,
    N_BOOTSTRAP,
    RANDOM_SEED,
    SCENARIO_GROUPS,
    _bootstrap_means,
    _extract_vector,
    _paired_indices,
)

ROBUSTNESS_DIR = BASE_DIR / "results" / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER = ROBUSTNESS_DIR / "step_4_5_bootstrap_ci_bca.csv"
OUT_GAP = ROBUSTNESS_DIR / "step_4_5_gap_ci_bca.csv"
OUT_HOLM = ROBUSTNESS_DIR / "step_4_5_holm_bonferroni.csv"
OUT_COHEN = ROBUSTNESS_DIR / "step_4_5_cohens_d.csv"

ALPHA = 0.05

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _bca_interval(theta_hat: float, boot: np.ndarray, jackknife: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) percentile interval.

    Args:
        theta_hat: Observed statistic on the full sample.
        boot: Bootstrap replicates of the statistic.
        jackknife: Leave-one-out replicates of the statistic.
        alpha: Two-sided significance level (0.05 gives a 95% interval).

    Returns:
        The (low, high) BCa interval endpoints.
    """
    if np.allclose(boot, boot[0]):
        return float(boot[0]), float(boot[0])
    prop = float(np.sum(boot < theta_hat)) / boot.size
    prop = min(max(prop, 1.0 / boot.size), 1.0 - 1.0 / boot.size)
    z0 = stats.norm.ppf(prop)
    diff = jackknife.mean() - jackknife
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    accel = float(np.sum(diff ** 3) / denom) if denom != 0.0 else 0.0
    z_lo, z_hi = stats.norm.ppf(alpha / 2.0), stats.norm.ppf(1.0 - alpha / 2.0)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    return float(np.percentile(boot, 100.0 * a1)), float(np.percentile(boot, 100.0 * a2))


def _jackknife_mean(vec: np.ndarray) -> np.ndarray:
    """Leave-one-out means of a vector."""
    return (vec.sum() - vec) / (vec.size - 1)


def _jackknife_paired_gap(v_d: np.ndarray, v_a0: np.ndarray) -> np.ndarray:
    """Leave-one-out paired gap mean(D) - mean(A0) over the same dropped index."""
    n = v_d.size
    return (v_d.sum() - v_d) / (n - 1) - (v_a0.sum() - v_a0) / (n - 1)


def _two_sided_pvalue(gap_boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: gap == 0, floored at 1/n_bootstrap."""
    frac_le = float(np.mean(gap_boot <= 0.0))
    frac_ge = float(np.mean(gap_boot >= 0.0))
    return max(2.0 * min(frac_le, frac_ge), 1.0 / gap_boot.size)


def _holm(pvalues: dict[tuple[str, str], float]) -> dict[tuple[str, str], dict]:
    """Holm-Bonferroni step-down over the family of gap p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[tuple[str, str], dict] = {}
    running = 0.0
    for rank, (key, p_raw) in enumerate(items):
        factor = m - rank
        running = max(running, min(factor * p_raw, 1.0))
        out[key] = {"p_raw": float(p_raw), "holm_factor": factor, "p_holm": float(running)}
    return out


def _cohens_d_paired(v_d: np.ndarray, v_a0: np.ndarray) -> float:
    """Paired Cohen's d (dz): mean of differences over SD of differences."""
    diff = v_d - v_a0
    sd = float(np.std(diff, ddof=1))
    return float(diff.mean() / sd) if sd > 0 else 0.0


def regenerate() -> dict:
    """Recompute per-group BCa, gap BCa, Holm and Cohen's d from ablation data.

    Returns:
        Mapping with keys per_group, gap, holm and cohens_d, each indexed by the
        relevant (config, group, metric) or (group, metric) tuple.
    """
    df = pd.read_csv(INPUT_CSV)
    per_group: dict = {}
    gap: dict = {}
    cohens: dict = {}
    p_raw: dict = {}

    for group, scen in SCENARIO_GROUPS.items():
        n_paired = min(
            df[(df["config"] == "A0") & (df["scenario"].isin(scen))].shape[0],
            df[(df["config"] == "D") & (df["scenario"].isin(scen))].shape[0],
        )
        indices = _paired_indices(n_paired, N_BOOTSTRAP, RANDOM_SEED)
        for metric in METRICS:
            v_a0 = _extract_vector(df, "A0", scen, metric)[:n_paired]
            v_d = _extract_vector(df, "D", scen, metric)[:n_paired]
            m_a0 = _bootstrap_means(v_a0, indices)
            m_d = _bootstrap_means(v_d, indices)
            for cfg, vec, m in (("A0", v_a0, m_a0), ("D", v_d, m_d)):
                lo, hi = _bca_interval(float(vec.mean()), m, _jackknife_mean(vec))
                per_group[(cfg, group, metric)] = {
                    "mean": round(float(vec.mean()), 4),
                    "ci95_lo_bca": lo,
                    "ci95_hi_bca": hi,
                    "std_bootstrap": round(float(m.std(ddof=1)), 4),
                    "n_obs": int(vec.size),
                }
            gap_boot = m_d - m_a0
            gap_hat = float(v_d.mean() - v_a0.mean())
            glo, ghi = _bca_interval(gap_hat, gap_boot, _jackknife_paired_gap(v_d, v_a0))
            sig = bool(glo > 0 or ghi < 0)
            gap[(group, metric)] = {
                "gap_mean": round(float(gap_boot.mean()), 4),
                "ci95_lo_bca": glo,
                "ci95_hi_bca": ghi,
                "significant_at_95_bca": sig,
                "std_bootstrap": round(float(gap_boot.std(ddof=1)), 4),
            }
            cohens[(group, metric)] = round(_cohens_d_paired(v_d, v_a0), 3)
            p_raw[(group, metric)] = _two_sided_pvalue(gap_boot)

    return {"per_group": per_group, "gap": gap, "holm": _holm(p_raw), "cohens_d": cohens}


def _write_csvs(res: dict) -> None:
    """Persist the four canonical robustness artefacts."""
    per_rows = [
        {"config": c, "scenario_group": g, "metric": me,
         "mean": v["mean"], "ci95_lo_bca": round(v["ci95_lo_bca"], 4),
         "ci95_hi_bca": round(v["ci95_hi_bca"], 4), "std_bootstrap": v["std_bootstrap"],
         "n_obs": v["n_obs"], "method": "BCa"}
        for (c, g, me), v in res["per_group"].items()
    ]
    pd.DataFrame(per_rows).to_csv(OUT_PER, index=False)

    gap_rows = [
        {"scenario_group": g, "metric": me, "gap_D_minus_A0_mean": v["gap_mean"],
         "ci95_lo_bca": round(v["ci95_lo_bca"], 4), "ci95_hi_bca": round(v["ci95_hi_bca"], 4),
         "significant_at_95_bca": v["significant_at_95_bca"], "std_bootstrap": v["std_bootstrap"],
         "n_bootstrap": N_BOOTSTRAP, "method": "BCa"}
        for (g, me), v in res["gap"].items()
    ]
    pd.DataFrame(gap_rows).to_csv(OUT_GAP, index=False)

    holm_rows = [
        {"arm": g, "metric": me, "gap_mean": res["gap"][(g, me)]["gap_mean"],
         "p_raw": round(hv["p_raw"], 4), "holm_factor": hv["holm_factor"],
         "p_holm": round(hv["p_holm"], 6), "sig_at_05_holm": hv["p_holm"] < 0.05}
        for (g, me), hv in res["holm"].items()
    ]
    pd.DataFrame(holm_rows).to_csv(OUT_HOLM, index=False)

    cohen_rows = [
        {"arm": g, "metric": me, "n_obs": res["per_group"][("D", g, me)]["n_obs"], "cohens_d": d}
        for (g, me), d in res["cohens_d"].items()
    ]
    pd.DataFrame(cohen_rows).to_csv(OUT_COHEN, index=False)


def main() -> None:
    """Regenerate the BCa, gap-BCa, Holm and Cohen artefacts and log a summary."""
    res = regenerate()
    _write_csvs(res)
    poa = res["per_group"][("A0", "9LC", "price_of_anarchy")]
    sh = res["gap"][("9LC", "shapley_fairness")]
    logger.info("Regenerated 4 robustness artefacts in %s", ROBUSTNESS_DIR.name)
    logger.info("=" * 70)
    logger.info(
        "  A0 9LC PoA %.4f BCa [%.4f, %.4f]; 9LC Shapley gap %.4f BCa [%.4f, %.4f] (Holm p=%.4g).",
        poa["mean"], poa["ci95_lo_bca"], poa["ci95_hi_bca"],
        sh["gap_mean"], sh["ci95_lo_bca"], sh["ci95_hi_bca"],
        res["holm"][("9LC", "shapley_fairness")]["p_holm"],
    )
    logger.info("  Cohen's d 9LC Shapley = %.3f.", res["cohens_d"][("9LC", "shapley_fairness")])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
