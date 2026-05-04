"""
Step 2.1 LC — IS-Match Score  [LIQUID COOLING]
===============================================
Phase 2: IS-Match Score (OS2) · Thesis Cap. 3.3.3 + 4.5

IS-Match Score = β · RI_temporal + γ · Exergy_DT_norm − δ · ΔTC_norm

RI(t) = α_Q·Q_norm(t) + α_T·T_compat(t) + α_t·Avail(t) − α_d·d_norm  [WHER D3]
RI_temporal = Σ_t [RI(t)·w(t)] / Σ_t w(t)   (manufacturing-shift weighted)

Exergy_DT_norm: per-scenario normalisation (mean / scenario_max).
  LC note: all LC scenarios have Exergy_DT_norm ≈ 0.75 (same load profile shape),
  reflecting high exergy quality fraction at ~48°C CDU return temperature.
  This is scale-invariant and meaningful for cross-DC comparison.

Q_NOMINAL in Watts (consistent with Phase 1 LC CSV where Q_available is in W):
  Edge_LC:       475,000 W  =  475 kW
  Mid_LC:      3,104,000 W  =  3.1 MW
  Hyperscale: 24,500,000 W  = 24.5 MW

Expected result: pre-upgrade IS-Match Scores LOW-MARGINAL for MidT/HighT pairs
(LC supply 46-50°C < industrial T requirement 90-130°C → HP/CO₂ HTHP needed).
LowT_60C pairs may reach MARGINAL (0.30-0.55) due to reduced temperature lift.
Physics is correct: IS-Match Score = raw thermodynamic compatibility pre-upgrade.

Input:  ../Phase1/LC-Opt/lc_dc_results_annual.csv
        ce_heat_inspired_dataset_lc.csv  (Step 2.2 LC)
Output: step_2_1_is_match_scores_lc.csv
        step_2_1_sensitivity_lc.csv
"""

from __future__ import annotations
import itertools
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent
# Phase 1 LC output — look in LC-Opt subfolder relative to repo root
# Phase 1 LC CSV — try canonical location, then sandbox sibling path
def _find_lc_csv() -> Path:
    candidates = [
        BASE_DIR.parent / 'Phase1' / 'LC-Opt' / 'lc_dc_results_annual.csv',  # user machine
        BASE_DIR.parent / 'LC-Opt' / 'lc_dc_results_annual.csv',              # sandbox mount
        BASE_DIR / 'lc_dc_results_annual.csv',                                 # same folder
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # will raise FileNotFoundError with correct message
PHASE1_LC_CSV = _find_lc_csv()
DATASET_CSV   = BASE_DIR / "ce_heat_inspired_dataset_lc.csv"
SCORES_CSV    = BASE_DIR / "step_2_1_is_match_scores_lc.csv"
SENS_CSV      = BASE_DIR / "step_2_1_sensitivity_lc.csv"

COL_SCENARIO = "scenario"
COL_T_SUPPLY = "T_supply"
COL_Q_AVAIL  = "Q_available"
COL_EXERGY   = "Exergy_DT"
COL_T_AVAIL  = "t_availability"

# Q_NOMINAL in Watts — must match Q_available column units in Phase 1 LC CSV
Q_NOMINAL_LC: dict[str, float] = {
    "Edge_LC":       475_000.0,    # W = 475 kW
    "Mid_LC":      3_104_000.0,    # W = 3.1 MW
    "Hyperscale_LC": 24_500_000.0, # W = 24.5 MW
}

ALPHA_Q, ALPHA_T, ALPHA_t, ALPHA_d = 0.25, 0.35, 0.25, 0.15
BETA_DEFAULT  = 0.40
GAMMA_DEFAULT = 0.40
DELTA_DEFAULT = 0.20

THRESHOLD_HIGH     = 0.60
THRESHOLD_MARGINAL = 0.30
T_LIFT_REF         = 100.0
D_MAX_KM           = 5.0


@dataclass
class ManufacturingSpec:
    process_name: str
    t_req_c: float
    distance_km: float
    shift_type: str
    q_demand_kw: float
    delta_tc_baseline: float


@dataclass
class ISMatchResult:
    dc_name: str
    process_name: str
    ri_temporal: float
    exergy_dt_norm: float
    delta_tc_norm: float
    is_match_score: float
    tier: str
    beta: float
    gamma: float
    delta: float
    n_weighted_steps: int
    exergy_norm_method: str


def make_shift_weights(n_steps: int, shift_type: str) -> np.ndarray:
    if shift_type == "continuous":
        return np.ones(n_steps, dtype=float)
    w = np.zeros(n_steps, dtype=float)
    spd = 96
    start_s, end_s = (24, 88) if shift_type == "2shift" else (32, 64)
    for i in range(n_steps):
        if (i // spd) % 7 < 5 and start_s <= (i % spd) < end_s:
            w[i] = 1.0
    return w


def compute_ri_series(
    t_supply: np.ndarray, q_available: np.ndarray, q_nominal: float,
    t_avail: np.ndarray, spec: ManufacturingSpec,
) -> np.ndarray:
    q_norm   = np.clip(q_available / q_nominal, 0.0, 1.0)
    t_gap    = np.clip(spec.t_req_c - t_supply, 0.0, None)
    t_compat = np.clip(1.0 - t_gap / T_LIFT_REF, 0.0, 1.0)
    avail    = t_avail.astype(float)
    d_norm   = min(spec.distance_km / D_MAX_KM, 1.0)
    return np.clip(ALPHA_Q*q_norm + ALPHA_T*t_compat + ALPHA_t*avail - ALPHA_d*d_norm,
                   0.0, 1.0)


def compute_ri_temporal(ri_series: np.ndarray, weights: np.ndarray) -> float:
    w_sum = weights.sum()
    return float(np.dot(ri_series, weights) / w_sum) if w_sum > 0 else 0.0


def compute_is_match(
    ri_temporal: float, exergy_norm: float, delta_tc_norm: float,
    beta: float = BETA_DEFAULT, gamma: float = GAMMA_DEFAULT,
    delta: float = DELTA_DEFAULT,
) -> float:
    return float(np.clip(beta*ri_temporal + gamma*exergy_norm - delta*delta_tc_norm,
                         0.0, 1.0))


def classify_tier(score: float) -> str:
    if score >= THRESHOLD_HIGH:     return "high_priority"
    if score >= THRESHOLD_MARGINAL: return "marginal"
    return "not_viable"


def load_specs_lc(path: Path) -> list[ManufacturingSpec]:
    """Load specs from LC dataset; fall back to inline defaults if not found."""
    if not path.exists():
        logger.warning(f"LC dataset not found: {path} → run step_2_2_dataset_builder_lc.py first")
        # Inline defaults — same plants with LC-calibrated delta_tc_baseline
        return [
            ManufacturingSpec("LowT_60C",   60.0,  1.5, "continuous", 450.0,  0.25),
            ManufacturingSpec("MidT_90C",   90.0,  2.5, "2shift",     800.0,  0.42),
            ManufacturingSpec("HighT_130C", 130.0, 3.5, "1shift",     1000.0, 0.55),
        ]
    df = pd.read_csv(path)
    agg = (df.groupby("process_name")
           .agg(t_req_c=("t_req_c","first"), distance_km=("distance_km","median"),
                shift_type=("shift_type","first"), q_demand_kw=("q_demand_kw","mean"),
                delta_tc_baseline=("delta_tc_baseline","mean"))
           .reset_index())
    return [ManufacturingSpec(
        str(r["process_name"]), float(r["t_req_c"]), float(r["distance_km"]),
        str(r["shift_type"]),   float(r["q_demand_kw"]), float(r["delta_tc_baseline"]),
    ) for _, r in agg.iterrows()]


def run_sensitivity(
    cache: dict[str, dict[str, dict]],
    beta_vals:  list[float] = [0.2, 0.3, 0.4, 0.5],
    gamma_vals: list[float] = [0.2, 0.3, 0.4, 0.5],
) -> pd.DataFrame:
    rows = []
    for beta, gamma in itertools.product(beta_vals, gamma_vals):
        delta = round(1.0 - beta - gamma, 4)
        if delta < 0.0:
            continue
        for dc, proc_dict in cache.items():
            for proc, comp in proc_dict.items():
                score = compute_is_match(comp["ri_temporal"], comp["exergy_norm"],
                                         comp["delta_tc_norm"], beta, gamma, delta)
                rows.append({"beta": beta, "gamma": gamma, "delta": delta,
                              "dc_name": dc, "process_name": proc,
                              "is_match_score": round(score, 4),
                              "tier": classify_tier(score)})
    return pd.DataFrame(rows)


def main() -> None:
    # ── 1. Load Phase 1 LC CSV ─────────────────────────────────────────────────
    if not PHASE1_LC_CSV.exists():
        raise FileNotFoundError(
            f"Phase 1 LC CSV not found: {PHASE1_LC_CSV}\n"
            "Expected at: ../Phase1/LC-Opt/lc_dc_results_annual.csv"
        )
    logger.info(f"Loading: {PHASE1_LC_CSV}")
    df_dt = pd.read_csv(PHASE1_LC_CSV)
    for col in [COL_SCENARIO, COL_T_SUPPLY, COL_Q_AVAIL, COL_EXERGY, COL_T_AVAIL]:
        if col not in df_dt.columns:
            raise ValueError(f"Column '{col}' missing. Available: {list(df_dt.columns)}")

    # ── 2. Per-scenario Exergy_DT max ──────────────────────────────────────────
    exergy_max_per_sc: dict[str, float] = (
        df_dt.groupby(COL_SCENARIO)[COL_EXERGY].max().to_dict()
    )
    logger.info("Exergy_DT max per scenario: "
                + "  ".join(f"{k}={v:.0f}W" for k, v in exergy_max_per_sc.items()))

    # ── 3. Load LC manufacturing specs ────────────────────────────────────────
    specs = load_specs_lc(DATASET_CSV)

    # ── 4. Compute IS-Match Score ─────────────────────────────────────────────
    results: list[ISMatchResult] = []
    cache:   dict[str, dict[str, dict]] = {}

    for dc_name in df_dt[COL_SCENARIO].unique():
        df_dc    = df_dt[df_dt[COL_SCENARIO] == dc_name].reset_index(drop=True)
        q_nom    = Q_NOMINAL_LC.get(dc_name, float(df_dc[COL_Q_AVAIL].max()))
        n_steps  = len(df_dc)
        ex_max   = exergy_max_per_sc.get(dc_name, 1.0)

        exergy_mean = float(df_dc[COL_EXERGY].mean())
        exergy_norm = float(np.clip(exergy_mean / ex_max, 0.0, 1.0))

        t_arr = df_dc[COL_T_SUPPLY].to_numpy(dtype=float)
        q_arr = df_dc[COL_Q_AVAIL].to_numpy(dtype=float)
        a_arr = df_dc[COL_T_AVAIL].to_numpy(dtype=float)

        cache[dc_name] = {}
        for spec in specs:
            w           = make_shift_weights(n_steps, spec.shift_type)
            ri_series   = compute_ri_series(t_arr, q_arr, q_nom, a_arr, spec)
            ri_temporal = compute_ri_temporal(ri_series, w)
            dtc_norm    = float(np.clip(spec.delta_tc_baseline, 0.0, 1.0))
            score       = compute_is_match(ri_temporal, exergy_norm, dtc_norm)
            tier        = classify_tier(score)

            results.append(ISMatchResult(
                dc_name=dc_name, process_name=spec.process_name,
                ri_temporal=round(ri_temporal, 4),
                exergy_dt_norm=round(exergy_norm, 4),
                delta_tc_norm=round(dtc_norm, 4),
                is_match_score=round(score, 4),
                tier=tier,
                beta=BETA_DEFAULT, gamma=GAMMA_DEFAULT, delta=DELTA_DEFAULT,
                n_weighted_steps=int(w.sum()),
                exergy_norm_method="per_scenario",
            ))
            cache[dc_name][spec.process_name] = {
                "ri_temporal": ri_temporal, "exergy_norm": exergy_norm,
                "delta_tc_norm": dtc_norm,
            }

    # ── 5. Ranking & output ───────────────────────────────────────────────────
    df_scores = (pd.DataFrame([asdict(r) for r in results])
                 .sort_values("is_match_score", ascending=False)
                 .reset_index(drop=True))
    df_scores.insert(0, "rank", df_scores.index + 1)

    logger.info("\n" + "="*70)
    logger.info(f"  IS-Match LC Ranking  (β={BETA_DEFAULT} γ={GAMMA_DEFAULT} δ={DELTA_DEFAULT}  "
                "Exergy_norm=per_scenario)")
    logger.info("="*70)
    for _, row in df_scores.iterrows():
        logger.info(f"  #{int(row['rank']):2d}  {row['dc_name']:14s} × {row['process_name']:14s}"
                    f"  RI={row['ri_temporal']:.3f}  Ex={row['exergy_dt_norm']:.3f}"
                    f"  Score={row['is_match_score']:.4f}  [{row['tier']}]")
    tier_counts = df_scores["tier"].value_counts()
    logger.info("-"*70)
    for t, c in tier_counts.items():
        logger.info(f"  {t}: {c}")
    logger.info("="*70)

    logger.info(
        "\n  THESIS INTERPRETATION (Cap. 5.2):\n"
        "  LC supply T=46-50°C → LowT_60C pairs: reduced lift (HP ~12-14°C).\n"
        "  MidT/HighT still require substantial upgrade; scores pre-upgrade LOW-MARGINAL.\n"
        "  Exergy_DT_norm ≈ 0.75 uniform across LC DCs (same load profile shape).\n"
        "  Differentiation by IS-Match vs RI_temporal captured in Step 2.3.\n"
    )

    df_scores.to_csv(SCORES_CSV, index=False)
    logger.info(f"  Scores → {SCORES_CSV}  ({len(df_scores)} rows)")

    # ── 6. Sensitivity ────────────────────────────────────────────────────────
    df_sens = run_sensitivity(cache)
    df_sens.to_csv(SENS_CSV, index=False)
    logger.info(f"  Sensitivity → {SENS_CSV}  ({len(df_sens)} rows)")

    unstable = (df_sens.groupby(["dc_name","process_name"])["tier"]
                .nunique().rename("n_tiers").reset_index()
                .query("n_tiers > 1"))
    if unstable.empty:
        logger.info("  Tier classification STABLE across all weight combinations.")
    else:
        logger.info(f"  {len(unstable)} pairs change tier across weight grid:")
        for _, row in unstable.iterrows():
            logger.info(f"    {row['dc_name']} × {row['process_name']}")


if __name__ == "__main__":
    main()
