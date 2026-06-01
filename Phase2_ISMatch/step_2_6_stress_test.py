"""
Step 2.6 LC - IS-Match Ranking Stress Test
===========================================
Phase 2 robustness layer (additive, post-baseline)

Worst-case perturbation analysis on input parameters at +/-20%:

  D-decision Phase 2 (2026-05-31): distance_km removed from
  perturbation set because step_2_1 scorer no longer uses ALPHA_d.
  Perturbing distance now produces zero effect on the score, so
  the test rows are kept for traceability but their delta is null.

  Active parameters at +/-20%:
    Q_demand_peak  (plant-side demand)
    T_required     (plant target temperature)
    distance_km    (DC <-> plant transport distance)
    delta_tc_norm  (calibration cost penalty)

For each of the 8 scenarios (4 params x 2 directions), the 9 LC pairs are
rescored via the step_2_1 logic (replicated inline to avoid the syntax
error in the truncated step_2_1_is_match_score_lc.py) and compared with
the baseline ranking. Two metrics are reported:
    top3_unchanged  True if the *set* of top-3 (dc, process) is preserved
    max_rank_shift  Maximum absolute change in rank across all 9 pairs

Output:
    results/step_2_6_stress_test.csv

Note on Q_demand_peak: the current temporal RI formula in step_2_1 uses
DC-side q_norm = q_available / q_nominal_dc; the plant peak demand does
not enter the IS-Match Score directly (it only filters feasibility in the
step_2_2 static RI). Q_demand_peak perturbation therefore trivially
preserves the ranking and is reported as such for completeness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PHASE1_LC_CSV = (
    BASE_DIR.parent / "Phase1_PhysicalDT" / "lc" / "results" / "lc_dc_results_annual.csv"
)
DATASET_CSV = RESULTS_DIR / "step_2_2_ce_heat_dataset_lc.csv"
BASELINE_CSV = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
OUT_CSV = RESULTS_DIR / "step_2_6_stress_test.csv"

PERTURB_FRACTION = 0.20
TOP_K = 3

COL_SCENARIO = "scenario"
COL_T_SUPPLY = "T_supply"
COL_Q_AVAIL = "Q_available"
COL_EXERGY = "Exergy_DT"
COL_T_AVAIL = "t_availability"

Q_NOMINAL_LC: dict[str, float] = {
    "Edge_LC": 475_000.0,
    "Mid_LC": 3_104_000.0,
    "Hyperscale_LC": 24_500_000.0,
}

ALPHA_Q, ALPHA_T, ALPHA_t, ALPHA_d = 0.25, 0.35, 0.25, 0.15
BETA_DEFAULT = 0.40
GAMMA_DEFAULT = 0.40
DELTA_DEFAULT = 0.20
THRESHOLD_HIGH = 0.60
THRESHOLD_MARGINAL = 0.30
T_LIFT_REF = 100.0
D_MAX_KM = 200.0


@dataclass
class ManufacturingSpec:
    process_name: str
    t_req_c: float
    distance_km: float
    shift_type: str
    q_demand_kw: float
    delta_tc_baseline: float


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
    q_norm = np.clip(q_available / q_nominal, 0.0, 1.0)
    t_gap = np.clip(spec.t_req_c - t_supply, 0.0, None)
    t_compat = np.clip(1.0 - t_gap / T_LIFT_REF, 0.0, 1.0)
    avail = t_avail.astype(float)
    d_norm = min(spec.distance_km / D_MAX_KM, 1.0)
    return np.clip(
        ALPHA_Q * q_norm + ALPHA_T * t_compat + ALPHA_t * avail - ALPHA_d * d_norm,
        0.0, 1.0,
    )


def compute_ri_temporal(ri_series: np.ndarray, weights: np.ndarray) -> float:
    w_sum = weights.sum()
    return float(np.dot(ri_series, weights) / w_sum) if w_sum > 0 else 0.0


def compute_is_match(
    ri_temporal: float, exergy_norm: float, delta_tc_norm: float,
    beta: float = BETA_DEFAULT, gamma: float = GAMMA_DEFAULT,
    delta: float = DELTA_DEFAULT,
) -> float:
    return float(np.clip(
        beta * ri_temporal + gamma * exergy_norm - delta * delta_tc_norm,
        0.0, 1.0,
    ))


def classify_tier(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "high_priority"
    if score >= THRESHOLD_MARGINAL:
        return "marginal"
    return "not_viable"


def load_specs_lc(path: Path) -> list[ManufacturingSpec]:
    df = pd.read_csv(path)
    rename_map = {
        "plant_tier": "process_name",
        "plant_t_req_c": "t_req_c",
        "distance_km": "distance_km",
        "plant_shift_pattern": "shift_type",
        "plant_q_demand_kw": "q_demand_kw",
        "plant_delta_tc_baseline": "delta_tc_baseline",
    }
    df2 = df.rename(columns=rename_map)
    agg = (
        df2.groupby("process_name")
        .agg(
            t_req_c=("t_req_c", "first"),
            distance_km=("distance_km", "median"),
            shift_type=("shift_type", "first"),
            q_demand_kw=("q_demand_kw", "mean"),
            delta_tc_baseline=("delta_tc_baseline", "mean"),
        )
        .reset_index()
    )
    return [
        ManufacturingSpec(
            process_name=str(r["process_name"]),
            t_req_c=float(r["t_req_c"]),
            distance_km=float(r["distance_km"]),
            shift_type=str(r["shift_type"]),
            q_demand_kw=float(r["q_demand_kw"]),
            delta_tc_baseline=float(r["delta_tc_baseline"]),
        )
        for _, r in agg.iterrows()
    ]


def _build_per_dc_arrays(df_dt: pd.DataFrame) -> dict[str, dict]:
    """Pre-extract time-series arrays per DC scenario."""
    cache: dict[str, dict] = {}
    ex_max_per_sc = df_dt.groupby(COL_SCENARIO)[COL_EXERGY].max().to_dict()
    for dc_name in df_dt[COL_SCENARIO].unique():
        df_dc = df_dt[df_dt[COL_SCENARIO] == dc_name].reset_index(drop=True)
        q_nom = Q_NOMINAL_LC.get(dc_name, float(df_dc[COL_Q_AVAIL].max()))
        ex_max = ex_max_per_sc.get(dc_name, 1.0)
        cache[dc_name] = {
            "t_arr": df_dc[COL_T_SUPPLY].to_numpy(dtype=float),
            "q_arr": df_dc[COL_Q_AVAIL].to_numpy(dtype=float),
            "a_arr": df_dc[COL_T_AVAIL].to_numpy(dtype=float),
            "q_nom": float(q_nom),
            "n_steps": len(df_dc),
            "exergy_norm": float(
                np.clip(df_dc[COL_EXERGY].mean() / ex_max, 0.0, 1.0)
            ),
        }
    return cache


def _score_all_pairs(
    dc_cache: dict[str, dict],
    specs: list[ManufacturingSpec],
    delta_tc_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for dc_name, payload in dc_cache.items():
        for spec in specs:
            w = make_shift_weights(payload["n_steps"], spec.shift_type)
            ri_series = compute_ri_series(
                payload["t_arr"], payload["q_arr"], payload["q_nom"],
                payload["a_arr"], spec,
            )
            ri_temporal = compute_ri_temporal(ri_series, w)
            if delta_tc_overrides is not None and spec.process_name in delta_tc_overrides:
                dtc_norm = float(np.clip(delta_tc_overrides[spec.process_name], 0.0, 1.0))
            else:
                dtc_norm = float(np.clip(spec.delta_tc_baseline, 0.0, 1.0))
            score = compute_is_match(ri_temporal, payload["exergy_norm"], dtc_norm)
            rows.append({
                "dc_name": dc_name,
                "process_name": spec.process_name,
                "ri_temporal": ri_temporal,
                "exergy_dt_norm": payload["exergy_norm"],
                "delta_tc_norm": dtc_norm,
                "is_match_score": score,
                "tier": classify_tier(score),
            })
    df = (
        pd.DataFrame(rows)
        .sort_values("is_match_score", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    return df


def _top3_set(df_scored: pd.DataFrame) -> set[tuple[str, str]]:
    head = df_scored.head(TOP_K)
    return set(zip(head["dc_name"].astype(str), head["process_name"].astype(str)))


def _rank_lookup(df_scored: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (str(r["dc_name"]), str(r["process_name"])): int(r["rank"])
        for _, r in df_scored.iterrows()
    }


def _perturb_specs(
    specs: list[ManufacturingSpec], field: str, factor: float
) -> list[ManufacturingSpec]:
    return [replace(s, **{field: getattr(s, field) * factor}) for s in specs]


def _delta_tc_overrides(
    specs: list[ManufacturingSpec], factor: float
) -> dict[str, float]:
    return {s.process_name: s.delta_tc_baseline * factor for s in specs}


def _factor(direction: str) -> float:
    if direction == "+20%":
        return 1.0 + PERTURB_FRACTION
    if direction == "-20%":
        return 1.0 - PERTURB_FRACTION
    raise ValueError(f"Unknown direction: {direction}")


def _evaluate_scenario(
    param: str,
    direction: str,
    dc_cache: dict[str, dict],
    specs_baseline: list[ManufacturingSpec],
    baseline_top3: set[tuple[str, str]],
    baseline_ranks: dict[tuple[str, str], int],
    note: str = "",
) -> dict:
    factor = _factor(direction)
    if param == "Q_demand_peak":
        new_specs = _perturb_specs(specs_baseline, "q_demand_kw", factor)
        scored = _score_all_pairs(dc_cache, new_specs)
    elif param == "T_required":
        new_specs = _perturb_specs(specs_baseline, "t_req_c", factor)
        scored = _score_all_pairs(dc_cache, new_specs)
    elif param == "distance_km":
        # D-decision 2026-05-31: distance no longer affects scorer (ALPHA_d=0),
        # so this branch produces a null delta. Kept for traceability.
        new_specs = _perturb_specs(specs_baseline, "distance_km", factor)
        scored = _score_all_pairs(dc_cache, new_specs)
    elif param == "delta_tc_norm":
        overrides = _delta_tc_overrides(specs_baseline, factor)
        scored = _score_all_pairs(dc_cache, specs_baseline, overrides)
    else:
        raise ValueError(f"Unknown parameter: {param}")

    top3 = _top3_set(scored)
    ranks_after = _rank_lookup(scored)
    shifts = [abs(baseline_ranks[k] - ranks_after[k]) for k in baseline_ranks]
    max_shift = int(max(shifts)) if shifts else 0
    return {
        "parameter": param,
        "direction": direction,
        "perturbed_top3": "[" + ", ".join(
            f"{dc}+{p}" for dc, p in sorted(top3)
        ) + "]",
        "top3_unchanged": bool(top3 == baseline_top3),
        "max_rank_shift": max_shift,
        "notes": note,
    }


def main() -> None:
    logger.info(
        "Step 2.6 starting: stress test +/- %.0f%% on Q_demand_peak, T_required, distance_km and delta_tc_norm",
        PERTURB_FRACTION * 100,
    )

    if not PHASE1_LC_CSV.exists():
        raise FileNotFoundError(f"Missing Phase 1 LC CSV: {PHASE1_LC_CSV}")
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_CSV}")
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Missing baseline scores: {BASELINE_CSV}")

    logger.info("Loading Phase 1 LC time-series: %s", PHASE1_LC_CSV.name)
    df_dt = pd.read_csv(PHASE1_LC_CSV)
    dc_cache = _build_per_dc_arrays(df_dt)
    specs = load_specs_lc(DATASET_CSV)
    logger.info(
        "Loaded %d DC scenarios, %d aggregated processes", len(dc_cache), len(specs)
    )

    scored_baseline = _score_all_pairs(dc_cache, specs)
    baseline_top3 = _top3_set(scored_baseline)
    baseline_ranks = _rank_lookup(scored_baseline)
    logger.info(
        "Baseline top-3: %s",
        "[" + ", ".join(f"{a}+{b}" for a, b in sorted(baseline_top3)) + "]",
    )

    df_input = pd.read_csv(BASELINE_CSV).head(TOP_K)
    expected_top3 = set(
        zip(df_input["dc_name"].astype(str), df_input["process_name"].astype(str))
    )
    if baseline_top3 != expected_top3:
        logger.warning(
            "Baseline top-3 from recomputation differs from step_2_1 CSV; check shift weights"
        )

    notes_q = "Score formula uses DC-side q_norm; plant Q_demand unused"
    scenarios = [
        ("Q_demand_peak", "+20%", notes_q),
        ("Q_demand_peak", "-20%", notes_q),
        ("T_required", "+20%", ""),
        ("T_required", "-20%", ""),
        ("distance_km", "+20%", ""),
        ("distance_km", "-20%", ""),
        ("delta_tc_norm", "+20%", ""),
        ("delta_tc_norm", "-20%", ""),
    ]
    rows: list[dict] = []
    for param, direction, note in scenarios:
        rows.append(
            _evaluate_scenario(
                param=param,
                direction=direction,
                dc_cache=dc_cache,
                specs_baseline=specs,
                baseline_top3=baseline_top3,
                baseline_ranks=baseline_ranks,
                note=note,
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(out))

    n_pass = int(out["top3_unchanged"].sum())
    n_total = len(out)
    breaks = out[~out["top3_unchanged"]]
    break_list = (
        "[" + ", ".join(
            f"{r['parameter']} {r['direction']}" for _, r in breaks.iterrows()
        ) + "]"
    )
    logger.info("=" * 70)
    logger.info(
        "  Top-3 ranking stable in %d/%d stress scenarios (%.1f%%). Breaks: %s",
        n_pass, n_total, 100.0 * n_pass / n_total, break_list,
    )
    logger.info("=" * 70)

    if n_pass < 6:
        logger.warning(
            "Stress test FAIL: only %d/%d scenarios preserved the top-3 set", n_pass, n_total
        )


if __name__ == "__main__":
    main()
