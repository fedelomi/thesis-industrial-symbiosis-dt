"""
Step 1.8 LC - Climate Sensitivity (3 T_amb scenarios)
======================================================
Phase 1 LC robustness layer (additive, post-baseline)

Evaluates RC model robustness across three ambient temperature climates
without re-running the synthetic-profile pipeline. Each climate is
evaluated analytically at the steady state with IT_load = 0.80 baseline:

    cold  T_amb =  5 degC  (nordic / Helsinki-class)
    mid   T_amb = 15 degC  (central EU baseline already used in canonical run)
    hot   T_amb = 25 degC  (mediterranean / Milan-class summer)

For each (scenario, climate) we compute:
    Q_available_mean_kW
    Exergy_DT_mean_kW   (exergy uses T0 = T_amb + 273.15 K as reference)
    t_availability      (fraction of time Q_available > 0.10 * P_rated)
    ERF_estimated       (= CAPTURE_FRACTION_LC ** 2 per audit P1-5)

Model invariance note: in the current LC formulation
    Q_available = alpha_i * P_rated * IT_load
T_amb does not enter the Q equation. T_CDU_supply is regulated to 38 degC
by the chiller, so T_supply (theta_ss) is also invariant. The only
T_amb-coupled output is Exergy_DT via the reference temperature T0.
This is documented in the output for thesis defence.

Output:
    results/step_1_8_climate_sensitivity.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from step_1_1_2_rc_model_lc import CAPTURE_FRACTION_LC, LC_SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = RESULTS_DIR / "step_1_8_climate_sensitivity.csv"

CLIMATES: list[tuple[str, float]] = [
    ("cold", 5.0),
    ("mid", 15.0),
    ("hot", 25.0),
]
BASELINE_IT_LOAD = 0.80
BASELINE_CLIMATE_LABEL = "mid"
Q_AVAIL_THRESHOLD_FRAC = 0.10
T_AVAIL_MIN = 0.85


def _exergy_ss(theta_ss_K: float, q_available_W: float, t_amb_C: float) -> float:
    if q_available_W <= 0:
        return 0.0
    t0_K = float(t_amb_C) + 273.15
    h0 = PropsSI("H", "T", t0_K, "P", 101325.0, "Water")
    s0 = PropsSI("S", "T", t0_K, "P", 101325.0, "Water")
    t_safe = float(np.clip(theta_ss_K, 273.17, 368.15))
    h = PropsSI("H", "T", t_safe, "P", 101325.0, "Water")
    s = PropsSI("S", "T", t_safe, "P", 101325.0, "Water")
    dh = h - h0
    if not np.isfinite(dh) or abs(dh) < 1.0:
        return 0.0
    m_dot = q_available_W / dh
    return float(m_dot * (dh - t0_K * (s - s0)))


def _evaluate(scenario_name: str, params: dict, t_amb_C: float) -> dict:
    sum_alpha_u = params["alpha_i"] * params["rated_power_W"] * BASELINE_IT_LOAD
    q_available = max(sum_alpha_u, 0.0)
    theta_ss = params["T_cdu_supply"] + params["R_d"] * sum_alpha_u
    exergy = _exergy_ss(theta_ss, q_available, t_amb_C)
    q_min = Q_AVAIL_THRESHOLD_FRAC * params["rated_power_W"]
    t_availability = 1.0 if q_available > q_min else 0.0
    erf_estimated = CAPTURE_FRACTION_LC ** 2
    return {
        "Q_available_mean_kW": round(q_available / 1e3, 2),
        "Exergy_DT_mean_kW": round(exergy / 1e3, 3),
        "t_availability": round(t_availability, 3),
        "ERF_estimated": round(erf_estimated, 4),
        "T_supply_ss_C": round(theta_ss - 273.15, 2),
    }


def main() -> None:
    logger.info(
        "Step 1.8 starting: climate sensitivity across %d T_amb scenarios at IT_load=%.2f",
        len(CLIMATES), BASELINE_IT_LOAD,
    )

    rows: list[dict] = []
    baseline_cache: dict[str, dict] = {}
    for scenario_name, params in LC_SCENARIOS.items():
        for climate_label, t_amb_C in CLIMATES:
            metrics = _evaluate(scenario_name, params, t_amb_C)
            if climate_label == BASELINE_CLIMATE_LABEL:
                baseline_cache[scenario_name] = metrics
            rows.append({
                "scenario_lc": scenario_name,
                "climate_label": climate_label,
                "T_amb_C": t_amb_C,
                **metrics,
            })

    for r in rows:
        base = baseline_cache[r["scenario_lc"]]
        for key, delta_col in [
            ("Q_available_mean_kW", "delta_Q_vs_baseline_pct"),
            ("Exergy_DT_mean_kW", "delta_Exergy_vs_baseline_pct"),
            ("t_availability", "delta_t_avail_vs_baseline_pct"),
        ]:
            base_v = base[key]
            cur = r[key]
            if base_v == 0:
                r[delta_col] = 0.0
            else:
                r[delta_col] = round((cur - base_v) / base_v * 100.0, 2)
        if r["climate_label"] == BASELINE_CLIMATE_LABEL:
            r["delta_Q_vs_baseline_pct"] = 0.0
            r["delta_Exergy_vs_baseline_pct"] = 0.0
            r["delta_t_avail_vs_baseline_pct"] = 0.0

    out_cols = [
        "scenario_lc", "climate_label", "T_amb_C",
        "Q_available_mean_kW", "Exergy_DT_mean_kW", "t_availability",
        "ERF_estimated", "T_supply_ss_C",
        "delta_Q_vs_baseline_pct",
        "delta_Exergy_vs_baseline_pct",
        "delta_t_avail_vs_baseline_pct",
    ]
    out = pd.DataFrame(rows)[out_cols]
    out.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(out))

    cold = out[out["climate_label"] == "cold"]
    hot = out[out["climate_label"] == "hot"]
    avg_delta_q_cold = float(cold["delta_Q_vs_baseline_pct"].mean())
    avg_delta_q_hot = float(hot["delta_Q_vs_baseline_pct"].mean())
    avg_delta_e_cold = float(cold["delta_Exergy_vs_baseline_pct"].mean())
    avg_delta_e_hot = float(hot["delta_Exergy_vs_baseline_pct"].mean())
    t_min = float(out["t_availability"].min())
    t_max = float(out["t_availability"].max())

    logger.info("=" * 70)
    logger.info(
        "  Climate sensitivity Q_available: cold %+.1f%%, hot %+.1f%% vs baseline; "
        "t_availability range [%.2f, %.2f]",
        avg_delta_q_cold, avg_delta_q_hot, t_min, t_max,
    )
    logger.info(
        "  Climate sensitivity Exergy_DT:   cold %+.1f%%, hot %+.1f%% vs baseline "
        "(driven by T0 reference shift)",
        avg_delta_e_cold, avg_delta_e_hot,
    )
    logger.info(
        "  Note: Q_available is T_amb-invariant in this LC formulation "
        "(Q = alpha_i * P * IT_load, T_amb does not enter). Only Exergy_DT "
        "varies via the exergy reference T0 = T_amb + 273.15 K."
    )
    logger.info("=" * 70)

    if t_min < T_AVAIL_MIN:
        logger.warning(
            "t_availability dropped below %.2f (min observed %.2f); model loses "
            "applicability outside the studied climate range",
            T_AVAIL_MIN, t_min,
        )


if __name__ == "__main__":
    main()
