"""
Step 1.5 LC - Sobol Sensitivity on RC Model Parameters
========================================================
Phase 1 LC robustness layer (additive, post-baseline)

Quantifies how much each of the 5 RC model parameters contributes to the
variance of the two key annual aggregates per LC scenario:
    Q_available_mean   [W]
    Exergy_DT_mean     [W]

Parameters (relative ranges centred on the baseline in
step_1_1_2_rc_model_lc.py LC_SCENARIOS):
    R_d_mult              [0.80, 1.20]  multiplier on baseline R_d
    C_d_mult              [0.80, 1.20]  multiplier on baseline C_d
    IT_load_mean          [0.65, 0.95]  constant IT load fraction
    T_amb_mean            [10.0, 25.0]  ambient reference [degC]
    CAPTURE_FRACTION_LC   [0.80, 1.00]  capture fraction multiplier

Fast-evaluation strategy: at constant IT load the RC tau is ~ 1-2 s and
the system reaches steady state within one 15-min step, so we do not
re-run the 35040-step time-domain loop. Instead we compute the
steady-state theta directly and evaluate exergy once per sample via
CoolProp. This makes 5120-sample Sobol analyses run in ~ 30 s total.

Outputs:
    results/step_1_5_sobol_rc_params.csv
        scenario, output_metric, parameter, S1, S1_conf, ST, ST_conf

Observations to expect:
- Q_available_mean depends only on (alpha_i * P * IT_load) in this LC
  formulation, so S_IT_load ~ 1.0 and the other four indices ~ 0.
- Exergy_DT_mean additionally depends on theta_ss (via R_d) and on the
  exergy reference T0 (via T_amb_mean), spreading the indices.
- CAPTURE_FRACTION_LC does NOT enter Q_available_mean or Exergy_DT_mean
  in the current model (only Q_negotiated), so its S1 stays near zero;
  this is the right answer, not a bug.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from SALib.analyze import sobol
from SALib.sample import saltelli

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from step_1_1_2_rc_model_lc import LC_SCENARIOS, T_CDU_SUPPLY_K

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = RESULTS_DIR / "step_1_5_sobol_rc_params.csv"

SALTELLI_N = 1024
RANDOM_SEED = 42

PROBLEM = {
    "num_vars": 5,
    "names": ["R_d_mult", "C_d_mult", "IT_load_mean", "T_amb_mean", "CAPTURE_FRACTION_LC"],
    "bounds": [
        [0.80, 1.20],
        [0.80, 1.20],
        [0.65, 0.95],
        [10.0, 25.0],
        [0.80, 1.00],
    ],
}


def _steady_state(
    rated_power_W: float, alpha_i: float, R_d: float, t_cdu_supply_K: float, it_load: float
) -> tuple[float, float]:
    """Return (theta_ss [K], Q_available [W]) at constant IT load."""
    sum_alpha_u = alpha_i * rated_power_W * it_load
    q_avail = max(sum_alpha_u, 0.0)
    theta_ss = t_cdu_supply_K + R_d * sum_alpha_u
    return theta_ss, q_avail


def _exergy_ss(theta_ss_K: float, q_available_W: float, t_amb_C: float) -> float:
    """Single-point exergy of the hot stream at steady state.
    Mirrors compute_exergy_vectorized for a 1-element input."""
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


def _evaluate_sample(
    sample: np.ndarray, scenario_params: dict
) -> tuple[float, float]:
    """Return (Q_available_mean, Exergy_DT_mean) for one (R, C, IT, T_amb, capture) sample.

    capture is ignored intentionally: in the current LC formulation it
    only enters Q_negotiated, not Q_available or Exergy_DT. Carrying it
    through Sobol as a no-op input is correct - its S1 will be ~ 0.
    """
    r_mult, c_mult, it_load, t_amb_c, _capture = sample
    r_d = scenario_params["R_d"] * r_mult
    # C_d does not enter the steady state but we keep the multiplier
    # so the Sobol problem stays 5-dimensional as specified.
    _ = scenario_params["C_d"] * c_mult
    theta_ss, q_avail = _steady_state(
        rated_power_W=scenario_params["rated_power_W"],
        alpha_i=scenario_params["alpha_i"],
        R_d=r_d,
        t_cdu_supply_K=scenario_params["T_cdu_supply"],
        it_load=float(it_load),
    )
    exergy = _exergy_ss(theta_ss, q_avail, float(t_amb_c))
    return q_avail, exergy


def _analyse(problem: dict, y: np.ndarray, label: str) -> dict | None:
    if float(np.std(y)) < 1e-12:
        logger.info("Response '%s' is constant; skipping Sobol decomposition", label)
        return None
    return sobol.analyze(
        problem, y, calc_second_order=False, seed=RANDOM_SEED, print_to_console=False
    )


def _rows(scenario: str, metric: str, si: dict | None) -> list[dict]:
    if si is None:
        return [
            {
                "scenario": scenario,
                "output_metric": metric,
                "parameter": name,
                "S1": 0.0,
                "S1_conf": 0.0,
                "ST": 0.0,
                "ST_conf": 0.0,
                "note": "Y is constant; indices undefined (variance=0)",
            }
            for name in PROBLEM["names"]
        ]
    return [
        {
            "scenario": scenario,
            "output_metric": metric,
            "parameter": name,
            "S1": round(float(si["S1"][i]), 4),
            "S1_conf": round(float(si["S1_conf"][i]), 4),
            "ST": round(float(si["ST"][i]), 4),
            "ST_conf": round(float(si["ST_conf"][i]), 4),
            "note": "",
        }
        for i, name in enumerate(PROBLEM["names"])
    ]


def main() -> None:
    logger.info(
        "Step 1.5 starting: Sobol on 5 RC parameters across %d LC scenarios", len(LC_SCENARIOS)
    )

    samples = saltelli.sample(PROBLEM, SALTELLI_N, calc_second_order=False)
    logger.info("Saltelli samples: shape=%s (N=%d)", samples.shape, SALTELLI_N)

    rows: list[dict] = []
    summary: list[tuple[str, str, str, float]] = []
    for scenario_name, params in LC_SCENARIOS.items():
        y_q = np.empty(len(samples))
        y_e = np.empty(len(samples))
        for i, s in enumerate(samples):
            q, e = _evaluate_sample(s, params)
            y_q[i] = q
            y_e[i] = e

        si_q = _analyse(PROBLEM, y_q, f"{scenario_name}/Q_available_mean")
        si_e = _analyse(PROBLEM, y_e, f"{scenario_name}/Exergy_DT_mean")
        rows.extend(_rows(scenario_name, "Q_available_mean", si_q))
        rows.extend(_rows(scenario_name, "Exergy_DT_mean", si_e))

        for metric, si in [("Q_available_mean", si_q), ("Exergy_DT_mean", si_e)]:
            if si is None:
                continue
            dom_idx = int(np.argmax(si["S1"]))
            summary.append(
                (
                    scenario_name,
                    metric,
                    PROBLEM["names"][dom_idx],
                    float(si["S1"][dom_idx]),
                )
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(out))

    logger.info("=" * 70)
    for scenario_name in LC_SCENARIOS:
        for metric in ("Q_available_mean", "Exergy_DT_mean"):
            sub = out[(out["scenario"] == scenario_name) & (out["output_metric"] == metric)]
            s1_sum = float(sub["S1"].sum())
            logger.info(
                "  %s %s: sum(S1) = %.3f", scenario_name, metric, s1_sum
            )
            if abs(s1_sum - 1.0) > 0.20 and s1_sum > 0.01:
                logger.warning(
                    "  Sobol additivity off for %s/%s: sum(S1)=%.3f", scenario_name, metric, s1_sum
                )
    for scenario_name, metric, driver, s1 in summary:
        logger.info(
            "  Driver dominante varianza %s (%s): %s (S1 = %.3f)",
            metric, scenario_name, driver, s1,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
