# ==============================================================================
# DATACENTER DT PROJECT - STEP 1.1 + 1.2 + 1.2-bis
# RC thermal model + exergy computation (arXiv:2604.15594)
# ==============================================================================

# Decision active: D1 — DT calibration on published KPIs (C1) instead of
#                       proprietary Frontier ORNL data. Wiki: [[implementation-decisions#D1]].

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

logger = logging.getLogger(__name__)

# Output directory (post-reorg, audit 2026-05-05)
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# STEP 1.1 - RC THERMAL MODEL (arXiv:2604.15594)
# Exact analytical solution (replaces unstable explicit Euler)
# theta(t+dt) = theta_ss + (theta(t) - theta_ss) * exp(-dt/tau)
# ------------------------------------------------------------------------------
def rc_thermal_model(
    theta_now: float,
    dt: float,
    C_d: float,
    sum_alpha_u: float,
    T_amb: float,
    R_d: float,
    Q_cool: float = 0.0,
) -> float:
    """
    Exact analytical solution of the RC thermal model (always stable).

    Explicit Euler diverges when dt >> tau = C_d * R_d.
    For Edge: tau = 12000 * 5.32e-4 = 6.4s << dt=900s -> guaranteed overflow.

    The linear ODE dtheta/dt = a - b*theta has exact solution:
      theta(t+dt) = theta_ss + (theta(t) - theta_ss) * exp(-dt/tau)
    """
    tau       = C_d * R_d                              # time constant [s]
    theta_ss  = T_amb + R_d * (sum_alpha_u - Q_cool)  # steady-state T [K]
    theta_next = theta_ss + (theta_now - theta_ss) * np.exp(-dt / tau)
    return theta_next

# R_d computed to yield a realistic T_supply at steady state:
# R_d = delta_T_target / (P_IT - Q_cool)
# Target: T_supply = T_amb + 25 K (e.g. 15+25 = 40 C for Edge)
DC_SCENARIOS = {
    "Edge": {
        "rated_power_W": 100e3,
        "C_d": 12000,
        # R_d = 25K / (0.92*100e3 - 45e3) = 25/47000 ~ 5.32e-4 K/W
        "R_d": 25.0 / (0.92 * 100e3 - 45e3),
        "alpha_i": 0.92,
        "T_amb": 15 + 273.15,
        "Q_cool": 45e3,
        "T_supply_target": 40.0,  # degC - for validation
    },
    "Mid": {
        "rated_power_W": 5e6,
        "C_d": 65000,
        # R_d = 20K / (0.95*5e6 - 2.8e6) = 20/1950000 ~ 1.03e-5 K/W
        "R_d": 20.0 / (0.95 * 5e6 - 2.8e6),
        "alpha_i": 0.95,
        "T_amb": 15 + 273.15,
        "Q_cool": 2.8e6,
        "T_supply_target": 35.0,
    },
    "Hyperscale": {
        "rated_power_W": 100e6,
        "C_d": 220000,
        # R_d = 15K / (0.97*100e6 - 58e6) = 15/39000000 ~ 3.85e-7 K/W
        "R_d": 15.0 / (0.97 * 100e6 - 58e6),
        "alpha_i": 0.97,
        "T_amb": 15 + 273.15,
        "Q_cool": 58e6,
        "T_supply_target": 30.0,
    }
}

# Capture fraction: fraction of Q_available negotiated with IS partner.
# Airside: 5% duct losses → 95% capturable. Audit fix P1-5.
CAPTURE_FRACTION_AIRSIDE: float = 0.95

# Realistic IT load profile: daily sinusoidal cycle + Gaussian noise
def it_load_profile(
    t: np.ndarray,
    N_STEP: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulates realistic IT load variability:
    - Daily cycle (96 steps = 24h at 15-min resolution)
    - Weekly cycle
    - Gaussian noise
    """
    rng = np.random.default_rng(seed)
    steps_per_day = 96
    # Base load 60-95% with daily cycle
    load = (
        0.775
        + 0.175 * np.sin(2 * np.pi * t / steps_per_day - np.pi/2)  # daily
        + 0.05  * np.sin(2 * np.pi * t / (steps_per_day * 7))       # weekly
        + rng.normal(0, 0.02, size=len(t))                           # noise +/-2%
    )
    return np.clip(load, 0.10, 1.0)

def compute_cv_rmse(
    sim_values: np.ndarray,
    ref_values: np.ndarray,
) -> float:
    """CV-RMSE in % -- uses real measured data as ref_values."""
    rmse = np.sqrt(np.mean((sim_values - ref_values) ** 2))
    return (rmse / np.mean(ref_values)) * 100

def compute_exergy(
    theta_K: float,
    Q_available: float,
    T0_K: float = 288.15,
) -> float:
    """
    Exergy_DT = m_dot * [(h - h0) - T0 * (s - s0)]
    T0_K: reference temperature already in Kelvin (default 15 C = 288.15 K)
    """
    # Guard: if theta is nan/inf or Q=0 -> zero exergy
    if not np.isfinite(theta_K) or Q_available <= 0:
        return 0.0

    # Clamp to liquid water range (T_triple+0.1 ... 95 C)
    theta_safe = float(np.clip(theta_K, 273.17, 368.15))

    try:
        h  = PropsSI('H', 'T', theta_safe, 'P', 101325, 'Water')
        h0 = PropsSI('H', 'T', T0_K,      'P', 101325, 'Water')
        s  = PropsSI('S', 'T', theta_safe, 'P', 101325, 'Water')
        s0 = PropsSI('S', 'T', T0_K,      'P', 101325, 'Water')
    except Exception as exc:
        logger.warning(
            "CoolProp call failed at theta_safe=%.3f K, T0_K=%.3f K (returning zero exergy). Error: %s",
            theta_safe, T0_K, exc,
        )
        return 0.0

    dh = h - h0
    if abs(dh) < 1.0:  # < 1 J/kg -> essentially at T_amb, exergy ~ 0
        return 0.0

    m_dot = Q_available / dh
    return float(m_dot * (dh - T0_K * (s - s0)))

# ------------------------------------------------------------------------------
# STEP 1.2 - SIMULATION LOOP 15-min steps (35,040 steps/year)
# ------------------------------------------------------------------------------
def run_simulation(
    scenario_name: str,
    params: dict,
) -> tuple[pd.DataFrame, bool]:
    t_ss = (params['T_amb'] - 273.15
            + params['R_d'] * (params['alpha_i'] * params['rated_power_W'] - params['Q_cool']))
    print(f"\nSimulation: {scenario_name} "
          f"({params['rated_power_W']/1e3:.0f} kW) | "
          f"R_d={params['R_d']:.2e} K/W | "
          f"T_supply_ss={t_ss:.1f} C")

    dt     = 15 * 60   # 900 s
    N_STEP = 35_040
    Q_min  = 0.10 * params["rated_power_W"]
    T0_K   = params["T_amb"]  # 288.15 K = 15 C

    t_idx        = np.arange(N_STEP)
    load_profile = it_load_profile(t_idx, N_STEP)

    results = {
        "step": [], "T_supply": [], "Q_available": [],
        "Exergy_DT": [], "Q_negotiated": [], "it_load_frac": []
    }

    theta = params["T_amb"]  # initial state = T_amb

    for t in range(N_STEP):
        u_t         = load_profile[t]
        sum_alpha_u = params["alpha_i"] * params["rated_power_W"] * u_t

        theta = rc_thermal_model(
            theta, dt, params["C_d"], sum_alpha_u,
            params["T_amb"], params["R_d"], params["Q_cool"]
        )
        T_supply_C = theta - 273.15

        Q_available  = max(sum_alpha_u - params["Q_cool"], 0.0)
        Q_negotiated = CAPTURE_FRACTION_AIRSIDE * Q_available  # hard constraint

        # STEP 1.2-bis: exergy with physical range check
        exergy = compute_exergy(theta, Q_available, T0_K=params["T_amb"])

        results["step"].append(t)
        results["T_supply"].append(T_supply_C)
        results["Q_available"].append(Q_available)
        results["Exergy_DT"].append(exergy)
        results["Q_negotiated"].append(Q_negotiated)
        results["it_load_frac"].append(u_t)

    df = pd.DataFrame(results)
    df["scenario"]       = scenario_name
    df["t_availability"] = (df["Q_available"] > Q_min).mean()

    # -------------------------------------------------------------------------
    # Calibration gate — physical plausibility check (audit fix P1-1 + P1-3).
    #
    # The previous code computed CV-RMSE against gaussian-perturbed simulation
    # output (tautological: CV-RMSE ≈ 1.5% by construction). This is removed.
    #
    # Calibration evidence: step_1_4d_benchmark_comparison.py demonstrates
    # Model TWH_mean (47.6°C for LC) is within 18°C of 4/5 real-world LC DC
    # cases (Benchmark_Comparison sheet in phase1_lc_results.xlsx).
    # For airside: T_supply ∈ [21, 29] °C consistent with ASHRAE A2 envelope.
    #
    # Physical plausibility bounds for airside (ASHRAE 90.4-2016, §6.3):
    T_SUPPLY_MIN_C: float = 21.0   # airside lower bound (ASHRAE A2)
    T_SUPPLY_MAX_C: float = 29.0   # airside upper bound
    T_AVAIL_MIN:    float = 0.95   # minimum expected t_availability (baseload)

    t_supply_mean = float(df["T_supply"].mean())
    t_avail       = float(df["t_availability"].iloc[0])
    plausible = (T_SUPPLY_MIN_C <= t_supply_mean <= T_SUPPLY_MAX_C
                 and t_avail >= T_AVAIL_MIN)

    print(f"  T_supply mean     : {t_supply_mean:.1f} C  "
          f"(bounds [{T_SUPPLY_MIN_C}, {T_SUPPLY_MAX_C}] C)")
    print(f"  t_availability    : {t_avail:.2%}  (bound >= {T_AVAIL_MIN:.0%})")
    print(f"  Q_available range : {df['Q_available'].min()/1e3:.1f} - "
          f"{df['Q_available'].max()/1e3:.1f} kW")
    print(f"  Plausibility gate : {'PASS' if plausible else 'FAIL'}")

    if not plausible:
        raise ValueError(
            f"Physical plausibility gate FAIL for {scenario_name}: "
            f"T_supply_mean={t_supply_mean:.1f}°C, t_avail={t_avail:.2%}"
        )

    return df, plausible

# ------------------------------------------------------------------------------
# MAIN — wrapped in guard so importing this module does not trigger simulation.
# Audit fix P1-2: prevents side-effects (CSV writes) on import for unit tests.
# ------------------------------------------------------------------------------
def main() -> None:
    df_all = pd.DataFrame()
    for name, params in DC_SCENARIOS.items():
        df, _ = run_simulation(name, params)
        df_all = pd.concat([df_all, df], ignore_index=True)

    out_csv = RESULTS_DIR / "datacenter_dt_results_annual.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nOutput saved: {out_csv}  ({len(df_all)} rows)")


if __name__ == "__main__":
    main()