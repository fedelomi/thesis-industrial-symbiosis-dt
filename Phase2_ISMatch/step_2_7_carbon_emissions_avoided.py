"""
Step 2.7 LC - CO2 Avoided per IS-Match Pair
============================================
Phase 2 robustness layer (additive, post-baseline)

Adds a sustainability dimension by estimating the CO2 emissions avoided
each year if the data center waste heat displaces natural-gas-fired
process heat at the manufacturing site:

    Q_recovery_annual_kWh = Q_dc_supply_kW * annual_active_hours * utilization
    CO2_avoided_tons_yr   = Q_recovery_annual_kWh * EMISSION_FACTOR / 1000
    equiv_cars_off_road   = CO2_avoided_tons_yr / 4.6

where:
    Q_dc_supply_kW           DC nominal thermal capacity (Edge / Mid / Hyperscale LC)
    annual_active_hours      from the aggregated shift pattern:
                             continuous=8760, 2shift=2500, 3shift=6000
    utilization              ri_temporal from step_2_1 (fraction effectively used)
    EMISSION_FACTOR          0.276 kgCO2/kWh - EU EEA 2024 natural gas grid mix
    4.6 tCO2/yr              average EU passenger car emissions (EEA)

Output:
    results/step_2_7_carbon_emissions_avoided.csv

Sanity reference (Mid_LC + LowT_60C, 2shift):
    3104 kW * 2500 h * ri_temporal * 0.276 / 1000 -> O(1-3 ktCO2/yr)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCORES_CSV = RESULTS_DIR / "step_2_1_is_match_scores_lc.csv"
DATASET_CSV = RESULTS_DIR / "step_2_2_ce_heat_dataset_lc.csv"
OUT_CSV = RESULTS_DIR / "step_2_7_carbon_emissions_avoided.csv"

Q_NOMINAL_KW_BY_DC: dict[str, float] = {
    "Edge_LC": 475.0,
    "Mid_LC": 3104.0,
    "Hyperscale_LC": 24500.0,
}

ANNUAL_HOURS_BY_SHIFT: dict[str, float] = {
    "continuous": 8760.0,
    "2shift": 2500.0,
    "3shift": 6000.0,
    "1shift": 2000.0,
}

EMISSION_FACTOR_KG_PER_KWH = 0.276
EMISSION_FACTOR_LABEL = "0.276 (gas natural, EU EEA 2024)"
CAR_TCO2_PER_YR = 4.6


def _shift_by_process(dataset_csv: Path) -> dict[str, str]:
    """Return process_name -> shift_pattern (matches step_2_1 'first' aggregation)."""
    df = pd.read_csv(dataset_csv, usecols=["plant_tier", "plant_shift_pattern"])
    df = df.rename(columns={"plant_tier": "process_name", "plant_shift_pattern": "shift_pattern"})
    return df.groupby("process_name")["shift_pattern"].first().to_dict()


def _hours_for(shift: str) -> float:
    if shift not in ANNUAL_HOURS_BY_SHIFT:
        logger.warning("Unknown shift pattern '%s'; defaulting to 2shift (2500 h/yr)", shift)
        return ANNUAL_HOURS_BY_SHIFT["2shift"]
    return ANNUAL_HOURS_BY_SHIFT[shift]


def main() -> None:
    logger.info(
        "Step 2.7 starting: CO2 avoided per pair (gas natural displaced, %.3f kgCO2/kWh)",
        EMISSION_FACTOR_KG_PER_KWH,
    )

    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing baseline scores: {SCORES_CSV}")
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_CSV}")

    df_scores = pd.read_csv(SCORES_CSV)
    shift_map = _shift_by_process(DATASET_CSV)
    logger.info(
        "Loaded %d pairs and shift map for %d processes",
        len(df_scores), len(shift_map),
    )

    rows: list[dict] = []
    for _, r in df_scores.iterrows():
        dc_name = str(r["dc_name"])
        process_name = str(r["process_name"])
        q_dc_kw = float(Q_NOMINAL_KW_BY_DC.get(dc_name, 0.0))
        if q_dc_kw == 0.0:
            logger.warning("Unknown DC '%s'; CO2 avoided set to 0", dc_name)
        shift = shift_map.get(process_name, "2shift")
        hours = _hours_for(shift)
        utilization = float(r["ri_temporal"])

        q_recovery_kwh = q_dc_kw * hours * utilization
        co2_tons = q_recovery_kwh * EMISSION_FACTOR_KG_PER_KWH / 1000.0
        cars = co2_tons / CAR_TCO2_PER_YR
        rows.append({
            "pair": f"{dc_name} + {process_name}",
            "dc_name": dc_name,
            "process_name": process_name,
            "shift_pattern": shift,
            "annual_hours": hours,
            "ri_temporal_utilization": round(utilization, 4),
            "Q_recovery_GWh_yr": round(q_recovery_kwh / 1e6, 4),
            "CO2_avoided_tons_yr": round(co2_tons, 1),
            "equiv_cars_off_road": round(cars, 1),
            "emission_factor_used": EMISSION_FACTOR_LABEL,
        })

    out = pd.DataFrame(rows).sort_values("CO2_avoided_tons_yr", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s (%d rows)", OUT_CSV.name, len(out))

    top = out.iloc[0]
    co2_min = float(out["CO2_avoided_tons_yr"].min())
    co2_max = float(out["CO2_avoided_tons_yr"].max())

    logger.info("=" * 70)
    logger.info(
        "  Top pair: %s avoids %.0f tCO2/yr (~ %.0f cars off road)",
        top["pair"], top["CO2_avoided_tons_yr"], top["equiv_cars_off_road"],
    )
    logger.info(
        "  Range across 9 pairs: [%.0f, %.0f] tCO2/yr", co2_min, co2_max,
    )
    logger.info("=" * 70)

    mid_low = out[(out["dc_name"] == "Mid_LC") & (out["process_name"] == "LowT_60C")]
    if not mid_low.empty:
        v = float(mid_low.iloc[0]["CO2_avoided_tons_yr"])
        if not 1000.0 <= v <= 3000.0:
            logger.warning(
                "Mid_LC + LowT_60C CO2 avoided = %.0f tCO2/yr is outside expected [1000, 3000] range",
                v,
            )


if __name__ == "__main__":
    main()
