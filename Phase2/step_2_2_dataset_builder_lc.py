# ==============================================================================
# STEP 2.2 LC — Dataset Builder (Sector-Parametric, Geography-Agnostic)
# ==============================================================================
#
# Builds the DC–manufacturing plant pairing dataset for IS-Match validation.
#
# DESIGN PHILOSOPHY
# ─────────────────────────────────────────────────────────────────────────────
# The framework is intentionally geography-agnostic: it characterises the
# conditions under which ANY liquid-cooled DC can form an industrial-symbiosis
# partnership with ANY manufacturing plant, independently of location.
#
# The 9 representative plants are sector-parametric profiles, not real
# georeferenced sites. Their parameters are grounded in published EU-wide
# statistics (see sources below). Geographic proximity is treated as a
# SCENARIO DIMENSION (short / medium / long), not as a computed distance
# between specific real-world coordinates.
#
# This approach avoids over-fitting to a single regional case study and
# allows the IS-Match Score formula (and the RL optimisation in Phase 4)
# to be interpreted as a general result rather than a site-specific one.
# An illustrative regional application (e.g. Northern Italian DC cluster)
# can be presented as a worked example in Cap. 5.4 without restricting
# the general validity of the framework.
#
# DATA SOURCES (statistical basis for parametric profiles)
# ─────────────────────────────────────────────────────────────────────────────
# MidT tier — Paper/Pulp sector
#   Hotmaps Industrial Database — Manz & Fleiter, Fraunhofer ISI (2018)
#   DOI: 10.5281/zenodo.4687147  |  CC-BY-4.0
#   → EU Paper & Printing: median Fuel_Demand 200–330 GWh/yr per site
#   → Temperature profile: 88% of process heat at 100–200°C (Fleiter et al. 2013)
#   → T_req = 90°C (lower bound of paper drying; achievable with HP upgrade
#     from LC supply 46–50°C, lift ~40°C, COP ≈ 3–4)
#
# LowT tier — Food / Agro-industrial sector
#   ENEA IS benchmark — "Simbiosi Industriale Territoriale" (2021)
#   → Typical food-processing heat demand: 600–1 200 kW thermal
#   → T_req = 60–65°C (pasteurisation, drying, CIP cleaning)
#
# HighT tier — Rubber / Chemical / Steam
#   CE-HEAT project taxonomy — WP3/WP4 deliverables (ce-heat.eu)
#   → Typical demand: 2 800–6 200 kW; T_req = 130–135°C
#   → HP/CO₂-HTHP mandatory: ΔT ≈ 80–90°C from LC supply 46–50°C
#
# PROXIMITY SCENARIOS (distance dimension)
# ─────────────────────────────────────────────────────────────────────────────
# Each plant is assigned a representative distance_km that encodes its
# proximity scenario.  Three bands are used (one per plant within each tier):
#   Short   ≤  50 km  — co-located industrial park or adjacent municipality
#   Medium  50–100 km — same regional grid (pipeline or truck-loop feasible)
#   Long   100–200 km — inter-regional connection (requires pipeline investment)
# Values are chosen to span the range documented in the DC WHR literature
# (see [B6] Yuan et al. 2023, [B7] Lund et al. 2018).
#
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
#   step_2_2_ce_heat_dataset_lc.csv        — all 27 pairs (9 plants × 3 DC)
#   step_2_2_feasibility_summary_lc.csv    — RI stats by tier
#
# THESIS DECLARATION (Cap. 4.5 and Cap. 6.2)
# ─────────────────────────────────────────────────────────────────────────────
#   "All manufacturing profiles are sector-parametric, grounded in published
#    EU statistics (Hotmaps, ENEA, CE-HEAT). The framework is geographically
#    agnostic; no specific plant or DC site is modelled. Proximity is treated
#    as a scenario dimension (short / medium / long) rather than a computed
#    geographic distance. An illustrative application to a Northern Italian
#    DC–paper-mill cluster is presented in Cap. 5.4 as a worked example
#    without restricting the general validity of the IS-Match Score."
# ==============================================================================

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
OUT_PAIRS = BASE_DIR / "step_2_2_ce_heat_dataset_lc.csv"
OUT_SUMM  = BASE_DIR / "step_2_2_feasibility_summary_lc.csv"

# ── RI weights ─────────────────────────────────────────────────────────────────
ALPHA_Q   = 0.35   # heat quantity match
ALPHA_T   = 0.40   # temperature compatibility
ALPHA_A   = 0.15   # availability overlap
ALPHA_D   = 0.10   # distance penalty
D_MAX_KM  = 200.0  # km at which distance score → 0
               #   (upper end of technically feasible WHR pipeline, [B6])

# ── DC Scenarios (Liquid Cooling) ──────────────────────────────────────────────
# Three representative DC scales, parametrised from the LC-Opt Frontier
# reference model (HPE/ORNL, NeurIPS 2025) — see Decision D5.
# Labels are scale-based; no geographic anchor is implied.
DC_SCENARIOS_LC: list[dict] = [
    {
        "dc_name"           : "Edge_LC",
        "q_nominal_kw"      : 475.0,    # small edge/enterprise DC
        "t_supply_mean_c"   : 47.60,
        "t_supply_max_c"    : 50.00,
        "t_avail_fraction"  : 1.0,
    },
    {
        "dc_name"           : "Mid_LC",
        "q_nominal_kw"      : 3104.0,   # mid-scale DC; anchored on Frontier HPE/ORNL
        "t_supply_mean_c"   : 47.60,
        "t_supply_max_c"    : 50.00,
        "t_avail_fraction"  : 1.0,
    },
    {
        "dc_name"           : "Hyperscale_LC",
        "q_nominal_kw"      : 24500.0,  # hyperscale; 8× Mid_LC linear scaling
        "t_supply_mean_c"   : 46.00,
        "t_supply_max_c"    : 49.50,
        "t_avail_fraction"  : 1.0,
    },
]

# ── Representative manufacturing plants ────────────────────────────────────────
# Nine sector-parametric profiles spanning 3 temperature tiers × 3 proximity
# scenarios (short / medium / long).  Parameters are rounded to the nearest
# order of magnitude typical for each sector (see sources above).
#
# Naming convention:  {Tier}_{Index:02d}_{Sector}_{ProximityBand}
PARAMETRIC_PLANTS: list[dict] = [

    # ── LowT_60C  (food / agro-industrial / dairy) ─────────────────────────────
    # Source: ENEA IS benchmark (2021)
    {
        "plant_name"   : "LowT_01_Food_Short",
        "description"  : "Food drying — short proximity scenario [parametric, ENEA]",
        "distance_km"  : 30.0,    # short band: co-located industrial park
        "t_req_c"      : 60.0,
        "q_demand_kw"  : 800.0,
        "shift"        : "2shift",
        "tier"         : "LowT_60C",
        "delta_tc"     : 0.25,
        "source"       : "Parametric — ENEA IS benchmark (2021)",
    },
    {
        "plant_name"   : "LowT_02_Agro_Medium",
        "description"  : "Agro-industrial processing — medium proximity scenario [parametric, ENEA]",
        "distance_km"  : 75.0,    # medium band
        "t_req_c"      : 60.0,
        "q_demand_kw"  : 1200.0,
        "shift"        : "2shift",
        "tier"         : "LowT_60C",
        "delta_tc"     : 0.28,
        "source"       : "Parametric — ENEA IS benchmark (2021)",
    },
    {
        "plant_name"   : "LowT_03_Dairy_Short",
        "description"  : "Dairy processing — short proximity scenario [parametric, ENEA]",
        "distance_km"  : 45.0,    # short band: adjacent municipality
        "t_req_c"      : 65.0,
        "q_demand_kw"  : 650.0,
        "shift"        : "continuous",
        "tier"         : "LowT_60C",
        "delta_tc"     : 0.22,
        "source"       : "Parametric — ENEA IS benchmark (2021)",
    },

    # ── MidT_90C  (paper / pulp sector) ────────────────────────────────────────
    # Parameters grounded in Hotmaps EU statistics [Manz & Fleiter 2018]:
    #   median FD for Paper & Printing sites: 200–330 GWh/yr (EU28)
    #   Q_demand_kw = FD_GWh × 0.88 × 1e6 / 8760  (Fleiter et al. 2013)
    # Proximity bands are representative of EU DC–industry co-location patterns
    # documented in [B6] (Yuan et al. 2023) and [B7] (Lund et al. 2018).
    {
        "plant_name"   : "MidT_04_PaperPulp_Medium",
        "description"  : "Paper/pulp mill — medium proximity [parametric, Hotmaps-informed]",
        "distance_km"  : 80.0,    # medium band; within common WHR pipeline range
        "t_req_c"      : 90.0,
        "q_demand_kw"  : round(280 * 0.88 * 1e6 / 8760, 1),  # FD≈280 GWh/yr → 28 118 kW
        "shift"        : "continuous",
        "tier"         : "MidT_90C",
        "delta_tc"     : 0.42,
        "source"       : "Parametric — Hotmaps EU Paper statistics [Manz & Fleiter 2018, "
                         "DOI:10.5281/zenodo.4687147]; T-profile: Fleiter et al. (2013)",
    },
    {
        "plant_name"   : "MidT_05_PaperPulp_Short",
        "description"  : "Paper/pulp mill — short proximity [parametric, Hotmaps-informed]",
        "distance_km"  : 50.0,    # short band: same industrial zone
        "t_req_c"      : 90.0,
        "q_demand_kw"  : round(310 * 0.88 * 1e6 / 8760, 1),  # FD≈310 GWh/yr → 31 143 kW
        "shift"        : "continuous",
        "tier"         : "MidT_90C",
        "delta_tc"     : 0.40,
        "source"       : "Parametric — Hotmaps EU Paper statistics [Manz & Fleiter 2018, "
                         "DOI:10.5281/zenodo.4687147]; T-profile: Fleiter et al. (2013)",
    },
    {
        "plant_name"   : "MidT_06_PaperPulp_Long",
        "description"  : "Paper/pulp mill — long proximity [parametric, Hotmaps-informed]",
        "distance_km"  : 140.0,   # long band: inter-regional
        "t_req_c"      : 90.0,
        "q_demand_kw"  : round(210 * 0.88 * 1e6 / 8760, 1),  # FD≈210 GWh/yr → 21 096 kW
        "shift"        : "continuous",
        "tier"         : "MidT_90C",
        "delta_tc"     : 0.45,
        "source"       : "Parametric — Hotmaps EU Paper statistics [Manz & Fleiter 2018, "
                         "DOI:10.5281/zenodo.4687147]; T-profile: Fleiter et al. (2013)",
    },

    # ── HighT_130C  (rubber / chemical / industrial steam) ─────────────────────
    # Source: CE-HEAT project taxonomy, WP3/WP4 deliverables (ce-heat.eu)
    {
        "plant_name"   : "HighT_07_Rubber_Medium",
        "description"  : "Rubber vulcanisation — medium proximity [parametric, CE-HEAT]",
        "distance_km"  : 100.0,
        "t_req_c"      : 130.0,
        "q_demand_kw"  : 2800.0,
        "shift"        : "2shift",
        "tier"         : "HighT_130C",
        "delta_tc"     : 0.55,
        "source"       : "Parametric — CE-HEAT taxonomy (WP3/WP4)",
    },
    {
        "plant_name"   : "HighT_08_ChemProcess_Short",
        "description"  : "Chemical process — short proximity [parametric, CE-HEAT]",
        "distance_km"  : 55.0,
        "t_req_c"      : 130.0,
        "q_demand_kw"  : 4500.0,
        "shift"        : "continuous",
        "tier"         : "HighT_130C",
        "delta_tc"     : 0.58,
        "source"       : "Parametric — CE-HEAT taxonomy (WP3/WP4)",
    },
    {
        "plant_name"   : "HighT_09_Steam_Long",
        "description"  : "Industrial steam — long proximity [parametric, CE-HEAT]",
        "distance_km"  : 160.0,
        "t_req_c"      : 135.0,
        "q_demand_kw"  : 6200.0,
        "shift"        : "continuous",
        "tier"         : "HighT_130C",
        "delta_tc"     : 0.60,
        "source"       : "Parametric — CE-HEAT taxonomy (WP3/WP4)",
    },
]


# ── RI components ──────────────────────────────────────────────────────────────
def ri_quantity(q_dc: float, q_plant: float) -> float:
    return float(np.clip(q_dc / max(q_plant, 1e-9), 0.0, 1.0))


def ri_temperature(t_dc: float, t_req: float) -> float:
    """Full credit if T_dc ≥ T_req; exponential decay for lift required."""
    if t_dc >= t_req:
        return 1.0
    return float(math.exp(-(t_req - t_dc) / 40.0))


def ri_availability(t_avail_dc: float, shift: str) -> float:
    shift_frac = {"continuous": 0.95, "2shift": 0.65, "1shift": 0.35}
    t_p = shift_frac.get(shift, 0.65)
    return float(min(t_avail_dc, t_p) / max(t_avail_dc, t_p, 1e-9))


def ri_distance(dist_km: float) -> float:
    return float(np.clip(1.0 - dist_km / D_MAX_KM, 0.0, 1.0))


def compute_ri(q_dc, q_plant, t_dc, t_req, t_avail, shift, dist_km) -> dict:
    rq = ri_quantity(q_dc, q_plant)
    rt = ri_temperature(t_dc, t_req)
    ra = ri_availability(t_avail, shift)
    rd = ri_distance(dist_km)
    ri = ALPHA_Q * rq + ALPHA_T * rt + ALPHA_A * ra - ALPHA_D * (1 - rd)
    return {
        "RI_quantity"    : round(rq, 4),
        "RI_temperature" : round(rt, 4),
        "RI_availability": round(ra, 4),
        "RI_distance"    : round(rd, 4),
        "RI_static"      : round(float(np.clip(ri, 0, 1)), 4),
        "feasible"       : int(ri >= 0.5),
    }


# ── Build dataset ──────────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    """
    Builds 27 DC–plant pairs (3 DC scenarios × 9 sector-parametric plants).

    Distance is read directly from plant["distance_km"] — a scenario parameter
    that encodes the proximity band (short / medium / long).  The same distance
    applies to all three DC scales, making the framework geographically agnostic
    at this stage.  A sensitivity analysis over distance_km is straightforward
    given this parametric structure.
    """
    rows = []
    for dc in DC_SCENARIOS_LC:
        for plant in PARAMETRIC_PLANTS:
            dist = plant["distance_km"]
            ri   = compute_ri(
                dc["q_nominal_kw"], plant["q_demand_kw"],
                dc["t_supply_mean_c"], plant["t_req_c"],
                dc["t_avail_fraction"], plant["shift"], dist,
            )
            rows.append({
                "dc_name"                : dc["dc_name"],
                "dc_q_nominal_kw"        : dc["q_nominal_kw"],
                "dc_t_supply_mean_c"     : dc["t_supply_mean_c"],
                "dc_t_supply_max_c"      : dc["t_supply_max_c"],
                "dc_t_avail_fraction"    : dc["t_avail_fraction"],
                "plant_name"             : plant["plant_name"],
                "plant_description"      : plant["description"],
                "plant_tier"             : plant["tier"],
                "plant_proximity_band"   : (
                    "short"  if dist <= 50 else
                    "medium" if dist <= 100 else "long"
                ),
                "plant_t_req_c"          : plant["t_req_c"],
                "plant_q_demand_kw"      : plant["q_demand_kw"],
                "plant_shift_pattern"    : plant["shift"],
                "plant_delta_tc_baseline": plant["delta_tc"],
                "plant_data_source"      : plant["source"],
                "distance_km"            : dist,
                **ri,
            })
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    src_label = {
        "LowT_60C"   : "Parametric (ENEA 2021)",
        "MidT_90C"   : "Parametric — Hotmaps-informed (Manz & Fleiter 2018)",
        "HighT_130C" : "Parametric (CE-HEAT WP3/WP4)",
    }
    rows = []
    for tier in ["LowT_60C", "MidT_90C", "HighT_130C"]:
        for dc in df["dc_name"].unique():
            sub = df[(df["plant_tier"] == tier) & (df["dc_name"] == dc)]
            rows.append({
                "dc_name"        : dc,
                "tier"           : tier,
                "data_source"    : src_label[tier],
                "n_pairs"        : len(sub),
                "n_feasible"     : int(sub["feasible"].sum()),
                "RI_static_mean" : round(sub["RI_static"].mean(), 4),
                "RI_static_max"  : round(sub["RI_static"].max(), 4),
                "RI_temp_mean"   : round(sub["RI_temperature"].mean(), 4),
                "dist_mean_km"   : round(sub["distance_km"].mean(), 1),
            })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  STEP 2.2 LC — Dataset Builder (Sector-Parametric, General Framework)")
    print("=" * 72)

    df = build_dataset()
    df.to_csv(OUT_PAIRS, index=False)
    print(f"\n[OK] {OUT_PAIRS.name}  ({len(df)} rows, "
          f"{df['dc_name'].nunique()} DCs × {df['plant_name'].nunique()} plants)")

    # Print table
    print(f"\n  {'DC':<18} {'Plant':<38} {'Tier':<14} "
          f"{'Band':<8} {'RI':>6} {'Dist':>7} {'OK':>4}")
    print("  " + "-" * 100)
    for _, r in df.iterrows():
        ok = "YES" if r["feasible"] else "no"
        print(f"  {r['dc_name']:<18} {r['plant_name']:<38} {r['plant_tier']:<14} "
              f"{r['plant_proximity_band']:<8} {r['RI_static']:>6.4f} "
              f"{r['distance_km']:>5.0f}km {ok:>4}")

    # Summary
    summ = build_summary(df)
    summ.to_csv(OUT_SUMM, index=False)
    print(f"\n  Feasibility summary (RI_static ≥ 0.5):")
    print(f"  {'Tier':<14} {'DC':<18} {'Feas/Tot':>8}  {'RI_mean':>8}  "
          f"{'RI_max':>8}  {'Dist_mean':>10}")
    print("  " + "-" * 80)
    for _, r in summ.iterrows():
        print(f"  {r['tier']:<14} {r['dc_name']:<18} "
              f"{r['n_feasible']}/{r['n_pairs']:>3}       "
              f"{r['RI_static_mean']:>8.4f}  {r['RI_static_max']:>8.4f}  "
              f"{r['dist_mean_km']:>8.1f} km")

    total_f = int(df["feasible"].sum())
    total   = len(df)
    print(f"\n  Total feasible: {total_f}/{total} ({100*total_f/total:.0f}%)")

    # Proximity band breakdown
    print(f"\n  Feasibility by proximity band:")
    for band in ["short", "medium", "long"]:
        sub = df[df["plant_proximity_band"] == band]
        nf  = int(sub["feasible"].sum())
        print(f"    {band:<8}: {nf}/{len(sub)}  RI_mean={sub['RI_static'].mean():.4f}"
              f"  dist_mean={sub['distance_km'].mean():.0f} km")

    print(f"\n  Outputs:")
    print(f"    {OUT_PAIRS.name}")
    print(f"    {OUT_SUMM.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
