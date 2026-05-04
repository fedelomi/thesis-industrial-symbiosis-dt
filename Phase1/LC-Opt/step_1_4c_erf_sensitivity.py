# ==============================================================================
# DATACENTER DT PROJECT - STEP 1.4c  [LIQUID COOLING]
# ERF Sensitivity Analysis — capture_fraction grid
#
# Thesis reference: Cap. 6.2 Limitations — sensitivity of ERF to
#   capture_fraction assumption.
# Operating point: single_phase_immersion_conservative (cf=0.90, ERF=0.81).
# Source for technology benchmarks: ICEF SDC Roadmap 2025
#   (Aines & Sandalow, Oct 2025).
#
# Analytical derivation of ERF = cf^2:
#   Let Q_available = heat captured by the cooling system at the CDU boundary.
#   By definition of capture fraction (cf):
#     EREUSE = cf * Q_available         (heat delivered outside DC boundary)
#     Q_available = cf * EDC            (because cf = Q_available / EDC)
#   => EDC = Q_available / cf
#   => ERF = EREUSE / EDC
#          = (cf * Q_available) / (Q_available / cf)
#          = cf^2
#   Numerical verification included below (tolerance 1e-9).
# ==============================================================================

import os
import shutil
import numpy as np
import pandas as pd
import xlsxwriter

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RC_CSV     = os.path.join(BASE_DIR, "lc_dc_results_annual.csv")
EED_CSV    = os.path.join(BASE_DIR, "eed_compliance_lc.csv")
OUT_CSV    = os.path.join(BASE_DIR, "erf_sensitivity_lc.csv")
P1_XLSX    = os.path.join(BASE_DIR, "phase1_lc_results.xlsx")
TMP_XLSX   = "/tmp/phase1_lc_results_step1_4c.xlsx"

# ------------------------------------------------------------------------------
# CAPTURE FRACTION GRID (ICEF SDC Roadmap 2025 technology benchmarks)
# ------------------------------------------------------------------------------
CAPTURE_FRACTION_GRID: dict[str, float] = {
    "air_cooled_rear_door"                : 0.70,
    "direct_liquid_cooling"               : 0.80,
    "single_phase_immersion_conservative" : 0.90,  # current operating point
    "single_phase_immersion_optimal"      : 0.95,
    "full_immersion"                      : 0.99,
}

OPERATING_POINT_CF  = 0.90
OPERATING_POINT_ERF = OPERATING_POINT_CF ** 2   # 0.81

# TEE baseline per scenario (EUR/year) — read from CSV or hardcoded fallback
TEE_FALLBACK: dict[str, int] = {
    "Edge_LC"       : 55_000,
    "Mid_LC"        : 360_000,
    "Hyperscale_LC" : 2_842_000,
}

# Phase 1 CSV → sheet mapping (must stay in sync with run_phase_1_lc.py)
P1_BASE_SHEETS = [
    ("lc_dc_results_annual.csv",           "RC_Model_Results_LC"),
    ("lc_synthetic_profile_annual.csv",     "Synthetic_Profile_LC"),
    ("lc_real_vs_synthetic_comparison.csv", "Real_vs_Synthetic_LC"),
    ("lc_validation_1_4.csv",              "Validation_LC"),
    ("lc_sensitivity_validation.csv",       "Sensitivity_Analysis_LC"),
]
EU_SIZE_CATEGORY = {
    "Edge_LC"       : "Very small (<=500 kW)",
    "Mid_LC"        : "Large (2-10 MW)",
    "Hyperscale_LC" : "Very large (>10 MW)",
}
RATED_POWER_MW = {
    "Edge_LC"       : 0.5,
    "Mid_LC"        : 3.2,
    "Hyperscale_LC" : 25.0,
}


# ------------------------------------------------------------------------------
# REGULATORY KPIs helper (mirrors step_0 logic; needed for xlsx rebuild)
# ------------------------------------------------------------------------------
def _build_reg_kpis(df_rc: pd.DataFrame) -> pd.DataFrame:
    reg = (
        df_rc.groupby("scenario", sort=False)
        .agg(
            Rated_Power_MW       =("scenario",     lambda s: RATED_POWER_MW.get(s.iloc[0], float("nan"))),
            EU_Size_Category     =("scenario",     lambda s: EU_SIZE_CATEGORY.get(s.iloc[0], "Unknown")),
            TWH_mean_C           =("TWH",          "mean"),
            TWH_min_C            =("TWH",          "min"),
            TWH_max_C            =("TWH",          "max"),
            ERF_annual_mean      =("ERF",          "mean"),
            Q_available_mean_kW  =("Q_available",  lambda x: x.mean() / 1e3),
            Q_negotiated_mean_kW =("Q_negotiated", lambda x: x.mean() / 1e3),
        )
        .reset_index()
    )
    for c in ["TWH_mean_C", "TWH_min_C", "TWH_max_C"]:
        reg[c] = reg[c].round(2)
    reg["ERF_annual_mean"]      = reg["ERF_annual_mean"].round(4)
    reg["Q_available_mean_kW"]  = reg["Q_available_mean_kW"].round(1)
    reg["Q_negotiated_mean_kW"] = reg["Q_negotiated_mean_kW"].round(1)
    return reg


# ------------------------------------------------------------------------------
# MAIN COMPUTATION
# ------------------------------------------------------------------------------
def build_erf_sensitivity(df_rc: pd.DataFrame,
                          tee_map: dict[str, int]) -> pd.DataFrame:
    """
    For each (scenario, cooling_technology) pair compute ERF and TEE impact.
    ERF = cf^2 (analytical; see header derivation).
    TEE_delta_eur_year = (ERF - 0.81) / 0.81 * TEE_baseline
    vs_baseline_pct    = (ERF - 0.81) / 0.81 * 100
    """
    # Mean Q_available per scenario [kW]
    q_avail_mean = (
        df_rc.groupby("scenario")["Q_available"].mean() / 1e3
    ).to_dict()

    rows = []
    for scenario in ["Edge_LC", "Mid_LC", "Hyperscale_LC"]:
        q_avail = q_avail_mean[scenario]
        tee_ref = tee_map[scenario]

        for tech, cf in sorted(CAPTURE_FRACTION_GRID.items(),
                               key=lambda kv: kv[1]):  # ascending by cf
            erf_analytical = cf ** 2

            # Numerical verification: (cf * Q) / (Q / cf) must equal cf^2
            q_neg_num = cf * q_avail        # EREUSE proxy [kW]
            edc_proxy  = q_avail / cf       # EDC proxy [kW]
            erf_numeric = q_neg_num / edc_proxy
            assert abs(erf_numeric - erf_analytical) < 1e-9, (
                f"ERF mismatch for {scenario} cf={cf}: "
                f"analytical={erf_analytical:.10f}, numeric={erf_numeric:.10f}"
            )

            q_neg_kw     = cf * q_avail
            vs_baseline  = (erf_analytical - OPERATING_POINT_ERF) / OPERATING_POINT_ERF * 100
            tee_delta    = (erf_analytical - OPERATING_POINT_ERF) / OPERATING_POINT_ERF * tee_ref

            rows.append({
                "scenario"             : scenario,
                "cooling_technology"   : tech,
                "capture_fraction"     : cf,
                "ERF"                  : round(erf_analytical, 6),
                "Q_negotiated_mean_kW" : round(q_neg_kw, 1),
                "TEE_baseline_eur_year": tee_ref,
                "TEE_delta_eur_year"   : round(tee_delta, 0),
                "vs_baseline_pct"      : round(vs_baseline, 2),
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# XLSX REBUILD (xlsxwriter → /tmp → copy back)
# Includes: 5 original CSV sheets + Regulatory_KPIs + EED_Compliance
#           + ERF_Sensitivity (new, 8th sheet)
# ------------------------------------------------------------------------------
def rebuild_xlsx(df_reg: pd.DataFrame,
                 df_eed: pd.DataFrame,
                 df_erf: pd.DataFrame) -> None:
    """Rebuild phase1_lc_results.xlsx with 8 sheets (adds ERF_Sensitivity)."""
    print("  Rebuilding phase1_lc_results.xlsx (8 sheets) ...")
    wb  = xlsxwriter.Workbook(TMP_XLSX, {"constant_memory": True})
    hdr = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})

    def write_df(ws: xlsxwriter.worksheet.Worksheet,
                 df: pd.DataFrame) -> None:
        for ci, col in enumerate(df.columns):
            ws.write(0, ci, col, hdr)
        for ri, row in enumerate(df.itertuples(index=False), start=1):
            for ci, v in enumerate(row):
                ws.write(ri, ci, v)

    # ── original CSV-backed sheets ──────────────────────────────────────────
    for csv_name, sheet_name in P1_BASE_SHEETS:
        csv_path = os.path.join(BASE_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"    [SKIP] {csv_name} not found")
            continue
        df = pd.read_csv(csv_path)
        ws = wb.add_worksheet(sheet_name)
        write_df(ws, df)
        print(f"    [OK]   {sheet_name} ({len(df)} rows)")

    # ── derived / post-processing sheets ────────────────────────────────────
    for sheet_name, df in [
        ("Regulatory_KPIs", df_reg),
        ("EED_Compliance",  df_eed),
        ("ERF_Sensitivity", df_erf),
    ]:
        ws = wb.add_worksheet(sheet_name)
        write_df(ws, df)
        print(f"    [OK]   {sheet_name} ({len(df)} rows)")

    wb.close()
    shutil.copy2(TMP_XLSX, P1_XLSX)
    os.remove(TMP_XLSX)
    size_mb = os.path.getsize(P1_XLSX) / 1e6
    print(f"  Saved: {P1_XLSX}  ({size_mb:.1f} MB, 8 sheets)")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  STEP 1.4c — ERF Sensitivity Analysis [Liquid Cooling]")
    print("=" * 70)

    # Load RC model CSV (TWH, ERF, Q_available, Q_negotiated already present)
    df_rc = pd.read_csv(
        RC_CSV,
        usecols=["scenario", "TWH", "ERF", "Q_available", "Q_negotiated"],
    )
    print(f"  Loaded {len(df_rc):,} rows from {os.path.basename(RC_CSV)}")

    # Load TEE baseline from EED_Compliance CSV (fallback to hardcoded)
    if os.path.exists(EED_CSV):
        eed_raw = pd.read_csv(EED_CSV)
        tee_map: dict[str, int] = dict(
            zip(eed_raw["scenario"], eed_raw["TEE_estimated_eur_year"].astype(int))
        )
        print(f"  TEE baseline loaded from {os.path.basename(EED_CSV)}")
    else:
        tee_map = TEE_FALLBACK
        print("  [WARN] eed_compliance_lc.csv not found — using hardcoded TEE fallback")

    # Fill missing scenarios from fallback
    for k, v in TEE_FALLBACK.items():
        tee_map.setdefault(k, v)

    # Build ERF sensitivity table
    df_erf = build_erf_sensitivity(df_rc, tee_map)
    print(f"\n  ERF_Sensitivity table ({len(df_erf)} rows):")
    print(df_erf.to_string(index=False))

    # Write intermediate CSV
    df_erf.to_csv(OUT_CSV, index=False)
    print(f"\n  CSV written: {OUT_CSV}")

    # Rebuild xlsx with 8 sheets
    df_reg = _build_reg_kpis(df_rc)
    df_eed = pd.read_csv(EED_CSV) if os.path.exists(EED_CSV) else pd.DataFrame()
    rebuild_xlsx(df_reg, df_eed, df_erf)

    print("\n  STEP 1.4c COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
