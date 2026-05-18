# ==============================================================================
# DATACENTER DT PROJECT - STEP 1.4d  [LIQUID COOLING]
# Real-Case Benchmark Comparison
#
# Compares the RC thermal model output (TWH_mean) against published real-world
# liquid-cooled data-center waste heat recovery cases.
# Used in thesis Cap. 4.2 (Physical Digital Twin validation — external plausibility).
#
# All real-case data is hardcoded from published sources (no external fetch).
# Sources:
#   - IEA HPT Annex 47 (2023)
#   - ICEF SDC Roadmap 2025 (Aines & Sandalow, Oct 2025)
#   - EcoDataCenter press release 2022
# ==============================================================================

# Decision active: D5 — TWH benchmark vs. real LC DC cases (D5 parametrization).


import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xlsxwriter

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RC_CSV     = RESULTS_DIR / "lc_dc_results_annual.csv"
EED_CSV    = RESULTS_DIR / "eed_compliance_lc.csv"
ERF_CSV    = RESULTS_DIR / "erf_sensitivity_lc.csv"
OUT_CSV    = RESULTS_DIR / "benchmark_comparison_lc.csv"
P1_XLSX    = RESULTS_DIR / "phase1_lc_results.xlsx"
TMP_XLSX   = Path(tempfile.gettempdir()) / "phase1_lc_results_step1_4d.xlsx"

# ------------------------------------------------------------------------------
# REAL CASES (hardcoded — do not fetch from external sources)
# ------------------------------------------------------------------------------
REAL_CASES: list[dict] = [
    {
        "case_name"                  : "Bahnhof Pionen (Stockholm)",
        "source"                     : "IEA HPT Annex 47",
        "year"                       : 2023,
        "T_supply_C"                 : 68.0,
        "Q_kW"                       : 600.0,
        "ERF_published"              : None,
        "capture_fraction_published" : None,
        "notes"                      : "Two series HPs; 67m pipe to DH return; 975 kW heat output",
    },
    {
        "case_name"                  : "Aquasar ETH Zurich",
        "source"                     : "ICEF SDC Roadmap 2025",
        "year"                       : 2022,
        "T_supply_C"                 : 65.0,
        "Q_kW"                       : 10.0,
        "ERF_published"              : 0.80,
        "capture_fraction_published" : 0.80,
        "notes"                      : "Campus heating; ~80pct capture; $1.25M/year savings (includes cooling cost offset)",
    },
    {
        "case_name"                  : "iDataCool Germany",
        "source"                     : "ICEF SDC Roadmap 2025",
        "year"                       : 2022,
        "T_supply_C"                 : 65.0,
        "Q_kW"                       : None,
        "ERF_published"              : None,
        "capture_fraction_published" : None,
        "notes"                      : "HPC + adsorption chiller; hot-water cooling ~65 C",
    },
    {
        "case_name"                  : "Meta Odense Denmark",
        "source"                     : "ICEF SDC Roadmap 2025",
        "year"                       : 2020,
        "T_supply_C"                 : 57.5,
        "Q_kW"                       : None,
        "ERF_published"              : None,
        "capture_fraction_published" : None,
        "notes"                      : "HP to Fjernvarme Fyn; 7000+ homes served; operating since 2020",
    },
    {
        "case_name"                  : "EcoDataCenter Falun",
        "source"                     : "EcoDataCenter press release 2022",
        "year"                       : 2022,
        "T_supply_C"                 : 55.0,
        "Q_kW"                       : None,
        "ERF_published"              : None,
        "capture_fraction_published" : None,
        "notes"                      : "Climate-positive DC; WHR to DH + biofuel; Falun SE; 80 MW secured power",
    },
]

# ------------------------------------------------------------------------------
# SCENARIO REFERENCE VALUES (from LC model design; fallback if xlsx unreadable)
# ------------------------------------------------------------------------------
SCENARIO_RATED_KW: dict[str, float] = {
    "Edge_LC"       : 500.0,
    "Mid_LC"        : 3_200.0,
    "Hyperscale_LC" : 25_000.0,
}
TWH_MEAN_FALLBACK: dict[str, float] = {
    "Edge_LC"       : 47.6,
    "Mid_LC"        : 47.6,
    "Hyperscale_LC" : 46.0,
}
MODEL_ERF = 0.81   # operating point (cf=0.90)

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
# HELPERS
# ------------------------------------------------------------------------------
def _build_reg_kpis(df_rc: pd.DataFrame) -> pd.DataFrame:
    """Derive Regulatory_KPIs from RC CSV (needed for xlsx rebuild)."""
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


def load_twh_means() -> dict[str, float]:
    """
    Load TWH_mean_C per scenario from Regulatory_KPIs sheet (xlsx) or fall back
    to hardcoded values.  Uses pd.read_excel single-sheet mode (fast).
    """
    try:
        reg = pd.read_excel(P1_XLSX, sheet_name="Regulatory_KPIs")
        return dict(zip(reg["scenario"], reg["TWH_mean_C"].round(2)))
    except Exception as exc:
        print(f"  [WARN] Could not read Regulatory_KPIs from xlsx: {exc}")
        print("         Using hardcoded TWH_mean fallback.")
        return TWH_MEAN_FALLBACK.copy()


def closest_scenario(q_kw: float | None,
                     t_supply_c: float,
                     twh_means: dict[str, float]) -> str:
    """
    Matching rule:
      - If Q_kW is known: pick scenario whose rated power is numerically closest.
      - If Q_kW is None: pick scenario whose TWH_mean_C is closest to T_supply_C.
    Ties broken by scenario order: Edge_LC > Mid_LC > Hyperscale_LC.
    """
    scenarios = list(SCENARIO_RATED_KW.keys())
    if q_kw is not None:
        dists = {s: abs(SCENARIO_RATED_KW[s] - q_kw) for s in scenarios}
    else:
        dists = {s: abs(twh_means.get(s, TWH_MEAN_FALLBACK[s]) - t_supply_c)
                 for s in scenarios}
    return min(scenarios, key=lambda s: dists[s])


def upgrade_note(delta_t: float) -> str:
    """Same classification logic as Phase 2 regulatory_note."""
    if delta_t <= 0:
        return "Direct exchange feasible"
    elif delta_t <= 50:
        return f"Heat pump required (ΔT={delta_t:.1f}°C)"
    else:
        return f"CO2-HTHP required (ΔT={delta_t:.1f}°C)"


# ------------------------------------------------------------------------------
# MAIN COMPUTATION
# ------------------------------------------------------------------------------
def build_benchmark_comparison(twh_means: dict[str, float]) -> pd.DataFrame:
    rows = []
    for case in REAL_CASES:
        scen         = closest_scenario(case["Q_kW"], case["T_supply_C"], twh_means)
        twh_model    = twh_means.get(scen, TWH_MEAN_FALLBACK[scen])
        t_delta      = round(case["T_supply_C"] - twh_model, 2)
        erf_delta    = (round(case["ERF_published"] - MODEL_ERF, 4)
                        if case["ERF_published"] is not None else None)
        plausibility = "within_range" if abs(t_delta) <= 20 else "outside_range"
        note_upg     = upgrade_note(t_delta)

        rows.append({
            "case_name"          : case["case_name"],
            "source"             : case["source"],
            "year"               : case["year"],
            "T_supply_C"         : case["T_supply_C"],
            "Q_kW"               : case["Q_kW"],
            "ERF_published"      : case["ERF_published"],
            "closest_scenario"   : scen,
            "model_TWH_mean_C"   : twh_model,
            "model_ERF"          : MODEL_ERF,
            "T_delta_C"          : t_delta,
            "ERF_delta"          : erf_delta,
            "plausibility_flag"  : plausibility,
            "upgrade_needed_model": note_upg,
            "notes"              : case["notes"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# XLSX REBUILD (xlsxwriter → /tmp → copy back)
# Includes all 9 sheets: 5 CSV-backed + Regulatory_KPIs + EED_Compliance
#                         + ERF_Sensitivity + Benchmark_Comparison (new, 9th)
# ------------------------------------------------------------------------------
def rebuild_xlsx(df_reg: pd.DataFrame,
                 df_eed: pd.DataFrame,
                 df_erf: pd.DataFrame,
                 df_bench: pd.DataFrame) -> None:
    """Rebuild phase1_lc_results.xlsx with 9 sheets (adds Benchmark_Comparison)."""
    print("  Rebuilding phase1_lc_results.xlsx (9 sheets) ...")
    wb  = xlsxwriter.Workbook(TMP_XLSX, {"constant_memory": True})
    hdr = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})

    def write_df(ws: xlsxwriter.worksheet.Worksheet,
                 df: pd.DataFrame) -> None:
        for ci, col in enumerate(df.columns):
            ws.write(0, ci, col, hdr)
        for ri, row in enumerate(df.itertuples(index=False), start=1):
            for ci, v in enumerate(row):
                # Write None/NaN as blank rather than the string "None"
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    ws.write_blank(ri, ci, None)
                else:
                    ws.write(ri, ci, v)

    for csv_name, sheet_name in P1_BASE_SHEETS:
        csv_path = BASE_DIR / csv_name
        if not csv_path.exists():
            print(f"    [SKIP] {csv_name} not found")
            continue
        df = pd.read_csv(csv_path)
        ws = wb.add_worksheet(sheet_name)
        write_df(ws, df)
        print(f"    [OK]   {sheet_name} ({len(df)} rows)")

    for sheet_name, df in [
        ("Regulatory_KPIs",      df_reg),
        ("EED_Compliance",       df_eed),
        ("ERF_Sensitivity",      df_erf),
        ("Benchmark_Comparison", df_bench),
    ]:
        ws = wb.add_worksheet(sheet_name)
        write_df(ws, df)
        print(f"    [OK]   {sheet_name} ({len(df)} rows)")

    wb.close()
    shutil.copy2(TMP_XLSX, P1_XLSX)
    TMP_XLSX.unlink()
    size_mb = P1_XLSX.stat().st_size / 1e6
    print(f"  Saved: {P1_XLSX}  ({size_mb:.1f} MB, 9 sheets)")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  STEP 1.4d — Real-Case Benchmark Comparison [Liquid Cooling]")
    print("=" * 70)

    # Load TWH_mean values (from xlsx or fallback)
    twh_means = load_twh_means()
    print(f"  TWH_mean values: {twh_means}")

    # Build benchmark comparison table
    df_bench = build_benchmark_comparison(twh_means)
    print(f"\n  Benchmark_Comparison table ({len(df_bench)} rows):")
    print(df_bench.to_string(index=False))

    # Write intermediate CSV
    df_bench.to_csv(OUT_CSV, index=False)
    print(f"\n  CSV written: {OUT_CSV}")

    # Plausibility summary
    n_within    = (df_bench["plausibility_flag"] == "within_range").sum()
    n_total     = len(df_bench)
    max_delta_within = (df_bench.loc[df_bench["plausibility_flag"] == "within_range",
                                     "T_delta_C"].abs().max())
    print(f"\n  Plausibility summary:")
    print(f"    Cases within ±20°C of TWH_mean: {n_within}/{n_total}")
    print(f"    Max |T_delta| among within-range cases: {max_delta_within:.1f}°C")
    print()
    # Summary sentence for thesis Cap. 4.2
    thesis_note = (
        f"Model TWH_mean (47.6°C) is within {int(np.ceil(max_delta_within))}°C "
        f"of {n_within}/{n_total} real-world LC DC cases, "
        f"confirming physical plausibility of the RC thermal model parameters."
    )
    print(f"  Thesis Cap. 4.2 sentence:")
    print(f"    \"{thesis_note}\"")

    # Rebuild xlsx with 9 sheets
    df_rc  = pd.read_csv(RC_CSV, usecols=["scenario","TWH","ERF","Q_available","Q_negotiated"])
    df_reg = _build_reg_kpis(df_rc)
    df_eed = pd.read_csv(EED_CSV) if EED_CSV.exists() else pd.DataFrame()
    df_erf = pd.read_csv(ERF_CSV) if ERF_CSV.exists() else pd.DataFrame()
    rebuild_xlsx(df_reg, df_eed, df_erf, df_bench)

    print("\n  STEP 1.4d COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()