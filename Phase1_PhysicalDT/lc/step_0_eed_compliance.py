# ==============================================================================
# DATACENTER DT PROJECT - STEP 0  [LIQUID COOLING]
# EED Compliance Matrix — EU Delegated Regulation 2024/1364 (implements EED Art.12)
#
# Reads:  lc_dc_results_annual.csv  (primary source for Regulatory KPIs data,
#         identical to what is stored in the Regulatory_KPIs sheet of
#         phase1_lc_results.xlsx — this avoids openpyxl re-parsing the large xlsx)
#
# Computes:
#   - EU size category per scenario (Annex IV, Del. Reg. 2024/1364)
#   - Art. 12 EED 2023/1791 reporting obligation (IT power >= 500 kW)
#   - Art. 26 EED 2023/1791 WHR utilisation obligation (total rated > 1,000 kW)
#   - WHR active status and Art. 26 exception flag
#   - TEE incentive estimate (Italian Certificati Bianchi, scaled from Mid_LC ref.)
#
# Outputs:
#   eed_compliance_lc.csv                  (intermediate CSV for traceability)
#   phase1_lc_results.xlsx (7 sheets)     (rebuilt with EED_Compliance added)
# ==============================================================================

# Decision active: D5 — LC scenarios from grey-box parametrization on LC-Opt FMU.
#                       Wiki: [[implementation-decisions#D5]].


import logging
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xlsxwriter

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RC_CSV     = RESULTS_DIR / "lc_dc_results_annual.csv"
EED_CSV    = RESULTS_DIR / "eed_compliance_lc.csv"
P1_XLSX    = RESULTS_DIR / "phase1_lc_results.xlsx"
TMP_XLSX   = Path(tempfile.gettempdir()) / "phase1_lc_results_step0.xlsx"

# ------------------------------------------------------------------------------
# KNOWN SCENARIO PARAMETERS
# PUE estimate = IT power / 0.85 for all scenarios (conservative liquid-cooling PUE)
# Reference: LC-Opt FMU model; design choices D1, D5 in project docs.
# ------------------------------------------------------------------------------
SCENARIO_PARAMS: dict[str, dict] = {
    "Edge_LC": {
        "it_power_kw"   : 500,
        "total_rated_kw": 590,      # 500 / 0.85 PUE ≈ 588 → rounded to 590
    },
    "Mid_LC": {
        "it_power_kw"   : 3_200,
        "total_rated_kw": 3_765,    # 3200 / 0.85 ≈ 3765
    },
    "Hyperscale_LC": {
        "it_power_kw"   : 25_000,
        "total_rated_kw": 29_412,   # 25000 / 0.85 ≈ 29412
    },
}

# TEE reference: 360,000 EUR/year for Mid_LC (3,200 kW IT)
# Source: Gestore Servizi Energetici (GSE) Certificati Bianchi literature estimate
TEE_REFERENCE_EUR   = 360_000
TEE_REFERENCE_SCEN  = "Mid_LC"

# Phase 1 CSV → Excel sheet mapping (must stay in sync with run_phase_1_lc.py)
P1_SHEETS = [
    ("lc_dc_results_annual.csv",           "RC_Model_Results_LC"),
    ("lc_synthetic_profile_annual.csv",     "Synthetic_Profile_LC"),
    ("lc_real_vs_synthetic_comparison.csv", "Real_vs_Synthetic_LC"),
    ("lc_validation_1_4.csv",              "Validation_LC"),
    ("lc_sensitivity_validation.csv",       "Sensitivity_Analysis_LC"),
]

# EU size categories — Annex IV, EU Delegated Regulation 2024/1364
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
# HELPER: EU size category from IT power [kW]
# ------------------------------------------------------------------------------
def eu_size_from_it_kw(it_kw: float) -> str:
    """Return EU size category string per Annex IV, Del. Reg. 2024/1364."""
    if it_kw <= 500:
        return "Very small (<=500 kW)"
    elif it_kw <= 1_000:
        return "Small (500 kW-1 MW)"
    elif it_kw <= 2_000:
        return "Medium (1-2 MW)"
    elif it_kw <= 10_000:
        return "Large (2-10 MW)"
    else:
        return "Very large (>10 MW)"


# ------------------------------------------------------------------------------
# STEP 1 — Derive Regulatory KPIs from RC CSV (same data as xlsx sheet)
# ------------------------------------------------------------------------------
def build_regulatory_kpis(df_rc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate lc_dc_results_annual.csv into per-scenario Regulatory KPIs.
    Mirrors the Regulatory_KPIs sheet already present in phase1_lc_results.xlsx.
    """
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
# STEP 2 — Compute EED Compliance Matrix
# ------------------------------------------------------------------------------
def build_eed_compliance(reg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EED compliance matrix for all LC scenarios.

    References:
      Art. 12 EED 2023/1791  — reporting obligation: IT power >= 500 kW
      Art. 26 EED 2023/1791  — WHR utilisation obligation: total rated > 1,000 kW
      Del. Reg. 2024/1364    — defines TWH, ERF, size categories
      TEE (Certificati Bianchi, GSE) — scaled from 360,000 EUR/year reference for Mid_LC
    """
    # TEE denominator: Q_negotiated_mean_kW of the reference scenario (Mid_LC)
    ref_mask      = reg["scenario"] == TEE_REFERENCE_SCEN
    q_neg_ref_kw  = float(reg.loc[ref_mask, "Q_negotiated_mean_kW"].iloc[0])

    rows = []
    for scenario, params in SCENARIO_PARAMS.items():
        it_kw    = params["it_power_kw"]
        total_kw = params["total_rated_kw"]

        mask       = reg["scenario"] == scenario
        erf_mean   = float(reg.loc[mask, "ERF_annual_mean"].iloc[0])
        q_neg_kw   = float(reg.loc[mask, "Q_negotiated_mean_kW"].iloc[0])

        # EU size category (recomputed from IT power for traceability)
        size_cat = eu_size_from_it_kw(it_kw)

        # Art. 12 — reporting obligation
        art12       = it_kw >= 500
        art12_note  = ("Reporting required (IT >= 500 kW)"
                       if art12 else "Below threshold (IT < 500 kW)")

        # Art. 26 — WHR utilisation obligation
        art26       = total_kw > 1_000
        art26_note  = (f"WHR obligation applies (total {total_kw:,} kW > 1,000 kW)"
                       if art26 else f"Below threshold (total {total_kw:,} kW <= 1,000 kW)")

        # WHR status and Art. 26 exception
        whr_active         = erf_mean > 0
        art26_exception    = art26 and not whr_active

        # TEE estimate (Certificati Bianchi) — linear scaling by Q_negotiated
        tee_raw     = TEE_REFERENCE_EUR * (q_neg_kw / q_neg_ref_kw)
        tee_rounded = int(round(tee_raw / 1_000) * 1_000)

        note = f"{art12_note}; {art26_note}"

        rows.append({
            "scenario"                : scenario,
            "IT_power_kW"             : it_kw,
            "total_rated_kW"          : total_kw,
            "EU_size_category"        : size_cat,
            "Art12_applicable"        : art12,
            "Art26_applicable"        : art26,
            "WHR_active"              : whr_active,
            "ERF_mean"                : round(erf_mean, 4),
            "Art26_exception_required": art26_exception,
            "TEE_estimated_eur_year"  : tee_rounded,
            "note"                    : note,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# STEP 3 — Rebuild phase1_lc_results.xlsx (xlsxwriter, write to /tmp then copy)
# This is required because openpyxl cannot efficiently append to a large
# xlsxwriter-generated file (18 MB). The rebuild approach is already validated
# by _gen_excel_fast.py and produces an identical, clean xlsx.
# ------------------------------------------------------------------------------
def rebuild_xlsx(reg: pd.DataFrame, eed: pd.DataFrame) -> None:
    """Rebuild phase1_lc_results.xlsx with 7 sheets (adds EED_Compliance)."""
    logger.info("  Rebuilding phase1_lc_results.xlsx ...")
    hdr_fmt_opts = {"bold": True, "bg_color": "#D9E1F2", "border": 1}

    wb = xlsxwriter.Workbook(TMP_XLSX, {"constant_memory": True})
    hdr = wb.add_format(hdr_fmt_opts)

    def write_df(ws: xlsxwriter.worksheet.Worksheet,
                 df: pd.DataFrame) -> None:
        for ci, col in enumerate(df.columns):
            ws.write(0, ci, col, hdr)
        for ri, row in enumerate(df.itertuples(index=False), start=1):
            for ci, v in enumerate(row):
                ws.write(ri, ci, v)

    # ── original CSV-backed sheets ───────────────────────────────────────────
    for csv_name, sheet_name in P1_SHEETS:
        csv_path = BASE_DIR / csv_name
        if not csv_path.exists():
            logger.info(f"    [SKIP] {csv_name} not found")
            continue
        df = pd.read_csv(csv_path)
        ws = wb.add_worksheet(sheet_name)
        write_df(ws, df)
        logger.info(f"    [OK]   {sheet_name} ({len(df)} rows)")

    # ── Regulatory_KPIs (derived from CSV, same as original sheet) ───────────
    ws_reg = wb.add_worksheet("Regulatory_KPIs")
    write_df(ws_reg, reg)
    logger.info(f"    [OK]   Regulatory_KPIs ({len(reg)} rows)")

    # ── EED_Compliance (new sheet) ───────────────────────────────────────────
    ws_eed = wb.add_worksheet("EED_Compliance")
    write_df(ws_eed, eed)
    logger.info(f"    [OK]   EED_Compliance ({len(eed)} rows)")

    wb.close()
    shutil.copy2(TMP_XLSX, P1_XLSX)
    TMP_XLSX.unlink()
    size_mb = P1_XLSX.stat().st_size / 1e6
    logger.info(f"  Saved: {P1_XLSX}  ({size_mb:.1f} MB, 7 sheets)")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 70)
    logger.info("  STEP 0 — EED Compliance Matrix [Liquid Cooling]")
    logger.info("=" * 70)

    # Load RC model output (provides TWH, ERF, Q_available, Q_negotiated)
    if not RC_CSV.exists():
        raise FileNotFoundError(f"RC CSV not found: {RC_CSV}")
    df_rc = pd.read_csv(
        RC_CSV, usecols=["scenario", "TWH", "ERF", "Q_available", "Q_negotiated"]
    )
    logger.info(f"  Loaded {len(df_rc):,} rows from {RC_CSV.name}")

    # Build Regulatory KPIs (mirrors Regulatory_KPIs sheet)
    reg = build_regulatory_kpis(df_rc)

    # Compute EED compliance matrix
    eed = build_eed_compliance(reg)
    logger.info("\n  EED Compliance Matrix:")
    logger.info(eed.to_string(index=False))

    # Write intermediate CSV
    eed.to_csv(EED_CSV, index=False)
    logger.info(f"\n  CSV written: {EED_CSV}")

    # Rebuild xlsx with 7 sheets
    rebuild_xlsx(reg, eed)

    logger.info("\n  STEP 0 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()