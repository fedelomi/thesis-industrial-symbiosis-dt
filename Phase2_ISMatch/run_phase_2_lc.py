# ==============================================================================
# DATACENTER DT PROJECT - PHASE 2 PIPELINE ORCHESTRATOR [LIQUID COOLING]
# Runs the full LC Phase 2 pipeline in sequence:
#
#   Step 2.0 LC  ->  step_2_0_epsilon_filter_lc.py      (ε-gate)
#   Step 2.2 LC  ->  step_2_2_dataset_builder_lc.py     (CE-HEAT dataset)
#   Step 2.1 LC  ->  step_2_1_is_match_score_lc.py      (IS-Match Score)
#   Step 2.3 LC  ->  step_2_3_ranking_validation_lc.py  (NDCG, precision@3)
#   Step 2.4 LC  ->  step_2_4_delta_tc_calibration_lc.py (ΔTC loop)
#
# Usage:
#   python run_phase_2_lc.py                  # re-run everything
#   python run_phase_2_lc.py --skip-existing  # skip steps whose output exists
#
# Final output:
#   phase2_lc_results.xlsx  — all CSV outputs consolidated into one workbook
# ==============================================================================

# Decisions active: D6 — Phase 2 LC-only orchestrator (D6 evidence-based exclusion).


import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Force UTF-8 on the orchestrator's own stdout/stderr so Unicode markers
# (e.g. epsilon, arrow, Delta) render on Windows cp1252 consoles too.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

_ENV_UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Paths in 'output' lists are relative to RESULTS_DIR (post-reorg, audit 2026-05-05).
# step_2_2 produces 'step_2_2_ce_heat_dataset_lc.csv' as canonical name; the legacy
# 'ce_heat_inspired_dataset_lc.csv' lives in data/ as the committed snapshot.
STEPS = [
    {
        "name"   : "Step 2.0 LC — ε-parameter Pre-screening Gate",
        "script" : "step_2_0_epsilon_filter_lc.py",
        "output" : ["step_2_0_epsilon_gate_lc.csv"],
    },
    {
        "name"   : "Step 2.2 LC — CE-HEAT Inspired Dataset Builder",
        "script" : "step_2_2_dataset_builder_lc.py",
        "output" : ["step_2_2_ce_heat_dataset_lc.csv",
                    "step_2_2_feasibility_summary_lc.csv"],
    },
    {
        "name"   : "Step 2.1 LC — IS-Match Score Computation",
        "script" : "step_2_1_is_match_score_lc.py",
        "output" : ["step_2_1_is_match_scores_lc.csv",
                    "step_2_1_sensitivity_lc.csv"],
    },
    {
        "name"   : "Step 2.3 LC — Ranking Validation (NDCG, precision@3)",
        "script" : "step_2_3_ranking_validation_lc.py",
        "output" : ["step_2_3_ranking_metrics_lc.csv",
                    "step_2_3_sensitivity_ranking_lc.csv"],
    },
    {
        "name"   : "Step 2.4 LC — ΔTC Calibration Loop (Phase 3 stub)",
        "script" : "step_2_4_delta_tc_calibration_lc.py",
        "output" : ["step_2_4_delta_tc_calibration_lc.csv",
                    "step_2_4_convergence_log_lc.csv"],
    },
    # ── Robustness layer (additive, post-baseline) ───────────────────────────
    {
        "name"   : "Step 2.5 LC — Sobol Sensitivity on IS-Match Weights",
        "script" : "step_2_5_sobol_weights.py",
        "output" : ["step_2_5_sobol_weights.csv"],
    },
    {
        "name"   : "Step 2.6 LC — Stress Test (+/- 20% on 4 inputs)",
        "script" : "step_2_6_stress_test.py",
        "output" : ["step_2_6_stress_test.csv"],
    },
    {
        "name"   : "Step 2.7 LC — CO2 Avoided per Pair",
        "script" : "step_2_7_carbon_emissions_avoided.py",
        "output" : ["step_2_7_carbon_emissions_avoided.csv"],
    },
    {
        "name"   : "Step 2.8 LC — PROMETHEE II Cross-Check vs IS-Match",
        "script" : "step_2_8_promethee_crosscheck.py",
        "output" : ["step_2_8_promethee_ranking.csv",
                    "step_2_8_promethee_correlation.csv"],
    },
]

EXCEL_SHEETS = [
    ("step_2_0_epsilon_gate_lc.csv",           "Epsilon_Gate_LC"),
    ("step_2_2_ce_heat_dataset_lc.csv",        "Dataset_CE-HEAT_LC"),
    ("step_2_2_feasibility_summary_lc.csv",    "Dataset_Summary_LC"),
    ("step_2_1_is_match_scores_lc.csv",        "IS_Match_Scores_LC"),
    ("step_2_1_sensitivity_lc.csv",            "IS_Match_Sensitivity_LC"),
    ("step_2_3_ranking_metrics_lc.csv",        "Ranking_Metrics_LC"),
    ("step_2_3_sensitivity_ranking_lc.csv",    "Ranking_Sensitivity_LC"),
    ("step_2_4_delta_tc_calibration_lc.csv",   "DeltaTC_Calibration_LC"),
    ("step_2_4_convergence_log_lc.csv",        "DeltaTC_Convergence_LC"),
    ("step_2_5_sobol_weights.csv",             "Sobol_Weights_LC"),
    ("step_2_6_stress_test.csv",               "Stress_Test_LC"),
    ("step_2_7_carbon_emissions_avoided.csv",  "Carbon_Avoided_LC"),
    ("step_2_8_promethee_ranking.csv",         "PROMETHEE_Ranking_LC"),
    ("step_2_8_promethee_correlation.csv",     "PROMETHEE_Correlation_LC"),
]

# Final consolidated workbook lives in results/ alongside the per-step CSVs.
EXCEL_OUTPUT = Path("results") / "phase2_lc_results.xlsx"


def sep(char="=", n=70):
    print(char * n)


def fmt_size(path):
    try:
        b = Path(path).stat().st_size
        return f"{b/1e6:.1f} MB" if b >= 1e6 else (f"{b/1e3:.0f} KB" if b >= 1e3 else f"{b} B")
    except Exception:
        return "n/a"


def outputs_exist(step):
    """Outputs are produced by sub-scripts inside RESULTS_DIR."""
    return all((RESULTS_DIR / f).exists() for f in step["output"])


def run_step(step, idx, total, skip_existing):
    script_path = BASE_DIR / step["script"]
    sep()
    print(f"[{idx}/{total}] {step['name']}")
    sep("-")
    sys.stdout.flush()

    if skip_existing and outputs_exist(step):
        for f in step["output"]:
            p = RESULTS_DIR / f
            print(f"  [SKIP]  {f:<50} {fmt_size(p):>8}  (already present)")
        print(f"\n  Time: 0.0s  (skipped)")
        return True, 0.0

    if not script_path.exists():
        print(f"  ERROR: script not found -- {script_path}")
        return False, 0.0

    t0     = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, cwd=str(BASE_DIR),
        env=_ENV_UTF8, encoding="utf-8",
    )
    elapsed = time.time() - t0

    for line in result.stdout.splitlines():
        print(f"  {line}")
    if result.stderr.strip():
        print("  --- STDERR ---")
        for line in result.stderr.splitlines():
            print(f"  {line}")

    print()
    all_ok = True
    for f in step["output"]:
        p = RESULTS_DIR / f
        if p.exists():
            print(f"  [OK]    {f:<50} {fmt_size(p):>8}")
        else:
            print(f"  [MISSING] {f}")
            all_ok = False

    if result.returncode != 0:
        print(f"\n  EXIT CODE: {result.returncode} -- step FAILED")
        all_ok = False

    print(f"\n  Time: {elapsed:.1f}s")
    return all_ok, elapsed


def build_p2_regulatory_kpis() -> pd.DataFrame:
    """
    Build the Phase 2 Regulatory_KPIs sheet (27 rows = 3 DC × 9 individual plants).

    Sources:
      step_2_2_ce_heat_dataset_lc.csv    -- 27 plant rows (plant metadata)
      step_2_4_delta_tc_calibration_lc.csv -- IS-Match scores, final iteration only
      phase1_lc_results.xlsx Regulatory_KPIs  -- TWH_mean, ERF per DC scenario
      phase1_lc_results.xlsx EED_Compliance   -- Art26_applicable per DC scenario

    Reference: EU Delegated Regulation 2024/1364, EED Art.12 / Art.26 (2023/1791)
    """
    import numpy as np

    # Repo layout (post-reorg, audit 2026-05-05):
    #   Github/Phase1_PhysicalDT/lc/results/phase1_lc_results.xlsx
    #   Github/Phase2_ISMatch/run_phase_2_lc.py   <- this script
    BASE          = BASE_DIR
    PHASE1_XLSX   = BASE.parent / "Phase1_PhysicalDT" / "lc" / "results" / "phase1_lc_results.xlsx"

    # ── 1. Load 27-plant dataset ─────────────────────────────────────────────
    # Look in results/ (regenerated outputs) first, then data/ (committed input).
    results_dir = BASE / "results"
    data_dir    = BASE / "data"
    ds_path = results_dir / "step_2_2_ce_heat_dataset_lc.csv"
    if not ds_path.exists():
        ds_path = data_dir / "ce_heat_inspired_dataset_lc.csv"
    if not ds_path.exists():
        # Backward compatibility with pre-reorg checkouts (flat Phase2/)
        ds_path = BASE / "step_2_2_ce_heat_dataset_lc.csv"
    if not ds_path.exists():
        ds_path = BASE / "ce_heat_inspired_dataset_lc.csv"
    ds = pd.read_csv(ds_path, usecols=[
        "dc_name", "plant_name", "plant_tier", "plant_t_req_c",
        "plant_q_demand_kw", "distance_km",
    ])

    # ── 2. Calibration scores — final iteration only ─────────────────────────
    # ΔTC calibration converges at N_iter=4; column 'iteration' marks each pass.
    cal = pd.read_csv(results_dir / "step_2_4_delta_tc_calibration_lc.csv")
    final_iter = cal["iteration"].max()
    cal_final  = cal[cal["iteration"] == final_iter][
        ["dc_name", "process_name", "is_match_score", "tier"]
    ].rename(columns={"process_name": "plant_tier"})

    # ── 3. Phase 1 Regulatory_KPIs (TWH, ERF, Q_negotiated per DC scenario) ─
    try:
        reg_p1 = pd.read_excel(PHASE1_XLSX, sheet_name="Regulatory_KPIs")
        reg_p1 = reg_p1.rename(columns={
            "scenario"           : "dc_name",
            "TWH_mean_C"         : "TWH_mean_C",
            "ERF_annual_mean"    : "ERF_scenario",
            "Q_negotiated_mean_kW": "Q_negotiated_mean_kW",
        })[["dc_name", "TWH_mean_C", "ERF_scenario", "Q_negotiated_mean_kW"]]
    except Exception as e:
        print(f"  [WARN]  Could not read Regulatory_KPIs from Phase1 xlsx: {e}")
        print("          Falling back to lc_dc_results_annual.csv ...")
        rc_csv = BASE.parent / "Phase1_PhysicalDT" / "lc" / "results" / "lc_dc_results_annual.csv"
        df_rc  = pd.read_csv(rc_csv, usecols=["scenario","TWH","ERF","Q_negotiated"])
        reg_p1 = (
            df_rc.groupby("scenario")
            .agg(TWH_mean_C=("TWH","mean"), ERF_scenario=("ERF","mean"),
                 Q_negotiated_mean_kW=("Q_negotiated", lambda x: x.mean()/1e3))
            .reset_index().rename(columns={"scenario":"dc_name"})
        )

    # ── 4. Art26_applicable from EED_Compliance sheet (fallback: hardcoded) ──
    # Hardcoded fallback reflects that all three scenarios have total rated > 1 MW
    # (Edge_LC 590 kW is below 1 MW threshold → False; Mid/Hyperscale → True).
    # The EED_Compliance sheet (step_0) uses the correct threshold of > 1,000 kW.
    ART26_FALLBACK = {
        "Edge_LC"       : False,   # total rated 590 kW < 1,000 kW
        "Mid_LC"        : True,
        "Hyperscale_LC" : True,
    }
    try:
        eed = pd.read_excel(PHASE1_XLSX, sheet_name="EED_Compliance")
        art26_map = dict(zip(eed["scenario"], eed["Art26_applicable"]))
    except Exception:
        print("  [WARN]  EED_Compliance sheet not found; using hardcoded Art26 values.")
        art26_map = ART26_FALLBACK

    # ── 5. Join all sources → 27-row result ──────────────────────────────────
    # Join dataset with calibration scores on (dc_name, plant_tier)
    merged = ds.merge(cal_final, on=["dc_name", "plant_tier"], how="left")
    # Join with Phase 1 Regulatory KPIs
    merged = merged.merge(reg_p1, on="dc_name", how="left")

    # ── 6. Derived columns ───────────────────────────────────────────────────
    # delta_T_residual_C: positive → heat pump needed; negative → direct exchange
    merged["delta_T_residual_C"] = (
        merged["plant_t_req_c"] - merged["TWH_mean_C"]
    ).round(2)

    # ERF_pair: capture fraction is DC-side only, independent of the downstream
    # industrial sink. Therefore ERF_pair = ERF_scenario for all pairs.
    # ERF_pair = Q_negotiated / (Q_negotiated / ERF_scenario) = ERF_scenario.
    merged["ERF_pair"] = merged["ERF_scenario"]

    # Art26_applicable: from EED_Compliance, keyed by dc_name
    merged["Art26_applicable"] = merged["dc_name"].map(art26_map)

    # IS-Match category from score thresholds
    def score_category(s: float) -> str:
        if s > 0.60:
            return "high-priority"
        elif s >= 0.30:
            return "marginal"
        else:
            return "not-viable"

    merged["is_match_category"] = merged["is_match_score"].apply(score_category)

    # Regulatory note based on delta_T_residual_C
    def reg_note(dt: float) -> str:
        if dt <= 0:
            return "Direct exchange feasible (ΔT < 0)"
        elif dt <= 50:
            return f"Heat pump required (ΔT = {round(dt)}°C)"
        else:
            return f"CO2-HTHP required (ΔT = {round(dt)}°C)"

    merged["regulatory_note"] = merged["delta_T_residual_C"].apply(reg_note)

    # ── 7. Select and rename final columns ───────────────────────────────────
    result = merged.rename(columns={
        "dc_name"            : "dc_scenario",
        "plant_name"         : "plant_id",
        "plant_tier"         : "t_tier",
        "plant_t_req_c"      : "T_req_C",
        "plant_q_demand_kw"  : "Q_demand_plant_kW",
    })[[
        "dc_scenario", "plant_id", "t_tier", "distance_km",
        "TWH_mean_C", "T_req_C", "delta_T_residual_C",
        "ERF_scenario", "Q_negotiated_mean_kW", "Q_demand_plant_kW",
        "ERF_pair", "Art26_applicable",
        "is_match_score", "is_match_category",
        "regulatory_note",
    ]]

    # Round numeric output columns
    result["is_match_score"]    = result["is_match_score"].round(4)
    result["Q_demand_plant_kW"] = result["Q_demand_plant_kW"].round(1)

    return result.reset_index(drop=True)


def consolidate_to_excel():
    sep()
    print("  CONSOLIDATING OUTPUTS -> results/phase2_lc_results.xlsx")
    sep("-")
    xlsx_path = BASE_DIR / EXCEL_OUTPUT
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for csv_name, sheet_name in EXCEL_SHEETS:
            csv_path = RESULTS_DIR / csv_name
            if not csv_path.exists():
                print(f"  [SKIP]   {csv_name} not found -- sheet '{sheet_name}' omitted")
                continue
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [OK]     {sheet_name:<30} <- {csv_name}  ({len(df)} rows)")

    # ── Regulatory_KPIs sheet (Phase 2) ─────────────────────────────────────
    # Appended separately with openpyxl mode='a' to avoid re-writing all sheets.
    # 27 rows = 3 DC scenarios × 9 individual manufacturing plants.
    # Reference: EU Delegated Regulation 2024/1364 / EED Art.12 & Art.26
    try:
        reg_kpis = build_p2_regulatory_kpis()
        with pd.ExcelWriter(
            xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer_app:
            reg_kpis.to_excel(writer_app, sheet_name="Regulatory_KPIs", index=False)
        print(f"  [OK]     {'Regulatory_KPIs':<30} (derived, {len(reg_kpis)} rows)")
        print()
        print("  Regulatory_KPIs preview:")
        print(reg_kpis.to_string(index=False))
    except Exception as exc:
        print(f"  [WARN]   Regulatory_KPIs sheet skipped: {exc}")

    sz = xlsx_path.stat().st_size
    print(f"\n  Saved: {xlsx_path}  ({sz/1e3:.0f} KB)")
    sys.stdout.flush()


def main():
    skip_existing = "--skip-existing" in sys.argv
    sep("=")
    print("  DATACENTER DT — PHASE 2 LIQUID COOLING PIPELINE")
    print(f"  Directory : {BASE_DIR}")
    print(f"  Mode      : {'skip-existing' if skip_existing else 'rerun-all'}")
    sep("=")
    print()
    sys.stdout.flush()

    t0      = time.time()
    results = []

    for idx, step in enumerate(STEPS, start=1):
        ok, elapsed = run_step(step, idx, len(STEPS), skip_existing)
        results.append((step["name"], ok, elapsed))
        print()
        sys.stdout.flush()
        if not ok:
            print(f"  Pipeline stopped at step {idx}. Fix errors before continuing.")
            break

    sep("=")
    print("  FINAL SUMMARY")
    sep("=")
    all_pass = all(ok for _, ok, _ in results)
    for name, ok, elapsed in results:
        tag      = "PASS" if ok else "FAIL"
        skip_tag = "  (skipped)" if elapsed == 0.0 and ok else ""
        print(f"  [{tag}]  {name:<55}  {elapsed:>5.1f}s{skip_tag}")

    print()
    print("  Output files (in results/):")
    for step in STEPS:
        for f in step["output"]:
            p = RESULTS_DIR / f
            if p.exists():
                size_kb = p.stat().st_size / 1024.0
                print(f"    {f:<55} {size_kb:>8.1f} KB")

    print()
    print(f"  Total time: {time.time()-t0:.1f}s")
    print()
    if all_pass:
        print("  GATE PHASE 2: PASS")
        print("  Ready for Phase 4 (Phase 3 KG ingest is independent).")
        consolidate_to_excel()
    else:
        print("  GATE PHASE 2: FAIL")
        print("  Fix the failed step(s) before running Phase 4.")


if __name__ == "__main__":
    main()
  