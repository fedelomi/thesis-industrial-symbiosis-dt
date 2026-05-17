# ==============================================================================
# DATACENTER DT PROJECT - PHASE 1 PIPELINE ORCHESTRATOR (LIQUID COOLING)
# Runs the full LC pipeline in sequence:
#
#   Step 1.1+1.2+1.2-bis  ->  step_1_1_2_rc_model_lc.py
#   Step 1.3              ->  step_1_3_synthetic_lc.py
#   Step 1.4              ->  step_1_4_validation_lc.py
#   Step 1.4b             ->  step_1_4b_sensitivity_lc.py
#
# Usage:
#   python run_phase_1_lc.py                  # re-run everything
#   python run_phase_1_lc.py --skip-existing  # skip steps whose output already exists
#
# Final output:
#   phase1_lc_results.xlsx  -- all LC CSV outputs consolidated into one workbook
# ==============================================================================

# Decisions active: D5 — grey-box LC parametrization (LC-Opt FMU).
#                   Outputs feed into D6 — LC-only Phase 2 pipeline.


import os, sys, subprocess, time
import pandas as pd

# Force UTF-8 in subprocesses on Windows (avoids UnicodeEncodeError)
_ENV_UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

# Output paths (post-reorg, audit 2026-05-05) — CSVs/XLSX live in results/.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STEPS = [
    {
        "name"   : "Step 1.1+1.2+1.2-bis -- LC RC Thermal Model + Exergy",
        "script" : "step_1_1_2_rc_model_lc.py",
        "output" : ["lc_dc_results_annual.csv"],
    },
    {
        "name"   : "Step 1.3 -- LC Privacy-Preserving Synthetic Generator",
        "script" : "step_1_3_synthetic_lc.py",
        "output" : ["lc_synthetic_profile_annual.csv", "lc_real_vs_synthetic_comparison.csv"],
    },
    {
        "name"   : "Step 1.4 -- LC Validation W1 + NDE (Phase 2 gate)",
        "script" : "step_1_4_validation_lc.py",
        "output" : ["lc_validation_1_4.csv"],
    },
    {
        "name"   : "Step 1.4b -- LC Sensitivity Analysis Privacy-Fidelity",
        "script" : "step_1_4b_sensitivity_lc.py",
        "output" : ["lc_sensitivity_validation.csv"],
    },
]

# Maps each CSV output to its Excel sheet name
EXCEL_SHEETS = [
    ("lc_dc_results_annual.csv",          "RC_Model_Results_LC"),
    ("lc_synthetic_profile_annual.csv",    "Synthetic_Profile_LC"),
    ("lc_real_vs_synthetic_comparison.csv","Real_vs_Synthetic_LC"),
    ("lc_validation_1_4.csv",             "Validation_LC"),
    ("lc_sensitivity_validation.csv",      "Sensitivity_Analysis_LC"),
]

EXCEL_OUTPUT = os.path.join("results", "phase1_lc_results.xlsx")


def sep(char="=", n=70):
    print(char * n)


def fmt_size(path):
    try:
        b = os.path.getsize(path)
        return f"{b/1e6:.1f} MB" if b >= 1e6 else (f"{b/1e3:.0f} KB" if b >= 1e3 else f"{b} B")
    except Exception:
        return "n/a"


def outputs_exist(step):
    """Outputs are produced inside RESULTS_DIR (post-reorg, audit 2026-05-05)."""
    return all(os.path.exists(os.path.join(RESULTS_DIR, f)) for f in step["output"])


def run_step(step, idx, total, skip_existing):
    script_path = os.path.join(BASE_DIR, step["script"])
    sep()
    print(f"[{idx}/{total}] {step['name']}")
    sep("-")
    sys.stdout.flush()

    # Skip mode
    if skip_existing and outputs_exist(step):
        for f in step["output"]:
            p = os.path.join(RESULTS_DIR, f)
            print(f"  [SKIP]  {f:<45} {fmt_size(p):>8}  (already present)")
        print(f"\n  Time: 0.0s  (skipped)")
        return True, 0.0

    if not os.path.exists(script_path):
        print(f"  ERROR: script not found -- {script_path}")
        return False, 0.0

    t0     = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=BASE_DIR,
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
        p = os.path.join(RESULTS_DIR, f)
        if os.path.exists(p):
            print(f"  [OK]    {f:<45} {fmt_size(p):>8}")
        else:
            print(f"  [MISSING] {f}")
            all_ok = False

    if result.returncode != 0:
        print(f"\n  EXIT CODE: {result.returncode} -- step FAILED")
        all_ok = False

    print(f"\n  Time: {elapsed:.1f}s")
    return all_ok, elapsed


def consolidate_to_excel():
    """Write all LC CSV outputs as sheets into a single Excel workbook."""
    sep()
    print("  CONSOLIDATING OUTPUTS -> phase1_lc_results.xlsx")
    sep("-")

    # EU Delegated Regulation 2024/1364 size categories (EED Art.12, Annex)
    # Boundaries: Very small 100-500 kW | Small 500 kW-1 MW |
    #             Medium 1-2 MW | Large 2-10 MW | Very large >10 MW
    EU_SIZE_CATEGORY = {
        "Edge_LC"        : "Very small (≤500 kW)",
        "Mid_LC"         : "Large (2–10 MW)",
        "Hyperscale_LC"  : "Very large (>10 MW)",
    }
    RATED_POWER_MW = {
        "Edge_LC"        : 0.5,
        "Mid_LC"         : 3.2,
        "Hyperscale_LC"  : 25.0,
    }

    xlsx_path = os.path.join(BASE_DIR, EXCEL_OUTPUT)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # ── existing sheets (unchanged) ──────────────────────────────────
        for csv_name, sheet_name in EXCEL_SHEETS:
            csv_path = os.path.join(RESULTS_DIR, csv_name)
            if not os.path.exists(csv_path):
                print(f"  [SKIP]   {csv_name} not found -- sheet '{sheet_name}' omitted")
                continue
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [OK]     {sheet_name:<28} <- {csv_name}  ({len(df)} rows)")

        # ── Regulatory_KPIs sheet ────────────────────────────────────────
        # Source: lc_dc_results_annual.csv (requires TWH and ERF columns
        # generated by step_1_1_2_rc_model_lc.py after EU reg update).
        # Reference: EU Delegated Regulation 2024/1364 implementing EED Art.12
        rc_csv = os.path.join(RESULTS_DIR, "lc_dc_results_annual.csv")
        if os.path.exists(rc_csv):
            df_rc = pd.read_csv(rc_csv)
            if {"TWH", "ERF", "Q_available", "Q_negotiated"}.issubset(df_rc.columns):
                reg = (
                    df_rc.groupby("scenario", sort=False)
                    .agg(
                        Rated_Power_MW    =("scenario", lambda s: RATED_POWER_MW.get(s.iloc[0], float("nan"))),
                        EU_Size_Category  =("scenario", lambda s: EU_SIZE_CATEGORY.get(s.iloc[0], "Unknown")),
                        # TWH: annual mean/min/max of CDU return temperature [°C]
                        TWH_mean_C        =("TWH", "mean"),
                        TWH_min_C         =("TWH", "min"),
                        TWH_max_C         =("TWH", "max"),
                        # ERF: Energy Reuse Factor annual mean (EREUSE/EDC)
                        ERF_annual_mean   =("ERF", "mean"),
                        # Supporting quantities [kW]
                        Q_available_mean_kW  =("Q_available",  lambda x: x.mean() / 1e3),
                        Q_negotiated_mean_kW =("Q_negotiated", lambda x: x.mean() / 1e3),
                    )
                    .reset_index()
                )
                # Round to readable precision
                reg["TWH_mean_C"]           = reg["TWH_mean_C"].round(2)
                reg["TWH_min_C"]            = reg["TWH_min_C"].round(2)
                reg["TWH_max_C"]            = reg["TWH_max_C"].round(2)
                reg["ERF_annual_mean"]      = reg["ERF_annual_mean"].round(4)
                reg["Q_available_mean_kW"]  = reg["Q_available_mean_kW"].round(1)
                reg["Q_negotiated_mean_kW"] = reg["Q_negotiated_mean_kW"].round(1)

                reg.to_excel(writer, sheet_name="Regulatory_KPIs", index=False)
                print(f"  [OK]     {'Regulatory_KPIs':<28} <- derived from lc_dc_results_annual.csv  ({len(reg)} rows)")
            else:
                print("  [SKIP]   Regulatory_KPIs -- TWH/ERF columns missing from lc_dc_results_annual.csv")
        else:
            print("  [SKIP]   Regulatory_KPIs -- lc_dc_results_annual.csv not found")

    print(f"\n  Saved: {xlsx_path}  ({fmt_size(xlsx_path)})")
    sys.stdout.flush()


def main():
    skip_existing = "--skip-existing" in sys.argv

    sep("=")
    print("  DATACENTER DT -- PHASE 1 LIQUID COOLING PIPELINE")
    print(f"  Directory : {BASE_DIR}")
    mode = "skip-existing" if skip_existing else "rerun-all"
    print(f"  Mode      : {mode}")
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

    # Summary
    sep("=")
    print("  FINAL SUMMARY")
    sep("=")
    all_pass = all(ok for _, ok, _ in results)
    for name, ok, elapsed in results:
        tag      = "PASS" if ok else "FAIL"
        skip_tag = "  (skipped)" if elapsed == 0.0 and ok else ""
        print(f"  [{tag}]  {name:<52}  {elapsed:>5.1f}s{skip_tag}")

    print()
    print("  Output files (in results/):")
    for step in STEPS:
        for f in step["output"]:
            p = os.path.join(RESULTS_DIR, f)
            if os.path.exists(p):
                print(f"    {f:<45} {fmt_size(p):>8}")

    sep("-")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print()
    if all_pass:
        print("  GATE PHASE 1 LC: PASS")
        print("  Ready for Phase 2 -- IS-Match Score (Liquid Cooling).")
        print()
        consolidate_to_excel()

        # ------------------------------------------------------------------
        # Step 0 — EED Compliance Matrix (post-consolidation)
        # Reads from the consolidated Excel (Regulatory_KPIs sheet already
        # present), appends EED_Compliance sheet, rebuilds phase1_lc_results.xlsx.
        # Placed after consolidation because it reads regulatory KPIs that
        # are derived during Step 1.1+1.2 and consolidated in the xlsx.
        # ------------------------------------------------------------------
        sep()
        print("  POST-PROCESSING: EED Compliance Matrix (Step 0)")
        sep("-")
        sys.stdout.flush()
        step0_script = os.path.join(BASE_DIR, "step_0_eed_compliance.py")
        if os.path.exists(step0_script):
            t0_step0 = time.time()
            result_step0 = subprocess.run(
                [sys.executable, step0_script],
                capture_output=True, text=True, cwd=BASE_DIR,
                env=_ENV_UTF8, encoding="utf-8",
            )
            elapsed_step0 = time.time() - t0_step0
            for line in result_step0.stdout.splitlines():
                print(f"  {line}")
            if result_step0.stderr.strip():
                for line in result_step0.stderr.splitlines():
                    print(f"  [STDERR] {line}")
            eed_ok = result_step0.returncode == 0
            tag = "PASS" if eed_ok else "FAIL"
            print(f"\n  [{tag}]  Step 0 EED Compliance  {elapsed_step0:.1f}s")
        else:
            print(f"  [SKIP]  step_0_eed_compliance.py not found -- skipped")

        # ------------------------------------------------------------------
        # Step 1.4c — ERF Sensitivity (capture_fraction grid)
        # Reads lc_dc_results_annual.csv + eed_compliance_lc.csv.
        # Adds ERF_Sensitivity sheet; rebuilds phase1_lc_results.xlsx (8 sheets).
        # Reference: ICEF SDC Roadmap 2025 technology benchmarks.
        # ------------------------------------------------------------------
        sep()
        print("  POST-PROCESSING: ERF Sensitivity (Step 1.4c)")
        sep("-")
        sys.stdout.flush()
        step1_4c_script = os.path.join(BASE_DIR, "step_1_4c_erf_sensitivity.py")
        if os.path.exists(step1_4c_script):
            t0_1_4c = time.time()
            result_1_4c = subprocess.run(
                [sys.executable, step1_4c_script],
                capture_output=True, text=True, cwd=BASE_DIR,
                env=_ENV_UTF8, encoding="utf-8",
            )
            elapsed_1_4c = time.time() - t0_1_4c
            for line in result_1_4c.stdout.splitlines():
                print(f"  {line}")
            if result_1_4c.stderr.strip():
                for line in result_1_4c.stderr.splitlines():
                    print(f"  [STDERR] {line}")
            tag = "PASS" if result_1_4c.returncode == 0 else "FAIL"
            print(f"\n  [{tag}]  Step 1.4c ERF Sensitivity  {elapsed_1_4c:.1f}s")
        else:
            print(f"  [SKIP]  step_1_4c_erf_sensitivity.py not found")

        # ------------------------------------------------------------------
        # Step 1.4d — Real-Case Benchmark Comparison
        # Reads TWH_mean from xlsx Regulatory_KPIs sheet (or fallback).
        # Adds Benchmark_Comparison sheet; rebuilds phase1_lc_results.xlsx (9 sheets).
        # Reference: IEA HPT Annex 47 + ICEF SDC Roadmap 2025 real cases.
        # ------------------------------------------------------------------
        sep()
        print("  POST-PROCESSING: Benchmark Comparison (Step 1.4d)")
        sep("-")
        sys.stdout.flush()
        step1_4d_script = os.path.join(BASE_DIR, "step_1_4d_benchmark_comparison.py")
        if os.path.exists(step1_4d_script):
            t0_1_4d = time.time()
            result_1_4d = subprocess.run(
                [sys.executable, step1_4d_script],
                capture_output=True, text=True, cwd=BASE_DIR,
                env=_ENV_UTF8, encoding="utf-8",
            )
            elapsed_1_4d = time.time() - t0_1_4d
            for line in result_1_4d.stdout.splitlines():
                print(f"  {line}")
            if result_1_4d.stderr.strip():
                for line in result_1_4d.stderr.splitlines():
                    print(f"  [STDERR] {line}")
            tag = "PASS" if result_1_4d.returncode == 0 else "FAIL"
            print(f