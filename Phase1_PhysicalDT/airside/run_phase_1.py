# ==============================================================================
# DATACENTER DT PROJECT - PHASE 1 PIPELINE ORCHESTRATOR
# Runs the full pipeline in sequence:
#
#   Step 1.1+1.2+1.2-bis  ->  step_1_1_2_rc_model.py
#   Step 1.3              ->  step_1_3_synthetic.py
#   Step 1.4              ->  step_1_4_validation.py
#   Step 1.4b             ->  step_1_4b_sensitivity.py
#
# Usage:
#   python run_phase_1.py                  # re-run everything
#   python run_phase_1.py --skip-existing  # skip steps whose output already exists
#
# Final output:
#   phase1_results.xlsx  -- all CSV outputs consolidated into one workbook
# ==============================================================================

# Decisions active: D1 — DT calibration on published KPIs (airside branch).


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
        "name"   : "Step 1.1+1.2+1.2-bis -- RC Thermal Model + Exergy",
        "script" : "step_1_1_2_rc_model.py",
        "output" : ["datacenter_dt_results_annual.csv"],
    },
    {
        "name"   : "Step 1.3 -- Privacy-Preserving Synthetic Generator",
        "script" : "step_1_3_synthetic.py",
        "output" : ["synthetic_profile_annual.csv", "real_vs_synthetic_comparison.csv"],
    },
    {
        "name"   : "Step 1.4 -- Validation W1 + NDE (Phase 2 gate)",
        "script" : "step_1_4_validation.py",
        "output" : ["validation_1_4.csv"],
    },
    {
        "name"   : "Step 1.4b -- Sensitivity Analysis Privacy-Fidelity",
        "script" : "step_1_4b_sensitivity.py",
        "output" : ["sensitivity_validation.csv"],
    },
]

# Maps each CSV output to its Excel sheet name
EXCEL_SHEETS = [
    ("datacenter_dt_results_annual.csv",  "RC_Model_Results"),
    ("synthetic_profile_annual.csv",       "Synthetic_Profile"),
    ("real_vs_synthetic_comparison.csv",   "Real_vs_Synthetic"),
    ("validation_1_4.csv",                 "Validation"),
    ("sensitivity_validation.csv",         "Sensitivity_Analysis"),
]

EXCEL_OUTPUT = os.path.join("results", "phase1_results.xlsx")


def sep(char="=", n=70):
    print(char * n)


def fmt_size(path):
    try:
        b = os.path.getsize(path)
        return f"{b/1e6:.1f} MB" if b >= 1e6 else (f"{b/1e3:.0f} KB" if b >= 1e3 else f"{b} B")
    except Exception:
        return "n/a"


def outputs_exist(step):
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
    """Write all CSV outputs as sheets into a single Excel workbook."""
    sep()
    print("  CONSOLIDATING OUTPUTS -> phase1_results.xlsx")
    sep("-")
    xlsx_path = os.path.join(BASE_DIR, EXCEL_OUTPUT)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for csv_name, sheet_name in EXCEL_SHEETS:
            csv_path = os.path.join(RESULTS_DIR, csv_name)
            if not os.path.exists(csv_path):
                print(f"  [SKIP]   {csv_name} not found -- sheet '{sheet_name}' omitted")
                continue
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [OK]     {sheet_name:<28} <- {csv_name}  ({len(df)} rows)")
    print(f"\n  Saved: {xlsx_path}  ({fmt_size(xlsx_path)})")
    sys.stdout.flush()


def main():
    skip_existing = "--skip-existing" in sys.argv

    sep("=")
    print("  DATACENTER DT -- PHASE 1 COMPLETE PIPELINE")
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
    print("  Output files:")
    for step in STEPS:
        for f in step["output"]:
            p = os.path.join(RESULTS_DIR, f)
            if os.path.exists(p):
                print(f"    {f:<45} {fmt_size(p):>8}")

    sep("-")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print()
    if all_pass:
        print("  GATE PHASE 1: PASS")
        print("  Ready for Phase 2 -- IS-