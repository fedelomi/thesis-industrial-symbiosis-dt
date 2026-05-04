# ==============================================================================
# DATACENTER DT PROJECT - STEP 1.4  [LIQUID COOLING]
# Synthetic profile validation (gate for Phase 2)
# Identical logic to step_1_4_validation.py -- only paths and scenarios differ.
#
# Gate: PASS if W1_norm < 0.10 and NDE < 0.20 for all features/scenarios
# ==============================================================================

KS_TEST_METHODOLOGICAL_NOTE = """
METHODOLOGICAL NOTE -- Why the KS test is not used as gate (Ch. 4.3)
See step_1_4_validation.py for full explanation.
W1_norm and NDE are used instead (Wasserstein + adapted NDE from SIDED).
"""

import os, sys
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REAL_CSV  = os.path.join(BASE_DIR, "lc_dc_results_annual.csv")
SYNT_CSV  = os.path.join(BASE_DIR, "lc_synthetic_profile_annual.csv")
OUT_CSV   = os.path.join(BASE_DIR, "lc_validation_1_4.csv")

FEATURES  = ["T_supply", "Q_available", "Exergy_DT"]
SCENARIOS = ["Edge_LC", "Mid_LC", "Hyperscale_LC"]

W1_THRESHOLD_EXCELLENT  = 0.05
W1_THRESHOLD_ACCEPTABLE = 0.10
NDE_THRESHOLD           = 0.20


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == m:
        return float(np.mean(np.abs(np.sort(a) - np.sort(b))))
    all_x = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), all_x, side='right') / n
    cdf_b = np.searchsorted(np.sort(b), all_x, side='right') / m
    dx    = np.diff(all_x, prepend=all_x[0])
    return float(np.sum(np.abs(cdf_a - cdf_b) * dx))


def nde_adapted(real: np.ndarray, synth: np.ndarray) -> float:
    real_sorted = np.sort(real)
    sigma_real  = real.std()
    if sigma_real == 0:
        return 0.0
    idx       = np.searchsorted(real_sorted, synth)
    idx       = np.clip(idx, 0, len(real_sorted) - 1)
    idx_left  = np.clip(idx - 1, 0, len(real_sorted) - 1)
    dist_right = np.abs(synth - real_sorted[idx])
    dist_left  = np.abs(synth - real_sorted[idx_left])
    nn_dist    = np.where(dist_left < dist_right, dist_left, dist_right)
    return float(np.mean(nn_dist) / sigma_real)


def main() -> None:
    print("=" * 70)
    print("STEP 1.4 LC -- Synthetic Profile Validation (Phase 2 Gate)")
    print("=" * 70)

    df_real = pd.read_csv(REAL_CSV)
    df_synt = pd.read_csv(SYNT_CSV)
    df_synt["scenario_base"] = df_synt["scenario"].str.replace("_synth", "", regex=False)

    records   = []
    gate_pass = True

    print(f"\n{'Scenario':<18} {'Feature':<14} {'W1_norm':>9} {'Gate W1':>10} {'NDE':>8} {'Gate NDE':>10}")
    print("-" * 74)

    for sc in SCENARIOS:
        r = df_real[df_real["scenario"] == sc]
        s = df_synt[df_synt["scenario_base"] == sc]

        for feat in FEATURES:
            real_vals  = r[feat].values
            synth_vals = s[feat].values

            w1_raw  = wasserstein_1d(real_vals, synth_vals)
            mu_real = real_vals.mean()
            w1_norm = w1_raw / mu_real if mu_real != 0 else 0.0
            nde_val = nde_adapted(real_vals, synth_vals)

            w1_ok  = w1_norm < W1_THRESHOLD_ACCEPTABLE
            nde_ok = nde_val < NDE_THRESHOLD
            if not w1_ok or not nde_ok:
                gate_pass = False

            w1_tag  = "PASS" if w1_norm < W1_THRESHOLD_EXCELLENT else ("WARN" if w1_ok else "FAIL")
            nde_tag = "PASS" if nde_ok else "FAIL"
            print(f"  {sc:<16} {feat:<14} {w1_norm:>8.4f}  {w1_tag:<10}  {nde_val:>6.4f}  {nde_tag}")

            records.append({
                "scenario"     : sc,
                "feature"      : feat,
                "w1_raw"       : round(w1_raw,  4),
                "w1_norm"      : round(w1_norm, 4),
                "w1_gate"      : "PASS" if w1_ok  else "FAIL",
                "nde"          : round(nde_val, 4),
                "nde_gate"     : "PASS" if nde_ok else "FAIL",
                "threshold_w1" : W1_THRESHOLD_ACCEPTABLE,
                "threshold_nde": NDE_THRESHOLD,
            })

    print("-" * 74)
    print()
    if gate_pass:
        print("GATE STEP 1.4 LC: PASS -- all metrics within threshold")
        print("  -> Cleared to proceed to Phase 2 (IS-Match Score)")
    else:
        fail_rows = [r for r in records if r["w1_gate"] == "FAIL" or r["nde_gate"] == "FAIL"]
        print("GATE STEP 1.4 LC: FAIL")
        for fr in fail_rows:
            issues = []
            if fr["w1_gate"] == "FAIL":
                issues.append(f"W1_norm={fr['w1_norm']:.4f} > {fr['threshold_w1']}")
            if fr["nde_gate"] == "FAIL":
                issues.append(f"NDE={fr['nde']:.4f} > {fr['threshold_nde']}")
            print(f"  FAIL {fr['scenario']}/{fr['feature']}: {', '.join(issues)}")

    df_out = pd.DataFrame(records)
    df_out["ks_note"] = "See KS_TEST_METHODOLOGICAL_NOTE in source file (Ch. 4.3)"
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(df_out)} rows)")

    print(f"\n{'-'*50}")
    for sc in SCENARIOS:
        vals = [r["w1_norm"] for r in records if r["scenario"] == sc]
        print(f"  W1_norm {sc:<18}: min={min(vals):.4f}  max={max(vals):.4f}  mean={np.mean(vals):.4f}")
    print(f"\n{'-'*50}")
    for sc in SCENARIOS:
        vals = [r["nde"] for r in records if r["scenario"] == sc]
        print(f"  NDE     {sc:<18}: min={min(vals):.4f}  max={max(vals):.4f}  mean={np.mean(vals):.4f}")
    print(f"{'-'*50}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
