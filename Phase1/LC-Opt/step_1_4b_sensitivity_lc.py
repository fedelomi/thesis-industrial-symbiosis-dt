# ==============================================================================
# DATACENTER DT PROJECT - STEP 1.4b  [LIQUID COOLING]
# Sensitivity analysis: privacy-fidelity tradeoff curve
# Identical logic to step_1_4b_sensitivity.py -- only paths and scenarios differ.
# ==============================================================================

THESIS_INTERPRETATION = (
    "Privacy-fidelity tradeoff -- Liquid Cooling (Ch. 4.3):\n"
    "Grid calibrated to the actual scale of the LC synthetic generator (Step 1.3 LC),\n"
    "which uses sigma = 0.05 x IQR. Interpretation same as air-side Phase 1.\n"
    "\n"
    "- sigma x0.025: maximum fidelity, near-zero perturbation. Privacy insufficient.\n"
    "- sigma x0.05 : operating point of Step 1.3 LC -- reference baseline.\n"
    "- sigma x0.10 : expected optimum -- fidelity still excellent (W1_norm < 0.05).\n"
    "- sigma x0.25 : maximum tested protection -- check W1_norm vs threshold.\n"
    "\n"
    "Liquid cooling note: T_supply variability is lower than air-side (~0.5 C std)\n"
    "because CDU supply is tightly controlled. Sigma calibration on IQR accounts\n"
    "for this automatically via the SIDED-style approach.\n"
)

import os, sys
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REAL_CSV    = os.path.join(BASE_DIR, "lc_dc_results_annual.csv")
OUT_CSV     = os.path.join(BASE_DIR, "lc_sensitivity_validation.csv")

SIGMA_MULTS = [0.025, 0.05, 0.10, 0.25]
FEATURES    = ["T_supply", "Q_available", "Exergy_DT"]
SCENARIOS   = ["Edge_LC", "Mid_LC", "Hyperscale_LC"]
SEED        = 42
BLOCK_SIZE  = 96
NDE_SAMPLE  = 1000
W1_THRESHOLD  = 0.05
NDE_THRESHOLD = 0.20


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.sort(x) - np.sort(y))))


def nde(x_synth: np.ndarray, x_real: np.ndarray) -> float:
    sigma = float(np.std(x_real))
    if sigma == 0:
        return 0.0
    x_sub   = x_synth[:NDE_SAMPLE]
    nearest = np.array([x_real[np.argmin(np.abs(x_real - v))] for v in x_sub])
    return float(np.mean(np.abs(x_sub - nearest)) / sigma)


def block_bootstrap(arr: np.ndarray, block_size: int,
                    rng: np.random.Generator) -> np.ndarray:
    N        = len(arr)
    n_blocks = int(np.ceil(N / block_size))
    starts   = rng.integers(0, N - block_size + 1, size=n_blocks)
    return np.concatenate([arr[s:s+block_size] for s in starts])[:N]


def generate_for_sigma(df_sc: pd.DataFrame, sigma_mult: float,
                       rng: np.random.Generator) -> dict:
    N         = len(df_sc)
    result    = {}
    all_feats = ["T_supply", "Q_available", "Exergy_DT", "Q_negotiated", "it_load_frac"]
    for feat in all_feats:
        arr      = df_sc[feat].values
        arr_boot = block_bootstrap(arr, BLOCK_SIZE, rng)
        sigma    = sigma_mult * np.std(arr)
        synth    = arr_boot + rng.normal(0, sigma, size=N)
        if feat == "T_supply":
            synth = np.clip(synth, 0.0, 100.0)
        elif feat in ("Q_available", "Exergy_DT", "Q_negotiated"):
            synth = np.clip(synth, 0.0, None)
        elif feat == "it_load_frac":
            synth = np.clip(synth, 0.10, 1.0)
        result[feat] = synth
    return result


def main() -> None:
    print("=" * 70)
    print("STEP 1.4b LC -- Sensitivity Analysis: Privacy-Fidelity Tradeoff Curve")
    print("=" * 70)
    sys.stdout.flush()

    df_real = pd.read_csv(REAL_CSV)
    print(f"\n  Input      : {REAL_CSV}")
    print(f"  Rows       : {len(df_real)}")
    print(f"  sigma_mult : {SIGMA_MULTS}")
    print(f"  Scenarios  : {SCENARIOS}")
    print(f"  Features   : {FEATURES}\n")
    sys.stdout.flush()

    rng     = np.random.default_rng(SEED)
    records = []

    for sigma_mult in SIGMA_MULTS:
        print(f"  [calc] sigma_mult = {sigma_mult:.3f}x ...")
        sys.stdout.flush()
        for sc in SCENARIOS:
            df_sc      = df_real[df_real["scenario"] == sc].reset_index(drop=True)
            synth_cols = generate_for_sigma(df_sc, sigma_mult, rng)
            for feat in FEATURES:
                real_vals  = df_sc[feat].values
                synth_vals = synth_cols[feat]
                w1_raw     = wasserstein_1d(real_vals, synth_vals)
                mu_real    = float(np.mean(real_vals))
                w1_norm    = w1_raw / mu_real if mu_real != 0 else 0.0
                nde_val    = nde(synth_vals, real_vals)
                records.append({
                    "sigma_mult": sigma_mult,
                    "scenario"  : sc,
                    "feature"   : feat,
                    "W1_norm"   : round(w1_norm, 5),
                    "NDE"       : round(nde_val, 5),
                    "W1_gate"   : "PASS" if w1_norm < W1_THRESHOLD  else "FAIL",
                    "NDE_gate"  : "PASS" if nde_val < NDE_THRESHOLD else "FAIL",
                })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}  ({len(df_out)} rows)\n")
    sys.stdout.flush()

    pivot = (df_out.groupby("sigma_mult")[["W1_norm", "NDE"]]
             .mean().round(5).reset_index())
    pivot.columns = ["sigma_mult", "W1_norm_mean", "NDE_mean"]

    print("=" * 62)
    print("PIVOT TABLE -- mean across all LC scenarios and features")
    print("=" * 62)
    print(f"  {'sigma_mult':>10}  {'W1_norm mean':>14}  {'NDE mean':>10}  {'Gate W1':>9}  {'Gate NDE':>9}")
    print("  " + "-" * 58)
    for _, row in pivot.iterrows():
        w1_ok   = row["W1_norm_mean"] < W1_THRESHOLD
        nde_ok  = row["NDE_mean"]     < NDE_THRESHOLD
        print(f"  {row['sigma_mult']:>10.3f}  {row['W1_norm_mean']:>14.5f}  "
              f"{row['NDE_mean']:>10.5f}  {'PASS' if w1_ok else 'FAIL':>9}  "
              f"{'PASS' if nde_ok else 'FAIL':>9}")
    sys.stdout.flush()

    optimal = pivot[(pivot["W1_norm_mean"] < W1_THRESHOLD) &
                    (pivot["NDE_mean"]     < NDE_THRESHOLD)]
    print()
    if not optimal.empty:
        sigma_opt = optimal["sigma_mult"].max()
        w1_opt    = optimal.loc[optimal["sigma_mult"] == sigma_opt, "W1_norm_mean"].values[0]
        nde_opt   = optimal.loc[optimal["sigma_mult"] == sigma_opt, "NDE_mean"].values[0]
        print(f"  OPTIMAL PRIVACY-FIDELITY: sigma_mult = {sigma_opt:.3f}")
        print(f"    W1_norm mean = {w1_opt:.5f}  (threshold <{W1_THRESHOLD})")
        print(f"    NDE mean     = {nde_opt:.5f}  (threshold <{NDE_THRESHOLD})")
    else:
        print("  No sigma_mult satisfies both thresholds.")

    print()
    print("=" * 62)
    print("THESIS INTERPRETATION (Ch. 4.3) -- Liquid Cooling:")
    print("=" * 62)
    print(THESIS_INTERPRETATION)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
