# ==============================================================================
# DATACENTER DT PROJECT - STEP 1.3
# Privacy-preserving synthetic profile generator
# Method: block bootstrap with replacement + calibrated Gaussian perturbation
# Inspired by SIDED (C4) -- no generative training required
# ==============================================================================
# Output:
#   synthetic_profile_annual.csv      -- synthetic annual profile
#   real_vs_synthetic_comparison.csv  -- comparative statistics per scenario
# ==============================================================================

# Decision active: D1 — derived from D1-calibrated DT output.


import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV   = RESULTS_DIR / "datacenter_dt_results_annual.csv"
OUT_SYNTH   = RESULTS_DIR / "synthetic_profile_annual.csv"
OUT_CONF    = RESULTS_DIR / "real_vs_synthetic_comparison.csv"

SEED       = 42
BLOCK_SIZE = 96          # 1 day (96 x 15 min) -- preserves daily autocorrelation
SIGMA_FRAC = 0.05        # sigma = 5% of empirical IQR (SIDED-style calibration)
FEATURES   = ["T_supply", "Q_available", "Exergy_DT", "Q_negotiated", "it_load_frac"]


def block_bootstrap(arr, block_size, rng):
    N        = len(arr)
    n_blocks = int(np.ceil(N / block_size))
    starts   = rng.integers(0, N - block_size + 1, size=n_blocks)
    blocks   = [arr[s : s + block_size] for s in starts]
    return np.concatenate(blocks)[:N]


def calibrate_sigma(arr):
    iqr   = np.subtract(*np.percentile(arr, [75, 25]))
    scale = iqr if iqr > 0 else np.std(arr)
    return SIGMA_FRAC * scale


def generate_synthetic(df_real, scenario, rng):
    df_s   = df_real[df_real["scenario"] == scenario].copy().reset_index(drop=True)
    df_out = df_s.copy()
    N      = len(df_s)
    for feat in FEATURES:
        arr       = df_s[feat].values
        arr_boot  = block_bootstrap(arr, BLOCK_SIZE, rng)
        sigma     = calibrate_sigma(arr)
        arr_synth = arr_boot + rng.normal(0, sigma, size=N)
        if feat == "T_supply":
            arr_synth = np.clip(arr_synth, 0.0, 100.0)
        elif feat in ("Q_available", "Exergy_DT", "Q_negotiated"):
            arr_synth = np.clip(arr_synth, 0.0, None)
        elif feat == "it_load_frac":
            arr_synth = np.clip(arr_synth, 0.10, 1.0)
        df_out[feat] = arr_synth
    Q_min = 0.10 * df_s["Q_available"].max()
    df_out["t_availability"] = (df_out["Q_available"] > Q_min).mean()
    df_out["scenario"]       = scenario + "_synth"
    return df_out


def ks2_numpy(r, s):
    """Two-sample KS test implemented in pure numpy."""
    r_s      = np.sort(r); s_s = np.sort(s)
    all_vals = np.sort(np.concatenate([r_s, s_s]))
    cdf_r    = np.searchsorted(r_s, all_vals, side='right') / len(r_s)
    cdf_s    = np.searchsorted(s_s, all_vals, side='right') / len(s_s)
    D        = float(np.max(np.abs(cdf_r - cdf_s)))
    n_eff    = (len(r) * len(s)) / (len(r) + len(s))
    p        = float(np.clip(2.0 * np.exp(-2.0 * n_eff * D**2), 0.0, 1.0))
    return D, p


def comparison_stats(df_r, df_s, scenario):
    records = []
    for feat in FEATURES:
        r = df_r[feat].values; s = df_s[feat].values
        cv_rmse          = (np.sqrt(np.mean((r - s)**2)) / (r.mean() + 1e-12)) * 100
        ks_stat, ks_p    = ks2_numpy(r, s)
        records.append({
            "scenario"    : scenario,
            "feature"     : feat,
            "mean_real"   : round(r.mean(), 4),
            "mean_synth"  : round(s.mean(), 4),
            "std_real"    : round(r.std(),  4),
            "std_synth"   : round(s.std(),  4),
            "min_real"    : round(r.min(),  4),
            "min_synth"   : round(s.min(),  4),
            "max_real"    : round(r.max(),  4),
            "max_synth"   : round(s.max(),  4),
            "cv_rmse_pct" : round(cv_rmse,  3),
            "ks_stat"     : round(ks_stat,  4),
            "ks_pvalue"   : round(ks_p,     4),
        })
    return pd.DataFrame(records)


def main():
    print("=" * 65)
    print("STEP 1.3 -- Privacy-Preserving Synthetic Generator")
    print("=" * 65)
    sys.stdout.flush()

    print(f"\nReading: {INPUT_CSV}")
    df_real = pd.read_csv(INPUT_CSV)
    print(f"  -> {len(df_real)} rows, scenarios: {list(df_real['scenario'].unique())}")
    sys.stdout.flush()

    rng = np.random.default_rng(SEED)
    list_synth, list_conf = [], []

    for scenario in df_real["scenario"].unique():
        print(f"\n[gen] Synthetic generation -- {scenario}")
        sys.stdout.flush()
        df_r = df_real[df_real["scenario"] == scenario].reset_index(drop=True)
        df_s = generate_synthetic(df_real, scenario, rng)
        df_c = comparison_stats(df_r, df_s, scenario)
        for _, row in df_c.iterrows():
            tag = "OK " if row["ks_pvalue"] > 0.05 else "(*)"
            print(f"   {tag} {row['feature']:18s}  "
                  f"mean_real={row['mean_real']:>12.2f}  mean_synth={row['mean_synth']:>12.2f}  "
                  f"CV-RMSE={row['cv_rmse_pct']:>6.2f}%  KS p={row['ks_pvalue']:.3f}")
        sys.stdout.flush()
        list_synth.append(df_s)
        list_conf.append(df_c)

    df_synth_all = pd.concat(list_synth, ignore_index=True)
    df_conf_all  = pd.concat(list_conf,  ignore_index=True)

    df_synth_all.to_csv(OUT_SYNTH, index=False)
    df_conf_all.to_csv(OUT_CONF,  index=False)

    print(f"\n{'='*65}")
    print(f"Saved: {OUT_SYNTH}  ({len(df_synth_all)} rows)")
    print(f"Saved: {OUT_CONF}  ({len(df_conf_all)} rows)")
    print(f"{'='*65}")
    print("\nMean CV-RMSE by scenario:")
    print(df_conf_all.groupby("scenario")["cv_rmse_pct"].mean().round(3).to_string())
    print("\n% features with KS p>0.05 (distribution preserved):")
    ks_ok = df_conf_all.groupby("scenario")["ks_pvalue"].apply(
        lambda x: (x > 0.05).mean() * 100).round(1)
    print(ks_ok.to_string())


if __name__ == "__main__":
    main()