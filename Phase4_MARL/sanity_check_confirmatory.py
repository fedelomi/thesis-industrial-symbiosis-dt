"""
sanity_check_confirmatory.py
=============================
Quick sanity check del CSV scaricato da Modal confirmatory run.

Legge results/modal_confirmatory_30seed_200k.csv e verifica:
1. 540 righe totali (30 seed x 9 scenari x 2 config A0+D)
2. Breakdown per (config, scale)
3. PoA Edge sotto D deve essere ~-0.20 con CI95 stretto (canonical finding)
4. PoA per ogni combinazione (config, scale) con CI95
5. welfare_dc Edge sotto D deve essere negativo (driver del gap)

Uso:
    cd "C:\\Users\\Feder\\Dev\\TESI\\Fasi Applicative\\Phase4"
    python sanity_check_confirmatory.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PHASE4 = Path(__file__).resolve().parent
CSV_PATH = PHASE4 / "results" / "modal_confirmatory_30seed_200k.csv"

SCALE_OF = {
    "S1": "Edge", "S2": "Edge", "S3": "Edge",
    "S4": "Mid", "S5": "Mid", "S6": "Mid",
    "S7": "Hyperscale", "S8": "Hyperscale", "S9": "Hyperscale",
}


def main() -> None:
    print("=" * 70)
    print("  Sanity check Modal confirmatory run")
    print("=" * 70)

    if not CSV_PATH.exists():
        print(f"ERRORE: file non trovato a {CSV_PATH}")
        print(f"        contenuto cartella results/:")
        for f in sorted((PHASE4 / "results").glob("*.csv")):
            print(f"          {f.name}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"\nFile letto: {CSV_PATH}")
    print(f"Righe totali: {len(df)}")
    print(f"Colonne: {list(df.columns)}")

    # 1. Sanity numero righe
    expected = 540
    if len(df) == expected:
        print(f"PASS  numero righe corretto ({expected})")
    else:
        print(f"WARN  attese {expected} righe, trovate {len(df)}")

    # 2. Map scale
    scenario_col = "scenario" if "scenario" in df.columns else "scenario_id"
    if scenario_col not in df.columns:
        print(f"ERRORE: nessuna colonna scenario o scenario_id in {list(df.columns)}")
        sys.exit(1)
    df["scale"] = df[scenario_col].map(SCALE_OF)

    # 3. Breakdown per (config, scale)
    print(f"\nBreakdown righe per (config, scale):")
    print(df.groupby(["config", "scale"]).size().unstack(fill_value=0))

    # 4. PoA aggregato per (config, scale)
    poa_col = "final_poa" if "final_poa" in df.columns else "poa"
    if poa_col not in df.columns:
        print(f"\nERRORE: nessuna colonna final_poa o poa in {list(df.columns)}")
        sys.exit(1)

    print(f"\nPoA aggregato per (config, scale) con CI95:")
    agg = df.groupby(["config", "scale"])[poa_col].agg(["mean", "std", "count"])
    agg["ci_half"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["ci_low"] = agg["mean"] - agg["ci_half"]
    agg["ci_high"] = agg["mean"] + agg["ci_half"]
    print(agg.round(4))

    # 5. Test canonical: gap PoA (D - A0) per scala, atteso:
    #    Edge -0.199, Mid +0.187, Hyperscale +0.103
    print(f"\nTest canonical finding (gap PoA = PoA_D - PoA_A0 per scala):")
    print(f"  Atteso:  Edge -0.199, Mid +0.187, Hyperscale +0.103")
    print()
    canonical = {"Edge": -0.199, "Mid": +0.187, "Hyperscale": +0.103}
    for scale in ["Edge", "Mid", "Hyperscale"]:
        A0 = df[(df["scale"] == scale) & (df["config"] == "A0")][poa_col]
        D = df[(df["scale"] == scale) & (df["config"] == "D")][poa_col]
        if len(A0) == 0 or len(D) == 0:
            continue
        # Gap per seed-pair se pareggiati, altrimenti gap di medie
        gap_mean = D.mean() - A0.mean()
        # Welch CI95 della differenza di medie
        se = np.sqrt(D.var(ddof=1) / len(D) + A0.var(ddof=1) / len(A0))
        ci_half = 1.96 * se
        expected = canonical[scale]
        within = abs(gap_mean - expected) < 0.05
        status = "PASS" if within else "WARN"
        sign_match = (gap_mean * expected > 0)
        sign_str = "(stesso segno)" if sign_match else "(segno DIVERSO da canonical)"
        print(f"  {scale:11s}  PoA_A0={A0.mean():+.4f}  PoA_D={D.mean():+.4f}  "
              f"gap={gap_mean:+.4f} +/- {ci_half:.4f}  "
              f"[atteso {expected:+.4f}]  {status} {sign_str}")

    # 6. welfare_dc Edge sotto D deve essere negativo (mechanism check)
    if "welfare_dc" in df.columns:
        edge_D_wdc = edge_D["welfare_dc"]
        print(f"\nMechanism check (welfare_dc Edge sotto D, expected negativo):")
        print(f"  mean = {edge_D_wdc.mean():+.2f}")
        print(f"  median = {edge_D_wdc.median():+.2f}")
        print(f"  fraction negative = {(edge_D_wdc < 0).mean() * 100:.1f}%")
        if edge_D_wdc.mean() < 0:
            print(f"  PASS  welfare_dc collapse confermato su Edge sotto D")
        else:
            print(f"  WARN  welfare_dc non collassa, mechanism diverso")

    print("\n" + "=" * 70)
    print("  Sanity check completato")
    print("=" * 70)


if __name__ == "__main__":
    main()
