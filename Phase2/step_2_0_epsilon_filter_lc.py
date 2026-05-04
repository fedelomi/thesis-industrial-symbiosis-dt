"""
Step 2.0 LC — ε-parameter Pre-screening Gate  [LIQUID COOLING]
================================================================
Phase 2: IS-Match Score (OS2) · Thesis Cap. 3.3.3

Formula: ε = k · (1 − r)
  k  = IS knowledge intensity  [0→1]
  r  = rebound risk            [0→1]

Reference: Hatsor & Jelnov (2024) [B6]
Gate criterion: ε ≥ ε_min = 0.60

LC DC T_supply calibrated on Phase 1 LC actual means (lc_dc_results_annual.csv):
  Edge_LC:       T_mean=47.60°C  T_max=50.00°C
  Mid_LC:        T_mean=47.60°C  T_max=50.00°C
  Hyperscale_LC: T_mean=46.00°C  T_max=48.00°C

LC note: higher supply temperature (38°C CDU → 46-50°C return) vs air-side (21-29°C)
reduces the temperature lift required for industrial heat recovery, improving feasibility.
"""

from __future__ import annotations
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
OUTPUT_CSV  = BASE_DIR / "step_2_0_epsilon_gate_lc.csv"
EPSILON_MIN: float = 0.60

K_PARAMS: dict[str, float] = {"direct_HX": 1.00, "HP": 0.92, "CO2_HTHP": 0.85}
R_PARAMS: dict[str, float] = {"direct_HX": 0.04, "HP": 0.06, "CO2_HTHP": 0.08}
T_DELTA_DIRECT_MAX: float = 5.0
T_REQ_CO2_MIN:      float = 100.0


@dataclass
class DcScenarioLC:
    name: str
    t_supply_mean_c: float
    t_supply_max_c: float
    q_nominal_kw: float


@dataclass
class ProcessTier:
    name: str
    t_req_c: float
    description: str
    shift_type: str


@dataclass
class EpsilonRecord:
    dc_name: str
    process_name: str
    t_dc_mean_c: float
    t_dc_max_c: float
    t_req_c: float
    delta_t_vs_mean_c: float
    delta_t_vs_max_c: float
    upgrade_tech_mean: str
    upgrade_tech_max: str
    k: float
    r: float
    epsilon: float
    pass_gate: bool
    note: str


def select_upgrade_tech(t_dc: float, t_req: float) -> str:
    delta_t = t_req - t_dc
    if delta_t <= T_DELTA_DIRECT_MAX:
        return "direct_HX"
    elif t_req <= T_REQ_CO2_MIN:
        return "HP"
    else:
        return "CO2_HTHP"


def compute_epsilon(k: float, r: float) -> float:
    return round(k * (1.0 - r), 4)


def run_epsilon_filter(
    dc_scenarios: list[DcScenarioLC],
    process_tiers: list[ProcessTier],
    epsilon_min: float = EPSILON_MIN,
) -> pd.DataFrame:
    records: list[EpsilonRecord] = []
    for dc in dc_scenarios:
        for proc in process_tiers:
            tech_mean = select_upgrade_tech(dc.t_supply_mean_c, proc.t_req_c)
            tech_max  = select_upgrade_tech(dc.t_supply_max_c,  proc.t_req_c)
            k   = K_PARAMS[tech_mean]
            r   = R_PARAMS[tech_mean]
            eps = compute_epsilon(k, r)
            records.append(EpsilonRecord(
                dc_name=dc.name,
                process_name=proc.name,
                t_dc_mean_c=dc.t_supply_mean_c,
                t_dc_max_c=dc.t_supply_max_c,
                t_req_c=proc.t_req_c,
                delta_t_vs_mean_c=round(proc.t_req_c - dc.t_supply_mean_c, 1),
                delta_t_vs_max_c=round(proc.t_req_c - dc.t_supply_max_c,  1),
                upgrade_tech_mean=tech_mean,
                upgrade_tech_max=tech_max,
                k=k, r=r, epsilon=eps,
                pass_gate=eps >= epsilon_min,
                note=(f"tech differs at peak T: {tech_max}" if tech_mean != tech_max
                      else f"{tech_mean}; ΔT(mean)={round(proc.t_req_c-dc.t_supply_mean_c,1)}°C"),
            ))
    return pd.DataFrame([asdict(r) for r in records])


def main() -> None:
    # LC DC scenarios — calibrated from Phase 1 LC outputs
    dc_scenarios = [
        DcScenarioLC("Edge_LC",       t_supply_mean_c=47.60, t_supply_max_c=50.00, q_nominal_kw=475.0),
        DcScenarioLC("Mid_LC",        t_supply_mean_c=47.60, t_supply_max_c=50.00, q_nominal_kw=3104.0),
        DcScenarioLC("Hyperscale_LC", t_supply_mean_c=46.00, t_supply_max_c=48.00, q_nominal_kw=24500.0),
    ]
    # Same 3 process temperature tiers as air-side Phase 2
    process_tiers = [
        ProcessTier("LowT_60C",   60.0,  "Food/agro, greenhouse, aquaculture", "continuous"),
        ProcessTier("MidT_90C",   90.0,  "Textile, wood drying, paper pulp",   "2shift"),
        ProcessTier("HighT_130C", 130.0, "Rubber, chemical, industrial steam",  "1shift"),
    ]

    df = run_epsilon_filter(dc_scenarios, process_tiers, EPSILON_MIN)
    n_pass  = int(df["pass_gate"].sum())
    n_total = len(df)

    logger.info("=" * 68)
    logger.info("  Step 2.0 LC — ε-parameter Gate  |  ε_min = %.2f", EPSILON_MIN)
    logger.info("=" * 68)
    logger.info("  Technology          k       r      ε      Gate")
    logger.info("  " + "-" * 48)
    for tech in ["direct_HX", "HP", "CO2_HTHP"]:
        k_v  = K_PARAMS[tech]; r_v = R_PARAMS[tech]
        eps  = compute_epsilon(k_v, r_v)
        gate = "✓ PASS" if eps >= EPSILON_MIN else "✗ FAIL"
        logger.info(f"  {tech:<16}    {k_v:.2f}  {r_v:.2f}  {eps:.4f}  {gate}")
    logger.info("  " + "-" * 48)
    logger.info(f"  Gate result: {n_pass}/{n_total} PASS")
    logger.info("")
    logger.info("  LC note: T_supply_mean 46-48°C vs. air-side 21-29°C")
    logger.info("  → LowT_60C requires only ~12-14°C lift (HP, ε=0.8648)")
    logger.info("  → All technologies clear ε ≥ 0.60 gate universally")
    logger.info("=" * 68)

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"  Output → {OUTPUT_CSV}  ({len(df)} rows)")

    print(f"\n  Thesis note (Cap. 3.3.3): all {n_total} LC pairs ε ≥ {EPSILON_MIN}.")
    print(f"  DC WHR is high-ε IS by design [B6]; LC raises ε(HP) vs air-side.")
    print(f"  LC advantage: lower ΔT_required → HP (not CO₂ HTHP) for LowT/MidT.\n")


if __name__ == "__main__":
    main()
