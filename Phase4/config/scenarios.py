"""9-scenario matrix for Phase 4 IS-negotiation experiments.

Matrix: 3 DC LC scales (Edge / Mid / Hyperscale) x 3 manufacturing temperature
tiers (LowT 60 C / MidT 90 C / HighT 130 C). Per ``implementation-decisions.md``
D6, only the liquid-cooling pipeline is in scope; airside is excluded
evidence-based after Phase 2 IS-Match screening.

The manufacturing-side profile picked per tier is the **medium-proximity**
representative from the Phase 2 dataset (see
``Phase2_ISMatch/results/step_2_2_ce_heat_dataset_lc.csv``):

    * LowT  -> LowT_02_Agro_Medium       (75 km, 1200 kW, 60 C)
    * MidT  -> MidT_04_PaperPulp_Medium  (80 km, 28128 kW, 90 C)
    * HighT -> HighT_07_Rubber_Medium    (100 km, 2800 kW, 130 C)

IS-Match scores are the iteration=4 final values from Phase 2 step 2.4
ΔTC calibration (``step_2_4_delta_tc_calibration_lc.csv``). The function
``load_is_match_scores_from_phase2()`` reloads them dynamically so that any
re-run of Phase 2 propagates automatically.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from env.models import DCProfile, MfgParams

# --------------------------------------------------------------------------- #
# CAPEX lookup (mid-range from Obsidian A9 / D4 references)                   #
# --------------------------------------------------------------------------- #
CAPEX_EUR_PER_KW: Dict[str, float] = {
    "direct_HX": 150.0,
    "HP": 700.0,
    "CO2_HTHP": 1500.0,
}

# COP assumptions for revenue / welfare calculation (literature midpoints).
COP_BY_UPGRADE: Dict[str, float] = {
    "direct_HX": 1.0,  # no electricity input, just pressure drop
    "HP": 3.5,         # T_sink 80-90 C, T_source 45-50 C
    "CO2_HTHP": 2.8,   # T_sink 130-150 C, T_source 45-50 C (NeurIPS 2025 ref.)
}

# --------------------------------------------------------------------------- #
# DC LC profiles (Phase 2 dataset, T_supply_mean and Q_nominal columns)       #
# --------------------------------------------------------------------------- #
# Q_nominal = 0.90 * Q_installed for Edge/Mid (capture fraction LC, D5),
# Hyperscale uses 0.98 effective capture per Phase 1 LC output.
DC_PROFILES: Dict[str, DCProfile] = {
    "Edge_LC": DCProfile(
        dc_id="Edge_LC",
        q_available_kw=475.0,        # 500 kW installed * 0.95 LC capture
        t_supply_c=47.6,
        exergy_dt=0.7529,            # normalised (Phase 2)
        t_availability=1.0,
        is_match_score=0.0,          # populated by load_scenarios()
    ),
    "Mid_LC": DCProfile(
        dc_id="Mid_LC",
        q_available_kw=3104.0,       # anchored to Frontier 3.104 MW (D5)
        t_supply_c=47.6,
        exergy_dt=0.7529,
        t_availability=1.0,
        is_match_score=0.0,
    ),
    "Hyperscale_LC": DCProfile(
        dc_id="Hyperscale_LC",
        q_available_kw=24500.0,      # 25 MW * 0.98 capture
        t_supply_c=46.0,
        exergy_dt=0.7583,
        t_availability=1.0,
        is_match_score=0.0,
    ),
}

# --------------------------------------------------------------------------- #
# Manufacturing tier representatives (medium proximity per D2 decision)       #
# --------------------------------------------------------------------------- #
MFG_PROFILES: Dict[str, MfgParams] = {
    "LowT": MfgParams(
        plant_id="LowT_02_Agro_Medium",
        t_req_c=60.0,
        q_process_kw=1200.0,
        distance_km=75.0,
        upgrade_required="HP",       # T_supply 47 C -> T_req 60 C, dT 13 C
        shift_pattern="2shift",
    ),
    "MidT": MfgParams(
        plant_id="MidT_04_PaperPulp_Medium",
        t_req_c=90.0,
        q_process_kw=28127.9,
        distance_km=80.0,
        upgrade_required="HP",       # T_supply 47 C -> T_req 90 C, dT 43 C
        shift_pattern="continuous",
    ),
    "HighT": MfgParams(
        plant_id="HighT_07_Rubber_Medium",
        t_req_c=130.0,
        q_process_kw=2800.0,
        distance_km=100.0,
        upgrade_required="CO2_HTHP", # dT > 50 C -> CO2_HTHP per Phase 2 KPI note
        shift_pattern="2shift",
    ),
}

# --------------------------------------------------------------------------- #
# Scenario-matrix indexing (S1..S9 follow methodology section 4.6).            #
# Phase 5 was integrated into Phase 4 -- see CLAUDE.md.                        #
# --------------------------------------------------------------------------- #
# Row-major: DC scale outer, tier inner -> S1=Edge_LC×LowT, S2=Edge_LC×MidT, ...
SCENARIO_INDEX: Dict[str, Tuple[str, str]] = {
    "S1": ("Edge_LC", "LowT"),
    "S2": ("Edge_LC", "MidT"),
    "S3": ("Edge_LC", "HighT"),
    "S4": ("Mid_LC", "LowT"),
    "S5": ("Mid_LC", "MidT"),
    "S6": ("Mid_LC", "HighT"),
    "S7": ("Hyperscale_LC", "LowT"),
    "S8": ("Hyperscale_LC", "MidT"),
    "S9": ("Hyperscale_LC", "HighT"),
}

# Phase 2 CSV path (resolved relative to repo root by default).
DEFAULT_PHASE2_CSV = (
    Path(__file__).resolve().parents[2]
    / "Phase2_ISMatch"
    / "results"
    / "step_2_4_delta_tc_calibration_lc.csv"
)


def load_is_match_scores_from_phase2(
    csv_path: Path | str = DEFAULT_PHASE2_CSV,
    iteration: int = 4,
) -> Dict[Tuple[str, str], float]:
    """Load final IS-Match scores from Phase 2 step 2.4 CSV.

    Parameters
    ----------
    csv_path:
        Path to ``step_2_4_delta_tc_calibration_lc.csv``.
    iteration:
        Which calibration iteration to use. Phase 2 LC converged at iter=4
        (Δ_max = 0.0094 < 0.01).

    Returns
    -------
    dict
        Mapping ``(dc_id, tier_tag) -> is_match_score`` where ``tier_tag`` is
        one of ``"LowT"``, ``"MidT"``, ``"HighT"`` (derived from the ``tier``
        column in the CSV by stripping the ``_<temp>C`` suffix).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 2 IS-Match CSV not found at {path}. "
            "Run Phase 2 (step_2_4) first or override csv_path."
        )

    scores: Dict[Tuple[str, str], float] = {}
    tier_map = {"LowT_60C": "LowT", "MidT_90C": "MidT", "HighT_130C": "HighT"}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["iteration"]) != iteration:
                continue
            dc_id = row["dc_name"]
            tier_tag = tier_map.get(row["process_name"])
            if tier_tag is None:
                continue
            scores[(dc_id, tier_tag)] = float(row["is_match_score"])
    if len(scores) != 9:
        raise ValueError(
            f"Expected 9 IS-Match scores from {path}, got {len(scores)}."
        )
    return scores


def load_airside_q_profile(
    csv_path: Path | str | None = None,
    scenario_label: str = "Mid",
    target_mean_kw: float = 3104.0,
) -> np.ndarray:
    """Load and rescale the Phase 1 airside annual Q_available profile.

    The raw airside DC profiles in Phase 1 have absolute Q values in a
    different scale band than the LC profiles (Mid airside mean ~890 MW
    vs Mid_LC 3.104 MW). For the Phase 4 Q1 sensitivity scenario we
    preserve the **relative variability** of the airside profile while
    rescaling so the mean matches the LC-equivalent rated capacity.

    Parameters
    ----------
    csv_path:
        Path to ``datacenter_dt_results_annual.csv``. Defaults to the
        canonical location under Phase 1 airside results.
    scenario_label:
        Filter column ``scenario`` (one of ``"Edge"``, ``"Mid"``,
        ``"Hyperscale"``).
    target_mean_kw:
        Target mean of the rescaled profile in kW (matches the LC rated
        Q for the same scale).

    Returns
    -------
    np.ndarray
        1-D array of length ~35040 (15-min steps over one year) with
        Q_available_kw rescaled so ``profile.mean() == target_mean_kw``.

    Notes
    -----
    Cited in Cap. 5.5 robustness analysis. The rescaling preserves the
    relative variability of the airside profile (CV, autocorrelation,
    fraction of zero-Q hours) and only adjusts the absolute level.
    """
    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "Phase1_PhysicalDT"
            / "airside"
            / "results"
            / "datacenter_dt_results_annual.csv"
        )
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 airside annual CSV not found at {path}. "
            "Run Phase 1 airside first."
        )

    raw: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("scenario") != scenario_label:
                continue
            try:
                raw.append(float(row["Q_available"]))
            except (KeyError, ValueError):
                continue
    if not raw:
        raise ValueError(
            f"No rows found for scenario={scenario_label} in {path}."
        )

    arr = np.asarray(raw, dtype=np.float64)
    raw_mean = float(arr.mean())
    if raw_mean <= 0:
        raise ValueError(
            f"Airside profile {scenario_label} has non-positive mean ({raw_mean})."
        )
    scale_factor = target_mean_kw / raw_mean
    rescaled = arr * scale_factor
    return rescaled


def build_scenarios(
    csv_path: Path | str = DEFAULT_PHASE2_CSV,
) -> Dict[str, Tuple[DCProfile, MfgParams]]:
    """Build the 9-scenario dictionary with IS-Match scores from Phase 2.

    Returns
    -------
    dict
        ``SCENARIOS[S_id] = (DCProfile, MfgParams)`` with the DC profile's
        ``is_match_score`` field populated from the Phase 2 step 2.4 output.
    """
    is_match = load_is_match_scores_from_phase2(csv_path)
    scenarios: Dict[str, Tuple[DCProfile, MfgParams]] = {}
    for s_id, (dc_id, tier_tag) in SCENARIO_INDEX.items():
        base_dc = DC_PROFILES[dc_id]
        dc = DCProfile(
            dc_id=base_dc.dc_id,
            q_available_kw=base_dc.q_available_kw,
            t_supply_c=base_dc.t_supply_c,
            exergy_dt=base_dc.exergy_dt,
            t_availability=base_dc.t_availability,
            is_match_score=is_match[(dc_id, tier_tag)],
        )
        mfg = MFG_PROFILES[tier_tag]
        scenarios[s_id] = (dc, mfg)
    return scenarios


# --------------------------------------------------------------------------- #
# Airside sensitivity scenario (Q1 of review_report.md, Cap. 5.5).            #
# --------------------------------------------------------------------------- #
AIRSIDE_T_AVAILABILITY_MID = 0.66  # from Phase 1 airside annual stats (Mid).


def build_airside_scenario() -> Tuple[DCProfile, MfgParams]:
    """Build the single airside robustness scenario S_AIR_M.

    Same scale and tier as S5 (Mid_LC x MidT) but with a dynamic
    Q_available profile and t_availability < 1.0, so the
    ``dt_grounded`` ablation flag becomes empirically meaningful.

    Returns
    -------
    (DCProfile, MfgParams)
        Tuple ready to be plugged into the env factory.
    """
    base_dc = DC_PROFILES["Mid_LC"]
    is_match = load_is_match_scores_from_phase2()
    profile = load_airside_q_profile(
        scenario_label="Mid",
        target_mean_kw=base_dc.q_available_kw,
    )
    dc = DCProfile(
        dc_id="Mid_AIR",
        q_available_kw=base_dc.q_available_kw,
        t_supply_c=base_dc.t_supply_c,
        exergy_dt=base_dc.exergy_dt,
        t_availability=AIRSIDE_T_AVAILABILITY_MID,
        is_match_score=is_match[("Mid_LC", "MidT")],
        q_available_profile=profile,
    )
    mfg = MFG_PROFILES["MidT"]
    return dc, mfg


# Eager-load at import (fail fast if Phase 2 output is missing).
try:
    SCENARIOS: Dict[str, Tuple[DCProfile, MfgParams]] = build_scenarios()
    try:
        SCENARIOS["S_AIR_M"] = build_airside_scenario()
    except FileNotFoundError:
        # Phase 1 airside CSV missing -- keep the LC scenarios usable.
        pass
except FileNotFoundError:
    # Fail gracefully so unit tests that don't need IS-Match can still import.
    SCENARIOS = {}


__all__ = [
    "AIRSIDE_T_AVAILABILITY_MID",
    "CAPEX_EUR_PER_KW",
    "COP_BY_UPGRADE",
    "DC_PROFILES",
    "MFG_PROFILES",
    "SCENARIO_INDEX",
    "SCENARIOS",
    "build_airside_scenario",
    "build_scenarios",
    "load_airside_q_profile",
    "load_is_match_scores_from_phase2",
]
