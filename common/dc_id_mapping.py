"""
common/dc_id_mapping.py
=======================
Single source of truth for the data center identifier mapping used
across all three implementation phases.

Phase 1 (Physical-DT) and Phase 2 (IS-Match) refer to the three liquid-cooled
data center scales as ``Edge_LC``, ``Mid_LC`` and ``Hyperscale_LC`` (descriptive
scenario tags carried in CSV outputs). Phase 3 (Graph RAG IS) uses short
graph-friendly identifiers ``DC-S``, ``DC-M`` and ``DC-L`` for Neo4j nodes.

Use the helpers below whenever code needs to translate between the two naming
conventions, instead of redefining ad-hoc dictionaries inline.

Wiki references:
- [[roadmap-fasi-1-2-3]] FASE 1 LC scenarios
- [[decisioni-implementative]] D5 (grey-box LC parametrization), D6 (LC-only)
"""

from __future__ import annotations
from typing import Final


# Canonical scale labels (human-readable)
SCALE_EDGE: Final[str]       = "Edge"
SCALE_MID: Final[str]        = "Mid-size"
SCALE_HYPERSCALE: Final[str] = "Hyperscale"

# Phase 1/2 scenario tags (used in CSV scenario column)
PHASE12_EDGE: Final[str]       = "Edge_LC"
PHASE12_MID: Final[str]        = "Mid_LC"
PHASE12_HYPERSCALE: Final[str] = "Hyperscale_LC"

# Phase 3 graph identifiers (Neo4j DataCenter.id)
PHASE3_EDGE: Final[str]       = "DC-S"
PHASE3_MID: Final[str]        = "DC-M"
PHASE3_HYPERSCALE: Final[str] = "DC-L"

# Rated IT capacity per scale [kW] — D5 grey-box LC parametrization
RATED_KW: Final[dict[str, float]] = {
    SCALE_EDGE:       500.0,
    SCALE_MID:       3200.0,   # anchored to Frontier LC-Opt (HPE/ORNL)
    SCALE_HYPERSCALE: 25000.0,
}

# --- Bidirectional mappings ---

PHASE12_TO_PHASE3: Final[dict[str, str]] = {
    PHASE12_EDGE:       PHASE3_EDGE,
    PHASE12_MID:        PHASE3_MID,
    PHASE12_HYPERSCALE: PHASE3_HYPERSCALE,
}

PHASE3_TO_PHASE12: Final[dict[str, str]] = {v: k for k, v in PHASE12_TO_PHASE3.items()}

PHASE12_TO_SCALE: Final[dict[str, str]] = {
    PHASE12_EDGE:       SCALE_EDGE,
    PHASE12_MID:        SCALE_MID,
    PHASE12_HYPERSCALE: SCALE_HYPERSCALE,
}

PHASE3_TO_SCALE: Final[dict[str, str]] = {
    PHASE3_EDGE:       SCALE_EDGE,
    PHASE3_MID:        SCALE_MID,
    PHASE3_HYPERSCALE: SCALE_HYPERSCALE,
}


def to_phase3(phase12_id: str) -> str:
    """Map a Phase 1/2 scenario tag (Edge_LC / Mid_LC / Hyperscale_LC) to its
    Phase 3 graph identifier (DC-S / DC-M / DC-L)."""
    try:
        return PHASE12_TO_PHASE3[phase12_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Phase 1/2 scenario id: {phase12_id!r}. "
            f"Expected one of {list(PHASE12_TO_PHASE3)}."
        ) from exc


def to_phase12(phase3_id: str) -> str:
    """Map a Phase 3 graph identifier (DC-S / DC-M / DC-L) to its
    Phase 1/2 scenario tag (Edge_LC / Mid_LC / Hyperscale_LC)."""
    try:
        return PHASE3_TO_PHASE12[phase3_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Phase 3 graph id: {phase3_id!r}. "
            f"Expected one of {list(PHASE3_TO_PHASE12)}."
        ) from exc


def rated_kw(any_id: str) -> float:
    """Return the rated IT capacity [kW] for a DC, accepting either a
    Phase 1/2 scenario tag or a Phase 3 graph identifier."""
    if any_id in PHASE12_TO_SCALE:
        scale = PHASE12_TO_SCALE[any_id]
    elif any_id in PHASE3_TO_SCALE:
        scale = PHASE3_TO_SCALE[any_id]
    else:
        raise KeyError(
            f"Unknown DC identifier: {any_id!r}. "
            f"Accepted: {list(PHASE12_TO_SCALE) + list(PHASE3_TO_SCALE)}"
        )
    return RATED_KW[scale]
