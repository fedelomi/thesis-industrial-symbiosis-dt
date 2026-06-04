"""Pytest setup: add the Phase3_Blind root to sys.path so step_3_* import cleanly."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE3_BLIND_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_BLIND_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_BLIND_ROOT))
