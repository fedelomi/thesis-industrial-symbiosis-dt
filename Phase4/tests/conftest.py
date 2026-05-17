"""Pytest setup: add Phase4 root to sys.path so ``env``, ``agents``, ``config`` import cleanly."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE4_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))
