"""Pytest configuration: expose Phase2_ISMatch root on sys.path so tests
can locate result CSVs and (optionally) import sibling modules."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parents[1]
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))
