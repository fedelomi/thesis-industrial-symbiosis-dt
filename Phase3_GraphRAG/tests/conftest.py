"""Pytest configuration: expose Phase3_GraphRAG on sys.path so tests
can read result CSVs and (optionally) import sibling modules."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE3_DIR = Path(__file__).resolve().parents[1]
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))
