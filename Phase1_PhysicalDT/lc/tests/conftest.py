"""Pytest configuration: expose Phase1_PhysicalDT/lc on sys.path so tests
can read result CSVs and (optionally) import sibling modules."""

from __future__ import annotations

import sys
from pathlib import Path

LC_DIR = Path(__file__).resolve().parents[1]
if str(LC_DIR) not in sys.path:
    sys.path.insert(0, str(LC_DIR))
