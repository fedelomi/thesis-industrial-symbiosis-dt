"""TDD tests for CRITICAL #3: step_1_4b runnability + orchestrator gate.

step_1_4b_sensitivity_lc.py was truncated mid-main with no
`if __name__ == "__main__"` guard, so running it produced nothing while the
orchestrator reported OK because outputs_exist only checks file presence. The fix
restores the entrypoint guard and adds a runnability check to the orchestrator
that detects a step whose script does not compile or has no main entrypoint.
"""
from __future__ import annotations

import importlib
from pathlib import Path

orch = importlib.import_module("run_phase_1_lc")
step = importlib.import_module("step_1_4b_sensitivity_lc")


def test_step_1_4b_has_main_guard() -> None:
    """step_1_4b exposes a __main__ entrypoint guard so it runs as a script."""
    src = Path(step.__file__).read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src


def test_step_1_4b_main_is_callable() -> None:
    """step_1_4b defines a callable main()."""
    assert callable(step.main)


def test_runnability_gate_accepts_complete_step() -> None:
    """The orchestrator runnability gate accepts the fixed step_1_4b script."""
    assert orch.script_is_runnable(Path(step.__file__)) is True


def test_runnability_gate_rejects_script_without_main_guard(tmp_path) -> None:
    """The gate rejects a compileable script that lacks a __main__ entrypoint."""
    bad = tmp_path / "truncated_step.py"
    bad.write_text("def main() -> None:\n    print('did work')\n", encoding="utf-8")
    assert orch.script_is_runnable(bad) is False
