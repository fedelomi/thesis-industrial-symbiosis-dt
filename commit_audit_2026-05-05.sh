#!/usr/bin/env bash
# =============================================================================
# commit_audit_2026-05-05.sh
# -----------------------------------------------------------------------------
# Helper script that materialises the audit-2026-05-05 changes as a sequence of
# logical Git commits on the current branch.
#
# Background: the audit reorg was performed by the assistant inside a sandbox
# that could not run `git commit` directly (OneDrive ACL holds the .git/index
# lock). All file edits and renames are already on disk; this script just
# stages them in the right groups and produces the final commit history.
#
# Run ONCE from the repo root, ideally on a clean working tree:
#     bash commit_audit_2026-05-05.sh
#
# Safe to run after `git status` confirms only the audit changes are pending.
# =============================================================================

set -euo pipefail

if [ ! -d ".git" ]; then
  echo "ERROR: must be run from the repository root."
  exit 1
fi

echo "==> 1/9  baseline (regenerated outputs + cluster-data-master removal)"
git add -u  # stages deletions of cluster-data-master and modified outputs
git add Phase1/ Phase2/ "Phase 3/" 2>/dev/null || true
git commit -m "chore: baseline pending outputs and remove cluster-data-master

Snapshots regenerated CSVs and stages the deletion of the unused
cluster-data-master mirror. Baseline commit prior to the audit-2026-05-05
repository reorganisation." || echo "  (nothing staged for this group)"

echo "==> 2/9  cleanup (remove venv + __pycache__)"
# Folders are already gone on disk; this stages their deletion in git.
git add -A "Phase 3" 2>/dev/null || true
git commit -m "chore: drop in-tree venv and pycache (audit FIX-9, FIX-10)

Phase 3/venv (606 MB) and __pycache__/ folders were removed from the
working tree. They remain ignored by .gitignore." || echo "  (nothing to commit)"

echo "==> 3/9  full repo reorganisation (rename phases + data/ + results/)"
git add -A
git commit -m "refactor: rename phases and split data/results subfolders

Phase1            -> Phase1_PhysicalDT/{airside,lc}
Phase2            -> Phase2_ISMatch
Phase 3 (space)   -> Phase3_GraphRAG
+ data/    holds committed reference inputs (Hotmaps, lc_opt_temperature_profile, etc.)
+ results/ holds generated CSVs and consolidated XLSX
Scripts auto-create results/ on import; all read/write paths updated." || echo "  (nothing to commit)"

echo "==> 4/9  Phase 3 dedupe + case fix (FIX-1, FIX-2)"
git add -A Phase3_GraphRAG/
git commit -m "fix(phase3): dedupe duplicate ingest script and rename to lower case

- Removed 'Step 3 1b ingest tier b it.py' (whitespace-only diff vs canonical).
- Renamed 'Step_3_1c_ingest_tier_b_dk.py' to 'step_3_1c_ingest_tier_b_dk.py'
  to match the snake_case convention used everywhere else." || echo "  (nothing to commit)"

echo "==> 5/9  Phase 1 path fix + shared dc_id_mapping (FIX-4, FIX-5)"
git add -A common/ Phase3_GraphRAG/step_3_5_phase1_integration.py Phase2_ISMatch/run_phase_2_lc.py Phase2_ISMatch/step_2_1_is_match_score_lc.py
git commit -m "fix: correct Phase 1 LC paths and add common/dc_id_mapping.py

- step_3_5_phase1_integration.py now points to Phase1_PhysicalDT/lc/results/
  (was 'Phase 1/LC-Opt/' which never resolved -> always fell back to wiki stats).
- run_phase_2_lc.py uses the post-reorg Phase 1 LC xlsx path.
- step_2_1_is_match_score_lc.py searches results/ first, then data/.
- New common/dc_id_mapping.py centralises Edge_LC <-> DC-S etc. translations." || echo "  (nothing to commit)"

echo "==> 6/9  Phase 3 orchestrator extension (FIX-3)"
git add -A Phase3_GraphRAG/run_phase_3.py Phase3_GraphRAG/requirements.txt Phase3_GraphRAG/.env.example
git commit -m "feat(phase3): full orchestrator covers ingest + RAG + integration

run_phase_3.py (renamed from run_phase_3_ingest.py) now drives all 12 steps
0..11 (was 0..4). New flags: --ingest-only, --skip-rag, --from-step N.
Added requirements.txt with pinned major versions for reproducible setup." || echo "  (nothing to commit)"

echo "==> 7/9  Extracted privacy gate (FIX-6)"
git add -A Phase3_GraphRAG/step_3_6_privacy_gate.py Phase3_GraphRAG/step_3_5_phase1_integration.py
git commit -m "refactor(phase3): extract step_3_6_privacy_gate.py from step_3_5

Privacy preservation gate (Passo 3.6 in roadmap-fasi-1-2-3) now lives in
its own script and is callable independently. Reuses PHASE1_LC_STATS from
step_3_5_phase1_integration to avoid drift on the wiki-validated reference
values." || echo "  (nothing to commit)"

echo "==> 8/9  Decision headers in scripts (FIX-8)"
git add -A
git commit -m "docs: declare D1-D6 decision headers in every implementation script

Each step_*.py and run_*.py now carries a short '# Decision active' header
right after the docstring, naming the implementation decision (D1-D6) it
implements. Cross-links to wiki [[decisioni-implementative]]." || echo "  (nothing to commit)"

echo "==> 9/9  Rewritten README (FIX-7)"
git add README.md
git commit -m "docs: rewrite README to match post-audit repo layout

- Correct institution: Politecnico di Torino (was Milano/Florence).
- Correct DC scales: Edge 100 kW, Mid 5 MW, Hyperscale 100 MW airside;
  500 kW / 3.2 MW / 25 MW LC.
- Repository structure section reflects the post-reorg layout.
- Per-phase Run sections updated with new folder paths.
- New 'Active implementation decisions (D1-D6)' table." || echo "  (nothing to commit)"

echo
echo "Done. Recent commit history:"
git log --oneline -10
