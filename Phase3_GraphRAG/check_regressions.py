"""check_regressions.py - free strict-EM diff between two graph-rag eval runs.

No API, no Neo4j. Compares the per-question `exact_match` field of two
evaluation_results_graph-rag_*.json files in data/ (these are timestamped and
preserved, unlike the judge per-query CSV which used to be overwritten).

Usage:
    python check_regressions.py                      # oldest vs newest graph-rag JSON
    python check_regressions.py data/old.json data/new.json   # explicit pair

Note: this is STRICT exact-match (the answer JSONs do not store the semantic
judge verdict). It catches strict regressions; a semantic diff would require
re-judging the old answers (API cost). Given the overall semantic rose, strict
is a reasonable free proxy for "did anything break".

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load_em(path: Path) -> dict[str, bool]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["question_id"]: bool(r.get("exact_match")) for r in rows}


def main() -> None:
    if len(sys.argv) == 3:
        old_p, new_p = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        files = sorted(DATA.glob("evaluation_results_graph-rag_*.json"))
        if len(files) < 2:
            print("Need at least two graph-rag result files in data/.")
            return
        old_p, new_p = files[0], files[-1]

    old, new = load_em(old_p), load_em(new_p)
    ids = sorted(set(old) | set(new))
    regr = [q for q in ids if old.get(q) and not new.get(q)]
    gain = [q for q in ids if not old.get(q) and new.get(q)]

    print(f"OLD: {old_p.name}  ({sum(old.values())}/{len(old)} strict EM)")
    print(f"NEW: {new_p.name}  ({sum(new.values())}/{len(new)} strict EM)")
    print(f"\nREGRESSIONS (was correct, now wrong): {len(regr)}")
    for q in regr:
        print(f"  - {q}")
    print(f"\nGAINS (was wrong, now correct): {len(gain)}")
    for q in gain:
        print(f"  + {q}")
    print(f"\nnet strict EM change: {sum(new.values()) - sum(old.values()):+d}")
    print("NOTE: strict EM only; semantic verdicts are not stored in these JSONs.")


if __name__ == "__main__":
    main()
