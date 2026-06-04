"""
Step 3.6 Blind - RQ3 benchmark evaluation (NSRR vs vector-only baseline)
========================================================================
Phase 3 Blind reconstruction (Institutional-LLM / Strato 2).

Runs the neuro-symbolic answerer and the vector-only RAG baseline over
data/benchmark.jsonl, scores both with deterministic clause-grounded scorers, and
reports per-category accuracy, the overall gap and groundedness. Both answerers
receive the same expected-unit hint, so the comparison is symmetric.

Scoring is fully deterministic (no API needed). An optional cross-model LLM judge
hook (`semantic_judge`) is provided for the interpretive narrative; it defaults to a
deterministic lexical-overlap proxy so the evaluation is hermetic and reproducible,
and exposes the keyword-vs-semantic gap the framework reports transparently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from step_3_0_config import BENCHMARK_JSONL, RESULTS_DIR, get_logger
from step_3_1_ingest import load_kb
from step_3_5_answerer import AnswerResult, BaselineAnswerer, NSRRAnswerer

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Deterministic scorers                                                        #
# --------------------------------------------------------------------------- #
def token_match(group: str, text: str) -> bool:
    """Return True if any "|"-separated alternative of a gold token matches text.

    An alternative matches if it is a substring, or (for "article N") an
    abbreviation variant, or all of its significant words appear.
    """
    x = text.lower()
    for alt in group.lower().split("|"):
        alt = alt.strip()
        if not alt:
            continue
        if alt in x:
            return True
        m = re.match(r"article (\d+)", alt)
        if m and re.search(rf"art\.?\s*{m.group(1)}", x):
            return True
        words = [w for w in re.findall(r"[a-z0-9]+", alt) if len(w) >= 2]
        if words and all(w in x for w in words):
            return True
    return False


def score_numeric(item: Dict, ans: AnswerResult) -> bool:
    """Numeric threshold scoring: value within 1% tolerance AND same unit family."""
    if ans.value is None:
        return False
    gold_v = float(item["gold_value"])
    gold_u = item["gold_unit"]
    if ans.unit != gold_u:
        return False
    return abs(ans.value - gold_v) <= max(0.01 * abs(gold_v), 1e-6)


def score_verdict(item: Dict, ans: AnswerResult) -> bool:
    """Verdict scoring: predicted verdict equals the gold verdict."""
    pred = ans.verdict or ""
    # allow the verdict to appear in the answer text for non-gate answerers
    if not pred and ans.answer_text:
        for v in ("non_compliant", "non-compliant", "conditional", "compliant"):
            if v in ans.answer_text.lower():
                pred = v.replace("-", "_")
                break
    return pred.replace("-", "_") == item["gold_verdict"]


def score_tokens(item: Dict, ans: AnswerResult, require_all: bool) -> bool:
    """Token scoring over the answer text. require_all=True for multi_hop (AND)."""
    tokens = item["gold_tokens"]
    text = ans.answer_text + " " + " ".join(ans.articles)
    matches = [token_match(t, text) for t in tokens]
    return all(matches) if require_all else any(matches)


def score_item(item: Dict, ans: AnswerResult) -> bool:
    """Dispatch to the correct scorer for an item."""
    mode = item["scoring"]
    if mode == "numeric":
        return score_numeric(item, ans)
    if mode == "verdict":
        return score_verdict(item, ans)
    if mode == "multi_token":
        return score_tokens(item, ans, require_all=True)
    if mode == "keyword":
        return score_tokens(item, ans, require_all=False)
    return False


def grounded(item: Dict, ans: AnswerResult) -> bool:
    """Groundedness: the gold document appears among the cited documents."""
    return item.get("grounding_doc", "") in ans.grounding_docs


def semantic_judge(question: str, gold: str, prediction: str, use_api: bool = False) -> bool:
    """Optional cross-model semantic judge (mockable).

    When ``use_api`` is False (default) this is a deterministic lexical-overlap
    proxy (token-F1 >= 0.5). A Sonnet-class judge can be plugged in behind the
    same signature; its outputs would be cached and its numeric claims re-verified
    by the symbolic layer. Kept off by default so the evaluation is hermetic.
    """
    if use_api:  # pragma: no cover - requires ANTHROPIC_API_KEY
        from step_3_6_judge_api import judge_with_sonnet  # type: ignore
        return judge_with_sonnet(question, gold, prediction)
    g = set(re.findall(r"[a-z0-9]+", gold.lower()))
    p = set(re.findall(r"[a-z0-9]+", prediction.lower()))
    if not g:
        return False
    inter = len(g & p)
    prec = inter / max(1, len(p))
    rec = inter / max(1, len(g))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1 >= 0.5


# --------------------------------------------------------------------------- #
# Evaluation driver                                                            #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SystemScore:
    """Accuracy accounting for one system."""

    name: str
    by_cat_correct: Dict[str, int] = field(default_factory=dict)
    by_cat_total: Dict[str, int] = field(default_factory=dict)
    grounded_correct: int = 0
    total: int = 0
    correct: int = 0

    def add(self, category: str, ok: bool, is_grounded: bool) -> None:
        self.by_cat_total[category] = self.by_cat_total.get(category, 0) + 1
        self.by_cat_correct[category] = self.by_cat_correct.get(category, 0) + (1 if ok else 0)
        self.total += 1
        self.correct += 1 if ok else 0
        self.grounded_correct += 1 if is_grounded else 0

    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def cat_accuracy(self, cat: str) -> float:
        t = self.by_cat_total.get(cat, 0)
        return self.by_cat_correct.get(cat, 0) / t if t else 0.0

    def groundedness(self) -> float:
        return self.grounded_correct / self.total if self.total else 0.0


def load_benchmark(path=BENCHMARK_JSONL) -> List[Dict]:
    """Load the benchmark JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found at {path}. Run build_benchmark.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(use_dense_baseline: bool = False) -> Tuple[SystemScore, SystemScore, List[Dict]]:
    """Run NSRR and baseline over the benchmark and return their scores.

    Returns:
        (nsrr_score, baseline_score, per_item_rows).
    """
    kb = load_kb()
    nsrr = NSRRAnswerer(kb)
    base = BaselineAnswerer(kb, use_dense=use_dense_baseline)
    bench = load_benchmark()

    nsrr_s = SystemScore("NSRR")
    base_s = SystemScore("VectorBaseline")
    rows: List[Dict] = []

    for item in bench:
        prefer_unit = item.get("gold_unit", "")
        a_nsrr = nsrr.answer(item["question"], prefer_unit=prefer_unit)
        a_base = base.answer(item["question"], prefer_unit=prefer_unit)
        ok_n = score_item(item, a_nsrr)
        ok_b = score_item(item, a_base)
        nsrr_s.add(item["category"], ok_n, grounded(item, a_nsrr))
        base_s.add(item["category"], ok_b, grounded(item, a_base))
        rows.append({
            "id": item["id"], "category": item["category"], "scoring": item["scoring"],
            "gold": item["gold_answer"],
            "nsrr_key": a_nsrr.answer_key, "nsrr_method": a_nsrr.method, "nsrr_correct": ok_n,
            "base_key": a_base.answer_key, "base_method": a_base.method, "base_correct": ok_b,
        })
    return nsrr_s, base_s, rows


def main(use_dense_baseline: bool = False) -> Dict:
    """Run the evaluation, print and persist the report."""
    nsrr_s, base_s, rows = evaluate(use_dense_baseline)
    cats = sorted(set(nsrr_s.by_cat_total))

    print("\n" + "=" * 74)
    print("  RQ3 BENCHMARK: Neuro-Symbolic Reasoner (NSRR) vs Vector-only RAG baseline")
    print("=" * 74)
    print(f"  {'category':22} {'n':>4} {'NSRR':>8} {'baseline':>9} {'gap':>8}")
    print("  " + "-" * 56)
    for c in cats:
        n = nsrr_s.by_cat_total[c]
        an, ab = nsrr_s.cat_accuracy(c), base_s.cat_accuracy(c)
        print(f"  {c:22} {n:>4} {an*100:>7.1f}% {ab*100:>8.1f}% {(an-ab)*100:>+7.1f}")
    print("  " + "-" * 56)
    print(f"  {'OVERALL':22} {nsrr_s.total:>4} {nsrr_s.accuracy()*100:>7.1f}% "
          f"{base_s.accuracy()*100:>8.1f}% {(nsrr_s.accuracy()-base_s.accuracy())*100:>+7.1f}")
    print(f"  {'groundedness':22} {'':>4} {nsrr_s.groundedness()*100:>7.1f}% {base_s.groundedness()*100:>8.1f}%")
    print("=" * 74)

    # Persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_questions": nsrr_s.total,
        "nsrr_accuracy": round(nsrr_s.accuracy(), 4),
        "baseline_accuracy": round(base_s.accuracy(), 4),
        "gap": round(nsrr_s.accuracy() - base_s.accuracy(), 4),
        "nsrr_groundedness": round(nsrr_s.groundedness(), 4),
        "baseline_groundedness": round(base_s.groundedness(), 4),
        "by_category": {
            c: {"n": nsrr_s.by_cat_total[c],
                "nsrr": round(nsrr_s.cat_accuracy(c), 4),
                "baseline": round(base_s.cat_accuracy(c), 4)}
            for c in cats
        },
        "baseline_backend": "dense" if use_dense_baseline else "tfidf",
    }
    (RESULTS_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # CSV of per-item rows
    import csv
    with (RESULTS_DIR / "eval_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote results/eval_summary.json and results/eval_results.csv")
    return summary


if __name__ == "__main__":
    main()
