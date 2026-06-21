"""
generate_ood_benchmark.py
=========================
Phase 3 - Out-of-distribution (OOD) benchmark generation (blind).

Closes the benchmark-template independence limitation (audit-merge N3,
2026-05-23, lesson-3-phase-3-graph-rag section 3.8.5; Future Work FW9). The
canonical 100-query benchmark (data/benchmark_qa_dataset.json) was written by
the same researcher who authored the 19 Cypher templates, so it is
in-distribution by construction. This script generates a held-out OOD set by
asking an independent LLM (Claude Sonnet 4.6) to write questions about the same
knowledge domain WITHOUT any sight of the templates, the existing question
phrasings, or the routing keywords.

Blindness contract (enforced here and re-checked for free in
curate_ood_benchmark.py):
  PROMPT MUST NOT CONTAIN: template ids, benchmark_qa_dataset phrasings, routing
    keyword lists, the internal CYPHER structure.
  PROMPT MUST CONTAIN: the domain, the available corpus, the four target query
    categories, and the difficulty scale.

Independence note: the generator (Sonnet 4.6) is independent from the template
author (the researcher) -- the axis the limitation is about. The semantic judge
in step_3_11 is also Sonnet 4.6, but it scores Haiku-produced answers against the
ground truth, so no self-enhancement bias arises from the overlap.

Output (no ground truth yet):
    data/benchmarks/ood_candidates_raw.json

Usage:
    python generate_ood_benchmark.py --dry-run        # 5 candidates, cheap probe
    python generate_ood_benchmark.py --n 40           # full generation

Cost: a single Sonnet 4.6 call (~1.5k input, ~4k output tokens) is well under
0.10 USD. Token usage and an estimated cost are printed.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BENCH_DIR = BASE_DIR / "data" / "benchmarks"
OUT_RAW = BENCH_DIR / "ood_candidates_raw.json"

GEN_MODEL = "claude-sonnet-4-6"
# Non-zero temperature for phrasing diversity; the OOD set should not read like
# one canonical template style.
GEN_TEMPERATURE = 0.7
# 40 questions with rationales run ~6k output tokens; 4096 truncates the JSON.
MAX_OUTPUT_TOKENS = 8192

# Sonnet 4.x list pricing (USD per 1M tokens) for the cost estimate only.
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0

CATEGORIES = [
    "compliance-multi-hop",
    "parameter-lookup",
    "cross-comparison",
    "regulatory-traversal",
]
DIFFICULTIES = ["easy", "medium", "hard"]
REQUIRED_FIELDS = ("category", "difficulty", "nl_question", "rationale")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_prompt(n: int) -> tuple[str, str]:
    """Return (system, user) for the blind generation call.

    The text deliberately describes only the public knowledge DOMAIN and corpus,
    never the internal query templates, the existing question phrasings, or the
    routing keywords.
    """
    system = (
        "You are an independent energy-policy and industrial-engineering analyst. "
        "You have NOT seen any database schema, query template, or pre-existing "
        "question list for this domain. Write the questions a policymaker, a plant "
        "energy manager, or a compliance officer would genuinely ask. Vary the "
        "phrasing naturally. Return ONLY valid JSON, no prose around it."
    )
    user = f"""Domain: industrial symbiosis between data centers and manufacturing, \
specifically the reuse of data center waste heat by industrial processes, and the \
regulatory and economic framework that governs it in the EU, Italy and Denmark.

Knowledge corpus available to answer such questions:
- EU Energy Efficiency Directive 2023/1791, in particular the articles on energy \
audits and management (Art. 12), the comprehensive heating and cooling assessment \
(Art. 23), the efficiency criteria for heating and cooling (Art. 24), and data \
center monitoring and waste heat disclosure (Art. 26), plus the delegated \
regulation defining data center reporting KPIs.
- Italy: the white-certificate incentive scheme (Titoli di Efficienza Energetica / \
Certificati Bianchi) supporting energy efficiency and waste heat recovery, with its \
monetary value and eligible technologies, and the public body that runs it.
- Denmark: the Heat Supply Act (Varmeforsyningsloven) and its district heating \
connection obligation, the national energy and climate plan target, and the \
operating parameters of third and fourth generation district heating networks.
- Standards: ISO 50001 energy management, ASHRAE 90.4 data center energy \
efficiency, ISO 23247 digital twin for manufacturing.
- A technical scenario grid of nine cases (S1 to S9) combining three data center \
scales (edge ~0.5 MW, mid-size, hyperscale) with three waste heat temperature \
bands (low, medium, high grade), each with an upgrade technology (direct heat \
exchange, heat pump, high temperature heat pump) feeding a target manufacturing \
process (for example pasteurization, paper drying, sterilization) in an industrial \
sector.

Write {n} distinct questions spread across these four categories:
1. compliance-multi-hop  : reasoning across a regulation and a technical case \
(for example which obligations a specific data center triggers and why).
2. parameter-lookup       : a specific value, threshold, temperature, year or name.
3. cross-comparison       : compare two scales, two countries, two networks or two \
technologies on a quantitative or regulatory dimension.
4. regulatory-traversal   : follow the chain from a rule to the actor, framework or \
instrument that enacts or enforces it.

Rules:
- Ground every question in the corpus above; do not ask things this corpus cannot answer.
- Prefer questions that need more than one fact (a reasoning hop), except in \
parameter-lookup where a single precise value is fine.
- Do not number the questions inside the text and do not hint at any database structure.
- Spread difficulty across easy, medium and hard.

Return a JSON array of exactly {n} objects, each:
{{"category": one of {CATEGORIES}, "difficulty": one of {DIFFICULTIES}, \
"nl_question": "the question text", "rationale": "one line on what fact or hop it tests"}}
"""
    return system, user


def parse_candidates(raw: str) -> list[dict]:
    """Extract the JSON array of candidates from the model output."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match is None:
        raise ValueError("No JSON array found in model output.")
    return json.loads(match.group(0))


def validate_schema(candidates: list[dict]) -> tuple[list[dict], list[str]]:
    """Free schema gate: keep well-formed candidates, report the rejects."""
    clean: list[dict] = []
    errors: list[str] = []
    for i, c in enumerate(candidates):
        missing = [f for f in REQUIRED_FIELDS if not str(c.get(f, "")).strip()]
        if missing:
            errors.append(f"candidate {i}: missing {missing}")
            continue
        if c["category"] not in CATEGORIES:
            errors.append(f"candidate {i}: bad category {c['category']!r}")
            continue
        if c["difficulty"] not in DIFFICULTIES:
            errors.append(f"candidate {i}: bad difficulty {c['difficulty']!r}")
            continue
        clean.append({
            "id": f"OOD{len(clean) + 1:02d}",
            "category": c["category"],
            "difficulty": c["difficulty"],
            "nl_question": str(c["nl_question"]).strip(),
            "rationale": str(c["rationale"]).strip(),
            "generator_model": GEN_MODEL,
        })
    return clean, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind OOD benchmark generation")
    parser.add_argument("--n", type=int, default=40, help="number of candidates (30-50)")
    parser.add_argument("--dry-run", action="store_true", help="generate only 5 (cheap probe)")
    parser.add_argument("--out", type=Path, default=OUT_RAW)
    args = parser.parse_args()

    n = 5 if args.dry_run else args.n
    if not args.dry_run and not (30 <= n <= 50):
        logger.warning("n=%d is outside the requested 30-50 range.", n)

    from config import ANTHROPIC_API_KEY
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        logger.error("anthropic SDK not installed: %s", exc)
        return

    system, user = build_prompt(n)
    logger.info("Generating %d OOD candidates with %s (temp=%.1f)", n, GEN_MODEL, GEN_TEMPERATURE)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=GEN_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=GEN_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(block.text for block in message.content if hasattr(block, "text"))

    tok_in = message.usage.input_tokens
    tok_out = message.usage.output_tokens
    cost = tok_in / 1e6 * PRICE_IN_PER_MTOK + tok_out / 1e6 * PRICE_OUT_PER_MTOK
    logger.info("tokens in=%d out=%d  est cost=$%.4f", tok_in, tok_out, cost)

    try:
        candidates = parse_candidates(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Could not parse model output: %s", exc)
        (args.out.parent).mkdir(parents=True, exist_ok=True)
        (args.out.with_suffix(".error.txt")).write_text(raw, encoding="utf-8")
        return

    clean, errors = validate_schema(candidates)
    logger.info("schema gate: %d valid / %d returned", len(clean), len(candidates))
    for e in errors:
        logger.warning("  reject: %s", e)

    by_cat = {cat: sum(1 for c in clean if c["category"] == cat) for cat in CATEGORIES}
    logger.info("category spread: %s", by_cat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d candidates -> %s", len(clean), args.out)
    print(f"\nGenerated {len(clean)} candidates (est cost ${cost:.4f}). "
          f"Next: python curate_ood_benchmark.py")


if __name__ == "__main__":
    main()
