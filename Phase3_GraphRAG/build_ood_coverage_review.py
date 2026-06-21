"""
build_ood_coverage_review.py
============================
Phase 3 - OOD coverage-gap review builder (ZERO API, deterministic).

Produces a manual-review view of the KG_COVERAGE_GAP queries in
benchmark_ood_v1.jsonl so the researcher can split each gap into
  - kg_missing    : the KG genuinely does not contain the requested fact
  - retrieval_miss : the fact IS in the KG but the top-3 templates did not fetch it
  - ambiguous     : needs a semantic decision

Everything is computed from existing data plus free, read-only Neo4j queries:
  - top-3 templates with semantic_router confidence,
  - the top-1 Cypher and its returned rows / entities,
  - deterministic KG existence checks for the entities the query references
    (a standard, an article, an actor, a DH network, a scenario),
  - a KG-grounded gap_type_hint with a high/med/low confidence and a one-line reason.

The hint is a PROPOSAL. The `decision` and `notes` fields are left empty for the
researcher. The source benchmark stays UNCURATED until those are filled and
check_ood_coverage_split.py is run.

Outputs:
    OOD_COVERAGE_GAP_REVIEW.md                       (sequential review, by category)
    data/benchmarks/benchmark_ood_v1_review.jsonl    (backing, step_3_11-compatible)

Usage:
    python build_ood_coverage_review.py

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase, exceptions as neo4j_exc

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from templates import CYPHER_TEMPLATES
from step_3_4_bis_neuro_symbolic import extract_entity_ids

BENCH_DIR = BASE_DIR / "data" / "benchmarks"
OOD_FILE = BENCH_DIR / "benchmark_ood_v1.jsonl"
OUT_MD = BASE_DIR / "OOD_COVERAGE_GAP_REVIEW.md"
OUT_JSONL = BENCH_DIR / "benchmark_ood_v1_review.jsonl"

TOP_K = 3
MAX_ENTITIES = 5

CATEGORY_ORDER = [
    "compliance-multi-hop",
    "parameter-lookup",
    "cross-comparison",
    "regulatory-traversal",
]

# Anchor specs: (label, query regex, KG existence query, the template that WOULD
# return the fact). existence query returns a count; > 0 means the entity exists.
# The "answer_template" is the template a correct route would have used.
ANCHOR_SPECS: list[dict] = [
    {"key": "ASHRAE 90.4", "rx": r"ashrae",
     "exists_cy": "MATCH (s:Standard) WHERE toUpper(s.id) CONTAINS 'ASHRAE' RETURN count(s) AS c",
     "answer_template": "GENERIC_STANDARD"},
    {"key": "ISO 23247 digital twin", "rx": r"iso\s*-?\s*23247|digital twin",
     "exists_cy": "MATCH (s:Standard) WHERE s.id CONTAINS '23247' RETURN count(s) AS c",
     "answer_template": "GENERIC_STANDARD"},
    {"key": "ISO 50001", "rx": r"iso\s*-?\s*50001|energy management system",
     "exists_cy": "MATCH (s:Standard) WHERE s.id CONTAINS '50001' RETURN count(s) AS c",
     "answer_template": "ISO50001_ARTICLES"},
    {"key": "EED Art.12 energy audit", "rx": r"article\s*12|art\.?\s*12|energy audit",
     "exists_cy": "MATCH (a:RegulatoryArticle) WHERE a.id CONTAINS 'ART-12' RETURN count(a) AS c",
     "answer_template": None},
    {"key": "EED Art.23 assessment", "rx": r"article\s*23|art\.?\s*23|comprehensive (heating|assessment)",
     "exists_cy": "MATCH (a:RegulatoryArticle {id:'EED-ART-23'}) RETURN count(a) AS c",
     "answer_template": "ALL_REGULATORY_ARTICLES"},
    {"key": "EED Art.24 efficiency criteria", "rx": r"article\s*24|art\.?\s*24|efficiency criteria",
     "exists_cy": "MATCH (a:RegulatoryArticle {id:'EED-ART-24'}) RETURN count(a) AS c",
     "answer_template": "ALL_REGULATORY_ARTICLES"},
    {"key": "GSE / Certificati Bianchi", "rx": r"certificati bianchi|white certificate|\bgse\b|tee/cb",
     "exists_cy": "MATCH (a:Actor {id:'GSE'}) RETURN count(a) AS c",
     "answer_template": "GENERIC_ACTOR"},
    {"key": "DH network generation temps", "rx": r"3gdh|4gdh|third-generation|fourth-generation",
     "exists_cy": "MATCH (n:DHNetwork) RETURN count(n) AS c",
     "answer_template": "DK_DH_COMPARE"},
    {"key": "cost obligation / who bears cost", "rx": r"cost obligation|who bears|bears the cost",
     "exists_cy": None, "answer_template": None},
    {"key": "procedural steps / legal sequence", "rx": r"steps? (to|needed)|sequence of (legal )?steps|what steps|pathway from that",
     "exists_cy": None, "answer_template": None},
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def kg_count(session, cypher: str) -> int:
    try:
        rec = session.execute_read(lambda tx, c=cypher: tx.run(c).single())
        return int(rec["c"]) if rec else 0
    except Exception as exc:
        logger.debug("count query failed: %s", exc)
        return 0


def execute_template(session, template_id: str) -> list[dict]:
    try:
        return session.execute_read(lambda tx, c=CYPHER_TEMPLATES[template_id]: tx.run(c).data())
    except Exception as exc:
        logger.debug("template %s failed: %s", template_id, exc)
        return []


def rows_to_text(rows: list[dict], cap: int = 20) -> str:
    return "\n".join(" | ".join(f"{k}: {v}" for k, v in r.items()) for r in rows[:cap])


def detect_anchors(session, question: str) -> list[dict]:
    """Detect referenced anchors and resolve their KG existence (deterministic)."""
    ql = question.lower()
    out: list[dict] = []
    for spec in ANCHOR_SPECS:
        if not re.search(spec["rx"], ql):
            continue
        if spec["exists_cy"] is None:
            exists = None  # a concept (cost, steps), not an entity to look up
        else:
            exists = kg_count(session, spec["exists_cy"]) > 0
        out.append({"key": spec["key"], "exists": exists,
                    "answer_template": spec["answer_template"]})
    return out


def classify(question: str, top3_ids: list[str], top1_rows: list[dict],
             union_rows: list[dict], anchors: list[dict], lcr: float) -> dict:
    """Deterministic gap_type_hint with confidence and a one-line reason.

    Precedence (most decisive signal first):
      1. a central referenced entity is ABSENT from the KG -> kg_missing
      2. EVERY referenced entity is present AND routed (rows returned) -> the gap
         is a conservative false flag, not a true KG gap -> retrieval_miss
      3. compound multi-hop (3+ anchors, mixed coverage) -> ambiguous
      4. some referenced entity is present but NOT in the route -> retrieval_miss
      5. only workflow/cost concepts (no node) requested -> kg_missing
    """
    entity_anchors = [a for a in anchors if a["exists"] is not None]
    missing = [a for a in entity_anchors if a["exists"] is False]
    present = [a for a in entity_anchors if a["exists"] is True]
    concept = [a for a in anchors if a["exists"] is None]
    unrouted_present = [a for a in present if a["answer_template"] and a["answer_template"] not in top3_ids]
    compound = len(anchors) >= 3

    # 1. A central referenced entity does not exist in the KG (simple query).
    if missing and not compound:
        keys = ", ".join(a["key"] for a in missing)
        conf = "high" if len(anchors) <= 2 else "med"
        return {"gap_type_hint": "kg_missing", "confidence": conf,
                "reason": f"referenced entity absent from KG: {keys}; propose KG enrichment."}

    # 2. Fully answerable: all present anchors routed, nothing missing or conceptual.
    if present and not missing and not unrouted_present and not concept and top1_rows:
        if lcr < 0.5:
            reason = ("routed template returns the fact but the neuro-symbolic regex "
                      "matched no node id (value columns); false gap, not a true KG gap.")
        else:
            tmpls = sorted({a["answer_template"] for a in present if a["answer_template"]})
            reason = (f"all referenced entities are retrieved by the routed template(s) "
                      f"{tmpls or top3_ids[:1]}; conservative false gap, not a true KG gap.")
        return {"gap_type_hint": "retrieval_miss", "confidence": "high", "reason": reason}

    # 3. Compound multi-hop with mixed coverage -> needs a human semantic split.
    if compound:
        miss = [a["key"] for a in missing] + [a["key"] for a in concept]
        retr = sorted({a["answer_template"] for a in unrouted_present if a["answer_template"]})
        return {"gap_type_hint": "ambiguous", "confidence": "med",
                "reason": f"compound multi-hop; missing/concept: {miss or 'none'}; "
                          f"retrievable-unrouted via {retr or 'none'}; needs multi-template "
                          "composition plus a semantic decision."}

    # 4. The fact exists in the KG but the route did not target its template.
    if unrouted_present:
        tmpls = sorted({a["answer_template"] for a in unrouted_present})
        conf = "high" if len(tmpls) == 1 and not concept else "med"
        return {"gap_type_hint": "retrieval_miss", "confidence": conf,
                "reason": f"fact(s) exist in KG via {tmpls} but top-3 routed to {top3_ids}; "
                          "routing or template fix, not KG enrichment."}

    # 5. Only workflow / cost-allocation concepts requested (no node to look up).
    if concept:
        keys = ", ".join(a["key"] for a in concept)
        return {"gap_type_hint": "kg_missing", "confidence": "med",
                "reason": f"requested at workflow/attribute granularity not modelled: {keys}."}

    # 6. Nothing retrieved and no resolvable anchor.
    if not union_rows and not anchors:
        return {"gap_type_hint": "kg_missing", "confidence": "med",
                "reason": "no template returns rows and no referenced entity resolves in KG."}

    return {"gap_type_hint": "ambiguous", "confidence": "low",
            "reason": "partial schema expressiveness; granularity between entity and "
                      "attribute; semantic decision required."}


def main() -> None:
    if not OOD_FILE.exists():
        logger.error("OOD benchmark not found: %s", OOD_FILE)
        return
    records = [json.loads(l) for l in OOD_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    gaps = [r for r in records if r.get("coverage") == "KG_COVERAGE_GAP"]
    logger.info("loaded %d records, %d coverage gaps", len(records), len(gaps))

    try:
        from semantic_router import get_default_router
    except Exception as exc:
        logger.error("semantic_router unavailable: %s", exc)
        return
    router = get_default_router()

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except (neo4j_exc.ServiceUnavailable, neo4j_exc.AuthError) as exc:
        logger.error("Neo4j unavailable: %s", exc)
        return

    review_rows: list[dict] = []
    with driver.session(database=NEO4J_DATABASE) as session:
        for g in gaps:
            q = g["nl_question"]
            top3 = router.route(q, top_k=TOP_K)
            top3_ids = [t for t, _ in top3]
            top1_id = top3_ids[0]
            top1_rows = execute_template(session, top1_id)
            union_rows: list[dict] = []
            for tid in top3_ids:
                union_rows.extend(execute_template(session, tid))

            ents = extract_entity_ids(rows_to_text(top1_rows or union_rows))
            kg_entities = [{"label": lbl, "node_id": eid} for eid, lbl in ents[:MAX_ENTITIES]]

            anchors = detect_anchors(session, q)
            lcr = g["neuro_symbolic"]["consistency_rate"]
            hint = classify(q, top3_ids, top1_rows, union_rows, anchors, lcr)

            review_rows.append({
                "query_id": g["id"],
                "category_GT": g["category"],
                "difficulty": g["difficulty"],
                "query": q,
                "template_top3": [{"template": t, "confidence": round(float(c), 3)} for t, c in top3],
                "cypher_executed_top1": CYPHER_TEMPLATES[top1_id].strip(),
                "top1_template": top1_id,
                "top1_row_count": len(top1_rows),
                "KG_entities_retrieved": kg_entities,
                "anchors_detected": anchors,
                "LCR": g["neuro_symbolic"]["consistency_rate"],
                "gap_type_hint": hint["gap_type_hint"],
                "hint_confidence": hint["confidence"],
                "hint_reason": hint["reason"],
                "decision": "",   # to be filled manually by Fede
                "notes": "",      # optional, to be filled manually by Fede
                "status": "UNCURATED",
            })

    driver.close()

    # backing jsonl
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in review_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_markdown(review_rows)
    logger.info("wrote %s (%d gaps) and %s", OUT_MD.name, len(review_rows), OUT_JSONL.name)
    by_hint = {}
    for r in review_rows:
        by_hint[r["gap_type_hint"]] = by_hint.get(r["gap_type_hint"], 0) + 1
    logger.info("proposed hint split: %s", by_hint)
    print(f"\nReview view for {len(review_rows)} coverage gaps -> {OUT_MD.name}. "
          f"Fede fills 'decision', then: python check_ood_coverage_split.py")


def _write_markdown(rows: list[dict]) -> None:
    n = len(rows)
    by_hint: dict[str, int] = {}
    for r in rows:
        by_hint[r["gap_type_hint"]] = by_hint.get(r["gap_type_hint"], 0) + 1

    lines: list[str] = []
    lines.append("# OOD coverage-gap review (benchmark_ood_v1)\n")
    lines.append(
        "Zero-API deterministic view of the KG_COVERAGE_GAP queries, for manual "
        "split into `kg_missing` / `retrieval_miss` / `ambiguous`. Hints are a "
        "PROPOSAL from KG-grounded heuristics; fill the **decision** field in "
        f"`data/benchmarks/benchmark_ood_v1_review.jsonl`. Source benchmark stays "
        f"UNCURATED until decisions are filled and `check_ood_coverage_split.py` is run.\n")
    kgm = by_hint.get("kg_missing", 0)
    lines.append(f"- Coverage gaps: **{n}** of 38 OOD queries = {round(100.0 * n / 38, 1)}% "
                 "(the OOD-report headline, an UPPER bound).")
    lines.append(f"- Proposed hint split (pre-review): {by_hint}")
    lines.append(f"- Provisional **true KG coverage gap** = kg_missing = {kgm}/38 = "
                 f"{round(100.0 * kgm / 38, 1)}% (a LOWER bound; the rest are retrieval "
                 "misses or compound-composition needs, not missing KG facts). "
                 "`check_ood_coverage_split.py` recomputes this from Fede's decisions.\n")
    lines.append("Legend: gap_type_hint = kg_missing (KG lacks the fact, -> Cap 6.3 "
                 "enrichment backlog) / retrieval_miss (fact present, top-3 routing "
                 "missed it, -> routing or template fix) / ambiguous (needs Fede).\n")

    patterns = _aggregate_patterns(rows)
    lines.append("## Patterns aggregati\n")
    for p in patterns:
        lines.append(f"- {p}")
    lines.append("")

    for cat in CATEGORY_ORDER:
        cat_rows = [r for r in rows if r["category_GT"] == cat]
        if not cat_rows:
            continue
        lines.append(f"\n## {cat}  ({len(cat_rows)} gaps)\n")
        for r in cat_rows:
            top3 = ", ".join(f"{t['template']}({t['confidence']})" for t in r["template_top3"])
            ents = ", ".join(f"{e['label']}:{e['node_id']}" for e in r["KG_entities_retrieved"]) or "(none)"
            cy = r["cypher_executed_top1"].replace("\n", " ")
            cy = re.sub(r"\s+", " ", cy)[:240]
            lines.append(f"### {r['query_id']}  [{r['difficulty']}]  -> hint: "
                         f"**{r['gap_type_hint']}** ({r['hint_confidence']})\n")
            lines.append(f"- **query**: {r['query']}")
            lines.append(f"- **template_top3** (semantic conf): {top3}")
            lines.append(f"- **KG_entities_retrieved**: {ents}  | top1_rows={r['top1_row_count']}  | LCR={r['LCR']}")
            lines.append(f"- **cypher_top1** (`{r['top1_template']}`): `{cy}`")
            lines.append(f"- **hint_reason**: {r['hint_reason']}")
            lines.append(f"- **decision**: _______   **notes**: _______")
            lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_patterns(rows: list[dict]) -> list[str]:
    """Detect systematic clusters for the Cap 6.3 enrichment roadmap."""
    out: list[str] = []
    # Cluster 1: Article 12 / energy-audit gaps (EED-ART-12 absent from KG).
    art12 = [r["query_id"] for r in rows
             if any(a["key"].startswith("EED Art.12") and a["exists"] is False
                    for a in r["anchors_detected"])]
    if art12:
        out.append(f"**Missing EED Article 12 (energy audits)**: {len(art12)} gaps "
                   f"({', '.join(art12)}) reference Art. 12 / energy-audit duties, but the "
                   f"KG only models EED Art. 23/24/26 + the delegated regulation. "
                   f"-> add an EED-ART-12 node (and Art. 11 audit-follow-up) to the KG.")
    # Cluster 2: standards (ASHRAE / ISO 23247) routed away from GENERIC_STANDARD.
    std = [r["query_id"] for r in rows
           if any(a["key"] in ("ASHRAE 90.4", "ISO 23247 digital twin") and a["exists"] is True
                  and a["answer_template"] not in [t["template"] for t in r["template_top3"]]
                  for a in r["anchors_detected"])]
    if std:
        out.append(f"**Standards retrieval miss**: {len(std)} gaps ({', '.join(std)}) ask for "
                   f"ASHRAE 90.4 or ISO 23247, which DO exist as Standard nodes, but the router "
                   f"sent them to DC/article templates. -> routing/anchor fix (GENERIC_STANDARD), "
                   f"not KG enrichment.")
    # Cluster 3: procedural / cost-allocation concepts not modelled.
    proc = [r["query_id"] for r in rows
            if any(a["exists"] is None for a in r["anchors_detected"])]
    if proc:
        out.append(f"**Procedural / cost-allocation granularity**: {len(proc)} gaps "
                   f"({', '.join(proc)}) ask for legal step sequences or who bears cost, which "
                   f"are workflow/attribute facts the entity-relationship KG does not model. "
                   f"-> decide scope: model as procedure nodes or declare out-of-scope in Cap 6.3.")
    # Cluster 4: compound multi-hop (>=3 anchors) needing composition.
    comp = [r["query_id"] for r in rows if len(r["anchors_detected"]) >= 3]
    if comp:
        out.append(f"**Compound multi-hop**: {len(comp)} gaps ({', '.join(comp)}) chain 3+ "
                   f"anchors (regulation + standard + incentive); even with full coverage the "
                   f"single-template architecture cannot compose them. -> FW9-bis: multi-template "
                   f"composition, not just routing.")
    if not out:
        out.append("No systematic cluster detected.")
    return out


if __name__ == "__main__":
    main()
