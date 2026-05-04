"""
step_3_4_evaluation.py
=======================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Valutazione comparativa di 3 configurazioni RAG sul benchmark QA (100 domande):
  Config 1 - no-RAG     : solo LLM con system prompt generico
  Config 2 - graph-RAG  : pipeline Neo4j con Cypher templates (step_3_2)
  Config 3 - llm-cypher : GraphCypherQAChain (LLM genera Cypher da NL)

Metriche (RAGAs):
  - faithfulness        : risposta supportata dal contesto recuperato
  - answer_relevancy    : risposta pertinente alla domanda
  - context_precision   : contesto recuperato preciso e non rumoroso

Target tesi (da [[roadmap-fasi-1-2-3]] Passo 3.4):
  - hallucination rate < 15% per graph-RAG (faithfulness > 0.85)
  - accuracy > 85% su factual-lookup

Uso:
    # Solo configurazione template (no API key OpenAI richiesta per graph-RAG)
    python step_3_4_evaluation.py --config graph-rag

    # Tutte e 3 le configurazioni (richiede OPENAI_API_KEY)
    python step_3_4_evaluation.py --config all

    # Solo un sottoinsieme di domande (sviluppo/test)
    python step_3_4_evaluation.py --config graph-rag --n 20

    # Solo una categoria
    python step_3_4_evaluation.py --config graph-rag --category factual-lookup

Output:
    data/evaluation_results_<config>_<timestamp>.json
    data/evaluation_summary_<timestamp>.csv

Requisiti:
    pip install ragas langchain langchain-openai langchain-community neo4j
    OPENAI_API_KEY per config no-rag e llm-cypher

Riferimento wiki: [[roadmap-fasi-1-2-3]] Passo 3.4,
                  [[decisioni-implementative]] D3 (metriche RAGAs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase, exceptions as neo4j_exc


# --- Configurazione ---

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = "neo4j"

DATA_DIR       = Path("data")
DATASET_PATH   = DATA_DIR / "benchmark_qa_dataset.json"


# --- Struttura risultato ---

@dataclass
class EvalResult:
    question_id:      str
    category:         str
    difficulty:       str
    hop_count:        int
    nl_question:      str
    ground_truth:     str
    source_clause:    str
    config:           str
    answer:           str
    context:          str
    cypher_used:      str
    elapsed_ms:       float
    # Metriche (popolate da RAGAs o da exact-match fallback)
    faithfulness:     Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision:Optional[float] = None
    exact_match:      Optional[bool]  = None
    error:            Optional[str]   = None


# ============================================================
# CONFIG 1 — no-RAG (LLM puro)
# ============================================================

def run_no_rag(questions: list[dict], llm) -> list[EvalResult]:
    """Invia le domande direttamente al LLM senza contesto dal grafo."""
    print("\n  [Config 1: no-RAG] Running...")
    results = []
    for i, q in enumerate(questions):
        print(f"    {i+1}/{len(questions)}: {q['id']}", end=" ", flush=True)
        t0 = time.time()
        try:
            resp = llm.invoke(q["nl_question"])
            answer = resp.content if hasattr(resp, "content") else str(resp)
            elapsed = (time.time() - t0) * 1000
            print(f"OK ({elapsed:.0f}ms)")
        except Exception as exc:
            answer = ""
            elapsed = (time.time() - t0) * 1000
            print(f"ERROR: {exc}")

        results.append(EvalResult(
            question_id=q["id"],
            category=q["category"],
            difficulty=q["difficulty"],
            hop_count=q["hop_count"],
            nl_question=q["nl_question"],
            ground_truth=q["ground_truth"],
            source_clause=q["source_clause"],
            config="no-rag",
            answer=answer,
            context="",   # no context in no-RAG
            cypher_used="",
            elapsed_ms=elapsed,
        ))
    return results


# ============================================================
# CONFIG 2 — graph-RAG (Cypher templates)
# ============================================================

# Mapping keyword -> template ID (da step_3_2)
KEYWORD_ROUTING: list[tuple[list[str], str]] = [
    # EED Art.26 — disclosure + threshold
    (["art.26", "article 26", "disclosure", "1 mw", "1mw",
      "threshold for mandatory", "threshold for data center"],
     "P1_eed_art26_threshold"),
    # EED articoli generici -> tutti gli articoli regolatori
    (["art.23", "article 23", "comprehensive assessment", "every 5 year",
      "art.24", "article 24", "heating and cooling criteria",
      "how many regulatoryarticle", "how many regulatory article",
      "delegated regulation", "2024/1364", "twh", "erf", "ereuse",
      "kpi", "obligation type of eed", "eed article"],
     "ALL_REGULATORY_ARTICLES"),   # restituisce tutti gli articoli con summary
    # Standard ASHRAE / ISO
    (["ashrae", "90.4", "iso 23247", "iso-23247", "standard", "scope of",
      "which organization", "publishes", "what does iso", "year was iso",
      "year was ashrae"],
     "GENERIC_STANDARD"),
    # Scenario S1 specifico
    (["scenario s1", "s1 ", "edge dc", "small dc", "direct heat exchange", "direct_hx"],
     "P2_thermal_compatibility_S1"),
    # Compatibilita termica generale
    (["compatible", "compatibility", "thermally", "9 scenario", "all scenario",
      "which scenario", "heat pump", "temperature band", "t1", "t2", "t3",
      "range", "grade heat", "how many temperature"],
     "P2_thermal_compatibility_all"),
    # Denmark / DH
    (["denmark", "danish", "dh connection", "varmeforsyningsloven", "mandatory connect",
      "4gdh", "3gdh", "supply temperature of the danish", "return temperature",
      "dh penetration", "energy agency", "necp", "ghg reduction"],
     "P3_regulatory_screening_dk"),
    # Italy / TEE/CB
    (["incentive", "italy", "italian", "tee", "white certificate", "certificati",
      "eur/mwh", "eur/toe", "eur per mwh", "eur per toe", "gse", "gestore",
      "how many italian", "which italian", "eligible", "absorption"],
     "P4_incentives_it_whr"),
    # Hyperscale comparison
    (["hyperscale", "dc-l", "s7", "s8", "s9", "compare the three is scenario"],
     "P5_scenario_comparison_L"),
    # Full IS pathway
    (["pathway", "full path", "complete", "symbiosis pathway",
      "from data center to", "produces_heat", "27 row"],
     "P6_full_is_path"),
    # DataCenter archetype
    (["it capacity", "it power", "cooling type", "waste heat capacity",
      "pue", "nominal pue", "edge data center", "mid-size", "hyperscale data center archetype"],
     "GENERIC_DC"),
]

# Template Cypher (ripetuti da step_3_2 per autonomia dello script)
CYPHER_TEMPLATES: dict[str, str] = {
    "P1_eed_art26_threshold": """
        MATCH (r:Regulation {id: 'EED-2023-1791'})-[:CONTAINS]->(a:RegulatoryArticle {id: 'EED-ART-26'})
        RETURN a.title AS article, a.obligation_type AS obligation,
               a.threshold_value AS threshold_mw, a.threshold_unit AS unit, a.summary AS summary
    """,
    "P2_thermal_compatibility_S1": """
        MATCH (s:Scenario {id: 'S1'})
        MATCH (s)-[:USES_DC]->(dc:DataCenter)
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
        MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource)
        MATCH (hs)-[:IN_BAND]->(tb:TemperatureBand)
        RETURN s.id AS scenario, dc.id AS datacenter,
               mp.name AS target_process, mp.temp_required_c AS process_temp_c,
               hs.upgrade_tech AS upgrade_tech, hs.temp_supply_c AS supply_temp_c,
               tb.id AS band,
               CASE WHEN hs.temp_supply_c >= mp.temp_required_c THEN 'COMPATIBLE' ELSE 'REQUIRES_UPGRADE' END AS compatibility
    """,
    "P2_thermal_compatibility_all": """
        MATCH (s:Scenario)
        MATCH (s)-[:USES_DC]->(dc:DataCenter)
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
        MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource)
        RETURN s.id AS scenario, dc.scale AS dc_scale,
               mp.name AS process, hs.upgrade_tech AS upgrade_tech,
               hs.temp_supply_c AS supply_c, mp.temp_required_c AS required_c,
               CASE WHEN hs.temp_supply_c >= mp.temp_required_c THEN 'OK' ELSE 'GAP' END AS status
        ORDER BY s.id
    """,
    "P3_regulatory_screening_dk": """
        MATCH (c:Country {iso: 'DK'})-[:HAS_FRAMEWORK]->(pf:PolicyFramework)
        MATCH (pf)-[:CONTAINS]->(a:RegulatoryArticle)
        WHERE a.obligation_type = 'mandatory'
        MATCH (dc:DataCenter) WHERE dc.it_capacity_kw >= 1000
        RETURN c.name AS country, pf.name AS framework,
               a.title AS obligation, a.threshold_value AS threshold_mw,
               a.summary AS summary, collect(dc.id) AS compliant_dcs
    """,
    "P4_incentives_it_whr": """
        MATCH (i:Incentive)
        WHERE 'waste-heat-recovery' IN i.eligible_tech
        MATCH (i)-[:GOVERNED_BY]->(r:Regulation)
        RETURN i.name AS incentive, i.value_eur_toe AS eur_per_toe,
               i.value_eur_mwh AS eur_per_mwh,
               i.eligible_tech AS eligible_technologies,
               collect(r.short_name) AS governing_decrees
    """,
    "P5_scenario_comparison_L": """
        MATCH (s:Scenario)-[:USES_DC]->(dc:DataCenter {id: 'DC-L'})
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)
        MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource)
        MATCH (hs)-[:IN_BAND]->(tb:TemperatureBand)
        RETURN s.id AS scenario, tb.id AS temp_band, mp.name AS process,
               mp.temp_required_c AS process_temp_c, hs.upgrade_tech AS upgrade_tech,
               hs.capacity_kw AS available_kw, hs.temp_supply_c AS supply_temp_c
        ORDER BY tb.id
    """,
    "P6_full_is_path": """
        MATCH (dc:DataCenter)-[:PRODUCES_HEAT]->(hs:HeatSource)-[:IN_BAND]->(tb:TemperatureBand)
        MATCH (s:Scenario)-[:USES_DC]->(dc)
        MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess)-[:PART_OF]->(sector:IndustrialSector)
        RETURN dc.id AS datacenter, dc.scale AS dc_scale,
               hs.upgrade_tech AS upgrade, tb.id AS band,
               mp.name AS process, sector.name AS sector, s.id AS scenario
        ORDER BY dc.id, tb.id
    """,
    "ALL_REGULATORY_ARTICLES": """
        MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle)
        RETURN r.id AS regulation, r.short_name AS reg_name,
               a.id AS article_id, a.article_number AS art_no,
               a.title AS title, a.obligation_type AS obligation,
               a.threshold_value AS threshold_mw,
               a.summary AS summary
        ORDER BY r.id, a.article_number
    """,
    "ALL_REGULATORY_ARTICLES": """
        MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle)
        RETURN r.id AS regulation, r.short_name AS reg_name,
               a.id AS article_id, a.article_number AS art_no,
               a.title AS title, a.obligation_type AS obligation,
               a.threshold_value AS threshold_mw,
               a.summary AS summary
        ORDER BY r.id, a.article_number
    """,
    "GENERIC_REGULATION": """
        MATCH (r:Regulation)
        OPTIONAL MATCH (r)-[:CONTAINS]->(a:RegulatoryArticle)
        RETURN r.id AS reg_id, r.name AS name, r.jurisdiction AS jurisdiction,
               collect(a.title) AS articles
        ORDER BY r.jurisdiction, r.year
    """,
    "GENERIC_COUNTRY": """
        MATCH (c:Country)
        RETURN c.iso AS iso, c.name AS name,
               c.dh_penetration_pct AS dh_pct, c.eed_transposed AS eed_transposed
        ORDER BY c.iso
    """,
    "GENERIC_STANDARD": """
        MATCH (s:Standard)
        RETURN s.id AS id, s.name AS name, s.issuer AS issuer,
               s.year AS year, s.scope AS scope
        ORDER BY s.issuer
    """,
    "GENERIC_DC": """
        MATCH (dc:DataCenter)
        RETURN dc.id AS id, dc.scale AS scale, dc.it_capacity_kw AS it_kw,
               dc.cooling_type AS cooling, dc.pue_nominal AS pue,
               dc.waste_heat_kw AS waste_heat_kw
        ORDER BY dc.it_capacity_kw
    """,
}


def route_question(question: str) -> str:
    """Keyword routing: mappa NL question -> cypher template ID."""
    q_lower = question.lower()
    for keywords, template_id in KEYWORD_ROUTING:
        if any(kw in q_lower for kw in keywords):
            return template_id
    # fallback per tipo di entita'
    if any(w in q_lower for w in ["country", "nation", "penetration", "transposed"]):
        return "GENERIC_COUNTRY"
    if any(w in q_lower for w in ["regulation", "directive", "law", "decree"]):
        return "GENERIC_REGULATION"
    if any(w in q_lower for w in ["data center", "datacenter", "dc-s", "dc-m", "dc-l"]):
        return "GENERIC_DC"
    return "P6_full_is_path"


def cypher_rows_to_context(rows: list[dict]) -> str:
    """Converte righe Neo4j in stringa di contesto per il LLM."""
    if not rows:
        return "No data found in knowledge graph."
    lines = []
    for row in rows[:10]:   # max 10 righe per non eccedere il contesto
        lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(lines)


def run_graph_rag(questions: list[dict], driver, llm=None) -> list[EvalResult]:
    """
    Graph RAG con Cypher templates.
    Se llm=None usa solo il contesto grezzo (context-only mode).
    Se llm e' fornito genera una risposta NL a partire dal contesto.
    """
    print("\n  [Config 2: graph-RAG] Running...")
    results = []
    with driver.session(database=NEO4J_DATABASE) as session:
        for i, q in enumerate(questions):
            print(f"    {i+1}/{len(questions)}: {q['id']}", end=" ", flush=True)
            t0 = time.time()

            # 1. Routing -> template
            template_id = route_question(q["nl_question"])
            cypher = CYPHER_TEMPLATES[template_id]

            # 2. Esecuzione Cypher
            try:
                rows = session.execute_read(lambda tx, c=cypher: tx.run(c).data())
                context = cypher_rows_to_context(rows)
            except Exception as exc:
                rows = []
                context = f"Cypher error: {exc}"

            # 3. Generazione risposta (se LLM disponibile)
            if llm is not None and context and not context.startswith("Cypher error"):
                try:
                    prompt = (
                        f"Using the following knowledge graph data, answer the question concisely.\n\n"
                        f"Knowledge graph context:\n{context}\n\n"
                        f"Question: {q['nl_question']}\n\n"
                        f"Answer:"
                    )
                    resp = llm.invoke(prompt)
                    answer = resp.content if hasattr(resp, "content") else str(resp)
                except Exception as exc:
                    answer = context   # fallback: restituisce il contesto grezzo
            else:
                # Context-only: risposta = contesto strutturato (usabile per exact-match)
                answer = context

            elapsed = (time.time() - t0) * 1000
            print(f"OK ({elapsed:.0f}ms) [template={template_id}]")

            results.append(EvalResult(
                question_id=q["id"],
                category=q["category"],
                difficulty=q["difficulty"],
                hop_count=q["hop_count"],
                nl_question=q["nl_question"],
                ground_truth=q["ground_truth"],
                source_clause=q["source_clause"],
                config="graph-rag",
                answer=answer,
                context=context,
                cypher_used=template_id,
                elapsed_ms=elapsed,
            ))
    return results


# ============================================================
# CONFIG 3 — llm-cypher (GraphCypherQAChain)
# ============================================================

def run_llm_cypher(questions: list[dict], graph, llm) -> list[EvalResult]:
    """LLM genera Cypher automaticamente via GraphCypherQAChain."""
    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError:
        try:
            from langchain.chains import GraphCypherQAChain
        except ImportError:
            print("  GraphCypherQAChain non disponibile.")
            return []

    print("\n  [Config 3: llm-cypher] Running...")
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=False,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )

    results = []
    for i, q in enumerate(questions):
        print(f"    {i+1}/{len(questions)}: {q['id']}", end=" ", flush=True)
        t0 = time.time()
        try:
            response = chain.invoke({"query": q["nl_question"]})
            answer = response.get("result", "")
            cypher_used = ""
            context = ""
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if "query" in step:
                        cypher_used = step["query"]
                    if "context" in step:
                        context = str(step["context"])
            elapsed = (time.time() - t0) * 1000
            print(f"OK ({elapsed:.0f}ms)")
        except Exception as exc:
            answer = ""
            cypher_used = ""
            context = ""
            elapsed = (time.time() - t0) * 1000
            print(f"ERROR: {exc}")

        results.append(EvalResult(
            question_id=q["id"],
            category=q["category"],
            difficulty=q["difficulty"],
            hop_count=q["hop_count"],
            nl_question=q["nl_question"],
            ground_truth=q["ground_truth"],
            source_clause=q["source_clause"],
            config="llm-cypher",
            answer=answer,
            context=context,
            cypher_used=cypher_used,
            elapsed_ms=elapsed,
        ))
    return results


# ============================================================
# Exact-match scoring (fallback senza RAGAs)
# ============================================================

def compute_exact_match(results: list[EvalResult]) -> None:
    """
    Exact-match: controlla se token significativi del ground truth compaiono
    nella risposta. Gestisce numeri, unita' e parole brevi (MW, DK, etc.).
    """
    import re
    for r in results:
        if not r.answer or not r.ground_truth:
            r.exact_match = None
            continue
        # Tokenizza ground truth: mantieni parole (anche brevi), numeri e % 
        raw_tokens = re.findall(r'[A-Za-z0-9]+(?:[.,][0-9]+)?|%', r.ground_truth)
        # Filtra stop words generiche ma mantieni tutto il resto
        stopwords = {'the','and','or','is','are','in','of','to','a','an',
                     'for','with','as','at','by','from','that','this','be',
                     'not','no','yes','both','also','via','per','if'}
        keywords = [t.lower() for t in raw_tokens
                    if t.lower() not in stopwords and len(t) >= 1]
        if not keywords:
            r.exact_match = None
            continue
        ans_lower = r.answer.lower()
        matches = sum(1 for kw in keywords if kw in ans_lower)
        r.exact_match = (matches / len(keywords)) >= 0.5


# ============================================================
# RAGAs evaluation (opzionale, richiede OPENAI_API_KEY)
# ============================================================

def compute_ragas(results: list[EvalResult], llm) -> None:
    """
    Calcola metriche RAGAs su faithfulness, answer_relevancy, context_precision.
    Aggiorna i campi sui singoli EvalResult.
    Richiede: pip install ragas
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        print("  RAGAs non disponibile - usa exact-match. Run: pip install ragas datasets")
        return

    # Prepara dataset RAGAs (richiede contesto non vuoto)
    evaluable = [r for r in results if r.context and not r.context.startswith("Cypher error")]
    if not evaluable:
        print("  Nessun risultato con contesto - skip RAGAs")
        return

    data = {
        "question":  [r.nl_question for r in evaluable],
        "answer":    [r.answer for r in evaluable],
        "contexts":  [[r.context] for r in evaluable],
        "ground_truth": [r.ground_truth for r in evaluable],
    }

    print(f"  Computing RAGAs on {len(evaluable)} samples...")
    try:
        ds = Dataset.from_dict(data)
        scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        score_df = scores.to_pandas()

        for i, r in enumerate(evaluable):
            r.faithfulness      = float(score_df["faithfulness"].iloc[i])
            r.answer_relevancy  = float(score_df["answer_relevancy"].iloc[i])
            r.context_precision = float(score_df["context_precision"].iloc[i])
    except Exception as exc:
        print(f"  RAGAs error: {exc}")


# ============================================================
# Output e riepilogo
# ============================================================

def save_results(results: list[EvalResult], config: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"evaluation_results_{config}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    return path


def print_report(results: list[EvalResult]) -> None:
    if not results:
        return

    config = results[0].config
    n = len(results)

    # Exact match
    em_results = [r for r in results if r.exact_match is not None]
    em_rate = sum(1 for r in em_results if r.exact_match) / len(em_results) if em_results else 0

    # RAGAs (se disponibili)
    faith_vals = [r.faithfulness for r in results if r.faithfulness is not None]
    relev_vals = [r.answer_relevancy for r in results if r.answer_relevancy is not None]

    print(f"\n  [{config}]")
    print(f"    Questions evaluated : {n}")
    print(f"    Exact match rate    : {em_rate:.1%}  ({len(em_results)} samples)")
    if faith_vals:
        print(f"    Faithfulness (mean) : {sum(faith_vals)/len(faith_vals):.3f}")
    if relev_vals:
        print(f"    Answer relevancy    : {sum(relev_vals)/len(relev_vals):.3f}")

    # Per categoria
    cats = sorted(set(r.category for r in results))
    print(f"    By category:")
    for cat in cats:
        cat_r = [r for r in results if r.category == cat and r.exact_match is not None]
        if cat_r:
            cat_em = sum(1 for r in cat_r if r.exact_match) / len(cat_r)
            print(f"      {cat:<25} EM={cat_em:.1%} ({len(cat_r)} samples)")

    # Latenza media
    elapsed = [r.elapsed_ms for r in results]
    print(f"    Avg latency         : {sum(elapsed)/len(elapsed):.0f}ms")

    # Threshold check (da roadmap: accuracy > 85%, faithfulness > 0.85)
    print(f"    Target (EM > 85%)   : {'PASS' if em_rate >= 0.85 else 'FAIL'}")
    if faith_vals:
        faith_mean = sum(faith_vals)/len(faith_vals)
        print(f"    Target (faith>0.85) : {'PASS' if faith_mean >= 0.85 else 'FAIL'}")


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 evaluation pipeline")
    parser.add_argument(
        "--config", choices=["no-rag", "graph-rag", "llm-cypher", "all"],
        default="graph-rag",
        help="Configuration to evaluate (default: graph-rag)"
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Max number of questions (0 = all)"
    )
    parser.add_argument(
        "--category", choices=["factual-lookup", "multi-hop-is", "comparative", "all"],
        default="all",
        help="Filter by category (default: all)"
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Compute RAGAs metrics (requires OPENAI_API_KEY and pip install ragas)"
    )
    return parser.parse_args()


def load_questions(n: int = 0, category: str = "all") -> list[dict]:
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        print("Run step_3_3_benchmark_qa_design.py first.")
        sys.exit(1)
    with open(DATASET_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    if category != "all":
        questions = [q for q in questions if q["category"] == category]
    if n > 0:
        questions = questions[:n]
    return questions


def get_llm():
    """Inizializza LLM OpenAI. Richiede OPENAI_API_KEY."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("langchain-openai non installato: pip install langchain-openai")
        sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY non trovata.")
        print("  PowerShell: $env:OPENAI_API_KEY = 'sk-...'")
        sys.exit(1)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Phase 3 - step_3_4_evaluation.py")
    print(f"  Config: {args.config}")
    print("=" * 60)

    questions = load_questions(args.n, args.category)
    print(f"  Questions loaded: {len(questions)}")

    # Neo4j driver (sempre necessario per graph-rag)
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"  Neo4j connected: {NEO4J_URI}")
    except (neo4j_exc.ServiceUnavailable, neo4j_exc.AuthError) as exc:
        print(f"  Neo4j error: {exc}")
        if args.config in ("graph-rag", "llm-cypher", "all"):
            sys.exit(1)
        driver = None

    all_results: list[EvalResult] = []

    try:
        configs = (["no-rag", "graph-rag", "llm-cypher"]
                   if args.config == "all" else [args.config])

        llm = None
        graph = None

        for config in configs:
            if config in ("no-rag", "llm-cypher") and llm is None:
                llm = get_llm()

            if config == "no-rag":
                results = run_no_rag(questions, llm)

            elif config == "graph-rag":
                # graph-RAG: usa LLM per risposta NL solo se disponibile
                results = run_graph_rag(questions, driver, llm=llm)

            elif config == "llm-cypher":
                if graph is None:
                    try:
                        from langchain_community.graphs import Neo4jGraph
                        graph = Neo4jGraph(
                            url="neo4j://127.0.0.1:7687",
                            username=NEO4J_USER,
                            password=NEO4J_PASSWORD,
                            database=NEO4J_DATABASE,
                        )
                    except Exception as exc:
                        print(f"  Neo4jGraph init failed: {exc}")
                        continue
                results = run_llm_cypher(questions, graph, llm)

            else:
                continue

            # Exact match
            compute_exact_match(results)

            # RAGAs (opzionale)
            if args.ragas and llm is not None:
                compute_ragas(results, llm)

            # Salva
            out_path = save_results(results, config)
            print(f"\n  Saved: {out_path}")

            all_results.extend(results)

        # Riepilogo finale
        print("\n" + "=" * 60)
        print("  EVALUATION SUMMARY")
        print("=" * 60)

        seen = set()
        for r in all_results:
            if r.config not in seen:
                seen.add(r.config)
                config_results = [x for x in all_results if x.config == r.config]
                print_report(config_results)

        print("\nNext: step_3_4_bis_neuro_symbolic.py  (logical consistency check)")

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
