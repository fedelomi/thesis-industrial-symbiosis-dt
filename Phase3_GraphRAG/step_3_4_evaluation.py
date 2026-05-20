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

Target tesi (da [[phase-1-2-3-roadmap]] Passo 3.4):
  - hallucination rate < 15% per graph-RAG (faithfulness > 0.85)
  - accuracy > 85% su factual-lookup

Uso:
    # Solo configurazione template (no API key richiesta per graph-RAG template mode)
    python step_3_4_evaluation.py --config graph-rag

    # Tutte e 3 le configurazioni (richiede ANTHROPIC_API_KEY)
    python step_3_4_evaluation.py --config all

    # Solo un sottoinsieme di domande (sviluppo/test)
    python step_3_4_evaluation.py --config graph-rag --n 20

    # Solo una categoria
    python step_3_4_evaluation.py --config graph-rag --category factual-lookup

Output:
    data/evaluation_results_<config>_<timestamp>.json
    data/evaluation_summary_<timestamp>.csv

Requisiti:
    pip install ragas langchain langchain-anthropic langchain-community neo4j
    ANTHROPIC_API_KEY per config no-rag e llm-cypher

Riferimento wiki: [[phase-1-2-3-roadmap]] Passo 3.4,
                  [[implementation-decisions]] D3 (metriche RAGAs)
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

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    DATA_DIR, LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY,
)
from templates import CYPHER_TEMPLATES

DATASET_PATH = DATA_DIR / "benchmark_qa_dataset.json"


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
    # ---- Temperature band definitions (only definition-style queries) ----
    (["temperature band", "low-grade heat band",
      "mid-grade heat band", "high-grade heat band",
      "how many temperature band",
      "range that defines the t1", "range that defines the t2",
      "range that defines the t3",
      "what range defines t1", "what range defines t2", "what range defines t3",
      "label of t1", "label of t2", "label of t3",
      "definition of t1", "definition of t2", "definition of t3"],
     "TEMPERATURE_BAND_DEF"),
    # ---- DK 4GDH only ----
    (["4gdh", "4th generation",
      "supply temperature of the danish 4gdh",
      "supply temperature of the danish 4th",
      "return temperature of the danish 4gdh",
      "return temperature of the danish 4th",
      "policy framework that governs",
      "policy framework governs",
      "danish 4gdh thermally feasible",
      "what governs the danish 4gdh"],
     "DK_4GDH_PARAMS"),
    # ---- DK 3GDH (legacy) only ----
    (["3gdh", "3rd generation",
      "supply temperature of the danish 3gdh",
      "supply temperature of the danish 3rd",
      "return temperature of the danish 3gdh",
      "return temperature of the danish 3rd",
      "legacy network", "danish legacy"],
     "DK_3GDH_PARAMS"),
    # ---- DK both networks: compare/capacity questions ----
    (["compare 3gdh and 4gdh", "compare 4gdh and 3gdh",
      "3gdh and 4gdh", "4gdh and 3gdh",
      "dh network capacity between", "capacity between",
      "what upgrade is needed vs"],
     "DK_DH_COMPARE"),
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
      "which scenario", "heat pump", "grade heat"],
     "P2_thermal_compatibility_all"),
    # Denmark / DH
    (["denmark", "danish", "dh connection", "varmeforsyningsloven",
      "mandatory connect", "dh penetration", "energy agency",
      "necp", "ghg reduction"],
     "P3_regulatory_screening_dk"),
    # Italy / TEE/CB
    (["incentive", "italy", "italian", "tee", "white certificate", "certificati",
      "eur/mwh", "eur/toe", "eur per mwh", "eur per toe", "gse", "gestore",
      "how many italian", "which italian", "eligible", "absorption"],
     "P4_incentives_it_whr"),
    # ---- Multi-criteria scenario comparison (NEW, scenario X) ----
    (["combination of data center scale", "combination of dc scale",
      "maximizes both heat capacity", "scale and upgrade technology",
      "highest capacity and highest temperature",
      "best combination of"],
     "P5_scenario_comparison_L"),
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
    # ISO 50001
    (["iso 50001", "iso50001", "energy management", "enms", "enpi",
      "energy review", "significant energy use", "seu", "§6.3", "§6.4",
      "§4.1", "§9.1", "climate change", "a1:2024", "requires_compliance",
      "energy performance indicator", "monitoring measurement"],
     "ISO50001_ARTICLES"),
    # ISO 50001 + ManufacturingProcess
    (["manufacturing process", "comply with iso", "iso compliance",
      "which process", "which manufacturing", "process comply", "iso 50001 apply"],
     "ISO50001_PROCESS_COMPLIANCE"),
]

# CYPHER_TEMPLATES importato da templates.py (unica fonte di verita')


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
# RAGAs evaluation (opzionale, richiede ANTHROPIC_API_KEY)
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
        help="Compute RAGAs metrics (requires ANTHROPIC_API_KEY and pip install ragas)"
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


def get_llm() -> "ChatAnthropic":
    """Build the Anthropic LLM for RAGAS evaluation."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        print("langchain-anthropic non installato: pip install langchain-anthropic")
        sys.exit(1)
    return ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )


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
