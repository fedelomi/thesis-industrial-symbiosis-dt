"""
step_3_2_graph_rag_pipeline.py
================================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Pipeline Graph RAG con LangChain + Neo4j.
Modalita disponibili:
  1. template  : Cypher templates pre-definiti per i 6 pattern query della tesi
  2. llm       : GraphCypherQAChain (LLM genera Cypher automaticamente)
  3. both      : esegue entrambe e confronta le risposte

Uso:
    python step_3_2_graph_rag_pipeline.py
    python step_3_2_graph_rag_pipeline.py --mode template
    python step_3_2_graph_rag_pipeline.py --mode llm
    python step_3_2_graph_rag_pipeline.py --mode both

Requisiti:
    pip install langchain langchain-community langchain-openai neo4j
    OPENAI_API_KEY settata come variabile d'ambiente oppure nel file .env

Riferimento wiki: [[roadmap-fasi-1-2-3]] Passo 3.2,
                  [[graph-rag-entity-schema]] sezione "Multi-hop Query Patterns"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, exceptions as neo4j_exc

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, LLM_MODEL, LLM_TEMPERATURE
from templates import CYPHER_TEMPLATES

# LangChain imports — lazy per template mode (non serve LLM)
_LANGCHAIN_OK = False
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain_openai import ChatOpenAI
    # GraphCypherQAChain si trova in langchain_community dal v0.2+
    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError:
        from langchain.chains import GraphCypherQAChain  # fallback v0.1.x
    _LANGCHAIN_OK = True
except ImportError as e:
    # In template mode le librerie LLM non servono
    _LANGCHAIN_OK = False
    _LANGCHAIN_ERR = str(e)


# --- Query Templates (Blocco D2 della roadmap) ---
#
# 6 pattern multi-hop definiti in [[graph-rag-entity-schema]]:
#   P1 - Lookup fattuale (regolatorio)
#   P2 - Compatibilita termica DC -> processo
#   P3 - Screening normativo per scenario
#   P4 - Incentivi per paese e tecnologia
#   P5 - Confronto scenari
#   P6 - Path IS completo (multi-hop)

QUERY_TEMPLATES: dict[str, dict[str, Any]] = {

    "P1_eed_art26_threshold": {
        "description": "P1 - Lookup: soglia obbligo disclosure EED Art.26",
        "category": "factual-lookup",
        "hop_count": 2,
        "cypher": CYPHER_TEMPLATES["P1_eed_art26_threshold"],
        "ground_truth": "1.0 MW - mandatory disclosure",
        "nl_question": "What is the EED Article 26 threshold for data center waste heat disclosure?",
    },

    "P2_thermal_compatibility_S1": {
        "description": "P2 - Compatibilita termica: Scenario S1 (DC-S x T1)",
        "category": "thermal-compatibility",
        "hop_count": 4,
        "cypher": CYPHER_TEMPLATES["P2_thermal_compatibility_S1"],
        "ground_truth": "direct_HX supply 47.6C, process requires 60C -> REQUIRES_UPGRADE",
        "nl_question": "Is Scenario S1 (small DC, low-temperature process) thermally compatible with direct heat exchange?",
    },

    "P2_thermal_compatibility_all": {
        "description": "P2 - Compatibilita termica: tutti i 9 scenari",
        "category": "thermal-compatibility",
        "hop_count": 4,
        "cypher": CYPHER_TEMPLATES["P2_thermal_compatibility_all"],
        "ground_truth": "9 rows; direct_HX always GAP (47.6 < 60/90/130); HP ok for T1/T2; CO2_HTHP ok for all",
        "nl_question": "Which of the 9 IS scenarios are thermally compatible without additional heat pump?",
    },

    "P3_regulatory_screening_dk": {
        "description": "P3 - Screening normativo: obblighi DK per DC > 1MW",
        "category": "regulatory-screening",
        "hop_count": 3,
        "cypher": CYPHER_TEMPLATES["P3_regulatory_screening_dk"],
        "ground_truth": "DK-DH-PLANNING-ACT connection mandate >= 1MW; DC-M and DC-L must connect",
        "nl_question": "What are the mandatory DH connection obligations for data centers in Denmark?",
    },

    "P4_incentives_it_whr": {
        "description": "P4 - Incentivi IT per waste heat recovery",
        "category": "incentive-lookup",
        "hop_count": 3,
        "cypher": CYPHER_TEMPLATES["P4_incentives_it_whr"],
        "ground_truth": "TEE/CB: 300 EUR/toe (25.8 EUR/MWh), governed by DM-2021 + DM-MASE-2025",
        "nl_question": "What financial incentives are available in Italy for data center waste heat recovery?",
    },

    "P5_scenario_comparison_L": {
        "description": "P5 - Confronto scenari DC-L (hyperscale) per fascia termica",
        "category": "comparative-screening",
        "hop_count": 5,
        "cypher": CYPHER_TEMPLATES["P5_scenario_comparison_L"],
        "ground_truth": "S7(T1/direct_HX), S8(T2/HP), S9(T3/CO2_HTHP); capacity ~35,400 kW each",
        "nl_question": "Compare the three IS scenarios for a hyperscale data center across temperature bands.",
    },

    "P6_full_is_path": {
        "description": "P6 - Path IS completo: DC -> HeatSource -> TemperatureBand -> ManufacturingProcess -> IndustrialSector",
        "category": "full-path",
        "hop_count": 5,
        "cypher": CYPHER_TEMPLATES["P6_full_is_path"],
        "ground_truth": "27 rows (3 DC x 3 upgrade x 3 processo-settore combinations mapped through scenari)",
        "nl_question": "Show the complete IS symbiosis pathways from data centers to manufacturing sectors.",
    },
}


# --- Template runner ---

@dataclass
class QueryResult:
    template_id: str
    nl_question: str
    category: str
    hop_count: int
    cypher: str
    rows: list[dict]
    elapsed_ms: float
    error: str | None = None


def run_template_queries(driver) -> list[QueryResult]:
    """Esegue tutti i template Cypher e ritorna i risultati."""
    results = []
    with driver.session(database=NEO4J_DATABASE) as session:
        for tid, tpl in QUERY_TEMPLATES.items():
            t0 = time.time()
            try:
                rows = session.execute_read(
                    lambda tx, q=tpl["cypher"]: tx.run(q).data()
                )
                elapsed = (time.time() - t0) * 1000
                results.append(QueryResult(
                    template_id=tid,
                    nl_question=tpl["nl_question"],
                    category=tpl["category"],
                    hop_count=tpl["hop_count"],
                    cypher=tpl["cypher"].strip(),
                    rows=rows,
                    elapsed_ms=elapsed,
                ))
            except Exception as exc:
                elapsed = (time.time() - t0) * 1000
                results.append(QueryResult(
                    template_id=tid,
                    nl_question=tpl["nl_question"],
                    category=tpl["category"],
                    hop_count=tpl["hop_count"],
                    cypher=tpl["cypher"].strip(),
                    rows=[],
                    elapsed_ms=elapsed,
                    error=str(exc),
                ))
    return results


def print_template_results(results: list[QueryResult]) -> None:
    print("\n" + "="*60)
    print("  TEMPLATE MODE RESULTS")
    print("="*60)
    for r in results:
        print(f"\n[{r.template_id}]  {r.category}  ({r.hop_count}-hop)")
        print(f"  Q: {r.nl_question}")
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            print(f"  Rows: {len(r.rows)}  |  Elapsed: {r.elapsed_ms:.1f}ms")
            for row in r.rows[:3]:   # mostra max 3 righe
                print(f"    {row}")
            if len(r.rows) > 3:
                print(f"    ... ({len(r.rows) - 3} more rows)")
        ground_truth = QUERY_TEMPLATES[r.template_id].get("ground_truth", "")
        print(f"  Expected: {ground_truth}")


# --- LLM pipeline ---

def build_llm_chain(graph: Neo4jGraph) -> GraphCypherQAChain:
    """Costruisce la chain LangChain con LLM."""
    if not _LANGCHAIN_OK:
        print(f"LangChain non disponibile: {_LANGCHAIN_ERR}")
        print("Run: pip install langchain langchain-community langchain-openai")
        sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY non trovata — setta la variabile d'ambiente.")
        print("  Windows PowerShell: $env:OPENAI_API_KEY = 'sk-...'")
        sys.exit(1)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )
    return chain


def run_llm_queries(chain: GraphCypherQAChain, questions: list[str]) -> list[dict]:
    """Esegue le domande via LLM e ritorna le risposte."""
    results = []
    for q in questions:
        print(f"\n  Q: {q}")
        t0 = time.time()
        try:
            response = chain.invoke({"query": q})
            elapsed = (time.time() - t0) * 1000
            cypher_used = ""
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if "query" in step:
                        cypher_used = step["query"]
                        break
            results.append({
                "question": q,
                "answer": response.get("result", ""),
                "cypher_generated": cypher_used,
                "elapsed_ms": elapsed,
                "error": None,
            })
            print(f"  A: {response.get('result', '')[:200]}")
            print(f"  Cypher: {cypher_used[:150]}...")
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            results.append({
                "question": q,
                "answer": "",
                "cypher_generated": "",
                "elapsed_ms": elapsed,
                "error": str(exc),
            })
            print(f"  ERROR: {exc}")
    return results


# --- Entry point ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Graph RAG pipeline")
    parser.add_argument(
        "--mode", choices=["template", "llm", "both"], default="template",
        help="Execution mode (default: template)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Phase 3 - step_3_2_graph_rag_pipeline.py")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    # Connessione Neo4j driver
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to {NEO4J_URI} (db: {NEO4J_DATABASE})")
    except neo4j_exc.ServiceUnavailable:
        print("Neo4j not reachable.")
        sys.exit(1)
    except neo4j_exc.AuthError:
        print("Auth error.")
        sys.exit(1)

    try:
        if args.mode in ("template", "both"):
            print("\n-- Running Cypher template queries -------------------------")
            template_results = run_template_queries(driver)
            print_template_results(template_results)

        if args.mode in ("llm", "both"):
            print("\n-- Running LLM chain queries --------------------------------")
            # LangChain Neo4jGraph (usa URL diverso — neo4j:// invece di bolt://)
            graph = Neo4jGraph(
                url="neo4j://127.0.0.1:7687",
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
            )
            chain = build_llm_chain(graph)

            # Domande di test: prime 3 dal dataset template
            test_questions = [
                tpl["nl_question"]
                for tpl in list(QUERY_TEMPLATES.values())[:3]
            ]
            llm_results = run_llm_queries(chain, test_questions)

            if args.mode == "both":
                print("\n-- Comparison (Template vs LLM) -------------------------")
                for tr, lr in zip(template_results[:3], llm_results):
                    print(f"\n  [{tr.template_id}]")
                    print(f"  Template rows : {len(tr.rows)}")
                    print(f"  LLM answer    : {lr['answer'][:150]}")
                    print(f"  LLM cypher    : {lr['cypher_generated'][:150]}")

        print("\nDone.")
        print("Next: step_3_3_benchmark_qa_design.py  (100-question dataset)")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
