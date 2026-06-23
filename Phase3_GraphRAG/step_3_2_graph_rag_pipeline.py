"""
step_3_2_graph_rag_pipeline.py
================================
Phase 3 - Graph RAG IS (OS3, Strato 2)

Pipeline Graph RAG con LangChain + Neo4j.
Modalita disponibili:
  1. template  : Cypher templates pre-definiti per i 6 pattern query della tesi
  2. llm       : GraphCypherQAChain (LLM genera Cypher automaticamente)
  3. both      : esegue entrambe e confronta le risposte
  4. v7        : multi-template retrieval (production path, FW9-ter)
                 route_question_multi(n=5) -> union_template_rows -> prune_rows
                 -> densify_context_v2 -> Haiku answer.
                 Measured: +29 pp third-judge semantic on OOD held-out at
                 constant generator (Section 6.2.2-ter of the thesis).
                 Recommended mode for deployment.

Uso:
    python step_3_2_graph_rag_pipeline.py
    python step_3_2_graph_rag_pipeline.py --mode template
    python step_3_2_graph_rag_pipeline.py --mode llm
    python step_3_2_graph_rag_pipeline.py --mode both
    python step_3_2_graph_rag_pipeline.py --mode v7

Requisiti:
    pip install langchain langchain-community langchain-anthropic neo4j anthropic
    ANTHROPIC_API_KEY settata come variabile d'ambiente oppure nel file .env

Riferimento wiki: [[phase-1-2-3-roadmap]] Passo 3.2,
                  [[graph-rag-entity-schema]] sezione "Multi-hop Query Patterns"
                  [[lesson-3]] sezione 3.10-bis cross-link 6.4.1 + v7
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, exceptions as neo4j_exc

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY,
)
from templates import CYPHER_TEMPLATES

# LangChain imports — lazy per template mode (non serve LLM)
_LANGCHAIN_OK = False
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain_anthropic import ChatAnthropic
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
        print("Run: pip install langchain langchain-community langchain-anthropic")
        sys.exit(1)

    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        anthropic_api_key=ANTHROPIC_API_KEY,
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


# --- v7 multi-template pipeline (FW9-ter production path) ---

def _v7_answer_prompt(context: str, question: str) -> str:
    """Strict-EM-preserving answer prompt used by the v7 pipeline.

    Mirrors the canonical step_3_4 generator prompt so that the only experimental
    variable when comparing canonical vs v7 is the retrieval (the answer template
    stays constant). This is the *flat* prompt that emits a plain string answer.
    For the *structured* chain-of-thought variant (v7-cot), use
    `_v7_answer_prompt_structured` below.
    """
    return (
        "You are answering a question about data centre waste heat recovery and "
        "European energy regulation. Use ONLY the facts in the context block. "
        "Preserve verbatim numeric values, article numbers, identifiers and unit "
        "labels (do not paraphrase them). If the context does not contain enough "
        "information to answer, reply with exactly 'Not available in the "
        "knowledge graph.' and nothing else.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# --- v7 structured chain-of-thought variant (Lever 2: structured CoT) ---

def _v7_answer_prompt_structured(context: str, question: str) -> str:
    """Structured chain-of-thought variant of the v7 answer prompt.

    Asks the model to emit a single JSON object with four fields:
      - extracted_facts:    a list of literal facts from the context that bear
                            on the question (verbatim, preserves anchor tokens).
      - reasoning_chain:    a list of 1-5 short reasoning steps that compose
                            the extracted facts into a conclusion.
      - final_answer:       the final answer string (the only field scored by
                            EM strict / EM semantic; preserves verbatim numbers,
                            article numbers and identifiers from extracted_facts).
      - confidence:         one of 'high', 'medium', 'low'.

    Compared to the flat prompt of `_v7_answer_prompt`, this variant is
    expected to lift the semantic EM on compound multi-hop queries by making
    the synthesis chain explicit (the model decomposes the question into
    atomic fact extraction plus controlled composition in a single forward
    pass, mitigating the synthesis_fail floor of Section 6.2.2-ter without the
    extra API calls of true decomposition or compose-then-verify).

    The 'final_answer' field is the only one scored downstream so that EM
    metrics remain comparable to the flat variant.
    """
    return (
        "You are answering a question about data centre waste heat recovery "
        "and European energy regulation. Use ONLY the facts in the context "
        "block. Preserve verbatim numeric values, article numbers, "
        "identifiers and unit labels.\n\n"
        "Output ONE JSON object with these four fields and nothing else:\n"
        "  - \"extracted_facts\": list of the verbatim facts from the context "
        "that are relevant to the question.\n"
        "  - \"reasoning_chain\": list of 1-5 short reasoning steps that "
        "compose the extracted facts into the answer.\n"
        "  - \"final_answer\": the final answer string. If the context does "
        "not contain enough information, set this to exactly \"Not available "
        "in the knowledge graph.\".\n"
        "  - \"confidence\": one of \"high\", \"medium\", \"low\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "JSON:"
    )


def _parse_v7_structured_answer(raw: str) -> tuple[str, dict]:
    """Extract `final_answer` and full structured payload from JSON output.

    Falls back to the raw string if JSON parsing fails (degraded mode keeps
    the pipeline robust on rare LLM output drift; the abstention sentinel
    string remains the canonical "no answer" marker).

    Returns:
        (final_answer, structured_payload) where structured_payload is the
        parsed dict (or {} on parse failure).
    """
    import json as _json
    import re as _re
    s = raw.strip()
    # Strip a leading "JSON:" prompt echo if present
    if s.lower().startswith("json:"):
        s = s[5:].strip()
    # Common LLM artefact: wrapping in markdown code fence
    fence = _re.match(r"^```(?:json)?\s*(.+?)\s*```\s*$", s, _re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        payload = _json.loads(s)
    except Exception:  # noqa: BLE001 - tolerate parse failures
        # Last attempt: find first {...} block in the string
        block = _re.search(r"\{.*\}", s, _re.DOTALL)
        if block:
            try:
                payload = _json.loads(block.group(0))
            except Exception:  # noqa: BLE001
                return raw.strip(), {}
        else:
            return raw.strip(), {}
    final = str(payload.get("final_answer", "")).strip()
    if not final:
        return raw.strip(), payload
    return final, payload


def run_v7_queries(driver, questions: list[str], top_n: int = 5,
                   row_cap: int = 40, use_cot: bool = False) -> list[dict]:
    """Run the v7 multi-template retrieval path on a list of questions.

    Per query the pipeline executes:
      1. route_question_multi(question, n=top_n) -> list of template ids
      2. for each template id, execute its Cypher and collect rows
      3. union_template_rows + prune_rows (dedup, cap, lookup-pruning)
      4. densify_context_v2 -> declarative one-fact-per-line context string
      5. Anthropic Haiku call with either the canonical step_3_4 answer prompt
         (flat string output, default) or the structured chain-of-thought
         variant (use_cot=True; emits {extracted_facts, reasoning_chain,
         final_answer, confidence} JSON, parsed downstream so that only
         final_answer reaches the EM scoring)
      6. emit answer + retrieval diagnostics for downstream eval

    Cost: ~one Haiku call per question (~0.001 USD on the canonical 100-q
    benchmark, ~0.003 USD on the OOD 38-q benchmark with longer contexts).
    The use_cot variant doubles the average output tokens (200-500 instead of
    50-200) but stays well below 0.005 USD per query.

    Returns:
        List of dicts with keys: question, answer, pool, union_rows,
        context_chars, elapsed_ms, error. When use_cot=True the dict also
        carries 'cot_payload' (parsed JSON) and 'cot_confidence' for the
        downstream diagnostics.
    """
    # Lazy imports: only needed in v7 mode, do not break template-only runs.
    from prompt5_retrieval import (
        densify_context_v2, prune_rows, route_question_multi,
        union_template_rows,
    )
    from templates import CYPHER_TEMPLATES
    try:
        from anthropic import Anthropic
    except ImportError:
        print("anthropic SDK non disponibile. Run: pip install anthropic")
        sys.exit(1)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    results: list[dict] = []
    with driver.session(database=NEO4J_DATABASE) as session:

        def run_template(tid: str) -> list[dict]:
            cypher = CYPHER_TEMPLATES.get(tid)
            if not cypher:
                return []
            try:
                return session.execute_read(lambda tx: tx.run(cypher).data())
            except Exception:  # noqa: BLE001 - template error => empty rows
                return []

        for q in questions:
            t0 = time.time()
            try:
                pool = route_question_multi(q, n=top_n, use_semantic=True)
                union = prune_rows(
                    union_template_rows(pool, run_template, row_cap=row_cap),
                    q,
                )
                context = densify_context_v2(union, template_ids=pool)
                if use_cot:
                    prompt = _v7_answer_prompt_structured(context, q)
                    max_tok = 1024  # extra room for the CoT JSON envelope
                else:
                    prompt = _v7_answer_prompt(context, q)
                    max_tok = 512
                response = client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=max_tok,
                    temperature=LLM_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                if use_cot:
                    answer, payload = _parse_v7_structured_answer(raw)
                    cot_confidence = str(payload.get("confidence", ""))
                else:
                    answer = raw
                    payload = {}
                    cot_confidence = ""
                elapsed = (time.time() - t0) * 1000
                results.append({
                    "question": q,
                    "answer": answer,
                    "pool": "|".join(pool),
                    "union_rows": len(union),
                    "context_chars": len(context),
                    "elapsed_ms": elapsed,
                    "error": None,
                    "cot_payload": payload if use_cot else None,
                    "cot_confidence": cot_confidence,
                })
                print(f"\n  Q: {q}")
                print(f"  A: {answer[:200]}")
                if use_cot and payload:
                    print(f"  CoT: confidence={cot_confidence}, "
                          f"facts={len(payload.get('extracted_facts', []))}, "
                          f"steps={len(payload.get('reasoning_chain', []))}")
                print(f"  Pool: {pool} | union_rows: {len(union)} | "
                      f"context: {len(context)} chars | "
                      f"elapsed: {elapsed:.0f}ms")
            except Exception as exc:  # noqa: BLE001 - keep the run alive
                elapsed = (time.time() - t0) * 1000
                results.append({
                    "question": q,
                    "answer": "",
                    "pool": "",
                    "union_rows": 0,
                    "context_chars": 0,
                    "elapsed_ms": elapsed,
                    "error": str(exc),
                    "cot_payload": None,
                    "cot_confidence": "",
                })
                print(f"\n  Q: {q}")
                print(f"  ERROR: {exc}")
    return results


# --- Entry point ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Graph RAG pipeline")
    parser.add_argument(
        "--mode", choices=["template", "llm", "both", "v7"], default="template",
        help=("Execution mode (default: template). "
              "template: deterministic Cypher templates only; "
              "llm: GraphCypherQAChain auto-Cypher; "
              "both: template + llm side-by-side; "
              "v7: multi-template retrieval + Haiku answer (production path, "
              "+29 pp third-judge semantic on OOD held-out, FW9-ter, "
              "thesis Section 6.2.2-ter).")
    )
    parser.add_argument(
        "--v7-top-n", type=int, default=5,
        help="v7 mode only: number of templates to union per query (default 5)."
    )
    parser.add_argument(
        "--v7-row-cap", type=int, default=40,
        help="v7 mode only: max rows after union before pruning (default 40)."
    )
    parser.add_argument(
        "--v7-cot", action="store_true",
        help=("v7 mode only: enable the structured chain-of-thought variant. "
              "The LLM emits a JSON object with extracted_facts, "
              "reasoning_chain, final_answer, confidence; only the "
              "final_answer reaches EM scoring. Lever 2 of the FW11 roadmap.")
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

        if args.mode == "v7":
            print("\n-- Running v7 multi-template pipeline -----------------------")
            print(f"  top_n={args.v7_top_n}, row_cap={args.v7_row_cap}, "
                  f"model={LLM_MODEL}")
            print("  Reference: thesis Section 6.2.2-ter (FW9-ter probe), "
                  "+29 pp third-judge semantic on OOD held-out at constant "
                  "generator. Recommended production path.")

            # Demo questions: same 7 nl_questions of the template set, so that
            # the v7 output is directly comparable to template mode out of the box.
            demo_questions = [
                tpl["nl_question"] for tpl in QUERY_TEMPLATES.values()
            ]
            v7_results = run_v7_queries(
                driver, demo_questions,
                top_n=args.v7_top_n, row_cap=args.v7_row_cap,
                use_cot=args.v7_cot,
            )
            n_ok = sum(1 for r in v7_results if r["error"] is None)
            print(f"\n  v7 summary: {n_ok}/{len(v7_results)} questions answered.")

        print("\nDone.")
        print("Next: step_3_3_benchmark_qa_design.py  (100-question dataset) "
              "or step_3_4_evaluation.py (canonical EM benchmark) or "
              "step_3_20_multitemplate_indist.py (v7 in-distribution benchmark).")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
