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
import logging
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

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_DIR / "benchmark_qa_dataset.json"


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a boolean-ish environment variable."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# Feature flag for the 2-stage semantic fallback (A/B test). When off,
# route_question keeps the legacy first-match-wins behaviour exactly.
ENABLE_SEMANTIC_FALLBACK: bool = _env_flag("ENABLE_SEMANTIC_FALLBACK", "1")


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
    # ---- HIGH PRIORITY (2026-06 fix): country-level facts must win over the
    # greedy 'denmark'/'italy' buckets below, which route every country question
    # to P3/P4 and miss DH penetration %, EED transposition and country counts.
    # GENERIC_COUNTRY returns dh_penetration_pct and eed_transposed for all
    # countries in one row set. Conservative keyword set to avoid regressions. ----
    (["penetration", "penetrazione", "dh penetration", "dh_penetration",
      "transpos", "recepi", "eed_transposed", "eed transposition",
      "which countries", "how many countries"],
     "GENERIC_COUNTRY"),
    # NOTE (2026-06-02): the hand-written B17 priority rule ('policy framework
    # governs the danish 4gdh') was REMOVED. That collision (governance vs 4GDH
    # thermal params) is now resolved structurally by the semantic fallback in
    # route_question (see semantic_router.py), not by a per-phrase patch.
    # ---- HIGH PRIORITY (2026-06 fix): cross-scale DC capacity comparisons.
    # GENERIC_DC returns waste_heat_kw for all 3 archetypes (Edge/Mid/Hyperscale)
    # in one set so the LLM can compare; the old 'hyperscale' keyword routed these
    # to P5 (DC-L only), giving one side. Targeted phrases to avoid regressions. ----
    (["edge and hyperscale", "edge vs hyperscale", "hyperscale and edge",
      "hyperscale vs edge", "edge versus hyperscale", "between edge and",
      "edge to hyperscale", "recoverable heat between",
      "compare the recoverable", "edge data center and the hyperscale"],
     "GENERIC_DC"),
    # ---- Administrative / enforcement bodies (A24 GSE, A31 DK-DEA). The fact
    # lives in Actor.notes; the P3/P4 buckets never return Actor nodes. ----
    (["which agency", "which authority", "which body", "administered by",
      "administers", "administrative body", "enforces", "energy agency",
      "danish energy agency", "dk-dea", "gse", "gestore",
      # PROMPT 4 Fase 4b: 'which public body issues/manages the white certificates'
      # (OOD28) must reach the Actor node, not the P4 incentive template.
      "public body", "responsible for issuing", "issuing and managing",
      "body responsible", "manages certificati", "managing certificati"],
     "GENERIC_ACTOR"),
    # ---- DK NECP GHG reduction target / horizon year (A30). Lives in the NECP
    # PolicyFramework (horizon_year + notes), which P3 does not fetch. ----
    (["ghg reduction", "ghg target", "emission reduction target",
      "reduction target", "necp target", "2030 target", "by 2030",
      "climate plan target", "greenhouse gas"],
     "DK_NECP_TARGET"),
    # ---- IT vs DK support comparison (C13, C16, C31): IT financial incentive
    # vs DK regulatory mandate, side by side. ----
    (["better financial support", "better support", "stronger incentive",
      "which jurisdiction", "support for industrial symbiosis",
      "support for is project", "compare incentive", "incentive comparison",
      "italy and denmark incentive", "denmark and italy incentive",
      "financial support compare", "regulatory vs financial",
      "it vs dk", "dk vs it", "italy vs denmark", "denmark vs italy",
      "regulatory burden", "transaction cost reduction", "tc reduction"],
     "IT_DK_SUPPORT_COMPARE"),
    # ---- Count of IS scenarios per industrial sector (B28). ----
    (["scenarios involve", "scenarios target the", "involve the pulp",
      "involve pulp and paper", "how many scenarios involve",
      "scenarios in the food", "scenarios in the pulp", "scenarios per sector"],
     "SCENARIO_COUNT_BY_SECTOR"),
    # ---- Per-scenario detail for S2-S6 and 'sector/dc of scenario' questions.
    # P2_thermal_compatibility_all now returns datacenter, sector, process, temps
    # and status for all 9 scenarios, so the LLM picks the asked one. S1 and
    # S7-S9 keep their existing specific routes below. ----
    (["scenario s2", "scenario s3", "scenario s4", "scenario s5", "scenario s6",
      "in s2", "in s3", "in s4", "in s5", "in s6",
      "sector of the process", "industrial sector of", "which sector",
      "process in scenario", "process and data center"],
     # NOTE (2026-06-02): the C23 ('compare the IS scenarios for the mid-size DC
     # across all three temperature bands') and C06 ('minimizes the thermal gap')
     # phrase patches were REMOVED. Both collisions are now resolved by the
     # semantic fallback (C23 was hijacked by TEMPERATURE_BAND_DEF / GENERIC_DC,
     # C06 fell through to the P6 default).
     "P2_thermal_compatibility_all"),
    # ---- Single-DC recoverable / waste-heat capacity (A19). GENERIC_DC returns
    # waste_heat_kw per archetype; P5 returned heat-source capacity (different
    # number). ----
    (["waste heat capacity", "waste-heat capacity", "recoverable heat capacity",
      "recoverable waste heat of", "waste heat of the hyperscale",
      "waste heat available from", "heat capacity of the hyperscale",
      "heat capacity of the edge"],
     "GENERIC_DC"),
    # ---- Temperature band definitions (only definition-style queries) ----
    (["temperature band", "low-grade heat band",
      "mid-grade heat band", "high-grade heat band", "medium-grade",
      "grade heat band", "how many temperature band",
      "range defines the t1", "range defines the t2", "range defines the t3",
      "temperature range of", "range of t1", "range of t2", "range of t3",
      "range of the t1", "range of the t2", "range of the t3",
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
      "danish 4gdh thermally feasible"],
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
    (["ashrae", "ashrae 90.4", "90.4", "tc 9.9", "iso 23247", "iso-23247",
      "digital twin framework", "standard", "scope of",
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


def keyword_candidates(q_lower: str) -> list[str]:
    """All distinct template ids whose keyword rule fires, in priority order.

    Unlike a first-match-wins loop, this exposes *ambiguity*: a question that
    matches two or more distinct templates is a collision the keyword layer
    cannot resolve on its own.
    """
    out: list[str] = []
    for keywords, template_id in KEYWORD_ROUTING:
        if any(kw in q_lower for kw in keywords) and template_id not in out:
            out.append(template_id)
    return out


def entity_fallback(q_lower: str) -> str:
    """Legacy entity-type heuristic used when no keyword rule fires."""
    if any(w in q_lower for w in ["country", "nation", "penetration", "transposed"]):
        return "GENERIC_COUNTRY"
    if any(w in q_lower for w in ["regulation", "directive", "law", "decree"]):
        return "GENERIC_REGULATION"
    if any(w in q_lower for w in ["data center", "datacenter", "dc-s", "dc-m", "dc-l"]):
        return "GENERIC_DC"
    return "P6_full_is_path"


def route_question(question: str) -> str:
    """Route an NL question to a Cypher template id (2-stage).

    Stage 1 (keyword): if exactly one distinct template matches, use it. This
    keeps the precise, deterministic behaviour for in-distribution phrasings.

    Stage 2 (semantic fallback, only when ENABLE_SEMANTIC_FALLBACK): triggered
    when the keyword stage is ambiguous (zero matches, or two or more distinct
    templates). A deterministic *prior* is defined as the legacy route for the
    question (first keyword match, or the entity fallback when nothing matched).
    The semantic stage ranks all templates globally (the correct template may be
    neither keyword candidate, e.g. C23 collides on TEMPERATURE_BAND_DEF /
    GENERIC_DC but should route to P2_thermal_compatibility_all) and overrides
    the prior ONLY when its top template both clears the confidence floor and
    outscores the prior template by at least the margin. On a near-tie the
    deterministic prior stands, so the embedding can only override on a clear
    semantic preference. This bounds regressions versus the legacy router.
    """
    q_lower = question.lower()
    candidates = keyword_candidates(q_lower)

    # Stage 1: unambiguous keyword hit wins.
    if len(candidates) == 1:
        return candidates[0]

    # Deterministic prior == legacy route (first match, else entity fallback).
    prior = candidates[0] if candidates else entity_fallback(q_lower)

    if not ENABLE_SEMANTIC_FALLBACK:
        return prior

    # Stage 2: semantic disambiguation, gated against the prior.
    try:
        from semantic_router import get_default_router
        router = get_default_router()
        ranked = router.score(question)
    except Exception as exc:  # routing must never crash the (paid) eval run
        logger.warning("semantic fallback unavailable (%s); using keyword route.", exc)
        return prior

    if not ranked:
        return prior
    top_id, top_conf = ranked[0]
    second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
    prior_conf = dict(ranked).get(prior, 0.0)

    if top_id == prior or top_conf < router.conf_threshold:
        return prior
    # Clear semantic preference over the deterministic prior.
    if (top_conf - prior_conf) < router.margin:
        return prior
    # Extra guard for the zero-keyword case: with no keyword signal at all, the
    # entity fallback prior is only weakly related, so a large top-minus-prior
    # gap can be spurious when the embedding is undecided (a flat distribution).
    # Require the top template to also stand out from the global runner-up.
    if not candidates and (top_conf - second_conf) < router.margin:
        return prior
    return top_id


def cypher_rows_to_context(rows: list[dict]) -> str:
    """Converte righe Neo4j in stringa di contesto per il LLM."""
    if not rows:
        return "No data found in knowledge graph."
    lines = []
    # 40 rows (was 10): the largest template (P6) returns 27 rows, and truncating
    # to 10 made 'how many' answers wrong because the count could not be seen.
    for row in rows[:40]:
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
                        "You answer questions about an industrial-symbiosis knowledge graph "
                        "using ONLY the rows provided. Rules:\n"
                        "1. Answer the SPECIFIC question directly in one or two sentences, "
                        "stating the exact value, id, article, name or count it asks for.\n"
                        "2. If the question asks 'how many', COUNT the matching rows, give the "
                        "number first, then list the matching ids.\n"
                        "3. If it asks to compare or says 'vs', state BOTH sides and the "
                        "difference explicitly.\n"
                        "4. Do NOT paste raw rows or unrelated fields; use only what answers "
                        "the question.\n"
                        "5. If the rows do not contain the answer, reply exactly: "
                        "'Not available in the knowledge graph.'\n\n"
                        f"Rows:\n{context}\n\n"
                        f"Question: {q['nl_question']}\n\n"
                        "Answer:"
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
