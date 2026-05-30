"""
ask_template.py
================
Interactive Q&A with **deterministic routing** to Cypher templates.

This preserves the 100% LCR guarantee of Phase 3's thesis claim:
1. User asks a natural-language question
2. Keyword router (`route_question`) selects a template ID
3. The pre-written Cypher template runs against Neo4j
4. Raw rows are formatted as context
5. (Optional) LLM rephrases the context as natural-language answer

If the question does not match any keyword route, the script either
falls back to a generic template (covered) or declares out-of-scope.

NO LLM-generated Cypher is ever executed. Hallucination of property
names (like the `threshold_value` that ask_llm.py invented) is
structurally impossible.

Uso:
    cd "Fasi Applicative/Phase3_GraphRAG"
    python ask_template.py
    python ask_template.py --no-llm           # raw context only, no NL reformulation
"""

from __future__ import annotations

import argparse
import sys
import time

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY,
)
from templates import CYPHER_TEMPLATES
from step_3_4_evaluation import route_question, cypher_rows_to_context

from neo4j import GraphDatabase, exceptions as neo4j_exc

# LLM optional — only for natural-language rephrasing of the deterministic context
try:
    from langchain_anthropic import ChatAnthropic
    _LLM_OK = True
except ImportError:
    _LLM_OK = False


def build_llm():
    if not _LLM_OK:
        return None
    return ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )


def ask_one(driver, llm, question: str) -> None:
    """Run one question through routing + template + (optional) rephrasing."""
    t0 = time.time()

    # 1. Routing
    template_id = route_question(question)
    cypher = CYPHER_TEMPLATES.get(template_id)
    if cypher is None:
        print(f"\n[OUT OF SCOPE] No template matches this question.")
        print(f"Routing returned: {template_id} (not in CYPHER_TEMPLATES)")
        print(f"Coverage is limited to the 14 query families; rephrase your question")
        print(f"in terms of EED articles, ISO 50001, scenarios S1-S9, DC scales,")
        print(f"manufacturing processes, Italy/Denmark incentives or DH mandates.")
        return

    # 2. Execute deterministic Cypher
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = session.execute_read(lambda tx, c=cypher: tx.run(c).data())
    except Exception as exc:
        print(f"\nERROR executing Cypher: {exc}")
        return

    elapsed_cypher_ms = (time.time() - t0) * 1000

    # 3. Format context
    context = cypher_rows_to_context(rows)

    # 4. (Optional) Natural-language rephrasing — LLM sees only the deterministic context
    answer_nl = None
    elapsed_llm_ms = 0.0
    if llm is not None and rows:
        prompt = (
            f"Using ONLY the following knowledge graph data, answer the user's question concisely "
            f"in the same language as the question. Do not invent facts not present in the data.\n\n"
            f"Knowledge graph context:\n{context}\n\n"
            f"User question: {question}\n\n"
            f"Answer:"
        )
        t1 = time.time()
        try:
            resp = llm.invoke(prompt)
            answer_nl = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            answer_nl = f"(LLM rephrasing failed: {exc})"
        elapsed_llm_ms = (time.time() - t1) * 1000

    # 5. Pretty print
    print()
    print("=" * 70)
    print(f"ROUTING -> template_id = {template_id}")
    print("=" * 70)
    print(f"Rows returned: {len(rows)}")
    print(f"Cypher elapsed: {elapsed_cypher_ms:.0f} ms")
    if elapsed_llm_ms:
        print(f"LLM rephrase: {elapsed_llm_ms:.0f} ms")

    if answer_nl:
        print()
        print("ANSWER")
        print("-" * 70)
        print(answer_nl.strip())

    print()
    print("RAW DETERMINISTIC RESULTS")
    print("-" * 70)
    if rows:
        for i, row in enumerate(rows[:10], 1):
            row_str = ", ".join(f"{k}={v}" for k, v in row.items())
            if len(row_str) > 250:
                row_str = row_str[:250] + " ..."
            print(f"  [{i}] {row_str}")
        if len(rows) > 10:
            print(f"  ... ({len(rows) - 10} more rows)")
    else:
        print("  (no rows)")

    print()
    print("CYPHER EXECUTED (deterministic, pre-written)")
    print("-" * 70)
    print(cypher.strip())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 100% LCR template Q&A")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM rephrasing, show only raw deterministic context")
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 3 Knowledge Graph - Template-routed Q&A (100% LCR)")
    print("=" * 70)

    # Connect Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to {NEO4J_URI}")
    except neo4j_exc.ServiceUnavailable:
        print("Neo4j not reachable. Start Neo4j Desktop and select DBMS Graph_RAG_FL.")
        sys.exit(1)
    except neo4j_exc.AuthError:
        print("Auth error. Check NEO4J_PASSWORD in .env")
        sys.exit(1)

    llm = None if args.no_llm else build_llm()
    if llm is None and not args.no_llm:
        print("LangChain not installed - falling back to raw context output")
    elif llm is not None:
        print(f"LLM ready for NL rephrasing: {LLM_MODEL}")

    print("\n" + "=" * 70)
    print("  Type your question. The router selects a Cypher template")
    print("  (no LLM-generated Cypher, 100% LCR).")
    print("")
    print("  Coverage (14 families): EED articles, ISO 50001, scenarios")
    print("  S1-S9, DC scales Edge/Mid/Hyperscale, manufacturing processes,")
    print("  IT/DK incentives, thermal compatibility, full IS paths.")
    print("")
    print("  Try: 'What is the EED Article 26 threshold for data centres?'")
    print("       'Which scenarios use direct heat exchange?'")
    print("       'What incentives are available in Italy for WHR?'")
    print("       'Show all IS pathways for hyperscale data centres.'")
    print("       'Quali processi manifatturieri devono fare ISO 50001?'")
    print("")
    print("  Type 'quit' or Ctrl+C to exit.")
    print("=" * 70)

    try:
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Bye.")
                return

            ask_one(driver, llm, question)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
