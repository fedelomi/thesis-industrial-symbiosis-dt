"""
ask_llm.py
==========
Interactive Q&A over the Phase 3 Knowledge Graph.

Loop: write a natural-language question, the LLM generates Cypher,
queries Neo4j, returns the answer + the generated Cypher + raw rows.

Uso:
    cd "Fasi Applicative/Phase3_GraphRAG"
    python ask_llm.py

Type your question and press Enter. Type 'quit' or Ctrl+C to exit.

Prerequisiti:
- Neo4j running su bolt://127.0.0.1:7687 (Knowledge Graph caricato)
- .env con NEO4J_PASSWORD e ANTHROPIC_API_KEY
- pip install langchain langchain-community langchain-anthropic neo4j python-dotenv
"""

from __future__ import annotations

import sys
import time

from config import (
    NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
    LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY,
)

# LangChain imports
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain_anthropic import ChatAnthropic
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
except ImportError as exc:
    print(f"LangChain not available: {exc}")
    print("Run: pip install langchain langchain-community langchain-anthropic neo4j")
    sys.exit(1)


def build_chain() -> GraphCypherQAChain:
    """Build the same GraphCypherQAChain used in step_3_2."""
    print("Connecting to Neo4j ...")
    graph = Neo4jGraph(
        url="neo4j://127.0.0.1:7687",
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    print(f"Connected. Schema preview:")
    try:
        print("  " + str(graph.schema)[:300] + " ...")
    except Exception:
        pass

    print(f"\nInstantiating LLM: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=False,                # set True for full chain tracing
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )
    return chain


def ask_one(chain: GraphCypherQAChain, question: str) -> None:
    """Run one question through the chain and pretty-print the answer."""
    t0 = time.time()
    try:
        response = chain.invoke({"query": question})
    except Exception as exc:
        print(f"\nERROR: {exc}\n")
        return

    elapsed_ms = (time.time() - t0) * 1000

    # Extract intermediate steps (Cypher + raw context)
    cypher_used = ""
    raw_context = []
    if "intermediate_steps" in response:
        for step in response["intermediate_steps"]:
            if isinstance(step, dict):
                if "query" in step:
                    cypher_used = step["query"]
                if "context" in step:
                    raw_context = step["context"]

    answer = response.get("result", "")

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer.strip() if answer else "(empty)")

    print("\n" + "-" * 70)
    print("GENERATED CYPHER")
    print("-" * 70)
    print(cypher_used.strip() if cypher_used else "(none)")

    if raw_context:
        print("\n" + "-" * 70)
        print(f"RAW GRAPH RESULTS ({len(raw_context)} record/i)")
        print("-" * 70)
        for i, row in enumerate(raw_context[:10], 1):
            row_str = str(row)
            if len(row_str) > 200:
                row_str = row_str[:200] + " ..."
            print(f"  [{i}] {row_str}")
        if len(raw_context) > 10:
            print(f"  ... ({len(raw_context) - 10} altri record)")

    print(f"\nElapsed: {elapsed_ms:.0f} ms")
    print()


def main() -> None:
    print("=" * 70)
    print("  Phase 3 Knowledge Graph - Interactive Q&A")
    print("=" * 70)

    chain = build_chain()

    print("\n" + "=" * 70)
    print("  Ready. Type your question in plain English or Italian.")
    print("  Examples:")
    print("    - What is the EED Article 26 obligation for data centres above 1 MW?")
    print("    - Which heat recovery technologies are listed for the LowT band?")
    print("    - Quali documenti del corpus citano ISO 50001?")
    print("  Type 'quit' (or Ctrl+C) to exit.")
    print("=" * 70)

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

        ask_one(chain, question)


if __name__ == "__main__":
    main()
