# FW9b - Confirmed signatures (Task 0, no spend)

Read on 2026-06-06. Real names found, to be reused (no re-derivation):

- `step_3_4_evaluation.py`
  - `route_question(question) -> str` : 2-stage single-template router (keyword stage 1, semantic fallback stage 2 gated by conf+margin). Stays unchanged (v7 single path).
  - `keyword_candidates(q_lower) -> list[str]` : DETERMINISTIC (iterates KEYWORD_ROUTING in order, no set). Not a source of non-determinism.
  - `entity_fallback(q_lower) -> str`, `KEYWORD_ROUTING: list[(keywords, tid)]`, `ENABLE_SEMANTIC_FALLBACK: bool`.
  - `cypher_rows_to_context(rows) -> str` : legacy "k: v | k: v" formatter, 40-row cap. Single path keeps this.
  - `run_graph_rag(questions, driver, llm=None)` : supports `llm=None` (context-only) for the dry-run gate.
- `semantic_router.py`
  - `SemanticRouter.score(query, candidates=None) -> list[(tid, conf)]` : sort is `key=lambda kv: kv[1], reverse=True` (ties break on dict insertion order). Task 1 target: change to `(-score, tid)` for an explicit stable order, plus a fixed embedding seed.
  - `get_default_router()` singleton; backend switch via env `SEMANTIC_ROUTER_BACKEND` (tfidf default, `st` = BGE BAAI/bge-small-en-v1.5). BGE anchor cache present under data/semantic_router_cache.
- `prompt5_retrieval.py` (v7 building blocks ALREADY present, reuse):
  - `route_question_multi(question, n=5, use_semantic, router, conf_floor)` : keyword candidates first, then semantic ranking, deterministic, capped at n.
  - `union_template_rows(template_ids, run_template, row_cap=40)` : union + dedup (json stable key) + cap. Pure (injected `run_template`).
  - `densify_context(rows, max_rows=40)` : generic verbalizer; Task 3 enhances to per-template + controlled standard-constant exception (OOD03).
  - `refine_error_class(...)` : splits under_retrieval from synthesis_fail.
- `step_3_11_ood_eval.py` : OOD harness. Wiring point lines 159-168 (route -> Cypher -> context -> Haiku answer -> Sonnet judge). Paid step = Haiku answer + Sonnet judge per scored query. Writes per_query.csv + summary.csv + evaluation_results_graph-rag-ood_<ts>.json. RETRIEVAL_MODE branch goes here (single = byte-equivalent).
- `_router_rank_check.py` : pool builder `combined_pool(question, router, k=8)` = keyword_candidates first then `router.score()[:k]`, deduped. Equivalent to `route_question_multi`. Reads `results/_ab_split_input.json`.
- `step_3_9_llm_judge.py` : `_build_judge_prompt`, `_parse_judge_output`, `JUDGE_MODEL` (Sonnet). Judge = paid.
- `config.py` : reads secrets from env/.env, raises if unset. Never print.

Environment (2026-06-06): .env present, secrets load via config. sentence-transformers + sklearn + neo4j installed. BGE anchor cache present (router runs offline). Frozen `results/_mt_retrieval_sim.json` (20 items with union_context) usable for offline oracle. NEO4J NOT reachable (ServiceUnavailable): Task 6 live dry-run and Task 7 paid run are blocked until Neo4j is started. All other tasks are no-spend and offline-verifiable.
