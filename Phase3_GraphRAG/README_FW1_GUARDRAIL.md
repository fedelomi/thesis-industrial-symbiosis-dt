# FW1 Guardrail Tool Chain (step_3_14 .. step_3_17)

Costruito il 2026-06-13 (sessione Cowork, items 1-3 del piano Phase 3).
Nessun file canonico toccato: `route_question`, `step_3_4_evaluation.py`,
benchmark v2 e tutti i CSV canonici restano invariati. Tutto atterra come
addendum nello spirito di Section 6.2.2-bis.

## Scoperta preliminare (step_3_14, GIA ESEGUITO)

`results/step_3_14_context_mode_audit.csv` documenta che **tutti e cinque i
run graph-rag dell'era canonica (incluso quello da EM strict 0.59) sono in
context-only mode**: `answer == context` su 100/100 query (ramo `llm=None`
di `run_graph_rag`). I baseline no-RAG (0.28) e LLM-Cypher (0.2737) sono
invece LLM-in-loop, come gli arm OOD del 2-3 giugno. Gate di parità del
matcher EM: PASS (1.0 su tutti gli artefatti).

Implicazione per il manoscritto (da triagare, nessun capitolo modificato):

- Il 59% canonico misura router + Cypher + densificazione SENZA il passo di
  rendering Haiku. I failure mode del judge (B02/B07/B08, "raw Cypher-result
  dictionaries") sono coerenti con questo.
- Lettura difensiva possibile e onesta: il benchmark canonico è
  deterministico e riproducibile a costo zero API; il contributo del layer
  LLM è misurato separatamente (step_3_15) e il confronto +31.1 pp resta
  valido come confronto fra pipeline-as-designed, da dichiarare con
  precisione in §4.3.3 / caption Table 5.5.
- VERIFICA con la tua memoria di progetto prima di editare i capitoli: se
  esisteva un run canonico with-LLM andato perso, lo step_3_15 lo
  ricostruisce comunque dagli stessi contesti.

## Ordine di esecuzione (sulla tua macchina, serve solo ANTHROPIC_API_KEY)

Nessuno step richiede Neo4j: i contesti sono già negli artefatti JSON.

```powershell
# 0. Test offline (nessuna API):
python -m pytest tests/test_fw1_guardrail.py -q     # atteso: 12 passed

# 1. Re-run con Haiku dai contesti salvati (item 1 rivisto):
python step_3_15_llm_answer_rerun.py                # dry-run: 3 query + stima costo
python step_3_15_llm_answer_rerun.py --full --yes   # ~100 chiamate Haiku, < 0.10 USD

# 2. Guardrail FW1 misurato (item 2):
python step_3_16_runtime_guardrail.py               # dry-run
python step_3_16_runtime_guardrail.py --full --yes  # ~0.30-0.80 USD (in-distribution)
python step_3_16_runtime_guardrail.py --arm ood --full --yes   # opzionale, 38 query

# 3. Third judge indipendente + spot-check umano (item 3):
python step_3_17_third_judge_eval.py                # dry-run
python step_3_17_third_judge_eval.py --full --yes   # default claude-opus-4-8
#    poi compila a mano human_verdict in results/step_3_17_spot_check_sample.csv
```

Ogni script ha `--dry-run` di default, gate di costo `--yes`, logging dei
token e costo indicativo nel summary CSV. La API key è letta dall'ambiente
(pattern config.py); non viene mai loggata né scritta negli artefatti.

## Separazione metodologica (item 3, da dichiarare nel manoscritto)

Tripla pairwise-distinct e ruoli:

| Ruolo | Modello | Vede la ground truth? |
|-------|---------|----------------------|
| Generatore | claude-haiku-4-5-20251001 | mai |
| Gate runtime (componente del sistema) | claude-sonnet-4-6 | mai (verifica grounding vs rows) |
| Evaluator terzo (fuori dal sistema) | claude-opus-4-8 (THIRD_JUDGE_MODEL) | sì (rubrica binaria stile step_3_9) |

Il test `test_guardrail_ground_truth_never_in_prompts` blocca a livello di
unit test ogni leak della ground truth nel path runtime. Lo spot-check umano
stratificato (5 accordi, 5 disaccordi, 5 astensioni) chiude il cerchio.

## Metriche riportate da step_3_16 (selective prediction)

- `em_effective`: EM strict con le astensioni contate come errore
  (confrontabile con il 59% canonico).
- `coverage` e `em_on_answered`: la coppia selettiva (quanto risponde, quanto
  è accurato quando risponde).
- `retry_rate`, `abstain_rate`, token e costo.

## Risultati misurati (run 2026-06-13, costo totale ~1.21 USD)

| Configurazione (stessi 100 query, stessi contesti) | Metrica | Valore |
|---|---|---|
| Context-only (canonical, step_3_4) | EM strict | 0.59 |
| Context-only (canonical, step_3_9) | Sonnet semantic | 0.41 |
| NL Haiku (step_3_15) | EM strict | 0.45 (-14 pp) |
| NL + guardrail (step_3_16) | EM strict effective | 0.41 (coverage 0.99, retry 5%, abstain 1%) |
| NL + guardrail (step_3_17) | Opus-4-8 semantic | 0.34 (agreement con EM 0.85) |

Per categoria (third judge): factual-lookup 0.74, multi-hop-is 0.21,
comparative 0.06. Le 16 query perse dal rendering NL su EM strict includono
esattamente i falsi positivi del judge canonico (A14, A19, B02, B07...).

## Estensione cross-model-class (run 2026-06-13, +~2.0 USD)

Stessi 100 query e stessi contesti salvati, tre classi di generatore, judge
terzo claude-opus-4-8 su tutte:

| Generatore | EM strict | Semantic (Opus) | Coverage | Sem. su risposte date |
|---|---|---|---|---|
| Haiku 4.5 + guardrail | 0.41 | 0.34 | 0.99 | 0.34 |
| Sonnet 4.6 (citabile) | 0.45 | 0.40 | 0.62 | 0.645 |
| Fable 5 (probe, contaminazione dichiarata) | 0.48 | 0.38 (0.398 su subset pulito) | 0.74 | 0.514 |

Per categoria (semantic): **factual-lookup 0.74 / 0.74 / 0.74, identico su
tutte e tre le classi**; multi-hop 0.21 / 0.30 / 0.30 (plateau a 0.30);
comparative 0.06 / 0.15 / 0.09. Scorecard delle predizioni pre-registrate:
3/4 centrate (EM strict 0.45 nel range 0.44-0.50; factual a 0.74; semantic
0.40 nel range 0.35-0.45), 1 mancata onestamente (abstention rate 38% sopra
il range previsto 5-20%: i modelli piu' capaci si astengono PIU' del
previsto, non meno).

Conclusione cross-class, la risposta misurata alla critica "bastava un
modello migliore": salire di due classi di modello a retrieval costante
compra +6 pp di semantic (0.34 -> 0.40) e soprattutto CALIBRAZIONE
(astensione dove il contesto non basta: coverage 0.99 -> 0.62, accuratezza
sulle risposte date 0.34 -> 0.645), mentre il soffitto factual resta
inchiodato a 0.74 e le categorie composizionali plateau a 0.30. Il limite
di Phase 3 e' retrieval piu' sintesi composizionale, non capacita' del
modello: "logica nel grafo, lingua nel modello" con quattro punti
sperimentali.

Tre conclusioni misurate (run guardrail):

1. **Il -14 pp di EM strict sotto rendering NL è la conferma indipendente
   dell'overstatement del keyword matcher**: il matcher premia il dump del
   contesto, che contiene le keyword per costruzione.
2. **Il collo di bottiglia è la sintesi composizionale, non grounding né
   rendering**: multi-hop 0.21 in-distribution converge col ceiling 0.22
   dell'OOD v7; comparative 0.06; factual-lookup regge a 0.74.
3. **FW1 validato come meccanismo di SAFETY, non come lift di accuracy
   in-distribution**: il gate Sonnet passa il 95% al primo colpo (le risposte
   sono grounded), 1 abstain, nessun lift semantico. L'aspettativa "from 41%
   upward" del paragrafo FW1 del manoscritto è falsificata onestamente; la
   leva di accuratezza confermata resta FW9-ter (decomposizione/compose-then-
   verify), già High in Table 6.1. Inter-judge gap Sonnet vs Opus ~7 pp:
   ulteriore datapoint sulla judge-variance di [@zheng2023judgellm].

## KG coverage audit (step_3_19, offline, 2026-06-13)

Domanda: "conviene rifare il grafo?". Risposta misurata: NO. Le 57 query
fallite semanticamente da tutte e tre le classi di generatore sono state
classificate incrociando contesti salvati e sorgenti di ingest
(`results/step_3_19_kg_coverage_audit.csv`):

| Classe | n | Leva corretta |
|---|---|---|
| routing / under-retrieval (fatto NEL grafo, mai recuperato) | 29 | multi-template union (v7, +55 pp) portata nel path canonico + FW9-quater |
| synthesis o metric artifact (fatto nel contesto) | 19 | FW9-ter (compose-then-verify) + semantic scoring |
| derived/borderline (risposta derivabile con aritmetica multi-fatto) | 8 | FW9-ter |
| **vero coverage gap** | **1** (B29, profile type baseload) | una riga di MERGE |

Il grafo contiene la risposta per almeno 50 delle 57 failure strutturali
(es. GSE, NECP horizon, DH penetration, 3GDH legacy: tutti presenti negli
ingest, mai raggiunti dal routing per quelle query). Ricostruire il grafo
attaccherebbe 1 failure su 57; le leve giuste sono retrieval reach e
sintesi composizionale, coerente con la decomposizione v7. Caveat: la
classificazione e' keyword-level automatica, revisione manuale consigliata
sul bucket synthesis (artefatti tipo A14 dove le keyword del GT compaiono
per coincidenza nel contesto).

## Multi-template in-distribution (step_3_20, 2026-06-13, ~1.14 USD)

Il path v7 (route top-5 + union + prune + densify) misurato per la prima
volta sul benchmark canonico in-distribution, a generatore costante (Haiku),
judge terzo Opus:

| Path (generatore Haiku, stessi 100 query) | EM strict | Semantic (Opus) |
|---|---|---|
| single-template + guardrail | 0.41 | 0.34 |
| **multi-template v7 (step_3_20)** | **0.69** | **0.63** |

Per categoria (semantic): factual-lookup 0.94, multi-hop 0.64 (da 0.21),
comparative 0.30 (da 0.06). Le 29 routing-failure dello step_3_19: contesto
con ground truth 97% (28/29), **17/29 sbloccate semanticamente (prima
0/29)**. Le 19 synthesis: 11/19 risolte (in larga parte erano under-retrieval
mascherato, coerente col round 4 del v7). Le 8 derived: 2/8 (residuo
composizionale, FW9-ter).

Scorecard predizioni step_3_20: P1 superata (97% contro soglia 40%), P2
centrata al bordo alto (EM 0.69 nel range 0.55-0.70), P3 mancata
onestamente nella direzione buona (le synthesis si sono mosse molto perche'
in parte erano routing mascherato).

**Catena dimostrata end-to-end con 6 esperimenti**: grafo sano (step_3_19:
1 solo coverage gap su 57 failure) -> il routing single-template era il
collo di bottiglia (step_3_20: +29 pp semantic a modello costante) -> la
classe di modello non lo era (cross-class: +6 pp) -> il residuo e'
composizione su query derived/comparative (FW9-ter). Il multi-template
in-distribution (0.63) supera anche l'OOD v7 (0.618), il canonico
context+judge (0.41) e ogni upgrade di modello (0.38-0.40). Canonici
invariati: tutto atterra in §6.2.2-ter come addendum.

## Snippet manoscritto (numeri finali; spot-check umano [TODO 11 righe])

Per nuova §6.2.2-ter (pattern addendum, canonici invariati):

> A post-freeze instrumentation audit (step_3_14) makes the answer-generation
> mode of every committed artefact explicit: the canonical 100-query Graph-RAG
> benchmark was evaluated in the deterministic context-only configuration
> (llm=None), which renders the canonical EM strict 0.59 fully reproducible
> without API access; the no-RAG and LLM-Cypher baselines are LLM-in-loop by
> construction. Re-generating the answers from the same stored contexts with
> the production Haiku model (step_3_15) yields EM strict 0.45: the -14 pp
> shift independently corroborates the keyword-matcher overstatement of
> Section 5.4.2, since the 16 queries lost under natural-language rendering
> include precisely the false positives surfaced by the canonical judge (A14,
> A19, B02, B07). Equipping the same path with a Sonnet-class runtime
> verification gate with one structured retry and abstention (FW1,
> step_3_16) yields an effective EM strict of 0.41 at a coverage of 99%
> (95% first-pass, 5% retry, 1% abstention), at ~2.1 API calls per query.
> Following the cross-model separation principle of [@zheng2023judgellm], the
> gate is a component of the system under test and is excluded from
> evaluation: final answers are scored by an independent third judge
> (claude-opus-4-8, step_3_17), yielding a semantic accuracy of 0.34 with
> 0.85 agreement against the EM matcher and a category profile that localises
> the residual ceiling on compositional synthesis (factual-lookup 0.74,
> multi-hop 0.21, comparative 0.06); the in-distribution multi-hop value
> converges with the 0.22 compliance-multi-hop ceiling measured on the OOD
> arm, identifying a single bottleneck across both distributions. The FW1
> guardrail is therefore validated as an abstention-safety mechanism rather
> than as an accuracy lift, and the accuracy lever is confirmed to be the
> compositional-synthesis item FW9-ter; the canonical Section 5.4 numbers
> remain unchanged.

Per la caption di Table 5.5 (precisione di wording, da triagare):

> EM strict measured in the deterministic context-only configuration of the
> Graph-RAG pipeline (step_3_4, llm=None); the no-RAG and LLM-Cypher
> baselines require the LLM by construction. See Section 6.2.2-bis for the
> LLM-in-loop and guardrail variants on the same retrieval contexts.

## Landing points

| Artefatto | Dove atterra |
|-----------|--------------|
| step_3_14 CSV | citabile in §4.3.3 / §6.2.2-bis (mode audit) |
| step_3_15 summary | addendum: contributo marginale del layer LLM |
| step_3_16 summary | FW1 row di Table 6.1 -> "preliminary validation delivered" |
| step_3_17 summary + spot-check | paragrafo di separazione metodologica |
