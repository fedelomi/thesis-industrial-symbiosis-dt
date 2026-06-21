# OOD coverage-gap review (benchmark_ood_v1)

Zero-API deterministic view of the KG_COVERAGE_GAP queries, for manual split into `kg_missing` / `retrieval_miss` / `ambiguous`. Hints are a PROPOSAL from KG-grounded heuristics; fill the **decision** field in `data/benchmarks/benchmark_ood_v1_review.jsonl`. Source benchmark stays UNCURATED until decisions are filled and `check_ood_coverage_split.py` is run.

- Coverage gaps: **16** of 38 OOD queries = 42.1% (the OOD-report headline, an UPPER bound).
- Proposed hint split (pre-review): {'retrieval_miss': 8, 'ambiguous': 4, 'kg_missing': 4}
- Provisional **true KG coverage gap** = kg_missing = 4/38 = 10.5% (a LOWER bound; the rest are retrieval misses or compound-composition needs, not missing KG facts). `check_ood_coverage_split.py` recomputes this from Fede's decisions.

Legend: gap_type_hint = kg_missing (KG lacks the fact, -> Cap 6.3 enrichment backlog) / retrieval_miss (fact present, top-3 routing missed it, -> routing or template fix) / ambiguous (needs Fede).

## Patterns aggregati

- **Missing EED Article 12 (energy audits)**: 3 gaps (OOD16, OOD21, OOD34) reference Art. 12 / energy-audit duties, but the KG only models EED Art. 23/24/26 + the delegated regulation. -> add an EED-ART-12 node (and Art. 11 audit-follow-up) to the KG.
- **Standards retrieval miss**: 5 gaps (OOD05, OOD07, OOD15, OOD26, OOD36) ask for ASHRAE 90.4 or ISO 23247, which DO exist as Standard nodes, but the router sent them to DC/article templates. -> routing/anchor fix (GENERIC_STANDARD), not KG enrichment.
- **Procedural / cost-allocation granularity**: 4 gaps (OOD11, OOD15, OOD31, OOD37) ask for legal step sequences or who bears cost, which are workflow/attribute facts the entity-relationship KG does not model. -> decide scope: model as procedure nodes or declare out-of-scope in Cap 6.3.
- **Compound multi-hop**: 4 gaps (OOD11, OOD15, OOD31, OOD36) chain 3+ anchors (regulation + standard + incentive); even with full coverage the single-template architecture cannot compose them. -> FW9-bis: multi-template composition, not just routing.


## compliance-multi-hop  (5 gaps)

### OOD11  [hard]  -> hint: **ambiguous** (med)

- **query**: Scenario S9 in the technical grid involves a hyperscale data center producing high-grade waste heat that is upgraded via a high-temperature heat pump to supply steam for sterilization in a pharmaceutical plant. If this facility is located in Italy, trace the full compliance pathway: which EU EED 2023/1791 articles are triggered, how would the plant manager certify energy management under ISO 50001, and what steps are needed to monetize the efficiency gain through the Certificati Bianchi scheme?
- **template_top3** (semantic conf): P1_eed_art26_threshold(0.764), ALL_REGULATORY_ARTICLES(0.762), ISO50001_PROCESS_COMPLIANCE(0.75)
- **KG_entities_retrieved**: Country:IT  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P1_eed_art26_threshold`): `MATCH (r:Regulation {id: 'EED-2023-1791'})-[:CONTAINS]->(a:RegulatoryArticle {id: 'EED-ART-26'}) RETURN a.title AS article, a.obligation_type AS obligation, a.threshold_value AS threshold_mw, a.threshold_unit AS unit, a.summary AS summary`
- **hint_reason**: compound multi-hop; missing/concept: ['procedural steps / legal sequence']; retrievable-unrouted via ['GENERIC_ACTOR', 'ISO50001_ARTICLES']; needs multi-template composition plus a semantic decision.
- **decision**: _______   **notes**: _______

### OOD13  [hard]  -> hint: **retrieval_miss** (med)

- **query**: An Italian mid-size data center wants to use a heat pump to upgrade its medium-grade waste heat for pasteurization in a food-processing plant next door. The plant manager intends to implement ISO 50001. How does the delegated regulation under EU EED 2023/1791 defining data center KPIs affect what the data center must report, and how can those reported KPIs serve as evidence when applying for Certificati Bianchi to cover the heat pump investment?
- **template_top3** (semantic conf): ALL_REGULATORY_ARTICLES(0.849), P1_eed_art26_threshold(0.787), P3_regulatory_screening_dk(0.748)
- **KG_entities_retrieved**: RegulatoryArticle:EED-ART-23, RegulatoryArticle:EED-ART-24, RegulatoryArticle:EED-ART-26, RegulatoryArticle:DEL-REG-ART-2, Regulation:EED-2023-1791  | top1_rows=4  | LCR=1.0
- **cypher_top1** (`ALL_REGULATORY_ARTICLES`): `MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle) RETURN r.id AS regulation, r.short_name AS reg_name, a.id AS article_id, a.article_number AS art_no, a.title AS title, a.obligation_type AS obligation, a.threshold_value AS threshold_m`
- **hint_reason**: fact(s) exist in KG via ['GENERIC_ACTOR', 'ISO50001_ARTICLES'] but top-3 routed to ['ALL_REGULATORY_ARTICLES', 'P1_eed_art26_threshold', 'P3_regulatory_screening_dk']; routing or template fix, not KG enrichment.
- **decision**: _______   **notes**: _______

### OOD14  [medium]  -> hint: **retrieval_miss** (high)

- **query**: Under EU EED 2023/1791 Article 24, what efficiency criteria must a waste heat recovery project meet to qualify as an efficient heating and cooling supply, and how would a scenario S4 mid-size data center using a standard heat pump to supply a food pasteurization process demonstrate compliance with those criteria?
- **template_top3** (semantic conf): ALL_REGULATORY_ARTICLES(0.795), P1_eed_art26_threshold(0.783), P3_regulatory_screening_dk(0.761)
- **KG_entities_retrieved**: RegulatoryArticle:EED-ART-23, RegulatoryArticle:EED-ART-24, RegulatoryArticle:EED-ART-26, RegulatoryArticle:DEL-REG-ART-2, Regulation:EED-2023-1791  | top1_rows=4  | LCR=1.0
- **cypher_top1** (`ALL_REGULATORY_ARTICLES`): `MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle) RETURN r.id AS regulation, r.short_name AS reg_name, a.id AS article_id, a.article_number AS art_no, a.title AS title, a.obligation_type AS obligation, a.threshold_value AS threshold_m`
- **hint_reason**: all referenced entities are retrieved by the routed template(s) ['ALL_REGULATORY_ARTICLES']; conservative false gap, not a true KG gap.
- **decision**: _______   **notes**: _______

### OOD15  [hard]  -> hint: **ambiguous** (med)

- **query**: If a hyperscale data center in Denmark is modelled using a digital twin based on ISO 23247, and the twin reveals that waste heat output exceeds what the local third-generation district heating network can absorb, what regulatory pathway under the Heat Supply Act and the EU EED Art. 23 assessment process should the operator follow to propose an upgrade to a fourth-generation network, and who bears the cost obligation?
- **template_top3** (semantic conf): P3_regulatory_screening_dk(0.836), P1_eed_art26_threshold(0.818), ALL_REGULATORY_ARTICLES(0.794)
- **KG_entities_retrieved**: DataCenter:DC-M, DataCenter:DC-L  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P3_regulatory_screening_dk`): `MATCH (c:Country {iso: 'DK'})-[:HAS_FRAMEWORK]->(pf:PolicyFramework) MATCH (pf)-[:CONTAINS]->(a:RegulatoryArticle) WHERE a.obligation_type = 'mandatory' MATCH (dc:DataCenter) WHERE dc.it_capacity_kw >= 1000 RETURN c.name AS country, pf.name`
- **hint_reason**: compound multi-hop; missing/concept: ['cost obligation / who bears cost']; retrievable-unrouted via ['DK_DH_COMPARE', 'GENERIC_STANDARD']; needs multi-template composition plus a semantic decision.
- **decision**: _______   **notes**: _______

### OOD16  [easy]  -> hint: **kg_missing** (high)

- **query**: A data center in Italy has never conducted an energy audit. Under Article 12 of EU EED 2023/1791, is it obligated to do so, and if the audit uncovers waste heat potential, which Italian incentive mechanism could fund the recovery infrastructure?
- **template_top3** (semantic conf): P4_incentives_it_whr(0.822), P1_eed_art26_threshold(0.793), ALL_REGULATORY_ARTICLES(0.773)
- **KG_entities_retrieved**: (none)  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P4_incentives_it_whr`): `MATCH (i:Incentive) WHERE 'waste-heat-recovery' IN i.eligible_tech MATCH (i)-[:GOVERNED_BY]->(r:Regulation) RETURN i.name AS incentive, i.value_eur_toe AS eur_per_toe, i.value_eur_mwh AS eur_per_mwh, i.eligible_tech AS eligible_technologies`
- **hint_reason**: referenced entity absent from KG: EED Art.12 energy audit; propose KG enrichment.
- **decision**: _______   **notes**: _______


## parameter-lookup  (2 gaps)

### OOD05  [medium]  -> hint: **retrieval_miss** (high)

- **query**: Which ASHRAE standard is specifically applicable to energy efficiency in data centers?
- **template_top3** (semantic conf): GENERIC_DC(0.797), P3_regulatory_screening_dk(0.787), P1_eed_art26_threshold(0.774)
- **KG_entities_retrieved**: DataCenter:DC-S, DataCenter:DC-M, DataCenter:DC-L  | top1_rows=3  | LCR=1.0
- **cypher_top1** (`GENERIC_DC`): `MATCH (dc:DataCenter) RETURN dc.id AS id, dc.scale AS scale, dc.it_capacity_kw AS it_kw, dc.cooling_type AS cooling, dc.pue_nominal AS pue, dc.waste_heat_kw AS waste_heat_kw ORDER BY dc.it_capacity_kw`
- **hint_reason**: fact(s) exist in KG via ['GENERIC_STANDARD'] but top-3 routed to ['GENERIC_DC', 'P3_regulatory_screening_dk', 'P1_eed_art26_threshold']; routing or template fix, not KG enrichment.
- **decision**: _______   **notes**: _______

### OOD07  [medium]  -> hint: **retrieval_miss** (high)

- **query**: What is the ISO standard number that defines a digital twin framework for manufacturing, relevant when modelling data center waste heat integration into a production line?
- **template_top3** (semantic conf): P1_eed_art26_threshold(0.761), ISO50001_PROCESS_COMPLIANCE(0.754), P2_thermal_compatibility_all(0.753)
- **KG_entities_retrieved**: Country:IT  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P1_eed_art26_threshold`): `MATCH (r:Regulation {id: 'EED-2023-1791'})-[:CONTAINS]->(a:RegulatoryArticle {id: 'EED-ART-26'}) RETURN a.title AS article, a.obligation_type AS obligation, a.threshold_value AS threshold_mw, a.threshold_unit AS unit, a.summary AS summary`
- **hint_reason**: fact(s) exist in KG via ['GENERIC_STANDARD'] but top-3 routed to ['P1_eed_art26_threshold', 'ISO50001_PROCESS_COMPLIANCE', 'P2_thermal_compatibility_all']; routing or template fix, not KG enrichment.
- **decision**: _______   **notes**: _______


## cross-comparison  (4 gaps)

### OOD19  [easy]  -> hint: **retrieval_miss** (high)

- **query**: Comparing third-generation and fourth-generation district heating networks in Denmark, what is the key difference in operating temperature that determines whether a data center can supply waste heat via direct heat exchange or must use a heat pump?
- **template_top3** (semantic conf): DK_DH_COMPARE(0.853), DK_4GDH_PARAMS(0.838), DK_3GDH_PARAMS(0.837)
- **KG_entities_retrieved**: (none)  | top1_rows=2  | LCR=0.0
- **cypher_top1** (`DK_DH_COMPARE`): `MATCH (dh:DHNetwork {country: 'DK'}) RETURN dh.generation AS generation, dh.name AS name, dh.supply_temp_c AS supply_temp_c, dh.return_temp_c AS return_temp_c, dh.capacity_mw AS capacity_mw, 'degrees Celsius' AS temp_unit, 'MW' AS capacity_`
- **hint_reason**: routed template returns the fact but the neuro-symbolic regex matched no node id (value columns); false gap, not a true KG gap.
- **decision**: _______   **notes**: _______

### OOD21  [medium]  -> hint: **kg_missing** (high)

- **query**: How do the energy audit requirements of EU EED Art. 12 and the energy management system certification under ISO 50001 differ in terms of frequency, scope, and the actor responsible for compliance in a data center that also supplies waste heat to a manufacturer?
- **template_top3** (semantic conf): P1_eed_art26_threshold(0.791), ISO50001_ARTICLES(0.787), ALL_REGULATORY_ARTICLES(0.787)
- **KG_entities_retrieved**: Country:IT  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P1_eed_art26_threshold`): `MATCH (r:Regulation {id: 'EED-2023-1791'})-[:CONTAINS]->(a:RegulatoryArticle {id: 'EED-ART-26'}) RETURN a.title AS article, a.obligation_type AS obligation, a.threshold_value AS threshold_mw, a.threshold_unit AS unit, a.summary AS summary`
- **hint_reason**: referenced entity absent from KG: EED Art.12 energy audit; propose KG enrichment.
- **decision**: _______   **notes**: _______

### OOD23  [hard]  -> hint: **retrieval_miss** (med)

- **query**: Scenario S5 (mid-size data center, medium-grade heat, heat pump, pasteurization) and scenario S8 (hyperscale data center, medium-grade heat, heat pump, paper drying) both use a standard heat pump as the upgrade technology. Comparing these two scenarios, which is more likely to satisfy the efficiency criteria of EU EED Art. 24, and how would the Certificati Bianchi valuation differ given the different scales of energy savings delivered?
- **template_top3** (semantic conf): P2_thermal_compatibility_S1(0.758), P3_regulatory_screening_dk(0.744), P5_scenario_comparison_L(0.736)
- **KG_entities_retrieved**: Scenario:S1, DataCenter:DC-S, TemperatureBand:T1  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P2_thermal_compatibility_S1`): `MATCH (s:Scenario {id: 'S1'}) MATCH (s)-[:USES_DC]->(dc:DataCenter) MATCH (s)-[:TARGETS_PROCESS]->(mp:ManufacturingProcess) MATCH (s)-[:HAS_HEATSOURCE]->(hs:HeatSource) MATCH (hs)-[:IN_BAND]->(tb:TemperatureBand) RETURN s.id AS scenario, dc`
- **hint_reason**: fact(s) exist in KG via ['ALL_REGULATORY_ARTICLES', 'GENERIC_ACTOR'] but top-3 routed to ['P2_thermal_compatibility_S1', 'P3_regulatory_screening_dk', 'P5_scenario_comparison_L']; routing or template fix, not KG enrichment.
- **decision**: _______   **notes**: _______

### OOD26  [easy]  -> hint: **retrieval_miss** (high)

- **query**: Comparing ASHRAE 90.4 and ISO 50001 in the context of a data center that supplies waste heat to a manufacturer, which standard focuses on the energy efficiency of the data center facility itself, and which focuses on the broader energy management system that would govern the heat supply arrangement?
- **template_top3** (semantic conf): ISO50001_ARTICLES(0.787), P1_eed_art26_threshold(0.769), ISO50001_PROCESS_COMPLIANCE(0.765)
- **KG_entities_retrieved**: Standard:ISO-50001-2018, Country:IT, Country:EU, Country:DK  | top1_rows=5  | LCR=1.0
- **cypher_top1** (`ISO50001_ARTICLES`): `MATCH (s:Standard {id: 'ISO-50001-2018'})-[:CONTAINS]->(a:RegulatoryArticle) RETURN s.id AS standard, s.name AS standard_name, s.amendment AS amendment, a.article_number AS article, a.title AS title, a.obligation_type AS obligation, a.summa`
- **hint_reason**: fact(s) exist in KG via ['GENERIC_STANDARD'] but top-3 routed to ['ISO50001_ARTICLES', 'P1_eed_art26_threshold', 'ISO50001_PROCESS_COMPLIANCE']; routing or template fix, not KG enrichment.
- **decision**: _______   **notes**: _______


## regulatory-traversal  (5 gaps)

### OOD28  [easy]  -> hint: **retrieval_miss** (high)

- **query**: In Italy, which public body is responsible for issuing and managing Certificati Bianchi, and what role does it play when a data center operator applies for white certificates for a waste heat recovery project?
- **template_top3** (semantic conf): P4_incentives_it_whr(0.705), P1_eed_art26_threshold(0.698), ALL_REGULATORY_ARTICLES(0.697)
- **KG_entities_retrieved**: (none)  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P4_incentives_it_whr`): `MATCH (i:Incentive) WHERE 'waste-heat-recovery' IN i.eligible_tech MATCH (i)-[:GOVERNED_BY]->(r:Regulation) RETURN i.name AS incentive, i.value_eur_toe AS eur_per_toe, i.value_eur_mwh AS eur_per_mwh, i.eligible_tech AS eligible_technologies`
- **hint_reason**: fact(s) exist in KG via ['GENERIC_ACTOR'] but top-3 routed to ['P4_incentives_it_whr', 'P1_eed_art26_threshold', 'ALL_REGULATORY_ARTICLES']; routing or template fix, not KG enrichment.
- **decision**: _______   **notes**: _______

### OOD31  [medium]  -> hint: **ambiguous** (med)

- **query**: If a manufacturing plant in Italy implements ISO 50001 and identifies data center waste heat as a significant energy opportunity, what is the regulatory pathway from that internal finding to obtaining Certificati Bianchi, and which national body validates the energy savings claim?
- **template_top3** (semantic conf): P1_eed_art26_threshold(0.759), ISO50001_PROCESS_COMPLIANCE(0.755), ALL_REGULATORY_ARTICLES(0.755)
- **KG_entities_retrieved**: Country:IT  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P1_eed_art26_threshold`): `MATCH (r:Regulation {id: 'EED-2023-1791'})-[:CONTAINS]->(a:RegulatoryArticle {id: 'EED-ART-26'}) RETURN a.title AS article, a.obligation_type AS obligation, a.threshold_value AS threshold_mw, a.threshold_unit AS unit, a.summary AS summary`
- **hint_reason**: compound multi-hop; missing/concept: ['procedural steps / legal sequence']; retrievable-unrouted via ['GENERIC_ACTOR', 'ISO50001_ARTICLES']; needs multi-template composition plus a semantic decision.
- **decision**: _______   **notes**: _______

### OOD34  [medium]  -> hint: **kg_missing** (high)

- **query**: Under EU EED 2023/1791, which article requires member states to ensure that energy audits lead to actionable recommendations, and how does that requirement create a duty on an Italian data center operator to consider waste heat supply as a recommended measure when a certified auditor identifies the opportunity?
- **template_top3** (semantic conf): ALL_REGULATORY_ARTICLES(0.821), P1_eed_art26_threshold(0.782), ISO50001_ARTICLES(0.728)
- **KG_entities_retrieved**: RegulatoryArticle:EED-ART-23, RegulatoryArticle:EED-ART-24, RegulatoryArticle:EED-ART-26, RegulatoryArticle:DEL-REG-ART-2, Regulation:EED-2023-1791  | top1_rows=4  | LCR=1.0
- **cypher_top1** (`ALL_REGULATORY_ARTICLES`): `MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle) RETURN r.id AS regulation, r.short_name AS reg_name, a.id AS article_id, a.article_number AS art_no, a.title AS title, a.obligation_type AS obligation, a.threshold_value AS threshold_m`
- **hint_reason**: referenced entity absent from KG: EED Art.12 energy audit; propose KG enrichment.
- **decision**: _______   **notes**: _______

### OOD36  [hard]  -> hint: **ambiguous** (med)

- **query**: A data center operator in Italy wants to use a digital twin built to ISO 23247 to continuously optimize waste heat delivery to a neighboring factory and use the resulting energy savings data as evidence for Certificati Bianchi claims. Trace the full regulatory and standards chain: how does ISO 23247 generate the performance data, how does ISO 50001 provide the management framework for validating savings, how does the EU EED delegated regulation on data center KPIs align with those savings metrics, and which Italian body ultimately certifies the white-certificate claim?
- **template_top3** (semantic conf): ALL_REGULATORY_ARTICLES(0.781), P1_eed_art26_threshold(0.771), ISO50001_ARTICLES(0.737)
- **KG_entities_retrieved**: RegulatoryArticle:EED-ART-23, RegulatoryArticle:EED-ART-24, RegulatoryArticle:EED-ART-26, RegulatoryArticle:DEL-REG-ART-2, Regulation:EED-2023-1791  | top1_rows=4  | LCR=1.0
- **cypher_top1** (`ALL_REGULATORY_ARTICLES`): `MATCH (r:Regulation)-[:CONTAINS]->(a:RegulatoryArticle) RETURN r.id AS regulation, r.short_name AS reg_name, a.id AS article_id, a.article_number AS art_no, a.title AS title, a.obligation_type AS obligation, a.threshold_value AS threshold_m`
- **hint_reason**: compound multi-hop; missing/concept: none; retrievable-unrouted via ['GENERIC_ACTOR', 'GENERIC_STANDARD']; needs multi-template composition plus a semantic decision.
- **decision**: _______   **notes**: _______

### OOD37  [medium]  -> hint: **kg_missing** (med)

- **query**: In Denmark, if a comprehensive heating and cooling assessment carried out under EU EED Art. 23 identifies a hyperscale data center as a significant waste heat source, what is the sequence of legal steps under the Heat Supply Act that transforms that finding into a binding heat supply obligation on the data center operator?
- **template_top3** (semantic conf): P3_regulatory_screening_dk(0.848), P1_eed_art26_threshold(0.821), ALL_REGULATORY_ARTICLES(0.812)
- **KG_entities_retrieved**: DataCenter:DC-M, DataCenter:DC-L  | top1_rows=1  | LCR=1.0
- **cypher_top1** (`P3_regulatory_screening_dk`): `MATCH (c:Country {iso: 'DK'})-[:HAS_FRAMEWORK]->(pf:PolicyFramework) MATCH (pf)-[:CONTAINS]->(a:RegulatoryArticle) WHERE a.obligation_type = 'mandatory' MATCH (dc:DataCenter) WHERE dc.it_capacity_kw >= 1000 RETURN c.name AS country, pf.name`
- **hint_reason**: requested at workflow/attribute granularity not modelled: procedural steps / legal sequence.
- **decision**: _______   **notes**: _______
