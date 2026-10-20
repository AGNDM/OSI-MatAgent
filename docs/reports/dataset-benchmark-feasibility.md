# Literature operations as a simulator benchmark: feasibility report

Date: 21 October 2026. Runtime baseline: `eb97c04` on `development`. Scope: dataset research, deterministic extraction audit and one native smoke run. This does not implement the deferred offline decision benchmark or run a model evaluation. No external LLM API requests were used.

## Recommendation and the actual gap

**Proceed with a small literature-grounded operation-conformance pilot; do not describe it as a materials-synthesis-success benchmark.** Kononova-style corpora provide real source-linked operation descriptions and numerical conditions. They do not provide our initial simulator state, executable device bindings, competing actions, counterfactual outcomes or the correct next-action probabilities. These must be defined and independently validated in the simulator. An operation mentioned in a successful paper is not automatically the unique best next action in a different state.

The best immediate licensed source is the solution-synthesis corpus. Its uncompressed payload matches the published CC BY 4.0 Figshare checksum. The larger 2020 solid-state and sol-gel releases are useful for feasibility statistics, but their precise release-level reuse terms were not established; clear those terms or obtain the explicitly licensed original solid-state payload before releasing record-derived tasks from them.

Our missing component is primarily **task construction and a valid executable outcome oracle**, rather than another decision model. The online operation chooser now exists, and its candidate menu can later receive these tasks. Existing demo outcomes cannot be recycled as independent evaluation labels.

## Sources inspected and what exists

| Source | Pinned material inspected | What it contributes | Limit |
|---|---|---|---|
| Kononova et al. (2019), inorganic solid-state recipes | Author repository commit `fca0a994ddb36b37d39b803e8a21880aaecb1d71`, `solid-state_dataset_20200713.json.xz` | Target/precursors, DOI, extracted operations and some temperatures, times, atmosphere, mixing attributes | The paper's 19,488 original entries are a different snapshot from the audited 31,782-record 2020 file |
| Same author's sol-gel release | Same commit, `sol-gel_dataset_20200713.json.xz` | Wet processing plus thermal operations; 9,518 records | No complete executable material-flow model; release license remains unverified |
| Wang, Kononova et al. (2022), solution synthesis | Commit `66cd72653eadab253ec04ca86a4901eb5329c022`, `solution-synthesis_dataset_2021-8-5.json.zip`; Figshare 16583387.v4 | 35,675 procedures; operations, solvents, some quantities and source identifiers | Extracted text/attributes remain incomplete and potentially misassociated |
| Open Reaction Database (ORD) | Official schema; data repository commit `704870642e5c93a141e6666436bc14a9b5f880ed` README, tree and licenses | Structured inputs, setup, conditions, observations, workup sequence, outcomes and provenance | Schema inspection only; no corpus-level coverage estimate or record audit in this study |

Primary sources: [Kononova 2019](https://www.nature.com/articles/s41597-019-0224-1), [Wang 2022](https://www.nature.com/articles/s41597-022-01317-2), [author solid-state repository](https://github.com/CederGroupHub/text-mined-synthesis_public/tree/fca0a994ddb36b37d39b803e8a21880aaecb1d71), [author solution repository](https://github.com/CederGroupHub/text-mined-solution-synthesis_public/tree/66cd72653eadab253ec04ca86a4901eb5329c022).

The original solid-state paper reports 19,488 entries from 53,538 synthesis paragraphs. This audit does not substitute that number for an updated release. Repository README counts also differ by historical release. We counted each downloaded payload directly. The original Figshare v3 license metadata was retrieved, but its payload request encountered a stale signed redirect and was stopped; its checksum was not validated and it is not part of our empirical counts.

ORD should be a secondary candidate for a later organic/workup track, not an assumed drop-in inorganic procedure collection. Its schema describes reactions; its `workups` field orders workup actions, not necessarily every reaction operation. A field being supported by the schema does not establish that it is populated in real records. The pinned current data tree uses Parquet files through Git LFS; a small Git blob is a pointer, not a downloaded reaction file. See [official schema](https://docs.open-reaction-database.org/en/stable/schema.html) and [pinned repository](https://github.com/open-reaction-database/ord-data/tree/704870642e5c93a141e6666436bc14a9b5f880ed).

## Empirical audit: what the numbers mean

The audit scanned **76,975 records** across three actual payloads, containing **136,872 extracted `HeatingOperation` entries**. It did not run the original extraction models, infer missing parameters with an LLM or fetch all source articles.

For each heating entry, we require exactly one explicit temperature and time, each with one value and equal non-null minimum/maximum, and recognized units. Bounds, ranges, multiple values, absent values and unknown units are rejected. Next we screen against the furnace input range (0–1100 degrees C and 0–24 hours hold). Finally we count the subset explicitly marked `air`. This records gas identity only: it does not distinguish static air from flowing air, controlled flow rate or pressure. Atmosphere missing is not silently treated as air. These are numeric/metadata screens, **not runnable-task counts**: a matching operation may still require pressure, a different vessel, missing ramp details or source correction. Sub-ambient targets also illustrate that schema validity alone does not imply achievable completion.

| Audited corpus | Records | Heating entries | Explicit scalar T + time | Within numeric envelope | Also explicit air | Records with at least one explicit-air match |
|---|---:|---:|---:|---:|---:|---:|
| Solid-state 2020 | 31,782 | 61,037 | 32,398 | 20,593 | 5,892 | 4,950 |
| Sol-gel 2020 | 9,518 | 19,769 | 9,306 | 7,861 | 2,573 | 2,092 |
| Solution 2021 | 35,675 | 56,066 | 30,243 | 28,245 | 4,893 | 4,438 |

The explicit-air heating-entry fractions are approximately 9.7%, 13.0% and 8.7%, respectively. They are conservative metadata-screen fractions, not estimates of scientific reproducibility. We do not infer that the other entries are impossible to execute. Excluding ranges deliberately loses potentially useful tasks, while accepting a scalar still does not validate extraction accuracy.

Solution records have nonempty `quantities` in 30,800/35,675 cases; that means at least one extracted quantity, not complete stoichiometry, reagent volumes or transferable material state. The two other releases do not expose that same `quantities` field. Balanced reaction coefficients are not weighed reagent masses.

Paragraph content is truncated with `<...>` in 31,778/31,782 solid-state, 9,518/9,518 sol-gel and 35,675/35,675 solution records. Therefore these public payloads are not a substitute for reading the source method and supporting information. Access to full papers and permission to redistribute their text are separate questions.

### Small deterministic record inspection

For the verified solution corpus, a fixed hash ordering of row indices selects 12 records. Their row indices, DOI, canonical row hashes, operation-type order and normalized heating attributes are retained in the [audit JSON](../evidence/dataset-feasibility-audit.json). This is a reproducible descriptive spot check, not a full-paper gold annotation or a representative estimate with confidence intervals. Only 2 of these 12 records contain a heating entry passing the strict explicit-air numeric screen.

Concrete examples from the spot check:

| Zero-based row | Source DOI | Extracted situation | Audit treatment |
|---:|---|---|---|
| 801 | `10.1016/j.cej.2014.03.093` | 200 C / 8 h, atmosphere absent; another heat entry lacks numeric conditions | Do not invent air or fill the incomplete step |
| 3182 | `10.1016/j.apsusc.2017.01.237` | 550 C with air but missing time; a separate entry has 550 C / 4 h but no atmosphere | Do not merge adjacent attributes into a supposedly complete step without the source |
| 6459 | `10.1016/S0025-5408(01)00647-X` | One entry has 800 C / 24 h / air; other heating entries differ or lack attributes | One numeric-screen match does not make the whole procedure supported |
| 24579 | `10.1016/j.jcat.2017.12.020` | One entry has 350 C / 2 h / air | Potential thermal substep; still needs source and apparatus review |

Operation list order is preserved as extracted. It is not certified as a temporally complete procedure; overlapping operations, repeated treatments and target variants need review. The fixed sample was inspected at the extracted-record level; no claim is made that these source papers were read in full.

## Native feasibility check and device mapping

We separately chose the **first** solution heating entry meeting explicit-air numeric conditions, above ambient and at most 600 C / 10 hours, as a bounded execution smoke check. This purposeful selection is separate from the 12-record spot check and cannot estimate corpus success rates.

It is row 1, operation 6, target field `TiO2`, source DOI `10.1016/j.apcata.2012.04.027`: the extracted conditions are **500 C, four hours, air**. We instantiated a fresh native `NaberthermL9Runtime`, enabled power, closed its door, configured and started a one-segment program, then advanced host simulation time and polled public state until completion.

The retained source paragraph explicitly mentions **flowing air**, which this furnace cannot model or control. The metadata air token therefore does not establish atmosphere compatibility. The **authored additions** were a 25 C empty-furnace start, 0.15 K/s ramp, interpreting the generic extracted four-hour operation time as a post-target hold, virtual apparatus substitution, a no-op scene callback and 60-second observation cadence. No geometry or rendered scene was validated. The corpus operation does not justify these additions. The full source method was not independently checked, and its DOI page was not accessible in this audit; this is an extracted-parameter execution check, not a paper replication.

The native program reached public `program.phase=completed` and `program.hold_elapsed_s=14400` after **17,580 simulated seconds**. The final indicated temperature was 771.95 K because the runtime returns to its ambient target after completion and the polling tick includes cooling. That does not mean it failed to reach its programmed hold. The evidence retains commands, parameters, public final state and source hash. **There is no chemical outcome label and no evidence that TiO2 was synthesized.** No LangGraph expert or operation-selector model was run in this check.

| Literature operation | Native contract inspected | What can be assessed now | What is not established |
|---|---|---|---|
| Calcination/heating | Nabertherm `thermal.configure_program`, `thermal.start_program`; 1–4 segments, 273.15–1373.15 K, ramp 0.001–1 K/s, hold up to 86,400 s | Program configuration, door/power conditions, time evolution and public program completion | Gas control, sample reaction, yield, phase formation and universal ramp feasibility; native contract declares atmosphere/sample calcination unsupported |
| Drying | Binder FD056 `sample.place`, `temperature.set` (273.15–573.15 K), `timer.set`, `ramp.set`, host dynamics | Candidate for source-grounded drying/mass-control tasks after matching modeled materials and loading geometry | Arbitrary salts, mixtures, precursor decomposition and faithful source apparatus; only contract inspected here, no new drying trajectory executed |
| Stirring/mixing | IKA RCT `stirring.set_speed` (0–157.08 rad/s), `stirring.set`, `vessel.place` | Speed/control and modeled-vessel preparation candidates | A general `MixingOperation` may mean grinding, dissolving or addition; native vessel material enum is restricted and does not represent arbitrary inorganic slurries |
| Centrifugation/separation | Eppendorf 5910Ri `centrifuge.set_method`, `centrifuge.get_curve`, `sedimentation.estimate` | Method settings and constrained model-based sedimentation estimates | Generic purification, actual pellet/supernatant inventory transfer or recovery; do not equate a configured method with physical separation |
| Filtration, washing, transfer, hydrothermal synthesis, quenching | No end-to-end adapter was validated in this audit | Keep unmapped in this pilot | A word match or registry entry does not prove a compatible native experiment path |

Actual public input schemas for the four inspected devices are retained in the audit JSON. This is a focused contract check, not an audit of every simulator device. Existing 194 single-device tasks mostly exercise controls (116 power tasks); they are integration tests, not evidence that these literature operations all run end to end. Existing TiO2 furnace/XRD traces use authored endpoint chemistry and synthetic diffraction, so they cannot supply independent chemical-success labels.

## A defensible benchmark construction

Use three explicitly separated layers:

1. **Literature layer:** source release/version/hash, DOI, source-method locator, materials, operation graph/partial order, units, intervals, apparatus and extraction uncertainty. Keep unavailable values unavailable. Have a human/scientific reviewer check the actual method for every admitted seed.
2. **Simulator task layer:** pinned native contract/model, supported substep, initial state created by a fresh runtime and deterministic setup, authored assumptions, public observation projection, evaluator-only fields, a finite candidate set and a measurable objective. Source parameters and authored environment fields must remain distinct.
3. **Outcome layer:** execute every candidate from an independently recreated identical start, with bounded host clock/polling; save public receipts, final observations, costs and failure categories. Score completed objectives and forbidden side effects, rather than command acceptance or similarity to the recipe sentence.

A suggested task schema is: `source_ref`, `source_verified_fields`, `authored_fixture`, `runtime_hashes`, `public_observation`, `goal`, `candidate_actions`, `reset_recipe`, `rollout_limits`, `outcome_predicate`, `cost_rule`, `acceptable_set`, `label_provenance`, `split_group`. This is a design, not an implemented dataset format.

A thermal example can ask which action advances a specified source-derived temperature/hold objective when a native program is paused by an open door. Candidate actions could close the door, alter the target, stop the program or inspect state. The initial fault and candidate alternatives are **authored**, not reported failures in the paper. The oracle must run each action and apply a declared time/cost policy; inspecting, waiting or multiple equivalent actions may otherwise also be acceptable. Do not manufacture a unique best answer by calling every alternative a failure.

For the collaborator proposal, T1 visible-state tasks are feasible first. T3 next-action tasks are feasible only with an explicit operational objective and independently checked option outcomes. T2 hidden/noisy-state tasks are not supplied by missing literature fields: they require paired states with the same agent-visible evidence but different relevant truth, and a native measurement that actually resolves the difference. The current ideal furnace temperature readout does not automatically provide that experiment. Defer T2 until its latent-state and measurement model is justified.

Keep model-facing candidates independent of evaluator labels. Our current menu contains 1–4 proposed operations; benchmark construction must use fixed, audited menus shared across methods, or explicitly evaluate candidate generation as a separate component. Never expose the reference acceptable set, branch label or executed counterfactual outcomes in model input.

Report task completion, false-success rate, unsafe/forbidden side effects, steps, elapsed simulation time, tool calls and model cost separately. Return full candidate probability vectors for decision metrics. For a unique labelled action subset, use multiclass Brier/log loss and calibration diagnostics. For tasks with several acceptable actions, report acceptable-set mass and correctness under a declared policy; do not turn an arbitrary member or a uniform acceptable-set vector into an unclaimed gold probability distribution. A maximum-mass objective alone is not a strictly proper estimate of an unknown true action distribution. Abstention, measurement and escalation need distinct actions and costs.

A fair mechanism experiment later needs the same base model across different readout methods, plus separately labelled cross-model comparisons. The present OpenJev-versus-DeepSeek setup cannot by itself identify the benefit of a decision mechanism. Cause-ID agreement also cannot establish faithfulness of an internal explanation.

## Splits, dependence and release rules

Never split rows randomly. The audited solid-state, sol-gel and solution releases have respectively 23,434, 7,579 and 29,881 unique nonempty DOIs, with maxima of 64, 13 and 11 records per DOI. Repeated `(DOI, truncated excerpt)` groups contain 12,456, 3,002 and 7,071 records. Those are warning indicators, not perfect full-paragraph identities: excerpts can collide or omit distinctions.

Construct source groups across **all imported releases** using normalized DOI plus known source-paragraph identity when available. Keep target variants, repeated recipes, generated faults, sibling states, candidate permutations, paraphrases and complete trajectories from a source group together. Deduplicate cross-corpus overlap before partitioning. When source lineage is ambiguous, keep linked records together or exclude them. Add chemistry-family and device/domain holdouts where the sample supports them; report performance by group and use source-clustered uncertainty estimates. Public-corpus pretraining contamination cannot be excluded simply by holding out dataset rows.

License facts: Figshare [9722159.v3](https://doi.org/10.6084/m9.figshare.9722159.v3) and [16583387.v4](https://doi.org/10.6084/m9.figshare.16583387.v4) metadata state CC BY 4.0. Only the solution payload was checksum-matched in this study. Neither inspected Ceder repository tree contains a root license establishing a blanket code or updated-data license. Do not infer dataset license from article license or import extraction code without checking its terms. ORD data/metadata use CC-BY-SA-4.0, while its code is Apache-2.0; see its [pinned license explanation](https://github.com/open-reaction-database/ord-data/blob/704870642e5c93a141e6666436bc14a9b5f880ed/README.md). Simulator code/assets have separate upstream terms already documented in `vendor/chemlab/`; do not relabel third-party material as our own. Keep attribution, source versions and artifact-level terms with derived tasks, and avoid redistributing article text or entire corpora in the code repository.

## Minimal follow-up, with explicit acceptance gates

| Stage | Concrete output | Acceptance condition |
|---|---|---|
| 0. Source validation | Up to 20 distinct DOI groups from the licensed solution corpus; inspect methods/SI and admit only supported thermal substeps | Every admitted temperature/time/atmosphere and action ordering has a source locator; unresolved fields are excluded or explicitly authored; exact release terms retained |
| 1. Executable pilot | For each admitted group, at most three authored visible-state situations and 3–4 fixed candidates; fresh runtime per counterfactual | Repeated reset/rollout reproduces outcome and cost; no hidden-answer leakage; explicit valid-action sets; all failures classified; 180–240 option rollouts only if all 20 groups qualify |
| 2. Workflow evaluation interface | Feed fixed observations/menus through the existing LangGraph chooser, procedure, gate, native execution and verifier | Reference rule baseline passes; stale evidence and optimistic false-success checks fail appropriately; trace records exact inputs and outcomes |
| 3. Limited model comparison | A separately approved small pilot after the oracle is stable | Predeclared same-model comparison, grouped splits and metrics; existing shared API ledger enforced; no bulk free OpenRouter requests |
| Later | Validated drying/stirring/separation mappings, then justified T2 and broader scientific outcomes | Each added device has an actual execution path and correct material/measurement assumptions; chemistry outcomes require an independent validated model or experimental data |

Twenty source groups are an engineering pilot, not a statistical claim of model superiority. Stage 1 numbers are proposed bounds, not completed work. No trained model, calibrated probabilities, source-verified synthesis reproduction or released benchmark results are claimed.

## Reproduce and inspect

Run from the canonical repository using the existing environment:

```bash
python -B tools/audit_synthesis_datasets.py --fetch --native-smoke
```

The script uses standard-library parsing for data and the already-installed native runtime for the optional smoke check. It downloads only the three pinned corpora and two small Figshare metadata records, verifies the solution member MD5, keeps raw files in ignored `results/dataset-feasibility-v1/raw`, and writes compact derived evidence. It makes no model calls. The original solid-state Figshare payload is not required.

Artifacts: [audit JSON](../evidence/dataset-feasibility-audit.json), [source/version/license manifest](../evidence/dataset-feasibility-sources.json), [audit script](../../tools/audit_synthesis_datasets.py). Raw ZIP/XZ payloads, metadata responses and inspected author schemas remain local for inspection. The source manifest records their sizes and SHA256 hashes; it is not a request to publish full third-party data.

Validation completed: scalar conversion, ambiguous-bound/range, unknown-unit, temperature-envelope and atmosphere assertions passed; native empty-furnace smoke completed. The feature's preceding release suite remains 111 tests; this research did not modify runtime workflow code or rerun a model experiment. Independent Astra report review is recorded separately when complete.
