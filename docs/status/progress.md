# Progress Against the Milestone Plan

Snapshot: 21 October 2026. The reference plan is [the original milestone document](../plans/archive/milestones.md). Implementation status and research acceptance are tracked separately.

| Milestone | Status | Evidence and remaining work |
|---|---|---|
| M0: Run upstream | Minimal scope complete | Pinned native runtimes execute locally; 570 upstream files have verified hashes. |
| M1: Expose tools | Complete for accepted task subsets | Typed actions, native schemas, identity/revision checks, request deduplication and logged rejections. |
| M2: Choose the case | Paper-grounded vertical slice implemented | TiO2 thermal-treatment/XRD scenario card and two distinguishable branches; qualitative phase analysis on synthetic reference-derived acquisitions. Chemistry remains authored. |
| M3: Smallest loop | Complete | Drying and titania contrasting repairs, wrong-repair controls, fresh evidence and independent scoring. |
| M4: Agent decisions | Minimum demonstrated | Real DeepSeek experts completed both drying and titania repair branches; titania used local OpenJev with zero routing fallback. Model stability is not established by a few development cases. |
| M5: Materials retrieval | Partial | Fixed role-query lexical protocol context exists. No task-conditioned materials retrieval, Materials Encoder integration or measured retrieval benefit. |
| M6: System 1 routing | Implemented with model substitution | Real local OpenJev replaces the planned Laya integration. Probabilities are uncalibrated; routing advantage is unproven. |
| M7: Visibility | Basic implementation | Drying UI shows live state, actions, evidence, routing, recordings and replay. No integrated GLB view or multi-device UI. |
| M8: Verify and freeze | Partial | Regression and development runs exist. Formal two-expert comparison, independent human failure annotation and final scientific demo acceptance remain pending. |

## Device expansion

The catalog contains 203 entries. Native sessions were established for 198; 194 have verified, resettable control/readback tasks spanning 16 categories. All 194 tasks pass the existing LangGraph using scripted specialist test doubles. This is interface coverage, not full instrument functionality: 116 tasks test power control and 78 test other controls/readbacks.

A predeclared 20-device sample with real DeepSeek specialists passed 20/20 final outcomes, with nine role-output validation/repair events. Two additional runs combining DeepSeek with local OpenJev passed 2/2 without routing fallback. Earlier device pilots passed 0/3 and 2/3 and remain recorded. Successful commands now return a fresh public native readback; writes and reads are separately counted against the action budget.

All device debugging consumed 311,022 tokens across 206 requests, including failures and repairs. There were no OpenRouter requests. The shared device budget ceiling is 5,000,000 tokens. Unknown-usage and outstanding reservations were both zero at the recorded snapshot.

## Earlier drying evidence

The explicitly authorized three-case SiliconFlow drying pilot passed 2/3, consuming 42,802 tokens. Process-setting and pan-offset repairs succeeded; an already-acceptable case failed after repeated verification outputs contradicted the supplied public mass criterion. These are development attempts, not the formal model comparison.

Scripted drying regression passed 16/16 attempts across eight test fixtures and two repetitions. Previous exploratory use of the decision-tuned OpenJev model as an expert is historical evidence, not the configured general-expert backend. Old free-model batch plans were retired.

## Validation and next acceptance conditions

The current project passes 111 tests. The upstream cache verifies against 570 pinned hashes. Credentials were absent from the device artifact scan, and all 206 returned model identities matched the configured DeepSeek model ID; no immutable provider weight revision was available.

Next acceptance work is meaningful acquisition tasks, shared multi-device sample/time semantics, scientific evidence retrieval, a second expert, a frozen comparison and independent human error review. The current results do not establish model-routing superiority, scientific fidelity or general materials-research capability.

See [Delivery checklist](checklist.md), [Device workflow](../guides/device-workflow.md) and [the committed evaluation summary](../reports/evaluation-summary.md). Detailed local traces remain under the ignored `results/` directory.

## Paper-grounded acquisition slice (21 October 2026)

The [titania scenario card](../demos/titania/scenario.md) follows the thermal-treatment and qualitative diffraction substep of Catauro et al. (2018), DOI 10.3390/ma11122364. Both branches run through the existing graph. Furnace programs, cooldown, XRD scan configuration/start/await/read and a shared simulation clock execute against unchanged native runtimes. Logical batch/aliquot lineage and finite chemistry endpoints are adapter-authored, not physical transfer or a calibrated reaction model.

Scripted branches passed 2/2 and both wrong repairs were independently rejected. Real DeepSeek specialists plus local OpenJev passed 2/2 with no routing fallback. Each real run needed one evidence-ID format repair. The 14 cloud calls consumed 22,617 tokens; shared cumulative usage is 333,639 across 220 settled calls, leaving 4,666,361 under the 5M ceiling. No unknown usage or outstanding reservations remained.

This advances M2–M4 for the preferred materials direction. M5 remains partial: fixed curated paper/reference context does not establish retrieval benefit. M7 remains the drying UI; titania currently runs in the CLI. M8 still requires a second expert, frozen comparisons and independent scientific review. See [execution evidence](../evidence/titania-validation.md).

## Pre-push review

A focused review fixed stale aliquot phase reuse after failed preparation and missing-usage report handling. All 84 tests pass. The prior real-model run remains historical evidence with its original hashes; no additional API calls were made. See [review findings and validation](../evidence/pre-push-review.md).

## Online next-operation selection

At the user's request, the workflow adds an opt-in operation-selection node while the offline decision benchmark is deferred. Diagnosis proposes typed candidates; a separately configured action chooser selects one; procedure and the deterministic gate must preserve that exact selection. The expert router remains independent. OpenJev uses the existing local service, and a budgeted DeepSeek action adapter is also available.

The local drying UI includes scripted regression and local-model operation modes with a separate candidate/probability panel. Public candidate/state projection, stale-state checks, probability validation, model/action/time budgets and failed handoffs have regression coverage. Current aliquot thermal evidence is explicitly separated from historical scans. The complete suite passes 108 tests; pinned native files are unchanged. See [operation guide](../guides/operation-decision.md) and [recorded integration evidence](../evidence/operation-decision-validation.md).

Final online integration reruns passed both titania branches with real DeepSeek candidate generation and local OpenJev routing/action selection, with no format repairs or router fallback in the final pair. Earlier failed pilots and an interrupted request remain recorded. The successful pair reported 23,958 tokens; all new development attempts added 116,858 conservatively charged tokens. The shared ledger now reports 430,167 tokens and charges 450,497/5,000,000 across 271 settled requests; one unknown-usage request retains its full reservation. There are no outstanding requests or accounting violations. Task success does not establish perfect diagnosis: the thermal run includes an incorrect hypothesis about the earlier scan range, preserved in the evidence report.

## Astra release review

Astra approved the bounded online feature after a required browser cloud-provider restriction and an operation-summary provenance fix. The full post-fix suite passed 111 tests in 87.88 seconds. These release fixes used offline tests only; the original real-model execution evidence and its historical 108-test snapshot remain unchanged. See [review and acceptance](../evidence/operation-feature-review.md).

## Literature-operation dataset feasibility

The dataset exploration goal is complete and independently reviewed by Astra. A reproducible audit counted 76,975 records across pinned solid-state, sol-gel and solution releases. The licensed solution payload was checksum-matched to Figshare. Its 4,893 explicit-air heating entries passing the strict numeric screen are potential source material, not executable benchmark tasks. A selected 500 C / four-hour parameter fixture completed in the native empty furnace; gas flow, source duration semantics, ramp and geometry limitations are explicitly documented.

The recommendation is a small source-verified operation-conformance pilot, with source parameters separated from authored simulator state and labels determined through independent candidate rollouts. Full chemical synthesis outcomes and hidden-state measurement tasks need additional validated models. The offline benchmark remains unimplemented; no model superiority or calibrated probability claim is added. Research and review used no external model API calls. See the [English report](../reports/dataset-benchmark-feasibility.md) and [Astra acceptance](../evidence/dataset-feasibility-review.md).

## Reference-free single-instrument execution

The new single-instrument CLI accepts goal-only task JSON and discovers native public tools without loading reference probes, solution steps or task-specific tool menus. It reuses the existing LangGraph, native runtime sessions, local router/action chooser, general expert transport and shared API ledger. Audited clock/poll bridges, fresh native evidence checks and structured bounded failure reports separate goal satisfaction, expert claims and protocol correctness.

Astra approved the design and accepted the revised implementation and real-model traces. Binder's one-minute timer task completed with fresh cited evidence. Nabertherm's short program independently completed; its expert report honestly remains unresolved at the model-call limit. Failed pilots remain recorded. This adds native single-device execution capability, not a released benchmark or validated synthesis chemistry. See the [guide](../guides/single-instrument-agent.md) and [reviewed validation](../evidence/single-instrument-validation.md).

## 21 October 2026: direct router waiting and OpenRouter switch

Context-aware bounded router waiting is implemented with ordinary gate/executor checks and public active-process guards. The early idle-wait regression was preserved and corrected. The single-instrument CLI now defaults to OpenRouter DeepSeek V4.1 Flash using the existing shared token ledger. A real request was rejected because the API key spending limit is $0; corrected Binder/furnace efficiency comparisons remain pending. No improvement is claimed. See [validation](../evidence/router-wait-validation.md).

## 21 October 2026: SiliconFlow efficiency reruns

Four same-task attempts were retained. A preliminary Binder success reduced total model calls from 28 to 20 and verification calls from four to one, with 6.2% fewer reported cloud tokens; wall time increased from 111.6 s to 131.5 s. The initial furnace attempt exhausted 48 calls in repeated diagnosis (14 times with an existing selected intent), without starting the device. The router now sees the concrete selected capability/arguments and revision validity, with explicit current-intent stage instructions. Final furnace/Binder reruns progressed without that loop but stopped on a SiliconFlow timeout/HTTP 503 respectively; complete two-task improvement is not established. No further requests were sent against the repeated provider errors.

The final suite passed 148 tests; 570 native hashes matched. Four attempts used 284,887 reported / 343,258 conservatively charged tokens. Cumulative ledger: 1,261,691 reported / 1,600,221 charged of 5M; all 372 reservations settled. See [detailed comparison](../evidence/router-wait-siliconflow-comparison.md).

## 21 October 2026: default OpenRouter expert and original-task reruns

OpenRouter `deepseek/deepseek-v4.1-flash` is now the default real expert for the single-instrument, device and titania entry points. A fresh interactive login shell loads the updated key; no key is stored in the repository. The default-switch regressions passed 24 tests; the routing/wait implementation previously passed the full 148-test suite.

Binder completed with supported evidence: 28→19 total model calls, 4→1 verification calls, 116,938→83,263 cloud tokens, 111.6→113.2 s wall time. The furnace failed after power-on/configuration: OpenJev repeatedly verified the same obs-5, never started the program, and made 12 verification requests including four repairs. Seven schema-valid reports cited the same obs-5: four unresolved and three contradicted. The unresolved reports mislabeled observed unmet predicates as missing evidence; an earlier summary incorrectly called all seven contradicted. Five responses returned empty content with all 3,072 completion tokens recorded as reasoning despite disabled reasoning in the request. The final empty response and repair caused role_error at 35 total calls / 303.0 s. Thus overall improvement is not established. Anti-repeat verification gating and empty-response/provider handling remain next work, not implemented claims.

The two runs used 153,502 reported/charged cloud tokens. Shared budget: 1,415,230 reported / 1,770,594 conservatively charged of 5M; all 400 reservations settled, no in-flight requests or accounting violation. See the [complete comparison](../evidence/router-wait-openrouter-comparison.md).

## 21 October 2026: Astra failure analysis

Astra independently reviewed the code, current trace/source hashes and API accounting without new inference calls. It confirmed an unchanged-evidence verification loop followed by empty/truncated completions; wait was correctly unavailable before program start. The verification-status prose was corrected to four unresolved and three contradicted outputs, all obs-5. The omitted estimated-quality rule in the instrument verifier prompt is a plausible semantic contributor, not a proven model cause. Priorities: semantic evidence-consumption guard across all routing boundaries, distinct response-envelope error/recovery handling, concrete verification feedback in router/diagnostic context, and shared evaluator quality semantics. Uncertainty escalation and usage-validity consistency are secondary changes. These are recommendations only; no runtime code was changed. See the [Astra review and acceptance plan](../reports/astra-router-verification-review.md).

## 21 October 2026: hosted Jev, shared verification semantics and output ceilings

Implemented an explicit OpenRouter Decisions API adapter for pinned TypeSafe Jev 1.14, including routing and operational choice, native probabilities/model snapshots, fail-closed execution and conservative shared-ledger reservations. Two exact recorded furnace states that local OpenJev routed to verify were routed to diagnose by hosted Jev. This is a different model, not just a hosted copy of the local 9B weights.

The verifier now explicitly accepts valid/estimated native state and validates its supported/contradicted/unresolved output against the shared predicate/freshness policy without rewriting verdicts. The expert output ceiling is 6,144 (hard cap 8,192); length/empty responses are classified before JSON parsing and do not trigger generic format repair. Earlier empties were per-request output-limit exhaustion, not account quota errors. Both new runs reported zero reasoning tokens, so the independent effect of the larger ceiling is not established.

The same tasks both completed with supported evidence and correct protocol: Binder 19 calls, one verification, two waits, 23.0 s; furnace 31 calls, one verification, nine 30-second waits, 27.9 s. Furnace includes a model-selected start and 270 native simulated seconds. New combined cloud tokens are 107,764 Binder (83,363 expert + 24,401 Jev) and 75,386 furnace (33,015 expert + 42,371 Jev). Cloud decision tokens increase total API usage relative to some local-router baselines; speed is not a token-count claim.

Full suite: 165 passed in 80.80 s. Native hashes: 570 matched. Two state replays plus two trajectories used 186,837 reported/charged tokens; cumulative ledger 1,602,067 reported / 1,957,431 conservatively charged of 5M; all 452 requests settled. The semantic duplicate-evidence guard was absent from this frozen run source; it was added in the later offline-reviewed changes below. Model, prompt and ceiling changed together in these full runs. See [validation and limitations](../evidence/hosted-jev-validation.md).

## 21 October 2026 detailed demo documentation for benchmark development

Added an English report covering the exact successful Binder and furnace trajectories: all 18 executed proposals, selected/rejected candidates, exact native invoke argument dictionaries and request IDs, workflow versus native revisions, all nine furnace wait observations, model/API/native call-count definitions, final verifier/evaluator evidence, authored environment setup and runtime limits. A machine-readable companion retains complete per-step public snapshots, decision probabilities, 50 request usage/reservation records, task/source/artifact hashes and independent extraction checks. The report proposes benchmark task/oracle/trace fields and held-out evaluation requirements without treating these two demos as a released chemistry benchmark.

Offline extraction reconciled all 18 proposal schemas, 33 native clock chunks, 50 unique reservations, 183,150 trajectory tokens and eight raw artifact hashes. No new model/API requests or runtime code changes were made. See [the report](../reports/single-instrument-demo-for-benchmark.md) and [detailed evidence](../evidence/single-instrument-demo-details.json).


## 21 October 2026: Astra code review and commit acceptance

Astra initially reproduced two blockers: repeated contract-valid verification of unchanged evidence until the call limit, and supported completion without a current-measurement citation. Both are fixed. Persistent semantic evidence consumption now gates every single-instrument verification route and survives transient-state resets; the last valid assessment reaches diagnosis and routing. Supported/contradicted outputs require current evidence through bounded validation/repair, without rewriting model verdicts.

Astra re-review passed 61 focused tests plus additional adversarial checks and found no remaining serious defect. The parent complete suite passed 172 tests in 80.91 seconds; all 570 native hashes matched. Historical demo artifacts and source hashes remain unchanged and predate these offline fixes. No new API usage was incurred. See the [review, scope and remaining limitations](../evidence/single-instrument-code-review.md).
