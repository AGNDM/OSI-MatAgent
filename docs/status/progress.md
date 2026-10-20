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
