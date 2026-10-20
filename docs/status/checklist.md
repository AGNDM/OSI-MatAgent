# Delivery Checklist

Status snapshot: 21 October 2026. Checked items have implementation or execution evidence. A passing scripted test does not establish model competence, and a device-control task does not establish scientific acquisition fidelity.

## Completed

- [x] Create a reproducible Python/conda environment with locked dependencies.
- [x] Pin upstream ChemLab code and data revisions and record source hashes/licenses.
- [x] Execute native laboratory actions locally and separate agent/evaluator permissions.
- [x] Build the LangGraph diagnostic, procedure, gate, executor and verification loop.
- [x] Give specialists separate prompts, public input projections and output schemas.
- [x] Validate capability names, arguments, identity, revisions and request IDs.
- [x] Enforce finite action, model-call, time and graph-step budgets.
- [x] Keep hidden evaluation state outside model inputs and score outcomes independently.
- [x] Implement drying process-failure and measurement-failure branches, acceptable controls and insufficient-evidence cases.
- [x] Test wrong corrections, stale proposals, duplicate requests and false-success prevention.
- [x] Deploy local OpenJev Flash 9B Q4 routing with explicit fallback reporting.
- [x] Integrate an independently configured DeepSeek general expert backend.
- [x] Provide a local drying UI with live events, saved recordings and reset/action replay.
- [x] Build configuration, frozen-plan and explicit execution/resume interfaces for future model comparisons.
- [x] Inventory all 203 upstream catalog entries; establish native sessions for 198.
- [x] Register one verified control/readback task for each of 194 devices across 16 categories.
- [x] Run all 194 device tasks through the same LangGraph with scripted specialists: 194/194 passed.
- [x] Run a predeclared real DeepSeek sample across all 16 categories: 20/20 final outcomes passed, including output repairs.
- [x] Verify real DeepSeek specialists plus local OpenJev routing on two device tasks: 2/2 passed without routing fallback.
- [x] Retain failed device pilots: 0/3 initially and 2/3 in the second pilot.
- [x] Fix stale device observations by returning and budgeting fresh native readbacks after successful commands.
- [x] Add a persistent, concurrency-safe 5M-token budget for device API debugging.
- [x] Account for failed calls and format repairs; record 311,022 tokens across 206 device-debugging requests.
- [x] Verify 570 upstream file hashes and pass 73 project tests.
- [x] Provide English project documentation and a checked-in evaluation summary; ignore local Chinese notes and generated traces.
- [x] Ground a TiO2 thermal-treatment/XRD substep in an original published paper and write a one-page scenario card.
- [x] Audit native furnace program completion and XRD running/partial/completed acquisition behavior.
- [x] Implement two titania branches with shared simulation time and logical batch/aliquot identity.
- [x] Pass both scripted branches and reject both wrong-repair controls independently.
- [x] Pass both titania branches with real DeepSeek specialists and local OpenJev, including two evidence-ID repairs and zero router fallbacks.
- [x] Test scan timeout, stale/wrong/unavailable evidence, unnecessary synthesis, replay safety and host-truth isolation; pass all 80 tests.
- [x] Record cumulative shared API usage of 333,639 tokens across 220 settled calls; no additional OpenRouter requests.

- [x] Review failure paths before publication, reject stale aliquot reuse after failed preparation, preserve incomplete-usage reports and pass all 84 tests.

## Partial or still pending

- [ ] Accept the remaining nine catalog entries with an appropriate native task/adapter.
- [ ] Generalize meaningful acquisitions beyond the new furnace/XRD titania slice. The 194-device control suite still includes 116 power-only tasks.
- [ ] Generalize sample identity/material state and physical transfer semantics. Titania currently has adapter-owned logical aliquot lineage and an authored endpoint bridge.
- [ ] Generalize multi-device scheduling and asynchronous jobs. Titania has a common two-runtime clock and bounded polling within synchronous macros.
- [ ] Add device selection and tasks to the browser UI; the current UI is the drying demo.
- [ ] Replace fixed role-query protocol context with task-conditioned materials retrieval and integrate the Materials Encoder.
- [ ] Demonstrate that retrieved domain evidence changes a check or corrective decision.
- [ ] Select a second independent general expert and execute the frozen formal comparison.
- [ ] Establish held-out routing evaluation, calibration and evidence of routing benefit.
- [ ] Obtain independent human causal review of at least six failure trajectories.
- [ ] Freeze and rehearse the final scientific demonstration with a labeled replay backup.

## Milestone interpretation

M0-M4 have a working minimal implementation. M6 routing is implemented with OpenJev in place of the originally proposed Laya. M7 has a basic live trace UI but no integrated GLB view. M5 remains partial, and M8 is not fully accepted as a scientific/model-comparison evaluation. Device expansion improves interface coverage without closing those research gaps.

See [Current progress](progress.md) for the milestone table and [Evaluation summary](../reports/evaluation-summary.md) for the result snapshot and limitations.
