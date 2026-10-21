# Delivery Checklist

Status snapshot: 21 October 2026. Checked items have implementation or execution evidence. A passing scripted test does not establish model competence, and a device-control task does not establish scientific acquisition fidelity.

## Native coverage correction — latest update

- [x] Remove the hard-coded clock/poll device whitelist; derive binding from native API signatures and asynchronous contracts.
- [x] Preserve all 2,855 public native tools across 194 registered devices.
- [x] Execute native clock/readback/replay checks on 191 devices; record the three native hosts with no clock API, without disabling their device tools.
- [x] Expose native poll/cancel for 149 devices with asynchronous declarations; verify one actual HX204 acquisition end to end. Do not interpret 149 as full acquisition acceptance.
- [x] Keep unacquired unrelated fields from blocking initialization; retain stale/missing-timestamp rejection for goal evidence.
- [x] Restore one missing same-revision washer data dependency. Current verification: 201 tests passed and 571 pinned files matched; no upstream runtime code changes.
- [x] Obtain Astra code/audit PASS and retain the earlier five offline integration failures and original model failure.
- [x] Complete the single new real-model balance rerun: supported goal, correct protocol, a legal wait candidate accepted, 20 requests and 63,106 tokens. Actual await execution is separately native-tested.

See the [coverage report](../reports/native-lifecycle-coverage.md). Earlier completed counts and reviews below refer to their original frozen sources.

## Current approved scope and implementation

This section supersedes the earlier audit's requested low-confidence escalation and scientific-demo script acceptance. The user selected direct legal argmax routing, requested diverse public instrument choices, deferred the scientific UI, and required **no external task-specific solution script, including TiO2**. The subsequent user instruction removes token-amount concerns for the authorized OpenRouter models; finite experiment/episode limits remain.

- [x] Default routing selects a legal maximum without a confidence threshold or deliberate review. Invalid model output/backend failure stops unresolved; legacy threshold behavior is explicitly named for historical compatibility only.
- [x] Expose all 194 registered devices to the first expert through a public directory and source-hashed detailed-contract cache. Search/describe is bounded; Jev chooses among the expert's described candidates. Selection cannot rewrite task predicates.
- [x] Bind one native `InstrumentLab` and reuse its public capability, validation, receipt, wait and evaluator interfaces. The new solver never imports reference tasks, scripted action sequences or Titania macros.
- [x] Run four diverse automatic-selection development cases: balance filter, IKA stirring, Binder timer and Nabertherm program. All selected an appropriate device; three met their goals, all four retained correct protocol outcomes. The balance stopped on a candidate outside the exposed menu, with no false success. Follow-up confirmed native time stepping works; our hard-coded bridge whitelist omitted this device, so the failure also exposes an adapter coverage gap.
- [x] Freeze an offline native-only acceptance pack: six cases repeated twice with explicitly offline scripted fixtures, four wrong-control negatives and 12 protocol checks; 28/28 checks passed. These doubles are test fixtures, not the live solver.
- [x] Add public-context lexical knowledge lookup, eight sourced/scoped passages, separate knowledge citations and no task-to-action recipe. Use identical native executor/oracle and lossless public-metadata encoding in paired arms.
- [x] Preserve the earlier two knowledge attempts, including a local reservation-budget stop. That censored attempt is not evidence that retrieval worsened planning.
- [x] Change authorized OpenRouter DeepSeek/Jev to provider/model-scoped record-only metering, preserve historical usage and isolate the remaining non-OpenRouter 5M budget. Astra reviewed this change before the four fixed new attempts.
- [x] Obtain final Astra code/report PASS, including independent reconciliation of 110 new requests, 70 raw artifact hashes and 83 frozen current source hashes.
- [x] Pass 193 offline project tests and verify all 570 pinned upstream files after the accounting update.
- [x] Complete and assess four new same-freeze knowledge on/off attempts: 4/4 goal and protocol success; same action sequences and model-call counts, with no demonstrated retrieval benefit. See the [current acceptance report](../reports/native-acceptance-and-knowledge.md) for final results rather than interpreting this work log as a benchmark claim.
- [ ] Establish that retrieved **materials** evidence improves a useful scientific decision. Generic instrument-fact retrieval is implemented; a held-out or causal benefit is not accepted.
- [ ] Replace the historical TiO2 macros with general native multi-instrument execution, shared sample/transfer semantics and an honest scientific oracle. The old TiO2 success is historical evidence, not acceptance of this new requirement.
- [ ] Final scientific presentation/UI: deferred by the user.
- [ ] Formal two-general-expert comparison, independent human error annotation and offline decision benchmark: still pending/deferred as recorded below.

Current code is uncommitted development work. The [report](../reports/native-acceptance-and-knowledge.md) distinguishes source-frozen real runs, later offline checks, model outcomes and remaining gaps.

## Earlier Astra alignment audit — baseline b46a09a

Reviewer: GPT-6 Astra. Baseline: development `b46a09a`, including the reviewed feature in `c059626`. This was a read-only source/evidence audit; no model requests, new trajectories or test-suite reruns were made for it. The latest recorded complete validation is 172 offline tests and 570 pinned native hashes. AI review does not substitute for independent human scientific review.

**Conclusion:** the bounded execution framework and contrasting materials demo are operational. Materials-knowledge use, complete uncertainty handling in the default single-instrument path, final scientific-demo packaging and formal comparative evaluation remain unaccepted. There is no defensible overall completion percentage across these differently sized deliverables.

### Requirements being compared

- [Original M0–M8 milestones](../plans/archive/milestones.md): a small materials troubleshooting demonstration with meaningful evidence, contrasting repairs, knowledge lookup and a frozen presentation.
- [Revised workflow proposal](../plans/workflow-proposal.md): the working architecture plus a controlled two-general-expert comparison and routing baseline.
- [Collaborator decision-mechanism proposal](../plans/decision-mechanisms-proposal.md): a new generated decision dataset, execution-derived labels and mechanism/calibration study. This is a separate research deliverable. Its [subsequent plan](../plans/decision-benchmark-review-and-plan.md) records the user's explicit deferral of the offline decision benchmark.

### Milestone acceptance

| Milestone | Audited status | Evidence or exact remaining acceptance |
|---|---|---|
| M0 — Run upstream | Minimum achieved | Pinned native runtimes, repeatable execution and [570-file validation](../evidence/single-instrument-code-review.md). |
| M1 — Expose tools | Achieved for supported subsets | Public contracts, typed invocation, identity/revision/receipt checks, deduplication and bounded time/poll bridges in the [single-instrument adapter](../../app/single_instrument.py). This does not accept all capabilities of all devices. |
| M2 — Choose the case | Minimum achieved | Paper-grounded [TiO2 scenario card](../demos/titania/scenario.md) with thermal-history and scan-coverage alternatives; native/fixture ownership is explicit. |
| M3 — Close the loop | Minimum achieved | Both contrasting branches and wrong-repair controls have [independent evaluation evidence](../evidence/titania-validation.md). Material outcomes/transfer remain authored fixtures, which the original demo scope allows. |
| M4 — Agent chooses tools | Minimum achieved | Real expert/router materials runs plus [goal-only Binder/furnace trajectories](../reports/single-instrument-demo-for-benchmark.md). The latest guard/citation changes have offline validation only; historical traces do not prove current-source live acceptance. |
| M5 — Materials retrieval | Partial; substantive gap | [Retrieval](../../app/retrieval.py) uses fixed role queries over protocol summaries; titania supplies fixed paper context. No task-conditioned materials lookup has been shown to change a selected check or correction. A lexical materials lookup is an allowed fallback; the Materials Encoder is not a prerequisite for minimum acceptance. |
| M6 — System 1 routing | Real integration achieved; full policy acceptance partial | Local OpenJev and explicit hosted TypeSafe Jev are live. However, [default single-instrument routing](../../app/single_instrument.py#L495) removes deliberate, so [low-confidence escalation](../../app/decision_backend.py#L57) is inactive there. Define and test the uncertainty policy or explicitly document this scope exception. No calibrated confidence or controlled routing-benefit claim is supported. |
| M7 — Visibility | Partial | Live drying UI and recordings exist. The titania scientific workflow is CLI-only; connect its sample, furnace/XRD evidence and decisions to a coherent view/replay. Integrated GLB is optional and is not the central blocker. |
| M8 — Verify and freeze | Partial | Regression, negative controls and retained failures exist. Freeze a current scientific demo, run its declared acceptance cases/baseline, package exact provenance and rehearse the live/replay presentation. The revised proposal's two-expert matrix and independent human error review are additional uncompleted requirements. |

### Current acceptance work, in priority order

- [ ] **P1 — Demonstrate materials evidence influencing one decision.** Add a small sourced material/process/characterization corpus (the original target is about 5–10 passages), construct a query from public task/evidence, return passage IDs/source sections, and retain an annotated trajectory showing why a retrieved fact changes a useful check or correction. Keep branch labels and evaluator truth out of retrieval. Report a declared no-retrieval control where practical; a pasted repair policy does not satisfy this item.
- [x] **Superseded by the user — Close the routing uncertainty policy.** Specify what a low-confidence/marginal single-instrument route should do while preserving bounded wait and execution authority. Verify that path with offline cases. Alternatively document a deliberate scope exception; do not claim the original escalation behavior in default mode.
- [ ] **P1 — Freeze and validate the current scientific demo.** Keep one versioned source/configuration/task manifest; cover contrasting process and measurement faults, acceptable and insufficient/conflicting-evidence controls, and wrong repairs. A six-case demo floor can reuse declared fixtures, but existing scripted drying tests do not themselves accept the current titania demo. Record all failures and include a simple matched baseline. A small explicitly budgeted live smoke run should check the latest guards; it is not a formal benchmark.
- [ ] **P2 — Package the scientific view and replay.** Show current sample, simulation time, processing history, acquisition result, cited evidence and selected operation in the same titania demo. Provide one start command and a clearly labeled saved replay; rehearse a five-minute run. Reuse native acquisition and existing event infrastructure.
- [ ] **Explanation quality — Separate correct outcomes from correct diagnosis.** Preserve unsupported claims in successful traces and label them separately. The historical thermal titania trajectory completed the goal despite incorrectly describing the earlier scan range; task success is not proof that every explanation was correct.
- [ ] **Later, after model selection — Complete the revised proposal's comparison.** Select a genuinely distinct second general expert. Two providers serving DeepSeek do not establish two independent expert configurations; Jev is a router/chooser, not the second general expert. Freeze matched inputs and run the declared 48 episodes, or label the 24-episode reduced pilot. First apply the shared token guard to the [historical matrix runner](../../bench/matrix.py), which currently constructs CompatibleExpert directly. Retain all attempts, repairs and provider failures.
- [ ] **Later research acceptance — Review and attribute failures independently.** Obtain the revised proposal's human review of at least six failure traces and report the actual comparison/error heatmap. Current AI review, development plots and conditional option probabilities do not establish human validation, causal routing benefit or calibration.

### Explicitly deferred or optional

- [ ] **Deferred by user:** the collaborator proposal's offline task generator, all-option outcome labels, acceptable-action sets, hidden-world identifiability checks, held-out splits and method/calibration comparison. Existing workflow traces are useful development evidence, not this released dataset.
- [ ] **Exploration complete; benchmark release pending:** [literature dataset feasibility](../reports/dataset-benchmark-feasibility.md) audited 76,975 source records and a narrow native furnace mapping. It did not produce a validated executable benchmark or chemistry ground truth. Source/license/operation semantics still gate task admission.
- [ ] **Optional expansion:** remaining nine catalog entries, more acquisition families, generic multi-device sample transfer/scheduling, Materials Encoder and GLB. None should displace the current materials-evidence and demo-acceptance gaps.

The 194-device suite establishes bounded control/readback coverage, including 116 power-only tasks; it is not 194 complete experiments. The two successful hosted-Jev demos establish two development outcomes, not held-out generalization. No new external-model budget is allocated by this checklist.

## Completed implementation and historical validation evidence

The counts below retain their historical snapshots. Use the audit above and the latest review for current acceptance status.

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

- [x] Add opt-in online operation selection between diagnostic candidates and procedure; retain independent expert routing and gate authority.
- [x] Add local OpenJev and budgeted DeepSeek action adapters, candidate/probability traces and browser demo modes.
- [x] Separate current aliquot processing evidence from historical scans; reject redundant valid-state reinspection in the titania profile.
- [x] Pass all 108 tests including selection/argument isolation, stale state, malformed candidates, budgets, UI/replay and legacy workflow coverage.

- [x] Pass both final real DeepSeek-candidate/local-OpenJev operation trajectories; retain earlier failures and the interrupted attempt with conservative accounting.
- [x] Record 450,497 conservatively charged tokens across 271 settled shared requests, including one full-charge unknown-usage request.

- [x] Pass Astra re-review after rejecting cloud browser expert configurations and separating chooser identities in summaries; pass all 111 release tests.

- [x] Execute declarative single-instrument tasks using discovered public native capabilities without task-specific reference scripts.
- [x] Add bounded model-selected direct waits through the ordinary execution gate and count router requests.
- [x] Support explicit hosted TypeSafe Jev routing/operation choice and the default OpenRouter DeepSeek expert through the shared 5M ledger.
- [x] Complete recorded Binder timer and furnace program tasks with fresh cited evidence; retain prior failures and uncontrolled-comparison limitations.
- [x] Publish detailed per-step tool/API/native-call reports and machine-readable provenance for benchmark design.
- [x] Classify empty/truncated expert responses before JSON repair and enforce shared verifier predicate/quality semantics.
- [x] Prevent duplicate verification of unchanged semantic evidence and require current citations for supported/contradicted reports.
- [x] Pass Astra code/report re-review, all 172 offline tests and all 570 pinned native hashes. Review fixes were not re-run with live models.
- [x] Retain the last settled shared ledger snapshot: 452 requests, 1,602,067 reported / 1,957,431 conservatively charged tokens; zero outstanding reservations. Earlier usage entries above are historical snapshots.

## Partial or still pending

- [ ] Accept the remaining nine catalog entries with an appropriate native task/adapter.
- [ ] Generalize meaningful acquisitions beyond the new furnace/XRD titania slice. The 194-device control suite still includes 116 power-only tasks.
- [ ] Generalize sample identity/material state and physical transfer semantics. Titania currently has adapter-owned logical aliquot lineage and an authored endpoint bridge.
- [ ] Generalize multi-device scheduling and asynchronous jobs. Titania has a common two-runtime clock and bounded polling within synchronous macros.
- [ ] Add device selection and tasks to the browser UI; the current UI is the drying demo.
- [x] Add task-conditioned lexical instrument-knowledge retrieval to the native profile; Materials Encoder integration and demonstrated material-decision benefit remain pending.
- [ ] Demonstrate that retrieved domain evidence changes a check or corrective decision.
- [ ] Select a second independent general expert and execute the frozen formal comparison.
- [ ] Establish held-out routing evaluation, calibration and evidence of routing benefit.
- [ ] Obtain independent human causal review of at least six failure trajectories.
- [ ] Freeze and rehearse the final scientific demonstration with a labeled replay backup.

## Milestone interpretation

M0-M4 have a historical working minimum; the current no-task-specific-script requirement is accepted for native single-instrument tasks, while the old multi-instrument TiO2 demo does not satisfy it. M6 routing is implemented with OpenJev in place of the originally proposed Laya. M7 has a basic live trace UI but no integrated GLB view. M5 remains partial, and M8 is not fully accepted as a scientific/model-comparison evaluation. Device expansion improves interface coverage without closing those research gaps.

See [Current progress](progress.md) for the milestone table and [Evaluation summary](../reports/evaluation-summary.md) for the result snapshot and limitations.

The offline decision-mechanism benchmark is explicitly deferred. Its generator, released task sets and comparative/calibration experiments are not part of the online operation-selection implementation.
