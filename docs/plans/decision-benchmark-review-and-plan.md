# Decision-mechanism benchmark: review and staged plan

Date: 21 October 2026. Status: proposed plan, not an adopted replacement for the existing workflow milestones. This request authorizes review and planning; no benchmark implementation, model experiment, or scope change is claimed here.


Scope update, 21 October 2026: the user approved only the online next-operation chooser and explicitly deferred the offline decision benchmark. The plan below is retained as a proposal, not an active benchmark implementation commitment. See [online operation decisions](../guides/operation-decision.md).

## Inputs and evidence boundary

The collaborator proposal is preserved verbatim in [decision-mechanisms-proposal.md](decision-mechanisms-proposal.md). Its SHA-256 matches the supplied `README-2.md`: `6e4554109cf4fbabb1ef9c6a5170ee51f30ca78bf190ce86827bac4d4bae6996`.

Review baseline: development commit `3406e11`, following failure-path fixes in `5e401a8` and the original titania demo in `ea292f3`. Current documentation records 84 passing tests; this planning exercise did not rerun tests. Historical real-model evidence retains its original executed source hashes.

Independent reviewer: Astra (`gpt-6-astra`), reviewing the proposal, current implementation and recorded evidence. The primary agent integrates that review with additional implementation checks. This is an AI technical review, not the independent human chemistry audit required by the proposal.

The shared API ledger was checked without sending requests: 333,639 charged tokens, 4,666,361 remaining under 5,000,000, and 220/220 requests settled. Planning consumes none of this external-model allowance.

## Alignment and reusable work

| Existing component | Reuse | What it does not establish |
|---|---|---|
| Native inventory and 194 device tasks | Device selection, contracts, public sessions, typed tool dispatch | A balanced decision dataset: 116 tasks are power-only; generator excludes acquisition and start/run operations |
| Gates, identity/revision checks and evaluator isolation | Execution safety, provenance and hidden-label boundary | A correct model decision: record raw answers before gate rejection, correction or fallback |
| Resettable fixtures and event traces | Basis for deterministic replay and candidate-option execution | Runtime snapshot/restore: `DeviceLab._snapshot` only captures observations |
| OpenJev deployment and token-probability reader | Local inference transport, option-token probability validation, latency traces | Operational-decision adapter or calibrated confidence: current prompt and route vocabulary are workflow-specific |
| DeepSeek transport and durable budget ledger | Existing authorized endpoint, metering, repairs and failure records | Decision-probability schema, sampling runner or API logprob support |
| Routing probe | Development checks and option-order sensitivity infrastructure | Independent held-out generalization: current dev/test largely reuse templates |
| Drying and titania demos | Sequential integration examples, visible evidence and abstention/failure checks | Native chemistry labels, generic decision-mechanism superiority, or an independent released test set |

Keep the LangGraph demos operational as a separate integration track. The benchmark core should operate on frozen decision records without requiring a multi-agent graph per single-shot example. Sequential/tool-enabled experiments may reuse LangGraph after the single-decision labels are validated.

## Astra review: prioritized findings

Astra recommends a small feasibility gate before adopting a new research mainline. The engineering supports the collaborator direction, but the current evidence primarily accepts workflow integration. Do not reinterpret historical control tests or authored titania endpoints as the new released test set.

The highest-priority revisions to the collaborator proposal are: (1) execution success is not automatically an optimal decision; (2) evaluator-only state is not automatically unidentifiable; (3) measurement, abstention and escalation need distinct semantics; (4) exact cause-ID citation does not establish internal rationale faithfulness; (5) decision probabilities are not presumed calibrated; and (6) comparing different models and different readouts does not isolate the readout mechanism.

Two additional implementation details matter. The measurement track must permit explicitly named measurement invocations and waits: the hood's `anemometer.measure` is an invoke capability, not just `describe/read`; calibration is a separate state-changing operation. The new error records should use independent axes for operational cause, observability, confidence, policy outcome and infrastructure failure, since these can overlap.

The proposal already correctly restricts the contract-rule reference to T1 and separates tool-access results. Preserve those choices, while validating any claim that the checker is exact against the runtime. For stochastic outcomes, deterministic seeded execution labels only the declared realization; expected-risk claims require an explicit distribution and repeated realizations.

Astra's preferred first deliverable is the specification plus a 24-family fume-hood audit, with no API calls. The primary agent adopts that ordering in the proposed plan below. The first existing-model smoke comparison is deliberately labelled a system comparison; a proper mechanism claim requires the later same-model readout experiment. No AI review supplies the proposal's required human expert audit.

## Design conditions before model evaluation

1. **Execution outcomes need a decision objective.** Successful invocation is not automatically the right operation. Define goal predicates, horizon, action costs, prohibited effects and accepted outcomes. Label by fresh-world execution against these conditions, with all candidate receipts preserved.
2. **Allow multiple valid choices.** Store acceptable option sets. For the first multiclass probability comparison, use questions with one justified mutually exclusive answer, such as a specified invocation's outcome. Keep multi-valid-action tasks separate until their probability scoring is defined; never select an arbitrary reference action as the only truth.
3. **Prove information insufficiency constructively.** Build paired worlds with identical agent-visible serialization and different relevant hidden outcomes. Define when measuring or abstaining is justified under the declared decision policy, and show that the permitted measurement resolves the distinction. Hidden visibility alone does not prove unknowability or imply a unique optimal action.
4. **Separate probabilities from policy.** Preserve raw model probabilities, raw choice, invalid-output mass where available, and final policy action. Workflow escalation thresholds (currently 0.6 probability / 0.15 margin), gate decisions, repairs and fallbacks are separate outcomes. Do not score a fallback as the model's prediction.
5. **Avoid model/mechanism confounding.** DeepSeek self-confidence versus OpenJev token probabilities is a comparison of two systems. To attribute differences to the readout mechanism, add a paired comparison using the same underlying model on identical records, such as verbal confidence versus sampling frequency. Dedicated decision-model comparisons remain explicitly model-plus-readout comparisons.
6. **Use metrics supported by the output.** A chosen-answer confidence permits top-label calibration, not a full multiclass Brier score. Require a valid complete probability vector for that score. For binary questions, define the event explicitly. Variable-option questions require format/option-count stratification; no automatic uniform vector may replace missing probabilities.
7. **Restrict interpretation of explanations and risk.** Citation agreement measures grounding, not internal causal faithfulness. Claimed decisive slots/preconditions should be checked by controlled interventions when supported. Keep unknown/multiple causes. Simulator risk tags describe the declared contract model and require review; they are not verified real-lab hazard labels.
8. **Split and report by independent groups.** Group base states, hidden-world pairs, paraphrases, option permutations and related trajectories before splitting. Hold out instrument domains and wording families for their respective claims. Cluster intervals by base scenario/instrument as justified, rather than treating repeated decisions as independent. Keep the T1 contract checker restricted to T1 and rank extra-tool-information tracks separately.

## Phased implementation

### P0 — Freeze the task and evaluation specification

Deliver `docs/benchmark/task-spec.md`, versioned task/result schemas, and a declared pilot protocol. Proposed task fields: task/instrument/domain IDs; native revision; state-recipe seed and clock schedule; tier; question and option IDs; agent-view hash; goal/horizon/cost policy; evaluation outcome for every option; acceptable answer set; source slot/precondition/error IDs; base-group/pair/template IDs; split; visibility and risk provenance.

Separate public task payloads from evaluator manifests. Public payloads omit hidden recipes, seeds that reveal latent parameters, labels, causes and branch IDs. Agent-visible contracts and allowed measurement tools must be fixed per track.

Acceptance: manually trace at least one T1, one paired T2 and one T3 record through the proposed labels; resolve multiple-answer and information-insufficiency semantics before generating a test set. No API calls are needed.

### P1 — One-instrument generator and label validation

Start with the existing Esco fume hood candidate (`CHEM-199`). Its pinned upstream example already exposes public sash control, anemometer measurement and monitor calibration, with evaluator-owned environment setup. These are audited candidate capabilities, not yet accepted new benchmark tasks.

Construct **24 development task families as a target**: approximately six T1, twelve T2 and six short T3 families. Each T2 family needs an observation-equivalent world pair where the question requires it; a pair, its paraphrases and permutations remain one statistical family. Counts are provisional: publish attainable coverage and rejected candidates rather than fabricating tiers to fill quotas. Balance outcome/option labels and include nontrivial failures and unsupported-operation exclusions. Use the T1 checker only as the visible-state reference.

For each candidate option, instantiate a fresh host and deterministically replay the full setup, clock and observation history. Do not deepcopy runtime objects/locks or treat graph checkpoints as world snapshots. Replay every accepted state/outcome at least three times, and check that changing candidate execution order cannot change labels.

Deliver generator CLI, reset/replay adapter, public tasks, private evaluation manifest, all-option receipts, rejection report, and group IDs.

Acceptance: all released labels reproduce; no cross-option contamination; metadata contains no answer shortcuts; supported native measurement completion is awaited; hidden-world pairs have equal public inputs and demonstrably different relevant outcomes. If a T2 distinction cannot be constructed, record it as a blocker and try another audited instrument rather than calling a visible-state question T2.

### P2 — Metrics and bounded existing-model smoke test

Implement a general decision interface independent of `LocalOpenJevBackend`'s route vocabulary. Reuse its validated probability extraction and transport; keep raw probabilities ahead of thresholds/fallbacks. Wrap DeepSeek decision requests in the existing shared token ledger, preserving unsuccessful and repaired attempts.

Initial systems: DeepSeek answer plus complete verbalized probability vector, local OpenJev option-token distribution, a T1-only contract checker, and a development-fitted majority/prior reference for appropriate formats. Confirm endpoint output features before adding any API logprob method. The existing expert-role ban on OpenJev is retained; the new adapter is for operational decisions, not repurposing it as a chemistry expert.

First run only **12–20 development decisions per model** spanning validated tiers. Use the same public state, question and answer options; log any readout-specific instructions. Test paired option orderings without counting permutations as independent new tasks. Reports distinguish raw first answers, repaired answers, abstentions, fallbacks and transport failures.

Prioritize correctness/balanced accuracy, well-defined Brier, selective accuracy/coverage, T2 measurement outcomes, failures/fallbacks, latency, tokens and hardware identity. Grounding agreement, ECE/reliability data and AUROC can follow once the core reports work. AUROC is unavailable when the evaluated group lacks both correct and incorrect predictions; do not invent a value. Tiny-sample calibration is diagnostic only.

Acceptance: metric fixtures verify known probability examples; exact public input hashes agree across systems; all raw failures are retained; per-track scores do not combine privileged tools with single-shot methods. An optional next pilot compares two readouts within the same model. Sampling uses at least ten independent draws per selected state, preserves all draws, and starts on a small development subset.

### P3 — Frozen multi-instrument pilot and human audit

After P1/P2 gates pass, choose approximately six instruments from at least three genuinely different capability domains based on validated nontrivial preconditions and identifiable hidden-state tasks. Existing inventory helps select candidates; do not assume every device supports every tier.

Publish a split manifest before running: base-group-disjoint development, validation and test sets, plus held-out wording variants. An exploratory target is **240 independent task families**: 160 source-domain families (80 train/development, 40 calibration, 40 in-domain test) and 80 families from one entirely held-out domain. Target roughly 25% T1, 50% T2 and 25% short T3 where native capabilities support it. Paraphrases and paired worlds do not increase the independent-family count. These are planning targets, not a sample-size justification or a promise of 99% reliable automation; one held-out domain supports only a limited transfer result.

Fit priors, thresholds, temperatures and any probes only on the permitted splits. Freeze task generation, translation, prompts and method settings before opening test results. Any existing debug/demo case or state used in prompt work remains development-only. Distinguish a domain-held-out evaluation from merely testing another instrument in a familiar domain.

Arrange a chemistry/operator review of approximately 60 stratified pilot task families as an early validity gate, including all critical hidden-state pairs in the first instrument. For a publication claim, obtain the collaborator proposal's 100–200-task independent expert audit, including hidden-state and rejected/ambiguous examples, and report disagreements and coverage. AI review cannot substitute for this step. Reviewer availability remains an external dependency.

Acceptance: no replay/visibility leaks; accepted label adjudications are recorded; paired system results include all attempts, group-aware uncertainty and exact model/runtime manifests. Report pilot scope, avoid rankings dominated by T1, and do not tune on test outcomes. Formal domain-generalization and rare-error claims require larger independent coverage.

### P4 — Add mechanisms and reconnect sequential demonstrations

Only after the dataset and core reports are accepted: expand same-model verbal/logprob/sampling comparisons, frozen-feature probes, embedding-temperature baselines and kNN; add independent domains, label-budget curves and group-aware statistical comparisons. Probe fitting and temperature fitting are explicitly permitted supervised components; base-model weights remain frozen. Downloading additional models or enrolling new API services is not a prerequisite for P0–P2.

Reuse the LangGraph drying/titania demos for a separately reported sequential/tool track with matched information and aggregate budgets. Native operation labels and authored materials outcomes retain separate provenance. More chemistry scenes, a new Materials Encoder, broad device expansion and additional UI features are lower priority until the benchmark's labeling and evaluation gates pass.

## Budget and stopping rules

The 5M ceiling remains shared across all existing and future debugging directories. Planning makes no external-model requests. Do not reset or fork the ledger to restart the allowance. No free OpenRouter batch evaluation.

Suggested internal allocation, subject to measured reservation bounds: at most 100,000 additional charged tokens for the initial P2 smoke test; at most 500,000 additional charged tokens cumulatively for P2/P3 pilots before reviewing cost and validity. These are stricter planning caps within the existing allowance, not authorization to exhaust the remaining 4,666,361 tokens. Projected cost includes repairs, failures, repeated draws and any LLM paraphrasing; prefer no-cost authored wording for the initial generator.

Precompute a conservative request bound and compare it with the phase cap and ledger before each batch. Pause the batch for unknown usage, authentication/quota errors, repeated transport failures, or evidence/label defects; fix offline before continuing. A budget reservation failure sends no request. Stop expanding the dataset if determinism, T2 identifiability or acceptable-label semantics remain unresolved.

A two-day engineering pilot can target a reduced P0–P2 slice only: day one specification/replay/generator, day two metrics and the bounded smoke test. The full eight-method study, expert audit, cross-domain evaluation and reliability claims are later deliverables with external dependencies.

## Immediate next work, if this direction is adopted

Implement the task schemas and the fume-hood replay/option-labeling adapter first. Produce ten manually inspectable labelled development examples, including at least one hidden-world pair and a multiple-valid-option case, before scaling toward the 24-family feasibility set. This provides a concrete go/no-go decision for the larger generator without committing to the entire research matrix.
