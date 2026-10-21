# Reference-free single-instrument agent

This entry point runs a declarative task against one registered native instrument. It discovers the complete implemented agent capability menu from `session.describe`, retains native parameter schemas/preconditions/effects/execution metadata, and reuses the existing diagnostic, operation-selection, procedure, gate, execution and verification graph. It does not load generated device task files, `reference_probe`, reference steps or task-selected tool menus.

## Run

```bash
python -m app.instrument_cli \
  --task cases/instrument_tasks/binder-timer.json \
  --output results/my-binder-attempt \
  --backend deepseek --router openrouter-jev --operation openrouter-jev
```

Use a new output directory for every attempt. This example uses the default OpenRouter expert with a 90-second request timeout. Start a fresh interactive login shell after changing credentials in shell startup settings; a previously running process may retain an old key. The two supported configurations use separate credentials and share the same ledger. DeepSeek now defaults to `configs/openrouter-deepseek-v4.1-flash.json`, model `deepseek/deepseek-v4.1-flash`, using `OPENROUTER_API_KEY`. This is a paid model and the key must have a nonzero spending allowance. The previous SiliconFlow configuration remains explicitly selectable with `--config`; it is not an automatic fallback. Both providers share the existing persistent 5M-token ledger, including conservative full reservations when usage is unavailable. The separate free-only OpenRouter debug transport is unchanged. The example selects hosted TypeSafe Jev 1.14 through OpenRouter for both routing and operation choice. This is a different model from local OpenJev Flash 9B. For local decisions, use `--router openjev --operation openjev`; the local service must be available on loopback port 8792 and the CLI checks readiness before sending expert requests. Hosted decisions require no local model service. CLI decision defaults remain the previous local mode; the example explicitly selects the hosted mode. Context size must accommodate public state and candidate descriptions. The CLI has no scripted backend or silent scripted fallback. `--backend local` requires an explicitly configured local general expert. `--operation disabled` uses the diagnostic intent directly and remains explicitly recorded as a different mode.

## Task format

A task supplies an instrument ID, natural-language objective, typed public-state predicates and optional observed-state constraints, readable fields and simulation limits. For example:

```json
{
  "task_id": "oven-timer",
  "device": "binder_fd056",
  "objective": "Make a one-minute native timer expire and verify it.",
  "predicates": [
    {"state_id": "timer.state", "path": ["expired"], "value": true, "unit": "1"},
    {"state_id": "timer.state", "path": ["set_s"], "value": 60, "unit": "1"}
  ],
  "max_simulated_seconds": 300,
  "max_wait_seconds": 120
}
```

There is no supplied action sequence. Operators are `eq`, `ge` and `le`; `path` addresses nested dictionary/list values. The predicate unit is the native **state slot** unit (the timer object itself has unit `1`), not an invented unit for an object member. Missing paths, wrong types, unavailable quality, non-finite values and mismatched units fail evaluation explicitly. State IDs must be declared public/readable. Existing registry environment initialization is retained and recorded separately; no goal-specific fixture or material-loading solution script is added.

## Execution semantics

- Native operations are namespaced as `<instrument-slug>::<native-capability>`. Optional arguments remain optional. Model output is an action proposal, never executable Python.
- `workflow.inspect` and `workflow.readback` read public state without advancing time. Both produce observation IDs. Initial goal satisfaction can be valid without a forced write.
- `workflow.await(duration_s)` advances only an audited native clock bridge in bounded chunks, records steps and supported polls, and obtains a fresh native readback. A synchronously accepted start command is not evidence that a physical program has completed.
- Audited delta-time bridges currently cover Binder FD056, IKA RCT digital and Nabertherm L9. Quantos Q2 has a separately audited explicit-world-time/poll/cancel interface. Other instruments still expose public commands but do not receive an invented clock or lifecycle adapter.
- Quantos asynchronous responses use the outer native status. A callable `poll` method alone does not establish support: Binder's placeholder `poll` is not used. Quantos host-loaded material/geometry fixtures are not supplied by this entry point, so missing prerequisites may prevent a real dosing task.
- Pending operation IDs stay visible. New native commands are blocked while pending; supported read, await, poll, cancellation and honest stop remain available. Unknown polling, missing operation IDs and identity mismatches terminate unresolved while retaining uncertainty.
- `workflow.stop(reason)` exits unresolved without pretending to cancel an active operation.
- No generic scene acknowledgement callback is installed. Native requirements for an actual scene adapter remain enforced; unavailable apparatus motion must fail honestly. The example furnace starts with its native closed-door initial state.

The gate enforces capability/schema, scope identity, revision, selected intent and request replay. Reusing a request ID with different content is rejected. Exact replay does not advance time or invoke again. Evidence is invalidated before native mutations, time advancement and cancellation, including uncertain/partial failures. Native response instance, clock, revision and evidence timestamps are checked, so a new adapter observation number cannot make an old native read fresh.

## Outcomes and budgets

`goal_met` and `task_success` are independent public-state outcomes. `agent_claim_supported`, `evidence_cited` and `false_success` separately describe the expert's report. `protocol_correct` tracks known execution/receipt violations; it is not a measure of optimal scientific planning. A failed goal with correct protocol handling is a valid trajectory. Protocol assertions are supported by trace receipts and adversarial tests, not a claim of physical laboratory safety.

`lifecycle_resolved` and `pending_operation` identify incomplete operations. Supported completion requires current native and cached evidence, satisfied predicates/constraints and no pending operation. Constraints are checked at observed states and native await chunks, not continuously between simulator integration points; they are not physical safety guarantees.

Model-call limits include expert, operation-selector and model-router attempts in this opt-in path, including failed attempts. Legacy workflow counters remain unchanged. Action budgets count agent proposals plus fresh readback; internal clock/read/poll counts are separately recorded as `native_io_counts`. Episode deadlines, simulation limits, action/model budgets and graph recursion limit bound execution. Initialization and graph failures retain structured unresolved reports.

Every attempt saves `manifest.json`, the discovered public contract, full model/native `trace.jsonl` and `result.json` under ignored `results/`. Model-generated actions, source hashes, fixture provenance, native responses, independent predicates, model claims and API accounting remain inspectable. These tasks are development integration checks, not the deferred offline benchmark or validated material synthesis.

## Context-aware direct waiting

The OpenJev router receives the objective, independent goal checks, current public readbacks, pending-operation identity, recent actions/errors and remaining budgets. It can choose short (up to 30 s) or long (up to 120 s) waits alongside ordinary expert routes. Each offer is also capped by the task's per-wait and total simulation limits. Direct waiting bypasses diagnostic/procedure/operation-selection model calls, but still passes the ordinary action gate and native executor, advances the audited clock in chunks and produces fresh evidence.

A direct wait is offered only with fresh public evidence and either a supported pending operation or a recognized running process in the goal's state family. Idle, stopped, failed or paused process markers suppress the shortcut; an unrelated running fan cannot enable it. Unknown lifecycle semantics retain the normal expert path. No wait is offered with an outstanding intent, satisfied goals, violated observed constraints or an execution error. `--no-router-wait` restores the previous routing path for comparisons.

This guard was added after a real debugging attempt wasted simulation time waiting before starting the timer. The router additionally receives the actual selected intent and whether its revision is current, rather than only an intent-present flag. New errors or stale selections can still trigger replanning. The current implementation passes 172 regression tests. The latest hosted Jev Binder and furnace runs both completed with one final verification each, in 19/31 model calls and 23.0/27.9 seconds. The earlier local-router failure remains recorded. See the [current comparison](../evidence/hosted-jev-validation.md).

## Verification quality and expert output

Both valid and estimated native state qualities are accepted, matching the existing public predicate evaluator. Evaluable false conditions require contradicted, unavailable/stale/identity-invalid evidence or pending work requires unresolved, and supported requires every declared predicate and constraint. The instrument role validator rejects inconsistent model verdicts without rewriting them. A known false conjunct establishes contradiction even if another field is unavailable.

The default expert completion ceiling is now 6,144 tokens (`--max-output-tokens`, hard maximum 8,192). This is separate from the unchanged shared 5M API allowance. Empty/whitespace output, missing choices/provider errors and length-truncated responses have explicit classifications; empty/truncated responses do not trigger the generic JSON-format repair loop. Malformed JSON/schema/evidence validation retains the existing bounded repair allowance. Private reasoning fields remain excluded from traces.

Hosted decision requests use the same durable ledger, normalized from Decision API input/output usage. Every request reserves conservatively before sending and settles on failure as well as success; unknown usage retains the entire reservation. The CLI's total cloud usage now includes hosted router and operational decisions, so compare expert-only and combined tokens separately.

The single-instrument adapter now prevents a second contract-valid verification of unchanged semantic evidence. Its persistent signature includes task/scope, native values/units/quality/clock, pending state and freshness; new observation IDs or revision counters alone do not reopen verification. Routing, deliberation, fallback and verification entry enforce this policy. Only valid assessments consume evidence, and their concrete feedback remains available for later diagnosis. Supported and contradicted reports must cite the current measurement ID; inconsistencies use the bounded repair path. Seven new offline regression cases and Astra re-review cover these fixes. The successful live demonstrations predate them; no model-only guarantee or productive-planning guarantee on unseen tasks is implied. See the [code review](../evidence/single-instrument-code-review.md).
