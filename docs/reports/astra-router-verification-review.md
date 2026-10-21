# Astra review: verification loops and empty expert responses

Later implementation: [hosted Jev, verifier consistency and output-budget validation](../evidence/hosted-jev-validation.md). This review remains a record of the earlier code; its semantic duplicate-evidence guard recommendation is still future work.

Recorded 21 October 2026. Reviewer: **GPT-6 Astra**. Scope: read-only analysis of the current single-instrument workflow, Binder/furnace traces, native public contracts, manifests and shared API ledger. No runtime code was changed and no project model/API requests or new trajectories were started for this review. Earlier reporting errors were corrected in documentation; raw traces are unchanged.

## Verdict

The furnace has two distinct problems: **the graph permits repeated verification of unchanged evidence**, then **an empty expert response and its empty repair terminate the run**. Fixing only response handling leaves the routing loop; fixing only the loop leaves a real response-compatibility issue. There is no evidence of an execution-gate failure, an HTTP outage in this run, or a false success claim.

Binder improved in call/token counts in its recorded run. The furnace failed, so an overall two-task efficiency improvement is not established. No additional decision model or furnace-specific action script is required by these findings.

## Independently checked evidence

| Measure | Binder historical → current | Furnace historical → current |
|---|---:|---:|
| Native goal met | Yes → Yes | Yes → No |
| Supported expert claim | Yes → Yes | No → No |
| Total model calls | 28 → 19 | 48 → 35 |
| Expert requests | 12 → 7 | 20 → 18 |
| Verification requests, including repairs | 4 → 1 | 6 → 12 |
| Reported cloud tokens | 116,938 → 83,263 | 64,963 → 70,239 |
| Wall time | 111.6 → 113.2 s | 175.3 → 303.0 s |

The historical furnace achieved its native goal but exhausted its call budget before final supported confirmation. Current trace hashes match their result files, and current manifest source hashes match the inspected app code.

**Correction:** seven furnace verification reports were schema-valid: **four unresolved and three contradicted**, all citing `obs-5`. Earlier prose incorrectly called all seven contradicted. Eight routed verification episodes produced eight initial requests and four repairs: seven parsed outputs and five empty responses. The three successful repairs returned contradicted; the final repair was empty. A parsed report is not automatically a semantically correct report.

## 1. The main failure is missing information-progress enforcement

The furnace executed inspection, power-on and configuration only. It never proposed/executed `thermal.start_program`, never awaited, and advanced native time by zero seconds. The action counter includes automatic readbacks. The last diagnostic response described configuration followed by starting, but control never returned to diagnostic planning after configuration. This does not show the operation chooser rejecting a correct start candidate: no new candidate-selection cycle occurred at that point.

The relevant code paths are:

- `app/graph.py:112–113`: a cached measurement is sufficient to offer verification.
- `app/graph.py:358–362`: a parsed verification result is stored.
- `app/graph.py:370–374`: either unresolved or contradicted can continue when the measurement envelope is valid.
- `app/graph.py:432`: continuation returns to the router without requiring new evidence.

There is no evidence-consumption guard, so the same measurement stays eligible. The router saw unmet phase/hold checks and later `verifier_disagreement=true`, but selected verification eight times. Choices were genuine model decisions without routing fallback. The first verify probability was about 0.766; later choices were marginal, about 0.518–0.564.

Concrete context/design weaknesses amplify this:

- `app/decision_backend.py:18` labels any measurement with no pending intent as a verify stage, even after that evidence was already assessed or shows unmet goals.
- Router context contains only a disagreement boolean, not the actual previous verdict, cited signature, missing-evidence list or repetition count.
- Diagnostic projection at `app/roles.py:27` also omits the last verification feedback.
- `app/single_instrument.py:479` removes deliberation, disabling the existing uncertainty escalation at `app/decision_backend.py:57–58` for these instrument runs.

These implementation conditions are verified. Which individual prompt feature caused the router's probability distribution remains a hypothesis; no causal ablation was run.

## 2. Quality and verdict semantics are not explicit enough

For `obs-5`, sample identity matches, revision is 2, and capture step 5 exceeds mutation step 4. The outer measurement is valid. The native phase and hold slots have quality **estimated**, with values idle and 0.

`evaluate()` at `app/single_instrument.py:64` accepts both valid and estimated native qualities. Under that policy, the phase/hold are evaluable mismatches. However, the instrument verifier prompt at `app/single_instrument.py:511` does not explicitly state this accepted-quality policy; the overridden device prompt at `app/device_roles.py:12` does.

The four unresolved outputs requested valid evidence of completed phase/full hold, despite acceptable public evidence of an actual mismatch. That conflicts with the intended evaluator policy. The omitted quality rule is a plausible contributor, but the model's internal cause is unknown.

Current schema validation checks report shape. The instrument validator mainly rejects supported while pending; it does not enforce the full distinction between supported, contradicted and unresolved. The final contradicted status was the last parsed report retained after later response failures. The terminal reason is role_error, not a successful final assessment.

## 3. Waiting is correctly unavailable before start

The public runtime separates configure from start. After configuration, phase remained idle and running was false. The lifecycle guard at `app/single_instrument.py:425–470` therefore correctly omitted direct wait options.

Waiting cannot replace a missing start command. More wall time, reads or repeated verification cannot advance this simulator. The workflow needed to return to ordinary diagnosis and operational selection over the existing public contracts.

Do not repair this by injecting a start command, relaxing idle-wait gating, or adding a furnace-specific action sequence.

## 4. Empty completions need their own response policy

All twelve furnace verification responses identify Novita in returned metadata. Five contain null content, finish reason length, 3,072 completion tokens and 3,072 reported reasoning tokens. Requests explicitly contain `reasoning.enabled=false`. Across furnace responses, 25,420 tokens are reported as reasoning. Binder's seven responses report zero reasoning tokens and come from other providers.

This proves a mismatch between requested output expectations and observed responses. It does not establish whether the gateway forwarded/transformed the parameter, the provider honored it, the model interpreted it differently, or usage classification is accurate. Provider, role and prompt differ together. Private reasoning text was not inspected or retained.

At `app/local_roles.py:61`, null content reaches JSON validation. The resulting validation error is treated as a format error by `app/graph.py:200–210`, triggering the normal repair allowance. That allowance is per role invocation, explaining four repairs despite format_repairs=1. The final initial request and repair were both empty, ending the run at 35 calls / 303.0 s. Neither the 48-call nor 360-second ceiling was reached.

## Prioritized repair plan

| Priority | Change | Intended effect |
|---|---|---|
| P0 | Consume successfully assessed semantic evidence; suppress repeat verification until meaningful evidence changes | Stop the demonstrated loop without choosing the next instrument action |
| P0 | Classify response-envelope failures before JSON/schema parsing; use explicit bounded recovery | Avoid treating output-budget exhaustion as an ordinary JSON typo |
| P1 | Supply actual last verification feedback to router and diagnostic; correct misleading stage labels | Make replanning aware of the prior result and unchanged evidence |
| P1 | Explicitly share valid/estimated quality policy and mismatch versus unavailable semantics | Align verifier instructions with the existing evaluator |
| P2 | Revisit bounded uncertainty escalation after the shared guard exists | Handle marginal routing choices without opening another loop |
| P2 | Share usage-validity checks between run summaries and the ledger | Avoid marking future partial usage as complete |

### Evidence guard design

Use task/predicate/constraint identity, sample identity, relevant native revision/time, values, units, qualities, freshness and pending-operation state to form a semantic signature. Observation IDs remain provenance, but a new ID alone must not reopen verification: redundant reads can mint new IDs without substantive evidence changes (`app/single_instrument.py:235`).

Record consumption only after a parsed, contract-valid assessment. Malformed/empty responses retain a separate bounded response-recovery allowance. Keep the consumption record separate from the last verification object, which diagnosis/execution currently clear. Enforce the same eligibility policy at router, deliberation and rules-fallback boundaries.

A general information-progress guard preserves real model planning, runtime contracts, gates and independent evaluation. It prevents this loop but cannot guarantee the next model-selected operation is correct. Blanket suppression of every verification until all predicates pass is broader than necessary and removes legitimate contradiction/unresolved assessments.

### Context and semantic validation

Include last status, cited evidence/signature, missing-evidence details and unchanged-evidence count in router and diagnostic projections. Keep the existing selected-intent context. Make stage describe eligible work rather than treating any measurement as a verification stage. Prompt changes alone are insufficient; current prompts already discourage intermediate verification.

Reuse `evaluate()` and the existing freshness rules. Explicitly accept valid and estimated under the present policy, while distinguishing stale, unavailable, wrong-sample, unit and path failures from evaluable false predicates. If semantic validation is added, retain the model's original verdict and report the inconsistency with bounded handling. Never silently rewrite a failed report into success.

### Response classification and provider diagnostics

Separate empty content, truncated content, malformed JSON, schema/evidence errors, explicit provider errors and transport failures. Retain sanitized provider/model identity, finish reason, usage and request identifiers; continue excluding private reasoning fields.

An empty length completion needs a distinct policy, not an automatic generic formatting retry. Any retry must have an explicit reason, a ceiling, normal accounting and an honest terminal result. Do not silently enable reasoning, enlarge limits, switch providers/models or synthesize an expert verdict from a failed response. Provider compatibility with disabled reasoning needs a separately scoped diagnostic before any future routing restriction is justified.

### Uncertainty and accounting

Restoring deliberation alone is insufficient: its current node uses the older summary and constructs its own route list. It first needs the same rich context and evidence eligibility guard, while respecting the one-token option limit and remaining budgets.

`app/instrument_cli.py:86` treats a nonempty usage object as complete, whereas `app/token_budget.py:94` validates nonnegative integer totals and arithmetic. Reuse one validity check. This discrepancy did not corrupt the reviewed runs, but can mislabel a future partial usage response.

## Accounting and review validation

Astra read the ledger without constructing or mutating it. The current runs contain 25 distinct expert reservation IDs: 83,263 Binder tokens plus 70,239 furnace tokens, totaling **153,502 reported/charged**. Empty responses are included. The shared ledger has **400 reservations and 400 settlements; 1,415,230 reported / 1,770,594 conservatively charged; 3,229,406 remaining**. There are eight historical unmetered requests, zero outstanding reservations and zero recorded accounting violations.

Reservation settlement precedes downstream JSON parsing; invalid output is still charged, and missing usage retains its full reservation. These are correct properties. Requested disabled reasoning is not proof of upstream compliance. The reservation also assumes the provider respects bounded completion usage and the framing/token bound; this run stayed within those bounds.

Astra attempted a narrow offline pytest run using /usr/bin/python3, which lacks pytest, so **no tests executed during the review**. No dependencies were installed. The earlier core 148-test and default-switch 24-test results remain prior evidence, not independently rerun tests. Source/trace hashes and accounting were independently checked. No project inference/API requests were made.

## Offline tests before another paid rerun

1. A router that repeatedly prefers verify cannot obtain multiple valid assessments of unchanged contradicted or unresolved evidence.
2. New observation ID alone remains suppressed; relevant changed state/time/quality, constraints or valid post-action evidence can reopen eligibility.
3. Suppression survives diagnosis and cannot be bypassed through deliberation or fallback.
4. Estimated-quality mismatches and genuinely unavailable/stale/identity/unit/path cases follow the same declared evaluator policy.
5. Empty length, whitespace, malformed JSON and provider errors have distinct classifications; retries stay bounded and each sent request is charged once.
6. Provider failures cannot synthesize supported claims or lose the provenance of the last valid report.
7. Idle/start/wait, stale authorization, action budget and pending-operation regressions continue to pass.

Use offline doubles and sanitized recorded payloads to reproduce these cases without more model calls.

## Acceptance for future original-task reruns

Freeze source/configuration, preserve the two public goals and budgets, use fresh result directories and the existing shared ledger. Count every attempt, repair and failure.

Both tasks should reach independently checked native goals and a supported expert claim citing fresh evidence, with correct protocol, no false success and no unresolved pending operation. Furnace must include a model-selected start and real native time progression; Binder must retain model-selected gated waits. There should be zero repeated valid verifications of semantically unchanged evidence.

Classify provider failures explicitly and reconcile every request settlement. Report calls/tokens separately from wall time; a failed run's lower call count is not an improvement. Do not attribute changes to waiting alone when providers, context and prompts also differ.

These are repair recommendations and acceptance criteria, **not implemented fixes or a claim that the furnace is repaired**.

Sources: [current comparison](../evidence/router-wait-openrouter-comparison.md), [machine evidence](../evidence/router-wait-openrouter-comparison.json), [earlier SiliconFlow attempts](../evidence/router-wait-siliconflow-comparison.md). Complete raw traces remain in ignored results directories.
