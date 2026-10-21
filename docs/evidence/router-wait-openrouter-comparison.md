# OpenRouter rerun of the original two instrument tasks

Implementation follow-up: [hosted Jev and verifier/output changes](hosted-jev-validation.md) records successful reruns of both tasks. Outcomes and budgets below remain historical.

Recorded 21 October 2026. OpenRouter DeepSeek V4.1 Flash is now the default real expert. OpenJev remains the local specialist router and operation chooser. Both tasks use unchanged public-state goals and the same action/model/wall budgets as their historical runs.

The result is mixed: **Binder completed with fewer calls/tokens and similar wall time; the furnace failed. A two-task efficiency improvement is not established.**

## Results

| Task | Goal / expert claim | Model calls | Expert calls | Verification calls | Cloud tokens | Wall time |
|---|---|---:|---:|---:|---:|---:|
| Binder timer | goal met; expert supported | 28 → 19 | 12 → 7 | 4 → 1 | 116,938 → 83,263 | 111.6 → 113.2 s |
| Furnace program | role_error | 48 → 35 | 20 → 18 | 6 → 12 | 64,963 → 70,239 | 175.3 → 303.0 s |

All counts include failed/repair attempts. Cloud tokens are provider-reported input plus output; local OpenJev tokens are outside the external API ledger. The historical furnace reached the native goal but did not produce a final supported claim before its 48-call budget. Failed tasks with fewer calls are never labelled an improvement.

## Why calls changed

The router receives public goal checks, current native state, recent actions, simulation and model budgets, and the concrete selected intent with revision validity. It can directly choose a bounded wait through the same gate/executor/readback path. Waiting skips diagnostic, operational-selection and procedure requests. Current selected intents are made explicit so routine preparation does not require renewed diagnosis. Final verification still uses the general expert.

- Binder timer: 2 direct waits; 0 diagnosis routes with an existing intent; 0 intermediate verification requests; 1 total verification requests.
- Furnace program: 0 direct waits; 0 diagnosis routes with an existing intent; 12 intermediate verification requests; 12 total verification requests.

Any intermediate verification is retained as overhead, not relabelled as final confirmation. The gate does not force goal success, and actual model probabilities still choose the router/operation options. No task-specific action program or reference task sequence was supplied.

## Furnace failure analysis

The furnace executed inspection, power-on and program configuration only. Its public state remained `program.phase=idle`, hold elapsed 0 s, with the requested segment configured. No `thermal.start_program` or await was executed. Native simulated time stayed at 0 s.

OpenJev repeatedly chose verification even though the goal checks showed the incomplete phase/hold and `verifier_disagreement=true`. The verifier produced seven schema-valid reports, all citing the same `obs-5`: **four `unresolved` and three `contradicted`**. The four unresolved reports incorrectly described evidence of the unmet phase/hold as missing evidence, despite fresh public values showing an actual mismatch. The three contradicted reports match the declared verification semantics. An earlier version incorrectly counted all seven as contradicted; the raw trace is unchanged. The native phase/hold slots are estimated, which evaluate() accepts, but the instrument-specific verifier prompt omits that explicit quality rule. This is a plausible contributor to the unresolved outputs, not a proven internal model cause. There were 12 verification requests including four repair requests, rather than 12 distinct evidence acquisitions. This is a workflow/router liveness problem: repeated checks had no new evidence and could not start the process. The direct-wait shortcut was correctly unavailable while the program was idle, so it could not resolve the preceding routing loop.

Five verification responses were also empty: returned metadata identified Novita, `finish_reason=length`, `content=null`, and all 3,072 completion tokens recorded as reasoning tokens, despite the outgoing request containing `reasoning.enabled=false`. Across all furnace responses, the provider reported 25,420 reasoning tokens. These facts demonstrate an output/parameter compatibility problem in this path; they do not establish why the upstream service behaved that way. Private reasoning text was not retained. Input/output usage, including empty completions, was fully reported and charged to the ledger.

The last initial verification request and its one permitted repair both returned empty content, so schema validation failed and the graph stopped with `role_error` after 35 total model calls and 303.0 s. It did not hit the 48-call or 360-second ceiling. No HTTP transport outage was required for this failure, and the verifier did not falsely claim success.

The next robustness work is to suppress repeat verification of the same observation/revision after a valid contradiction unless new evidence or an action intervenes, expose the actual previous verification result in router context, and explicitly classify empty reasoning-only responses before JSON parsing. Provider compatibility with disabled reasoning also needs checking. These changes are proposed here; they were not applied during these two frozen comparison runs.

## Exact model-generated operations

### Binder timer

| Step | Capability | Arguments |
|---:|---|---|
| 1 | `workflow.inspect` | `{}` |
| 2 | `binder_fd056::power.set` | `{"on": true}` |
| 3 | `binder_fd056::timer.set` | `{"days": 0, "hours": 0, "minutes": 1, "fan_after": "on"}` |
| 4 | `workflow.await` | `{"duration_s": 30.0}` |
| 5 | `workflow.await` | `{"duration_s": 30.0}` |

Goal met: `True`; report: `supported`; cited evidence: `obs-9`; protocol correct: `True`; false success: `False`. Simulated time: 60.0 s.

### Furnace program

| Step | Capability | Arguments |
|---:|---|---|
| 1 | `workflow.inspect` | `{}` |
| 2 | `nabertherm_l9::power.set` | `{"enabled": true}` |
| 3 | `nabertherm_l9::thermal.configure_program` | `{"segments": [{"temperature_K": 323.15, "ramp_rate_K_s": 0.1, "hold_s": 10}]}` |

Goal met: `False`; report: `contradicted`; cited evidence: `obs-5`; protocol correct: `True`; false success: `False`. Simulated time: 0.0 s.

## Defaults, verification and budget

The single-instrument runner, real-model device runner and titania runner select `configs/openrouter-deepseek-v4.1-flash.json` by default for DeepSeek. Scripted offline modes remain explicit/offline. SiliconFlow is available through an explicit configuration on runners exposing `--config`. The separate free-only OpenRouter adapter was not loosened. Actual credentials are loaded from a fresh interactive login shell because the old process environment retained the previous key; no key value is stored.

The routing/wait code passed the full 148-test suite before this provider-default change. The default/configuration change then passed 24 relevant device/titania/transport regressions. All 570 native hashes matched. Actual executed proposals were independently validated against discovered schemas, trace hashes checked, and inspected expert/router inputs contained no reference/fixture/private-state keys. Both new trajectories have identical current app source hashes. Full requests/responses, effective expert settings, local router deployment settings, contracts and traces remain under ignored results directories.

These two runs used **153,502 reported / 153,502 conservatively charged tokens**. OpenRouter reported a total cost of **$0.038716**. The shared ledger is **1,415,230 reported / 1,770,594 charged of 5,000,000**, with **3,229,406 remaining**. Settled requests: 400/400; in-flight/crashed: 0; accounting violation: False.

## Limits of this comparison

This compares historical SiliconFlow trajectories with the current OpenRouter workflow. The provider, richer routing context and selected-intent prompts changed together; historical furnace used a 2K local context while current runs use 8K. Network/provider latency, caching, token accounting and concurrent offline checks are uncontrolled. Counts and observed tool paths demonstrate these particular outcomes, but neither wall-time changes nor all token differences can be attributed solely to direct waiting. Two tasks do not establish a general success rate. The furnace is empty-furnace control, not material synthesis.

Earlier failed and partial runs remain documented in the [SiliconFlow comparison](router-wait-siliconflow-comparison.md) and [initial wait validation](router-wait-validation.md). See [machine-readable evidence](router-wait-openrouter-comparison.json) and the [usage guide](../guides/single-instrument-agent.md).

An independent [Astra review and prioritized repair plan](../reports/astra-router-verification-review.md) analyzes the liveness, quality-policy and response-compatibility issues. No runtime fixes were applied during that review.
