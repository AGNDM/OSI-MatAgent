# SiliconFlow rerun: context-aware router waiting

Later follow-up: the [OpenRouter two-task comparison](router-wait-openrouter-comparison.md) records the current default and latest complete outcomes. Results and budgets below remain historical.

Recorded 21 October 2026. The same Binder timer and empty-furnace program goals were rerun using SiliconFlow DeepSeek V4.1 Flash and local OpenJev. No task action sequence was supplied to the agents. All attempts, including failed ones, remain in the shared 5M-token ledger.

One completed preliminary Binder rerun reduced model and verification calls, but took longer in wall time. The first furnace rerun exposed repeated diagnosis; after a context correction, final reruns were blocked by a SiliconFlow timeout and HTTP 503. A completed two-task efficiency improvement is **not established**.

## Final comparison

| Task / version | Goal met | Expert supported | Total model calls | Expert calls | Verification calls | Direct waits | Reported cloud tokens | Wall time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Binder baseline | true | true | 28 | 12 | 4 | 0 | 116,938 | 111.6 |
| Binder final rerun | false | false | 10 | 4 | 0 | 0 | 51,300 (partial) | 114.6 |
| Furnace baseline | true | false | 48 | 20 | 6 | 0 | 64,963 | 175.3 |
| Furnace final rerun | false | false | 12 | 5 | 0 | 0 | 14,686 (partial) | 153.3 |

Total model calls count router, operational chooser and expert requests, including failed/repair attempts. A lower count from a failed task is not an efficiency improvement. Reported tokens exclude unknown usage; the ledger retains the full reservation for unmetered requests. Independent goal satisfaction and the expert claim are reported separately. The historical furnace reached its goal but exhausted the call budget before final expert confirmation.

## What was corrected during this comparison

The first guarded-wait Binder rerun succeeded: 20 total calls versus 28 historically, verification calls 1 versus 4, and 109,735 cloud tokens versus 116,938. Its wall time was 131.5 s versus 111.6 s, so fewer calls did not establish faster execution. One invalid timer candidate was rejected and repaired before execution; this request is included. OpenJev chose two direct 30 s waits, and expiry was confirmed at 60 simulated seconds.

The corresponding furnace rerun failed at 48 model calls after inspection alone. OpenJev routed back to diagnosis 14 times despite an already selected operation; no verifier was called. This is an observed routing/planning failure, not an external API stop or a verifier rejecting progress. Public device states and goal checks were present, but the router saw only a boolean indicating an intent existed, not the selected capability and arguments.

The context was then expanded with the actual model-selected intent and whether its observation/selection revision is current. A current intent with no new error gets an explicit prepare-action instruction. Stale selections and new errors retain diagnosis; the offered routes and real model probability selection remain in use. No native action sequence or deterministic next-operation script was added. This addresses the missing-context hypothesis; these few demonstrations are not a causal ablation proving model reliability.

The next furnace attempt progressed through inspection and power-on without that loop, but its subsequent diagnostic request timed out at 90 s. The partial 14,686 reported tokens are not its complete usage. It remains an unresolved failed attempt with no completion/speedup claim.

The final Binder rerun reached the power-on procedure-generation stage, then SiliconFlow returned **HTTP 503**, code `50508`, “System is too busy now. Please try again later.” No power-on command was executed. This is a distinct provider failure, not a model argument/gate rejection. Its partial 51,300 reported tokens exclude the failed request; the full reservation remains charged. Further API calls stopped after these consecutive provider failures.

## Retained current-session attempts

| Run directory under ignored results/ | Outcome | Calls | Waits | Reported / conservatively charged tokens |
|---|---|---:|---:|---:|
| router-wait-siliconflow-binder-v3 | goal and expert confirmed | 20 | 2 | 109,735 / 109,735 |
| router-wait-siliconflow-furnace-v3 | llm_budget | 48 | 0 | 109,166 / 109,166 |
| router-wait-siliconflow-furnace-v4 | role_error | 12 | 0 | 14,686 partial / 54,360 |
| router-wait-siliconflow-binder-v4 | role_error | 10 | 0 | 51,300 partial / 69,997 |

## Final model-generated operations

### Binder: router-wait-siliconflow-binder-v4

| Step | Capability | Exact arguments |
|---:|---|---|
| 1 | `workflow.inspect` | `{}` |

Terminal reason: `role_error`. Goal met: `False`; expert report: `unresolved`; protocol correct: `True`; false success: `False`. Simulated time: 0.0 s. Concrete selected-intent re-diagnoses: 0.

### Furnace: router-wait-siliconflow-furnace-v4

| Step | Capability | Exact arguments |
|---:|---|---|
| 1 | `workflow.inspect` | `{}` |
| 2 | `nabertherm_l9::power.set` | `{"enabled": true}` |

Terminal reason: `role_error`. Goal met: `False`; expert report: `unresolved`; protocol correct: `True`; false success: `False`. Simulated time: 0.0 s. Concrete selected-intent re-diagnoses: 0.

## Validation and interpretation limits

The final code passed 148 regression tests in 107.18 s, including selected-intent visibility/freshness/error tests. All 570 frozen native source hashes matched. Separate deterministic adapter checks confirmed no direct-wait offers while either native process was idle, and valid offers after actual native start commands. Those adapter checks are not agent-generated trajectories and do not count as additional model task successes.

Both goal specifications match their historical manifests exactly. Goals, model-call/action/wall budgets and task simulation limits were retained. Current requests use a 90 s timeout and bounded 3,072 output tokens. Baseline furnace used the older 2K local context; current routing uses 8K. Rich routing context, lifecycle gating, selected-intent prompts and historical serialization differences also vary. These are development examples, not a controlled direct-wait ablation or a statistically established success rate. No chemistry or physical-hardware result is claimed.

A separate trace audit validated all nine actually executed tool proposals against their discovered schemas and found no reference/fixture/private-state keys in inspected expert/router inputs. No reference action files were supplied, no native code was modified, and cloud errors did not trigger scripted fallback. Full model requests/responses, exact non-secret expert configuration, contracts, manifests and trace hashes remain under each ignored results directory. The machine-readable evidence retains source hashes, error counts, exact predicates, actual returned model IDs and role/native counts.

Cumulative shared API ledger: **1,261,691 reported; 1,600,221 conservatively charged / 5,000,000; 3,399,779 remaining**. Settled 372/372; unknown-usage requests 6; in-flight/crashed 0; accounting violation False.

See [machine-readable evidence](router-wait-siliconflow-comparison.json), [earlier implementation/failure record](router-wait-validation.md), and [usage guide](../guides/single-instrument-agent.md).

Current-session usage across four attempts: **284,887 reported tokens; 343,258 conservatively charged**. Two requests had unknown usage. No budget was reset and no OpenRouter request was sent in this session.
