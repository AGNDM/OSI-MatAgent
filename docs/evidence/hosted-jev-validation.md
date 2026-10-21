# Hosted Jev decisions, verifier consistency and output-budget changes

Recorded 21 October 2026. The requested changes are implemented and both original instrument tasks completed with supported expert claims, independent native goal satisfaction, correct protocol and no false success. Hosted TypeSafe Jev is a different model from local OpenJev Flash 9B; this is a replacement experiment, not a deployment-only comparison.

## Same failed inputs, different decisions

OpenRouter serves typed decisions at `https://openrouter.ai/api/alpha/decisions`. The adapter requests pinned `typesafe/jev-1.14`; returned identity was `typesafe/jev-1.14-20261014`. Native choice probabilities are validated and retained; none are fabricated.

| Recorded furnace state | Local OpenJev choice / probability | Hosted Jev choice / probability |
|---|---|---|
| 43 | verify / 0.766 | diagnose / 1.000 |
| 47 | verify / 0.564 | diagnose / 1.000 |

These two calls reused the exact stored routing summary and offered routes, before applying any new anti-repeat guard. The hosted API wraps that same prompt in its typed Choice question. This is targeted failure replay, not a decision benchmark or proof of general accuracy.

## Original tasks: end-to-end comparison

| Task / version | Native goal / supported claim | Model calls | Verification calls | Total cloud tokens | Wall time |
|---|---|---:|---:|---:|---:|
| Binder: historical SiliconFlow + local OpenJev | True / True | 28 | 4 | 116,938 | 111.6 s |
| Binder: previous OpenRouter expert + local OpenJev | True / True | 19 | 1 | 83,263 | 113.2 s |
| Binder: hosted Jev + corrected verifier | True / True | 19 | 1 | 107,764 | 23.0 s |
| Furnace: historical SiliconFlow + local OpenJev | True / False | 48 | 6 | 64,963 | 175.3 s |
| Furnace: previous OpenRouter expert + local OpenJev | False / False | 35 | 12 | 70,239 | 303.0 s |
| Furnace: hosted Jev + corrected verifier | True / True | 31 | 1 | 75,386 | 27.9 s |

All actual requests, including repairs, are counted. Previous local decision calls consumed local compute, while hosted decisions now consume API tokens. A failed prior run is not a valid efficiency target. Relative to the historical furnace, current calls are lower and the expert now confirms completion; relative to the last successful Binder, total calls remain 19 while wall time is lower. These are observed runs, not controlled speed benchmarks.

## New cloud-token breakdown

| Task | DeepSeek expert requests / tokens | Jev decision requests / tokens | Direct waits | Simulated time |
|---|---:|---:|---:|---:|
| Binder | 7 / 83,363 | 12 / 24,401 | 2 | 60.0 s |
| Furnace | 9 / 33,015 | 22 / 42,371 | 9 | 270.0 s |

The furnace model-selected configure and start commands, then chose nine 30-second waits. It completed after 270 simulated seconds and cited obs-25. Binder selected its one-minute timer and two 30-second waits, then cited obs-9. Both used only one final verification request, with zero expert repair/error events. Full ordered operations, actual decision model snapshots and hashes are retained in the machine-readable evidence.

## Verification policy repair

The instrument verifier explicitly accepts both valid and estimated native state qualities, matching the existing evaluate() implementation. Structured policy in the prompt distinguishes evaluable false predicates (contradicted), unavailable/stale/identity/unit/path evidence or pending operation (unresolved), and all satisfied goals/constraints (supported). A known false conjunct establishes contradiction even if another field is unavailable.

The contract validator reuses evaluate() plus the existing scope/revision/freshness rules. It rejects inconsistent model verdicts and retains the original output in the trace; it does not rewrite a model failure into success. Validation repair remains bounded. Offline replay of the old seven parsed reports rejects the four unresolved classifications and accepts the three contradicted ones under the declared policy. Final supported reports on the two new runs passed this validator.

## Was empty output caused by insufficient quota?

The old five empty completions were output-budget exhaustion: finish_reason=length and all 3,072 completion tokens recorded as reasoning. They were not account/key quota errors. Why upstream reasoning occurred despite reasoning.enabled=false remains unproven.

The OpenRouter expert configuration and single-instrument CLI output ceiling are raised from 3,072 in the earlier runs to 6,144 per request; the global 5M budget is unchanged and the hard per-request ceiling remains 8,192. A response-envelope classifier distinguishes truncation, empty/whitespace content, missing choices/provider error and JSON/schema errors. Truncation/empty responses terminate with explicit error kinds rather than entering generic JSON-format repair. Offline replay classifies all five old failures as expert_output_truncated.

Neither new run returned empty or truncated output; both reported zero reasoning tokens. Therefore the run does not isolate the benefit of a larger ceiling from the corrected prompt and changed provider choices. It would be incorrect to claim that raising the ceiling alone solved the old provider behavior.

## Implementation and budget

OpenRouter Jev is available for both router and operational choice with `--router openrouter-jev --operation openrouter-jev`. The existing local mode is still available. Hosted routing fails closed if the decision API fails; it does not execute a silently substituted rules decision. The normal gate, identity/revision checks, audited native stepping, independent evaluator and actual model-generated operations remain in use. No reference task program or automatically inserted start command was added.

The CLI now includes routed decision usage as well as expert usage, without double counting operation metadata. Typed Decision API usage is normalized from input_tokens/output_tokens for the same durable ledger. Usage validity is shared between run summaries and ledger settlement. Hosted decisions have no max_tokens parameter: requests are restricted to one text Choice with at most four options and at most 24,000 serialized bytes, with a conservative full-model-context reservation plus request/framing before sending. Missing usage retains the full reservation.

The two state replays and two trajectories used **186,837 reported/charged tokens**. Shared cumulative ledger: **1,602,067 reported / 1,957,431 conservatively charged of 5,000,000**, **3,042,569 remaining**. Settled 452/452; in-flight/crashed 0; accounting violation False. No allowance was reset or increased.

## Validation and limits

165 tests passed in 80.80 seconds, including hosted typed response/probability validation, operation selection, failed-request accounting, fail-closed routing, estimated-quality semantics, stale/identity/pending cases, and output-limit classification without generic retries. Actual executed tool proposals were schema-audited; no forbidden reference/fixture/private-state keys appeared in inspected public inputs. All 570 pinned native hashes match. Both new runs share the same recorded app source hashes.

The semantic unchanged-evidence verification guard was not implemented in the frozen source used for this experiment; hosted Jev avoided the failing branches in these examples. A subsequent [Astra code review](single-instrument-code-review.md) reproduced the remaining loop and a missing-citation defect; both were fixed and validated offline. These historical runs were not repeated after those fixes. Model, decision transport, verifier policy and output limits changed together in the full runs. Provider latency and concurrent offline testing are uncontrolled. The two recorded-state replays give narrower evidence of a changed decision, but do not prove calibration or robustness on new tasks. The later guard enforces duplicate-verification suppression, but cannot guarantee useful planning on unseen tasks. No chemistry or hardware result is claimed.

Official references: [Jev decision model/API](https://openrouter.ai/blog/insights/what-is-jev/), [TypeSafe model limits](https://docs.typesafe.ai/models). See [machine-readable evidence](hosted-jev-validation.json), [previous comparison](router-wait-openrouter-comparison.md), and [Astra review](../reports/astra-router-verification-review.md). Raw requests, response metadata, manifests and traces remain in ignored results directories; no credentials or private reasoning are retained.
