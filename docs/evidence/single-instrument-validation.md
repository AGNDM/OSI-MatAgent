# Reference-free single-instrument validation and Astra review

Recorded 21 October 2026. Scope: reuse native interfaces for a single-instrument task trajectory without a task-specific action script. Goal failure is acceptable when execution/lifecycle handling is correct. This is development integration evidence, not the deferred offline decision benchmark or a scientific synthesis result.

## What changed

`InstrumentTask` accepts an objective and independent typed public-state predicates, plus optional observed-state constraints/readable fields and simulation limits. Unknown fields are rejected, including `reference_probe`, `steps` and `tool_menu`. The runner does not load generated device task files or `ScriptedDeviceRoles`.

`InstrumentLab` reuses registry-based native construction and the authenticated session's `describe`, `read` and `invoke`. The complete implemented agent-visible capability menu retains native schemas, optional parameters, preconditions, effects and execution information. The existing LangGraph, specialist/operation/router backends, gates, traces and API ledger remain in use. Public context for the action chooser is compacted without changing the supplied candidate identities or arguments. Procedure receives the selected capability's contract rather than a repeated full manual.

Native command acceptance, independent task satisfaction and the expert claim are separate. Native instance/operation identity, clocks, monotonically observed native revisions and valid/estimated observation timestamps are checked. Evidence is invalidated before potential effects; exact replay cannot invoke or advance time again. Pending operations stay visible and permit only supported lifecycle/read/stop actions. Model-call limits include router attempts in this new opt-in path; legacy counters are unchanged.

Clock bridges use explicit audited runtime interfaces. Binder/IKA/Nabertherm use native delta-time stepping; Quantos has an explicit-world-time and instance-bound poll/cancel bridge. Binder's placeholder poll is not treated as support. No generic scene acknowledgement is supplied. Model requests cannot invent clock resets, host material fixtures or scene confirmations. Constraints are checked at observed states and await chunks, not continuously between integration steps.

## Astra review

Astra approved the design before implementation. Its first source review required revisions to remove automatic scene confirmation, validate terminal operation identity, capture initialization failures, count router model calls, separate independent success from claims, and use actual Quantos response envelopes. A later review required real native evidence timestamps rather than allowing undated observations. These were fixed and covered by adversarial regressions.

Astra's final verdict was **PASS: implementation and both completed real-model demonstrations are accepted within the agreed scope, with no remaining blockers.** It independently ran the final **28 targeted tests**, checked native receipts and trace hashes, verified actual DeepSeek response identities and model-call counts, and found no reference-script/hidden-state leakage in reviewed inputs. It also accepted the CLI fallback preserving unresolved operation identity when the independent final read fails. This is an independent AI technical review, not human scientific validation.

## Real-model trajectories

Both use DeepSeek-V4.1-Flash through the existing budgeted SiliconFlow transport and local OpenJev for specialist routing and operational selection. The model receives native tool contracts, task predicates and current public observations, without a reference sequence. Native examples describe argument syntax; they are not a supplied task solution.

| Run | Actual model-generated operations | Independent result | Expert report | Protocol |
|---|---|---|---|---|
| Binder v3 | Inspect; `power.set(on=true)`; `timer.set(days=0,hours=0,minutes=1,fan_after=on)`; await 65 s | `timer.state.expired=true`, `set_s=60`; fresh evidence | Supported, citing `obs-7` | Correct |
| Nabertherm v1 | Inspect; power on; configure a 323.15 K / 0.1 K/s / 10 s segment; start program; await 120 s three times | `program.phase=completed`, hold 10 s; configured segment verified; 360 simulated seconds | Unresolved because the 48-call budget was exhausted before final confirmation | Correct |

The furnace outcome deliberately remains `goal_met=true` and `agent_claim_supported=false`. `task_success` reflects independent goal satisfaction, not whether the expert produced a positive report. The unresolved report and missing final citation are preserved; this is not relabelled as an expert-confirmed success.

| Accounting | Binder v3 | Furnace v1 |
|---|---:|---:|
| Specialist requests | 12 | 20 |
| Model-router calls | 12 | 21 |
| Operation-selector calls | 4 | 7 |
| Total model-call counter | 28 | 48 |
| Native invoke calls | 2 | 3 |
| Native public reads, including independent evaluation | 6 | 9 |
| Native clock steps | 7 | 36 |
| Reported cloud tokens | 116,938 | 64,963 |
| False success | false | false |

The goal predicates and permitted instruments are authored task specifications, not inferred evaluation truth. No Python procedure was authored for either task. No `thermal.prepare_reference` or `xrd.acquire` macro is used. The furnace's initial native door is already closed; the runner does not fake a door-motion scene callback.

## Failed attempts retained

- **Binder v1:** local OpenJev was stopped. One expert format/evidence repair occurred; operational selection failed before any device operation. Goal not met, no false success. Reported cloud usage: 90,401 tokens.
- **Binder v2:** an expert output hit the previous completion limit and was repaired. Inspection ran, but the local chooser returned HTTP 400 at the old 2K context size. No native write occurred. Reported cloud usage: 144,517 tokens.
- The owned service was restarted with 8K context using the existing startup helper's new optional `--context` parameter. Expert JSON serialization now uses compact UTF-8 instead of escaped Unicode, preserving JSON values while reducing prompt overhead. Output remains explicitly bounded, and candidate prose is requested to be brief. Binder v3 then completed.

Failures were not overwritten or excluded from API accounting. The successful thermal trajectory's limits and historical source hashes remain as executed. Its manifest precedes later CLI-only refinements; it is not claimed to have an identical current source tree. Per-run source hashes and current differences are retained in the machine-readable evidence.

## Verification, provenance and limitations

The final full suite passed **139 tests in 88.73 seconds**, including the last pending-identity regression. Astra independently passed the 28 targeted tests. All 570 pinned upstream native hashes matched. The new targeted suite covers optional arguments, precondition/schema/identity/revision rejection, replay, pending-to-running/terminal states, partial failure, cancellation, unknown polling, missing/wrong IDs, unavailable/old/undated observations, initialization/final-read failures, router budgets, graph/action/deadline exits and reference-file isolation.

A separate audit checked each retained tool proposal against its discovered native schema and recomputed trace hashes, role/router/operation counts and cloud usage. No forbidden reference/fixture/private-state keys were present in inspected model role inputs. The schema/lifecycle audit validates interface handling; it does not establish optimal planning or completeness of the candidate menu.

Total newly reported cloud usage across all four attempts: **416,819 tokens**. Shared cumulative usage: **846,986 reported**, **867,316 conservatively charged / 5,000,000**, leaving **4,132,684**. All 310 reservations are settled; the one unknown-usage reservation is historical and remains fully charged. No accounting violation or in-flight reservation remains. No OpenRouter calls were made.

Quantos pending/partial/cancel tests use transport doubles shaped like the native interface; no real material-loaded Quantos acquisition is claimed. Unavailable fixtures, unsupported clocks and absent scene adapters can still prevent tasks on other devices. Neither demonstration validates material chemistry, hardware, geometry, continuous safety or a released benchmark. Graph-bound and honest unresolved trajectories are valid outputs of this entry point.

See the [usage guide](../guides/single-instrument-agent.md) and [compact evidence](single-instrument-validation.json). Full requests/responses, native contracts, manifests and traces remain in ignored `results/single-instrument-*` directories. The existing three untracked user report/proposal artifacts were preserved.
