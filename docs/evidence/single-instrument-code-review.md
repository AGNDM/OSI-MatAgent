# Astra code and demonstration review

Reviewed on 21 October 2026 by GPT-6 Astra against the canonical development checkout. Final verdict: **PASS**, after fixing two independently reproduced blockers. No remaining serious defect was found in the reviewed changes. This is a scoped review, not a guarantee for arbitrary instruments or model outputs.

## Scope

The reference-free single-instrument runner, native public-session adapter, LangGraph routing and operation selection, direct bounded waits, verification semantics, hosted TypeSafe Jev transport, general-expert response handling, shared token ledger, configurations, regression tests and demonstration reports were reviewed together. Particular attention went to execution authority, stale observations, pending-operation identity, repeated model work, citation contracts, cancellation/accounting and whether reported results were supported by saved evidence.

The detailed Binder/furnace report and its machine-readable companion were reconciled with local frozen manifests and traces. Reviewer work used offline tests and doubles only: no new project model/API requests, dependency installations or live trajectories.

## Initial findings and fixes

| Severity | Reproduced defect | Implemented correction |
|---|---|---|
| P1 | A router could repeatedly verify one unchanged observation until the model-call budget was exhausted. Eight valid contradicted reports cited the same evidence without advancing native time. | Persist consumed semantic evidence signatures independently of the transient verification object. Filter verification at routing, deliberation and fallback boundaries; reject duplicate entry before calling the expert. |
| P1 | A supported report with an empty citation list passed the validator, terminated the graph and was later scored as false success. | Require the current measurement ID for supported and contradicted reports. Reject inconsistencies through the existing bounded repair path, preserving model outputs and independent scoring. |
| Minor | The device-runner module description incorrectly claimed it never used OpenRouter. | Correct the description to reflect the configured expert and shared ledger. |

The signature contains task goals/constraints/scope, native public values, units, quality, clock, pending state and evidence-freshness conditions. Observation IDs, capture counters and workflow/native revision counters alone do not establish information progress. A meaningful readback change can reopen verification. Only a parsed, contract-valid assessment consumes a signature; response failures retain bounded handling. The last valid assessment remains available to later diagnosis and routing after transient state is cleared.

This guard prevents repeated valid verification of identical semantic evidence. It does not guarantee useful future planning: a model may still select redundant reads or other unproductive actions, bounded by ordinary episode budgets. It is currently enabled through the single-instrument adapter hook; legacy profiles without that hook retain their existing behavior.

## Independent Astra re-review

Astra independently passed **61 focused tests in 0.77 seconds**, including all seven new regression cases. Additional offline counterexamples established:

- Reintroducing a consumed verify route into the offered menu terminates at the verification entry guard with zero additional verifier calls.
- Returning supported without citations on both the initial attempt and repair ends unresolved, consumes no evidence signature and produces no false success.
- Genuinely changed goal evidence permits a new assessment, including contradicted to supported.
- Evidence consumption survives diagnosis and execution; changing observation IDs or either revision counter alone does not reopen verification.
- Both hosted Jev and budgeted expert requests cancelled during HTTP handling settle exactly once, retain the full reservation when usage is unknown, and leave no in-flight request or accounting violation.

The citation validator retains bounded repair and does not manufacture a successful verdict. No further serious flaw was established in fail-closed hosted execution, pending-operation identity, uncertain native reads, response classification or current usage aggregation.

## Parent validation and report provenance

The complete offline suite passed **172 tests in 80.91 seconds**. All **570 pinned upstream hashes matched**. Whitespace, English-documentation and local-link checks are recorded with the prepared commit. Unrelated artwork and separate literature-proposal/Titania reports are excluded.

Historical demonstrations remain unchanged: Binder has 19 unique API reservations, 107,764 tokens and five executed proposals; furnace has 31 reservations, 75,386 tokens and thirteen proposals. All eight raw artifact hashes match; recorded source hashes identify the frozen implementation actually run. Both saved independent scores show goal satisfaction and no false success. The review fixes above were added afterward and validated offline; these two live demonstrations are not reruns of the fixed code.

No new API requests were made for this review. The last settled ledger snapshot remains 452 requests, 1,602,067 reported tokens and 1,957,431 conservatively charged tokens under the unchanged 5M ceiling, with zero outstanding reservations.

See the [detailed demo report](../reports/single-instrument-demo-for-benchmark.md), [recorded hosted-model validation](hosted-jev-validation.md), [current guide](../guides/single-instrument-agent.md) and [review metadata](single-instrument-code-review.json). Formal benchmark construction, held-out evaluation, chemistry calibration and general multi-instrument orchestration remain open work.
