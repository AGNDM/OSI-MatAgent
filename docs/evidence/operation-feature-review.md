# Astra review of the online operation-selection feature

Date: 21 October 2026. Reviewer: Astra (`gpt-6-astra`). Baseline: development commit `3406e11`; review covers the pending online operation-selection changes and the overall affected execution/publication paths. The user authorized push only after review. This is an AI technical review, not a human scientific audit or benchmark acceptance.

## Initial outcome: revise

1. **Required: reject cloud experts in the browser.** Although the browser was documented as cloud-free, a configured SiliconFlow endpoint could expose `configured-openjev`, then create an unbudgeted `CompatibleExpert`. This pre-existing path bypassed the shared cloud ledger. `DemoController` now rejects non-local providers before output creation or transport/mode exposure. The budgeted SiliconFlow CLI remains available. Offline regressions verify rejection without network/output and continued support for configured loopback experts.
2. **Recommended and fixed: preserve operation-selector provenance in summaries.** Grouping only by specialist/router merged operation-disabled and operation-enabled runs. Summary grouping and output now include `operation_backend`, with missing historical fields treated as disabled. An offline regression checks separate denominators and chooser identity.

## Re-review outcome: pass

Astra re-read both fixes and independently executed the three new offline regressions: **3 passed**. Its explicit final recommendation was **PASS, no remaining blockers for this bounded integration feature**, subject to completing the full regression run and updating the current test count.

The review accepted copied candidate inputs, schema/citation/probability checks, selected-intent preservation at procedure and gate, revision checks, deadline/action/model limits, explicit selector failure without silent fallback, replay behavior and current-aliquot evidence isolation. The reviewer independently passed the preceding 108-test suite and verified all 570 native hashes. Its durable-ledger recomputation matched 430,167 reported and 450,497 charged tokens, with the unknown request fully charged and no outstanding reservation or accounting violation. Candidate publication files passed English, local-link and available-secret-value checks.

Final post-fix full regression: **111 passed in 87.88 seconds**. Current project status records this release result. The new fixes use offline tests; no paid model calls were made for review or re-review. Existing real-model evidence keeps its executed source/trace hashes and historical test snapshot; it is not relabelled as a new experiment after these browser/summary fixes.

## Accepted limits

The operation node is opt-in; the specialist router remains separate. The browser uses local models or scripted specialists only. The titania chemistry/transfer remain authored and diffraction is synthetic. The final real-model pair passed task outcomes, but a recorded expert hypothesis misstates an earlier scan range. Candidate completeness, calibrated probabilities, model superiority and physical laboratory reliability are not established. The offline decision benchmark was deferred and was not a release prerequisite.
