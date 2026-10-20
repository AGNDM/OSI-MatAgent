# Online operation-selection integration evidence

Recorded 21 October 2026. Scope: [online next-operation decisions](../guides/operation-decision.md). The offline decision benchmark is deferred. These are development integration trajectories, not a released decision dataset, calibrated confidence study or model ranking.

## Final real-model runs

DeepSeek generated candidate operations and procedure/verification records. The same local OpenJev service independently handled specialist routing and operation selection.

| Branch | Executed actions | Independent outcome | Cloud requests | Local operation calls | Reported cloud tokens |
|---|---|---|---:|---:|---:|
| Thermal history | Inspect → prepare retained reference aliquot → full XRD | Passed | 7 | 3 | 14,506 |
| Scan coverage | Inspect → full XRD, preserving the original aliquot | Passed | 5 | 2 | 9,452 |

Both final runs used exact current source hashes, had no specialist format/contract repair and no specialist-router fallback. The gate preserved the selected operation, and final native acquisitions supplied fresh cited phase evidence. The earlier scripted and local-OpenJev-with-scripted-specialists runs also passed 2/2 each, but used earlier recorded source hashes and are not relabelled as new final-version runs.

Task success is independent of perfect expert explanations. In the successful thermal trajectory, one hypothesis incorrectly calls the old scan truncated even though the original thermal-branch scan covered 20–60 degrees. The executed repair, current aliquot identity and acquired final evidence satisfy the target; this does not establish error-free diagnostic reasoning.

## Retained failures and fixes

- `results/titania-operation-deepseek-v1/`: 0/2, call budgets exhausted by repeated inspection. Diagnosis repeatedly treated present inspection as absent. Current inspection presence is now made explicit in the decision context and diagnostic system prompt; redundant valid-current titania reinspection is rejected.
- `results/titania-operation-deepseek-v2/`: 0/2, diagnostic contract failures / repeated thermal preparation. A newly prepared aliquot and an old aliquot's scan appeared together in the public state. Observations now expose an explicit current-aliquot thermal receipt and clear current scan fields until that aliquot is measured. Historical receipts remain labelled history.
- `results/titania-operation-deepseek-v3/`: process received SIGTERM without a final result. It is an interrupted attempt, not a model success or a completed failure trajectory. One unknown-usage request was conservatively settled at its full reservation (20,330 tokens). The interrupted trace and executed source hashes remain recorded.
- `results/titania-operation-deepseek-v4-thermal/` and `results/titania-operation-deepseek-v4-scan/`: independently recorded final reruns, 2/2 passed. Earlier attempts were neither overwritten nor excluded from this report.

The fixes strengthen public-state clarity and action handoffs; they do not introduce hidden evaluator answers or choose an operation from the branch label.

## Verification and budget

The full suite passes **108 tests**. All **570 pinned upstream hashes** match. Regressions cover candidate and citation validation, duplicate options, complete probabilities/argmax, backend mutation isolation, selected-intent preservation at procedure and gate, stale state, action/model/time budgets, operation errors without silent fallback, UI modes/replay, redundant inspection and current-versus-historical aliquot evidence. Existing legacy workflow modes remain operational.

The final successful pair reported **23,958 tokens** across 12 cloud requests. Including failed and interrupted online development attempts, additional conservatively charged usage is **116,858 tokens**. Shared cumulative totals: **430,167 reported**, **450,497 charged / 5,000,000**, **4,549,503 remaining**. All 271 reservations are settled; one request has unknown reported usage and retains its full charge. No accounting violation or outstanding reservation remains. No OpenRouter requests were made.

The browser demo was reloaded at `http://127.0.0.1:8793/`, its new operation panel was verified in a scripted browser episode, and saved recordings/replay were preserved. Browser specialists are test doubles; actual DeepSeek candidate generation is exercised through the CLI.

The adjacent [machine-readable evidence](operation-decision-validation.json) includes raw scores, selected option probabilities, all attempt directories, source hashes, trace hashes and budget state. Full API inputs/responses and local probability records stay in the ignored local trace directories. Neither synthetic titania endpoints nor the supplied-option probabilities establish physical laboratory reliability.
