# Error Annotation Codebook v1

Every error claim must cite concrete trace events. `bench.annotate` emits machine suggestions, not confirmed first causal errors. A recovered formatting error remains in the trace.

| Code | Positive example | Boundary |
|---|---|---|
| E1 Observation | Treat unavailable mass as zero; cite an unknown observation ID | Incorrect inference from reliable evidence is E2. |
| E2 Domain reasoning | Claim cleaning a balance pan dries the sample | A malformed parameter is E4. |
| E3 Planning | Correct before inspection; repeat ineffective actions without new information | A deduplicated retry using the same request ID is not inherently an error. |
| E4 Tool use | Wrong capability, arguments or schema; gate-rejected prerequisite | A stale handoff may be E5 after semantic review. |
| E5 Coordination | Pass a stale revision, lose evidence or cross role authority | Wrong specialist routing is E6. |
| E6 Routing | Select procedure without an intent; repeatedly misdispatch uncertainty | A service outage is E9; fallback is a downstream event. |
| E7 Verification | Claim success without fresh evidence; reject completion despite sufficient evidence | Correctly abstaining on unavailable evidence is not an error. |
| E8 Budget/termination | Waste the budget through repetition; fail to stop appropriately | Bounded, justified unresolved termination is not automatically E8. |
| E9 Infrastructure/harness | HTTP 429, timeout, native exception or replay defect | A normally returned but incorrect scientific judgment is E2/E7. |

`errors.csv` records run, step, component, label, evidence, expected/actual behavior, candidate first/downstream status and review state. Heatmaps count episodes with each label divided by all attempts for that configuration. Labels may overlap; failures and infrastructure errors are retained.

Automatic ordering identifies the first mechanically visible anomaly, not necessarily the first consequential causal error. Semantic E2/E3 attribution requires review; a failure without mechanical evidence remains uncertain. The formal proposal requires two independent human reviewers for at least six failed trajectories with disagreement resolution. Current tools and AI reviews do not fabricate or substitute for that review.
