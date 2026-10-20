# Titania acquisition development evidence

Recorded 21 October 2026. [Scenario](../demos/titania/scenario.md), [run instructions and limitations](../demos/titania/workflow.md), [compact machine-readable evidence](titania-validation.json), [paper provenance](titania-source.json).

| Branch | Real expert action sequence after inspection | Independent result | Cloud tokens | Expert requests | Router fallbacks | Citation repairs |
|---|---|---|---:|---:|---:|---:|
| Thermal history | Retained dried sister aliquot → native reference thermal program → cooldown → full XRD | Passed | 13,081 | 8 | 0 | 1 |
| Scan coverage | Preserve aliquot → full XRD | Passed | 9,536 | 6 | 0 | 1 |

Specialists used SiliconFlow `deepseek-ai/DeepSeek-V4.1-Flash`; routing used the local OpenJev Flash 9B Q4 model. Both real trajectories and both scripted trajectories passed. Both final scans acquired 801 points and matched all five listed reference windows for each phase. This is qualitative evidence in the declared synthetic fixture, not a quantitative assay.

Both real trajectories initially cited `paper-protocol` and `public-xrd-reference` as observation IDs. The evidence validator rejected these citations; one bounded repair per trajectory produced valid records. These failures and exact API bodies/responses are retained in the local JSONL traces. No claim of error-free first attempts is made.

The native audit recorded scan start with 0/801 points, an intermediate 440/801 points while still running, and completion at 801/801. Scheduler, furnace and XRD clocks agreed at 26,784 simulated seconds in the audit. The furnace public program reported a completed 7,200-second hold; cooldown reached the authored 313.15 K transfer threshold. The agent's hidden-template-truth query was denied.

A full rescan after the wrong thermal history still lacked rutile. Extra heat followed by the truncated scan still lacked adequate coverage. Both controls failed independent evaluation even when given deliberately optimistic `supported` reports. A separate regression also rejects unnecessary thermal replacement even if followed by a valid full scan.

All **80 project tests passed**. The new tests cover contrasting graph paths, acquired partial/completed data, shared clocks, wrong repairs, preservation policy, timeout with no valid measurement, wrong/stale/unavailable evidence, missing citations, request conflicts/idempotency and host-truth isolation. All **570 pinned upstream files** verified without modification.

This development run used **22,617 tokens across 14 requests**, including repairs. Shared cumulative budget: **333,639 / 5,000,000**, leaving **4,666,361**; all 220 requests settled with complete reported usage and no outstanding reservations. No OpenRouter requests were made.

Raw artifacts remain local under `results/titania-deepseek-openjev-v1/` and `results/titania-scripted-v2/`; earlier scripted pilot `results/titania-scripted-v1/` is retained. Trace paths and SHA-256 hashes, executed source hashes, model IDs, audit values and scores are preserved in the adjacent JSON.

Scientific limits: the original paper anchors the selected treatment and qualitative endpoint. Furnace/sample coupling, phase endpoints, retained dry stock and transfers are fixture assumptions. Native XRD integrates RRUFF-derived synthetic profiles, not the paper's measured curves. Fixed source snippets and an explicit decision policy do not demonstrate retrieval benefit or general materials reasoning. The browser UI still shows drying; this scenario currently runs from the CLI.
