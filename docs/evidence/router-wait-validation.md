# Context-aware router wait: implementation and current validation

Follow-up: SiliconFlow was subsequently selected again. See the [same-task comparison](router-wait-siliconflow-comparison.md) for a successful preliminary Binder run, a routing-context correction and retained final provider failures. Status and budget below describe the earlier OpenRouter switch.

Later follow-up: the [OpenRouter two-task comparison](router-wait-openrouter-comparison.md) records the current default and latest complete outcomes. Results and budgets below remain historical.

Recorded 21 October 2026. This is integration debugging, not an offline benchmark or a material synthesis result.

## Change

The OpenJev router now receives the task objective/predicates, current public states, recent action outcomes, pending operations and remaining simulation/model/action/time budgets. It can directly propose bounded short/long waits. The same trusted action gate, identity/revision checks, replay protection, native clock bridge, constraint checks and fresh readback remain in use. Diagnostic, operation-selection and procedure model calls are skipped for these waits. Final verification remains a separate expert role.

An early real run showed why public lifecycle gating is necessary: the timer was idle, but the model repeatedly chose to wait. The corrected offer guard requires fresh evidence of an active process in the goal's state family (or supported pending work), suppresses explicit paused/blocked/failed/stopped/idle markers and rejects unrelated activity such as a fan. Unknown lifecycle semantics keep the ordinary expert path. Durations are capped by task and remaining simulation limits. This is a lifecycle guard, not a task-specific action sequence.

## Retained attempts

| Attempt | Outcome | Implication |
|---|---|---|
| Historical Binder v3 | Goal satisfied; supported report; 28 model calls, 116,938 cloud tokens | Baseline |
| Historical furnace v1 | Goal satisfied; final expert confirmation absent at 48-call limit; 64,963 cloud tokens | Baseline efficiency problem |
| Router-wait Binder/furnace v1 | First cloud request timed out; no instrument actions | No efficiency conclusion; unknown usage fully reserved |
| Router-wait Binder v2, before lifecycle fix | Goal failed; 48 calls, 8 direct waits, 300 simulated seconds; 86,897 cloud tokens | Idle waiting is a planning failure, not an improvement |
| Router-wait furnace v2, before lifecycle fix | Interrupted after inspection while switching providers; no final score | Preserved partial trace; excluded from completion comparisons |
| Corrected OpenRouter Binder v1 | First expert request rejected with HTTP 403, `key_limit`, limit $0 | No model-generated operation; second task not sent against the known account block |

The prior SiliconFlow child was stopped explicitly after the provider switch. Its partial trace is retained with `interruption.json`. No interrupted result is fabricated. The blocked OpenRouter request is conservatively charged at its full reservation because usage was unavailable. Provider failures are not reported as zero token usage.

## Provider and reproducibility

The CLI now defaults to `configs/openrouter-deepseek-v4.1-flash.json`, endpoint `https://openrouter.ai/api/v1`, model `deepseek/deepseek-v4.1-flash`, with `OPENROUTER_API_KEY`. The official live model catalog confirmed this exact ID. Reasoning is disabled using OpenRouter's `reasoning.enabled=false`, responses are non-streaming, output is bounded and no automatic retries or model/provider substitutions are added. The generic free-only OpenRouter debug adapter is unchanged. The user-authorized paid DeepSeek adapter reuses the existing durable shared 5M-token ledger.

A meaningful before/after comparison remains pending a nonzero OpenRouter key spending allowance. No speedup or improved completion rate is claimed. Only the original two tasks will be rerun. Historical furnace used a 2K local context; the current local router uses 8K. The provider switch, context difference and corrected prompts preclude attributing any future difference solely to direct waiting. Each run retains frozen source hashes rather than being relabelled as the current tree.

## Local validation

Eight targeted tests pass for direct-wait gating/execution, stale decisions, action/time bounds, idle/unrelated/paused process rejection, OpenRouter request construction, failed-request accounting and explicit model authorization. All 570 pinned upstream native hashes match. The full regression suite passed **147 tests in 108.09 seconds**; final transport/configuration checks also pass. These tests do not establish actual-model efficiency.

See [machine-readable evidence](router-wait-validation.json) and [usage guide](../guides/single-instrument-agent.md). Full native contracts, manifests, requests and traces remain under ignored `results/`.

Shared cumulative API budget after all retained attempts: **976,804 reported tokens; 1,256,963 conservatively charged / 5,000,000; 3,743,037 remaining**. All 339 reservations are settled. Four requests have unknown usage and retain full charges; no accounting violation remains.
