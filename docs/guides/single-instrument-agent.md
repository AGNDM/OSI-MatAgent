# Native instrument tasks

## Run

```bash
python -m app.instrument_cli --backend deepseek \
  --router openrouter-jev --operation openrouter-jev \
  --task cases/acceptance/v1/binder-timer.json \
  --output results/my-task
```

Set `OPENROUTER_API_KEY` before starting the process. The default expert is configured in `configs/openrouter-deepseek-v4.1-flash.json`. `--config` selects another supported endpoint explicitly. SiliconFlow uses `SILICONFLOW_API_KEY`; there is no automatic provider fallback. Use a new output directory for each attempt.

## Task input

A task declares a device (or `auto`), objective and public-state predicates. Optional constraints and simulation limits bound the episode. It contains no action sequence.

```json
{
  "task_id": "oven-timer",
  "device": "binder_fd056",
  "objective": "Make a one-minute timer expire and verify it.",
  "predicates": [
    {"state_id": "timer.state", "path": ["expired"], "value": true, "unit": "1"},
    {"state_id": "timer.state", "path": ["set_s"], "value": 60, "unit": "1"}
  ],
  "max_simulated_seconds": 300,
  "max_wait_seconds": 120
}
```

Operators are `eq`, `ge` and `le`. Units refer to the native state slot, including object-valued slots. Missing, stale, unavailable or incorrectly typed evidence cannot establish success.

For `device: "auto"`, the expert sees the public directory, searches/describes candidate devices and proposes up to four for the decision model. Exactly one native instrument is bound; predicates remain fixed. `--knowledge on` adds retrieved facts with separate citations, without providing an execution recipe.

## Execution

Native capabilities use `<instrument-slug>::<capability>`. Public schemas, preconditions and execution metadata come from native `describe`. Models propose structured data, never executable Python.

- `workflow.inspect` and `workflow.readback` obtain public observations.
- `workflow.await` advances the supported native simulation clock and obtains fresh evidence; it does not sleep for physical process time.
- Declared asynchronous operations expose supported poll/cancel methods and retain pending operation identity.
- `workflow.stop` records an unresolved outcome.

The gate checks scope, revision, capability arguments, selected intent and replay identity. Evidence is invalidated by mutations and clock advancement. A successful start receipt does not establish process completion. Where supported, the router can choose a bounded direct wait using current task, process and pending-operation context; `--no-router-wait` disables that shortcut.

The default routing policy selects a legal maximum-probability option without confidence-threshold escalation. Backend failures remain explicit. Verification must cite fresh observations, agree with the public predicates and resolve pending work; unchanged evidence cannot trigger repeated valid verification.

## Outcomes and limits

`goal_met` / `task_success` describe public-state outcomes. `agent_claim_supported`, `evidence_cited` and `false_success` describe the agent's report. `protocol_correct` and `lifecycle_resolved` describe execution integrity, independently of whether the goal was achieved.

The CLI bounds actions, model calls, wall time, simulated time and graph steps. Its `--max-output-tokens` defaults to 6,144 with an 8,192 maximum; benchmark expert configurations use separate allowances. Supported OpenRouter calls record usage in the durable ledger; SiliconFlow retains its shared 5M-token limit. Never reset the ledger to bypass accounting.

Each attempt saves its manifest, public contract, trace and result under `results/`. Use `python -m app.instrument_cli --help` for all options. Generic material transfer and multi-instrument synthesis are not implemented by this entry point.
