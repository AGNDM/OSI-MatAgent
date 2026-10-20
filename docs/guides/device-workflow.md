# Native Device Workflow

The existing diagnostic, procedure, gate, executor and verification LangGraph now accepts a device-specific profile. The original drying profile remains available. Native actions are named `device_slug::capability_id`, and their arguments are validated against the pinned agent-session contract.

## Coverage and boundaries

- 203 catalog entries were inventoried; 198 native sessions were established.
- 194 devices have one verified control/readback fixture each. Nine entries remain unaccepted; the full reasons are recorded in `cases/devices/registry.json`.
- A task exposes only its required capability subset. Passing a power, setpoint or door-control task does not validate acquisition, chemical behavior, hardware, robotics or all instrument capabilities.
- Each task resets an independent single-device world. There is no shared material transfer or common multi-device clock.
- Task fixtures are authored integration tests, not an external scientific benchmark. The reference actions, initialization arguments and reference probes are not passed to experts.

## Native execution and evidence

The host performs explicit environment initialization through native evaluator-owned setup. Where required, mechanical preconditions follow pinned upstream tests. These setup events are recorded in the fixture and score, separately from the agent action budget.

The agent receives only public observations, its task goal, a bounded capability menu and the latest action receipt. A successful native command immediately reads public state back into `observations.measurement`; the command and read each consume an action. This prevents the next specialist from treating the old inspection as current state.

The gate checks device scope, revision, request ID, prerequisites, capability allowlisting and JSON Schema. Host setup, fault injection and simulation-clock controls are not exposed as model tools. Reusing a completed request returns its recorded result without executing twice. Uncertain native execution does not trigger an automatic replacement action.

Each device episode allows six laboratory calls, twelve expert calls, 180 seconds and bounded graph execution. One schema repair is allowed and counts as another model call. The independent evaluator re-reads native public state and requires an initially nontrivial goal, an accepted native action, a matching final state and cited fresh readback. A model's `supported` response alone is insufficient.

## Commands

```bash
conda activate osi-hack

# Offline regression: no external API requests.
python -m bench.device_tasks --output results/my-device-offline

# One device, scripted specialists and rules routing.
python -m bench.device_tasks --device mt_xpr204e --output results/my-balance-test

# Real DeepSeek specialists; uses SILICONFLOW_API_KEY from the environment.
python -m bench.device_tasks --backend deepseek --device mt_xpr204e \
  --output results/my-balance-deepseek

# Real local routing, after starting the OpenJev service.
python -m bench.device_tasks --backend deepseek --router openjev \
  --device mt_xpr204e --output results/my-balance-combined
```

Every output directory must be new. The default backend is explicitly labeled scripted; the default router is rules. A declared category sample is stored in `configs/device-debug-sample.json` and can be selected with `--sample`. Real API calls require an explicit `--backend deepseek`; they are not part of normal tests.

The browser on port 8793 remains the original drying demonstration. New device tasks currently use the CLI.

## Shared 5M-token device debugging budget

All device DeepSeek requests share `results/device-expansion-budget-v1/ledger.jsonl`. New output directories do not reset it. Do not delete or recreate the ledger to restart the allowance.

Before transport, the ledger reserves the UTF-8 request-byte bound plus 8,192 framing tokens and the configured output ceiling. File locking prevents concurrent over-reservation. Complete provider prompt/completion usage settles the reservation; failed requests, missing usage and interrupted processes retain the full reserved amount. An accounting anomaly blocks subsequent requests.

Requests disable thinking and streaming, use a bounded output limit and have no transport retries. Output repairs are separately metered calls. The ledger contains request IDs, model IDs and token accounting, not credentials. It controls token usage; it is not a monetary billing statement. Other historical experiment entry points are not a substitute for this budgeted device runner.

## Reproduce the native subset

```bash
python tools/fetch_device_runtime.py --download
python tools/fetch_device_runtime.py
python tools/discover_devices.py
python tools/build_device_tasks.py
python -m pytest -q
```

The first command restores the ignored upstream cache from the committed manifests; the second only verifies it. Discovery and fixture generation execute native code without model requests. They regenerate developer-owned fixtures, so use a new experiment output after changing them.

Code revision: `123cf78593469871e41b8dbd63a11dfb7c5ca251`. Dataset revision: `4ca4a42dab2ca0a884e449d202e9ad96625d5648`. Preserve the separate code/data licenses and attribution. No geometry download is required.

## Recorded outcomes

194/194 offline tasks passed. Real DeepSeek completed the predeclared 20-device sample across 16 categories; two additional real DeepSeek/OpenJev combinations passed without routing fallback. Failed earlier pilots remain recorded. Total device API use was 311,022 tokens across 206 requests. See [the evidence summary](../reports/evaluation-summary.md) and [the remaining-work checklist](../status/checklist.md).
