# Decision-Routed Lab

A local LangGraph prototype for evidence-driven virtual laboratory control. Diagnostic, procedure and verification specialists propose structured actions; a deterministic gate checks them before native ChemLab execution. Routing uses a rules baseline or a separately deployed local OpenJev model. The general expert backend supports the explicitly configured SiliconFlow DeepSeek endpoint.

The repository contains three workflow profiles:

- **Drying troubleshooting:** distinguish a process-setting problem from a measurement problem, apply a justified correction, and verify fresh evidence with an independent evaluator.
- **Device control:** run a small native control/readback task for each of 194 registered devices across 16 categories. These are independently reset single-device integration tasks, not complete scientific experiments.
- **Paper-grounded titania:** distinguish a delivered thermal-program error from truncated XRD coverage; execute native furnace and diffraction acquisition macros. See the [scenario card](docs/titania-scenario.md) and [run instructions](docs/titania-workflow.md). Material chemistry remains an explicitly authored endpoint fixture.

## Setup

```bash
conda env create -f environment.yml
conda activate osi-hack
python tools/fetch_device_runtime.py --download
python tools/fetch_device_runtime.py
python -m pytest -q
```

The download script uses only the standard library. It restores the exact upstream files specified by the committed manifests and verifies SHA-256 hashes. The approximately 221 MB upstream cache, model weights, API credentials and generated experiment traces are excluded from Git.

## Run without external API calls

```bash
# Run all registered device tasks with explicitly labeled scripted specialists.
python -m bench.device_tasks --output results/my-device-offline

# Run one device task.
python -m bench.device_tasks --device mt_xpr204e --output results/my-balance-test

# Start the original drying demonstration.
python -m app.demo --port 8793 --output results/my-drying-demo
```

Open `http://127.0.0.1:8793` for the drying UI. Device tasks currently use the CLI. Defaults use scripted specialist test doubles and rules routing; they make no cloud requests.

## Run real models

Deploy the local router using [LOCAL_OPENJEV.md](LOCAL_OPENJEV.md), then add `--router openjev` to a device command. OpenJev is the router, not the general expert.

Set `SILICONFLOW_API_KEY` in your environment, then explicitly request a real expert run:

```bash
python -m bench.device_tasks --backend deepseek --router openjev \
  --device mt_xpr204e --output results/my-balance-model-run
```

This sends billable requests to the configured `deepseek-ai/DeepSeek-V4.1-Flash` model. Device debugging shares a durable 5,000,000-token reservation ledger across output directories. Do not remove the ledger to restart the allowance. Free OpenRouter models are restricted to small, explicit debugging requests; free batch evaluation is disabled.

## Verified snapshot

| Check | Result |
|---|---:|
| Native session inventory | 198 of 203 catalog entries |
| Registered device control tasks | 194 |
| Offline LangGraph device regression | 194/194 passed |
| Real DeepSeek sample across 16 categories | 20/20 final outcomes passed |
| Real DeepSeek + local OpenJev device runs | 2/2 passed, no routing fallback |
| Project tests | 73 passed |
| Device debugging API usage, including failed pilots and repairs | 311,022 tokens |

The first real device pilot passed 0/3; the second passed 2/3. Their failures were retained. After fixing stale observation handoffs and returning fresh native readbacks after commands, the declared 20-device sample passed. These development results are not a statistical benchmark or evidence of routing superiority. See [the checked-in evidence summary](docs/evaluation-summary.md).

## Documentation

- [Completed and remaining checklist](CHECKLIST.md)
- [Current progress against milestones](PROGRESS.md)
- [Device workflow, scope and budget](DEVICE_WORKFLOW.md)
- [Evaluation and result interpretation](EXPERIMENTS.md)
- [Five-minute drying demo](DEMO.md)
- [Error annotation codebook](ERROR_CODEBOOK.md)
- [Configuration guide](configs/README.md)
- [Full proposal](materials_lab_agent_proposal.md)
- [Original milestone plan](materials_lab_agent_proposal_previous_milestones.md)

## Boundaries and attribution

The drying workflow uses the native ChemLab BINDER FD56 process runtime; its balance offset, cleaning and observation layer are authored fixtures. The device expansion uses native agent sessions and a task-specific capability subset. Of the 194 device tasks, 116 test power control; the remaining 78 test other controls/readbacks. There is no shared sample transfer, unified multi-device clock, validated hardware/robot behavior or complete synthesis/characterization workflow.

ChemLab code is pinned to `123cf78593469871e41b8dbd63a11dfb7c5ca251`; data is pinned to `4ca4a42dab2ca0a884e449d202e9ad96625d5648`. Upstream code and data have separate licenses: retain [LICENSE](vendor/chemlab/LICENSE), [LICENSE-ASSETS.md](vendor/chemlab/LICENSE-ASSETS.md) and [NOTICE.md](vendor/chemlab/NOTICE.md). The manifests record 570 verified upstream files. No 3D geometry is required for the current tasks.
