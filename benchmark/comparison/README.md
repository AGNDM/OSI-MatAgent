# Local comparison backends

Inference adapters cover HF generative models, native OpenJev/APUS decision heads, Ollama and frozen-feature readouts. The sibling [instrument release](../instrument/README.md) owns questions, actions and grading. The shared application workflow is provided separately by [graph](../graph/README.md).

## Setup and execution

Run from `benchmark/` inside a genuine Slurm compute allocation. Supply complete local checkpoints and trusted fitted artifacts; these tools do not download model weights.

```bash
python3.12 -m comparison.setup --profile hf
python3.12 -m comparison.setup --profile features
python3.12 -m comparison.setup --profile apus
```

Use the created environment for the selected backend. Configure explicit paths in a copy of `comparison/configs/*.example.json` (or their documented environment bindings). Then:

```bash
python -m comparison.run --config comparison/configs/hf-qwen.example.json \
  --only qwen-hf-logprobs-seed0 --track mcqa --preflight --output output/preflight
```

Remove `--preflight` to execute the configured cohort; `--limit 1` runs a diagnostic prefix. MCQA defaults to 114 held-out questions. Use the explicit `all` split only when training/validation diagnostics are intended. Sequential mode is separately configured.

| Module | Function |
|---|---|
| `setup` | Create/verify environments and explicit Ollama installations |
| `run` | Preflight and inference |
| `features` | Fit/read frozen-feature artifacts |
| `study_matrix`, `study` | Construct, freeze and execute studies |
| `aggregate`, `taxonomy`, `study_report` | Process saved runs |
| `tool_track` | Separate read-only information track |

Use each module's `--help` for arguments. Keep output outside source packages. APUS has its own pinned interpreter; Ollama requires a separately managed loopback service and the pinned runtime in `ollama-runtime.lock.json`. Keep backend contexts, source identities and missing/failed cases in saved study metadata.
