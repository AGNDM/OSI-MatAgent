# Development

The shared workflow is `app.graph.build_graph`. Hosted sequential evaluation uses `bench.instrument_eval_langgraph.LangGraphPolicy`; local evaluation uses `benchmark/graph/graph_local/policy.py`. Both retain a graph checkpoint and return native actions to the independent evaluator.

`app.lab_adapter` defines the structural interface and shared gate exception. It does not load a second simulator. The previous standalone drying adapter, demo server, synthesis macros and device-discovery cache are outside this benchmark release. Public data and task fixtures live in the self-contained `benchmark/instrument` release.

## Validate

```bash
python benchmark/instrument/eval/validate.py
python -m pytest -q
```

The tests exercise public contracts, hosted request configuration and both graph bridges without model calls. The release validator authenticates the distributed questions, answers, splits and evaluator/runtime inventory. Do not edit frozen dataset files to make a test pass.

Local graph preparation snapshots `app/`, `corpus/`, `configs/`, `cases/` and `vendor/`; the latter directories are retained for that stable interface. The actual instrument runtime is bundled under `benchmark/instrument/sequential_120/runtime/`.

## Contributions

Keep model credentials, checkpoints, fitted artifacts, logs, prediction records and run directories out of commits. Use `results/` for local evidence. Commit source, test inputs, configuration, concise documentation and final report assets only.

The initial mainline release preserves all 104 Python/HTML source files introduced by David's PR #1 byte-for-byte. Frozen dataset files are also unchanged. Source integrity, local tests and documentation links are checked before publication; model weights and a Slurm allocation are required for full local-model inference.

Package-specific dependencies and runtime/data licenses remain with their original components. Reporting utilities consume user-supplied local runs and retain the original scoring conventions.
