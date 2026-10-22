# Benchmarking

Run hosted commands from the repository root. Use a fresh output directory for each experiment; keep the exact model, reasoning effort, dataset inventory and execution mode with the result.

## Hosted sequential tasks

`bench.instrument_eval_langgraph` preserves one graph checkpoint per task. It yields a validated action to the benchmark, then resumes from actual simulator feedback. `free_run`, `teacher_forced` and `both` are supported. Reference corrections appear only after the original action has been scored.

```bash
python -m bench.instrument_eval_langgraph \
  --bundle benchmark/instrument --execution local-cpu \
  --mode teacher_forced --case-id 769124c42b5476d2af84eb6b \
  --expert-config configs/openrouter-deepseek-v4.1-flash-benchmark.json \
  --decision-model openai/gpt-6-luna-decisions \
  --expert-timeout 900 --output results/my-case
```

The local-CPU entry point verifies and copies the release, enables the local branch in two scheduler guards, and checks that the remaining evaluator code and data are unchanged. It does not impersonate a Slurm job.

For the full 120-task free-run cohort in isolated workers:

```bash
python -m bench.instrument_eval_batch --bundle benchmark/instrument \
  --expert-config configs/openrouter-minimax-m3.1-flash-max.json \
  --decision-model typesafe/jev-1.14 --workers 8 \
  --output results/my-batch
```

The batch runner freezes its source and records all terminal cases, including failures. Its shared request-start interval is 3.4 seconds. The single-run CLI defaults to no pacing; a positive `--rate-interval` requires `--rate-state`.

## Hosted MCQA

```bash
# Decision model alone, unchanged original answer options.
python -m bench.instrument_mcqa_jev all --bundle benchmark/instrument \
  --workers 4 --output results/my-mcqa-jev

# General expert answers directly.
python -m bench.instrument_mcqa_minimax --bundle benchmark/instrument \
  --config configs/openrouter-minimax-m3.1-flash-low.json \
  --workers 4 --output results/my-mcqa-direct

# General expert annotates options, then Jev selects an original option.
python -m bench.instrument_mcqa_assisted --bundle benchmark/instrument \
  --config configs/openrouter-minimax-m3.1-flash-low.json \
  --workers 4 --output results/my-mcqa-assisted
```

The two expert entry points also accept the supported DeepSeek configuration. Names reflect their initial MiniMax implementation. Annotations preserve original option identity; the bundle's scorer evaluates the selected original option. Public prompts exclude answer keys.

## Local models and teacher forcing

David's [local graph guide](../benchmark/graph/README.md) documents environment setup, immutable preparation, explicit `--mode teacher_forced` and worker execution. [Comparison backends](../benchmark/comparison/README.md) provide HF, native decision heads, Ollama and frozen-feature methods. These runners require a genuine Slurm compute allocation and separately staged weights/artifacts.

`benchmark/evaluation/configs/workflow-e-plan.json` retains the complete 22-configuration recipe. This is a configuration template, not a claim that all its cells are suitable for inclusion in the final report. The two static plans under `benchmark/evaluation/configs/static-388/` are also retained.

## Configuration and results

Hosted calls use `OPENROUTER_API_KEY`. Decision choices are `typesafe/jev-1.14` and `openai/gpt-6-luna-decisions`. MiniMax low/max and DeepSeek low configurations are explicit; retained `:free` files request that exact endpoint with no paid fallback. Reasoning text is excluded while reported usage remains recorded. Never put keys in files.

MCQA accuracy, autonomous outcome success and assisted completion measure different things. Original-action accuracy in teacher-forced mode is computed before correction. Missing native proposals have no action-accuracy denominator. Attribute failures from concrete evidence: model output/call errors, interface defects and service failures are separate categories.

Outputs contain source/configuration manifests, native records, model traces and accounting. They are ignored by Git. [Result processing tools](../benchmark/evaluation/README.md) read supplied local evidence; no saved runs are bundled here.
