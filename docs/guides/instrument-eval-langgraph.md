# Instrument benchmark

The external benchmark bundle contains **388 MCQA questions** and **120 sequential tasks** across 14 workflow templates. The sequential adapter runs the repository LangGraph against the original evaluator's simulator and scoring logic.

## Retrieve the bundle

The report uses dataset commit `22a8192e4e40f3a7d37e1164a8755b006f932d60`. Export it without merging its result files into this branch:

```bash
git fetch origin instrument_eval
mkdir -p /tmp/osi-instrument-eval
git archive 22a8192e4e40f3a7d37e1164a8755b006f932d60 benchmark/instrument | tar -x -C /tmp/osi-instrument-eval
```

The runner verifies the release manifest, copies the bundle into the run directory and enables an explicit local-CPU branch in two scheduler guards. It checks that other runtime code and grading data remain unchanged.

## Stateful LangGraph evaluation

```bash
python -m bench.instrument_eval_langgraph \
  --bundle /tmp/osi-instrument-eval/benchmark/instrument \
  --execution local-cpu --mode free_run --all --continue-on-error \
  --expert-config configs/openrouter-deepseek-v4.1-flash-benchmark.json \
  --decision-model typesafe/jev-1.14 \
  --expert-timeout 900 --output results/my-workflow-eval
```

Use repeated `--case-id` arguments instead of `--all` for a subset. `--decision-model openai/gpt-6-luna-decisions` replaces both router and operation chooser. Expert configurations support explicit `low`, `medium` and `max` reasoning; the report compares low and max. Paid MiniMax configurations are `configs/openrouter-minimax-m3.1-flash-low.json` and `configs/openrouter-minimax-m3.1-flash-max.json`. The retained `:free` configurations request that exact endpoint and never fall back to paid service.

For isolated parallel workers over the complete cohort:

```bash
python -m bench.instrument_eval_batch \
  --bundle /tmp/osi-instrument-eval/benchmark/instrument \
  --workers 8 \
  --expert-config configs/openrouter-minimax-m3.1-flash-max.json \
  --decision-model openai/gpt-6-luna-decisions \
  --output results/my-parallel-eval
```

Use fresh output directories. The batch runner freezes its source and records configuration and worker outcomes. Its shared pacing interval is 3.4 seconds; the single-run CLI defaults to no pacing and accepts a positive `--rate-interval` together with `--rate-state` when needed. A provider-wide endpoint, quota or payment error stops new cases rather than changing model identity.

## Evaluation modes

| Mode | Model input and execution | Metric |
|---|---|---|
| MCQA | Public question and original immutable options | Correct original answer selections |
| `free_run` | Public state, menu, actual execution history and feedback | Autonomous outcome success |
| `teacher_forced` | Same decision interface, followed by reference correction when required | Original-action accuracy and assisted completion |

`--mode both` runs both sequential modes. Assisted completion is not autonomous success. The evaluator owns private case specifications, checkpoints and faults. The agent receives the public view and offered capabilities/presets, not reference actions in advance. Actual teacher-forced corrections are visible in subsequent execution feedback.

The graph persists across callbacks. It gates a proposal and yields at `execute`; the evaluator runs the native action, advances its own simulator and returns feedback to that same graph. No second instrument world is created. Finish claims are assessed independently by the evaluator.

## MCQA

```bash
python -m bench.instrument_mcqa_jev all \
  --bundle /tmp/osi-instrument-eval/benchmark/instrument \
  --output results/my-mcqa --workers 4
```

This uses unchanged original options and the bundle's scorer. `bench.instrument_mcqa_assisted` adds expert annotations before Jev selection, while `bench.instrument_mcqa_minimax` is the direct MiniMax preset. Their MiniMax presets retain the explicitly selected free endpoint; check configuration and availability before running them. `bench.instrument_open_question` provides a separate generated-answer experiment and is not the original-option MCQA score.

## Outputs

All detailed outputs remain under ignored `results/`: source/configuration manifests, model/native traces, episode records, scorer summaries and token accounting. Inspect failures before assigning cause. Malformed model outputs and calls violating supplied schemas are model errors; adapter defects and external service failures are separate categories. Keep unsuccessful cases in the denominator and distinguish missing coverage from attempted failures.

The [final report](../reports/instrument-evaluation-report.html) presents aggregate comparisons. Runtime models and dataset revisions should be recorded for every new experiment.
