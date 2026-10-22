# Instrument benchmark packages

| Package | Purpose |
|---|---|
| [instrument](instrument/README.md) | 388 MCQA questions, 120 sequential tasks, native runtime and graders |
| [comparison](comparison/README.md) | Local models, decision heads, feature readouts and studies |
| [graph](graph/README.md) | Shared LangGraph execution with free-run or teacher-forced feedback |
| [evaluation](evaluation/README.md) | Explicit-input exports, metrics and workflow audits |
| [review](review/README.md) | Browser-based human review tools |

The complete code from PR #1 is retained. Model weights, fitted artifacts and recorded runs are supplied separately. Local-model entry points require real Slurm allocations; the hosted `bench.instrument_eval_langgraph` runner has an explicit local-CPU mode. See the root [benchmark guide](../docs/benchmarking.md).
