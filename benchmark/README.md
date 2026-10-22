# Instrument evaluation

This directory contains the datasets, inference adapters, study runners and
reporting tools used for the instrument evaluations. Published summaries live in
[`results/388/`](../results/388/README.md) and
[`results/120/`](../results/120/README.md).

| Package | Purpose | Start here |
| --- | --- | --- |
| `instrument/` | Frozen 388-question MCQA dataset, 120 interactive cases, native runtime and graders | [Dataset and scorer commands](instrument/README.md) |
| `comparison/` | Local HF/Ollama inference, native Jev/APUS heads, feature baselines and static studies | [Model setup and MCQA studies](comparison/README.md) |
| `graph/` | Expert × chooser combinations through the application graph, in autonomous or teacher-forced mode | [Workflow preparation, arrays and reporting](graph/README.md) |
| `evaluation/` | Saved-result exports, calibration/resource metrics and final workflow audits | [Reporting and reproduction](evaluation/README.md) |
| `review/` | Blinded human-review interface and rating template | [Review instructions](review/README.md) |

The static benchmark scores answer selection. The sequential benchmark scores
native actions and outcomes across interactive episodes. Teacher forcing grades
each original action before replacing an incorrect proposal with an accepted
reference action; completion on that trajectory is assisted completion.

The published static results contain **34 configurations**, scored on **114
held-out questions** from the 388-question corpus. The final sequential study
contains **22 configurations × 120 cases**, seed 0, teacher forcing: **2,640
closed records**, **1,344 assisted goals (50.91%)**, and **5,477 / 12,466 correct
original native proposals (43.94%)**. Routing and internal schema checks remain
unassisted. Read the result READMEs for denominators, aborts and coverage.

## Running evaluations

On SCITAS, run setup, Python, tests, simulation, inference and report generation
in real Slurm compute allocations. Use login nodes for editing, submission,
scheduler queries and log inspection. Cluster account/partition defaults in the
templates can be overridden with `sbatch` options.

1. Validate the frozen dataset with the [instrument setup/check jobs](instrument/README.md).
2. Stage model checkpoints separately and install the appropriate
   [comparison environments](comparison/README.md). Native heads have their own
   dependency requirements; model weights are not included in Git.
3. For static inference, use the comparison runner or its frozen study matrix.
   For interactive episodes, use the [graph runner](graph/README.md), which
   freezes source, configuration and case identities before submitting workers.
4. Export saved predictions and episodes using [evaluation/](evaluation/README.md).
   Write new runs under ignored `results/` directories, outside source packages.

Portable recipes cover the [two used static study plans](evaluation/configs/static-388/mcqa-used-profiles.md)
and the [final 22-cell teacher-forced grid](evaluation/configs/workflow-e-plan.json).
Paths and environments must be provided for the receiving cluster. A new source
snapshot is a new run; the historical frozen study and published JSONs retain
their original identities.

The combined check also runs the application's full regression suite, which
requires its pinned native runtime under `vendor/chemlab/`. Restore that cache
using `tools/fetch_device_runtime.py --download` inside a compute allocation if
it is missing; the eight-device benchmark bundle is a separate runtime.

From the repository root, check all packages using the graph/features environment
(the check job performs no model inference or downloads):

```bash
export BENCHMARK_ROOT="$PWD"
export EVALUATION_PYTHON=/absolute/path/to/graph-environment/bin/python
sbatch benchmark/slurm/check.sbatch
```

The job writes per-suite logs, JUnit reports and a source-bound receipt under
`results/benchmark-check-JOBID/`. The optional `GRAPH_QWEN25_14B_MODEL` variable
enables a checkpoint-configuration check without loading weights.

## Methods and provenance

- [Dataset construction and scoring](docs/methodology.md): authored scenarios,
  native reference labels, splits, L1–L5 categories, metrics and graph adapters.
- [Published 388 evaluation](../results/388/README.md) and
  [published 120 evaluation](../results/120/README.md): exact protocols and fields.
- [Study history](docs/study-history.md): dated pilots, scheduler recovery and
  final closure. Its local raw-evidence paths are not distributed in Git.
- [Historical installation evidence](docs/provenance/README.md): receipts for
  earlier validated source snapshots, distinct from checks on the current tree.
- [Cleanup validation](docs/provenance/cleanup-validation.md): 796 passed tests,
  portable run-profile checks and result-export replay evidence.
- [Research plan](PLAN.md): design targets, not evidence of completed results.

The graph study uses the two upstream application fixes from `development`
through `e116c6b`: rounded hosted-Jev scores and order-insensitive readback
presets. The benchmark-specific behavior lives in adapters under `graph/` and
`comparison/`; the manifest-protected instrument release stays frozen.

See [evaluation validation](evaluation/README.md) for package check commands.
Preserve the instrument [licenses](instrument/LICENSE),
[asset terms](instrument/LICENSE-ASSETS.md) and [notices](instrument/NOTICE.md)
when redistributing the benchmark. Independent expert validity review is pending.
