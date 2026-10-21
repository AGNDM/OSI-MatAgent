# Instrument decision benchmark

A portable distribution of two benchmarks authored from ChemLab instrument contracts and native simulations. Copy this entire folder; it does not depend on the original experiment checkout.

| Track | Task | Guide |
| --- | --- | --- |
| **388 MCQA questions** | Select a supplied action or operating plan from 2–4 choices. | [MCQA guide](mcqa_388/README.md) |
| **120 sequential cases** | Choose an instrument and successive actions, receiving simulator feedback after each action. | [Sequential guide](sequential_120/README.md) |

The sequential track contains 14 workflow templates, eight instruments, 146 public commands and 1,954 finite argument presets. Teacher forcing scores the original proposal, corrects wrong actions and continues; free execution measures autonomous performance. These are authored synthetic scenarios. Independent expert review remains pending.

## Get started on SCITAS Kuma

Run these submission commands from this folder. All installation, Python commands, tests and simulation run on compute nodes through Slurm.

```bash
mkdir -p output
sbatch slurm/setup.sbatch
```

Wait for setup to finish successfully, then submit the integrity checks and a native reference replay:

```bash
sbatch slurm/check.sbatch
sbatch slurm/sequential.sbatch --policy reference --mode both
```

Setup creates an ignored local `.venv` using Python 3.12 and pinned native dependencies. It needs access to the package index or a configured local package mirror. Logs and run outputs go under `output/`. The launchers use this local environment by default; an absolute `BENCHMARK_PYTHON` can select an existing compatible environment. Slurm defaults use account `liac` and partition `h100`; colleagues with a different allocation should override these through `sbatch`.

The reference replay is an evaluator check, not a model result. Every evaluation needs a fresh output directory. See [examples](examples/README.md) to implement a policy. To prepare the optional local Qwen environment, submit `sbatch slurm/setup.sbatch --model`; this creates `.venv-models`. Supply model weights separately and request GPU resources for inference. The [sequential guide](sequential_120/README.md) gives the launch command.

## Export and score MCQA

The following are job payloads: run them inside a genuine Slurm compute allocation on SCITAS, using the interpreter created by setup.

```bash
.venv/bin/python eval/prompts.py --split heldout --output output/prompts.jsonl
.venv/bin/python eval/mcqa.py --predictions my-predictions.jsonl --split heldout --output output/scores.json
```

Give models only the exported `prompt`, retaining `task_id` for matching predictions. Predictions are JSON lines such as `{"task_id":"...","selected_option_id":"B"}`. Keep answer keys and sequential evaluator specifications out of model inputs.

## Folder layout

```text
mcqa_388/        Frozen questions, answer keys, splits and provenance
sequential_120/ Public cases, action presets, manuals and evaluator fixtures
  runtime/      Frozen native simulator and dependency locks
  provenance/   Original execution and context-validation receipts
eval/           Prompt export, interactive evaluation, scoring and validation
examples/       Prediction examples and policy adapters
tests/          Evaluator and release-integrity checks
slurm/          Environment setup, verification and sequential launchers
tools/          Portable environment and release-maintenance utilities
manifest.json   Current version and checksums of published files
validation.json Recorded package verification evidence
```

Local environments, caches and `output/` are excluded from the release inventory. Copy the source folder without those generated directories when sharing it. Native runtime code, selected assets and licences are required; no model weights, 3D meshes or live experiment outputs are shipped.

## Reproducibility and interpretation

`slurm/check.sbatch` verifies every published checksum, dataset consistency and unit tests. The recorded [validation receipt](validation.json) also describes isolated-copy and native replay checks. See [maintenance instructions](tools/README.md) before changing release files; the maintenance utility protects dataset, answer-key, split, core-grader and vendored-runtime bytes.

The 388-question track has 114 held-out questions. Full-corpus accuracy includes training and validation diagnostics. The sequential track has no assigned train/test split; preserve its workflow grouping when creating one. Proposed-action accuracy, accuracy excluding informational queries, assisted completion and autonomous success are distinct metrics. Saved-episode scoring aggregates trusted engine records; native replay establishes action outcomes. Detailed metrics, baseline protocol and limitations are in each track's guide.

## Attribution

Keep `LICENSE`, `LICENSE-ASSETS.md`, `NOTICE.md`, `CITATION.cff` and upstream runtime notices with every copy. Python utilities use MIT terms; ChemLab-derived assets have CC BY-NC 4.0 and third-party terms. The included citation identifies upstream ChemLab, not a separate benchmark paper. Manifests pin source revisions. This project authored the benchmark scenarios and grading criteria; simulation does not establish real-equipment reliability or chemical outcomes.
