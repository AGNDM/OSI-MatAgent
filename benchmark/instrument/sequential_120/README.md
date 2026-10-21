# Sequential instrument benchmark: 120 cases

This track evaluates **one action at a time**, with a persistent goal and native simulator feedback. It contains 120 authored parameterized cases across 14 workflow templates and eight instruments. These are synthetic workflows constructed from ChemLab's public instrument contracts and native runtimes, rather than pre-existing ChemLab benchmark episodes or recorded laboratory failures.

## Files

| Path | Content |
| --- | --- |
| `cases.jsonl` | 120 public initial contexts and opaque task IDs. |
| `action-presets.json` | The fixed shared bank of 1,954 schema-valid presets for all 146 public commands. |
| `public-contracts.json` | All eight public static instrument manuals, keyed by their original hashes. |
| `evaluator/case-specs.jsonl` | Trusted evaluator fixtures, checkpoints, accepted actions, timing, fault schedules, goals and budgets. Keep this file out of model inputs. |
| `runtime/` | Frozen evaluator/adapter sources, selected native assets, dependency instructions and copy provenance. |
| `provenance/` | Original case/menu/source fingerprints, native execution evidence and context-validation receipts. |
| `manifest.json` | Dataset inventory, grouping policy and file hashes. |
| `../eval/sequential.py` | Interactive runner and scorer for saved trusted engine episodes. |
| `../eval/validate_sequential.py` | Static consistency and provenance checks, also run by the parent validator. |

Public contexts are preserved byte for byte from validated export **4632791**. Evaluator specifications are shipped separately because adaptive scoring needs state, prerequisites and accepted alternatives; a list of fixed next-action answer strings would be insufficient. Pass only the policy's public context and offered menu to the model.

## Interaction

The first choice selects one of eight instruments. Thereafter every implemented public command for the selected machine remains available, including commands with unmet operating prerequisites. The model can also read public observations, wait, inspect the manual, switch machines, declare completion or escalate a fault. Hidden truth access and evaluator fault injection are excluded.

Parameterized commands offer finite shared presets. The bank is compiled once across the corpus, with representative public-schema values; it is not filtered using the current private case or correct checkpoint. It does not enumerate every possible number or string. A policy may select a menu index for the command and its parameters or return an offered fully specified native action.

Each native action is scored once. A choice of machine, command and parameters may involve multiple menu selections, which do not count as separate scored native actions. In **teacher-forced** mode, the evaluator records the original proposal, replaces an incorrect proposal with an offered acceptable action and runs it. The model receives the executed action, correction flag and native response before its next decision. In **free-run** mode, its proposed action is executed without correction. State persists across actions and machine switches. Waiting advances simulated time; new measurements require a public read or a command that actually returns observations.

The included Qwen adapter preserves the validated v6 rendering: selected-machine manual, complete temporal history, readable reading and parameter tables, and references to repeated public evidence. Backend integrity checksums remain in the journals. The model chooses finite labels and does not generate command JSON. Other policies must record their own observation, menu and decoding protocol for meaningful comparisons.

## Run and score

From the distribution root, prepare the local environment with `sbatch slurm/setup.sbatch`, then submit a reference replay:

```bash
mkdir -p output
sbatch slurm/sequential.sbatch --policy reference --mode both
```

The launcher uses `.venv/bin/python` and a fresh job-specific output directory. These direct commands are compute-job payloads; run them inside a genuine Slurm allocation on SCITAS:

```bash
# Deterministic privileged reference: checks execution, not model performance.
.venv/bin/python eval/sequential.py run --policy reference --mode both --output output/sequential-reference

# Replace the demonstration policy with your model adapter.
.venv/bin/python eval/sequential.py run --policy examples.sequential_policy:factory --mode both --output output/sequential-model

# Aggregate trusted episode records produced by the native runner.
.venv/bin/python eval/sequential.py score --episodes output/sequential-model/episodes.jsonl
```

See the CLI help for case limits used in smoke checks. A limited run reports its selected cohort and explicit `diagnostic_subset` and `complete_full_dataset` flags. Scoring that file with the default scorer retains all 120 cases in each mode's goal-success denominator and reports missing episodes and observed-step coverage. It does not invent decisions for unvisited steps. The shipped example policy deliberately chooses the first offered option; its purpose is to demonstrate the callback API.

The optional `examples.sequential_qwen:factory` adapter uses `SEQUENTIAL_MODEL_PATH` to locate separately staged local weights. Model weights and inference dependencies are not required for static validation, static scoring or native reference replay.

For the optional Qwen adapter, first submit `sbatch slurm/setup.sbatch --model` and wait for successful completion. Then supply your staged weights and request a GPU:

```bash
SEQUENTIAL_MODEL_PATH=/absolute/path/to/local/model \
BENCHMARK_PYTHON="$PWD/.venv-models/bin/python" \
  sbatch --qos=normal --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=04:00:00 \
  slurm/sequential.sbatch --policy examples.sequential_qwen:factory --mode both
```

The optional model lock targets Python 3.12, Linux x86_64 and CUDA 12.6. The launcher defaults to CPU resources. Adjust the GPU request and wall time for your model and allocation; record these settings with results. Weights are not included or downloaded by the adapter.

## Organization and metrics

| Instrument | Cases |
| --- | ---: |
| Binder FD 56 oven | 18 |
| Eppendorf CellXpert C170i incubator | 6 |
| Esco fume hood | 18 |
| IKA RCT digital hotplate | 18 |
| Nabertherm L9 furnace | 18 |
| New Era NE-1000 syringe pump | 24 |
| Peak Precision Hydrogen generator | 6 |
| Vaisala DMT152 dew-point transmitter | 12 |

The 14 workflows cover furnace programming and interruption recovery; oven USB recording and standby restoration; hotplate stirring and motor diagnosis; pump delivery and timed monitoring; hood illumination and sash access; dew-point startup/diagnosis and offline calibration; hydrogen startup/diagnosis; and incubator sensor-cover access. There are 27 matched healthy/failure pairs. Cases include healthy controls and different setup conditions, so 120 cases do not mean 120 distinct failure mechanisms.

Report proposed-action accuracy, control-action accuracy excluding auxiliary queries, corrected completion and autonomous success separately. Aggregate breakdowns by workflow, instrument, condition and operational stage help identify where a policy fails. A corrected trajectory reaching the goal is assisted completion. Saved-episode scoring trusts the native engine's grading and only validates/reaggregates its records; arbitrary reported successes require native replay to establish correctness.

No train/test split has been assigned. Preserve `expansion.split_group` when creating a future holdout so that variants and matched conditions of the same workflow remain together. The five legacy L1–L5 construction categories of the 388-question dataset are not assigned to these sequential cases, and an increasing difficulty order has not been established.

## Evidence and limits

The original native validation checked 480 diagnostic episodes. Later job **4632791** verified unchanged runtime/menu/case fingerprints before reusing that evidence, then passed 60 tests, all 120 reference trajectories, every tested command path and the documented long-history context stresses. Its maximum tested prompt was 26,621 of 30,000 tokens. Those receipts are supplied in `provenance/`; they do not establish model accuracy or exhaustively test every possible future history. The package's own validation receipt records isolated-copy checks and fresh native replay separately.

Independent domain-expert review remains pending. Accepted action credit uses an authored partial-order curriculum, which can omit otherwise valid procedures. Instruments run as independent native simulators; this track does not model cross-instrument chemical/material transfers or verify real hardware reliability. The parent `LICENSE`, `LICENSE-ASSETS.md`, `NOTICE.md` and `CITATION.cff` apply, with the upstream notices also preserved alongside the vendored native files.
