# Local models through the application graph

This package evaluates local expert and chooser backends through the repository's
`app.graph.build_graph`. It supplies a public benchmark adapter, model transports,
preparation tools and read-only reports. It does not vendor a second application,
train models, or provide a hosted inference backend. Keep `app/` at repository
root and the sibling `benchmark/instrument/` and `benchmark/comparison/` packages.

The model proposes alternatives, the chooser selects a workflow route and an
operation, and the graph reaches its typed action gate before native execution.
A saved native grade remains the authority for action correctness and outcomes.
The same graph source can run different injected profiles and transports; that
is not an identical end-to-end pipeline to the instrument CLI.

| Location | Purpose |
| --- | --- |
| `graph_local/` | Local specialist/chooser transports and native handoff adapter |
| `graph_local/reporting/` | Reusable matrix and workflow-reachability readers |
| `tools/` | Small command-line entry points |
| `slurm/` | Supported preparation, execution, validation and report launchers |
| `tests/` | Runtime, portability and reporter tests |
| `examples/` | Editable input plans, not completed experiments |
| [Protocol history](docs/protocol-history.md) | Earlier pilots, completed E profile and study provenance |
| [Reporting provenance](docs/provenance/README.md) | Original validation receipts and archived validation inputs |

## Completed study E and new runs

E is a completed historical study: **22 combinations × 120 cases, model seed 0,
`teacher_forced` only**. Its frozen source used `menu-ids-v3`,
`decision-view-v4`, guided JSON and an explicitly extended Qwen2.5-14B profile.
It produced 2,640 closed records: 1,344 assisted goals and 1,296 model-output or
budget aborts. The [full result](../../results/120/workflow-teacher-forced-results.json),
[compact summary](../../results/120/workflow-teacher-forced-status.json) and
[final audit](../../results/120/workflow-teacher-forced-final-audit.json) contain
coverage, source identity and original-proposal metrics. Historical details and
comparison limits are in the [study notes](docs/protocol-history.md).

The commands below prepare a **new** study. The live package retains its original
unguided interface by default; optional transports must be explicit in the plan.
The shipped `matrix.json` and `stronger-matrix.json` are 2 × 9 input templates,
not aliases for E's 22-cell configuration. `provider-design.json` is a research
outline; its hosted/MiniMax placeholders are not executable local providers.
Never modify an existing study snapshot to adopt a newer package version.

## Environment and checks

On SCITAS, installation, imports, tests, hashing, simulation, inference and report
rendering all belong in a genuine Slurm allocation. Login-node editing, plain
file inspection and scheduler commands are sufficient to prepare the commands.
The supplied account and partition defaults are site examples; override them
with `sbatch` flags on another cluster. Normal SCITAS `liac` jobs require a GPU
allocation even for preparation or reporting.

From the repository root, replace these absolute placeholders with your staged
paths. No launcher assumes a particular user's checkout or cache directory.

```bash
export GRAPH_PACKAGE="$PWD/benchmark/graph"
export GRAPH_BASE_PYTHON=/absolute/path/to/model-environment/bin/python
export GRAPH_ENVIRONMENT=/absolute/path/to/fresh-graph-environment
export GRAPH_PYTHON="$GRAPH_ENVIRONMENT/bin/python"
export GRAPH_ENVIRONMENT_RECEIPT=/absolute/path/to/graph-environment-receipt.json
export GRAPH_STUDY=/absolute/path/to/fresh-graph-study
export MODEL_ROOT=/absolute/path/to/staged-models
export ARTIFACT_ROOT=/absolute/path/to/fitted-feature-artifacts
export APUS_PYTHON=/absolute/path/to/apus-environment/bin/python
```

Reuse an already validated graph environment/receipt, or create a fresh isolated
layer. Setup never installs into the existing model environment and refuses an
existing destination. `uv` must be on `PATH`, or pass its absolute path with
`--uv`. APUS requires its separately pinned Transformers 5.16.1 interpreter;
the graph expert environment uses the recorded 4.57.1 model stack.

```bash
sbatch "$GRAPH_PACKAGE/slurm/setup.sbatch" \
  --environment "$GRAPH_ENVIRONMENT" \
  --receipt "$GRAPH_ENVIRONMENT_RECEIPT" \
  --app-source "$PWD" --comparison-source "$PWD/benchmark"
```

Run the complete package checks without loading model weights. Native integration
fixtures are explicitly privileged offline diagnostics and do not count as model
performance. An optional test reads the released Qwen configuration only; set
`GRAPH_QWEN25_14B_MODEL="$MODEL_ROOT/qwen2.5-14b"` if it is staged. Otherwise
that one test skips with setup guidance.

```bash
sbatch "$GRAPH_PACKAGE/slurm/check.sbatch" \
  --app-source "$PWD" --comparison-source "$PWD/benchmark" \
  --junitxml /absolute/path/to/checks/graph.xml

# Reporter checks alone require no native benchmark bootstrap.
sbatch "$GRAPH_PACKAGE/slurm/check-reporting.sbatch" \
  --junitxml /absolute/path/to/checks/reporting.xml
```

## Prepare and execute a new study

Plans accept an explicit `experiments` list or an `experts` × `choosers` matrix.
Relative model/artifact paths resolve against the JSON file. `{MODEL_ROOT}` and
`{ARTIFACT_ROOT}` expand only from explicit CLI arguments; shell commands and
arbitrary environment expansion are not evaluated. Unsupported/hosted provider
kinds are rejected, and all model files must already exist locally.

The opt-in guided interface uses these fields:

| Specification | Explicit fields |
| --- | --- |
| Expert | `record_protocol: "menu-ids-v3"`, `structured_output: "json_schema_lmfe_v4"` |
| Chooser | `context_protocol: "decision-view-v4"`, an explicit `context_max_bytes` |
| Qwen2.5-14B extension, when desired | `context_extension: "qwen25-yarn-v1"`, `max_context: 65536` |

Guided experts propose IDs from the complete currently offered public menu; the
adapter binds those IDs to exact actions and preserves their arguments. The
procedure envelope copies the selected action without another model call.
Diagnostic/verification citations are restricted to public wrapper IDs, and
bounded verifier prose prevents repeated output loops. Readable grouped menus
and exact duplicate references reduce repeated text. The decision view omits
older/unrelated public context, so it is not lossless; complete inputs remain in
the trace. No private evaluator answer or ranking is supplied to a model.

These transports require `lm-format-enforcer==0.11.3` and `interegular==0.3.3` in
the validated interpreter. Default unguided runs do not import them. YaRN changes
the loaded Qwen positional configuration, not checkpoint weights; it is not a
training step. Different transport/context/model profiles require separate
frozen studies and separate interpretation. See the [protocol details](docs/protocol-history.md).

For the original input template, explicitly select the evaluation mode:

```bash
sbatch "$GRAPH_PACKAGE/slurm/prepare.sbatch" \
  --app-source "$PWD" --comparison-source "$PWD/benchmark" \
  --environment-receipt "$GRAPH_ENVIRONMENT_RECEIPT" \
  --plan "$GRAPH_PACKAGE/examples/matrix.json" \
  --model-root "$MODEL_ROOT" --artifact-root "$ARTIFACT_ROOT" \
  --apus-python "$APUS_PYTHON" \
  --mode both --shards 12 --output "$GRAPH_STUDY"
```

Use `--mode free_run` for autonomous execution or `--mode teacher_forced` for an
assisted diagnostic. An optional `--app-revision` checks the exact source commit.
Preparation hashes all copied files, snapshots the chosen application, adapter,
comparison code, original instrument bundle, input plan and environment receipt,
and submits no jobs. Existing output directories are refused.

Inspect `prepared.json` and `jobs.json` before submission. The unchanged 2 × 9
input template with 12 shards has 216 workers, hence indices `0-215`; a different
plan or shard count requires its own declared range. Each worker owns disjoint
case IDs and preserves the full 120-case denominator.

```bash
mkdir -p "$GRAPH_STUDY/logs"
export GRAPH_PACKAGE="$GRAPH_STUDY/snapshot/package"
sbatch --array=0-215%24 \
  --output="$GRAPH_STUDY/logs/graph-%A_%a.log" \
  "$GRAPH_PACKAGE/slurm/run.sbatch"
```

Workers authenticate frozen inputs and environment versions. Keep their source,
configuration and failed attempts unchanged. Record actual GPU/CPU allocations
for timing and budget comparisons; H100, L40S and MIG are different profiles.

## Grading and interpretation

Teacher forcing grades an original native proposal first, replaces an incorrect
action with an acceptable native reference, and returns the resulting feedback.
It does not correct graph routing or rescue records that fail before a native
proposal. Repeated diagnosis can exhaust the budget without a correctable
proposal. Specialist/schema/model failures and valid native budget aborts remain
closed results; missing workers remain missing.

Report original-proposal accuracy, reference-correction counts/rates and native
decision coverage alongside all 120 cases per cell. Assisted goal completion is
separate from autonomous success. Native route probabilities have no route
oracle here; graph ECE/Brier calibration is not claimed. Supporting labeled MCQA
analyses belong to `benchmark/comparison`.

## Reports and regeneration

Build the comparison summary from the full frozen plan and raw runs first:

```bash
sbatch "$GRAPH_PACKAGE/slurm/aggregate.sbatch" \
  --bundle "$GRAPH_STUDY/snapshot/instrument" \
  --runs "$GRAPH_STUDY/runs" --plan "$GRAPH_STUDY/configuration.json" \
  --output "$GRAPH_STUDY/reports/combined" \
  --bootstrap 1000 --permutations 10000

sbatch "$GRAPH_PACKAGE/slurm/report.sbatch" \
  --study "$GRAPH_STUDY" \
  --combined-report "$GRAPH_STUDY/reports/combined/report.json" \
  --output "$GRAPH_STUDY/reports/matrix"
```

The matrix wrapper also writes `matrix/reachability/`. The appendix separates
router calls, operation choices, nontrivial candidate menus, procedure/gate
progress and native grading. It reads existing evidence and awards no new grade.
For a separate appendix destination:

```bash
sbatch "$GRAPH_PACKAGE/slurm/reachability.sbatch" \
  --matrix "$GRAPH_STUDY/reports/matrix/matrix.json" \
  --output "$GRAPH_STUDY/reports/reachability"
```

Reusable implementations live in `graph_local/reporting/`; their tests live in
`tests/`. Old Python and positional Slurm entry points under `tools/reporting/`
forward to the packaged implementations. Original validation receipts and the
original machine-specific launchers/tests are preserved as evidence in
[docs/provenance](docs/provenance/README.md), not as active execution instructions.

Multiple `--study` arguments may combine compatible frozen studies. The wrapper
checks matching profile, policy, backend and application graph bytes before
pooling. Different protocols require separate primary reports; calling the
matrix module directly does not add that wrapper guard. Retain missing cases
and model/interface aborts in cohort denominators and state exclusions for
completed-only metrics. Independent human validity review remains outstanding.

`tools/regenerate_mcqa.py` authenticates the original bundle/generator bytes and
calls the original generator without requiring Git. Run it inside Slurm with
`--bundle <benchmark/instrument> --output <fresh-directory>`. It preserves private
oracle file mode 0600 and never overwrites the canonical instrument release.
