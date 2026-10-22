# Local models through the application graph

This package evaluates local expert and chooser backends through the repository's
`app.graph.build_graph`. It provides a public benchmark adapter, local model
transports, immutable study preparation and saved-result reports. Keep `app/` at
repository root and the sibling `benchmark/instrument/` and
`benchmark/comparison/` packages available. Models and fitted feature artifacts
must already exist locally.

| Location | Purpose |
| --- | --- |
| `graph_local/` | Specialist/chooser transports and native handoff adapter |
| `graph_local/reporting/` | Matrix and workflow-reachability readers |
| `tools/` | Direct Python command-line entry points |
| `examples/` | Editable input plans |
| [Protocol history](docs/protocol-history.md) | Earlier pilots and completed study E |
| [Reporting provenance](docs/provenance/README.md) | Historical qualification receipts |

The model proposes alternatives, the chooser selects a workflow route and an
operation, and the graph reaches its typed action gate before native execution.
The native grade determines action correctness and outcomes. Injected profiles
and transports change the end-to-end pipeline despite using the same graph.

## Completed study and new execution

Historical study E used **22 combinations × 120 cases, seed 0,
`teacher_forced` only**, with `menu-ids-v3`, `decision-view-v4`, guided JSON and
the explicit Qwen2.5-14B YaRN profile. All 2,640 records closed: 1,344 assisted
goals and 1,296 model-output or budget aborts. See the
[full result](../../results/120/workflow-teacher-forced-results.json),
[status](../../results/120/workflow-teacher-forced-status.json) and
[final audit](../../results/120/workflow-teacher-forced-final-audit.json).

Commands below prepare a new study. The live default is the original unguided
interface; optional transports must be declared explicitly. The shipped
`matrix.json` and `stronger-matrix.json` are 2 × 9 templates, not E's 22-cell
configuration. `provider-design.json` is a research outline with unimplemented
hosted placeholders. Never change an existing study snapshot to adopt new code.

## Environment

Run installation, imports, hashing, simulation, inference and reporting inside
your own real Slurm compute allocation. Python entry points retain allocation
guards. This package includes no software tests or scheduler scripts; arrange
the GPU, CPU, memory and time allocation through your cluster's tooling.

From the repository root, set explicit paths inside that allocation:

```bash
export GRAPH_PACKAGE="$PWD/benchmark/graph"
export GRAPH_BASE_PYTHON=/absolute/path/to/model-environment/bin/python
export GRAPH_ENVIRONMENT=/absolute/path/to/fresh-graph-environment
export GRAPH_PYTHON="$GRAPH_ENVIRONMENT/bin/python"
export GRAPH_ENVIRONMENT_RECEIPT=/absolute/path/to/environment-receipt.json
export GRAPH_STUDY=/absolute/path/to/fresh-study
export MODEL_ROOT=/absolute/path/to/staged-models
export ARTIFACT_ROOT=/absolute/path/to/fitted-features
export APUS_PYTHON=/absolute/path/to/apus-environment/bin/python
```

Reuse a validated environment and receipt, or create a fresh isolated layer:

```bash
"$GRAPH_BASE_PYTHON" "$GRAPH_PACKAGE/tools/setup.py" \
  --base-python "$GRAPH_BASE_PYTHON" \
  --environment "$GRAPH_ENVIRONMENT" \
  --receipt "$GRAPH_ENVIRONMENT_RECEIPT" \
  --app-source "$PWD" --comparison-source "$PWD/benchmark"
```

Setup refuses existing destinations and leaves the base model environment
unchanged. `uv` must be available, or supplied through `--uv`. APUS uses its
separate pinned Transformers 5.16.1 interpreter; the expert environment uses
the recorded 4.57.1 stack. Setup authenticates dependencies and imports the
actual application graph without loading model weights.

## Prepare and run

Plans accept an explicit `experiments` list or an `experts` × `choosers` matrix.
Relative model/artifact paths resolve against the input JSON file.
`{MODEL_ROOT}` and `{ARTIFACT_ROOT}` expand only through explicit CLI arguments.
Unsupported or hosted backend kinds are rejected.

| Opt-in specification | Fields |
| --- | --- |
| Expert | `record_protocol: "menu-ids-v3"`, `structured_output: "json_schema_lmfe_v4"` |
| Chooser | `context_protocol: "decision-view-v4"`, explicit `context_max_bytes` |
| Qwen2.5-14B extension | `context_extension: "qwen25-yarn-v1"`, `max_context: 65536` |

Guided experts propose IDs from the complete offered public menu; the adapter
binds them to exact actions. The procedure envelope copies the selected action
without another model call. Public wrapper IDs constrain citations; bounded
verification prose and duplicate references reduce repeated text. The decision
view omits older or unrelated public context; complete inputs remain in traces.
Models receive no private evaluator answers or ranking.

Guidance requires `lm-format-enforcer==0.11.3` and `interegular==0.3.3`; unguided
runs do not import them. YaRN changes positional configuration, not checkpoint
weights. Keep different protocols and model/context profiles in separate studies.

```bash
PYTHONPATH="$GRAPH_PACKAGE" "$GRAPH_PYTHON" -m graph_local.prepare \
  --app-source "$PWD" --comparison-source "$PWD/benchmark" \
  --environment-receipt "$GRAPH_ENVIRONMENT_RECEIPT" \
  --plan "$GRAPH_PACKAGE/examples/matrix.json" \
  --model-root "$MODEL_ROOT" --artifact-root "$ARTIFACT_ROOT" \
  --apus-python "$APUS_PYTHON" \
  --mode teacher_forced --shards 12 --output "$GRAPH_STUDY"
```

Choose `free_run` for autonomous execution, `teacher_forced` for assisted
diagnostics, or `both` for separate trajectories in each mode. An optional
`--app-revision` requires an exact source commit. Preparation hashes the copied
application, adapter, comparison code, original bundle, plan and environment
receipt, submits no jobs, and refuses existing output directories.

Inspect `prepared.json` and `jobs.json`. Each entry owns disjoint case IDs.
Execute each declared index once, in an appropriate compute allocation, using
the frozen worker. For example, the first declared worker is:

```bash
"$GRAPH_PYTHON" "$GRAPH_STUDY/snapshot/package/tools/worker.py" \
  --study "$GRAPH_STUDY" --job-index 0
```

Repeat for the remaining declared indices using your allocation tooling, with
one worker per job to preserve distinct launch provenance. Index 0 alone does
not cover the full study. Workers authenticate inputs and package
versions. Preserve failed attempts, source/configuration and actual GPU/CPU
allocation metadata. H100, L40S and MIG timing results need separate hardware
interpretation.

## Grading

Teacher forcing grades the original native proposal first, replaces an
incorrect action with an acceptable native reference, and returns its feedback.
It does not correct routing or rescue records that fail before a native
proposal. Repeated diagnosis may exhaust a budget without a correctable action.
Closed model/schema/budget aborts remain results; missing workers remain missing.

Report original-proposal accuracy, reference-correction counts/rates and native
decision coverage over all 120 cases per cell. Assisted goal completion is
separate from autonomous success. Graph route probabilities have no route
oracle here; graph ECE/Brier calibration is not claimed. Labeled MCQA analyses
belong to `benchmark/comparison`. Independent human validity review is pending.

## Reports

First aggregate the full frozen plan and saved runs, then render the matrix:

```bash
PYTHONPATH="$GRAPH_STUDY/snapshot" "$GRAPH_PYTHON" -m comparison.study_report \
  --bundle "$GRAPH_STUDY/snapshot/instrument" \
  --runs "$GRAPH_STUDY/runs" --plan "$GRAPH_STUDY/configuration.json" \
  --output "$GRAPH_STUDY/reports/combined" \
  --bootstrap 1000 --permutations 10000

"$GRAPH_PYTHON" "$GRAPH_STUDY/snapshot/package/tools/report.py" \
  --study "$GRAPH_STUDY" \
  --combined-report "$GRAPH_STUDY/reports/combined/report.json" \
  --output "$GRAPH_STUDY/reports/matrix"
```

The wrapper also writes `matrix/reachability/`, separating routing, operation
choices, candidate menus, procedure/gate progression and native grading. It
reads existing evidence and awards no new grade. Multiple `--study` arguments
require matching profile, policy, backend and application graph bytes. Retain
missing cases and aborts in cohort denominators and disclose completed-only
metric exclusions. Legacy Python paths under `tools/reporting/` remain usable.

`tools/regenerate_mcqa.py --bundle <benchmark/instrument> --output <fresh-directory>`
authenticates the original generator/bundle and regenerates records inside a
compute allocation. It preserves private oracle mode 0600 and never overwrites
the canonical release.
