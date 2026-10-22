# Local models through LangGraph

The complete local-model adapter uses `app.graph.build_graph`. The evaluator executes native actions; the persistent graph resumes from acknowledged feedback. Teacher forcing grades each original proposal before applying a reference correction. It does not repair failures that occur before a native proposal.

The sibling `comparison/` and `instrument/` packages and root `app/` are required. `graph_local/` contains the adapters and study runner; `tools/` contains executable setup, worker and report wrappers. Model checkpoints and fitted artifacts are staged separately.

## Environment

Run installation, imports, hashing, simulation, inference and reporting inside
your own real Slurm compute allocation. Python entry points retain allocation
guards. Arrange
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

Resolve the included `benchmark/evaluation/configs/workflow-e-plan.json` using its bindings example and `evaluation/workflow/render_plan.py`, or use an editable matrix under `examples/`. Pass the resulting plan below.

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
  --plan /absolute/path/to/rendered-workflow-plan.json \
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


## Saved-result reports

`tools/report.py --help` describes matrix/reachability reports from an existing frozen study. `comparison.study_report` aggregates saved shards using the original scorers. Keep generated outputs under local run directories. No saved model results are included in this source release.
