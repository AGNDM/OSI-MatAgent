# Saved-result reporting and reproduction

This package contains the small postprocessing closure used to produce the final
[388-question results](../../results/388/README.md) and
[120-case results](../../results/120/README.md). It reads saved predictions/episodes;
it does not load models, fit readouts, run native simulations or retry cases.
The original helpers and frozen studies under `results/` remain unchanged.

The portable copies add explicit input/output bindings. Their source hashes,
timestamps and provenance consequently differ from the original exports; regenerated
files must not be described as byte-identical historical artifacts.

## Included sources

| Reusable file | Actual original source and purpose |
| --- | --- |
| `mcqa/export_overall.py` | New narrow snapshot-to-overall copier, reconstructing the schema1.0 input used by the two original export stages. |
| `mcqa/export_groups.py` | `results/.mcqa-group-export/export.py`, used in Slurm4639272 for L1–L5 and2/3/4-choice breakdowns. |
| `mcqa/enrich.py`, `metrics.py`, `accounting.py` | `results/.mcqa-metrics-export/`, used in Slurm4640085 for confidence, usage and allocation estimates. |
| `workflow/report_grid.py` | `results/.deadline-rescue/report_grid.py`, final full120 export in Slurm4644452. |
| `workflow/compact_status.py` | Original compact exporter, run after final full export; copies authenticated fields without recalculating scores. |
| `workflow/audit_failures.py` | Original `audit_all_closed_failures.py`, final all22-cell error audit in Slurm4646285; not the earlier weak-only audit. |
| `workflow/final_audit.py` | Original independent closure auditor, final pass in Slurm4646411. |
| `tests/` | Copied final-audit/compact regressions, plus confidence/resource and explicit-path-binding fixtures. |

[configs/original-source-provenance.json](configs/original-source-provenance.json)
records the original file paths, SHA256 values, final E source/cohort bindings and
which originals were pinned by published provenance. It is generated once from
the actual historical artifacts by `prepare_profiles.py`, without changing them.
The dated [tariff capture](configs/site-tariffs-20261022.yaml) is historical evidence,
not a live price source or invoice.

One-off scheduler recovery, monitor/drain controllers, CPU-binding fixes, memory
probes and development pilots are **historical evidence only**. Their hundreds of
operational scripts are not copied here. The final E auditor still authenticates
the evidence they produced; excluding their executables does not waive those checks.

## Execution and dependencies

On SCITAS, run Python, hashing, validation and all exports **inside a genuine Slurm
allocation**. The scripts enforce `SLURM_JOB_ID` and `SLURMD_NODENAME`. Python3.11+
and standard-library modules suffice for postprocessing, with the frozen scoring
modules described below available. Models, weights, fitted artifacts, environments
and raw run journals are deliberately not redistributed by this package.

`slurm/run.sbatch` is a small postprocessing wrapper. Set absolute
`EVALUATION_ROOT` (this directory) and `EVALUATION_PYTHON` (an existing interpreter),
then use `sbatch "$EVALUATION_ROOT/slurm/run.sbatch" COMMAND ...`. Its default
SCITAS resources may be overridden with normal `sbatch` flags. It clears inherited
CPU-binding masks and runs a single explicit command. It performs no scheduler
mutations other than the allocation explicitly submitted by the caller.

The commands below are **compute-allocation payloads**, not login-node commands.
`EVAL`, `PYTHON`, `OUT`, `STUDY`, `EVIDENCE_ROOT`, `SNAPSHOT`, `TAXONOMY` and
`SCORER_SOURCE` denote explicit absolute paths supplied by the caller. Use fresh
output directories outside the frozen study. Published JSON is never an implicit
default destination.

## Reproduce the 388 exports

Required inputs are the authenticated original static `snapshot.json`, its
recorded scorer/bundle tree, taxonomy, saved prediction shards and their worker
configurations. Hashes in the snapshot and bundle manifest are checked before
scoring. The committed
[metadata profile](configs/mcqa-overall-metadata.json) declares the historical
34-configuration dataset/model scope and expected snapshot hash explicitly.

```bash
"$PYTHON" "$EVAL/mcqa/export_overall.py" --snapshot "$SNAPSHOT" \
  --metadata "$EVAL/configs/mcqa-overall-metadata.json" --output "$OUT/base/mcqa-results.json"
"$PYTHON" "$EVAL/mcqa/export_groups.py" --overall "$OUT/base/mcqa-results.json" \
  --snapshot "$SNAPSHOT" --taxonomy "$TAXONOMY" --source-root "$SCORER_SOURCE" --output-dir "$OUT/base"
"$PYTHON" "$EVAL/mcqa/enrich.py" --input-dir "$OUT/base" --output-dir "$OUT/enriched" \
  --work-dir "$OUT/enrichment-evidence" --snapshot "$SNAPSHOT" --taxonomy "$TAXONOMY" \
  --source-root "$SCORER_SOURCE" --tariffs "$EVAL/configs/site-tariffs-20261022.yaml"
```

The closure imports authenticated `comparison.study_report`, `study_metrics`,
`taxonomy`, `aggregate` and `instrument.eval.mcqa`; it reads272 saved prediction
files. Held-out scores and all existing score fields/denominators must reconcile
before augmentation. The three outputs retain separate full-cohort/held-out scopes,
the restricted T1 reference and missing-confidence/usage coverage. Selection Brier,
AUROC ties, threshold ties and reference zero-timing placeholders retain their
original semantics. No significance tests or new inference are performed.

Accounting queries `sacct` read-only. Currency estimates cover full retained MCQA
allocations, including loading/overhead; they are not prorated to held-out/category/
choice-count subsets. Unavailable historical accounting stays partial/null. The
captured rates do not establish actual bills or active GPU time.

## Re-export and independently audit historical E

The 120 tools retain the exact **22 × 120, seed 0, teacher-forced** completion gates.
They are not a generic promise that a newly prepared study passes E's closure audit.
The authenticated study must include `prepared.json`, source/configuration/jobs/
scope manifests, frozen `snapshot/`, exact production `runs/`, worker identities,
launch-batch metadata, dispatch receipts, scheduler recovery manifests and hardware
override evidence. Its original recorded evidence paths must remain accessible.

The reporter imports only the frozen standard-library saved-record scorer. Failure
and final audits also use frozen `graph_local.run.authenticate` to verify the full
inventory. Final closure requires complete actual Slurm ownership/accounting and
source-bound hardware/memory evidence; missing evidence is never silently skipped.
There is no automatic path rewriting, source rebinding or model reload.

```bash
"$PYTHON" "$EVAL/workflow/report_grid.py" --study "$STUDY" --evidence-root "$EVIDENCE_ROOT" \
  --scheduler --report-launcher "$EVAL/slurm/run.sbatch" --output "$OUT/workflow-results.json"
"$PYTHON" "$EVAL/workflow/compact_status.py" --study "$STUDY" \
  --input "$OUT/workflow-results.json" --output "$OUT/workflow-status.json"
"$PYTHON" "$EVAL/workflow/audit_failures.py" --study "$STUDY" --output "$OUT/workflow-failures.json"
"$PYTHON" "$EVAL/workflow/final_audit.py" --study "$STUDY" --evidence-root "$EVIDENCE_ROOT" \
  --main "$OUT/workflow-results.json" --status "$OUT/workflow-status.json" \
  --failure-audit "$OUT/workflow-failures.json" --output "$OUT/workflow-final-audit.json" --require-complete
```

`EVIDENCE_ROOT` explicitly bounds study/recovery evidence reads. To audit an existing
historical failure report, additionally supply `--failure-helper` with the exact
original classifier source it records; its bytes must match that report's hash.
For the commands above, the default is this package's `workflow/audit_failures.py`.
All input exports/raw records remain read-only, and audit outputs refuse replacement.

Metrics preserve original-before-correction grading. All-closed action totals include
saved abort decisions and auxiliary queries; `native_score` action/checkpoint/group
metrics remain completed-episode-only. Every cell retains120 cases in its assisted
goal denominator. Null/unknown usage is preserved. Technical, startup, unknown,
missing or inconsistent evidence blocks final completion; evidenced model-output
and budget aborts remain valid failed attempts. See [results/120](../../results/120/README.md).

## Portable model/protocol recipe and checks

The [static inference profiles](configs/static-388/mcqa-used-profiles.md) preserve
the two actual input plans behind the 34 static configurations: 22 R2 methods
and 12 feature/Qwen3 methods. Each has its own source/configuration provenance,
eight-shard cohort and explicit path bindings. Their generation in Slurm
authenticated all 272 saved MCQA workers and checked that resolving the path
placeholders preserves every experiment field. Run them separately with
`--track mcqa`; their historical sequential settings are excluded.

[workflow-e-plan.json](configs/workflow-e-plan.json) copies the 22 used model/method
specifications and budgets, replacing model/artifact locations with explicit named
placeholders. [workflow-e-bindings.example.json](configs/workflow-e-bindings.example.json)
lists every required binding. Fill a separate copy with your own absolute paths,
then run `workflow/render_plan.py --plan ... --bindings ... --output ...` in Slurm.
Pass the resolved plan to the [graph preparation workflow](../graph/README.md)
with **`--mode teacher_forced`**, an explicit APUS interpreter/environment and
fresh source receipts. `analysis.mode` is descriptive metadata; the preparation
CLI's `--mode` selects the actual execution mode and otherwise defaults to both.
Weights/artifacts are not downloaded or inferred from labels. This creates a new
study identity, not the historical E scheduler/evidence envelope.

The combined validation entrypoint is [../slurm/check.sbatch](../slurm/check.sbatch).
Evaluation regressions can also be run within its compute allocation, without
raw production runs or model files:

```bash
"$PYTHON" -m pytest -q --tb=short -p no:cacheprovider "$EVAL/tests"
```

Tests use synthetic evidence and the shipped instrument saved-record scorer.
Optionally set `EVALUATION_STUDY` to use an authenticated historical study's frozen
scorer instead; full archival audit commands still require all original evidence.
The focused checks cover nested/group/per-case score tampering, same-count
stale-source evidence, technical/unknown/startup failures, legitimate model/budget
failures, incomplete cohorts, null usage, confidence ties and explicit bindings.
They do not launch models, replay native workflows or alter published results.
