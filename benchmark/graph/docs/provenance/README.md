# Historical reporting validation

These immutable receipts record the original reporting qualifications. They do
not certify later package reorganization or model-study results.

| Archive | Original Slurm qualification | Current reusable implementation |
| --- | --- | --- |
| [Matrix reporter](matrix-report-4635920/receipt.json) | 4635920, 26 tests | `graph_local/reporting/matrix_report.py` |
| [Reachability appendix](reachability-4635941/receipt.json) | 4635941, 10 tests | `graph_local/reporting/reachability.py` |

The reporter implementation bytes were retained during relocation. Original
test and launcher bytes are stored as `test_*.py.txt` and `run.sbatch.txt` next
to each unchanged receipt, so every historical `source_sha256` remains
checkable. The archived launchers contain the original machine-specific paths;
they are evidence, not supported execution entry points.

Current reporter tests live in `tests/` with package imports. Run them through
`slurm/check-reporting.sbatch`, or include them in `slurm/check.sbatch`.
Current matrix and appendix launchers are `slurm/report.sbatch` and
`slurm/reachability.sbatch`. Legacy `tools/reporting/` paths contain only
compatibility entry points.
