# Historical reporting validation

These immutable receipts record the original reporting qualifications. They do
not certify later package reorganization or model-study results.

| Archive | Original Slurm qualification | Current reusable implementation |
| --- | --- | --- |
| [Matrix reporter](matrix-report-4635920/receipt.json) | 4635920, 26 tests | `graph_local/reporting/matrix_report.py` |
| [Reachability appendix](reachability-4635941/receipt.json) | 4635941, 10 tests | `graph_local/reporting/reachability.py` |

The reporter implementation bytes were retained during relocation. The receipts
retain hashes and counts from their original qualification. Historical software
tests and scheduler scripts, including archived text copies, have since been
removed from this package; those receipt entries describe the historical run.

Current execution uses the Python entry points in the [package guide](../../README.md).
Legacy Python paths under `tools/reporting/` remain compatibility entry points.
