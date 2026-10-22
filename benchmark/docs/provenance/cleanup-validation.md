# Evaluation cleanup validation — 2026-10-22

All checks ran in genuine Slurm allocations on Kuma (`kh062`). No model
inference or hosted API calls were made. Native simulator integration fixtures
are explicitly privileged diagnostics, not additional model results.

| Scope | Passed tests | Slurm evidence |
| --- | ---: | --- |
| Frozen instrument integrity and scorers | 82 | [4647976 receipt](cleanup-benchmark-check-4647976.json) |
| Static comparison runner and launchers | 133 | Same receipt |
| Application-graph bridge, native integration, portability and reporters | 327 | Same receipt |
| Export metrics, compact projection, audit gates and path bindings | 37 | Same receipt |
| Full application regressions, including the two upstream fixes | 217 | [4647988 receipt](cleanup-app-check-4647988.json) |
| **Total** | **796** | |

The first combined job's benchmark suites passed, as did its shell/Python syntax
checks and source/result integrity guards. Its overall status remains `failed`
because full application collection required a legacy simulator absent from the
eight-device benchmark bundle. The application checker was corrected to use the
normal application import path and verify its complete pinned runtime. Job
4647988 authenticated all 571 manifest entries, restored 568 missing ignored
cache files from matching existing upstream bytes, and passed all application
tests. It performed no downloads; tracked source and published files were
unchanged during that check. The original failed receipt is preserved as recorded.

[Export replay 4647979](cleanup-export-replay-4647979.json) ran all seven saved
export/audit stages into a fresh ignored result directory. All three static
exports preserved their scores, denominators, confidence metrics, selective
curves, measured timing and token fields. All 22 sequential cells preserved
native/group/checkpoint/per-case metrics, usage and completed/all-closed totals.
The fresh final audit passed **23/23 gates with zero issues**: 2,640 closed
records, 1,344 assisted goals, 1,296 aborts, 5,477 correct original proposals out
of 12,466, and 6,989 corrections. Source and published-result hashes were
unchanged during replay. New export timestamps, helper identities and live
allocation-accounting provenance are intentionally separate from historical bytes.

Job **4647975** authenticated the used reporting-helper identities and generated
the [portable 22-cell workflow recipe](../../evaluation/configs/original-source-provenance.json).
Job **4647987** authenticated the two used static configurations, all **272**
MCQA worker bindings, fitted-artifact hashes and saved runtime evidence. It
checked that resolving the [34 static configuration recipes](../../evaluation/configs/static-388/mcqa-used-profiles-provenance.json)
preserves every original experiment field. These profiles retain separate
historical source identities; they do not imply newly executed model runs.

The receipts retain their original paths and source hashes. Documentation and
portable static profiles were completed after the initial benchmark-suite
snapshot. The graph/comparison/evaluation runtime and scoring modules tested in
that snapshot were unchanged; the corrected application check helper was
exercised by job 4647988. Full raw studies, cache files and logs stay ignored.
