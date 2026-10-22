# Instrument evaluation

Evaluation datasets, model adapters and result exporters are organized here.
Published summaries live in [`results/388/`](../results/388/README.md) and
[`results/120/`](../results/120/README.md).

| Folder | Purpose |
| --- | --- |
| [instrument/](instrument/README.md) | Frozen 388-question dataset, 120 sequential cases, native runtime and graders |
| [comparison/](comparison/README.md) | Local model inference, confidence readouts, feature baselines and static study preparation |
| [graph/](graph/README.md) | Expert × chooser evaluation through the shared application graph |
| [evaluation/](evaluation/README.md) | Saved-result exports, calibration/resource metrics and workflow audits |
| [review/](review/README.md) | Blinded human-review interface |
| [docs/](docs/methodology.md) | Dataset construction, scoring methods and historical provenance |

Use the Python entrypoints documented in each package. Software tests and cluster
submission scripts are excluded. On SCITAS, execute Python, installation,
simulation, inference and analysis inside your own compute allocation. Model
weights, fitted artifacts and environments are staged separately; generated runs
belong outside source packages.

Portable recipes retain the [two used static study plans](evaluation/configs/static-388/mcqa-used-profiles.md)
and the [22-cell sequential study](evaluation/configs/workflow-e-plan.json).
Provide explicit model/artifact paths and a fresh output directory. For the
sequential recipe, pass `--mode teacher_forced` explicitly during preparation.

The static results contain 34 configurations with 114 held-out questions. The
sequential results contain 22 × 120 cases, seed 0, using teacher forcing. Original
actions are graded before incorrect proposals are corrected; assisted completion
is separate from autonomous success. See the result READMEs for exact metrics.

The graph runner uses shared `app/` code from `development` through `e116c6b`
(or a compatible later revision). This includes the upstream rounded-Jev and
readback-preset fixes. No duplicate application implementation is bundled here.

See [methodology](docs/methodology.md), [study history](docs/study-history.md) and
[historical validation evidence](docs/provenance/README.md). Packaging removal of
tests and launchers does not change dataset, answer, split or native-grader bytes;
its manifest is refreshed to describe the files actually distributed.
