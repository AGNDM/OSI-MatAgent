# Historical installation evidence

For the current reorganization, see the [cleanup validation](cleanup-validation.md):
796 passed tests, portable run-profile checks and both saved-result export replays.

Software tests, cluster launchers and their archived source copies were removed
after these qualifications. The receipts remain historical evidence; current
package integrity is checked with `instrument/eval/validate.py`.

These receipts authenticate earlier installation/validation snapshots on Kuma.
They retain the original paths, source hashes, job IDs and scope. They do not
certify the current working tree after subsequent packaging changes.

| Receipt | Scope |
| --- | --- |
| [comparison-validation.json](comparison-validation.json) | Installed comparison controller, model-environment pins and prior integration checks |
| [graph-validation.json](graph-validation.json) | Original graph environment/package validation |
| [graph-installed-validation.json](graph-installed-validation.json) | Installed-copy portability/reporting checks |
| [graph-manifest.json](graph-manifest.json) | Original graph source inventory |

The files were relocated from the benchmark root without changing their
contents. Absolute paths identify historical evidence on the original cluster;
they are not current configuration defaults. Raw studies, model weights and
environments are kept outside versioned source.

Additional graph reporter receipts and original site-bound launchers are in
[`../../graph/docs/provenance/`](../../graph/docs/provenance/).
See [study history](../study-history.md), the
[published workflow audit](../../../results/120/workflow-teacher-forced-final-audit.json)
and the [reporting package](../../evaluation/README.md) for final-study provenance.
