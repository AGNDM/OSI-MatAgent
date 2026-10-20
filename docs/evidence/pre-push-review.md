# Development branch pre-push review

Date: 21 October 2026. Scope: a focused logic review of the current workflow, titania acquisition/evaluator, token-budget enforcement, and publication contents. This is not an exhaustive audit or benchmark acceptance.

## Findings and fixes

1. **Failed thermal replacement could retain an earlier aliquot's phase fixture.** Starting a new preparation now invalidates the old phase data. Acquisition requires a completed thermal receipt for the current aliquot. A failed preparation no longer attaches the previous completed run to its result. The preservation policy counts attempted replacements, including failures, rather than only completed thermal runs. A regression injects a furnace timeout and checks that acquisition and optimistic success are rejected.
2. **Missing provider usage could crash the titania report writer.** Summaries now tolerate absent or malformed usage and mark `usage_complete: false`. Only valid reported usage enters the reported-token subtotal; the existing durable budget ledger continues conservatively charging unknown calls. Three regression variants cover absent, empty and inconsistent usage.

Acquisition also invalidates the earlier pattern before starting a rescan, so a timeout leaves no current valid pattern. Existing graph tests exercise both successful repair paths and independent negative controls.

## Validation and publication

The full project suite passes 84 tests. Existing pinned upstream files remain unchanged. The three earlier commits were scanned across 310 historical blobs for private-key and API-key patterns; no matches were found. Generated raw traces, downloaded assets, local models, environment files and the persistent budget ledger remain ignored. English scenario, provenance and compact execution evidence are versioned.

No cloud-model requests were made for this review. The successful real-model evidence in `titania-validation.json` remains an historical execution of commit `ea292f3`, identified by its original source hashes; it was not relabelled as a new model run of the reviewed revision.

The explicitly authored chemistry bridge, synthetic XRD profiles, fixed repair policy, CLI-only titania demo and missing formal decision-mechanism benchmark remain documented limitations. Within this focused review, no further blocking issue was identified after the fixes.
