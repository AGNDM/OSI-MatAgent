# Bundled replay example

`rc1mx-hydrolysis.zip` contains a sanitized copy of the saved simulator recording
`rc1mx-live-20261022-trial13` (2026-10-22), including model requests/responses,
successful operation receipts and independent monitor samples. It is recorded
simulation output, not a new run or a physical laboratory experiment.

Secret-bearing fields and model reasoning fields are redacted using the history
viewer rules. Personal absolute paths are replaced with placeholders. The original
recording is unchanged. The archive contains only the five files used by replay.

When `results/` has no recordings, the launcher extracts this example as
`results/example-rc1mx-hydrolysis/`. Existing histories are never overwritten.
Copy other complete run directories into `results/` and choose them in the page.
