# Titania workflow: acquisition audit and reproduction

The [scenario card](scenario.md) defines the scientific question and evidence boundary. Implementation is in `app/titania.py`, runner in `bench/titania.py`, and regression cases in `tests/test_titania.py`. No dependency was added.

## What actually executes

| Component | Verified native capability | Host adaptation and limits |
|---|---|---|
| Nabertherm L9, CHEM-185 | Power/door controls, program configuration/start, time-stepped ramp/hold, public completion and indicated-temperature reads | Empty-furnace dynamics; no sample chemistry or thermal mass. Logged in-memory scene callback; no rendered scene. Start near room temperature, await completed program, power off, await transfer guard |
| MiniFlex600, CHEM-160 | Door/sample-slot/tube/slit configuration, scan configuration/start, time-stepped acquisition, acquired-pattern read | Explicit ready cooling fixture. Host supplies a reference-derived sample response. Successful start is followed by running/partial/completed public reads |
| Sample handoff | Native XRD sample connection ID | Adapter owns batch/aliquot lineage and retained dry material. Loading, transfer and finite endpoint lookup are authored; no robot motion, weighing, shared chemical engine or mass balance |
| Time | Furnace delta stepping; XRD absolute stepping | One adapter clock translates the two APIs; polling is bounded within synchronous workflow macros. General graph-level asynchronous jobs are still pending |
| Phase check | Native integration of the supplied profile; reference metadata query | Deterministic local-maxima analysis is our code, not a native phase-identification service or fitted quantitative assay |

A complete full scan has 801 points at 0.05° spacing and 0.05 s integration per point. The audit records start with zero completed points, an intermediate partial scan, completion, matching clocks, furnace hold completion and an agent query for `template_sample_truth` rejected with `ACCESS_DENIED`. Native references and all raw acquired points are saved. Acquisition data are explicitly labelled synthetic even though upstream's template reader uses `synthetic: false` to describe reference provenance.

Native code is pinned by the existing expansion manifests; its source is not modified. Reference library SHA-256: `8a8f0a0418a541e570dbd55c999a6cfd191284cdd4eb8af7d572cc88ae9864e5`. See the reset artifacts for exact DIF filenames, public reflection lists, provenance, limitations and profile assumptions. Existing upstream attribution and data license conditions continue to apply.

## Reproduce locally

From the repository root with the existing `osi-hack` conda environment:

```bash
python -m bench.titania --output results/titania-scripted-new --controls
python -m pytest -q
```

The scripted specialist is an integration-test double and makes no cloud requests. `--controls` additionally runs both wrong-repair controls and the native acquisition audit. A new output directory is required for every attempt.

For explicitly authorized small real-model debugging, start the local OpenJev server as described in [the deployment guide](../../guides/local-openjev.md), export `SILICONFLOW_API_KEY`, then run:

```bash
python -m bench.titania --backend deepseek --router openjev \
  --output results/titania-deepseek-new
```

This uses `deepseek-ai/DeepSeek-V4.1-Flash` at the configured SiliconFlow endpoint and the existing durable **shared 5,000,000-token ledger**. It never calls OpenRouter. Calls, failures and repairs are charged by `BudgetedExpert`; the ledger must not be reset. Two branches run sequentially, each with at most 16 expert calls, eight budgeted workflow actions and a 240-second graph deadline. A full acquisition counts as a command plus a readback. Native commands are separately logged inside each macro.

Artifacts include a source-hash manifest, per-branch reset evidence, exact model inputs/outputs and routing events in JSONL traces, final raw patterns, independent scores, and optional controls/audit. Reports and patterns are generated locally under ignored `results/`; a compact checked-in execution summary is in [the execution summary](../../evidence/titania-validation.md).

## Remaining acceptance work

This adds one compound-specific vertical slice. It does not turn the other 194 control tasks into acquisition tasks, establish physical transfers, add task-conditioned retrieval/Materials Encoder, add a second expert, demonstrate a routing advantage, or add this scenario to the browser UI. The paper snippets and reference windows are curated fixed context. Broader retrieval, blind held-out cases, independent scientific review, formal model comparisons and visual integration remain open.
