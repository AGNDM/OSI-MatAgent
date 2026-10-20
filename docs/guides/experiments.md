# Experiments and Evaluation

The project distinguishes scripted integration regression, real development debugging and a future formal model comparison. A passing scripted policy does not establish model competence, and the current development samples do not establish generalization or routing superiority.

## Budgeted device debugging

```bash
python -m bench.device_tasks --output results/my-offline-devices
python -m bench.device_tasks --backend deepseek --device mt_xpr204e \
  --output results/my-real-device-debug
```

The default is offline, using scripted specialists and rules routing. Explicit DeepSeek device runs share the durable 5M-token ledger described in [Device workflow](device-workflow.md). `--router openjev` selects the separate local router. All outputs require a new directory; previous failures are retained.

The recorded device expansion passed 194/194 offline tasks, 20/20 final real-expert category-sample outcomes and 2/2 real-expert/local-router combinations. Initial pilots passed 0/3 and 2/3. Device debugging used 311,022 tokens including failures and repairs. A committed summary is in [Evaluation summary](../reports/evaluation-summary.md); raw traces remain local and ignored.

## Drying fixtures and metrics

`cases/dev` contains four authored development fixtures. `cases/test` contains eight authored test fixtures: two process faults, two measurement faults, two acceptable controls and two insufficient-evidence cases. They initialize native drying state and an authored balance layer; they are not measured laboratory data or an upstream task benchmark.

```bash
python -m bench.run --split test --repeats 2
```

This command defaults to scripted specialists. The independent evaluator checks world state, identity, fresh evidence and scenario policy. The agent cannot read hidden truth or set the official score.

- `task_success` is an independent outcome check.
- `expected_cause_covered` means an evidence-bearing candidate includes the expected cause; it is not final diagnosis accuracy. Historical `diagnosis_correct` fields retain their old coverage meaning.
- Trace-level failure counts include failed attempts and infrastructure errors.
- Router probabilities are conditional candidate probabilities, not calibrated correctness estimates.
- Fixed role-query protocol context is not task-conditioned materials RAG. The Materials Encoder is not integrated.

The earlier, explicitly bounded three-case real DeepSeek drying pilot passed 2/3 and consumed 42,802 tokens. Repeated verification mistakes in the acceptable case were retained. This pilot is separate from device debugging and is not a formal benchmark.

## Formal model-comparison interface

`configs/experts.template.json` separates reference and contrast general experts from OpenJev routing. Supply model, endpoint, optional revision and a credential environment-variable name. Local endpoints must use loopback HTTP; the explicitly authorized SiliconFlow HTTPS endpoint is also supported. Model identities returned by the provider are retained, but alias or immutable-weight equivalence is not assumed.

```bash
# Plan only: no model request.
python -m bench.matrix --config configs/experts.template.json \
  --output results/my-comparison-plan
```

The default two-repeat plan contains 48 attempts: two experts across eight cases and two repeats, plus a reference-expert rules-router comparison. An explicitly chosen single repeat gives 24 attempts. `--execute` is required to send requests and fails early if experts are unconfigured. Two aliases of the same endpoint/model are rejected as independent experts.

`--resume --execute` continues unattempted entries under the same frozen configuration/source fingerprint; it does not replace failures. A second general expert has not been selected, and the formal comparison has not been completed. The device runner's token guard is not automatically applied to historical matrix/CLI entry points; do not use them to bypass the current device allowance.

## Free-model restriction and annotation

OpenRouter free models are limited to small explicitly requested debugging. Free batch matrix execution is disabled. `bench.run --models` rejects multiple models, multiple cases, repeats and test-set free-model evaluation before network access. Retired plans must not be resumed as free batches.

`python -m bench.annotate <output-directory>` produces machine-suggested errors, heatmaps and recorded reports. These do not replace independent causal annotation. See [Error codebook](../reference/error-codebook.md). The proposal's independent human review and held-out routing evaluation remain pending.
