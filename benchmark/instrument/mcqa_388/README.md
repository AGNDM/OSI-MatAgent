# Frozen 388-question MCQA dataset

`questions.jsonl.gz` is the original public archive, preserved byte for byte. Its uncompressed question hash and split IDs match the frozen v1 release. Each record includes the question, public instrument observations, contract, operating goal and shuffled answer choices. The 388 questions represent 194 base cases and 112 scenario groups across eight instruments: 182 have two choices, 136 have three and 70 have four. Some choices are complete operating plans.

`answers.jsonl` contains evaluator-only reference answers checked through native simulation and authored criteria. Each current question has one correct option; the list format supports a future release with several accepted options:

```json
{"task_id": "opaque question ID", "correct_option_ids": ["B"]}
```

The answer file omits private fixtures and full native traces, which are unnecessary for static scoring. Independent expert review remains pending.

## Model inputs and predictions

Run commands from the distribution root, inside a Slurm compute allocation on SCITAS:

```bash
.venv/bin/python eval/prompts.py --split heldout --output output/prompts.jsonl
.venv/bin/python eval/mcqa.py --predictions my-predictions.jsonl --split heldout --output output/scores.json
```

Each exported row has `task_id` and `prompt`. Give the model the prompt only; keep the ID for matching predictions. The formatter reproduces the original `public-menu-union-v2` public evidence projection, including documentation for candidate actions symmetrically. Raw question rows also contain bookkeeping fields; do not send those or the answer keys to the model.

The exported text is the evidence portion. Record the model's system message, answer instructions, decoding and aggregation separately. [The recorded Qwen protocol](../examples/qwen-baseline-protocol.json) describes the initial baseline; exporting prompts does not run inference.

Predictions use exact original option IDs:

```json
{"task_id": "opaque question ID", "selected_option_id": "B"}
```

`status` defaults to `ok`; explicit other statuses count as incorrect. Probabilities and extra method metadata do not change accuracy. Missing predictions and invalid choices also count as incorrect, retaining the full requested denominator. Malformed JSON, duplicate IDs and unknown task IDs are rejected. Known predictions outside the selected cohort are ignored and counted separately.

## Splits and metrics

| Split | Questions |
| --- | ---: |
| `train` | 232 |
| `validation` | 42 |
| `test` | 60 |
| `test_template` | 36 |
| `test_fault` | 8 |
| `test_instrument` | 10 |

The default `heldout` cohort is the union of the four `test*` splits: **114 questions**. `--split all` scores all 388, including train/validation diagnostics; label that result separately. Exact split names are supported.

The scorer reports question accuracy, equal-tier macro accuracy, valid-answer coverage, missing/invalid counts, and breakdowns by menu size, category, tier, instrument, domain, task type and split. Random-guessing expectations use each question's actual menu size. Probability calibration and other study metrics are outside this minimal scorer.

Legacy `difficulty` IDs describe construction categories:

| ID | Task category |
| --- | --- |
| L1 | Single-action checks |
| L2 | Direct diagnosis and simple recovery |
| L3 | Evidence assessment, testing and escalation |
| L4 | Procedure and service plan selection |
| L5 | Interruption response plan selection |

An increasing empirical difficulty ordering has not been established. These categories are separate from failure mechanisms and T1/T2/T3 tiers.

## Initial baseline and provenance

The initial `qwen-generative-sampling-0` run used Qwen2.5-3B-Instruct and a vote over ten independently generated valid answers per question. Held-out accuracy was **69/114 = 60.5%**, with **62.2% tier-macro accuracy**. Its four-choice held-out subset scored **23/24 = 95.8%**. Choice-count cohorts have different task mixes. This is one sampling seed, not a completed model comparison. The distribution includes aggregate results and the protocol, without model weights or raw model outputs.

`manifest.json` records current dataset counts, filenames and hashes. `source-release-manifest.json` preserves the original upstream release manifest. [Historical bundle receipts](provenance/README.md) describe the earlier MCQA-only package. Their historical paths and sequential-placeholder statements do not describe the current combined distribution.
