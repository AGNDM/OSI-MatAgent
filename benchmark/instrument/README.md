# Instrument decision benchmark

Portable datasets and native graders for **388 multiple-choice questions** and
**120 sequential instrument cases**. The simulator-derived questions, answer
keys, case specifications, splits and runtime retain their original contents.
Software tests and cluster submission scripts are excluded from this package.

## Setup and evaluation

Run these commands from this directory inside your compute allocation. On
SCITAS, installation, imports, simulation, inference and scoring belong on
compute nodes. Supply model weights separately.

```bash
python3.12 tools/setup.py
.venv/bin/python eval/validate.py
.venv/bin/python eval/prompts.py --split heldout --output output/prompts.jsonl
.venv/bin/python eval/mcqa.py --predictions my-predictions.jsonl \
  --split heldout --output output/scores.json
.venv/bin/python eval/sequential.py run --policy reference --mode both \
  --output output/sequential-reference
```

Setup creates or verifies the pinned native `.venv`; `--model` selects the
optional `.venv-models` inference environment. Neither downloads model weights.
Use fresh output directories. The privileged reference policy checks evaluator
execution and is not a learned-agent result. See the
[sequential guide](sequential_120/README.md) and [policy examples](examples/README.md).

Give models only exported prompts. Retain opaque task IDs to match predictions;
keep answer keys and private sequential specifications out of model inputs.

## Contents

| Folder | Contents |
| --- | --- |
| `mcqa_388/` | Frozen questions, answer keys, cohort splits and provenance |
| `sequential_120/` | Public cases, menus, native runtime and private evaluator fixtures |
| `eval/` | Prompt export, graders, interactive runner and integrity validation |
| `examples/` | Prediction records and policy adapters |
| `tools/` | Pinned environment setup and packaging maintenance |

`manifest.json` authenticates the current package inventory. Packaging cleanup
changes that inventory, not the benchmark data or scoring rules. Existing
validation receipts describe their original source snapshots. See
[release maintenance](tools/README.md) before changing manifest-protected files.

The static track has 114 held-out questions; full-corpus metrics include training
and validation diagnostics. The sequential track has no assigned train/test
split. Report autonomous outcomes, original-action accuracy and assisted
completion separately. Simulation does not establish real-equipment reliability.

Keep [LICENSE](LICENSE), [asset terms](LICENSE-ASSETS.md), [NOTICE](NOTICE.md)
and [CITATION.cff](CITATION.cff) with every copy.
