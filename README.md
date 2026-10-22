# OSI MatAgent

A LangGraph agent for virtual laboratory instruments, with a benchmark spanning **388 multiple-choice questions** and **120 stateful workflow tasks**.

The general expert proposes evidence-grounded actions. A decision model selects the next route and operation; typed validation and a deterministic gate check the proposal before native execution. Fresh observations feed the next decision. Tasks specify goals and public-state predicates, without supplying a reference action script.

**[Evaluation report](docs/reports/instrument-evaluation-report.html)** · **[PDF](output/pdf/instrument-evaluation-report.pdf)** · **[Documentation](docs/README.md)**

## Setup

```bash
conda env create -f environment.yml
conda activate osi-hack
python tools/fetch_device_runtime.py --download
python tools/fetch_device_runtime.py
python -m pytest -q
```

The runtime manifests pin upstream code and data. Downloaded runtime files, model weights, credentials and experiment outputs stay outside version control.

## Run a native task

Set `OPENROUTER_API_KEY` in your shell, then run:

```bash
python -m app.instrument_cli --backend deepseek \
  --router openrouter-jev --operation openrouter-jev \
  --task cases/acceptance/v1/binder-timer.json \
  --knowledge on --output results/my-binder
```

This uses DeepSeek V4.1 Flash and hosted TypeSafe Jev 1.14. A task with `device: "auto"` exposes the 194-device public directory for discovery and binds one selected instrument per episode. Public interface availability does not imply support for every scientific acquisition. Multi-instrument material transfer and general synthesis remain outside this solver's current scope.

## Evaluate

The [benchmark guide](docs/guides/instrument-eval-langgraph.md) covers bundle retrieval, local CPU execution, expert/decision configurations, free-run trajectories, teacher-forced evaluation and MCQA. Model calls require credentials and may incur provider charges; configuration changes never silently switch providers or endpoints.

The final report combines hosted models and collaborator-provided compact models. MCQA measures answer selection; free-run measures autonomous task completion; teacher-forced evaluation measures original actions along corrected trajectories.

## Repository

| Path | Purpose |
|---|---|
| `app/` | LangGraph workflow, device discovery, native adapters and model transports |
| `bench/` | Evaluation runners and offline regression harnesses |
| `cases/`, `corpus/`, `configs/` | Task inputs, public contracts, knowledge and model configuration |
| `tests/`, `tools/` | Offline checks and runtime installation |
| `docs/` | Final report, usage, architecture and attribution |
| `results/` | Ignored local experiment artifacts and usage ledger |

Legacy scripted drying/device/TiO2 runners remain as regression utilities, separate from the reference-free solver. Titania requires `--legacy-demo`.

See [third-party attribution](docs/reference/third-party.md). Native runtime licenses and notices remain under `vendor/chemlab/`.
