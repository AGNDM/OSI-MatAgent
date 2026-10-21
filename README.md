# Decision-Routed Lab

A local LangGraph workflow for native virtual-instrument tasks. DeepSeek V4.1 Flash generates typed candidates, procedures and evidence-based reports. Hosted TypeSafe Jev 1.14 chooses the instrument, next operation and expert route. A deterministic gate validates actions before the pinned ChemLab runtime executes them. The default routing policy takes a legal maximum-probability choice; it has no low-confidence escalation or silent model-failure fallback.

The primary entry point accepts a goal and public-state predicates, discovers public instrument contracts and executes without a reference solution script. The expert can inspect the complete **194-device directory** and propose up to four described devices for Jev to choose from. Only one native instrument is bound per episode. Full directory visibility does not mean that every instrument supports every acquisition or time-control operation.

## Setup

```bash
conda env create -f environment.yml
conda activate osi-hack
python tools/fetch_device_runtime.py --download
python tools/fetch_device_runtime.py
python -B -m pytest -q
```

The manifests pin 571 upstream files, including the restored washer dependency. The approximately 221 MB cache, model weights, credentials and generated experiment traces are excluded from Git.

## Run a goal without a solution script

Set `OPENROUTER_API_KEY` in the environment. After changing shell startup credentials, start a fresh interactive login shell. No key belongs in a task file or report.

```bash
python -m app.instrument_cli --backend deepseek \
  --router openrouter-jev --operation openrouter-jev \
  --task cases/acceptance/v1/binder-timer.json \
  --knowledge on --output results/my-auto-binder
```

This task declares `device: "auto"`, an objective and fixed outcome predicates. It supplies no action sequence, instrument shortlist or answer script. The expert searches/describes public contracts; Jev selects a proposed device, then the ordinary diagnostic → operation choice → procedure → gate → native execution → fresh evidence loop runs. The task oracle remains fixed throughout selection. Unsupported material-state goals stop honestly rather than invoke a chemistry fixture.

`--knowledge on` adds task/evidence-conditioned lexical retrieval with separate knowledge and observation citations. It supplies facts, not an action plan. The corpus includes native instrument behavior and one scoped paper passage; it does not implement synthesis chemistry. See the [native acceptance and knowledge report](docs/reports/native-acceptance-and-knowledge.md) for results and limitations.

The authorized OpenRouter DeepSeek/Jev transports record actual usage without a local token-amount cap, following the user's updated instruction. Finite action, model-call, time and output limits remain. SiliconFlow is explicitly selectable and retains the persistent 5M-token budget; no provider fallback is automatic. Free OpenRouter models remain limited to small explicit debug calls. See the [single-instrument guide](docs/guides/single-instrument-agent.md).

## Offline checks and historical demonstrations

```bash
# Frozen development acceptance with clearly labeled offline doubles.
python -m bench.acceptance --phase offline --output results/my-native-acceptance

# Historical device integration fixtures; not the generic live task solver.
python -m bench.device_tasks --output results/my-device-regression

# Historical drying UI with scripted specialists.
python -m app.demo --port 8793 --output results/my-drying-demo
```

The new live path uses `InstrumentLab` and native public methods only. The old drying/device fixtures and paper-grounded TiO2 demonstration remain for regression and historical comparison. **Titania uses authored `thermal.prepare_reference` / `xrd.acquire` macros and a material endpoint fixture. It does not satisfy the current no-task-specific-script requirement.** Its runner requires `--legacy-demo`. A generic multi-instrument TiO2 synthesis/transfer/acquisition workflow is still pending.

## Evidence and boundaries

The historical device suite passed 194/194 control/readback fixtures, including 116 power-only cases; this is interface coverage rather than 194 complete experiments. Current automatic native acceptance covers four diverse devices and records failed attempts as well as successful outcomes. Consult the [current report](docs/reports/native-acceptance-and-knowledge.md), [checklist](docs/status/checklist.md) and [evidence index](docs/evidence/README.md) for dated results, source freezes and exact counts.

Native lifecycle adaptation now follows API signatures and declared asynchronous contracts without a device whitelist. Actual clock/readback/replay checks passed on 191 of 194 registered devices; the other three have no native host clock API and retain their public device tools. Poll/cancel interfaces are exposed for 149 declared asynchronous devices. See the [coverage report](docs/reports/native-lifecycle-coverage.md); these counts do not establish complete scientific acquisitions. Goal satisfaction, model claims and protocol correctness are reported separately. Knowledge retrieval benefit, held-out model generalization, a second general-expert comparison, calibrated routing advantage and physical sample transfer remain unaccepted. The scientific UI and offline decision benchmark are deferred.

## Layout and attribution

| Directory | Contents |
|---|---|
| `app/` | LangGraph, public device discovery, native adapter and model transports |
| `bench/` | Frozen acceptance and historical evaluation runners |
| `cases/`, `corpus/`, `configs/` | Declarative tasks, public contract cache, knowledge facts and configuration |
| `tests/`, `tools/` | Offline regression and pinned-runtime setup |
| `docs/`, `web/` | English documentation, evidence and historical drying UI |
| `vendor/` | Upstream manifests and attribution |
| `results/` | Ignored local traces, frozen sources and durable usage ledger |

ChemLab code is pinned to `123cf78593469871e41b8dbd63a11dfb7c5ca251`; data to `4ca4a42dab2ca0a884e449d202e9ad96625d5648`. Preserve [LICENSE](vendor/chemlab/LICENSE), [LICENSE-ASSETS.md](vendor/chemlab/LICENSE-ASSETS.md) and [NOTICE.md](vendor/chemlab/NOTICE.md). No 3D geometry is required. Start with the [documentation index](docs/README.md).
