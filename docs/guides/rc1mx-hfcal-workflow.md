# RC1mx + HFCal workflow

This project includes a bounded, real-model workflow for the native ChemLab RC1mx + HFCal simulator. DeepSeek proposes instrument actions and observations, and Jev selects devices, actions, and graph routes. The host initializes a fixed batch and checks boundaries and completion independently every two seconds of simulated time. The model chooses the instrument, control mode, stirrer settings, observation fields, and wait durations. Instrument state reads do not advance the simulation; `workflow.await` does.

The current scenario starts with **0.800 kg water and 0.020 kg acetic anhydride charged at 25 °C**. The objective is to reach 35 °C, hold the reactor between 34 and 36 °C, and reach at least 99% model conversion and stoichiometric product yield. During warm-up, the permitted temperature range is 24–36 °C. When a two-second monitor sample first enters 34–36 °C, the holding range latches; leaving that range then stops the run. Material addition, discharge, and changes to the kinetic priors are outside this task. This is a simulator experiment, not a procedure for physical equipment.

## Run locally on Windows

Use Python 3.11 and restore the pinned ChemLab runtime files described in the repository [setup guide](../../README.md). The Windows launchers use a local `.venv` and a local `.env` file. For example:

```powershell
uv sync --group dev
& .\.venv\Scripts\python.exe -B tools/fetch_device_runtime.py --download
& .\.venv\Scripts\python.exe -B tools/fetch_device_runtime.py
Copy-Item .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Keep it local. A live run makes paid OpenRouter calls. Check the initial state without any model calls or clock advance:

```powershell
& .\.venv\Scripts\python.exe -B -m app.rc1mx_experiment --preflight
```

Then start one bounded live run:

```powershell
.\Run-RC1mx.ps1
```

The script creates a fresh directory under `results/` and opens the read-only result viewer at `http://127.0.0.1:8793/`. The viewer shows model questions and answers, selected actions, instrument receipts, the result JSON, and the final checks. It reads saved records without making model calls. Use `Start-Viewer.cmd` to open it again later.

Each live run stops at the first completion or boundary. The current limits are 500 model calls, 200 lab actions, 120 seconds per wait, 5,400 seconds of simulated time, and 1,200 seconds of wall-clock time. Logs include source hashes, original API requests and replies, independent monitor samples, native export, and a temperature/conversion CSV. `results/` and `.env` are ignored by Git.

The reaction kinetics are example simulation priors rather than measured kinetics. Model conversion is known to the private evaluator, not exposed to the agent as an instrument measurement. The two-second monitor catches sampled violations; it does not establish a continuous-time temperature guarantee.
