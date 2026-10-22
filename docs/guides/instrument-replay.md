# Instrument replay

For a guided explanation of the experiment and screen, start with the
[demo walkthrough](demo-walkthrough.md).

## One-click launch

Use the `rc1mx-hfcal` branch. Choose **Code > Download ZIP** on that branch, or:

```bash
git clone --branch rc1mx-hfcal --single-branch https://github.com/AGNDM/OSI-MatAgent.git
cd OSI-MatAgent
```

Extract the **entire folder** to a writable location. On Windows, double-click
`Start-Replay.cmd` in the project root. The page opens in your default browser.
Keep the console window open; close it or press Ctrl+C to stop the server.

The launcher uses Python 3.11+ when available. Otherwise it downloads a pinned,
SHA-256-verified official Python into `.replay-runtime/` inside this folder.
It needs no administrator privileges, package installation, API key, or simulator
runtime. First launch requires internet if Python or the 2.6 MB instrument model
is missing; subsequent replay is local. The standalone ZIP includes the model.

On macOS/Linux with Python 3.11+ installed, run:

```bash
bash start-replay.sh
```

The launcher normally uses `http://127.0.0.1:8793/replay`. If that port is occupied,
it chooses a free port and opens the matching URL. Repeated launches reuse this
folder's existing replay service. Other running services are left alone.
The existing history viewer is available through **Original logs** at `/`.
Both pages read the same saved files in `results/`.

When no saved runs exist, the bundled sanitized example is installed as
`results/example-rc1mx-hydrolysis/`. To view your own runs, copy complete run folders
into `results/`, then reload the page and choose a recording. Preserve `result.json`,
`manifest.json`, `selection-trace.jsonl`, `execution/trace.jsonl`, and, when available,
`monitor.jsonl`. The 3D scene currently models RC1mx + HFCal; unsupported instruments
are identified in the page rather than given a fabricated animation.

Optional checks (without starting a server):

```powershell
.\Start-Replay.cmd -CheckOnly
# Exercise the local-Python download path even if a system Python exists:
.\Start-Replay.cmd -PortablePython -CheckOnly
```

```bash
bash start-replay.sh --check
```

Advanced scripts for live model experiments and the original workflow demo are
kept under `tools/launchers/`. Those scripts still require the full development
environment and (for real model calls) an API key; they are separate from replay.

The replay page uses recorded events, successful command receipts, sensor readings,
and optional `monitor.jsonl` samples. It does not advance the simulator, send model
requests, or operate an instrument.

## Open your own recording

Copy the run directory into `results/` with this layout, then click **Refresh**
and choose it in **Run record**:

```text
results/
  my-run/
    result.json
    manifest.json
    selection-trace.jsonl
    execution/
      trace.jsonl
    monitor.jsonl
```

Keep the files produced by a single run together. `result.json` alone cannot
reconstruct model conversations or the sequence of instrument operations.
`monitor.jsonl` supplies simulator temperature and composition; it is optional,
and missing monitor data is displayed as unavailable. The generic history reader
also supports a top-level `trace.jsonl` used by some older runs.

For a separate recordings directory or a specific port, use the lightweight
Python entry point:

```bash
python tools/launch_replay.py --results /path/to/recordings --port 8893 --no-browser
```

Then open the URL printed by the launcher. This entry point uses only Python's
standard library; the full development dependencies are not needed for replay.

## What is shared through Git

The repository includes source, English guides, launchers, tests, local Three.js
modules and their license, ChemLab manifests/attribution, and one sanitized sample
archive in `examples/replay/`. This sample enables replay on a clean checkout.

`results/` contains only a tracked `.gitkeep` placeholder. Your run histories,
Codex scratch work and exports (`work/`, `outputs/`, or `result/`), video exports,
`.env`, virtual environments, the portable Python cache, and downloaded ChemLab
assets remain local. The launcher recreates its ignored runtime files as needed.
The bundled sample is documented in [examples/replay](../../examples/replay/README.md).

## Controls

- Select a run at the top. Use Previous/Next, the step menu, the slider, or the
  left/right arrow keys to move through its recorded history.
- **All interactions** includes requests to DeepSeek and Jev as well as instrument
  actions. **Action keyframes** skips intermediate model calls and keeps actions,
  stop events, and the final result.
- Inspect **Request**, **Response**, or **Raw event** in the left panel. Messages
  are loaded from the original trace when selected; sensitive fields are redacted
  using the same rules as the original history viewer.
- Payload sections have descriptive headings and JSON syntax highlighting.
  Jev requests show the actual question criteria as lettered candidate cards
  before the longer context. Expand a card for its full parameters and rationale.
  The response view marks the chosen letter and displays saved model-reported
  probabilities without renormalizing them. The request view contains no answer
  highlights. **Raw event** and **Copy** retain the complete recorded payloads.
- **Play** steps forward automatically. A wait of 120 simulated seconds takes
  one short transition, rather than 120 seconds of wall time. Choose **Instant**
  to remove transitions. Playback pauses when the page is hidden.
- Drag the central model to orbit, scroll to zoom, and use **Focus reactor** or
  **Reset view** to reposition the camera.

## How to interpret state

The right panel deliberately separates two sources:

1. **Simulator ground truth** comes from the independent monitor. It is internal
   simulation data, not a real physical measurement, and was not necessarily
   available to the model.
2. **Last agent readings** show what the workflow actually read. Each field keeps
   its own observation timestamp and revision. Waiting does not refresh a sensor
   reading; old readings remain labeled with their age.

Only samples at or before the selected step are shown. Discrete control changes
come from successful receipts or recorded readings, never merely from a proposed
action. A failed command does not change the replayed instrument state. Runs
without monitor data show that ground truth is unavailable.

The heating indicator prefers recorded heater power if present. Otherwise it
shows **Estimated heating** when the controlled temperature is more than 0.10 °C
below the last accepted target (reactor temperature in Tr mode, jacket temperature
in Tj mode). An accepted ramp's end target is used, so this indicates estimated
heat demand rather than reproducing the controller's instantaneous power. Monitor
temperature is preferred; older agent readings are the fallback. Power off,
emergency, or thermostat off disables the estimated glow. Missing data remains
unknown. The oil heater and visible jacket glow together. The separate HFCal
calibration rod lights only from its recorded `cal_heater.on` state; the orange
agitator color is unrelated to heating. No new simulation is run for this effect.

## Rendering and assets

The RC1mx + HFCal scene uses the original ChemLab GLB. The camera, screen readouts,
status lights, drain angle, and agitator are rendered locally with Three.js.
The agitator rotates continuously at the displayed RPM, even when step playback
is paused. Its speed changes smoothly between steps and stops at zero RPM.
This motion does not advance the replay clock. Transitions between recorded
states compress elapsed simulation time. These visuals are not a new fluid
simulation, an exact reconstruction of the shaft phase, or additional observations.
The replay currently animates RC1mx + HFCal; other instruments retain the message
and state panels with a clear notice that no 3D replay model is available.

Three.js 0.180.0 is vendored with its MIT license under `web/vendor/three/`.
The ChemLab asset is not committed because `vendor/chemlab/assets/` is ignored.
To restore it on another machine, run:

```powershell
.\.venv\Scripts\python.exe tools/fetch_replay_assets.py
```

The downloader pins the upstream revision and verifies the model's SHA-256.
ChemLab asset license and attribution are retained in `vendor/chemlab/`.
Once installed, the replay needs no CDN, external rendering service, or API key.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Missing files or no launcher | Extract the whole ZIP first; use the `rc1mx-hfcal` branch. Do not run from inside an archive. |
| Browser does not open | Copy the URL printed in the launcher window into your browser. Keep that window open. |
| Port 8793 is busy | Use the URL the launcher prints. It checks ports 8793 through 8813 automatically. Alternatively use `Start-Replay.cmd -Port 8893` or `bash start-replay.sh --port 8893`. |
| Python is missing | Windows prepares it inside `.replay-runtime/`. On macOS/Linux, install Python 3.11+ and retry. |
| First-time download fails | Check access to `python.org` (Windows Python) and `huggingface.co` (instrument model), then rerun the launcher. |
| Blank or failed 3D scene | Use a current browser with WebGL enabled. Retry the launcher to verify or restore the model. The message/state panels can still explain a recording. |
| My recording is not listed | Check the folder layout above and that `result.json` is valid JSON. Click **Refresh**. |
| Temperature stays unchanged | Agent readings update only on recorded reads. Check the ground-truth card, simulation clock, and the reading's age. |
| Stirrer moves while paused | Pausing stops step advancement. A recorded nonzero RPM keeps the visual agitator rotating. |

## Validation

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_replay_viewer.py tests/test_result_viewer.py
```

These checks cover old readings across waits, partial reads, rejected operations,
same-time monitor samples, actual stop times, original request preservation,
redaction, and isolation of the original history routes.
