# Five-Minute Local Demonstration

> Historical drying demonstration with authored observations/fixtures and scripted defaults. It is not the current generic native task solver; use the [single-instrument guide](../guides/single-instrument-agent.md).

The browser demonstrates bounded drying troubleshooting: a native oven process runtime plus an explicitly authored balance layer. Device expansion is available separately through the CLI; the browser does not currently list all devices.

## Start

```bash
conda activate osi-hack
python tools/serve_openjev.py
```

In another terminal:

```bash
conda activate osi-hack
python -m app.demo --port 8793 --output results/my-drying-demo
```

Open `http://127.0.0.1:8793`. Do not start another model process if port 8792 already serves the intended model. The default UI modes use scripted specialists with either rules or real local OpenJev routing. They do not call the cloud expert.

Stop the foreground demo with Ctrl+C. Stop the model started by the helper with `python tools/serve_openjev.py --stop`. No system startup service is installed.

## Suggested sequence

1. **0:00-0:40:** Identify the specialist backend and router. Explain native versus authored behavior and the independent evaluator.
2. **0:40-2:00:** Run `dev-01` with scripted specialists and real OpenJev routing. Follow the inspection, proposed correction, gate and fresh weighing.
3. **2:00-2:40:** Load the saved recording. Show the RECORDED label and reset/action replay. Replay makes no model calls and is not mid-episode checkpoint recovery.
4. **2:40-4:00:** Show a contrasting measurement-fault case and a retained real-model failure. Explain which evidence warrants a different action.
5. **4:00-5:00:** Show [the checked-in result summary](../reports/evaluation-summary.md), the device expansion and [remaining work](../status/checklist.md). Separate scripted regression from real-expert evidence.

Every new episode resets its world. Saved runs can be reloaded after a server restart. The UI's observer is isolated from experiment execution, so a display failure cannot change the score or trace.

## Real device experts

Use the budgeted CLI for the current DeepSeek device debugging allowance:

```bash
python -m bench.device_tasks --backend deepseek --router openjev \
  --device mt_xpr204e --output results/my-device-demo
```

This explicitly sends API requests and uses the shared device ledger. Do not use older demo/expert entry points to bypass that ledger. See [Device workflow](../guides/device-workflow.md).

## Interface language

The local drying interface uses English for controls, mode names, workflow nodes, status, metrics, help text and replay controls. The page declares `lang="en"`. A browser-verified offline `dev-01` run completed successfully with no Chinese visible text or UI error; no cloud calls were made.
