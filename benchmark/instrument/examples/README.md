# Examples and policy adapters

## Static MCQA

`prediction.jsonl` is one schema example using an actual task ID and its first offered choice, without consulting the answer key. It is not a complete result; omitted questions count as incorrect.

`public-prompt.jsonl` contains the same case's exact public evidence text. `qwen-baseline-protocol.json` records the initial Qwen run's decoding, aggregation and aggregate results. These files do not invoke inference.

## Sequential decisions

`sequential-case.json` shows a public initial case. `sequential_policy.py` demonstrates the public menu callback and deliberately picks the first option. It is an API example, not a solver:

```bash
# Submit from the distribution root after setup.
.venv/bin/python eval/sequential.py run --policy examples.sequential_policy:factory --mode both --limit 1 --output output/example-policy
```

A factory receives public context containing the protocol, mode, shared parameter bank and menu builder. Its policy implements `choose(view, history, options)` and returns a zero-based offered-menu index, or implements `next_action(view, history)` and returns an offered action. The evaluator retains private fixtures, reference actions and scoring rules. Machine/command/parameter selections form one scored native action. See the [sequential guide](../sequential_120/README.md) for correction feedback and metrics.

`sequential_qwen.py` adapts separately staged local Qwen weights to finite-label selection. Prepare `.venv-models` with `python3.12 tools/setup.py --model`, set `SEQUENTIAL_MODEL_PATH` to the absolute local model directory, and use `--policy examples.sequential_qwen:factory` with a GPU allocation. Weights are never downloaded by this adapter. Record weights, decoding and resource settings with each result.
