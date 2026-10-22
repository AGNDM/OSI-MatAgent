# Model configuration

Expert endpoint identity is independent of the decision model. Credentials are environment-variable names, never literal keys.

| Configuration | Use |
|---|---|
| `openrouter-deepseek-v4.1-flash.json` | Default native-task expert |
| `openrouter-deepseek-v4.1-flash-benchmark.json` | Low-reasoning benchmark expert |
| `openrouter-minimax-m3.1-flash-low.json` / `-max.json` | Paid MiniMax benchmark expert with explicit effort |
| `openrouter-minimax-m3.1-flash-free*.json` | Exact free endpoint; no automatic paid fallback |
| `siliconflow-deepseek-v4.1-flash.json` | Explicit SiliconFlow alternative |
| `experts.template.json` | Separate reference and contrast endpoints |

OpenRouter uses `OPENROUTER_API_KEY`. SiliconFlow uses `SILICONFLOW_API_KEY`. Loopback endpoints serve optional local experts. The sequential benchmark selects Jev or Luna through `--decision-model`; the native CLI selects hosted Jev through `--router openrouter-jev --operation openrouter-jev`.

Reasoning effort, completion allowance and request timeout are explicit. Reasoning text is excluded from stored responses while token usage is retained. Supported OpenRouter calls use record-only accounting; SiliconFlow retains its shared 5M-token policy. Finite episode and request limits still apply.

`proposal_v1.json` and `proposal_v2.json` are retired plan inputs, not supported batch entry points. `device-debug-sample.json` is an offline integration sample. See the [benchmark guide](../docs/guides/instrument-eval-langgraph.md).
