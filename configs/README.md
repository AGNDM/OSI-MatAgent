# Configuration Guide

`experts.template.json` defines separate reference and contrast general experts. Unselected models remain null. The expert endpoint/model identity is separate from the local OpenJev router, and credentials are represented only by environment-variable names.

`siliconflow-deepseek-v4.1-flash.json` configures the explicitly authorized `https://api.siliconflow.cn/v1` endpoint and `deepseek-ai/DeepSeek-V4.1-Flash` model using `SILICONFLOW_API_KEY`. The contrast expert is still unselected. The budgeted device entry point is `python -m bench.device_tasks --backend deepseek`; the shared 5M ledger survives new output directories.

`device-debug-sample.json` declares the 20-device, 16-category development sample. It is a tool-interface debugging sample, not a scientific benchmark or a randomized performance study.

Local OpenAI-compatible expert endpoints use loopback HTTP. An optional revision is a declared provenance value; returned model identity is recorded separately. No implicit fallback replaces an unconfigured expert with OpenJev.

`proposal_v1.json` and `proposal_v2.json` describe retired free-model batch plans. They must not be executed or resumed as OpenRouter free batches. Current free-model use is restricted to small, explicitly requested debugging.
