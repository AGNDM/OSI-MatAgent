# Local model and decision comparisons

This package evaluates frozen local models on the sibling [instrument release](../instrument/README.md): 388 static multiple-choice questions and 120 sequential instrument cases. The simulator owns native transitions, correctness, correction feedback and termination. The runner never downloads weights or trains neural models.

The comparison-owned expert shortlist is separate from the actual repository [LangGraph bridge](../graph/README.md). Published protocols and denominators are documented in [results/388](../../results/388/README.md) and [results/120](../../results/120/README.md). Saved-result exporters live in [evaluation/](../evaluation/README.md).

## Entry points

| Module | Purpose |
| --- | --- |
| `comparison.setup` | Create or verify pinned environments; verify/import an explicitly staged Ollama runtime |
| `comparison.run` | Preflight or execute a declared configuration |
| `comparison.features` | Fit classical readouts from frozen features on training/validation cohorts |
| `comparison.study_matrix` | Generate an example model/readout matrix |
| `comparison.study` | Freeze sources, partition cohorts, run one worker and merge saved shards |
| `comparison.aggregate` | Export a flat CSV from a run report |
| `comparison.taxonomy` | Produce evaluator-side failure-mechanism labels |
| `comparison.study_report` | Produce research JSON/CSV, HTML and figures from saved records |
| `comparison.tool_track` | Separate read-only information cohort with transcript-dependent labels |

`backends.py`, `native.py` and `policies.py` implement inference and the benchmark boundary. `agentic.py` implements the comparison-owned shortlist. `feature_io.py`, `feature_models.py` and `features.py` own feature artifacts; `study_metrics.py` owns report statistics.

## Environment and local inputs

Run commands from the repository's **benchmark/ directory, inside an existing Slurm compute allocation**. Allocation guards remain active for setup, preflight, hashing, scoring and inference. Resource allocation and process scheduling belong to the caller; this package contains no scheduler scripts or software-test suite.

```bash
python3.12 -m comparison.setup --profile hf
python3.12 -m comparison.setup --profile features
python3.12 -m comparison.setup --profile apus
```

The profiles create `.venv-comparison-hf`, `.venv-comparison-features` and `.venv-comparison-apus` with exact dependencies. `--profile native` creates the lightweight `.venv-comparison` environment. Existing environments are verified without changing packages. APUS uses its separate Transformers environment. Setup installs no weights.

Use a compatible existing interpreter as `$PYTHON` below. Stage complete local checkpoints/tokenizers and trusted fitted artifacts outside source packages. Examples use `${COMPARISON_QWEN_MODEL}`, `${COMPARISON_OPENJEV_MODEL}` and `${COMPARISON_APUS_MODEL}`; only selected experiments need their paths resolved. Relative custom model/artifact paths resolve against the configuration directory.

Keep output outside `comparison/` and `instrument/`. Each run requires a fresh directory, normally ignored `benchmark/output/` or repository `results/`. Set environment controls for the allocated process:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
unset PYTHONHOME PYTHONPATH
```

## Run a configuration

```bash
export COMPARISON_QWEN_MODEL=/absolute/path/to/qwen2.5-3b
"$PYTHON" -m comparison.run --config comparison/configs/hf-qwen.example.json \
  --only qwen-hf-logprobs-seed0 --track mcqa --preflight --output output/preflight
"$PYTHON" -m comparison.run --config comparison/configs/hf-qwen.example.json \
  --only qwen-hf-logprobs-seed0 --track mcqa --limit 1 --output output/diagnostic
```

Preflight validates configuration and the release without loading models. `--limit 1` is a diagnostic prefix; omit it for the full configured cohort. MCQA defaults to 114 held-out questions; `tracks.mcqa.split="all"` includes all 388 training/validation/test diagnostics. Use `--track sequential` explicitly for the separate sequential workload; its mode comes from `tracks.sequential.mode`, and the release has no assigned train/test split.

HF supports plain choice, complete-menu token logits, verbal confidence and strict sampling. Sampling retains all draws and invalidates the decision if any declared draw is malformed. A verbal scalar never becomes an invented full probability vector. Native head probabilities are not presumed calibrated.

OpenJev requires its released pointer head and `wev-ai==0.1.0`. APUS requires its shipped `openjet_runtime`; low/high use actual trained depths. Native choice/context limits remain explicit errors. The optional `qwen25-yarn-v1` extension is an explicitly recorded variant; default loads retain checkpoint configuration.

## Ollama

Stage the pinned executable identified by `ollama-runtime.lock.json`, a dedicated local model store and a matching HF tokenizer. Manage a loopback server inside the allocation yourself; no pull or server launch is automatic.

```bash
export COMPARISON_OLLAMA_BIN=/absolute/path/to/ollama
export OLLAMA_MODELS=/absolute/path/to/dedicated-model-store
export OLLAMA_HOST=127.0.0.1:18000
export COMPARISON_OLLAMA_URL=http://127.0.0.1:18000
export OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=0
"$COMPARISON_OLLAMA_BIN" serve > /absolute/path/to/server.log 2>&1 &
OLLAMA_SERVER_PID=$!
# Once the server is ready:
"$PYTHON" -m comparison.setup --verify-ollama
"$PYTHON" -m comparison.setup --import-ollama \
  --model-path "$COMPARISON_QWEN_MODEL" --ollama-model qwen25-3b-local:imported
"$PYTHON" -m comparison.run --config comparison/configs/ollama-qwen.example.json \
  --only qwen-ollama-choice-seed0 --track mcqa --output output/ollama-run
kill "$OLLAMA_SERVER_PID"
```

Import is needed only for a fresh store and never overwrites a tag. Preserve the recorded runtime, model digest, conversion precision, tokenizer and template identities. The strict adapter checks locally formatted prompt tokens against server consumption. Ollama supports generative readouts here, not complete-menu logprobs.

## Frozen features and shortlist

`comparison.features` uses a frozen SentenceTransformer or frozen LLM hidden states. Only 232 MCQA training questions fit candidate classifiers; 42 validation questions fit cosine temperature. The 114 held-out questions remain separate. kNN stores training features. Sequential use transfers the MCQA readout without fitting sequential labels; all-388 figures include training/validation diagnostics.

Each decision artifact requires `decision.pkl` and `manifest.json`. The majority reference requires `majority.json` and `majority-manifest.json`. Manifests authenticate fitting cohorts, feature sources, model/tokenizer identity and runtime versions. Load only trusted local pickles. Neural weights are not trained.

Set `architecture="expert_decision"`, a local HF `expert`, a chooser `backend` and `method` for the comparison-owned expert shortlist. Its schema/menu gate checks proposals; the native evaluator checks correctness. Probabilities remain conditional on that shortlist. This is distinct from the repository graph's specialist/menu protocols.

## Prepare and execute disjoint workers

```bash
"$PYTHON" -m comparison.study prepare --config /absolute/path/to/config.json \
  --output /absolute/path/to/fresh-study --track mcqa --shards 8 \
  --hf-python /absolute/path/to/hf/python --apus-python /absolute/path/to/apus/python
```

Configurations containing Ollama additionally need `--ollama-bin` and `--ollama-models`. Preparation freezes code, configuration and the instrument release, then writes `study.json` and `worker-commands.json`. It launches no workers. The manifest records an interpreter/argument list, working directory and additional environment for every worker. Start each entry once inside a suitable allocation. For example, from `STUDY/snapshot/`:

```bash
"$HF_PYTHON" -m comparison.study worker --study "$STUDY" --group hf --index 0
"$HF_PYTHON" -m comparison.study merge --study "$STUDY"
```

Run all declared workers before a complete merge. `--allow-partial` exposes missing coverage. `--max-parallel` remains a recorded concurrency hint; the caller owns actual concurrency. Ollama workers require the caller-managed server described above. Never edit a prepared snapshot or rerun a completed worker into existing output.

`study_matrix` is an example generator, not the canonical 34-configuration result manifest. Prepared studies pin resolved paths and source identities. `tool_track prepare` likewise emits `worker-commands.json`; execute its `comparison.tool_track worker --study STUDY --index N` entries from the frozen snapshot, then merge. Its derived conditional-label cohort stays separate from static MCQA.

## Saved reports and interpretation

```bash
"$PYTHON" -m comparison.aggregate --report output/RUN/report.json \
  --output output/RUN/comparison.csv
"$PYTHON" -m comparison.study_report --bundle instrument --taxonomy "$TAXONOMY" \
  --runs "$STUDY/runs" --plan "$STUDY/snapshot/configuration.json" --output "$FRESH_REPORT"
```

Reports preserve accuracy and attempt/valid/confidence denominators; calibration, selected-correctness Brier, multiclass Brier, AUROC, selective coverage and uncertainty are reported only where supported. Missing and invalid outputs remain visible. Wall latency and known token counts are not active GPU time or invoices.

Sequential reports separate original proposals, reference-corrected completion and autonomous success. Transport/context/unsupported-runtime failures remain explicit. Hierarchical node probabilities are not calibrated full-action probabilities. L1–L5 are construction labels, not an established difficulty ordering. Independent expert review and real-hardware validity remain separate limitations. Models, native runtime code and data retain their own licences and attribution.
