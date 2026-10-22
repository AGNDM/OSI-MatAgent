# Local model and decision-mechanism comparisons

Evaluate frozen local models on the sibling `../instrument/` release: 388 static multiple-choice questions and 120 sequential instrument cases. No model training, hosted API credentials or model download is required by the runner. Each experiment uses the same public evidence and offered choices. The benchmark simulator owns state transitions, correctness, correction feedback and termination.

This package owns the static/direct comparison and its optional expert-shortlist pipeline. The actual repository LangGraph bridge, specialist menu protocols and full teacher-forced expert/chooser grid live in [`../graph/`](../graph/README.md). These are distinct evaluation protocols. Published static exports are documented in [`results/388`](../../results/388/README.md), and the completed graph study in [`results/120`](../../results/120/README.md).

| Backend | Supported readouts | Probability meaning |
| --- | --- | --- |
| Hugging Face causal LLM | Plain choice, token logits, verbal confidence, sampling | Conditional menu-token softmax; selected-answer self-confidence; or observed frequencies of at least ten full generated answers |
| Ollama local model | Plain choice, verbal confidence, sampling | Same generative readouts; this adapter does not implement complete-menu logprobs |
| OpenJev-8B with `wev-ai` | Native decision | Released learned pointer-head distribution |
| APUS-Reflex | Native low/high effort | Released candidate-projection model at its trained depths |
| Frozen MiniLM / local LLM features | Cosine, logistic probe, kNN | Candidate softmax; cosine temperature fitted on validation only |
| References | Uniform, training-label prior, public T1 rules | Explicit reference scope; rules are undefined outside audited cases |

HF and Ollama are inference runtimes. A model checkpoint is a separate choice: add another experiment with a different local snapshot/tag to compare models. Use the same checkpoint and evidence to compare readouts; comparisons between different models measure the complete model/readout systems. Native decision-head outputs are not presumed calibrated on these tasks.

## Layout and environments

Run submission commands from the repository's **`benchmark/` directory**, where `comparison/` and `instrument/` are siblings. Every installation, preflight, checksum, score and inference command runs in Slurm on SCITAS Kuma. Launchers use account `liac`; override it if necessary.

| Entry point / directory | Responsibility |
| --- | --- |
| `run.py`, `configs/` | Run declared model/readout experiments, or preflight without loading weights |
| `backends.py`, `native.py`, `policies.py` | Offline HF/Ollama/native inference and the benchmark decision boundary |
| `agentic.py` | Comparison-owned expert shortlist; separate from the repository graph |
| `features.py`, `feature_io.py`, `feature_models.py` | Frozen encoders, train/validation fitting and authenticated feature artifacts |
| `study.py`, `study_matrix.py` | Freeze sources/configuration, partition cohorts, generate submissions and merge shards |
| `aggregate.py`, `study_report.py`, `study_metrics.py`, `taxonomy.py` | Tables, research reports, metrics and evaluator-side grouping |
| `tool_track.py` | Separate derived read-only information-gathering cohort with transcript-dependent labels |
| `setup.py`, `requirements*.txt`, `slurm/` | Pinned environments and cluster launcher templates |
| `tests/` | Unit and integration checks, executed inside a Slurm allocation |

Keep checkpoints, fitted artifacts and generated runs outside `comparison/` and the frozen `instrument/` release. The direct runner and study preparer reject outputs nested in their source packages. Typical writable destinations are `benchmark/output/` for diagnostics and repository `results/` for retained studies. Generated receipts, environments, caches and Python bytecode are not source files; preserve any retained receipts with the study that they authenticate.

```bash
mkdir -p output
sbatch comparison/slurm/setup.sbatch --profile native
sbatch comparison/slurm/setup.sbatch --profile hf
# Frozen features and report rendering use their own dependency profile.
sbatch comparison/slurm/setup.sbatch --profile features
# APUS uses a separate environment because it requires Transformers 5.16.1.
sbatch comparison/slurm/setup.sbatch --profile apus
```

Profiles create `.venv-comparison`, `.venv-comparison-hf`, `.venv-comparison-features` and `.venv-comparison-apus` with pinned dependencies. Setup verifies an existing environment without changing its packages. Alternatively set an absolute `COMPARISON_PYTHON` to a compatible staged environment. No weights are installed by setup. Environments, model caches and generated `output/` are ignored by Git.

The `.sbatch` files are Kuma resource templates, not portable hardware detection. Override account, partition, QoS, memory, CPU/GPU count and time on `sbatch` for the target cluster. The launchers share `slurm/common.sh`: it requires a compute allocation, removes ambient Python paths, and clears inherited CPU-binding masks before each `srun`. Generated study submissions also clear parent-step binding before invoking `sbatch`. Thread counts still follow the new allocation's `SLURM_CPUS_PER_TASK`; model settings and evaluation budgets are unchanged by this environment setup.

## Hugging Face models

Provide an already downloaded local model snapshot, including tokenizer, configuration and all weight shards. HF loading is offline and does not execute custom remote model code. The starter configuration uses Qwen2.5-3B-Instruct; other causal models supported by the installed Transformers version can use the same backend.

Example configuration model paths use `${COMPARISON_QWEN_MODEL}`, `${COMPARISON_OPENJEV_MODEL}` and `${COMPARISON_APUS_MODEL}`. Export the variables needed by the selected experiments; unselected examples need not resolve. Relative model/artifact paths in a custom configuration resolve against that configuration's directory. Weights and fitted pickles are external inputs and must not be added to this package. The optional `qwen25-yarn-v1` context extension is an explicitly recorded Qwen2.5 configuration variant; default model loading keeps the checkpoint configuration unchanged.

```bash
export COMPARISON_QWEN_MODEL=/absolute/path/to/qwen2.5-3b
sbatch comparison/slurm/run.sbatch --config comparison/configs/hf-qwen.example.json \
  --only qwen-hf-logprobs-seed0 --limit 1
```

That command is a diagnostic prefix: one MCQA question and one sequential case in each mode. Omit `--limit` only when ready for the full configured cohorts. MCQA defaults to the **114 held-out questions**, with `tracks.mcqa.split="all"` available for separately labelled all-388 diagnostics. Sequential defaults to all 120 cases in teacher-forced and free-run modes. The sequential release has no assigned train/test split.

The HF example declares plain-choice, token-logit, verbal-confidence and ten-sample experiments. `--only` selects a named experiment; omit it to execute the whole configuration. Repeat configurations with predeclared seeds for a model study. Each invocation needs a fresh output directory; launchers choose a job-specific directory by default.

## Ollama on a compute node

Use a pinned **Ollama 0.34.0** executable from an existing installation or the official release identified in `ollama-runtime.lock.json`. The launcher starts a loopback server inside the assigned GPU job and shuts it down afterward. It uses a dedicated model store and never pulls weights automatically.

To import the same local Qwen checkpoint for a runtime comparison:

```bash
export COMPARISON_OLLAMA_BIN=/absolute/path/to/ollama
export COMPARISON_QWEN_MODEL=/absolute/path/to/qwen2.5-3b
sbatch comparison/slurm/import-ollama.sbatch \
  --model-path "$COMPARISON_QWEN_MODEL" --ollama-model qwen25-3b-local:imported
```

Wait for a successful import, then submit:

```bash
sbatch comparison/slurm/ollama.sbatch --config comparison/configs/ollama-qwen.example.json \
  --only qwen-ollama-choice-seed0 --limit 1
```

The default store is `benchmark/.cache/comparison-ollama/models`; `COMPARISON_OLLAMA_MODELS` selects another dedicated store. Import records original weight hashes, conversion details, Ollama version, binary hash and resulting model digest. An existing tag is never overwritten. GGUF conversion/runtime precision can affect outputs, even with the same source checkpoint; record those details rather than assuming identical inference.

The checked Qwen import produced F16 GGUF weights; the HF example also selects float16 for this runtime comparison. The model tag is a user-defined name; the recorded model details and digest identify the actual precision and weights.

The strict adapter requires the matching local HF tokenizer. It formats the full prompt locally, sends raw generation, checks the context budget and verifies the server consumed the expected tokens. A mismatch is an explicit failure. Complete-menu logprobs are not implemented here; returned top-k token information would need separate coverage validation. See the official [Ollama import guide](https://docs.ollama.com/import) and [generation API](https://docs.ollama.com/api/generate).

## Dedicated decision models

```bash
export COMPARISON_OPENJEV_MODEL=/absolute/path/to/openjev8b
sbatch comparison/slurm/run.sbatch --config comparison/configs/openjev.example.json --limit 1

export COMPARISON_APUS_MODEL=/absolute/path/to/apus-reflex/4B-5949
COMPARISON_PYTHON="$PWD/.venv-comparison-apus/bin/python" \
  sbatch comparison/slurm/run.sbatch --config comparison/configs/apus.example.json --limit 1
```

OpenJev requires its learned pointer head and `wev-ai==0.1.0`. APUS requires the official `openjet_runtime` shipped with its snapshot and the separate pinned Transformers environment. APUS supports at most 16 choices per native decision. The sequential interface uses hierarchy nodes with up to 20 labels, so some APUS nodes are explicitly unsupported. OpenJev also has a limited native context. These failures remain visible; neither evidence nor menus are silently pruned. APUS low/high select actual trained depths, not different prompting language.

## Outputs and interpretation

Each run records source/configuration hashes, resolved model paths, runtime identities, weight/tokenizer hashes, raw replies, actual token counts where available, and timing. Each experiment writes MCQA predictions/scores or sequential episodes, per-step feedback and scores; `report.json` combines them. This compute-job payload exports a flat table:

```bash
.venv-comparison/bin/python -m comparison.aggregate --report output/RUN/report.json --output output/RUN/comparison.csv
```

MCQA reports frozen accuracy/coverage/breakdowns, top-label ECE and Brier of selected-answer correctness when confidence exists. Full multiclass Brier requires a complete probability vector. A verbal confidence scalar never becomes an invented vector. Sampling keeps every draw; under the strict protocol any malformed draw invalidates the decision, with its actual responses and usage retained.

Sequential scores separate proposed-action/control accuracy, corrected completion and autonomous success. A malformed model reply is scored wrong and corrected in teacher-forced mode; transport, context or unsupported-model errors remain infrastructure failures. Hierarchical command/page/argument probabilities are recorded at their nodes. They are not presented as calibrated full-action probabilities. Diagnostic prefix summaries retain a separate all-120-case report with missing coverage explicit.

## Agentic pipeline

Set `architecture="expert_decision"`, provide a local HF `expert` specification and the chooser `backend`, and select its readout with `method`. At every decision the expert receives the complete public context and proposes a fixed number of offered candidates (up to four), with a diagnosis and rationale. A schema and menu-membership gate validates those proposals. The independent chooser then selects from the shortlist; the benchmark executes the action, supplies native feedback, applies any teacher-forcing correction and verifies the goal. Direct and agentic architectures are reported separately.

This comparison-owned pipeline implements the expert/proposal/decision pattern. It does not claim to run the colleague application's hosted graph unchanged. The gate checks the proposal schema and public identifiers; correctness is checked by the native evaluator. Private reference-path candidate coverage is measured only after execution. Shortlist probabilities remain conditional on that shortlist and never become invented full-menu Brier scores.

## Frozen features and fitting

The `features` setup profile adds exact feature and plotting dependencies. Fit artifacts with `python -m comparison.features` inside a GPU allocation, specifying the bundle, local model path, repository, revision and a fresh output directory. `--encoder-kind embedding` uses frozen SentenceTransformer features; `hf_hidden` uses mean-pooled frozen LLM hidden states. Every token contributes through declared nonoverlapping chunks. The text protocol preserves the complete public messages and candidate descriptions, including all sequential history and corrections.

Only the 232 MCQA training questions fit the candidate classifiers. Only the 42 validation questions fit the cosine temperature; 114 held-out questions remain separate. kNN stores training features. The optional `--label-efficiency` curve samples nested budgets of independent training scenario groups with three seeds. Sequential use transfers the fitted MCQA feature decision without fitting on sequential labels. All-388 figures contain training/validation diagnostics and must not be presented as held-out performance.

Artifacts include mandatory checksums, fitting identities, disjoint cohort IDs/groups, encoder weight/tokenizer identities and exact runtime versions. Only locally produced, trusted pickle artifacts should be loaded. No neural model weights are trained.

Each fitted decision directory contains `decision.pkl` and `manifest.json`; keep them together and point a `features` backend's `artifact_path` at that directory. The majority-label reference requires `majority.json` and `majority-manifest.json` in its artifact directory. Model/encoder paths and artifact paths are independently configured: moving one does not silently substitute another. Graph use consumes these same authenticated artifacts through its chooser adapter, with its separately documented projected context.

## Parallel studies and research report

`python -m comparison.study_matrix` writes a declared local matrix. `python -m comparison.study prepare` freezes source, configuration and the sibling instrument release into a fresh study directory, partitions each configured cohort into disjoint shards, and writes `study.json` plus a reviewable `submit.sh`. Preparation does not submit jobs. Execute that script through normal Slurm submission authorization. The generated arrays use separate HF, Ollama and APUS environments; they can run concurrently. Never edit a study's snapshot while jobs use it.

`study_matrix` is an example matrix generator, not the canonical list of the 34 completed static configurations. Its `--project` argument expects staged weights beneath `external/models/`; `--feature-root` and `--qwen3` are explicit optional inputs. For another weight layout, supply a configuration using the same schema instead of changing evaluation code. Preparation resolves and records absolute local model/interpreter paths; a prepared study is an immutable execution record, not a relocatable template.

A prepared study separates `snapshot/` (copied code, configuration and instrument release), `runs/` (shard outputs), `worker-receipts/`, `logs/`, `study.json`, `submit.sh` and `submitted.tsv`. Merge outputs default to `merged/`. The submission receipt prevents accidental duplicate arrays; partial merges expose missing tasks rather than dropping them. `tool_track` uses its own `tool-study.json` and copied read-only runtime and must not be pooled with the original static MCQA cohort.

`python -m comparison.study merge --study STUDY` authenticates source and model identities and rejects overlaps, foreign IDs and incomplete coverage unless partial reporting is explicitly requested. `python -m comparison.taxonomy` creates evaluator-side failure-mechanism labels from declared fixture, workflow and fault metadata. Healthy members of matched pairs retain the target mechanism with a separate healthy-condition flag. The original task IDs and evaluator remain unchanged.

`python -m comparison.study_report --bundle instrument --taxonomy TAXONOMY.json --runs STUDY/runs --plan STUDY/snapshot/configuration.json --output FRESH_REPORT` produces standalone HTML, JSON/CSV results, task records and exported figures. It reports accuracy, calibration coverage, Brier/ECE/AUROC, selective coverage, hazard and information-request metrics where supported, latency/known token counts, clustered confidence intervals, paired comparisons with Holm correction, per-seed results and explicit pending/error coverage. Mechanism, condition and instrument are the primary breakdowns; L1-L5 remain legacy construction labels. Repeated seeds and related scenario variants are not treated as independent cases.

Rationale citation coverage is shown explicitly; a label-only prediction supplies no citation. Agreement with contract-derived cause IDs is evidence grounding, not proof of causal explanation. Wall latency is not relabelled as active GPU time. Independent human expert review and real-hardware validity remain pending. Newly added study, feature, agentic and reporting code requires Slurm validation before a full-study result is claimed.

## Verify

On Kuma, submit from the frozen snapshot directory. Newly generated submission
scripts enter that directory before invoking `sbatch`, so jobs inherit the
correct `SLURM_SUBMIT_DIR` even when the caller starts elsewhere. `--chdir` alone
does not replace setting the submission directory for these launchers.

The generic launchers default to an H100 allocation; they do not automatically
route methods between MIG, L40s and H100 resources. Retained studies can have
their own separately recorded scheduling. Check GPU names, memory, CPU limits
and available quotas against the target Slurm partition before overriding a
template. Hardware compatibility and completed shard coverage remain separate
checks; a submitted array is not a finished experiment.

Inside a Slurm CPU allocation, from `benchmark/`:

```bash
.venv-comparison-features/bin/python -m unittest discover -s comparison/tests -v
.venv-comparison/bin/python -m comparison.run --config comparison/configs/hf-qwen.example.json \
  --only qwen-hf-logprobs-seed0 --preflight --output output/preflight
```

Preflight validates the frozen release, schema, selected configuration and output policy without loading models. Real model/context compatibility still requires a small GPU integration run. The sibling instrument package's licences, notices and dataset limitations continue to apply. Model checkpoints and runtime executables retain their own licences.
