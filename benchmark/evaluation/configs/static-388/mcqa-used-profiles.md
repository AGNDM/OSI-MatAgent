# Exact used static inference profiles

These two profiles preserve the actual 22 + 12 configuration inputs behind the
published 34-configuration MCQA snapshot. They retain separate historical source
identities in mcqa-used-profiles-provenance.json; do not describe them as one
historical source freeze or as the later repository graph protocol.

Fill a COPY of mcqa-bindings.example.json with absolute staged paths. The existing
evaluation/workflow/render_plan.py resolves either profile with those explicit
bindings. This only changes paths and removes historical sequential track settings;
all experiment inference fields are checked against authenticated worker inputs.

Run all preparation, hashing and Python below inside real Slurm allocations.
For EACH profile, render a fresh configuration with render_plan.py --plan PROFILE
--bindings YOUR_BINDINGS --output FRESH_CONFIG. Then from benchmark/ prepare a
separate fresh study:

    "$PYTHON" -m comparison.study prepare --config "$FRESH_CONFIG" \
      --output "$FRESH_STUDY" --track mcqa --shards 8 \
      --hf-python "$HF_PYTHON" --apus-python "$APUS_PYTHON"

The 22-row R2 profile additionally requires --ollama-bin "$OLLAMA_BIN" and
--ollama-models "$OLLAMA_MODELS". Stage the pinned Ollama executable and dedicated
store with the exact qwen25-3b-local:imported tag/digest and recorded F16/model,
tokenizer and template identities. No pull/import is automatic; recreating a tag
or changing conversion settings requires a new runtime identity. Read the original
binary/runtime lock and per-experiment identities in the provenance manifest.

Inspect each fresh study.json and worker-commands.json before execution. The command
manifest records exact argv, working directory and environment for each task;
execute each recorded command through your own compute allocation. For direct
diagnostics use comparison.run --track mcqa explicitly. Neither plan
authorizes or implies any 120-case sequential workload. All 388 MCQA rows include
232 train and 42 validation diagnostics; report the 114 held-out rows separately.

Preserve strict ten-draw sampling with declared seeds 0/1/2, cited-confidence
variants, original menu seed 42, native effort depths, the four comparison-owned
shortlist pairs (384-token expert, max 4 candidates), direct 192-token caps and
restricted T1 reference. The 12-row profile requires externally staged, trusted
frozen feature artifacts and their complete original manifests; no pickle is
bundled or loaded by this generator. Retraining or changing library/source/encoder
identities defines another artifact. This is not the v11 teacher-forced graph grid.
