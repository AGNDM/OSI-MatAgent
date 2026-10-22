#!/usr/bin/env bash
# Source only from the allocated Slurm launchers below.
source comparison/slurm/common.sh
comparison_inference_environment
# The strict Ollama adapter uses the matching local HF tokenizer for context
# accounting, so evaluation requires the HF profile even though HTTP is stdlib.
comparison_python="${COMPARISON_PYTHON:-$PWD/.venv-comparison-hf/bin/python}"
comparison_ollama="${COMPARISON_OLLAMA_BIN:?Set COMPARISON_OLLAMA_BIN to the pinned Ollama executable}"
[[ "$comparison_python" == /* && -x "$comparison_python" && "$comparison_ollama" == /* && -x "$comparison_ollama" ]] || {
  printf '%s\n' 'Absolute executable COMPARISON_PYTHON and COMPARISON_OLLAMA_BIN paths are required.' >&2
  exit 2
}
export COMPARISON_OLLAMA_BIN="$comparison_ollama"
export OLLAMA_HOST="127.0.0.1:$((18000 + SLURM_JOB_ID % 10000))"
export COMPARISON_OLLAMA_URL="http://$OLLAMA_HOST"
export OLLAMA_MODELS="${COMPARISON_OLLAMA_MODELS:-$PWD/.cache/comparison-ollama/models}"
export OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH="${COMPARISON_OLLAMA_CONTEXT:-32768}"
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=0
mkdir -p output "$OLLAMA_MODELS"
"$comparison_ollama" serve >"output/comparison-ollama-server-$SLURM_JOB_ID.log" 2>&1 &
comparison_ollama_pid=$!
trap 'kill "$comparison_ollama_pid" 2>/dev/null || true; wait "$comparison_ollama_pid" 2>/dev/null || true' EXIT
comparison_server_ready=false
for ((comparison_attempt=0; comparison_attempt<60; comparison_attempt++)); do
  kill -0 "$comparison_ollama_pid" 2>/dev/null || {
    printf '%s\n' 'Allocated Ollama server exited; inspect its log.' >&2
    exit 2
  }
  if curl --noproxy '*' -fsS "$COMPARISON_OLLAMA_URL/api/version" >/dev/null 2>&1; then
    comparison_server_ready=true
    break
  fi
  sleep 1
done
[[ "$comparison_server_ready" == true ]] || {
  printf '%s\n' 'Ollama did not start inside the allocation.' >&2
  exit 2
}
srun "$comparison_python" -m comparison.setup --verify-ollama \
  >"output/comparison-ollama-runtime-$SLURM_JOB_ID.json"
