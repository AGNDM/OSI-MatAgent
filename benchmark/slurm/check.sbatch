#!/usr/bin/env bash
#SBATCH --job-name=benchmark-check
#SBATCH --account=liac
#SBATCH --partition=mig12gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=results/benchmark-check-%j.log
set -euo pipefail
: "${SLURMD_NODENAME:?Use a real Slurm compute allocation}"
: "${SLURM_JOB_ID:?Use sbatch}"
: "${BENCHMARK_ROOT:?Set the absolute repository root}"
: "${EVALUATION_PYTHON:?Set the absolute graph/features interpreter}"
if [[ "$BENCHMARK_ROOT" != /* || "$EVALUATION_PYTHON" != /* || ! -x "$EVALUATION_PYTHON" ]]; then
  printf '%s\n' 'BENCHMARK_ROOT and EVALUATION_PYTHON must be absolute; the interpreter must exist.' >&2
  exit 2
fi
for benchmark_binding_variable in ${!SLURM_CPU_BIND@}; do unset "$benchmark_binding_variable"; done
export SLURM_CPU_BIND=none
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
unset PYTHONHOME PYTHONPATH
cd "$BENCHMARK_ROOT"
srun "$EVALUATION_PYTHON" -B benchmark/tools/check.py \
  --root "$BENCHMARK_ROOT" \
  --output "${BENCHMARK_CHECK_OUTPUT:-$BENCHMARK_ROOT/results/benchmark-check-$SLURM_JOB_ID}" "$@"
