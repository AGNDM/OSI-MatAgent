#!/usr/bin/env bash
# Shared execution environment; source after entering the benchmark/snapshot root.
[[ -n "${SLURMD_NODENAME:-}" && -n "${SLURM_JOB_ID:-}" ]] || {
  printf '%s\n' 'Comparison commands require a real Slurm compute allocation.' >&2
  return 2
}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
# A submission made inside another srun step can inherit that step's CPU mask.
# Let the new allocation supply its CPU set instead of reusing the parent mask.
unset SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
unset SRUN_CPU_BIND_LIST SRUN_CPU_BIND_TYPE SRUN_CPU_BIND_VERBOSE
export SLURM_CPU_BIND=none SRUN_CPU_BIND=none

comparison_inference_environment() {
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
  export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
  export TOKENIZERS_PARALLELISM=false
}
