#!/bin/bash
: "${SLURMD_NODENAME:?Use sbatch or srun in a real compute allocation}"
: "${SLURM_JOB_ID:?A real Slurm job is required}"
: "${GRAPH_PACKAGE:?Set GRAPH_PACKAGE to the absolute graph package directory}"
# A controller's CPU mask is allocation-specific and must not reach child steps.
for graph_binding_variable in ${!SLURM_CPU_BIND@}; do unset "$graph_binding_variable"; done
export SLURM_CPU_BIND=none
GRAPH_RUNTIME_ROOT="$GRAPH_PACKAGE"
if [[ ! -d "$GRAPH_RUNTIME_ROOT/graph_local" && -d "$GRAPH_RUNTIME_ROOT/../graph_local" ]]; then
  GRAPH_RUNTIME_ROOT="$(cd "$GRAPH_RUNTIME_ROOT/.." && pwd)"
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
unset PYTHONHOME PYTHONPATH
