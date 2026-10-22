#!/usr/bin/env bash
#SBATCH --job-name=evaluation-postprocess
#SBATCH --account=liac
#SBATCH --partition=mig12gb,mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=evaluation-%j.log
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-}" ]] || exit 2
evaluation_root="${EVALUATION_ROOT:?Set the absolute benchmark/evaluation directory}"
evaluation_python="${EVALUATION_PYTHON:?Set an existing absolute Python interpreter}"
[[ "$evaluation_root" = /* && "$evaluation_python" = /* && -x "$evaluation_python" ]] || exit 2
for evaluation_binding in ${!SLURM_CPU_BIND@}; do unset "$evaluation_binding"; done
export SLURM_CPU_BIND=none PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONHOME PYTHONPATH
evaluation_command="${1:?Choose a documented postprocessing command}"
shift
case "$evaluation_command" in
  workflow-report) evaluation_script=workflow/report_grid.py ;;
  workflow-status) evaluation_script=workflow/compact_status.py ;;
  workflow-failures) evaluation_script=workflow/audit_failures.py ;;
  workflow-audit) evaluation_script=workflow/final_audit.py ;;
  workflow-render-plan) evaluation_script=workflow/render_plan.py ;;
  mcqa-overall) evaluation_script=mcqa/export_overall.py ;;
  mcqa-groups) evaluation_script=mcqa/export_groups.py ;;
  mcqa-enrich) evaluation_script=mcqa/enrich.py ;;
  prepare-profiles) evaluation_script=prepare_profiles.py ;;
  *) printf 'Unknown evaluation command: %s\n' "$evaluation_command" >&2; exit 2 ;;
esac
srun --cpu-bind=none --exact --ntasks=1 --cpus-per-task=2 \
  "$evaluation_python" "$evaluation_root/$evaluation_script" "$@"
