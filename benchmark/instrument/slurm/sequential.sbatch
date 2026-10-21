#!/usr/bin/env bash
#SBATCH --job-name=instrument-sequential
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=output/sequential-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark/instrument}"
[[ -n "${SLURMD_NODENAME:-}" && -n "${SLURM_JOB_ID:-}" ]] || exit 2
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$BENCHMARK_PYTHON" != /* || ! -x "$BENCHMARK_PYTHON" ]]; then
  printf '%s\n' 'No benchmark interpreter: submit slurm/setup.sbatch first, or set an absolute BENCHMARK_PYTHON.' >&2
  exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
unset PYTHONPATH PYTHONHOME
srun "$BENCHMARK_PYTHON" eval/sequential.py run \
  --output "${BENCHMARK_OUTPUT:-output/sequential-$SLURM_JOB_ID}" "$@"
