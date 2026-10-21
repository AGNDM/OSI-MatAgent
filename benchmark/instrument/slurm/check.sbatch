#!/usr/bin/env bash
#SBATCH --job-name=instrument-bundle-check
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=output/bundle-check-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark/instrument}"
[[ -n "${SLURMD_NODENAME:-}" && -n "${SLURM_JOB_ID:-}" ]] || exit 2
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-$PWD/.venv/bin/python}"
if [[ "$BENCHMARK_PYTHON" != /* || ! -x "$BENCHMARK_PYTHON" ]]; then
  printf '%s\n' 'No benchmark interpreter: submit slurm/setup.sbatch first, or set an absolute BENCHMARK_PYTHON.' >&2
  exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
srun "$BENCHMARK_PYTHON" eval/validate.py
srun "$BENCHMARK_PYTHON" -m unittest discover -s tests -v
