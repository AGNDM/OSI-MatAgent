#!/usr/bin/env bash
#SBATCH --job-name=instrument-setup
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=output/setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark/instrument}"
[[ -n "${SLURMD_NODENAME:-}" && -n "${SLURM_JOB_ID:-}" ]] || exit 2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
srun "${BENCHMARK_BOOTSTRAP_PYTHON:-python3.12}" tools/setup.py "$@"
