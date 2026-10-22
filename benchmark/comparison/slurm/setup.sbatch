#!/usr/bin/env bash
#SBATCH --job-name=instrument-comparison-setup
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=output/comparison-setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository benchmark directory}"
source comparison/slurm/common.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="${COMPARISON_PIP_CACHE:-$PWD/.cache/comparison-pip}"
srun "${COMPARISON_BOOTSTRAP_PYTHON:-python3.12}" -m comparison.setup "$@"
