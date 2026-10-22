#!/usr/bin/env bash
#SBATCH --job-name=instrument-study-merge
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen study snapshot}"
source comparison/slurm/common.sh
srun "${COMPARISON_MERGE_PYTHON:-python3.12}" -m comparison.study merge --study "${COMPARISON_STUDY:?}" --allow-partial "$@"
