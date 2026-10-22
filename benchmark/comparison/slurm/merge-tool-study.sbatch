#!/usr/bin/env bash
#SBATCH --job-name=instrument-tools-merge
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen tool-study snapshot}"
source comparison/slurm/common.sh
srun "${COMPARISON_PYTHON:?}" -m comparison.tool_track merge --study "${COMPARISON_TOOL_STUDY:?}" --allow-partial "$@"
