#!/usr/bin/env bash
#SBATCH --job-name=instrument-readonly-tools
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen tool-study snapshot}"
source comparison/slurm/common.sh
comparison_inference_environment
srun "${COMPARISON_PYTHON:?}" -m comparison.tool_track worker --study "${COMPARISON_TOOL_STUDY:?}"
