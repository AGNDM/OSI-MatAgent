#!/usr/bin/env bash
#SBATCH --job-name=instrument-study
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
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen study snapshot}"
source comparison/slurm/common.sh
comparison_inference_environment
if [[ "${COMPARISON_RUNTIME:?}" == ollama ]]; then
  source comparison/slurm/ollama-common.sh
fi
srun "${COMPARISON_PYTHON:?}" -m comparison.study worker --study "${COMPARISON_STUDY:?}" --group "$COMPARISON_RUNTIME"
