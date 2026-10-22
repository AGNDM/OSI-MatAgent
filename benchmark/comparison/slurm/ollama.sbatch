#!/usr/bin/env bash
#SBATCH --job-name=instrument-ollama
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=output/comparison-ollama-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository benchmark directory}"
source comparison/slurm/ollama-common.sh
srun "$comparison_python" -m comparison.run \
  --output "${COMPARISON_OUTPUT:-output/comparison-ollama-$SLURM_JOB_ID}" "$@"
