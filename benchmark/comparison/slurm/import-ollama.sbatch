#!/usr/bin/env bash
#SBATCH --job-name=instrument-import-local-hf
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=output/comparison-import-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository benchmark directory}"
source comparison/slurm/ollama-common.sh
srun "$comparison_python" -m comparison.setup --import-ollama "$@"
