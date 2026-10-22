#!/usr/bin/env bash
#SBATCH --job-name=instrument-local-models
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=output/comparison-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository benchmark directory}"
source comparison/slurm/common.sh
comparison_inference_environment
comparison_python="${COMPARISON_PYTHON:-$PWD/.venv-comparison-hf/bin/python}"
[[ "$comparison_python" == /* && -x "$comparison_python" ]] || {
  printf '%s\n' 'Run comparison/slurm/setup.sbatch --profile hf first, or supply an absolute COMPARISON_PYTHON.' >&2
  exit 2
}
srun "$comparison_python" -m comparison.run \
  --output "${COMPARISON_OUTPUT:-output/comparison-$SLURM_JOB_ID}" "$@"
