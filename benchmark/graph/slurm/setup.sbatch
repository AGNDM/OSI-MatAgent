#!/bin/bash
#SBATCH --job-name=graph-local-setup
#SBATCH --account=liac
#SBATCH --partition=mig12gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=graph-setup-%j.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_BASE_PYTHON:?Set the existing frozen inference interpreter}"
export UV_CACHE_DIR="${GRAPH_UV_CACHE:-${TMPDIR:-/tmp}/graph-uv-${SLURM_JOB_ID}}"
srun "$GRAPH_BASE_PYTHON" "$GRAPH_PACKAGE/tools/setup.py" --base-python "$GRAPH_BASE_PYTHON" "$@"
