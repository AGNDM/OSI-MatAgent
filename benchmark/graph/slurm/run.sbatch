#!/bin/bash
#SBATCH --job-name=graph-local-array
#SBATCH --account=liac
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=graph-%A_%a.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_PYTHON:?Set the validated local graph interpreter}"
: "${GRAPH_STUDY:?Set the absolute prepared study directory}"
srun "$GRAPH_PYTHON" "$GRAPH_STUDY/snapshot/package/tools/worker.py" --study "$GRAPH_STUDY" "$@"
