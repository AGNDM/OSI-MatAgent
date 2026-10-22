#!/bin/bash
#SBATCH --job-name=graph-local-prepare
#SBATCH --account=liac
#SBATCH --partition=mig12gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=graph-prepare-%j.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_PYTHON:?Set the validated local graph interpreter}"
export PYTHONPATH="$GRAPH_PACKAGE"
srun "$GRAPH_PYTHON" -m graph_local.prepare "$@"
