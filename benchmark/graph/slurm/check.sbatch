#!/bin/bash
#SBATCH --job-name=graph-local-check
#SBATCH --account=liac
#SBATCH --partition=mig12gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=graph-check-%j.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_PYTHON:?Set the validated local graph interpreter}"
srun "$GRAPH_PYTHON" "$GRAPH_PACKAGE/tools/check.py" "$@"
