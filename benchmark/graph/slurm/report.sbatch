#!/bin/bash
#SBATCH --job-name=graph-local-report
#SBATCH --account=liac
#SBATCH --partition=l40s,h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=graph-report-%j.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_PYTHON:?Set the validated local graph interpreter}"
srun "$GRAPH_PYTHON" "$GRAPH_PACKAGE/tools/report.py" "$@"
