#!/usr/bin/env bash
#SBATCH --job-name=graph-reachability
#SBATCH --account=liac
#SBATCH --partition=mig12gb,mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=graph-reachability-%j.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_PYTHON:?Set the validated graph/reporting interpreter}"
export PYTHONPATH="$GRAPH_RUNTIME_ROOT"
srun --cpu-bind=none "$GRAPH_PYTHON" -m graph_local.reporting.reachability "$@"
