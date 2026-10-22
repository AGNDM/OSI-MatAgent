#!/usr/bin/env bash
#SBATCH --job-name=graph-reporting-check
#SBATCH --account=liac
#SBATCH --partition=mig12gb,mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=graph-reporting-check-%j.log
set -euo pipefail
source "${GRAPH_PACKAGE:?}/slurm/common.sh"
: "${GRAPH_PYTHON:?Set the validated graph/reporting interpreter}"
export PYTHONPATH="$GRAPH_RUNTIME_ROOT" MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
srun --cpu-bind=none "$GRAPH_PYTHON" -m pytest -q --tb=short -p no:cacheprovider \
  "$GRAPH_RUNTIME_ROOT/tests/test_matrix_report.py" \
  "$GRAPH_RUNTIME_ROOT/tests/test_reachability.py" "$@"
