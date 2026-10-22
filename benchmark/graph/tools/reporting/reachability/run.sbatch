#!/usr/bin/env bash
#SBATCH --job-name=graph-reachability-compat
#SBATCH --account=liac
#SBATCH --partition=mig12gb,mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=graph-reachability-%j.log
set -euo pipefail
: "${GRAPH_PACKAGE:?Set the absolute benchmark/graph or frozen snapshot/package path}"
if [[ "${1:-}" == --validate ]]; then
  shift
  exec bash "$GRAPH_PACKAGE/slurm/check-reporting.sbatch" "$@"
fi
[[ $# -eq 2 ]] || { echo 'Usage: run.sbatch MATRIX_JSON FRESH_OUTPUT, or --validate' >&2; exit 2; }
exec bash "$GRAPH_PACKAGE/slurm/reachability.sbatch" --matrix "$1" --output "$2"
