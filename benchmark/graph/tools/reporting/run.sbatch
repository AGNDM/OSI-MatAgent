#!/usr/bin/env bash
#SBATCH --job-name=graph-matrix-compat
#SBATCH --account=liac
#SBATCH --partition=l40s,h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=graph-matrix-%j.log
set -euo pipefail
: "${GRAPH_PACKAGE:?Set the absolute benchmark/graph or frozen snapshot/package path}"
if [[ "${1:-}" == --validate ]]; then
  shift
  exec bash "$GRAPH_PACKAGE/slurm/check-reporting.sbatch" "$@"
fi
[[ $# -ge 3 ]] || { echo 'Usage: run.sbatch STUDY COMBINED_REPORT FRESH_OUTPUT [ADDITIONAL_STUDY ...], or --validate' >&2; exit 2; }
study="$1" report="$2" output="$3"
shift 3
arguments=(--study "$study" --combined-report "$report" --output "$output")
for extra in "$@"; do arguments+=(--study "$extra"); done
exec bash "$GRAPH_PACKAGE/slurm/report.sbatch" "${arguments[@]}"
