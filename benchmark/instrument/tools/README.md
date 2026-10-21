# Setup and release maintenance

Run these tools inside genuine Slurm compute allocations on SCITAS. The
submission commands below run from the distribution root. Source files must
stay unchanged while an evaluation or validation job is using them.

## Environment

```bash
mkdir -p output
sbatch slurm/setup.sbatch
```

This creates `.venv` with Python 3.12 and all six pinned native dependencies.
`sbatch slurm/setup.sbatch --model` creates `.venv-models` with the optional
Linux x86_64 / CUDA 12.6 dependency lock. Neither mode downloads model weights.
Installation needs a reachable package index or configured mirror; model setup
also uses the PyTorch CUDA wheel index. The model environment is much larger.

Set `BENCHMARK_BOOTSTRAP_PYTHON` to an absolute Python 3.12 interpreter if
`python3.12` is unavailable. Pass `--environment /absolute/fresh/path` for an
alternative environment location. Existing environments are verified and
reused only when their pinned packages match; they are never upgraded or
replaced. A partial or mismatched environment requires a fresh path.

`BENCHMARK_PYTHON=/absolute/environment/bin/python` overrides the interpreter
used by `slurm/check.sbatch` and `slurm/sequential.sbatch`.

## Verify a copy

`sbatch slurm/check.sbatch` checks published checksums, dataset consistency and
the unit tests. For an isolated-copy native smoke check, use this compute-job
payload:

```bash
.venv/bin/python tools/check_copy.py --output output/isolated-check
```

The output directory must be new. This checks imports outside the original
checkout, runs the tests in a temporary copy, and executes one case with
reference, deliberately wrong and demonstration policies in both modes. These
six episodes test portability and correction behavior, not model performance
or full-dataset coverage. Environments, caches and outputs are not copied.

## Refresh a maintained release

Colleagues running the unchanged release do not need to refresh anything.
For reviewed documentation or packaging-tool changes, these are compute-job
payloads:

```bash
.venv/bin/python tools/release.py refresh --version 1.1.2 --output output/release-refresh.json
.venv/bin/python tools/release.py check --output output/release-check.json
```

Refresh uses the previous root manifest to protect frozen questions, answer
keys, splits, scenario specifications, core graders and vendored runtime. It
updates the parent and sequential inventories, then validates the result;
failed validation rolls back both manifests. Dependency locks and documentation
are packaging metadata. Changed benchmark data or grader behavior requires a
separate reviewed release, not an inventory refresh.

`manifest.json` is the release version authority. `validation.json` describes
the distributed release and is excluded from its own checksums. Refresh does
not rewrite that receipt: before redistributing a changed release, replace it
with current verification evidence and the new manifest hash. Keep run-specific
receipts under `output/`, and exclude generated environments, caches and
outputs when sharing a copy. Keep all licences and provenance files.
