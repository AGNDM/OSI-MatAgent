# Frozen sequential runtime

The `vendor/` tree contains unmodified copies of the experiment evaluator,
shared action menus, context projection, native simulation code, and selected
ChemLab assets for all eight supported instruments. `provenance.json` records
each original relative path, byte count and SHA-256. The furnace uses the
included basic40 core and thermal runtime; the other seven instruments use the
included device runtimes and JSON data directories.

Copy the entire `benchmark/instrument/` directory to another location. From its
root, submit `sbatch slurm/setup.sbatch` to create `.venv` with Python 3.12 and
the complete pinned native dependency lock. Submit `sbatch slurm/check.sbatch`
to verify the copy, then use `slurm/sequential.sbatch` for native evaluation.
An absolute `BENCHMARK_PYTHON` can select a compatible existing environment.
No environment, model
weights, 3D meshes or links to the original checkout are included. The native
simulator and model policies retain the original requirement for a real Slurm
compute allocation. Static inspection and scoring of existing output files do
not start native simulations.

Custom adapters import the bundled modules as follows:

```python
from sequential_120.runtime.bootstrap import activate
activate()
from stepwise_choice.engine import run_case
from stepwise_choice.menus import MenuBuilder

# The distribution CLI loads the release's fixed case specification and bank.
# run_case(case, policy, mode="teacher_forced", menu_builder=MenuBuilder(bank))
```

The policy implements `next_action(public_view, history)` and returns one action
offered by `MenuBuilder`. The evaluator retains the private fixture and checkpoint
specification. Provide the fixed exported bank explicitly; rebuilding a bank or
generating cases is unnecessary for evaluating the release.

`requirements-model.txt` locks the optional local Hugging Face finite-label
policy dependencies for Python 3.12, Linux x86_64 and CUDA 12.6. Submit
`sbatch slurm/setup.sbatch --model` to create the separate `.venv-models`.
Supply pinned model files and a compatible GPU driver separately. The runtime also
includes the lossless public history projection and its integrity checks.

The source modules retain their original internal names for byte identity. Use
the distribution's `eval/sequential.py` interface. The original experiment
module CLIs use the experiment directory layout and are not release entrypoints.

ChemLab code is MIT licensed. Its JSON data and metadata are covered by
CC BY-NC 4.0 and the third-party terms in the included upstream `NOTICE.md` and
`LICENSES/`; original attribution and license files are preserved in
`vendor/external/ChemLab-Assets/`.
