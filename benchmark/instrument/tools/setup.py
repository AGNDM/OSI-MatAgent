"""Prepare the locked benchmark environment inside a genuine Slurm allocation.

Default: native-only .venv. --model: optional CUDA 12.6 model dependencies in
.venv-models. This never downloads model weights or replaces an existing venv.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv


BUNDLE = Path(__file__).resolve().parents[1]
REQUIREMENTS = BUNDLE / "sequential_120" / "runtime"


def _run(arguments):
    subprocess.run([str(argument) for argument in arguments], check=True)


def _pins(path):
    """Read the complete transitive lock, including nested native requirements."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            pins.update(_pins(path.parent / line[3:].strip()))
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or " " in line:
            raise ValueError(f"Expected an exact package pin in {path.name}: {line}")
        pins[name.lower().replace("_", "-")] = version
    return pins


def _verify(interpreter, pins):
    code = """import importlib.metadata, json, sys
expected = json.loads(sys.argv[1])
actual = {name: importlib.metadata.version(name) for name in expected}
wrong = {name: [expected[name], actual[name]] for name in expected
         if actual[name] != expected[name]}
if wrong:
    raise SystemExit('Environment differs from lock: ' + json.dumps(wrong))
print(json.dumps({'python': sys.version, 'packages': actual}, sort_keys=True))
"""
    result = subprocess.run([str(interpreter), "-c", code, json.dumps(pins)],
                            check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true",
                        help="Install optional CUDA model dependencies; no model weights")
    parser.add_argument("--environment", type=Path,
                        help="Use a new environment path instead of .venv or .venv-models")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        parser.error("Submit slurm/setup.sbatch: installation requires a Slurm compute allocation")
    if sys.version_info[:2] != (3, 12):
        parser.error("The dependency lock requires Python 3.12; set BENCHMARK_BOOTSTRAP_PYTHON")
    if args.model and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        parser.error("The model lock targets Linux x86_64 and CUDA 12.6")
    requirement = REQUIREMENTS / ("requirements-model.txt" if args.model else "requirements.txt")
    pins = _pins(requirement)
    environment = (args.environment or (BUNDLE / (".venv-models" if args.model else ".venv"))).absolute()
    interpreter = environment / "bin" / "python"
    if environment.exists():
        if not interpreter.is_file():
            parser.error("Environment path already exists without an interpreter; choose a fresh --environment")
        # Reuse only an exact match. Never mutate an environment used by a run.
        details = _verify(interpreter, pins)
        _run([interpreter, "-m", "pip", "check"])
        print(json.dumps({"status": "reused", "environment": str(environment), **details}, sort_keys=True))
        return 0
    venv.EnvBuilder(with_pip=True).create(environment)
    command = [interpreter, "-m", "pip", "--disable-pip-version-check", "install",
               "--only-binary=:all:", "--no-deps", "-r", requirement]
    if args.model:
        command.extend(["--index-url", "https://pypi.org/simple", "--extra-index-url",
                        "https://download.pytorch.org/whl/cu126"])
    _run(command)
    _run([interpreter, "-m", "pip", "check"])
    details = _verify(interpreter, pins)
    receipt = {
        "status": "created", "environment": str(environment),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_node": os.environ["SLURMD_NODENAME"],
        "platform": platform.platform(), "model_weights_downloaded": False,
        "lock_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in {requirement, REQUIREMENTS / "requirements.txt"}},
        **details,
    }
    output = BUNDLE / "output" / f"setup-{os.environ['SLURM_JOB_ID']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
