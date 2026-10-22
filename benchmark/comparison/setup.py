"""Create exact local comparison environments, or import local HF weights.

Every installation, checksum and Ollama operation requires a Slurm compute node.
This tool never downloads model weights and never changes an existing environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import venv


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
PROFILES = {
    "native": ("requirements.txt", ".venv-comparison"),
    "hf": ("requirements-hf.txt", ".venv-comparison-hf"),
    "features": ("requirements-features.txt", ".venv-comparison-features"),
    "apus": ("requirements-apus.txt", ".venv-comparison-apus"),
}


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Run this command inside a real Slurm compute allocation")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_pins(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return {}
    seen.add(path)
    pins = {}
    for original in path.read_text(encoding="utf-8").splitlines():
        line = original.partition("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            entries = read_pins(path.parent / line[3:].strip(), seen)
        else:
            name, separator, version = line.partition("==")
            if not separator or not name or not version or " " in line:
                raise ValueError(f"Expected an exact package pin in {path}: {line}")
            entries = {name.lower().replace("_", "-"): version}
        for name, version in entries.items():
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting pin for {name}")
            pins[name] = version
    return pins


def verify_environment(interpreter, pins):
    program = """import importlib.metadata, json, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('The target environment must use Python 3.12')
expected = json.loads(sys.argv[1])
actual = {name: importlib.metadata.version(name) for name in expected}
wrong = {name: [expected[name], actual[name]] for name in expected if actual[name] != expected[name]}
if wrong:
    raise SystemExit('Environment differs from lock: ' + json.dumps(wrong))
print(json.dumps({'python': sys.version, 'packages': actual}, sort_keys=True))
"""
    result = subprocess.run([str(interpreter), "-c", program, json.dumps(pins)],
                            text=True, capture_output=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError("Environment verification failed: " + detail[-3000:])
    # uv-created environments can intentionally omit pip. Verifying these must
    # not install or mutate them just to reuse their exact existing packages.
    probe = subprocess.run([str(interpreter), "-c",
                            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pip') else 1)"])
    if probe.returncode == 0:
        subprocess.run([str(interpreter), "-m", "pip", "check"], check=True)
    return json.loads(result.stdout)


def receipt_path(prefix):
    output = ROOT / "output" / f"{prefix}-{os.environ['SLURM_JOB_ID']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_receipt(path, content):
    content = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_node": os.environ["SLURMD_NODENAME"],
        **content,
    }
    Path(path).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": content.get("status", "validated")}))


def prepare_environment(args):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The dependency locks require Python 3.12")
    if args.profile != "native" and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise ValueError("The model dependency locks target Linux x86_64 / CUDA 12.6")
    requirement_name, default_environment = PROFILES[args.profile]
    requirement = SOURCE / requirement_name
    lock_files = set()
    pins = read_pins(requirement, lock_files)
    environment = (args.environment or ROOT / default_environment).absolute()
    interpreter = environment / "bin" / "python"
    if environment.exists():
        if not interpreter.is_file():
            raise ValueError("Environment path exists without an interpreter; use a fresh --environment")
        details = verify_environment(interpreter, pins)
        status = "reused"
    else:
        venv.EnvBuilder(with_pip=True).create(environment)
        command = [str(interpreter), "-m", "pip", "--disable-pip-version-check", "install",
                   "--only-binary=:all:", "--no-deps", "-r", str(requirement)]
        if args.profile != "native":
            command.extend(["--index-url", "https://pypi.org/simple", "--extra-index-url",
                            "https://download.pytorch.org/whl/cu126"])
        subprocess.run(command, check=True)
        details = verify_environment(interpreter, pins)
        status = "created"
    write_receipt(receipt_path("comparison-setup"), {
        "status": status, "profile": args.profile, "environment": str(environment),
        "lock_files_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sorted(lock_files)},
        "model_weights_downloaded": False, "verification": "all_exact_lock_pins", **details,
    })


def ollama_request(path, payload=None):
    base = os.environ.get("COMPARISON_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    if not (base.startswith("http://127.0.0.1:") or base.startswith("http://localhost:")):
        raise ValueError("The comparison setup requires an allocated-node loopback Ollama server")
    request = urllib.request.Request(base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.load(response)


def verify_ollama():
    lock_path = SOURCE / "ollama-runtime.lock.json"
    lock = json.loads(lock_path.read_text())
    actual = ollama_request("/api/version")
    if actual.get("version") != lock["version"]:
        raise ValueError(f"Ollama runtime differs from lock: {actual!r}")
    executable = Path(os.environ["COMPARISON_OLLAMA_BIN"]).resolve()
    return {"runtime_version": actual["version"], "executable": str(executable),
            "executable_sha256": digest(executable), "runtime_lock_sha256": digest(lock_path)}


def import_ollama(args):
    runtime = verify_ollama()
    if args.model_path is None or not args.ollama_model:
        raise ValueError("Import requires --model-path and --ollama-model")
    model_path = args.model_path.resolve(strict=True)
    if not model_path.is_dir() or any(character in str(model_path) for character in '\n\r"'):
        raise ValueError("Use a local HF model directory without quotes or newlines")
    configuration = json.loads((model_path / "config.json").read_text())
    architectures = configuration.get("architectures", [])
    allowed = {"Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", "GemmaForCausalLM"}
    if not architectures or any(name not in allowed for name in architectures):
        raise ValueError("Unsupported import architecture: use HF/native backends for decision heads or other models")
    tags = ollama_request("/api/tags").get("models", [])
    if any(args.ollama_model in {item.get("name"), item.get("model")} for item in tags):
        raise ValueError("That Ollama tag already exists; choose a new tag to preserve experiment provenance")
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise ValueError("No local safetensors weights found")
    weight_provenance = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in weights}
    work = ROOT / "output" / f"comparison-import-{os.environ['SLURM_JOB_ID']}"
    work.mkdir(parents=True, exist_ok=False)
    modelfile = work / "Modelfile"
    modelfile.write_text(f'FROM "{model_path}"\n', encoding="utf-8")
    subprocess.run([os.environ["COMPARISON_OLLAMA_BIN"], "create", args.ollama_model,
                    "--file", str(modelfile)], check=True)
    tags = ollama_request("/api/tags").get("models", [])
    imported = next((item for item in tags
                     if args.ollama_model in {item.get("name"), item.get("model")}), None)
    if imported is None:
        raise RuntimeError("Import completed without the requested local tag")
    write_receipt(work / "receipt.json", {
        "status": "imported", "source_model_path": str(model_path),
        "source_architectures": architectures, "source_config_sha256": digest(model_path / "config.json"),
        "source_weights": weight_provenance, "model_weights_downloaded": False,
        "explicit_quantization_requested": False, "ollama_model": imported,
        "model_store": os.environ.get("OLLAMA_MODELS"), **runtime,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="native")
    parser.add_argument("--environment", type=Path, help="New environment destination; existing environments are verified")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--verify-ollama", action="store_true")
    operation.add_argument("--import-ollama", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--ollama-model")
    args = parser.parse_args(argv)
    require_slurm()
    if args.verify_ollama:
        print(json.dumps(verify_ollama(), sort_keys=True))
    elif args.import_ollama:
        import_ollama(args)
    else:
        prepare_environment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
