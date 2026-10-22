"""Validate the evaluation source tree without inference or network calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Run validation in a real Slurm compute allocation")


def app_checks(root, output):
    """Use the application's full pinned runtime, separate from the eight-device bundle."""
    if not (root / "vendor/chemlab/scripts/basic40_runtime_measure.py").is_file():
        raise RuntimeError("Restore the application's pinned native runtime in Slurm with "
                           "tools/fetch_device_runtime.py --download before full app checks")
    verified = subprocess.run([sys.executable, str(root / "tools/fetch_device_runtime.py")],
                              cwd=root, check=False)
    if verified.returncode:
        return verified.returncode
    sys.path.insert(0, str(root))
    import pytest

    return pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider",
                       str(root / "tests"), "--junitxml", str(output / "app.xml")])


def source_files(root):
    """Only inventory distributable code; never traverse environments or runs."""
    files = {root / "README.md", root / ".gitignore", root / "benchmark/README.md",
             root / "benchmark/.gitignore"}
    for relative in ("app", "tests", "benchmark/comparison", "benchmark/graph",
                     "benchmark/evaluation", "benchmark/docs", "benchmark/tools",
                     "benchmark/slurm"):
        for directory, names, filenames in os.walk(root / relative):
            names[:] = sorted(name for name in names
                              if not name.startswith(".") and name not in {"__pycache__", "output"})
            files.update(Path(directory) / name for name in filenames
                         if not name.endswith((".pyc", ".pyo")))
    return sorted(path for path in files if path.is_file())


def main():
    require_allocation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh directory outside benchmark source, normally under results/")
    parser.add_argument("--app-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root / "benchmark"):
        raise ValueError("Write validation output outside source packages")
    if args.app_only:
        return app_checks(root, output)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    files = source_files(root)
    before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    python = sys.executable
    suites = [
        ("instrument-integrity", [python, "eval/validate.py"], root / "benchmark/instrument"),
        ("instrument-tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
         root / "benchmark/instrument"),
        ("comparison", [python, "-m", "unittest", "discover", "-s", "comparison/tests", "-v"],
         root / "benchmark"),
        ("graph", [python, "graph/tools/check.py", "--app-source", str(root),
                   "--comparison-source", str(root / "benchmark"),
                   "--junitxml", str(output / "graph.xml")], root / "benchmark"),
        ("evaluation", [python, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
                        "evaluation/tests", "--junitxml", str(output / "evaluation.xml")],
         root / "benchmark"),
        ("application", [python, str(Path(__file__).resolve()), "--root", str(root),
                         "--output", str(output), "--app-only"], root),
    ]
    checks = []
    for name, command, cwd in suites:
        print(f"Checking {name}; log: {output / (name + '.log')}", flush=True)
        with (output / (name + ".log")).open("w") as log:
            completed = subprocess.run(command, cwd=cwd, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        checks.append({"name": name, "command": command, "returncode": completed.returncode})
        print(f"{name}: {'passed' if completed.returncode == 0 else 'FAILED'}", flush=True)
    for path in files:
        if path.suffix in {".sh", ".sbatch"}:
            completed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            checks.append({"name": "shell:" + str(path.relative_to(root)),
                           "returncode": completed.returncode, "stderr": completed.stderr})
        elif path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")
    unchanged = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source_files(root)} == before
    checks.append({"name": "source-unchanged-during-checks", "returncode": 0 if unchanged else 1})
    frozen = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "benchmark/instrument",
                             "results/388", "results/120"], cwd=root, check=False)
    checks.append({"name": "frozen-release-and-published-results-unchanged", "returncode": frozen.returncode})
    passed = all(check["returncode"] == 0 for check in checks)
    receipt = {"status": "passed" if passed else "failed", "started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(),
               "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "compute_node": os.environ["SLURMD_NODENAME"], "python": python,
               "inference_performed": False, "hosted_inference_calls": 0,
               "source_sha256": before, "checks": checks}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
