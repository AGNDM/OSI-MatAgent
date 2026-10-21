#!/usr/bin/env python3
"""Verify this distribution from an isolated copy, including native smoke cases.

Run inside a Slurm CPU allocation with the native dependencies installed. This
checks portability, not model performance or full-dataset scientific validity.
The original benchmark and any ongoing experiment outputs are left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


DEFAULT_BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFAULT_BUNDLE))
from eval.validate import _ignored
from tools.release import check_bundle


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _run(command, cwd, environment, log):
    result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Isolated check failed ({result.returncode}); inspect {log}")
    return result.stdout


def check_copy(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    _require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
             "--output must be a fresh empty directory")
    if output.is_relative_to(bundle):
        _require(output.is_relative_to(bundle / "output"),
                 "Check output inside the bundle must be under output/")
    output.mkdir(parents=True, exist_ok=True)
    source = check_bundle(bundle)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="instrument-release-copy-") as temporary:
        root = Path(temporary)
        copied = root / "instrument"

        def ignore(directory, names):
            relative_dir = Path(directory).relative_to(bundle)
            return [name for name in names
                    if (relative_dir / name).as_posix() not in {"manifest.json", "validation.json"}
                    and _ignored(relative_dir / name)]

        shutil.copytree(bundle, copied, ignore=ignore)
        _run([sys.executable, "tools/release.py", "check"], copied, environment,
             output / "static-check.log")
        tests_log = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                         copied, environment, output / "unit-tests.log")
        matches = re.findall(r"Ran (\d+) tests? in", tests_log)
        _require(bool(matches), "Unit test log does not contain a test count")
        # Start outside both the original checkout and the copied bundle. Every
        # benchmark/native module must still resolve to the copied distribution.
        probe = root / "imports.py"
        probe.write_text(
            "import pathlib,sys\n"
            "bundle=pathlib.Path(sys.argv[1]).resolve()\n"
            "sys.path.insert(0,str(bundle))\n"
            "from sequential_120.runtime.bootstrap import activate,PACKAGES\n"
            "activate()\n"
            "import eval.sequential,stepwise_choice.engine,stepwise_expanded.engine,instrument_bench.adapters\n"
            "for name,module in list(sys.modules.items()):\n"
            " if name.split('.')[0] in {*PACKAGES,'eval','sequential_120'}:\n"
            "  if getattr(module,'__file__',None): pathlib.Path(module.__file__).resolve().relative_to(bundle)\n"
            "print('Native and evaluator imports resolve inside the isolated bundle')\n",
            encoding="utf-8")
        _run([sys.executable, str(probe), str(copied)], root, environment, output / "imports.log")
        summaries = {}
        for label, policy in (("reference", "reference"), ("wrong-menu", "wrong-menu"),
                              ("public-plugin", "examples.sequential_policy:make_policy")):
            _run([sys.executable, "eval/sequential.py", "run", "--policy", policy,
                  "--mode", "both", "--limit", "1", "--output", f"output/{label}"],
                 copied, environment, output / f"{label}.log")
            summary = json.loads((copied / f"output/{label}/summary.json").read_text(encoding="utf-8"))
            _require(summary["completed_episodes"] == 2 and summary["diagnostic_subset"],
                     f"Incomplete isolated {label} smoke cases")
            _require(not summary["complete_full_dataset"], "A smoke subset was mislabeled as full evaluation")
            summaries[label] = summary
            shutil.copyfile(copied / f"output/{label}/summary.json", output / f"{label}.json")
        for mode in ("teacher_forced", "free_run"):
            scored = summaries["reference"]["by_mode"][mode]
            _require(scored["goals_reached"] == 1 and scored["interventions"] == 0
                     and scored["observed_control_accuracy_micro"] == 1,
                     f"Native reference failed in {mode}")
        negative = summaries["wrong-menu"]["by_mode"]
        _require(negative["teacher_forced"]["goals_reached"] == 1
                 and negative["teacher_forced"]["observed_control_accuracy_micro"] == 0
                 and negative["free_run"]["goals_reached"] == 0,
                 "Wrong-choice and teacher-forcing smoke checks failed")
        receipt = {"status": "passed", "isolated_copy": True,
                   "manifest_sha256": source["manifest_sha256"],
                   "original_project_imports_used": False, "native_replay_performed": True,
                   "native_reference_cases": 1, "native_reference_episodes": 2,
                   "native_wrong_menu_episodes": 2, "public_plugin_episodes": 2,
                   "unit_tests_passed": int(matches[-1]), "full_dataset_replayed": False,
                   "model_weights_loaded": False, "neural_inference_performed": False,
                   "host": os.environ["SLURMD_NODENAME"], "slurm_job_id": os.environ["SLURM_JOB_ID"]}
        (output / "isolated-copy.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _require(bool(os.environ.get("SLURMD_NODENAME") and os.environ.get("SLURM_JOB_ID")),
                 "Run inside a real Slurm compute allocation")
        print(json.dumps(check_copy(args.bundle, args.output), sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
