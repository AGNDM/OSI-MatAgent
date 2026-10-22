"""Freeze, partition, run and merge a local-model comparison study.

Preparation produces explicit worker commands for an existing compute allocation;
it never submits jobs. A shard always retains the complete cohort denominator.
Merging authenticates the frozen source and rejects overlapping or foreign rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .run import (experiment_mcqa_cohort, experiment_mcqa_dataset, experiment_tracks,
                  fingerprint, load_bundle, require_allocation, resolve_configuration,
                  select_mcqa, select_tasks, source_inventory, validate_config, write_json)
from .aggregate import mcqa_calibration, sequential_calibration


PROTOCOL = "frozen-local-comparison-study-v1"


def _ignored(directory, names):
    return [name for name in names if name in {"output", "__pycache__", ".git", ".cache", ".pytest_cache"}
            or name.startswith(".venv") or name.endswith((".pyc", ".pyo"))]


def runtime_group(experiment):
    backend = experiment["backend"]
    if backend.get("kind") == "ollama":
        return "ollama"
    if backend.get("kind") in {"native", "decision"} and backend.get("family") == "apus":
        return "apus"
    return "hf"


def _load_rows(path):
    if not path.is_file():
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as error:
            raise ValueError(f"Malformed JSONL at {path}:{index}; interrupted final rows require an explicit recovery copy") from error
    return rows


def _write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def prepare(args):
    require_allocation()
    source_config = validate_config(json.loads(args.config.read_text()))
    config = resolve_configuration(source_config, args.config,
                                   {experiment["name"] for experiment in source_config["experiments"]})
    bundle = ((args.config.resolve().parent / config["bundle"]).resolve()
              if "bundle" in config else Path(__file__).resolve().parent.parent / "instrument")
    helpers = load_bundle(bundle)
    validation = helpers["validate"].validate_bundle(bundle)
    if args.shards < 1 or args.max_parallel < 1:
        raise ValueError("shards and max-parallel must be positive")
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("Study output must be a fresh directory")
    if destination.is_relative_to(bundle.resolve()) or destination.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("A study cannot be created inside frozen comparison/instrument source")
    tasks, _ = helpers["common"].load_dataset(bundle / "mcqa_388/questions.jsonl.gz", bundle / "mcqa_388/answers.jsonl")
    public_cases = helpers["common"].load_jsonl(bundle / "sequential_120/cases.jsonl")
    cohorts = {"mcqa": select_mcqa(tasks, config.get("tracks", {}).get("mcqa", {}).get("split", "heldout")),
               "sequential": public_cases}
    tracks = ("mcqa", "sequential") if args.track == "both" else (args.track,)
    # Preserve the venv executable path: resolving its symlink would select the
    # base interpreter and silently lose the environment's installed packages.
    python_paths = {"hf": str(args.hf_python.absolute()),
                    "ollama": str(args.hf_python.absolute()),
                    "apus": str(args.apus_python.absolute()) if args.apus_python else None}
    groups, experiment_cohorts = {}, {}
    for experiment in config["experiments"]:
        current_tracks = experiment_tracks(experiment, tracks)
        if not current_tracks:
            continue
        current_cohorts = {"mcqa": experiment_mcqa_cohort(tasks,
                            config.get("tracks", {}).get("mcqa", {}).get("split", "heldout"), experiment),
                           "sequential": public_cases}
        if any(len(current_cohorts[track]) < args.shards for track in current_tracks):
            raise ValueError("Each requested experiment shard must contain at least one task")
        experiment_cohorts[experiment["name"]] = {
            track: [task["task_id"] for task in current_cohorts[track]] for track in current_tracks}
        group = runtime_group(experiment)
        interpreter = python_paths[group]
        if not interpreter or not Path(interpreter).is_file():
            raise ValueError(f"Missing staged Python interpreter for runtime group {group}")
        if group == "ollama" and (args.ollama_bin is None or args.ollama_models is None):
            raise ValueError("An Ollama study requires --ollama-bin and --ollama-models")
        if group == "ollama" and not args.ollama_bin.is_file():
            raise ValueError("Ollama executable is missing")
        for track in current_tracks:
            for shard_index in range(args.shards):
                selected, selection = select_tasks(current_cohorts[track], selection={"shard_count": args.shards,
                                                                            "shard_index": shard_index})
                group_jobs = groups.setdefault(group, [])
                group_jobs.append({"array_index": len(group_jobs), "experiment": experiment["name"],
                    "analysis_family": experiment.get("analysis_family", experiment["name"]),
                    "seed": experiment.get("seed", 0), "track": track,
                    "shard_index": shard_index, "shard_count": args.shards,
                    "selected_task_ids": [task["task_id"] for task in selected],
                    "selection_sha256": selection["selection_sha256"],
                    "output": f"runs/{experiment['name']}/{track}/shard-{shard_index:03d}"})
    snapshot = destination / "snapshot"
    snapshot.mkdir(parents=True)
    shutil.copytree(Path(__file__).resolve().parent, snapshot / "comparison", ignore=_ignored)
    shutil.copytree(bundle, snapshot / "instrument", ignore=_ignored)
    config["bundle"] = "instrument"
    write_json(snapshot / "configuration.json", config)
    # The authoritative source inventory must be computed by the copied module,
    # whose PACKAGE constant points into the immutable snapshot.
    program = "from pathlib import Path; from comparison.run import source_inventory; import json; print(json.dumps(source_inventory(Path('configuration.json'), Path('instrument'))))"
    result = subprocess.run([sys.executable, "-c", program], cwd=snapshot,
                            check=True, text=True, capture_output=True,
                            env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}})
    inventory = json.loads(result.stdout)
    study = {"protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by_job": os.environ["SLURM_JOB_ID"], "compute_node": os.environ["SLURMD_NODENAME"],
        "configuration_sha256": fingerprint(config), "snapshot_source_sha256": inventory,
        "snapshot_source_inventory_sha256": fingerprint(inventory),
        "source_release_validation": validation,
        "tracks": list(tracks), "shards": args.shards, "max_parallel_per_group": args.max_parallel,
        "cohorts": {track: [task["task_id"] for task in cohorts[track]] for track in tracks},
        "experiment_cohorts": experiment_cohorts,
        "python_paths": python_paths,
        "ollama_bin": str(args.ollama_bin.resolve()) if args.ollama_bin else None,
        "ollama_models": str(args.ollama_models.resolve()) if args.ollama_models else None,
        "groups": groups,
        "execution_policy": "Preparation does not submit jobs. Run explicit worker-commands.json entries inside suitable compute allocations, then merge. Do not execute the same worker twice.",
        "data_policy": "All-388 scores include authored train/validation rows. Heldout-114 scores are reported separately. Sequential cases have no assigned heldout split."}
    study["study_manifest_sha256"] = fingerprint(study)
    write_json(destination / "study.json", study)
    (destination / "logs").mkdir()
    commands = []
    for group, jobs in groups.items():
        environment = {}
        if group == "ollama":
            environment.update(COMPARISON_OLLAMA_BIN=study["ollama_bin"],
                               OLLAMA_MODELS=study["ollama_models"])
        for job in jobs:
            commands.append({"group": group, "index": job["array_index"],
                "cwd": str(snapshot), "environment": environment,
                "argv": [python_paths[group], "-m", "comparison.study", "worker",
                    "--study", str(destination), "--group", group,
                    "--index", str(job["array_index"])]})
    command_manifest = destination / "worker-commands.json"
    write_json(command_manifest, {"schema_version": 1,
        "study_manifest_sha256": study["study_manifest_sha256"], "workers": commands,
        "runtime_note": "An Ollama worker needs a caller-managed pinned loopback server and COMPARISON_OLLAMA_URL. Allocation, environment and concurrency are caller-owned.",
        "merge_after_workers": {"cwd": str(snapshot), "argv": [python_paths["hf"],
            "-m", "comparison.study", "merge", "--study", str(destination), "--allow-partial"]}})
    print(json.dumps({"status": "prepared", "study": str(destination),
                      "array_tasks": sum(map(len, groups.values())),
                      "groups": {key: len(value) for key, value in groups.items()},
                      "worker_commands": str(command_manifest)}))


def read_study(path, *, authenticate=True):
    root = Path(path).resolve()
    study = json.loads((root / "study.json").read_text())
    if study.get("protocol") != PROTOCOL:
        raise ValueError("Unknown study protocol")
    unsigned = {key: value for key, value in study.items() if key != "study_manifest_sha256"}
    if fingerprint(unsigned) != study.get("study_manifest_sha256"):
        raise ValueError("Study manifest changed after preparation")
    if authenticate:
        snapshot = root / "snapshot"
        if Path(__file__).resolve().parent != snapshot / "comparison":
            raise ValueError("Run study workers/merge from the frozen snapshot, not the mutable original source")
        actual = source_inventory(snapshot / "configuration.json", snapshot / "instrument")
        if actual != study["snapshot_source_sha256"]:
            raise ValueError("Frozen study source/configuration changed; create a fresh study")
        load_bundle(snapshot / "instrument")["validate"].validate_bundle(snapshot / "instrument")
    return root, study


def authenticate_validation_receipt(path):
    """Require the source/configuration bytes that passed semantic validation."""
    root = Path(path).resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt.get("status") != "validated" or receipt.get("semantic_smoke_assertions_passed") is not True:
        raise ValueError("A successful semantic validation receipt is required")
    snapshot = root / "snapshot"
    if Path(__file__).resolve().parent != snapshot / "comparison":
        raise ValueError("Verify validation through its frozen snapshot module")
    inventory = source_inventory(root / "configuration.json", snapshot / "instrument")
    if (receipt.get("snapshot_source_sha256") != inventory
            or receipt.get("snapshot_source_inventory_sha256") != fingerprint(inventory)):
        raise ValueError("Validated source/configuration changed after validation")
    load_bundle(snapshot / "instrument")["validate"].validate_bundle(snapshot / "instrument")
    return receipt


def worker(args):
    require_allocation()
    root, study = read_study(args.study)
    group = args.group or os.environ.get("COMPARISON_RUNTIME")
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    jobs = study["groups"].get(group)
    if jobs is None or not 0 <= index < len(jobs):
        raise ValueError("Unknown runtime group or array index")
    job = jobs[index]
    output = root / job["output"]
    command = [study["python_paths"][group], "-m", "comparison.run",
        "--config", str(root / "snapshot/configuration.json"), "--output", str(output),
        "--only", job["experiment"], "--track", job["track"],
        "--shard-index", str(job["shard_index"]), "--shard-count", str(job["shard_count"])]
    result = subprocess.run(command, cwd=root / "snapshot")
    write_json(root / "worker-receipts" / group / f"{index:04d}.json", {
        "study_manifest_sha256": study["study_manifest_sha256"], "job": job,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "compute_node": os.environ["SLURMD_NODENAME"], "exit_code": result.returncode,
        "command": command})
    return result.returncode


def validate_partition(expected_ids, selections, *, complete=True):
    """Reject overlaps and foreign tasks even when partial results are allowed."""
    observed = []
    expected = set(expected_ids)
    for selected in selections:
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate task inside a shard")
        if set(selected) - expected:
            raise ValueError("Foreign task IDs in a shard")
        observed.extend(selected)
    if len(observed) != len(set(observed)):
        raise ValueError("Overlapping shards cannot be merged")
    if complete and set(observed) != expected:
        raise ValueError("Shards do not exactly cover the configured cohort")
    return {"expected": len(expected_ids), "observed": len(observed),
            "missing_task_ids": [task_id for task_id in expected_ids if task_id not in set(observed)]}


def identity_signature(value):
    """Exclude node/timing/ephemeral transport details, preserve model bytes."""
    if isinstance(value, list):
        return [identity_signature(item) for item in value]
    if isinstance(value, dict):
        return {key: identity_signature(item) for key, item in value.items()
                if key not in {"load_wall_time_s", "hardware", "base_url", "latency_s"}}
    return value


def authenticate_identities(identities):
    """Compare every recorded model/runtime artifact, allowing failed lazy loads.

    An identity that never loaded weights may lack artifact fields. Every field
    present in two identities must agree, so a changed checkpoint or runtime
    cannot be hidden behind the allowed missing fields of another failed shard.
    """
    artifact_keys = {"weight_sha256", "local_files_sha256", "model_digest", "tokenizer_sha256",
                     "config_sha256", "loaded_generation_config_sha256", "loaded_chat_template_sha256", "server_version",
                     "torch", "transformers", "dtype", "effort", "trained_depths",
                     "native_package", "mechanism", "head_type", "model_type",
                     "artifact_sha256", "manifest_sha256", "encoder_signature",
                     "model_files_sha256", "feature_source_sha256", "model_dtype",
                     "sentence_transformers", "sklearn", "numpy", "scipy"}
    observed = {}
    def visit(value, prefix=()):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            path = prefix + (key,)
            if key in artifact_keys and item is not None:
                signature = fingerprint(item)
                if path in observed and observed[path] != signature:
                    raise ValueError("Incompatible model/runtime artifacts across shards: " + ".".join(path))
                observed[path] = signature
            elif isinstance(item, dict):
                visit(item, path)
    for identity in identities:
        visit(identity)
    return {"compared_artifact_fields": len(observed),
            "artifact_signature_sha256": fingerprint({".".join(key): value for key, value in observed.items()})}


def merge(args):
    require_allocation()
    root, study = read_study(args.study)
    output = args.output.resolve() if args.output else root / "merged"
    if output.exists():
        raise ValueError("Merge output must be a fresh directory")
    snapshot = root / "snapshot"
    config = json.loads((snapshot / "configuration.json").read_text())
    helpers = load_bundle(snapshot / "instrument")
    bundle = snapshot / "instrument"
    tasks, answers = helpers["common"].load_dataset(bundle / "mcqa_388/questions.jsonl.gz", bundle / "mcqa_388/answers.jsonl")
    cases = helpers["common"].load_jsonl(bundle / "sequential_120/evaluator/case-specs.jsonl")
    jobs = [job for group in study["groups"].values() for job in group]
    output.mkdir(parents=True)
    report = {"schema_version": 1, "protocol": PROTOCOL,
        "study_manifest_sha256": study["study_manifest_sha256"],
        "source_inventory_sha256": study["snapshot_source_inventory_sha256"],
        "configuration_sha256": study["configuration_sha256"], "experiments": [], "job_status": []}
    all_finished = True
    for experiment in config["experiments"]:
        name = experiment["name"]
        if name not in study["experiment_cohorts"]:
            continue
        row = {"name": name, "analysis_family": experiment.get("analysis_family", name),
               "seed": experiment.get("seed", 0), "method": experiment.get("method", "choice"),
               "architecture": experiment.get("architecture", "direct"),
               "reporting_scope": experiment.get("reporting_scope", "primary"),
               "primary_comparison": experiment.get("reporting_scope") != "T1_reference",
               "backend_kind": experiment["backend"].get("kind"), "tracks": {}}
        identities = []
        for track in study["experiment_cohorts"][name]:
            predictions, episodes, selected_sets = [], [], []
            planned = [job for job in jobs if job["experiment"] == name and job["track"] == track]
            for job in planned:
                run_output = root / job["output"]
                run_config = run_output / "configuration.json"
                run_report = run_output / "report.json"
                status = {"experiment": name, "track": track, "shard_index": job["shard_index"],
                          "output": str(run_output), "report_present": run_report.is_file()}
                report["job_status"].append(status)
                if not run_config.is_file():
                    all_finished = False
                    status["status"] = "not_started"
                    if not args.allow_partial:
                        raise ValueError(f"Missing run configuration: {run_output}")
                    continue
                metadata = json.loads(run_config.read_text())
                if metadata.get("source_inventory_sha256") != study["snapshot_source_inventory_sha256"]:
                    raise ValueError(f"Shard source differs from frozen study: {run_output}")
                if metadata.get("configuration_sha256") != study["configuration_sha256"]:
                    raise ValueError(f"Shard model/configuration differs from study: {run_output}")
                if metadata.get("selected_experiments") != [name] or metadata.get("track") != track:
                    raise ValueError("Shard contains unexpected experiments or tracks")
                selection = metadata.get("selection", {})
                if selection != {"shard_index": job["shard_index"], "shard_count": job["shard_count"], "task_ids": None} or metadata.get("limit") is not None:
                    raise ValueError("Shard selection/diagnostic limit differs from study plan")
                if run_report.is_file():
                    shard_report = json.loads(run_report.read_text())
                    status.update(status="finished" if "complete" in shard_report else "running",
                                  complete=shard_report.get("complete"))
                    all_finished = all_finished and "complete" in shard_report
                else:
                    status["status"] = "running_or_failed"
                    all_finished = False
                if status["status"] != "finished" and not args.allow_partial:
                    raise ValueError(f"Shard has no final report: {run_output}")
                directory = run_output / name / track
                selection_file = directory / "selection.json"
                if selection_file.is_file():
                    track_selection = json.loads(selection_file.read_text())
                    if (track_selection.get("selected_task_ids") != job["selected_task_ids"]
                            or track_selection.get("selection_sha256") != job["selection_sha256"]):
                        raise ValueError("Track-level shard IDs differ from prepared plan")
                current = _load_rows(directory / ("predictions.jsonl" if track == "mcqa" else "episodes.jsonl"))
                if any(record.get("task_id") not in set(job["selected_task_ids"]) for record in current):
                    raise ValueError("Shard produced a foreign task ID")
                if track == "mcqa":
                    if len({record["task_id"] for record in current}) != len(current):
                        raise ValueError("Duplicate MCQA predictions in shard")
                    predictions.extend(current)
                    selected_sets.append([record["task_id"] for record in current])
                else:
                    keys = [(record.get("task_id"), record.get("mode")) for record in current]
                    if len(keys) != len(set(keys)):
                        raise ValueError("Duplicate sequential episodes in shard")
                    episodes.extend(current)
                    selected_sets.append(list(dict.fromkeys(record["task_id"] for record in current)))
                identity_file = run_output / name / "identity.json"
                if identity_file.is_file():
                    identities.append(json.loads(identity_file.read_text()))
                status["records"] = len(current)
            coverage = validate_partition(study["experiment_cohorts"][name][track], selected_sets, complete=not args.allow_partial)
            destination = output / name / track
            destination.mkdir(parents=True)
            if track == "mcqa":
                _write_rows(destination / "predictions.jsonl", predictions)
                split = config.get("tracks", {}).get("mcqa", {}).get("split", "heldout")
                grading_tasks, grading_answers = experiment_mcqa_dataset(tasks, answers, experiment)
                summary = helpers["mcqa"].score_predictions(grading_tasks, grading_answers, predictions, split)
                summary.update(calibration=mcqa_calibration(experiment_mcqa_cohort(tasks, split, experiment), grading_answers, predictions),
                               heldout_report=helpers["mcqa"].score_predictions(grading_tasks, grading_answers, predictions, "heldout"),
                               heldout_calibration=mcqa_calibration(experiment_mcqa_cohort(tasks, "heldout", experiment), grading_answers, predictions),
                               reporting_scope=experiment.get("reporting_scope", "primary"),
                               primary_comparison=experiment.get("reporting_scope") != "T1_reference")
            else:
                _write_rows(destination / "episodes.jsonl", episodes)
                mode = config.get("tracks", {}).get("sequential", {}).get("mode", "both")
                modes = helpers["sequential"].MODES if mode == "both" else (mode,)
                summary = helpers["sequential"].score_episodes(cases, episodes, modes)
                summary.update(full_dataset_size=len(cases), selected_cases=coverage["observed"],
                               conditional_menu_confidence=sequential_calibration(episodes))
            summary.update(shard_merge_coverage=coverage)
            write_json(destination / "summary.json", summary)
            row["tracks"][track] = summary
        signatures = {fingerprint(identity_signature(identity)) for identity in identities}
        # A failed lazy model load can leave a less detailed identity. Preserve
        # all identities for auditing instead of dropping explicit failures.
        row["identity_signatures"] = sorted(signatures)
        row["artifact_authentication"] = authenticate_identities(identities)
        row["identities"] = identities
        report["experiments"].append(row)
    report["all_jobs_finished"] = all_finished
    report["complete_full_benchmark"] = all_finished and all(
        all((summary.get("coverage") == 1) if track == "mcqa" else summary.get("complete", False)
            for track, summary in row["tracks"].items()) for row in report["experiments"])
    report["diagnostic_subset"] = not report["complete_full_benchmark"]
    read_study(root)
    write_json(output / "report.json", report)
    print(json.dumps({"status": "merged", "output": str(output),
                      "all_jobs_finished": all_finished,
                      "complete_full_benchmark": report["complete_full_benchmark"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--track", choices=("mcqa", "sequential", "both"), default="both")
    prepare_parser.add_argument("--shards", type=int, default=8)
    prepare_parser.add_argument("--max-parallel", type=int, default=8,
                                help="recorded caller concurrency hint; preparation launches no workers")
    prepare_parser.add_argument("--hf-python", type=Path, required=True)
    prepare_parser.add_argument("--apus-python", type=Path)
    prepare_parser.add_argument("--ollama-bin", type=Path)
    prepare_parser.add_argument("--ollama-models", type=Path)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--study", type=Path, required=True)
    worker_parser.add_argument("--group")
    worker_parser.add_argument("--index", type=int)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--study", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path)
    merge_parser.add_argument("--allow-partial", action="store_true")
    validation_parser = commands.add_parser("verify-validation")
    validation_parser.add_argument("--validated", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require_allocation()
        if args.command == "verify-validation":
            authenticate_validation_receipt(args.validated)
            print(json.dumps({"status": "validation_source_authenticated", "validated": str(args.validated)}))
            return 0
        result = {"prepare": prepare, "worker": worker, "merge": merge}[args.command](args)
        return 0 if result is None else result
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
