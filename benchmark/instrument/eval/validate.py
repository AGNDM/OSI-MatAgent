"""Validate the portable release without loading simulators or models.

The manifest authenticates the files shipped together, rather than establishing
the scientific validity of their contents. Native replay receipts are provenance;
this command checks the static release and does not rerun those simulations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

if __package__:
    from .common import load_dataset
    from .prompts import solver_view
else:
    from common import load_dataset
    from prompts import solver_view


DEFAULT_BUNDLE = Path(__file__).resolve().parents[1]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE_PROMPT_FIELDS = frozenset({
    "correct_option_ids", "answer", "oracle", "oracles", "reference_steps",
    "canonical_actions", "setup_steps", "scheduled_events", "fault_family",
    "fault_type", "seed", "base_case_id", "task_id", "group_id", "split",
    "difficulty", "tier", "family_id",
})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative_path(bundle, name):
    """Resolve a canonical portable path without following an escape."""
    _require(isinstance(name, str) and bool(name), "Manifest paths must be nonempty strings")
    path = PurePosixPath(name)
    _require("\\" not in name and "\x00" not in name
             and not path.is_absolute() and not PureWindowsPath(name).drive
             and all(part not in {"", ".", ".."} for part in name.split("/")),
             f"Unsafe bundle path: {name!r}")
    candidate = bundle.joinpath(*path.parts)
    try:
        candidate.resolve().relative_to(bundle)
    except (ValueError, OSError, RuntimeError) as error:
        raise ValueError(f"Bundle path escapes its root: {name!r}") from error
    return candidate


def _ignored(relative):
    parts = relative.parts
    return (any(part in {"__pycache__", ".git", ".venv", ".venv-models",
                        ".pytest_cache", ".cache", "output"} for part in parts)
            or relative.suffix in {".pyc", ".pyo"}
            or relative.as_posix() in {"manifest.json", "validation.json"})


def _publishable_files(bundle):
    """Inventory files without traversing ignored or linked directories."""
    found = set()
    pending = [bundle]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            relative = path.relative_to(bundle)
            if _ignored(relative):
                continue
            _require(not path.is_symlink(), f"Publishable symlinks are not portable: {relative.as_posix()}")
            if path.is_dir():
                pending.append(path)
            else:
                _require(path.is_file(), f"Unsupported bundle entry: {relative.as_posix()}")
                found.add(relative.as_posix())
    return found


def _verify_files(bundle, entries):
    _require(isinstance(entries, dict) and bool(entries), "Manifest files must be a nonempty mapping")
    for name, expected in entries.items():
        path = _relative_path(bundle, name)
        _require(not _ignored(PurePosixPath(name)), f"Excluded runtime or receipt file listed in manifest: {name}")
        _require(isinstance(expected, dict), f"Invalid file metadata: {name}")
        digest = expected.get("sha256")
        size = expected.get("bytes")
        _require(isinstance(digest, str) and _SHA256.fullmatch(digest), f"Invalid sha256 metadata: {name}")
        _require(type(size) is int and size >= 0, f"Invalid byte count metadata: {name}")
        _require(path.is_file(), f"Missing manifest file: {name}")
        parts = PurePosixPath(name).parts
        _require(not any(bundle.joinpath(*parts[:end]).is_symlink() for end in range(1, len(parts) + 1)),
                 f"Manifest file uses a symlink: {name}")
        _require(path.stat().st_size == size, f"Byte count mismatch: {name}")
        actual = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                actual.update(block)
        _require(actual.hexdigest() == digest, f"SHA256 mismatch: {name}")
    actual_names = _publishable_files(bundle)
    unlisted = sorted(actual_names - entries.keys())
    missing = sorted(entries.keys() - actual_names)
    _require(not unlisted, f"Unlisted publishable files: {', '.join(unlisted)}")
    _require(not missing, f"Missing manifest files: {', '.join(missing)}")
    return len(entries)


def _declared_count(metadata, key):
    value = metadata.get(key)
    _require(type(value) is int and value > 0, f"Dataset {key} must be a positive integer")
    return value


def _verify_splits(bundle, directory_name, tasks, files):
    directory = _relative_path(bundle, directory_name)
    _require(directory.is_dir(), f"Missing split directory: {directory_name}")
    expected = {}
    groups = {}
    for task in tasks:
        task_id, split, group = task["task_id"], task.get("split"), task.get("group_id")
        _require(isinstance(split, str) and bool(split), f"Missing split for task {task_id}")
        _require(isinstance(group, str) and bool(group), f"Missing group_id for task {task_id}")
        _require(group not in groups or groups[group] == split, f"Group crosses split boundaries: {group}")
        groups[group] = split
        expected.setdefault(split, set()).add(task_id)
    actual = {}
    seen = set()
    for path in sorted(directory.glob("*.txt")):
        _require(path.relative_to(bundle).as_posix() in files,
                 f"Split file is outside the hashed inventory: {path.name}")
        split = path.stem
        ids = path.read_text(encoding="utf-8").splitlines()
        _require(all(value and value == value.strip() for value in ids), f"Empty or padded ID in split {split}")
        _require(len(ids) == len(set(ids)), f"Duplicate IDs in split {split}")
        overlap = seen.intersection(ids)
        _require(not overlap, f"Split files overlap: {', '.join(sorted(overlap))}")
        seen.update(ids)
        actual[split] = set(ids)
    _require(actual.keys() == expected.keys(), "Split files do not match the task split names")
    for split in expected:
        _require(actual[split] == expected[split], f"Split IDs do not match task assignments: {split}")
    _require(seen == {task["task_id"] for task in tasks}, "Split files are not exhaustive")
    return {split: len(ids) for split, ids in sorted(actual.items())}, len(groups)


def _verify_prompt_boundary(tasks):
    """Check only the solver projection; raw release metadata stays available."""
    for task in tasks:
        view = solver_view(task)
        _require(not _PRIVATE_PROMPT_FIELDS.intersection(view),
                 f"Evaluator metadata entered the prompt for task {task['task_id']}")
        poisoned = dict(task)
        poisoned.update({key: "PRIVATE_VALIDATION_SENTINEL" for key in _PRIVATE_PROMPT_FIELDS})
        _require(solver_view(poisoned) == view,
                 f"Evaluator metadata changes the prompt for task {task['task_id']}")
        contract = view.get("contract", {})
        _require("fault.inject" not in contract.get("capabilities", {}), "Evaluator fault injection exposed in a prompt")
        _require(not any(key.startswith("truth.") for key in contract.get("state_slots", {})),
                 "Evaluator truth exposed in a prompt")


def validate_bundle(bundle=DEFAULT_BUNDLE):
    """Return a compact receipt, or raise ValueError for an invalid release."""
    bundle = Path(bundle).resolve()
    _require(bundle.is_dir(), f"Bundle directory does not exist: {bundle}")
    manifest_path = _relative_path(bundle, "manifest.json")
    _require(not manifest_path.is_symlink(), "Manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "Manifest must be one JSON object")
    version = manifest.get("bundle_version")
    _require(isinstance(version, str) and bool(version), "Missing bundle_version")
    files = _verify_files(bundle, manifest.get("files"))
    datasets = manifest.get("datasets")
    _require(isinstance(datasets, dict) and isinstance(datasets.get("mcqa_388"), dict),
             "Manifest lacks the mcqa_388 dataset")
    dataset = datasets["mcqa_388"]
    question_count = _declared_count(dataset, "questions")
    group_count = _declared_count(dataset, "groups")
    # Minimal answer records deliberately omit private base-case identities.
    # Retain this source-provenance count without deriving it from task IDs.
    base_cases = _declared_count(dataset, "base_cases")
    _require(base_cases <= question_count and group_count <= question_count,
             "Declared groups or base cases exceed the number of questions")
    question_file = dataset.get("questions_file", "mcqa_388/questions.jsonl.gz")
    answer_file = dataset.get("answers_file", "mcqa_388/answers.jsonl")
    questions = _relative_path(bundle, question_file)
    answers = _relative_path(bundle, answer_file)
    _require(question_file in manifest["files"] and answer_file in manifest["files"],
             "Dataset questions and answers must belong to the hashed file inventory")
    tasks, keys = load_dataset(questions, answers)
    _require(len(tasks) == question_count, f"Question count mismatch: expected {question_count}, found {len(tasks)}")
    _require(len(keys) == question_count, "Answer count does not match the question count")
    for task in tasks:
        correct = keys[task["task_id"]].get("correct_option_ids")
        _require(isinstance(correct, list) and len(correct) == 1,
                 f"Expected one correct option for task {task['task_id']}")
        offered = {option["option_id"] for option in task["options"]}
        _require(correct[0] in offered, f"Answer is outside the option menu for task {task['task_id']}")
    split_counts, groups = _verify_splits(bundle, dataset.get("split_dir", "mcqa_388/splits"), tasks,
                                         manifest["files"])
    _require(groups == group_count, f"Group count mismatch: expected {group_count}, found {groups}")
    _verify_prompt_boundary(tasks)
    report = {"status": "passed", "bundle_version": version,
            "counts": {"questions": len(tasks), "answers": len(keys), "groups": groups,
                       "base_cases_declared": base_cases, "files": files},
            "split_counts": split_counts, "prompt_boundary_checked": True,
            "native_replay_performed": False}
    sequential = datasets.get("sequential_120", {})
    if sequential.get("data_included"):
        if __package__:
            from .validate_sequential import validate_sequential_dataset
        else:
            from validate_sequential import validate_sequential_dataset
        report["sequential_120"] = validate_sequential_dataset(bundle, sequential, manifest["files"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_bundle(args.bundle)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
