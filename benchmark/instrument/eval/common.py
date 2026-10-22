"""Strict JSONL loading and release-integrity checks; no simulator required."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read plain or gzip JSONL, rejecting ambiguous or malformed records.

    Empty lines are skipped. Duplicate object keys and non-finite numbers are
    rejected even inside optional fields, preventing different JSON readers
    from silently interpreting the same prediction file differently.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_invalid_constant,
                    parse_float=_finite_float,
                )
                if not isinstance(row, dict):
                    raise ValueError("each JSONL record must be an object")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append(row)
    return rows


def _record_id(row: dict[str, Any], context: str) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"{context}: record must be an object")
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{context}: task_id must be a nonempty string")
    return task_id


def validate_dataset(
    tasks: list[dict[str, Any]], answers: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Check IDs, menu uniqueness, answer coverage, and correct-option references."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("questions must be a nonempty list")
    if not isinstance(answers, dict):
        raise ValueError("answers must map task IDs to answer records")
    index: dict[str, dict[str, Any]] = {}
    menus: dict[str, set[str]] = {}
    for position, task in enumerate(tasks, 1):
        task_id = _record_id(task, f"question {position}")
        if task_id in index:
            raise ValueError(f"duplicate question task_id: {task_id!r}")
        for field in ("instrument_id", "domain", "task_type", "category", "difficulty", "tier", "split"):
            if field in task and (not isinstance(task[field], str) or not task[field]):
                raise ValueError(f"{task_id}: {field} must be a nonempty string")
        options = task.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"{task_id}: options must contain at least two choices")
        option_ids: set[str] = set()
        for option in options:
            option_id = option.get("option_id") if isinstance(option, dict) else None
            if not isinstance(option_id, str) or not option_id:
                raise ValueError(f"{task_id}: option_id must be a nonempty string")
            if option_id in option_ids:
                raise ValueError(f"{task_id}: duplicate option_id {option_id!r}")
            option_ids.add(option_id)
        index[task_id] = task
        menus[task_id] = option_ids
    missing = set(index) - set(answers)
    extra = set(answers) - set(index)
    if missing or extra:
        raise ValueError(
            "answer IDs must exactly cover question IDs "
            f"(missing={len(missing)}, unknown={len(extra)})"
        )
    for task_id, answer in answers.items():
        if not isinstance(answer, dict):
            raise ValueError(f"{task_id}: answer must be an object")
        if "task_id" in answer and answer["task_id"] != task_id:
            raise ValueError(f"{task_id}: answer record task_id disagrees with its key")
        correct_ids = answer.get("correct_option_ids")
        if (
            not isinstance(correct_ids, list)
            or not correct_ids
            or any(not isinstance(option_id, str) or not option_id for option_id in correct_ids)
        ):
            raise ValueError(f"{task_id}: correct_option_ids must be a nonempty string list")
        if len(set(correct_ids)) != len(correct_ids):
            raise ValueError(f"{task_id}: duplicate correct_option_ids")
        if not set(correct_ids).issubset(menus[task_id]):
            raise ValueError(f"{task_id}: correct_option_ids reference unknown choices")
    return index


def load_dataset(
    questions: str | Path, answers: str | Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return public task records and a checked task-ID-to-answer mapping."""
    tasks = load_jsonl(questions)
    answer_map: dict[str, dict[str, Any]] = {}
    for position, answer in enumerate(load_jsonl(answers), 1):
        task_id = _record_id(answer, f"answer {position}")
        if task_id in answer_map:
            raise ValueError(f"duplicate answer task_id: {task_id!r}")
        answer_map[task_id] = answer
    validate_dataset(tasks, answer_map)
    return tasks, answer_map
