"""Score saved MCQA predictions against the frozen instrument answer keys.

Run from any working directory::

    python /path/to/benchmark/instrument/eval/mcqa.py --predictions predictions.jsonl

All target questions remain in the accuracy denominator. Missing responses,
non-menu selections, and non-"ok" statuses score zero. Probabilities are ignored:
this evaluator scores the submitted choice, never its most likely probability.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

if __package__:
    from .common import _record_id, load_dataset, load_jsonl, validate_dataset
else:
    from common import _record_id, load_dataset, load_jsonl, validate_dataset


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_SPLITS = frozenset({"test", "test_template", "test_fault", "test_instrument"})
SPLIT_CHOICES = (
    "heldout", "all", "train", "validation", "test", "test_template", "test_fault", "test_instrument"
)
CATEGORY_NAMES = {
    "L1": "Single-action checks",
    "L2": "Direct diagnosis and simple recovery",
    "L3": "Evidence assessment, testing and escalation",
    "L4": "Procedure and service plan selection",
    "L5": "Interruption response plan selection",
}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(row["correct"] for row in rows)
    valid = sum(row["valid"] for row in rows)
    missing = sum(row["missing"] for row in rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "coverage": valid / n if n else None,
        "valid": valid,
        "submitted": n - missing,
        "invalid": n - missing - valid,
        "missing": missing,
        "uniform_random_accuracy": sum(row["random_accuracy"] for row in rows) / n if n else None,
    }


def score_predictions(
    tasks: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    split: str = "heldout",
) -> dict[str, Any]:
    """Score one prediction per known task; reject duplicate or unknown IDs.

    A complete prediction file may include known tasks outside the selected
    cohort. Those rows are counted in ``ignored_out_of_cohort`` and contribute
    no score. Coverage is the fraction of cohort tasks with a valid menu choice
    and status ``ok``. Accuracy, including every subgroup, keeps missing and
    invalid responses in its denominator. Probabilities do not affect validity.
    """
    if split not in SPLIT_CHOICES:
        raise ValueError(f"unknown split {split!r}; choose from {', '.join(SPLIT_CHOICES)}")
    index = validate_dataset(tasks, answers)
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list of records")
    prediction_map: dict[str, dict[str, Any]] = {}
    for position, prediction in enumerate(predictions, 1):
        task_id = _record_id(prediction, f"prediction {position}")
        if task_id in prediction_map:
            raise ValueError(f"duplicate prediction task_id: {task_id!r}")
        if task_id not in index:
            raise ValueError(f"unknown prediction task_id: {task_id!r}")
        prediction_map[task_id] = prediction
    cohort = [
        task for task in tasks
        if split == "all"
        or (split == "heldout" and task.get("split") in HELDOUT_SPLITS)
        or (split != "heldout" and task.get("split") == split)
    ]
    if not cohort:
        raise ValueError(f"no questions match split {split!r}")
    rows: list[dict[str, Any]] = []
    for task in cohort:
        task_id = task["task_id"]
        prediction = prediction_map.get(task_id)
        option_ids = {option["option_id"] for option in task["options"]}
        selected = prediction.get("selected_option_id") if prediction is not None else None
        valid = (
            prediction is not None
            and prediction.get("status", "ok") == "ok"
            and isinstance(selected, str)
            and selected in option_ids
        )
        category = task.get("category")
        if not isinstance(category, str) or not category:
            category = CATEGORY_NAMES.get(task.get("difficulty"), task.get("difficulty", "unspecified"))
        rows.append({
            "task_id": task_id,
            "correct": bool(valid and selected in answers[task_id]["correct_option_ids"]),
            "valid": bool(valid),
            "missing": prediction is None,
            "random_accuracy": len(answers[task_id]["correct_option_ids"]) / len(option_ids),
            "option_count": str(len(option_ids)),
            "category": category,
            "instrument": task.get("instrument_id", "unspecified"),
            "domain": task.get("domain", "unspecified"),
            "task_type": task.get("task_type", "unspecified"),
            "split": task.get("split", "unspecified"),
            "tier": task.get("tier", "unspecified"),
        })
    result = _summarize(rows)
    result["split"] = split
    result["dataset_size"] = len(tasks)
    result["prediction_rows"] = len(predictions)
    result["ignored_out_of_cohort"] = len(prediction_map) - sum(not row["missing"] for row in rows)
    for field in ("option_count", "category", "instrument", "domain", "task_type", "split", "tier"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[f"by_{field}"] = {key: _summarize(value) for key, value in sorted(groups.items())}
    tier_accuracies = [group["accuracy"] for group in result["by_tier"].values()]
    result["tier_macro_accuracy"] = sum(tier_accuracies) / len(tier_accuracies)
    result["definitions"] = {
        "accuracy": "correct / all questions in the selected cohort; missing and invalid count as wrong",
        "coverage": "valid submitted menu selections with status ok / all cohort questions",
        "invalid": "submitted cohort rows with a non-menu/non-string/missing choice or status other than ok",
        "heldout": "union of test, test_template, test_fault and test_instrument",
        "tier_macro_accuracy": "unweighted mean accuracy over tiers present in the selected cohort",
        "uniform_random_accuracy": "mean(number of correct choices / number of offered choices)",
        "probabilities": "ignored; selected_option_id is scored directly",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="plain or gzip JSONL model predictions")
    parser.add_argument("--questions", type=Path, default=BUNDLE_ROOT / "mcqa_388/questions.jsonl.gz")
    parser.add_argument("--answers", type=Path, default=BUNDLE_ROOT / "mcqa_388/answers.jsonl")
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="heldout")
    parser.add_argument("--output", type=Path, help="also save the JSON score report here")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() in {
            args.predictions.resolve(), args.questions.resolve(), args.answers.resolve()
        }:
            raise ValueError("--output must not overwrite a predictions, questions, or answers input file")
        tasks, answers = load_dataset(args.questions, args.answers)
        report = score_predictions(tasks, answers, load_jsonl(args.predictions), args.split)
        rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError, EOFError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
