"""Export authenticated static MCQA subgroup scores on a Slurm compute node."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
    raise RuntimeError("A Slurm compute allocation is required")
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--overall', type=Path, required=True)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--taxonomy', type=Path, required=True)
parser.add_argument('--source-root', type=Path,
                    help='Optional relocated scorer/bundle root; its recorded hashes are still checked')
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
ORIGINAL, SNAPSHOT, TAXONOMY = (path.resolve() for path in (args.overall, args.snapshot, args.taxonomy))
RESULTS = args.output_dir.resolve()
RESULTS.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(actual, expected, label):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Metric mismatch {label}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"Mismatch {label}: {actual!r} != {expected!r}")


snapshot = json.loads(SNAPSHOT.read_text())
original_bytes = ORIGINAL.read_bytes()
original = json.loads(original_bytes)
same(original["snapshot_generated_utc"], snapshot["generated_utc"], "snapshot timestamp")
same(original["methods"], snapshot["methods"] and [
    {key: value for key, value in row.items() if key not in {"teacher_forced_attempts", "free_run_attempts"}}
    for row in snapshot["methods"]
], "original export methods")
source = args.source_root.resolve() if args.source_root else Path(snapshot["source"])
for name, expected in snapshot["report_source_sha256"].items():
    same(sha256(source / "comparison" / name), expected, f"frozen scoring source {name}")
bundle = source / "instrument"
same(sha256(bundle / "manifest.json"), snapshot["bundle_sha256"], "frozen bundle")
same(sha256(TAXONOMY), snapshot["taxonomy_sha256"], "frozen taxonomy")

sys.path.insert(0, str(source))
from comparison.study_report import mcqa_rows
from comparison.study_metrics import calibration, mean
from comparison.taxonomy import load_jsonl
from instrument.eval.mcqa import score_predictions

manifest_path = bundle / "mcqa_388" / "manifest.json"
manifest = json.loads(manifest_path.read_text())
questions_path = bundle / "mcqa_388" / "questions.jsonl.gz"
answers_path = bundle / "mcqa_388" / "answers.jsonl"
same(sha256(questions_path), manifest["question_archive_sha256"], "question archive")
same(sha256(answers_path), manifest["answers_sha256"], "answers")
taxonomy = json.loads(TAXONOMY.read_text())
same(sha256(questions_path), taxonomy["mcqa_questions_sha256"], "taxonomy questions")
tasks = load_jsonl(questions_path)
answers_records = {row["task_id"]: row for row in load_jsonl(answers_path)}
answers = {task_id: row["correct_option_ids"] for task_id, row in answers_records.items()}
task_index = {task["task_id"]: task for task in tasks}
same(len(tasks), 388, "question count")
same(len(task_index), len(tasks), "unique questions")
same(set(task_index), set(answers), "answer IDs")
category_names = manifest["category_names"]
categories = ["L1", "L2", "L3", "L4", "L5"]
choice_counts = ["2", "3", "4"]
methods_by_name = {method["experiment"]: method for method in original["methods"]}
same(len(methods_by_name), 34, "method count")
predictions = {name: {} for name in methods_by_name}
authenticated_files = []

for recorded_source in snapshot["sources"]:
    mcqa_files = {relative: expected for relative, expected in recorded_source.get("files", {}).items()
                  if relative.endswith("/mcqa/predictions.jsonl")}
    if not mcqa_files:
        continue
    run = Path(recorded_source["path"])
    same(sha256(run / "configuration.json"), recorded_source["configuration_sha256"], f"configuration {run}")
    for relative, expected in mcqa_files.items():
        path = run / relative
        name = Path(relative).parts[0]
        if name not in methods_by_name:
            raise ValueError(f"Unknown experiment in authenticated snapshot: {name}")
        same(sha256(path), expected, f"prediction file {path}")
        authenticated_files.append({"path": str(path), "sha256": expected, "experiment": name})
        for record in load_jsonl(path):
            task_id = record["task_id"]
            if task_id not in task_index:
                raise ValueError(f"Unknown question {task_id}")
            if task_id in predictions[name]:
                raise ValueError(f"Duplicate question {task_id} for {name}")
            predictions[name][task_id] = record


def summarize(rows):
    n = len(rows)
    attempted = sum(row["attempted"] for row in rows)
    valid = sum(row["valid"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    confident = [row for row in rows if row.get("confidence") is not None]
    brier_rows = [row for row in rows if row.get("brier") is not None]
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "attempted": attempted,
        "attempt_coverage": attempted / n if n else None,
        "valid": valid,
        "response_coverage": valid / n if n else None,
        "invalid": attempted - valid,
        "missing": n - attempted,
        "infrastructure_errors": sum(row["status"] == "infrastructure_error" for row in rows),
        "model_output_invalid": sum(row["status"] == "model_output_invalid" for row in rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "uniform_random_accuracy": mean(len(answers[row["task_id"]]) / len(task_index[row["task_id"]]["options"]) for row in rows),
        "ece": calibration(rows, denominator=n).get("ece") if n else None,
        "confidence_n": len(confident),
        "multiclass_brier": mean(row["brier"] for row in brier_rows),
        "brier_n": len(brier_rows),
    }


def group_rows(rows, axis, keys):
    groups = {key: [] for key in keys}
    for row in rows:
        if axis == "category":
            key = row["legacy_level"]
        else:
            key = str(len(task_index[row["task_id"]]["options"]))
        if key not in groups:
            raise ValueError(f"Unexpected {axis}: {key}")
        groups[key].append(row)
    grouped = {key: summarize(value) for key, value in groups.items()}
    total = summarize(rows)
    for field in ("n", "correct", "attempted", "valid", "invalid", "missing", "infrastructure_errors", "model_output_invalid"):
        same(sum(group[field] for group in grouped.values()), total[field], f"{axis} reconciliation {field}")
    return grouped


def check_portable(group, portable, label):
    for left, right in (("n", "n"), ("correct", "correct"), ("accuracy", "accuracy"),
                        ("attempted", "submitted"), ("valid", "valid"),
                        ("response_coverage", "coverage"), ("invalid", "invalid"),
                        ("missing", "missing"), ("uniform_random_accuracy", "uniform_random_accuracy")):
        same(group[left], portable[right], f"portable scorer {label} {left}")


category_methods = []
choice_methods = []
for method in original["methods"]:
    name = method["experiment"]
    selected_tasks = tasks if method["reporting_scope"] != "T1_reference" else [task for task in tasks if task.get("tier") == "T1"]
    selected_ids = {task["task_id"] for task in selected_tasks}
    if set(predictions[name]) - selected_ids:
        raise ValueError(f"Predictions outside declared scope for {name}")
    rows = mcqa_rows(selected_tasks, answers, predictions[name], taxonomy)
    heldout = [row for row in rows if str(row.get("split", "")).startswith("test")]
    total = summarize(rows)
    heldout_total = summarize(heldout)
    same(total["n"], method["mcqa_cohort_n"], f"{name} full denominator")
    same(total["attempted"], method["mcqa_attempts"], f"{name} full attempts")
    for old_key, new_key in (
        ("heldout_n", "n"), ("heldout_accuracy", "accuracy"),
        ("heldout_response_coverage", "response_coverage"), ("heldout_ece", "ece"),
        ("heldout_confidence_n", "confidence_n"), ("heldout_multiclass_brier", "multiclass_brier"),
        ("heldout_brier_n", "brier_n"), ("heldout_infrastructure_envelopes", "infrastructure_errors"),
        ("heldout_invalid_model_outputs", "model_output_invalid")
    ):
        same(heldout_total[new_key], method[old_key], f"{name} existing {old_key}")

    normalized = []
    for task_id, prediction in predictions[name].items():
        backend = prediction.get("backend_result", prediction)
        normalized.append({"task_id": task_id,
                           "status": prediction.get("status", "ok"),
                           "selected_option_id": prediction.get("selected_option_id", backend.get("selected_option_id"))})
    portable_answers = {task_id: answers_records[task_id] for task_id in selected_ids}
    common = {key: method[key] for key in ("experiment", "architecture", "comparison_group", "reporting_scope")}
    category_result = dict(common)
    choice_result = dict(common)
    for scope, cohort_rows in (("all_questions", rows), ("heldout", heldout)):
        current_total = summarize(cohort_rows)
        by_category = group_rows(cohort_rows, "category", categories)
        by_choices = group_rows(cohort_rows, "choice_count", choice_counts)
        portable = score_predictions(selected_tasks, portable_answers, normalized, "all" if scope == "all_questions" else "heldout")
        check_portable(current_total, portable, f"{name}/{scope}")
        for key, subgroup in by_category.items():
            if subgroup["n"]:
                check_portable(subgroup, portable["by_category"][category_names[key]], f"{name}/{scope}/{key}")
            else:
                same(subgroup["accuracy"], None, f"empty category {name}/{key}")
        for key, subgroup in by_choices.items():
            if subgroup["n"]:
                check_portable(subgroup, portable["by_option_count"][key], f"{name}/{scope}/{key}-choice")
            else:
                same(subgroup["accuracy"], None, f"empty option count {name}/{key}")
        category_result[scope] = {"overall": current_total, "groups": by_category}
        choice_result[scope] = {"overall": current_total, "groups": by_choices}
    category_methods.append(category_result)
    choice_methods.append(choice_result)
    print(f"Verified {name}", flush=True)

heldout_tasks = [task for task in tasks if str(task.get("split", "")).startswith("test")]
category_distribution = {label: {
    "name": category_names[label],
    "all_questions_n": sum(task["difficulty"] == label for task in tasks),
    "heldout_n": sum(task["difficulty"] == label for task in heldout_tasks),
} for label in categories}
choice_distribution = {count: {
    "offered_choice_count": int(count),
    "all_questions_n": sum(len(task["options"]) == int(count) for task in tasks),
    "heldout_n": sum(len(task["options"]) == int(count) for task in heldout_tasks),
} for count in choice_counts}
same([category_distribution[key]["all_questions_n"] for key in categories], [158, 50, 88, 62, 30], "full categories")
same([category_distribution[key]["heldout_n"] for key in categories], [44, 20, 18, 22, 10], "heldout categories")
same([choice_distribution[key]["all_questions_n"] for key in choice_counts], [182, 136, 70], "full option counts")
same([choice_distribution[key]["heldout_n"] for key in choice_counts], [64, 26, 24], "heldout option counts")

common_document = {
    "schema_version": "1.0",
    "benchmark": original["benchmark"],
    "track": "static_mcqa",
    "snapshot_generated_utc": snapshot["generated_utc"],
    "export_generated_utc": datetime.now(timezone.utc).isoformat(),
    "primary_reporting_scope": "heldout",
    "dataset": original["dataset"],
    "pretrained_models": original["pretrained_models"],
    "method_count": len(category_methods),
    "metric_definitions": {
        "accuracy": "correct / all questions in the selected subgroup; invalid, missing and non-ok answers count as incorrect",
        "response_coverage": "valid offered selections with status ok / all questions in the subgroup",
        "attempt_coverage": "recorded predictions / all questions in the subgroup",
        "invalid": "recorded predictions that are not valid, including infrastructure/model errors",
        "n": "number of questions in the declared method scope and selected subgroup",
        "empty_groups": "n = 0 and accuracy = null; empty groups are not scored as zero accuracy",
        "all_questions": "full method cohort, normally 388; includes train and validation diagnostics",
        "heldout": "union of test, test_fault, test_instrument and test_template; normally 114 questions",
        "rates": "fractions in [0,1], not percentages",
        "uniform_random_accuracy": "mean(number of accepted answers / number of offered choices)",
        "ece": "canonical expected calibration error on available selected-answer confidence values",
        "multiclass_brier": "canonical multiclass Brier score where a full probability distribution is available",
    },
    "provenance": {
        "source_snapshot": str(SNAPSHOT),
        "source_snapshot_sha256": sha256(SNAPSHOT),
        "overall_results": "mcqa-results.json",
        "overall_results_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "question_archive_sha256": sha256(questions_path),
        "answers_sha256": sha256(answers_path),
        "taxonomy_sha256": snapshot["taxonomy_sha256"],
        "frozen_scoring_source_sha256": snapshot["report_source_sha256"],
        "export_script_sha256": sha256(__file__),
        "source_snapshot_slurm_job_id": snapshot["slurm_job_id"],
        "export_slurm_job_id": os.environ["SLURM_JOB_ID"],
        "export_compute_node": os.environ["SLURMD_NODENAME"],
        "authenticated_prediction_files": len(authenticated_files),
        "prediction_files": authenticated_files,
        "neural_inference_performed": False,
        "hosted_calls": 0,
    },
    "verification": {
        "all_existing_34_heldout_summaries_match": True,
        "source_prediction_hashes_match_snapshot": True,
        "subgroup_counts_and_correct_counts_reconcile": True,
        "independent_portable_scorer_agrees": True,
        "empty_scoped_reference_groups_are_null": True,
    },
    "interpretation": list(original["interpretation"]),
}
category_document = dict(common_document, group_by="legacy_task_category", source_task_field="difficulty",
    empirical_difficulty_order_established=False, group_definitions=category_distribution,
    methods=category_methods)
category_document["interpretation"] = common_document["interpretation"] + [
    "L1-L5 are task-construction categories, not a validated increasing difficulty ranking.",
    "L4/L5 select offered plans in static MCQA; they do not measure autonomous sequential execution.",
]
choice_document = dict(common_document, group_by="offered_choice_count", source_task_field="len(options)",
    group_definitions=choice_distribution, methods=choice_methods)
choice_document["interpretation"] = common_document["interpretation"] + [
    "Groups 2, 3 and 4 count offered answer choices, not semantic classes.",
    "Different option-count groups have different task-category mixtures; their accuracies are descriptive, not a controlled effect of choice count.",
]

outputs = {
    RESULTS / "mcqa-results-by-category.json": category_document,
    RESULTS / "mcqa-results-by-option-count.json": choice_document,
}
for path in outputs:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing export {path}")
same(ORIGINAL.read_bytes(), original_bytes, "original export remains unchanged")
for path, document in outputs.items():
    serialized = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(serialized, encoding="utf-8")
    same(json.loads(path.read_text()), document, f"saved JSON round trip {path.name}")
    print(f"WROTE {path} SHA256 {sha256(path)}", flush=True)
print(json.dumps({"status": "passed", "methods": len(category_methods),
                  "authenticated_prediction_files": len(authenticated_files),
                  "category_counts": category_distribution, "choice_counts": choice_distribution,
                  "all_existing_heldout_metrics_reproduced": True,
                  "independent_scorer_agrees": True,
                  "neural_inference_performed": False,
                  "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "compute_node": os.environ["SLURMD_NODENAME"]}, indent=2), flush=True)
