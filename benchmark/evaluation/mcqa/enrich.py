"""Enrich the three frozen MCQA exports without model inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
    raise RuntimeError("A Slurm compute allocation is required")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir', type=Path, required=True,
                    help='Schema1.0 overall and group exports; always read-only')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True,
                    help='New directory for immutable before copies and accounting evidence')
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--taxonomy', type=Path, required=True)
parser.add_argument('--source-root', type=Path)
parser.add_argument('--tariffs', type=Path, required=True,
                    help='Captured partition tariffs, used only for explicitly labeled estimates')
args = parser.parse_args()
INPUT, OUTPUT, RUN = (path.resolve() for path in (args.input_dir, args.output_dir, args.work_dir))
if INPUT == OUTPUT or RUN == INPUT or RUN == OUTPUT:
    raise ValueError('Input, output and scratch directories must be distinct')
OUTPUT.mkdir(parents=True, exist_ok=True)
RUN.mkdir(parents=True, exist_ok=False)
NAMES = ("mcqa-results.json", "mcqa-results-by-category.json", "mcqa-results-by-option-count.json")
if any((OUTPUT / name).exists() for name in NAMES):
    raise FileExistsError('Refusing to replace a previously published output')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(actual, expected, label):
    if type(actual) in (int, float) and type(expected) in (int, float):
        valid = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        valid = actual == expected
    if not valid:
        raise ValueError(f"Mismatch {label}: {actual!r} != {expected!r}")


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + chr(10),
                    encoding="utf-8")


before_bytes = {name: (INPUT / name).read_bytes() for name in NAMES}
documents = {name: json.loads(data) for name, data in before_bytes.items()}
original = documents[NAMES[0]]
for name, data in before_bytes.items():
    (RUN / (name + ".before")).write_bytes(data)

snapshot_path = args.snapshot.resolve()
snapshot = json.loads(snapshot_path.read_text())
source = args.source_root.resolve() if args.source_root else Path(snapshot["source"])
bundle = source / "instrument"
taxonomy_path = args.taxonomy.resolve()
same(original["snapshot_generated_utc"], snapshot["generated_utc"], "snapshot identity")
same(sha256(bundle / "manifest.json"), snapshot["bundle_sha256"], "frozen bundle")
same(sha256(taxonomy_path), snapshot["taxonomy_sha256"], "frozen taxonomy")
for filename, expected in snapshot["report_source_sha256"].items():
    same(sha256(source / "comparison" / filename), expected, "frozen scorer " + filename)

sys.path.insert(0, str(source))
from comparison.study_report import mcqa_rows
from comparison.study_metrics import calibration, correctness_auc, selective_curve, mean, quantile
from comparison.taxonomy import load_jsonl
from instrument.eval.mcqa import score_predictions
from metrics import summarize_extra
from accounting import collect_accounting

canonical = dict(calibration=calibration, correctness_auc=correctness_auc,
                 selective_curve=selective_curve, mean=mean, quantile=quantile)


manifest = json.loads((bundle / "mcqa_388" / "manifest.json").read_text())
questions_path = bundle / "mcqa_388" / "questions.jsonl.gz"
answers_path = bundle / "mcqa_388" / "answers.jsonl"
same(sha256(questions_path), manifest["question_archive_sha256"], "question archive")
same(sha256(answers_path), manifest["answers_sha256"], "answers")
taxonomy = json.loads(taxonomy_path.read_text())
tasks = load_jsonl(questions_path)
task_index = {task["task_id"]: task for task in tasks}
answer_records = {answer["task_id"]: answer for answer in load_jsonl(answers_path)}
answers = {task_id: answer["correct_option_ids"] for task_id, answer in answer_records.items()}
same(len(tasks), 388, "task count")
same(len(task_index), 388, "unique task IDs")
same(set(task_index), set(answers), "answer IDs")
methods = {method["experiment"]: method for method in original["methods"]}
same(len(methods), 34, "method count")
same(set(methods), {method["experiment"] for method in snapshot["methods"]}, "experiment identities")
for method in snapshot["methods"]:
    for key, value in method.items():
        if key not in {"teacher_forced_attempts", "free_run_attempts"}:
            same(methods[method["experiment"]][key], value, "original metric " + method["experiment"] + "/" + key)
for name in NAMES[1:]:
    same(documents[name]["snapshot_generated_utc"], snapshot["generated_utc"], name + " source snapshot")
    same(documents[name]["provenance"]["source_snapshot_sha256"], sha256(snapshot_path), name + " snapshot hash")
    same(documents[name]["provenance"]["overall_results_sha256"], sha256(INPUT / NAMES[0]), name + " overall export hash")
    same({method["experiment"] for method in documents[name]["methods"]}, set(methods), name + " methods")

predictions = {name: {} for name in methods}
source_jobs = {}
prediction_files = []
for recorded_source in snapshot["sources"]:
    files = {name: expected for name, expected in recorded_source.get("files", {}).items()
             if name.endswith("/mcqa/predictions.jsonl")}
    if not files:
        continue
    run = Path(recorded_source["path"])
    same(sha256(run / "configuration.json"), recorded_source["configuration_sha256"], "configuration " + str(run))
    configuration = json.loads((run / "configuration.json").read_text())
    same(configuration["track"], "mcqa", "dedicated MCQA allocation")
    for relative, expected in files.items():
        name = Path(relative).parts[0]
        if name not in methods:
            raise ValueError("Unknown experiment " + name)
        same(configuration["selected_experiments"], [name], "dedicated experiment allocation")
        path = run / relative
        same(sha256(path), expected, "prediction file " + str(path))
        job_id = str(recorded_source["slurm_job_id"])
        if not job_id.isdigit() or job_id in source_jobs:
            raise ValueError("Nonunique or invalid raw MCQA job ID: " + job_id)
        source_jobs[job_id] = {
            "experiment": name, "track": "mcqa", "source_path": str(run),
            "compute_node": recorded_source.get("compute_node"),
        }
        prediction_files.append({"path": str(path), "sha256": expected, "experiment": name, "raw_slurm_job_id": job_id})
        for record in load_jsonl(path):
            task_id = record["task_id"]
            if task_id not in task_index or task_id in predictions[name]:
                raise ValueError("Unknown or duplicated prediction task for " + name)
            predictions[name][task_id] = record
same(len(prediction_files), 272, "authenticated prediction files")


def legacy_summary(rows):
    n = len(rows)
    attempted = sum(row["attempted"] for row in rows)
    valid = sum(row["valid"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    probability_rows = [row for row in rows if row.get("brier") is not None]
    confidence_rows = [row for row in rows if row.get("confidence") is not None]
    return dict(
        n=n, correct=correct, accuracy=correct / n if n else None,
        attempted=attempted, attempt_coverage=attempted / n if n else None,
        valid=valid, response_coverage=valid / n if n else None,
        invalid=attempted - valid, missing=n - attempted,
        infrastructure_errors=sum(row["status"] == "infrastructure_error" for row in rows),
        model_output_invalid=sum(row["status"] == "model_output_invalid" for row in rows),
        status_counts=dict(sorted(Counter(row["status"] for row in rows).items())),
        uniform_random_accuracy=mean(len(answers[row["task_id"]]) / len(task_index[row["task_id"]]["options"]) for row in rows),
        ece=calibration(rows, denominator=n)["ece"] if n else None,
        confidence_n=len(confidence_rows),
        multiclass_brier=mean(row["brier"] for row in probability_rows),
        brier_n=len(probability_rows),
    )


def check_prior_group(prior, rows, label):
    current = legacy_summary(rows)
    for key, expected in current.items():
        same(prior[key], expected, label + "/" + key)


rows_by_experiment = {}
extras_by_experiment = {}
all_resources = {}
for name, method in methods.items():
    selected = tasks if method["reporting_scope"] != "T1_reference" else [task for task in tasks if task.get("tier") == "T1"]
    selected_ids = {task["task_id"] for task in selected}
    if set(predictions[name]) - selected_ids:
        raise ValueError("Predictions outside scoped cohort " + name)
    rows = mcqa_rows(selected, answers, predictions[name], taxonomy)
    heldout = [row for row in rows if str(row.get("split", "")).startswith("test")]
    normalized = []
    for task_id, prediction in predictions[name].items():
        backend = prediction.get("backend_result", prediction)
        normalized.append(dict(task_id=task_id, status=prediction.get("status", "ok"),
                               selected_option_id=prediction.get("selected_option_id", backend.get("selected_option_id"))))
    portable = score_predictions(selected, {key: answer_records[key] for key in selected_ids}, normalized, "heldout")
    same(portable["accuracy"], method["heldout_accuracy"], name + " independent accuracy")
    same(portable["n"], method["heldout_n"], name + " independent denominator")
    same(legacy_summary(heldout)["ece"], method["heldout_ece"], name + " previous ECE")
    same(legacy_summary(heldout)["multiclass_brier"], method["heldout_multiclass_brier"], name + " previous multiclass Brier")
    rows_by_experiment[name] = {"all_questions": rows, "heldout": heldout}
    extras = summarize_extra(heldout, name, canonical)
    extras_by_experiment[name] = extras
    all_resources[name] = summarize_extra(rows, name, canonical)["resources"]
    for key, value in extras.items():
        method["heldout_" + key] = value
    method["all_questions_resources"] = all_resources[name]
    print("Verified prior scores and added metrics: " + name, flush=True)

print("Collecting Slurm allocation accounting for 272 dedicated MCQA shard jobs.", flush=True)
global_accounting, accounting_by_experiment = collect_accounting(source_jobs, RUN, tariff_path=args.tariffs)


def apply_full_allocation(resources, name):
    allocation = accounting_by_experiment.get(name, {})
    resources["gpu_allocation_hours"] = allocation.get("gpu_allocation_hours")
    resources["gpu_cost_chf_estimate"] = allocation.get("gpu_cost_chf_estimate")
    resources["allocation_unavailable_reason"] = (
        None if allocation.get("cost_estimate_complete") else
        "Slurm allocation or CHF tariff coverage is incomplete; inspect compute_accounting coverage and errors."
    )
    resources["allocation_scope"] = "Full declared MCQA cohort, including setup/loading/overhead; not the held-out or subgroup allocation."
    resources["gpu_allocation_by_partition"] = allocation.get("by_partition", {})
    resources["actual_invoiced_cost_chf"] = None
    resources["cost_is_estimate"] = True


for name, method in methods.items():
    method["compute_accounting"] = accounting_by_experiment.get(name, {})
    apply_full_allocation(method["all_questions_resources"], name)

category_doc = documents[NAMES[1]]
choice_doc = documents[NAMES[2]]
for document, axis in ((category_doc, "category"), (choice_doc, "choice_count")):
    for method in document["methods"]:
        name = method["experiment"]
        for scope in ("all_questions", "heldout"):
            rows = rows_by_experiment[name][scope]
            block = method[scope]
            check_prior_group(block["overall"], rows, name + "/" + scope)
            extra = summarize_extra(rows, name, canonical)
            if scope == "all_questions":
                apply_full_allocation(extra["resources"], name)
            block["overall"].update(extra)
            if scope == "heldout":
                same(extra, extras_by_experiment[name], name + " consistent held-out extension")
            for group, summary in block["groups"].items():
                subset = [row for row in rows if (
                    row["legacy_level"] == group if axis == "category"
                    else str(len(task_index[row["task_id"]]["options"])) == group
                )]
                check_prior_group(summary, subset, name + "/" + scope + "/" + group)
                summary.update(summarize_extra(subset, name, canonical))
            for field in ("n", "correct", "attempted", "valid", "invalid", "missing"):
                same(sum(value[field] for value in block["groups"].values()), block["overall"][field],
                     name + "/" + axis + "/" + scope + " reconciled " + field)
            for token in ("input", "output"):
                known = [value["resources"]["tokens"][token]["known_total"] for value in block["groups"].values()
                         if value["resources"]["tokens"][token]["known_total"] is not None]
                same(sum(known) if known else None, block["overall"]["resources"]["tokens"][token]["known_total"],
                     name + "/" + axis + "/" + scope + " token sum " + token)
            group_latency = [value["resources"]["latency"]["total_s"] for value in block["groups"].values()
                             if value["resources"]["latency"]["total_s"] is not None]
            same(sum(group_latency) if group_latency else None, block["overall"]["resources"]["latency"]["total_s"],
                 name + "/" + axis + "/" + scope + " observed latency sum")
        method["compute_accounting"] = accounting_by_experiment.get(name, {})

augmentation = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "compute_node": os.environ["SLURMD_NODENAME"],
    "source_snapshot_sha256": sha256(snapshot_path),
    "source_snapshot_generated_utc": snapshot["generated_utc"],
    "previous_export_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in before_bytes.items()},
    "scripts_sha256": {path.name: sha256(path) for path in (Path(__file__), HERE / "metrics.py", HERE / "accounting.py")},
    "authenticated_prediction_files": len(prediction_files),
    "neural_inference_performed": False,
    "hosted_calls": 0,
    "new_confidence_intervals_or_significance_tests": False,
}
definitions = {
    "binary_brier_selected_correctness": "Mean (selected-answer confidence - correctness)^2 over available valid confidence rows.",
    "confidence_coverage": "Available confidence rows / all cohort questions.",
    "correctness_auroc": "Probability that a correct answer receives higher confidence than an incorrect answer, with half credit for ties; null without both classes.",
    "selective_curve": "One point per confidence threshold including all ties; accuracy=correct accepted/accepted, coverage=accepted/full subgroup size. Descriptive held-out curve, not a validation-fitted deployment guarantee.",
    "latency": "Observed backend/pipeline wall time; excludes explicit zero reference placeholders. Direct timing normally excludes model loading; first expert-to-native calls may include lazy chooser initialization.",
    "tokens": "Known top-level input/output token sums and per-field coverage. Sampling totals already include all draws; expert/chooser totals already include both stages. Do not double-add nested records.",
    "feature_usage": "Feature methods record cold-encoding tokens only and include cache/CPU work in wall timing. Token units differ between encoders; zero generated output tokens is valid.",
    "api_cost_usd": "Zero hosted API spend for these local configurations; not a measured GPU cost.",
    "gpu_allocation_hours": "Allocated GPU units multiplied by Slurm elapsed hours for retained dedicated MCQA shard jobs. A MIG unit is a slice, not a full-GPU equivalent.",
    "gpu_cost_chf_estimate": "Full retained MCQA allocation hours multiplied by installed SCITAS partition tariff. Estimate only; excludes preparation/fitting/downloads/other tracks/unretained attempts.",
    "subgroup_allocation": "Held-out and task/choice subgroups share whole-job allocations; GPU allocation and monetary cost remain null for these subgroups. No latency-based prorating is invented.",
    "actual_invoiced_cost_chf": "Unavailable; tariff estimates are not invoices.",
}
interpretation = [
    "Additional metrics are recomputed from the same authenticated saved predictions; pretrained models were not rerun.",
    "Confidence, latency and token availability are reported explicitly; missing observations are not replaced with zero.",
    "Selected-answer Brier uses confidence only; multiclass Brier still requires a genuine full probability distribution.",
    "Reference uniform/majority zero latency values are placeholders and excluded from measured timing.",
    "Allocation costs cover complete retained MCQA shard allocations, not isolated held-out questions or subgroups; they are estimates, not final billed charges.",
]
for filename, document in documents.items():
    document["schema_version"] = "1.1"
    document["metrics_augmentation"] = augmentation
    document["compute_accounting"] = global_accounting
    document.setdefault("metric_definitions", {}).update(definitions)
    document["interpretation"].extend(note for note in interpretation if note not in document["interpretation"])
    document["verification"] = {
        **document.get("verification", {}),
        "all_existing_scores_and_denominators_preserved": True,
        "all_272_prediction_hashes_match_snapshot": True,
        "independent_portable_accuracy_scorer_agrees": True,
        "group_counts_usage_and_observed_timing_reconcile": True,
        "reference_timing_placeholders_excluded": True,
    }
original["provenance"]["export"] = "Original score fields preserved and verified; added confidence/resource metrics recomputed from authenticated saved predictions without new model inference."

staged = {name: RUN / (name + ".new") for name in NAMES}
write_json(staged[NAMES[0]], original)
new_overall_hash = sha256(staged[NAMES[0]])
for name in NAMES[1:]:
    documents[name]["provenance"]["overall_results_sha256"] = new_overall_hash
    documents[name]["provenance"]["metrics_augmentation_slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    write_json(staged[name], documents[name])

for name in NAMES:
    same(json.loads(staged[name].read_text()), documents[name], name + " JSON round trip")
    same((INPUT / name).read_bytes(), before_bytes[name], name + " unchanged during processing")
for name in NAMES:
    os.replace(staged[name], OUTPUT / name)
for name in NAMES[1:]:
    document = json.loads((OUTPUT / name).read_text())
    same(document["provenance"]["overall_results_sha256"], sha256(OUTPUT / NAMES[0]), name + " published overall link")
summary = {
    "status": "passed",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "compute_node": os.environ["SLURMD_NODENAME"],
    "method_count": len(methods),
    "binary_brier_methods": sum(method.get("heldout_binary_brier_selected_correctness") is not None for method in methods.values()),
    "auroc_methods": sum(method.get("heldout_correctness_auroc") is not None for method in methods.values()),
    "selective_curve_methods": sum(bool(method.get("heldout_selective_curve")) for method in methods.values()),
    "measured_latency_methods": sum(method["heldout_resources"]["latency"]["observed_n"] > 0 for method in methods.values()),
    "authenticated_prediction_files": len(prediction_files),
    "all_existing_scores_preserved": True,
    "neural_inference_performed": False,
    "compute_accounting": {key: value for key, value in global_accounting.items()
                           if key not in {"jobs", "raw_job_records", "accounting_source"}},
    "files": {name: {"sha256": sha256(OUTPUT / name), "bytes": (OUTPUT / name).stat().st_size} for name in NAMES},
}
print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
