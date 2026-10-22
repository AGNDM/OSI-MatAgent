"""Calibration with explicit coverage; never manufacture unavailable confidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def calibration(rows, *, bins=10, denominator=None):
    """Rows have confidence/correct; missing confidence remains uncovered.

    Multiclass Brier requires a genuine complete probability vector and exactly
    one gold label. Verbalized confidence supports ECE and binary Brier of the
    selected answer's correctness, not a fabricated multiclass distribution.
    """
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be positive")
    rows = list(rows)
    denominator = len(rows) if denominator is None else denominator
    if type(denominator) is not int or denominator < len(rows):
        raise ValueError("calibration denominator must cover every observed row")
    valid = [row for row in rows if type(row.get("confidence")) in (float, int)
             and math.isfinite(row["confidence"]) and 0 <= row["confidence"] <= 1
             and type(row.get("correct")) is bool]
    buckets = []
    for index in range(bins):
        bucket = [row for row in valid if min(int(row["confidence"] * bins), bins - 1) == index]
        if not bucket:
            continue
        buckets.append({"lower": index / bins, "upper": (index + 1) / bins,
                        "n": len(bucket), "accuracy": sum(row["correct"] for row in bucket) / len(bucket),
                        "mean_confidence": sum(row["confidence"] for row in bucket) / len(bucket)})
    briers = []
    for row in valid:
        probabilities, gold = row.get("probabilities"), row.get("gold_labels")
        if not isinstance(probabilities, dict) or not isinstance(gold, list) or len(gold) != 1:
            continue
        if gold[0] not in probabilities:
            continue
        if (not probabilities or not all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1
                                        for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5, rel_tol=1e-5)):
            continue
        briers.append(sum((value - float(label == gold[0])) ** 2 for label, value in probabilities.items()))
    n = len(valid)
    return {"cohort_size": denominator, "confidence_rows": n,
            "confidence_coverage": n / denominator if denominator else None,
            "ece": sum(bucket["n"] * abs(bucket["accuracy"] - bucket["mean_confidence"])
                       for bucket in buckets) / n if n else None,
            "binary_brier_selected_correctness": sum((row["confidence"] - row["correct"]) ** 2
                                                      for row in valid) / n if n else None,
            "multiclass_brier": sum(briers) / len(briers) if briers else None,
            "multiclass_brier_rows": len(briers), "bins": buckets,
            "definitions": {"ece": "fixed-width bins over available selected-answer confidence; see coverage",
                            "binary_brier_selected_correctness": "mean squared error of selected-answer confidence versus its correctness",
                            "multiclass_brier": "sum of squared full-vector errors, single-gold rows only; unsupported vectors omitted"}}


def mcqa_calibration(tasks, answers, predictions):
    index = {row["task_id"]: row for row in predictions}
    rows = []
    for task in tasks:
        prediction = index.get(task["task_id"], {})
        labels = [option["option_id"] for option in task["options"]]
        if prediction.get("status") != "ok" or prediction.get("selected_option_id") not in labels:
            continue
        selected = prediction["selected_option_id"]
        correct = answers[task["task_id"]]["correct_option_ids"]
        rows.append({"confidence": prediction.get("confidence"), "correct": selected in correct,
                     "probabilities": prediction.get("probabilities"), "gold_labels": correct})
    return calibration(rows, denominator=len(tasks))


def sequential_calibration(episodes):
    """Do not misrepresent hierarchy-node confidence as action confidence."""
    summaries = {}
    for mode in sorted({episode["mode"] for episode in episodes}):
        current = [episode for episode in episodes if episode["mode"] == mode]
        records = [record for episode in current if episode.get("status") == "ok"
                   for record in episode.get("decision_records", [])]
        events = [event for record in records for event in record.get("usage", {}).get("selection_events", [])]
        summaries[mode] = {"observed_actions": len(records), "observed_hierarchy_selections": len(events),
                           "selections_with_probability_vector": sum(event.get("label_probabilities") is not None for event in events),
                           "selections_with_confidence": sum(event.get("confidence") is not None for event in events),
                           "action_ece": None, "action_multiclass_brier": None,
                           "reason": "Command, argument, and page probabilities are conditional menu-node readouts. Native correctness grades complete actions; no unsupported action calibration is inferred."}
    return summaries


def flatten_report(report):
    rows = []
    for experiment in report.get("experiments", []):
        common = {"experiment": experiment["name"], "method": experiment.get("method"),
                  "backend": experiment.get("backend_kind"), "diagnostic_subset": report.get("diagnostic_subset")}
        for track, summary in experiment.get("tracks", {}).items():
            if track == "mcqa":
                rows.append({**common, "track": track, "mode": "static", "n": summary.get("n"),
                             "accuracy": summary.get("accuracy"), "coverage": summary.get("coverage"),
                             "ece": summary.get("calibration", {}).get("ece"),
                             "multiclass_brier": summary.get("calibration", {}).get("multiclass_brier")})
            elif track == "sequential":
                for mode, value in summary.get("by_mode", {}).items():
                    rows.append({**common, "track": track, "mode": mode, "n": value.get("n"),
                                 "accuracy": value.get("observed_control_accuracy_micro"),
                                 "coverage": value.get("coverage"),
                                 "goal_completion": value.get("cohort_goal_completion_rate")})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not __import__("os").environ.get("SLURMD_NODENAME") or not __import__("os").environ.get("SLURM_JOB_ID"):
        parser.error("Aggregation requires a Slurm compute-node allocation")
    report = json.loads(args.report.read_text())
    rows = flatten_report(report)
    fields = sorted({field for row in rows for field in row})
    if args.output.exists():
        parser.error("--output must be a new file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
