"""Research metrics with explicit cohorts and dependence-aware uncertainty."""
from __future__ import annotations

from collections import defaultdict
import math
import random

from .aggregate import calibration


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_interval(rows, field="correct", *, samples=1000, seed=42):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups[row["group_id"]].append(float(value))
    if not groups:
        return {"estimate": None, "low": None, "high": None, "groups": 0, "resamples": 0}
    clusters = [(sum(values), len(values)) for values in groups.values()]
    estimate = sum(value for value, size in clusters) / sum(size for value, size in clusters)
    if len(clusters) < 2:
        return {"estimate": estimate, "low": None, "high": None, "groups": 1, "resamples": 0}
    rng, draws = random.Random(seed), []
    for _ in range(samples):
        selected = [rng.choice(clusters) for _ in clusters]
        draws.append(sum(value for value, size in selected) / sum(size for value, size in selected))
    return {"estimate": estimate, "low": quantile(draws, .025), "high": quantile(draws, .975),
            "groups": len(clusters), "resamples": samples,
            "estimand": "row-weighted mean; whole correlated scenario/workflow clusters resampled"}


def correctness_auc(rows):
    values = [row for row in rows if row.get("confidence") is not None]
    positive = [row["confidence"] for row in values if row["correct"]]
    negative = [row["confidence"] for row in values if not row["correct"]]
    if not positive or not negative:
        return None
    return sum(float(p > n) + .5 * float(p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def ece_interval(rows, *, samples=1000, seed=42, bins=10):
    """Resample cached whole-group reliability-bin sufficient statistics."""
    groups = defaultdict(lambda: [[0., 0., 0.] for _ in range(bins)])
    for row in rows:
        confidence = row.get("confidence")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            continue
        index = min(int(confidence * bins), bins - 1)
        bucket = groups[row["group_id"]][index]
        bucket[0] += 1
        bucket[1] += confidence
        bucket[2] += bool(row["correct"])
    clusters = list(groups.values())
    def value(selected):
        counts, difference = 0., 0.
        for index in range(bins):
            counts += sum(cluster[index][0] for cluster in selected)
            difference += abs(sum(cluster[index][2] - cluster[index][1] for cluster in selected))
        return difference / counts if counts else None
    estimate = value(clusters)
    if len(clusters) < 2:
        return {"estimate": estimate, "low": None, "high": None,
                "groups": len(clusters), "resamples": 0}
    rng = random.Random(seed)
    draws = [value([rng.choice(clusters) for _ in clusters]) for _ in range(samples)]
    return {"estimate": estimate, "low": quantile(draws, .025), "high": quantile(draws, .975),
            "groups": len(clusters), "resamples": samples,
            "estimand": "fixed-width ECE among available confidences; whole confidence-bearing scenario/workflow groups resampled"}


def selective_curve(rows):
    # A threshold includes every tied prediction. Splitting a tied-confidence
    # group would make the curve depend on an arbitrary question ordering.
    ordered = sorted((row for row in rows if row.get("confidence") is not None),
                     key=lambda row: -row["confidence"])
    points, successes = [], 0
    for index, row in enumerate(ordered):
        successes += bool(row["correct"])
        if index + 1 == len(ordered) or ordered[index + 1]["confidence"] != row["confidence"]:
            points.append({"threshold": row["confidence"], "accepted": index + 1,
                           "coverage": (index + 1) / len(rows), "accuracy": successes / (index + 1)})
    return points


def summarize_decisions(rows, *, bootstrap=1000, seed=42):
    rows = list(rows)
    if not rows:
        return {"n": 0, "accuracy": None, "attempt_coverage": None, "response_coverage": None}
    by_label = defaultdict(list)
    for row in rows:
        if row.get("semantic_label") is not None:
            by_label[row["semantic_label"]].append(row["correct"])
    confident = [row for row in rows if row.get("confidence") is not None]
    brier_rows = [row for row in rows if row.get("brier") is not None]
    curve = selective_curve(rows)
    citation_eligible = [row for row in rows if row.get("expected_cause_ids")]
    citations = [row for row in citation_eligible if row.get("causes_cited")]
    lexical = [row for row in citation_eligible if row.get("expert_mention_evidence_available")]
    information = [row for row in rows if type(row.get("expected_information_request")) is bool]
    true_requests = sum(bool(row.get("information_request")) and row["expected_information_request"] for row in information)
    requests = sum(bool(row.get("information_request")) for row in information)
    necessary = sum(row["expected_information_request"] for row in information)
    risky = [row for row in confident if row.get("authored_risk") not in
             {None, "unspecified", "reversible_operation", "read_only"}]
    latencies = [row["latency_s"] for row in rows if row.get("latency_s") is not None]
    result = {
        "n": len(rows), "unique_tasks": len({row["task_id"] for row in rows}),
        "groups": len({row["group_id"] for row in rows}),
        "accuracy": mean(row["correct"] for row in rows),
        "accuracy_ci95": cluster_interval(rows, samples=bootstrap, seed=seed),
        "balanced_accuracy": mean(mean(values) for values in by_label.values()),
        "balanced_accuracy_semantic_classes": len(by_label),
        "attempt_coverage": mean(row.get("attempted", False) for row in rows),
        "response_coverage": mean(row.get("valid", False) for row in rows),
        "missing": sum(row.get("status") == "missing" for row in rows),
        "model_output_invalid": sum(row.get("status") == "model_output_invalid" for row in rows),
        "infrastructure_errors": sum(row.get("status") == "infrastructure_error" for row in rows),
        "calibration": calibration(rows, denominator=len(rows)),
        "ece_ci95": ece_interval(rows, samples=bootstrap, seed=seed),
        "correctness_auroc": correctness_auc(rows),
        "auroc_available_classes": sorted({bool(row["correct"]) for row in confident}),
        "selective_curve": curve,
        "exploratory_coverage_at_precision": {str(target): max((point["coverage"] for point in curve
                                                                if point["accuracy"] >= target), default=0)
                                                for target in (.95, .99)},
        "multiclass_brier": mean(row["brier"] for row in brier_rows),
        "multiclass_brier_n": len(brier_rows),
        "multiclass_brier_ci95": cluster_interval(rows, field="brier", samples=bootstrap, seed=seed),
        "brier_skill_uniform_menu": 1 - mean(row["brier"] for row in brier_rows) /
            mean(row["uniform_brier"] for row in brier_rows) if brier_rows else None,
        "information_request_precision": true_requests / requests if requests else None,
        "information_request_recall": true_requests / necessary if necessary else None,
        "information_request_eligible_n": len(information),
        "cause_citation_eligible_n": len(citation_eligible), "cause_citation_reported_n": len(citations),
        "cause_citation_coverage": len(citations) / len(citation_eligible) if citation_eligible else None,
        "cause_citation_exact_match": mean(set(row["causes_cited"]) == set(row["expected_cause_ids"]) for row in citations),
        "cause_citation_any_match": mean(bool(set(row["causes_cited"]) & set(row["expected_cause_ids"])) for row in citations),
        "cause_citation_precision": mean(len(set(row["causes_cited"]) & set(row["expected_cause_ids"])) / len(set(row["causes_cited"])) for row in citations),
        "cause_citation_recall": mean(len(set(row["causes_cited"]) & set(row["expected_cause_ids"])) / len(set(row["expected_cause_ids"])) for row in citations),
        "expert_expected_identifier_mention_recall": mean(len(row["expected_cause_identifier_mentions"]) / len(row["expected_cause_ids"]) for row in lexical),
        "expert_expected_identifier_mention_any": mean(bool(row["expected_cause_identifier_mentions"]) for row in lexical),
        "expert_identifier_mention_n": len(lexical),
        "expert_identifier_mention_note": "Expected public-ID mention recall/hit only, extracted from unverified expert checks; not citation precision, exact-match faithfulness or causal reasoning validation.",
        "authored_risk_confident_error_rate": mean(not row["correct"] and row["confidence"] >= .9 for row in risky),
        "authored_risk_confidence_n": len(risky),
        "shortlist_coverage": mean(row.get("shortlist_covers_gold") for row in rows),
        "shortlist_coverage_n": sum(row.get("shortlist_covers_gold") is not None for row in rows),
        "latency_mean_s": mean(latencies), "latency_p50_s": quantile(latencies, .5),
        "latency_p95_s": quantile(latencies, .95), "latency_observed_n": len(latencies),
        "observed_wall_s": sum(latencies),
        "local_api_cost_usd": 0.0,
        "known_input_tokens": sum(row.get("input_tokens") or 0 for row in rows),
        "known_output_tokens": sum(row.get("output_tokens") or 0 for row in rows),
        "input_token_usage_coverage": mean(row.get("input_tokens") is not None for row in rows),
        "output_token_usage_coverage": mean(row.get("output_tokens") is not None for row in rows),
    }
    return result


def validation_thresholds(rows, targets=(.95, .99)):
    validation = [row for row in rows if row.get("split") == "validation"]
    heldout = [row for row in rows if str(row.get("split", "")).startswith("test")]
    curve = selective_curve(validation)
    report = {}
    for target in targets:
        candidates = [point for point in curve if point["accuracy"] >= target]
        chosen = max(candidates, key=lambda point: point["coverage"]) if candidates else None
        accepted = [row for row in heldout if chosen and row.get("confidence") is not None
                    and row["confidence"] >= chosen["threshold"]]
        report[str(target)] = {"threshold": chosen["threshold"] if chosen else None,
                               "validation_n": len(validation),
                               "validation_accepted": chosen["accepted"] if chosen else 0,
                               "heldout_accepted": len(accepted), "heldout_n": len(heldout),
                               "heldout_coverage": len(accepted) / len(heldout) if heldout else None,
                               "heldout_precision": mean(row["correct"] for row in accepted),
                               "note": "Threshold fitted on validation only; empirical precision is not a deployment guarantee."}
    return report


def paired_permutation(left, right, *, samples=10000, seed=42):
    """Swap methods within whole correlated groups, not individual variants."""
    left_index = {row["task_id"]: row for row in left}
    right_index = {row["task_id"]: row for row in right}
    common = sorted(set(left_index) & set(right_index))
    differences = defaultdict(list)
    for key in common:
        first, second = left_index[key], right_index[key]
        if first["group_id"] != second["group_id"]:
            raise ValueError("Paired task has inconsistent correlation groups")
        differences[first["group_id"]].append(float(first["correct"]) - float(second["correct"]))
    values = [mean(group) for group in differences.values()]
    if not values:
        return {"p_value": None, "paired_tasks": 0, "groups": 0}
    observed = abs(sum(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        extreme += abs(sum(value if rng.getrandbits(1) else -value for value in values)) >= observed - 1e-12
    return {"difference_group_macro": mean(values), "p_value": (extreme + 1) / (samples + 1),
            "paired_tasks": len(common), "groups": len(values), "permutations": samples,
            "estimand": "equal-weight scenario/workflow group paired accuracy difference"}


def holm_adjust(comparisons):
    eligible = sorted((index for index, row in enumerate(comparisons) if row.get("p_value") is not None),
                      key=lambda index: comparisons[index]["p_value"])
    previous = 0.
    for rank, index in enumerate(eligible):
        previous = min(1., max(previous, comparisons[index]["p_value"] * (len(eligible) - rank)))
        comparisons[index]["holm_p_value"] = previous
    return comparisons
