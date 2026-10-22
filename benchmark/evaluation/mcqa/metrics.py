"""Add descriptive confidence and resource metrics to canonical MCQA rows.

No model calls or statistical resampling occur here. The caller supplies frozen
scoring functions and authenticates the rows and their source artifacts.
"""

from __future__ import annotations

import math


def _finite_nonnegative(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
    )


def _experiment_name(experiment):
    if isinstance(experiment, str):
        return experiment
    return experiment.get("experiment", experiment.get("name", ""))


def _token_summary(rows, field):
    values = [row.get(field) for row in rows]
    known = [value for value in values if type(value) is int and value >= 0]
    n = len(rows)
    return {
        "known_total": sum(known) if known else None,
        "known_n": len(known),
        "unknown_n": n - len(known),
        "coverage": len(known) / n if n else None,
    }


def summarize_extra(rows, experiment, canonical):
    """Summarize one complete method cohort or subgroup without new inference.

    Confidence curves use the full cohort denominator and include every tie at
    a threshold. Resource totals use only top-level canonical row values, which
    already combine sampling draws and expert/chooser stages. Unknown usage is
    never replaced with zero. Wall latency is not active GPU time or a bill.
    """
    rows = list(rows)
    n = len(rows)
    # Canonical rows already sanitize confidence. Retain that boundary if a
    # caller supplies an invalid value, without changing the cohort denominator.
    confidence_rows = []
    scoring_rows = []
    for row in rows:
        value = row.get("confidence")
        if _finite_nonnegative(value) and value <= 1:
            confidence_rows.append(row)
        else:
            row = dict(row, confidence=None)
        scoring_rows.append(row)
    confident_n = len(confidence_rows)
    correct_n = sum(bool(row["correct"]) for row in confidence_rows)
    incorrect_n = confident_n - correct_n
    if not n:
        auroc_reason = "empty_cohort"
    elif not confident_n:
        auroc_reason = "no_confidence"
    elif not correct_n:
        auroc_reason = "no_correct_confident_answers"
    elif not incorrect_n:
        auroc_reason = "no_incorrect_confident_answers"
    else:
        auroc_reason = None
    calibration = canonical["calibration"](scoring_rows, denominator=n)

    recorded_latency = [
        row["latency_s"] for row in rows
        if _finite_nonnegative(row.get("latency_s"))
    ]
    name = _experiment_name(experiment)
    placeholder_reference = (
        name.startswith("reference-uniform")
        or name.startswith("reference-majority")
    )
    placeholder_n = (
        sum(value == 0 for value in recorded_latency)
        if placeholder_reference else 0
    )
    observed_latency = [
        value for value in recorded_latency
        if not (placeholder_reference and value == 0)
    ]
    complete_usage_n = sum(
        all(type(row.get(field)) is int and row[field] >= 0
            for field in ("input_tokens", "output_tokens"))
        for row in rows
    )
    recorded_attempt_n = sum(bool(row.get("attempted")) for row in rows)
    failed_attempt_n = sum(
        bool(row.get("attempted")) and not bool(row.get("valid"))
        for row in rows
    )
    resources = {
        "latency": {
            "recorded_n": len(recorded_latency),
            "recorded_sum_s": sum(recorded_latency) if recorded_latency else None,
            "placeholder_n": placeholder_n,
            "observed_n": len(observed_latency),
            "coverage": len(observed_latency) / n if n else None,
            "mean_s": canonical["mean"](observed_latency),
            "p50_s": canonical["quantile"](observed_latency, .5),
            "p95_s": canonical["quantile"](observed_latency, .95),
            "total_s": sum(observed_latency) if observed_latency else None,
        },
        "tokens": {
            "input": _token_summary(rows, "input_tokens"),
            "output": _token_summary(rows, "output_tokens"),
            "complete_usage_n": complete_usage_n,
            "complete_usage_coverage": complete_usage_n / n if n else None,
        },
        "recorded_attempt_n": recorded_attempt_n,
        "failed_attempt_n": failed_attempt_n,
        "api_cost_usd": 0.0 if n else None,
        "gpu_active_seconds": None,
        "gpu_allocation_hours": None,
        "gpu_cost_chf_estimate": None,
        "allocation_unavailable_reason": (
            "Question cohorts share Slurm jobs with other questions or tracks; "
            "per-cohort GPU allocation and monetary cost cannot be inferred "
            "from recorded inference latency."
            if n else "No questions in this cohort."
        ),
    }
    return {
        "binary_brier_selected_correctness": calibration[
            "binary_brier_selected_correctness"
        ],
        "confidence_coverage": calibration["confidence_coverage"],
        "correctness_auroc": (
            canonical["correctness_auc"](scoring_rows)
            if auroc_reason is None else None
        ),
        "auroc_confidence_n": confident_n,
        "auroc_correct_n": correct_n,
        "auroc_incorrect_n": incorrect_n,
        "auroc_unavailable_reason": auroc_reason,
        "selective_curve": canonical["selective_curve"](scoring_rows),
        "selective_unavailable_reason": (
            "empty_cohort" if not n else "no_confidence" if not confident_n else None
        ),
        "resources": resources,
    }
