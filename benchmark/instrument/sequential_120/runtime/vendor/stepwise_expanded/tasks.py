"""Private expanded cases; never pass these dictionaries to a decision model."""
from __future__ import annotations

import copy
from collections import Counter

from .new_workflows import build_new_workflows
from .variants import build_variants


def build_tasks(seed=42):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tasks = copy.deepcopy(build_variants(seed=seed) + build_new_workflows(seed=seed))
    if len(tasks) != 120 or len({case["task_id"] for case in tasks}) != len(tasks):
        raise ValueError("Expected exactly 120 distinct expanded cases")
    for case in tasks:
        metadata = case["expansion"]
        if metadata["template_id"] != case["workflow_id"]:
            raise ValueError("Template and workflow identity must agree")
        # Group every parameter variant of a template together for any future
        # holdout. There are deliberately no automatic train/test assignments.
        metadata["split_group"] = metadata["template_id"]
        metadata.setdefault("pair_id", None)
        metadata["declared_checkpoint_count"] = len(case["checkpoints"]) + 1
        metadata["difficulty_is_empirically_validated"] = False
        case["budget"]["max_decisions"] = max(32, len(case["checkpoints"]) * 3 + 8)
    if len({case["workflow_id"] for case in tasks}) != 14:
        raise ValueError("Expected fourteen workflow templates")
    if len({case["instrument_id"] for case in tasks}) != 8:
        raise ValueError("Expected all eight supported native instruments")
    return tasks


def inventory(tasks):
    def counts(key):
        return dict(sorted(Counter(case[key] for case in tasks).items()))

    pairs = {}
    for case in tasks:
        pair_id = case["expansion"].get("pair_id")
        if pair_id:
            pairs.setdefault(pair_id, []).append(case)
    return {
        "cases": len(tasks),
        "workflow_templates": len({case["workflow_id"] for case in tasks}),
        "instruments": len({case["instrument_id"] for case in tasks}),
        "matched_pairs": len(pairs),
        "workflow_counts": counts("workflow_id"),
        "instrument_counts": counts("instrument_id"),
        "condition_counts": dict(sorted(Counter(case["expansion"]["condition"] for case in tasks).items())),
        "expected_episodes_both_modes": len(tasks) * 2,
        "independent_expert_review_completed": False,
        "empirical_difficulty_order_established": False,
        "train_test_split_assigned": False,
        "future_split_policy": "Keep all parameter variants and matched conditions of each workflow template together using expansion.split_group.",
        "provenance": "Authored synthetic workflows and parameter variations using ChemLab public instrument contracts and executable native runtimes. These are not pre-existing ChemLab benchmark questions or recorded laboratory incidents.",
    }
