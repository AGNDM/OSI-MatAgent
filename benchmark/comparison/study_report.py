"""Build an auditable local-study report from completed or partial shard outputs.

Run on a Slurm compute node. The output includes standalone HTML, JSON, CSV,
task-level research records, an evidence-backed taxonomy, and exportable plots.
Original answer keys, evaluator code and legacy level identifiers are unchanged.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics

from .study_metrics import (cluster_interval, holm_adjust, mean, paired_permutation,
                            summarize_decisions, validation_thresholds)
from .taxonomy import (MECHANISMS, build_taxonomy, load_jsonl, require_allocation, sha256)

VERSION = "local-instrument-study-report-v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n")


def _read_output(path, warnings):
    """A live writer's unfinished final line is visible as a warning, not data."""
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    warnings.append(f"Live incomplete final line: {path}:{number}")
                    return
                raise


def _compact_episode(episode):
    keep = {key: value for key, value in episode.items() if key not in {
        "history", "decision_records", "final_public_observation", "policy_identity"}}
    records = []
    for record in episode.get("decision_records", []):
        events = record.get("usage", {}).get("selection_events", [])
        records.append({key: record.get(key) for key in
                        ("decision_index", "checkpoint_id", "stage", "proposal_correct",
                         "auxiliary_query", "corrected", "policy_error", "invalid_model_reply")}
                       | {"hierarchy_selections": len(events),
                          "node_confidence_rows": sum(event.get("confidence") is not None for event in events),
                          "node_probability_rows": sum(event.get("label_probabilities") is not None for event in events),
                          "shortlist_evaluations": [event["shortlist_coverage"] for event in events
                                                   if "shortlist_coverage" in event]})
    keep["decision_records"] = records
    return keep


def discover_experiments(roots, bundle, *, include_diagnostics=False):
    """Merge disjoint shards only when their experiment specification matches."""
    expected_release = sha256(Path(bundle) / "manifest.json")
    experiments, sources, warnings, seen_configurations = {}, [], [], set()
    for root in roots:
        root = Path(root).resolve()
        candidates = [root] if root.name == "configuration.json" else sorted(root.rglob("configuration.json"))
        for path in candidates:
            path = path.resolve()
            if path in seen_configurations:
                continue
            seen_configurations.add(path)
            configuration = json.loads(path.read_text())
            resolved = configuration.get("resolved_configuration")
            if not isinstance(resolved, dict):
                continue  # Native per-episode configuration, not a study run.
            run = path.parent
            if configuration.get("limit") is not None and not include_diagnostics:
                warnings.append(f"Excluded diagnostic prefix: {run}")
                continue
            report_file = run / "report.json"
            if report_file.is_file() and json.loads(report_file.read_text()).get("preflight"):
                continue
            recorded_release = configuration.get("source_sha256", {}).get("instrument/manifest.json")
            if recorded_release != expected_release:
                warnings.append(f"Excluded incompatible or unauthenticated release: {run}")
                continue
            selected = configuration.get("selected_experiments", [row["name"] for row in resolved.get("experiments", [])])
            source = {"path": str(run), "configuration_sha256": sha256(path),
                      "comparison_source_inventory_sha256": configuration.get("source_inventory_sha256"),
                      "slurm_job_id": configuration.get("slurm_job_id"),
                      "compute_node": configuration.get("compute_node"),
                      "diagnostic_prefix": configuration.get("limit"), "files": {}}
            sources.append(source)
            for specification in resolved.get("experiments", []):
                if specification["name"] not in selected:
                    continue
                key = fingerprint(specification)
                entry = experiments.setdefault(key, {"specification": specification,
                    "name": specification["name"], "seed": specification.get("seed", 0),
                    "analysis_family": specification.get("analysis_family", specification["name"]),
                    "architecture": specification.get("architecture", "direct"),
                    "method": specification.get("method", "choice"),
                    "backend": specification.get("backend", {}), "identity": {}, "identities": [],
                    "comparison_source_signatures": set(),
                    "mcqa": {}, "sequential": {}, "requested_tracks": set(), "sources": []})
                entry["sources"].append(str(run))
                source_files = {key: value for key, value in configuration.get("source_sha256", {}).items()
                                if key.startswith("comparison/")}
                if source_files:
                    entry["comparison_source_signatures"].add(fingerprint(source_files))
                    if len(entry["comparison_source_signatures"]) > 1:
                        raise ValueError("Comparison source changed across shards of the same experiment")
                track = configuration.get("track", "both")
                requested = set(("mcqa", "sequential") if track == "both" else (track,))
                declared_tracks = specification.get("tracks", specification.get("evaluation_tracks"))
                if declared_tracks is not None:
                    requested &= set([declared_tracks] if isinstance(declared_tracks, str) else declared_tracks)
                entry["requested_tracks"].update(requested)
                experiment_dir = run / specification["name"]
                identity_path = experiment_dir / "identity.json"
                if identity_path.is_file():
                    entry["identities"].append(json.loads(identity_path.read_text()))
                    source["files"][str(identity_path.relative_to(run))] = sha256(identity_path)
                for track, filename in (("mcqa", "predictions.jsonl"), ("sequential", "episodes.jsonl")):
                    data = experiment_dir / track / filename
                    if not data.is_file():
                        continue
                    source["files"][str(data.relative_to(run))] = sha256(data)
                    for record in _read_output(data, warnings):
                        identity = record["task_id"] if track == "mcqa" else (record["task_id"], record["mode"])
                        if identity in entry[track]:
                            raise ValueError(f"Duplicate {track} task for the same experiment/seed: {identity!r}; select one run or disjoint shards")
                        if track == "sequential" and record.get("status") == "infrastructure_error":
                            step_dir = data.parent / "steps" / hashlib.sha256(record["task_id"].encode()).hexdigest()[:24] / record["mode"]
                            partial = []
                            for step in sorted(step_dir.glob("*.json")):
                                partial.append(json.loads(step.read_text()))
                                source["files"][str(step.relative_to(run))] = sha256(step)
                            record["partial_decisions_observed"] = len(partial)
                            record["partial_policy_wall_s"] = sum(step.get("policy_latency_s", 0) for step in partial)
                            record["partial_input_tokens_known"] = sum(step.get("usage", {}).get("input_tokens", 0) or 0 for step in partial)
                            record["partial_output_tokens_known"] = sum(step.get("usage", {}).get("output_tokens", 0) or 0 for step in partial)
                        entry[track][identity] = record if track == "mcqa" else _compact_episode(record)
    from .study import authenticate_identities
    for entry in experiments.values():
        entry["artifact_authentication"] = authenticate_identities(entry["identities"])
        if entry["identities"]:
            entry["identity"] = max(entry["identities"], key=lambda value: len(json.dumps(value, sort_keys=True)))
    return list(experiments.values()), sources, warnings


def apply_plan(experiments, plan):
    """Make declared but not-yet-started experiments visible in the report."""
    index = defaultdict(list)
    for entry in experiments:
        index[(entry["name"], entry["seed"])].append(entry)
    planned = set()
    for specification in plan.get("experiments", []):
        identity = (specification["name"], specification.get("seed", 0))
        if identity in planned:
            raise ValueError("Study plan repeats an experiment name and seed")
        planned.add(identity)
        existing = index.get(identity, [])
        if len(existing) > 1:
            raise ValueError(f"Study includes incompatible configurations for planned experiment {identity}")
        if existing:
            for field in ("analysis_family", "architecture", "method"):
                expected = specification.get(field, "direct" if field == "architecture" else "choice" if field == "method" else specification["name"])
                if existing[0][field] != expected:
                    raise ValueError(f"Study plan disagrees with recorded experiment {identity}: {field}")
            continue
        tracks = specification.get("tracks", specification.get("evaluation_tracks", ["mcqa", "sequential"]))
        if isinstance(tracks, str):
            tracks = [tracks]
        entry = {"specification": specification, "name": specification["name"],
                 "seed": specification.get("seed", 0),
                 "analysis_family": specification.get("analysis_family", specification["name"]),
                 "architecture": specification.get("architecture", "direct"),
                 "method": specification.get("method", "choice"), "backend": specification.get("backend", {}),
                 "identity": {}, "mcqa": {}, "sequential": {}, "requested_tracks": set(tracks), "sources": []}
        experiments.append(entry)
    return experiments


def discover_tool_tracks(roots):
    """Read method-7 evidence without merging information-dependent labels."""
    tracks, warnings, seen = [], [], set()
    for root in roots:
        root = Path(root).resolve()
        candidates = [root] if root.name == "report.json" else sorted(root.rglob("report.json"))
        for path in candidates:
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            result = json.loads(path.read_text())
            if result.get("protocol") != "native-readonly-conditional-label-comparison-v1":
                continue
            methods = result.get("methods", {})
            valid = all(row.get("static_answer_key_used_for_grading") is False and
                        row.get("checkpoint_replay_verified") is True for row in methods.values())
            if not methods or not valid:
                warnings.append(f"Excluded unverified information-dependent tool report: {path}")
                continue
            tracks.append({"path": str(path), "report_sha256": sha256(path), **result})
    return tracks, warnings


def _probabilities(result, labels):
    values = result.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(labels):
        return None
    if (not all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values.values())
            or not math.isclose(sum(values.values()), 1., rel_tol=1e-5, abs_tol=1e-5)):
        return None
    return values


def _cause_ids(oracle):
    result = set()
    for cause in oracle.get("causes") or []:
        if isinstance(cause, str):
            result.add(cause)
        elif isinstance(cause, dict) and cause.get("visibility") != "evaluator":
            for key in ("id", "state_id", "precondition_id", "slot_id", "code"):
                if cause.get(key):
                    result.add(str(cause[key]))
    return sorted(result)


def mcqa_rows(tasks, answers, predictions, taxonomy):
    labels_index = {row["task_id"]: row for row in taxonomy["labels"] if row["track"] == "mcqa"}
    metadata = taxonomy.get("mcqa_evaluator_metadata", {})
    rows = []
    if set(predictions) - {task["task_id"] for task in tasks}:
        raise ValueError("Prediction file includes unknown frozen MCQA IDs")
    for task in tasks:
        task_id = task["task_id"]
        prediction = predictions.get(task_id, {})
        result = prediction.get("backend_result", prediction)
        option_ids = [option["option_id"] for option in task["options"]]
        selected = prediction.get("selected_option_id", result.get("selected_option_id"))
        status = prediction.get("status", "missing" if not prediction else "ok")
        valid = status == "ok" and selected in option_ids
        gold = answers[task_id]
        probabilities = _probabilities(result, option_ids) if valid else None
        confidence = prediction.get("confidence", result.get("confidence")) if valid else None
        if confidence is None and probabilities is not None:
            confidence = probabilities[selected]
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            confidence = None
        brier = sum((value - float(label == gold[0])) ** 2 for label, value in probabilities.items()) if probabilities and len(gold) == 1 else None
        oracle = metadata.get(task_id, {})
        selected_option = next((option for option in task["options"] if option["option_id"] == selected), {})
        information_request = selected_option.get("information_request")
        if information_request is None:
            information_request = any(step.get("answer") == "insufficient_information" or
                                      (task.get("task_type") == "measure" and step.get("kind") == "read") or
                                      str(step.get("capability_id", "")).endswith(".measure")
                                      for step in selected_option.get("steps", []))
        trace = result.get("metadata", {}).get("agentic_trace", {})
        shortlist = trace.get("shortlist_ids")
        citations = result.get("causes_cited", trace.get("causes_cited", []))
        citations = citations if isinstance(citations, list) and all(isinstance(item, str) for item in citations) else []
        evidence_checks = result.get("metadata", {}).get("evidence_citations_model_generated_unverified", [])
        expected_mentions = []
        mention_evidence_available = isinstance(evidence_checks, list) and bool(evidence_checks) and all(isinstance(text, str) for text in evidence_checks)
        if mention_evidence_available:
            text = "\n".join(evidence_checks)
            expected_mentions = [identifier for identifier in _cause_ids(oracle)
                                 if re.search(r"(?<![\w.])" + re.escape(identifier) + r"(?![\w.])", text)]
        tag = labels_index[task_id]
        rows.append({"task_id": task_id, "group_id": task.get("group_id", task_id),
                     "base_case_id": oracle.get("base_case_id"), "split": task.get("split"),
                     "instrument_id": task.get("instrument_id"), "domain": task.get("domain"),
                     "task_type": task.get("task_type"), "semantic_label": oracle.get("label_class"),
                     "primary_failure_mode": tag["primary_failure_mode"], "condition": tag["condition"],
                     "legacy_level": task.get("difficulty"), "status": status,
                     "attempted": bool(prediction), "valid": bool(valid),
                     "correct": bool(valid and selected in gold), "confidence": confidence,
                     "probabilities": probabilities, "gold_labels": gold,
                     "brier": brier, "uniform_brier": 1 - 1 / len(option_ids),
                     "expected_cause_ids": _cause_ids(oracle), "causes_cited": citations,
                     "cause_citation_source": "explicit_identifier_list",
                     "expected_cause_identifier_mentions": expected_mentions,
                     "expert_mention_evidence_available": mention_evidence_available,
                     "expected_information_request": oracle.get("expected_information_request")
                     if task.get("task_type") in {"predict", "measure", "diagnose"} else None,
                     "information_request": bool(information_request),
                     "authored_risk": task.get("risk"),
                     "shortlist_covers_gold": bool(set(shortlist) & set(gold)) if isinstance(shortlist, list) else None,
                     "latency_s": result.get("latency_s", prediction.get("latency_s")),
                     "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens")})
    return rows


def sequential_rows(cases, episodes, taxonomy, mode):
    labels = {row["task_id"]: row for row in taxonomy["labels"] if row["track"] == "sequential"}
    valid_ids = {case["task_id"] for case in cases}
    if any(key[0] not in valid_ids for key in episodes):
        raise ValueError("Episode file includes unknown frozen sequential IDs")
    result = []
    for case in cases:
        task_id = case["task_id"]
        episode = episodes.get((task_id, mode), {})
        completed = episode.get("status") == "ok"
        records = episode.get("decision_records", []) if completed else []
        controls = [record for record in records if not record["auxiliary_query"]]
        tag = labels[task_id]
        result.append({"task_id": task_id, "group_id": case.get("expansion", {}).get("split_group", case["workflow_id"]),
                       "workflow_id": case["workflow_id"], "instrument_id": case["instrument_id"],
                       "mode": mode, "primary_failure_mode": tag["primary_failure_mode"],
                       "condition": tag["condition"], "status": episode.get("status", "missing"),
                       "correct": bool(completed and episode.get("trajectory_outcome_met")),
                       "attempted": bool(episode), "valid": completed,
                       "control_accuracy": mean(record["proposal_correct"] for record in controls),
                       "proposal_accuracy": mean(record["proposal_correct"] for record in records),
                       "control_correct": sum(bool(record["proposal_correct"]) for record in controls),
                       "control_count": len(controls), "decision_count": len(records),
                       "interventions": sum(bool(record["corrected"]) for record in records),
                       "model_output_invalid": sum(bool(record.get("invalid_model_reply")) or
                           str(record.get("policy_error", "")).startswith("InvalidModelDecision:") for record in records),
                       "false_completion_claim": episode.get("false_completion_claim") if completed else None,
                       "latency_s": episode.get("policy_latency_s"),
                       "input_tokens": episode.get("input_tokens") if episode.get("token_accounting_complete") else None,
                       "output_tokens": episode.get("output_tokens") if episode.get("token_accounting_complete") else None,
                       "hierarchy_selections": sum(record["hierarchy_selections"] for record in records),
                       "node_confidence_rows": sum(record["node_confidence_rows"] for record in records),
                       "node_probability_rows": sum(record["node_probability_rows"] for record in records),
                       "expert_reference_coverage": episode.get("posthoc_expert_reference_coverage")})
        result[-1].update(partial_decisions_observed=episode.get("partial_decisions_observed", 0),
                          partial_policy_wall_s=episode.get("partial_policy_wall_s", 0),
                          partial_input_tokens_known=episode.get("partial_input_tokens_known", 0),
                          partial_output_tokens_known=episode.get("partial_output_tokens_known", 0))
    return result


def summarize_sequential(rows, *, bootstrap=1000):
    completed = [row for row in rows if row["valid"]]
    controls = sum(row["control_count"] for row in completed)
    reference_reports = [row["expert_reference_coverage"] for row in completed
                         if isinstance(row.get("expert_reference_coverage"), dict)]
    reference_events = sum(row.get("reference_offered_events", 0) for row in reference_reports)
    reference_included = sum(row.get("reference_in_shortlist_events", 0) for row in reference_reports)
    return {"n": len(rows), "unique_cases": len({row["task_id"] for row in rows}), "completed": len(completed),
            "coverage": len(completed) / len(rows) if rows else None,
            "attempt_coverage": mean(row["attempted"] for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
            "infrastructure_errors": sum(row["status"] == "infrastructure_error" for row in rows),
            "cohort_goal_completion": mean(row["correct"] for row in rows),
            "goal_completion_ci95": cluster_interval(rows, samples=bootstrap),
            "observed_control_accuracy_micro": sum(row["control_correct"] for row in completed) / controls if controls else None,
            "observed_control_accuracy_episode_macro": mean(row["control_accuracy"] for row in completed),
            "control_accuracy_ci95": cluster_interval(completed, field="control_accuracy", samples=bootstrap),
            "control_decisions_observed": controls,
            "interventions": sum(row["interventions"] for row in completed),
            "model_output_invalid": sum(row["model_output_invalid"] for row in completed),
            "false_completion_claims": sum(bool(row["false_completion_claim"]) for row in completed),
            "known_input_tokens": sum(row.get("input_tokens") or 0 for row in completed),
            "known_output_tokens": sum(row.get("output_tokens") or 0 for row in completed),
            "observed_policy_wall_s": sum(row.get("latency_s") or 0 for row in completed),
            "aborted_episode_partial_decisions_observed": sum(row.get("partial_decisions_observed", 0) for row in rows),
            "aborted_episode_partial_policy_wall_s": sum(row.get("partial_policy_wall_s", 0) for row in rows),
            "aborted_episode_partial_input_tokens_known": sum(row.get("partial_input_tokens_known", 0) for row in rows),
            "aborted_episode_partial_output_tokens_known": sum(row.get("partial_output_tokens_known", 0) for row in rows),
            "hierarchy_selections": sum(row["hierarchy_selections"] for row in completed),
            "conditional_confidence_rows": sum(row["node_confidence_rows"] for row in completed),
            "conditional_probability_rows": sum(row["node_probability_rows"] for row in completed),
            "canonical_reference_shortlist_coverage": reference_included / reference_events if reference_events else None,
            "canonical_reference_offered_events": reference_events,
            "canonical_reference_coverage_note": "Post hoc coverage of the authored canonical reference path only; other admissible actions may also exist.",
            "full_action_ece": None, "full_action_brier": None,
            "confidence_note": "Hierarchy-node probabilities are conditional; no full-action calibration is inferred."}


def _breakdowns(rows, summarize):
    return {axis: {value: summarize([row for row in rows if str(row.get(axis, "unknown")) == value])
                   for value in sorted({str(row.get(axis, "unknown")) for row in rows})}
            for axis in ("primary_failure_mode", "condition", "instrument_id")}


def analyze(experiments, tasks, answers, cases, taxonomy, *, bootstrap=1000, permutations=10000):
    entries, research_rows, comparison_rows = [], [], defaultdict(dict)
    family_raw = defaultdict(lambda: {"members": [], "mcqa": [], "sequential": defaultdict(list)})
    for entry in experiments:
        label = entry["name"]
        result = {key: entry[key] for key in ("name", "seed", "analysis_family", "architecture", "method", "backend", "identity", "sources")}
        result["artifact_authentication"] = entry.get("artifact_authentication")
        result["identities"] = entry.get("identities", [])
        result["experiment_fingerprint"] = fingerprint(entry["specification"])
        result["specification"] = entry["specification"]
        result["reporting_role"] = entry["specification"].get("reporting_role", "competitor")
        result["reporting_scope"] = entry["specification"].get("reporting_scope", "full_cohorts")
        artifact_directory = entry["identity"].get("artifact_path", entry["backend"].get("artifact_path"))
        if artifact_directory:
            efficiency = Path(artifact_directory) / "label-efficiency.json"
            if efficiency.is_file():
                result["label_efficiency"] = {"path": str(efficiency), "sha256": sha256(efficiency),
                                               "curve": json.loads(efficiency.read_text()).get("results", []),
                                               "scope": "Held-out evaluation after fitting nested train-only scenario-group budgets and validation calibration."}
        if "mcqa" in entry["requested_tracks"]:
            rows = mcqa_rows(tasks, answers, entry["mcqa"], taxonomy)
            if result["reporting_scope"] == "T1_reference":
                tier_index = {task["task_id"]: task.get("tier") for task in tasks}
                rows = [row for row in rows if tier_index[row["task_id"]] == "T1"]
            heldout = [row for row in rows if str(row.get("split", "")).startswith("test")]
            summarize = lambda group: summarize_decisions(group, bootstrap=bootstrap)
            result["mcqa"] = {"all_388": summarize(rows), "heldout_114": summarize(heldout),
                              "heldout_breakdowns": _breakdowns(heldout, summarize),
                              "validation_fitted_thresholds": validation_thresholds(rows)}
            comparison_rows["mcqa_heldout"][result["experiment_fingerprint"]] = heldout
            family_raw[entry["analysis_family"]]["mcqa"].extend(rows)
            research_rows.extend({**row, "experiment": label, "seed": entry["seed"], "track": "mcqa"} for row in rows)
        if "sequential" in entry["requested_tracks"]:
            result["sequential"] = {}
            for mode in ("teacher_forced", "free_run"):
                rows = sequential_rows(cases, entry["sequential"], taxonomy, mode)
                summarize = lambda group: summarize_sequential(group, bootstrap=bootstrap)
                result["sequential"][mode] = {**summarize(rows), "breakdowns": _breakdowns(rows, summarize)}
                comparison_rows["sequential_" + mode][result["experiment_fingerprint"]] = rows
                family_raw[entry["analysis_family"]]["sequential"][mode].extend(rows)
                research_rows.extend({**row, "experiment": label, "seed": entry["seed"], "track": "sequential"} for row in rows)
        entries.append(result)
        family_raw[entry["analysis_family"]]["members"].append(result)
    pairwise = {}
    index = {entry["experiment_fingerprint"]: entry for entry in entries}
    # Primary significance comparisons are between declared method families,
    # averaging seed repetitions within each task before whole-group swaps.
    # Seed 0 versus seed 1 is not an additional method comparison.
    family_results, family_pair_rows = [], defaultdict(dict)
    for family, contents in family_raw.items():
        members = contents["members"]
        signatures = set()
        # Pending plan entries may contain unresolved environment variables;
        # compare concrete configurations once their jobs have produced them.
        concrete = [member for member in members if member["sources"]] or members
        for member in concrete:
            specification = dict(member["specification"])
            for key in ("name", "seed", "analysis_family"):
                specification.pop(key, None)
            signatures.add(fingerprint(specification))
        if len(signatures) != 1:
            raise ValueError(f"analysis_family {family!r} mixes specifications beyond the declared random seed")
        value = {"name": family, "analysis_family": family, "seed": "pooled",
                 "seeds": sorted({member["seed"] for member in members}),
                 "architecture": members[0]["architecture"], "method": members[0]["method"],
                 "backend": members[0]["backend"], "members": [member["name"] for member in members],
                 "reporting_role": members[0]["reporting_role"],
                 "reporting_scope": members[0]["reporting_scope"]}
        if contents["mcqa"]:
            rows = contents["mcqa"]
            heldout = [row for row in rows if str(row.get("split", "")).startswith("test")]
            summarize = lambda group: summarize_decisions(group, bootstrap=bootstrap)
            value["mcqa"] = {"all_388": summarize(rows), "heldout_114": summarize(heldout),
                             "heldout_breakdowns": _breakdowns(heldout, summarize)}
            if value["reporting_role"] != "reference":
                family_pair_rows["mcqa_heldout"][family] = _average_seed_rows(heldout)
        if contents["sequential"]:
            value["sequential"] = {}
            for mode, rows in contents["sequential"].items():
                summarize = lambda group: summarize_sequential(group, bootstrap=bootstrap)
                value["sequential"][mode] = {**summarize(rows), "breakdowns": _breakdowns(rows, summarize)}
                if value["reporting_role"] != "reference":
                    family_pair_rows["sequential_" + mode][family] = _average_seed_rows(rows)
        family_results.append(value)
    prior_families = [entry["analysis_family"] for entry in entries if entry["backend"].get("kind") == "reference"
                      and entry["backend"].get("reference") == "majority"]
    if prior_families:
        prior_family = prior_families[0]
        prior_rows = family_raw[prior_family]["mcqa"]
        prior_index = {row["task_id"]: row for row in prior_rows}
        prior_complete = bool(prior_rows) and all(row["attempted"] and row["status"] != "infrastructure_error"
                                                 and row.get("brier") is not None for row in prior_rows)
        for entry in entries + family_results:
            if "mcqa" not in entry:
                continue
            pool = family_raw[entry["analysis_family"]]["mcqa"]
            if "specification" in entry:
                # Per-seed records are not replaced by a family-average score.
                pool = [row for row in research_rows if row["track"] == "mcqa" and
                        row["experiment"] == entry["name"] and row["seed"] == entry["seed"]]
            for scope, summary in (("all_388", entry["mcqa"]["all_388"]),
                                   ("heldout_114", entry["mcqa"]["heldout_114"])):
                cohort = pool if scope == "all_388" else [row for row in pool if str(row.get("split", "")).startswith("test")]
                matched = [row for row in cohort if row.get("brier") is not None and row["task_id"] in prior_index
                           and prior_index[row["task_id"]].get("brier") is not None]
                baseline = mean(prior_index[row["task_id"]]["brier"] for row in matched)
                supported = prior_complete and summary["attempt_coverage"] == 1 and summary["infrastructure_errors"] == 0
                summary["brier_skill_train_menu_prior"] = 1 - mean(row["brier"] for row in matched) / baseline if supported and baseline else None
                summary["brier_skill_train_menu_prior_n"] = len(matched) if supported else 0
                summary["brier_skill_train_menu_prior_reference"] = prior_family
    for scope, method_rows in family_pair_rows.items():
        comparisons = []
        for left, right in itertools.combinations(method_rows, 2):
            a, b = method_rows[left], method_rows[right]
            eligible = all(row.get("attempted") and row.get("status") != "infrastructure_error" for row in a + b)
            value = paired_permutation(a, b, samples=permutations) if eligible else {
                "p_value": None, "reason": "Missing/infrastructure coverage prevents a model-performance significance claim."}
            comparisons.append({"left": left, "right": right, "eligible": eligible, **value})
        pairwise[scope] = holm_adjust(comparisons)
    families = defaultdict(list)
    for entry in entries:
        families[entry["analysis_family"]].append(entry)
    seed_summaries = {}
    for family, members in families.items():
        accuracies = [entry.get("mcqa", {}).get("heldout_114", {}).get("accuracy") for entry in members]
        values = [value for value in accuracies if value is not None]
        seed_summaries[family] = {"seeds": sorted({entry["seed"] for entry in members}),
                                  "n_distinct_seeds": len({entry["seed"] for entry in members}),
                                  "experiments": [entry["name"] for entry in members],
                                  "heldout_accuracy_mean": mean(values),
                                  "heldout_accuracy_sd": statistics.stdev(values) if len(values) > 1 else None,
                                  "three_seed_sampling_complete": len({entry["seed"] for entry in members}) >= 3
                                  and all(_entry_complete(entry) for entry in members)
                                  if any(entry["method"] == "sampling" for entry in members) else None}
    return {"experiments": entries, "analysis_families": family_results,
            "paired_comparisons": pairwise, "seed_summaries": seed_summaries}, research_rows


def _average_seed_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    return [{"task_id": key, "group_id": values[0]["group_id"],
             "correct": mean(value["correct"] for value in values),
             "attempted": all(value.get("attempted") for value in values),
             "status": "infrastructure_error" if any(value.get("status") == "infrastructure_error" for value in values)
             else "missing" if any(value.get("status") == "missing" for value in values) else "ok"}
            for key, values in groups.items()]


def _entry_complete(entry):
    checks = []
    if "mcqa" in entry:
        summary = entry["mcqa"]["all_388"]
        checks.append(summary["attempt_coverage"] == 1 and summary["infrastructure_errors"] == 0)
    if "sequential" in entry:
        checks.extend(summary["coverage"] == 1 for summary in entry["sequential"].values())
    return bool(checks) and all(checks)


def _entry_outputs_present(entry):
    checks = []
    if "mcqa" in entry:
        checks.append(entry["mcqa"]["all_388"]["attempt_coverage"] == 1)
    if "sequential" in entry:
        checks.extend(summary["attempt_coverage"] == 1 for summary in entry["sequential"].values())
    return bool(checks) and all(checks)


def flat_rows(report):
    values = []
    for entry in report.get("analysis_families", report["experiments"]):
        row = {"experiment": entry["name"], "seed": entry["seed"], "family": entry["analysis_family"],
               "architecture": entry["architecture"], "method": entry["method"],
               "backend": entry["backend"].get("kind", "unknown"),
               "reporting_role": entry.get("reporting_role", "competitor"),
               "reporting_scope": entry.get("reporting_scope", "full_cohorts")}
        for scope in ("all_388", "heldout_114"):
            summary = entry.get("mcqa", {}).get(scope, {})
            for field in ("accuracy", "response_coverage", "attempt_coverage", "missing", "infrastructure_errors", "balanced_accuracy", "multiclass_brier", "correctness_auroc", "shortlist_coverage", "latency_mean_s"):
                row[scope + "_" + field] = summary.get(field)
            row[scope + "_ece"] = summary.get("calibration", {}).get("ece")
        if row["reporting_role"] == "reference":
            for scope in ("all_388", "heldout_114"):
                summary = entry.get("mcqa", {}).get(scope, {})
                row["reference_" + scope + "_accuracy"] = summary.get("accuracy")
                row["reference_" + scope + "_n"] = summary.get("unique_tasks", summary.get("n"))
        for mode in ("teacher_forced", "free_run"):
            summary = entry.get("sequential", {}).get(mode, {})
            for field in ("coverage", "cohort_goal_completion", "observed_control_accuracy_micro", "infrastructure_errors", "interventions"):
                row[mode + "_" + field] = summary.get(field)
        row["status"] = "complete" if _entry_complete(entry) else "partial_or_pending"
        values.append(row)
    return values


def compute_accounting(entries, raw_accounting=None):
    model_wall = 0.
    for entry in entries:
        model_wall += entry.get("mcqa", {}).get("all_388", {}).get("observed_wall_s", 0)
        model_wall += sum(summary.get("observed_policy_wall_s", 0) + summary.get("aborted_episode_partial_policy_wall_s", 0)
                          for summary in entry.get("sequential", {}).values())
    jobs, observed = [], set()
    for row in (raw_accounting or {}).get("jobs", []):
        job_id = str(row.get("job_id", row.get("JobID", "")))
        if not job_id or "." in job_id or job_id in observed:
            continue
        observed.add(job_id)
        elapsed = row.get("elapsed_seconds", row.get("elapsed_raw", row.get("ElapsedRaw")))
        gpu_count = row.get("allocated_gpus")
        if gpu_count is None:
            tres = row.get("alloc_tres", row.get("AllocTRES", ""))
            if isinstance(tres, str):
                match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
                if match:
                    gpu_count = int(match.group(1))
        try:
            seconds = float(elapsed)
            gpus = int(gpu_count)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(seconds) or seconds < 0 or gpus < 0:
            continue
        jobs.append({"job_id": job_id, "elapsed_seconds": seconds, "allocated_gpus": gpus,
                     "allocated_gpu_wall_seconds": seconds * gpus,
                     "hardware": row.get("hardware"), "state": row.get("state", row.get("State"))})
    return {"local_api_cost_usd": 0., "recorded_model_policy_wall_seconds": model_wall,
            "allocated_gpu_wall_seconds": sum(row["allocated_gpu_wall_seconds"] for row in jobs) if jobs else None,
            "accounted_allocation_jobs": jobs,
            "definitions": "Allocated GPU wall seconds = actual allocation elapsed seconds × allocated GPU count, excluding job steps. This is neither GPU kernel time nor API cost; model wall time excludes unrecorded failed calls."}


def publication_plots(report, output):
    """Standalone PDF/SVG/PNG artifacts; no browser screenshot dependencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    output = Path(output)
    output.mkdir()
    artifacts = []
    def save(fig, name):
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
        artifacts.append({"name": name, "png": str(output / f"{name}.png"),
                          "pdf": f"plots/{name}.pdf", "svg": f"plots/{name}.svg"})
    entries = [entry for entry in report.get("analysis_families", report["experiments"])
               if "mcqa" in entry and entry.get("reporting_role") != "reference"]
    if entries:
        entries.sort(key=lambda entry: entry["mcqa"]["heldout_114"]["accuracy"] or 0)
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(entries) * .36)))
        summaries = [entry["mcqa"]["heldout_114"] for entry in entries]
        y = np.arange(len(entries))
        estimates = [summary["accuracy"] or 0 for summary in summaries]
        colors = ["#b96e32" if entry["architecture"] != "direct" else "#267b87" for entry in entries]
        ax.barh(y, estimates, color=colors, height=.7)
        low = [max(0., value - (summary["accuracy_ci95"]["low"] if summary["accuracy_ci95"]["low"] is not None else value))
               for value, summary in zip(estimates, summaries)]
        high = [max(0., (summary["accuracy_ci95"]["high"] if summary["accuracy_ci95"]["high"] is not None else value) - value)
                for value, summary in zip(estimates, summaries)]
        ax.errorbar(estimates, y, xerr=[low, high], fmt="none", color="#20343d", capsize=2, lw=1)
        ax.set(yticks=y, yticklabels=[entry["name"] for entry in entries], xlim=(0, 1.06),
               xlabel="Accuracy on all 114 held-out questions", title="Held-out decisions · scenario-cluster 95% intervals")
        for index, summary in enumerate(summaries):
            ax.text(min(.99, estimates[index] + .015), index,
                    f"{estimates[index]:.1%}  ({summary['attempt_coverage']:.0%} attempted)", va="center", fontsize=8)
        save(fig, "heldout-accuracy")
        mechanisms = [key for key in report["taxonomy"]["counts"]["mcqa"]]
        matrix = [[entry["mcqa"]["heldout_breakdowns"]["primary_failure_mode"].get(key, {}).get("accuracy", float("nan"))
                   for key in mechanisms] for entry in entries]
        fig, ax = plt.subplots(figsize=(max(9, len(mechanisms) * 1.25), max(3.5, len(entries) * .36)))
        image = ax.imshow(np.array(matrix, dtype=float), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        ax.set(yticks=range(len(entries)), yticklabels=[entry["name"] for entry in entries],
               xticks=range(len(mechanisms)), xticklabels=[MECHANISMS[key] for key in mechanisms],
               title="Held-out accuracy by authored instrument mechanism")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.colorbar(image, ax=ax, label="Accuracy")
        save(fig, "failure-mechanisms")
        probabilistic = [entry for entry in entries if entry["mcqa"]["heldout_114"]["calibration"]["confidence_rows"]]
        if probabilistic:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].plot([0, 1], [0, 1], "--", color=".5", lw=1)
            for entry in probabilistic:
                summary = entry["mcqa"]["heldout_114"]
                bins = summary["calibration"]["bins"]
                axes[0].plot([value["mean_confidence"] for value in bins], [value["accuracy"] for value in bins],
                             marker="o", markersize=3, lw=1, label=entry["name"])
                curve = summary["selective_curve"]
                axes[1].plot([point["coverage"] for point in curve], [point["accuracy"] for point in curve], lw=1)
            axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Selected-answer confidence", ylabel="Observed accuracy", title="Reliability · available confidence only")
            axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage of all 114 questions", ylabel="Accuracy above threshold", title="Exploratory selective accuracy")
            axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -.22), ncol=2)
            save(fig, "calibration-selective")
    sequential = [entry for entry in report.get("analysis_families", report["experiments"])
                  if "sequential" in entry and entry.get("reporting_role") != "reference"]
    if sequential:
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, len(sequential) * .36)), sharey=True)
        y = np.arange(len(sequential))
        axes[0].barh(y, [entry["sequential"]["free_run"]["cohort_goal_completion"] or 0 for entry in sequential], color="#267b87")
        axes[1].barh(y, [entry["sequential"]["teacher_forced"]["observed_control_accuracy_micro"] or 0 for entry in sequential], color="#b96e32")
        axes[0].set(yticks=y, yticklabels=[entry["name"] for entry in sequential], xlim=(0, 1), xlabel="Goals reached / all 120 cases", title="Free run · autonomous outcome")
        axes[1].set(xlim=(0, 1), xlabel="Correct original controls / observed controls", title="Teacher forcing · proposal accuracy")
        save(fig, "sequential-outcomes")
    efficiency_entries = [entry for entry in report["experiments"] if entry.get("label_efficiency", {}).get("curve")]
    if efficiency_entries:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for entry in efficiency_entries:
            budgets = defaultdict(list)
            for row in entry["label_efficiency"]["curve"]:
                budgets[row["actual_group_budget"]].append(row["accuracy"])
            ordered = sorted(budgets)
            estimates = [mean(budgets[budget]) for budget in ordered]
            deviations = [statistics.stdev(budgets[budget]) if len(budgets[budget]) > 1 else 0 for budget in ordered]
            ax.errorbar(ordered, estimates, yerr=deviations, marker="o", capsize=3, label=entry["name"])
        ax.set(xlabel="Label budget · independent training scenario groups", ylabel="Held-out accuracy",
               ylim=(0, 1), title="Frozen-feature label efficiency · mean ± seed standard deviation")
        ax.legend(fontsize=8)
        save(fig, "label-efficiency")
    return artifacts


def render_html(report, plots):
    payload = json.dumps([row for row in flat_rows(report) if row.get("reporting_role") != "reference"],
                         ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    figures = "".join('<figure><img alt="' + html.escape(item["name"]) + '" src="data:image/png;base64,' +
                      base64.b64encode(Path(item["png"]).read_bytes()).decode() + '"><figcaption>' +
                      html.escape(item["name"].replace("-", " ")) + ' · <a href="' + item["pdf"] + '">PDF</a> · <a href="' + item["svg"] + '">SVG</a></figcaption></figure>' for item in plots)
    counts = report["taxonomy"]["counts"]
    mechanism_rows = "".join("<tr><td>" + html.escape(MECHANISMS[key]) + "</td><td>" + str(counts["mcqa"].get(key, 0)) + "</td><td>" + str(counts["sequential"].get(key, 0)) + "</td></tr>"
                             for key in sorted(set(counts["mcqa"]) | set(counts["sequential"])))
    warnings = "".join("<li>" + html.escape(message) + "</li>" for message in report["warnings"])
    references = [entry for entry in report.get("analysis_families", report["experiments"])
                  if entry.get("reporting_role") == "reference"]
    reference_rows = "".join("<tr><td>" + html.escape(entry["name"]) + "</td><td>" +
                             html.escape(entry.get("reporting_scope", "reference")) + "</td><td>" +
                             str(entry.get("mcqa", {}).get("heldout_114", {}).get("unique_tasks", "—")) + "</td><td>" +
                             (f"{entry['mcqa']['heldout_114']['accuracy']:.1%}" if entry.get("mcqa", {}).get("heldout_114", {}).get("attempt_coverage") else "Pending") +
                             "</td></tr>" for entry in references)
    reference_table = ("<h2>Reference baselines</h2><section><p class='method-note'>Defined reference scopes are separate from full-cohort competitors and excluded from ranking and method significance tests.</p><table><thead><tr><th>Reference</th><th>Defined scope</th><th>Held-out tasks in scope</th><th>Accuracy</th></tr></thead><tbody>" + reference_rows + "</tbody></table></section>") if references else ""
    tool_rows = []
    for track in report.get("tool_tracks", []):
        for name, summary in track["methods"].items():
            def percentage(key):
                return f"{summary[key]:.1%}" if type(summary.get(key)) in (int, float) else "—"
            tool_rows.append("<tr><td>" + html.escape(name) + "</td><td>" +
                             percentage("full_source_accuracy") + "</td><td>" +
                             percentage("full_source_heldout_accuracy") + "</td><td>" +
                             percentage("conditional_label_coverage") + "</td><td>" +
                             str(summary.get("labels_changed_after_information", "—")) + "</td></tr>")
    tool_table = ("<h2>Tool access · a separate information track</h2><section><p class='intro'>The model can read or describe the simulator before deciding. Answers are re-labelled against the actual revealed transcript and checkpoint replay is verified. These results receive different information and do not join the static-key ranking or paired tests.</p><table><thead><tr><th>Readout after tools</th><th>All 388 source accuracy</th><th>Held-out 114 source accuracy</th><th>Conditional-label coverage</th><th>Labels changed</th></tr></thead><tbody>" + "".join(tool_rows) + "</tbody></table></section>") if tool_rows else ("<h2>Tool access · pending</h2><section>No verified conditional-label tool report has been collected in this snapshot.</section>") if report.get("tool_track_required") else ""
    status = ("Complete declared evaluation" if report.get("complete_declared_study") else
              "All requested outputs collected · unsupported or infrastructure cases remain" if report.get("all_expected_outputs_present") else
              "Partial snapshot · pending outputs remain")
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Instrument decisions · local model study</title><style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#23353e;background:#f4f6f5;font-size:15px}body{max-width:1440px;margin:0 auto;padding:34px}h1{font-size:clamp(28px,4vw,46px);letter-spacing:-1.3px;margin:12px 0}h2{font-size:24px;margin-top:34px}.eyebrow{color:#286f79;text-transform:uppercase;font-size:12px;letter-spacing:2px;font-weight:750}.intro{max-width:900px;line-height:1.6;color:#52636a}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:25px 0}.card,section,figure{background:white;border:1px solid #dce4e2;border-radius:13px;padding:20px}.card strong{display:block;font-size:31px;color:#276f79}.card span{font-size:13px;color:#607177}.notice{border-left:4px solid #ba773b;background:#fff7ec;padding:15px 20px;line-height:1.6}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}input,select{font:inherit;padding:10px 13px;border:1px solid #c9d7d4;border-radius:7px;background:white}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:12px 10px;text-align:left;border-bottom:1px solid #e5ebe8;white-space:nowrap}th{font-size:11px;text-transform:uppercase;letter-spacing:.5px;cursor:pointer;color:#63747a;background:#f8faf9;position:sticky;top:0}td:first-child{font-weight:650}.bad{color:#a35f25}.plots{display:grid;gap:20px}figure{margin:0;overflow:auto}figure img{width:100%;height:auto}figcaption{font-size:12px;color:#697b80;padding-top:12px}a{color:#286f79}.method-note{font-size:12px;line-height:1.5;color:#627278}details{margin-top:18px}summary{cursor:pointer;font-weight:650}.footer{padding:26px 0;color:#74848a;font-size:12px}li{margin:7px 0}.pill{display:inline-block;border-radius:99px;padding:4px 9px;background:#e9f2ef;font-size:11px}
</style><header><div class="eyebrow">ChemLab-derived instrument benchmark · local study</div><h1>How should a lab agent decide?</h1><span class="pill">''' + html.escape(status) + '''</span><p class="intro">Same public instrument evidence and offered actions, different local models and decision mechanisms. This report separates static choices, adaptive control with teacher-forced correction, and autonomous free runs.</p></header>
<div class="cards"><div class="card"><strong>388</strong><span>Static questions · all-set diagnostics</span></div><div class="card"><strong>114</strong><span>Held-out questions · model comparisons</span></div><div class="card"><strong>120</strong><span>Sequential cases · two evaluation modes</span></div><div class="card"><strong>''' + str(len(report["experiments"])) + '''</strong><span>Recorded experiment / seed configurations</span></div></div>
<div class="notice"><b>Interpret the denominators.</b> Missing outputs and infrastructure failures stay in the question/case cohort. Step accuracy uses observed proposals; read it with episode coverage. Teacher-forced goal completion includes oracle corrections and is not autonomous success. Synthetic simulator validity has not received independent instrument-expert review.</div>
<h2>Model and mechanism comparison</h2><div class="toolbar"><input id="search" placeholder="Filter model, method or name" aria-label="Filter experiments"><select id="architecture"><option value="">All architectures</option><option value="direct">Direct decisions</option><option value="expert_decision">Expert + decision model</option></select></div>
<section class="table-wrap"><table id="results"><thead><tr></tr></thead><tbody></tbody></table><p class="method-note">Click a column header to sort. “—” means unavailable. Accuracy includes invalid model replies; coverage and infrastructure counts explain incomplete execution. Confidence and Brier apply only where the readout actually supports them. Rows pool predeclared seeds within each method family; report.json retains every seed and its variation. Correlated repeats stay in the same uncertainty cluster.</p></section>
''' + reference_table + tool_table + '''<h2>Instrument failure mechanisms</h2><p class="intro">Primary groups follow authored fault families and workflow targets. Healthy members of a matched pair retain that mechanism and are marked separately as healthy controls. Readiness, missing prerequisites, hardware faults and ambiguous evidence are distinct conditions. Legacy L1–L5 IDs remain unchanged in the dataset.</p><section class="table-wrap"><table><thead><tr><th>Authored target mechanism</th><th>Static questions</th><th>Sequential cases</th></tr></thead><tbody>''' + mechanism_rows + '''</tbody></table></section>
<h2>Results and uncertainty</h2><div class="plots">''' + figures + '''</div>
<h2>Research interpretation</h2><section><ul><li>Model comparisons and readout comparisons are different experiments. HF and Ollama are inference runtimes; conversion, precision and tokenizer identities are recorded.</li><li>Confidence intervals resample whole correlated scenario groups for MCQA and whole workflows for sequential cases. Paired method swaps use those groups; Holm correction is applied within each declared comparison scope.</li><li>ECE uses selected-answer confidence. Verbal confidence never becomes a fabricated full vector. Multiclass Brier uses complete genuine probabilities and single-gold questions only. AUROC is unavailable without both correct and incorrect confident answers.</li><li>Selective curves on the test set are exploratory. Separate thresholds are fitted on validation and then assessed on held-out questions. They are not hardware-safety guarantees.</li><li>Correct cited public cause IDs establish citation agreement with authored causes, not faithfulness of internal reasoning. Missing citations remain uncovered. Risk tags describe authored simulator tasks, not verified real-world hazards.</li><li>Local API spend is zero. Model-call wall time, token-accounting coverage, hardware identities and optional Slurm allocation accounting remain separate; wall time is never renamed GPU kernel time.</li></ul>
<p><a href="report.json">Complete results and provenance</a> · <a href="summary.csv">Comparison CSV</a> · <a href="taxonomy.json">Evidence-backed taxonomy</a> · <a href="research-records.jsonl">Question and case research records</a></p></section>
<details><summary>Source coverage and warnings (''' + str(len(report["warnings"])) + ''')</summary><ul>''' + warnings + '''</ul></details><div class="footer">Snapshot ''' + html.escape(report["generated_at"]) + ''' · ''' + VERSION + ''' · Full hashes and definitions accompany this page.</div>
<script>const rows=''' + payload + ''';const columns=[['experiment','Experiment'],['architecture','Architecture'],['all_388_accuracy','All 388'],['heldout_114_accuracy','Held-out 114'],['heldout_114_ece','ECE'],['heldout_114_multiclass_brier','Brier'],['heldout_114_response_coverage','Valid answers'],['heldout_114_infrastructure_errors','Infra'],['free_run_cohort_goal_completion','Free-run goals'],['free_run_coverage','Episode coverage'],['teacher_forced_observed_control_accuracy_micro','TF controls'],['teacher_forced_cohort_goal_completion','TF assisted goals']];let sortKey='heldout_114_accuracy',descending=true;const tr=document.querySelector('#results thead tr');columns.forEach(([key,label])=>{const th=document.createElement('th');th.textContent=label;th.onclick=()=>{descending=sortKey===key?!descending:true;sortKey=key;render()};tr.append(th)});function format(k,v){if(v===null||v===undefined)return'—';if(typeof v==='number'){if(k.includes('accuracy')||k.includes('coverage')||k.includes('completion'))return(100*v).toFixed(1)+'%';if(Number.isInteger(v))return String(v);return v.toFixed(3)}return String(v)}function render(){const query=document.querySelector('#search').value.toLowerCase(),arch=document.querySelector('#architecture').value;const selected=rows.filter(r=>(!arch||r.architecture===arch)&&JSON.stringify(r).toLowerCase().includes(query));selected.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];if(x==null)return 1;if(y==null)return-1;return(typeof x==='number'?x-y:String(x).localeCompare(String(y)))*(descending?-1:1)});const body=document.querySelector('#results tbody');body.replaceChildren();selected.forEach(row=>{const item=document.createElement('tr');columns.forEach(([key])=>{const cell=document.createElement('td');cell.textContent=format(key,row[key]);if(key.includes('infrastructure')&&row[key]>0)cell.className='bad';item.append(cell)});body.append(item)})}document.querySelector('#search').oninput=render;document.querySelector('#architecture').onchange=render;render();</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--plan", type=Path, help="Frozen study matrix with all declared experiments")
    parser.add_argument("--mcqa-oracles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--slurm-accounting", type=Path, help="Optional already-exported Slurm accounting JSON")
    parser.add_argument("--tool-runs", type=Path, action="append", default=[])
    parser.add_argument("--require-tool-track", action="store_true")
    args = parser.parse_args(argv)
    require_allocation()
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.bootstrap < 1 or args.permutations < 1:
        parser.error("resample counts must be positive")
    bundle = args.bundle.resolve()
    taxonomy = json.loads(args.taxonomy.read_text()) if args.taxonomy else build_taxonomy(bundle, args.mcqa_oracles)
    if (taxonomy["mcqa_questions_sha256"] != sha256(bundle / "mcqa_388/questions.jsonl.gz") or
            taxonomy["sequential_cases_sha256"] != sha256(bundle / "sequential_120/cases.jsonl")):
        raise ValueError("Taxonomy does not match the frozen task files")
    tasks = load_jsonl(bundle / "mcqa_388/questions.jsonl.gz")
    answers = {row["task_id"]: row["correct_option_ids"] for row in load_jsonl(bundle / "mcqa_388/answers.jsonl")}
    cases = load_jsonl(bundle / "sequential_120/evaluator/case-specs.jsonl")
    experiments, sources, warnings = discover_experiments(args.runs, bundle,
                                                         include_diagnostics=args.include_diagnostics)
    tool_tracks, tool_warnings = discover_tool_tracks(args.runs + args.tool_runs)
    warnings.extend(tool_warnings)
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if plan is not None:
        experiments = apply_plan(experiments, plan)
    result, research_rows = analyze(experiments, tasks, answers, cases, taxonomy,
                                   bootstrap=args.bootstrap, permutations=args.permutations)
    if not experiments:
        warnings.append("No authenticated study run configuration was found; this is a dataset/taxonomy report only.")
    if taxonomy.get("oracle_source") is None:
        warnings.append("Original authenticated MCQA oracle metadata was not supplied; mechanism, semantic-label and rationale metrics may be unavailable.")
    report = {"schema_version": 1, "report_protocol": VERSION,
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
              "report_source_sha256": {name: sha256(Path(__file__).parent / name)
                                        for name in ("taxonomy.py", "study_metrics.py", "study_report.py", "aggregate.py")},
              "compute_node": os.environ["SLURMD_NODENAME"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "taxonomy": {key: taxonomy[key] for key in ("taxonomy_version", "counts", "condition_counts", "oracle_source")},
              "sources": sources, "warnings": warnings,
              "tool_tracks": tool_tracks, "tool_track_required": args.require_tool_track,
              "study_plan": {"sha256": sha256(args.plan), "experiments": len(plan.get("experiments", []))}
              if plan is not None else None,
              "slurm_accounting": json.loads(args.slurm_accounting.read_text()) if args.slurm_accounting else None,
              "definitions": {"all_388": "All released static questions; includes train/validation diagnostics, not an unseen-only result.",
                              "heldout_114": "Primary static comparison: all test/test_template/test_fault/test_instrument questions.",
                              "sequential": "All 120 authored cases; no train/test split assigned. Teacher-forced and free-run modes are separate.",
                              "missing_policy": "Missing and invalid outputs remain in question/case denominators; observed-step metrics do not invent absent action opportunities.",
                              "confidence": "Selected-answer scalar and full menu vectors are distinct; full-action calibration unavailable for hierarchical sequential selections.",
                              "brier_skill_uniform_menu": "Relative to fixed uniform probability over each offered menu; not a fitted training prior.",
                              "uncertainty": "Whole semantic scenario clusters for MCQA; whole workflows for sequential. Missing/infrastructure coverage suppresses significance claims.",
                              "limitations": "Independent expert review pending; authored simulated operations, no coupled material/chemical evolution or physical-hardware validation."},
              **result}
    report["complete_declared_study"] = bool(experiments) and all(_entry_complete(entry) for entry in report["experiments"])
    report["all_expected_outputs_present"] = bool(experiments) and all(_entry_outputs_present(entry) for entry in report["experiments"])
    if args.require_tool_track:
        tool_complete = bool(tool_tracks) and all(summary.get("complete_source_cohort") is True
                                                  for track in tool_tracks for summary in track["methods"].values())
        report["complete_declared_study"] &= tool_complete
    report["completed_experiment_scopes"] = sum(_entry_complete(entry) for entry in report["experiments"])
    report["compute_accounting"] = compute_accounting(report["experiments"], report["slurm_accounting"])
    args.output.mkdir(parents=True)
    write_json(args.output / "taxonomy.json", taxonomy)
    write_json(args.output / "report.json", report)
    flat = flat_rows(report)
    fields = sorted({key for row in flat for key in row}) or ["experiment"]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    with (args.output / "research-records.jsonl").open("w") as stream:
        for row in research_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    plots = [] if args.no_plots else publication_plots(report, args.output / "plots")
    (args.output / "index.html").write_text(render_html(report, plots))
    print(json.dumps({"status": "report_written", "experiments": len(experiments),
                      "output": str(args.output), "warnings": len(warnings)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
