"""Run interactive instrument episodes or summarize saved evaluator records.

``run`` needs the bundled native runtimes and jsonschema. ``score`` uses only
the standard library and summarizes evaluator-produced records; it does not
replace simulator grading with static answers. On SCITAS both commands belong
inside a Slurm compute-node allocation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__:
    from .common import _record_id, load_jsonl
else:
    from common import _record_id, load_jsonl


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BUNDLE_ROOT / "sequential_120"
MODES = ("teacher_forced", "free_run")
PROTOCOL = "instrument-sequential-public-choices-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _index_tasks(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _require(isinstance(tasks, list) and bool(tasks), "tasks must be a nonempty list")
    index = {}
    for position, task in enumerate(tasks, 1):
        task_id = _record_id(task, f"task {position}")
        _require(task_id not in index, f"duplicate task_id: {task_id!r}")
        index[task_id] = task
    return index


def _metadata(task: dict[str, Any], field: str) -> str:
    value = task.get(field)
    if value is None and field == "condition":
        value = task.get("expansion", {}).get("condition")
    return value if isinstance(value, str) and value else "unspecified"


def _validate_episode(row: dict[str, Any]) -> None:
    status = row.get("status", "ok")
    _require(isinstance(status, str) and status in {"ok", "infrastructure_error"}, "episode status must be ok or infrastructure_error")
    if status != "ok":
        _require(isinstance(row.get("error"), str) and bool(row["error"]),
                 "infrastructure_error requires a nonempty error message")
        return
    records = row.get("decision_records")
    _require(isinstance(records, list) and bool(records), "completed episode requires decision_records")
    _require(type(row.get("trajectory_outcome_met")) is bool, "trajectory_outcome_met must be boolean")
    _require(row.get("menu_membership_verified") is True, "episode must verify public-menu membership")
    _require(row.get("termination") != "invalid_curriculum_correction", "invalid correction is infrastructure_error")
    for position, record in enumerate(records, 1):
        _require(isinstance(record, dict), "each decision record must be an object")
        _require(type(record.get("decision_index")) is int and record["decision_index"] == position,
                 "decision_index must be contiguous and start at one")
        for field in ("proposal_correct", "auxiliary_query", "corrected"):
            _require(type(record.get(field)) is bool, f"decision {position}: {field} must be boolean")
        expected_correction = row["mode"] == "teacher_forced" and not record["proposal_correct"]
        _require(record["corrected"] == expected_correction,
                 f"decision {position}: correction flag disagrees with mode and original grade")
        _require(isinstance(record.get("proposed_action"), dict), "proposed_action must be an offered action object")
        _require(isinstance(record.get("response"), dict), "decision response must be an object")
        if not record["auxiliary_query"]:
            _require(isinstance(record.get("checkpoint_id"), str) and bool(record["checkpoint_id"]),
                     "control decision requires checkpoint_id")
            _require(isinstance(record.get("stage"), str) and bool(record["stage"]),
                     "control decision requires stage")
    if "decisions" in row:
        _require(type(row["decisions"]) is int and row["decisions"] == len(records), "decisions disagrees with records")
    if "interventions" in row:
        _require(type(row["interventions"]) is int and row["interventions"] == sum(r["corrected"] for r in records),
                 "interventions disagrees with records")
    expected_field = "autonomous_success" if row["mode"] == "free_run" else "corrected_trajectory_outcome_met"
    if expected_field in row:
        _require(row[expected_field] is row["trajectory_outcome_met"], f"{expected_field} disagrees with outcome")
    unavailable_field = "corrected_trajectory_outcome_met" if row["mode"] == "free_run" else "autonomous_success"
    _require(row.get(unavailable_field) is None, f"{unavailable_field} must be null in this mode")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    completed = [row["episode"] for row in rows if row["status"] == "ok"]
    steps = [step for episode in completed for step in episode["decision_records"]]
    control = [step for step in steps if not step["auxiliary_query"]]
    queries = [step for step in steps if step["auxiliary_query"]]
    successes = sum(episode["trajectory_outcome_met"] for episode in completed)
    episode_all = [_mean([int(step["proposal_correct"]) for step in episode["decision_records"]]) for episode in completed]
    episode_control = [value for episode in completed
                       if (value := _mean([int(step["proposal_correct"]) for step in episode["decision_records"]
                                           if not step["auxiliary_query"]])) is not None]
    checkpoint_grades = []
    stage_grades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in completed:
        grades: dict[str, list[bool]] = defaultdict(list)
        for step in episode["decision_records"]:
            if not step["auxiliary_query"]:
                grades[step["checkpoint_id"]].append(step["proposal_correct"])
                stage_grades[step["stage"]].append(step)
        checkpoint_grades.append(_mean([int(all(values)) for values in grades.values()]))
    result = {
        "n": len(rows), "completed": len(completed),
        "missing": sum(row["status"] == "missing" for row in rows),
        "infrastructure_failures": sum(row["status"] == "infrastructure_error" for row in rows),
        "coverage": len(completed) / len(rows) if rows else None,
        "complete": len(completed) == len(rows),
        "goals_reached": successes,
        "cohort_goal_completion_rate": successes / len(rows) if rows else None,
        "completed_episode_goal_rate": successes / len(completed) if completed else None,
        "decisions_observed": len(steps),
        "correct_original_proposals": sum(step["proposal_correct"] for step in steps),
        "observed_proposal_accuracy_micro": _mean([int(step["proposal_correct"]) for step in steps]),
        "observed_proposal_accuracy_episode_macro": _mean(episode_all),
        "control_decisions_observed": len(control),
        "observed_control_accuracy_micro": _mean([int(step["proposal_correct"]) for step in control]),
        "observed_control_accuracy_episode_macro": _mean(episode_control),
        "query_decisions_observed": len(queries),
        "observed_query_admissibility": _mean([int(step["proposal_correct"]) for step in queries]),
        "observed_checkpoint_accuracy_episode_macro": _mean([value for value in checkpoint_grades if value is not None]),
        "interventions": sum(step["corrected"] for step in steps),
        "privileged_diagnostic_episodes": sum(bool(episode.get("privileged_policy", episode.get("privileged_diagnostic", False)))
                                               for episode in completed),
        "by_stage": {
            stage: {"observed_decisions": len(values),
                    "original_correct": sum(value["proposal_correct"] for value in values),
                    "observed_accuracy": _mean([int(value["proposal_correct"]) for value in values])}
            for stage, values in sorted(stage_grades.items())
        },
    }
    result["autonomous_success_rate" if mode == "free_run" else "assisted_goal_completion_rate"] = result["cohort_goal_completion_rate"]
    return result


def score_episodes(tasks: list[dict[str, Any]], episodes: list[dict[str, Any]],
                   modes: tuple[str, ...] | list[str] = MODES) -> dict[str, Any]:
    """Summarize one simulator-produced episode per task/mode.

    All selected tasks remain in episode-level denominators. Step accuracies
    describe observed original proposals only; absent trajectories have no
    invented action count. A report with incomplete coverage is provisional.
    """
    index = _index_tasks(tasks)
    _require(isinstance(modes, (tuple, list)) and bool(modes)
             and all(isinstance(mode, str) and mode in MODES for mode in modes)
             and len(modes) == len(set(modes)), "invalid or duplicate modes")
    _require(isinstance(episodes, list), "episodes must be a list")
    episode_map = {}
    for position, episode in enumerate(episodes, 1):
        task_id = _record_id(episode, f"episode {position}")
        _require(task_id in index, f"unknown episode task_id: {task_id!r}")
        mode = episode.get("mode")
        _require(mode in MODES, f"unknown episode mode: {mode!r}")
        key = (task_id, mode)
        _require(key not in episode_map, f"duplicate episode task/mode: {key!r}")
        _validate_episode(episode)
        episode_map[key] = episode
    reports = {}
    for mode in modes:
        rows = []
        for task_id, task in index.items():
            episode = episode_map.get((task_id, mode))
            rows.append({"task": task, "episode": episode,
                         "status": episode.get("status", "ok") if episode is not None else "missing"})
        report = _summarize(rows, mode)
        for field in ("workflow_id", "instrument_id", "condition"):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[_metadata(row["task"], field)].append(row)
            report[f"by_{field}"] = {key: _summarize(group, mode) for key, group in sorted(groups.items())}
        workflow_reports = list(report["by_workflow_id"].values())
        report["workflow_macro_goal_completion_rate"] = _mean([row["cohort_goal_completion_rate"] for row in workflow_reports])
        report["observed_workflow_macro_proposal_accuracy"] = _mean([
            row["observed_proposal_accuracy_episode_macro"] for row in workflow_reports
            if row["observed_proposal_accuracy_episode_macro"] is not None])
        report["observed_workflow_macro_control_accuracy"] = _mean([
            row["observed_control_accuracy_episode_macro"] for row in workflow_reports
            if row["observed_control_accuracy_episode_macro"] is not None])
        report["workflows_with_observations"] = sum(row["completed"] > 0 for row in workflow_reports)
        reports[mode] = report
    expected = len(tasks) * len(modes)
    completed = sum(row["completed"] for row in reports.values())
    return {
        "protocol": PROTOCOL, "dataset_size": len(tasks), "expected_episodes": expected,
        "submitted_episode_rows": len(episodes), "completed_episodes": completed,
        "ignored_other_mode_rows": sum(mode not in modes for _, mode in episode_map),
        "complete": completed == expected,
        "by_mode": reports,
        "definitions": {
            "cohort_goal_completion_rate": "goals reached / all selected tasks; missing and infrastructure episodes contribute no completed goal; incomplete reports are provisional coverage-dependent lower bounds",
            "completed_episode_goal_rate": "goals reached / completed episodes only; always read with coverage",
            "observed_proposal_accuracy_micro": "correct original proposals / all observed original proposals, including information queries",
            "observed_control_accuracy_micro": "correct original control proposals / observed control proposals, excluding auxiliary queries",
            "episode_macro": "unweighted mean of completed episode rates, not pooled steps",
            "workflow_macro": "unweighted mean across workflow rates; observed variants exclude workflows with no observations and report their coverage",
            "observed_checkpoint_accuracy_episode_macro": "per episode, each visited checkpoint scores one iff every proposal at that checkpoint was correct; then average completed episode rates; unvisited checkpoints are not inferred",
            "assisted_goal_completion_rate": "teacher-forced execution may reach the goal after incorrect model proposals; this is not autonomous success",
            "autonomous_success_rate": "free-run goal completion over the full selected cohort without corrections",
            "by_stage": "observed original control decisions grouped by evaluator checkpoint stage; no fixed future action denominator is invented",
            "trust": "offline scoring checks record consistency and aggregates native evaluator grades; it does not re-simulate or authenticate arbitrary submitted correctness flags",
        },
    }


class OfferedChoicePolicy:
    """Adapt a plugin to finite machine/command/argument choices.

    ``plugin.choose(view, history, options)`` returns a zero-based integer index.
    It is called again with argument options if needed. Every native action is
    still scored once. Alternatively ``next_action(view, history)`` may return
    an offered native action directly. No private case or reference is passed.
    """
    def __init__(self, plugin: Any, menus: Any, error_type: type[Exception] = ValueError):
        self.plugin, self.menu_builder = plugin, menus
        self.error_type = error_type
        self.privileged = bool(getattr(plugin, "privileged", False))
        self.identity = copy.deepcopy(getattr(plugin, "identity", {}))
        self.last_action_usage = {}

    def next_action(self, view: dict, history: list) -> dict:
        try:
            if hasattr(self.plugin, "choose"):
                options = self.menu_builder.action_options(view, history)
                selection = self.plugin.choose(copy.deepcopy(view), copy.deepcopy(history), copy.deepcopy(options))
                _require(type(selection) is int and 0 <= selection < len(options), "choose() must return an offered zero-based index")
                selected = options[selection]
                if "action" in selected:
                    return copy.deepcopy(selected["action"])
                arguments = self.menu_builder.argument_options(view, history, selected)
                selection = self.plugin.choose(copy.deepcopy(view), copy.deepcopy(history), copy.deepcopy(arguments))
                _require(type(selection) is int and 0 <= selection < len(arguments), "argument choose() must return an offered zero-based index")
                return copy.deepcopy(arguments[selection]["action"])
            return self.plugin.next_action(copy.deepcopy(view), copy.deepcopy(history))
        except Exception as error:
            raise self.error_type(f"Policy interface failure: {type(error).__name__}: {error}") from error
        finally:
            self.last_action_usage = copy.deepcopy(getattr(self.plugin, "last_action_usage", {}))


def _runtime():
    if str(BUNDLE_ROOT) not in sys.path:
        sys.path.insert(0, str(BUNDLE_ROOT))
    from sequential_120.runtime.bootstrap import activate
    activate()
    from stepwise_choice.engine import MenuProtocolError, MenuReferencePolicy, WrongMenuPolicy, run_case
    from stepwise_choice.menus import MenuBuilder
    from stepwise_bench.policies import require_allocation
    require_allocation()
    return MenuReferencePolicy, WrongMenuPolicy, run_case, MenuBuilder, MenuProtocolError


def _source_provenance() -> dict[str, str]:
    paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    paths.extend(sorted((DATA_ROOT / "runtime").rglob("*.py")))
    for name in ("manifest.json", "runtime/provenance.json"):
        path = DATA_ROOT / name
        if path.is_file():
            paths.append(path)
    return {path.relative_to(BUNDLE_ROOT).as_posix(): _digest(path) for path in paths}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the fixed cases through the unchanged native sequential engine."""
    reference, wrong, run_case, menu_type, interface_error = _runtime()
    tasks, cases = load_jsonl(args.tasks), load_jsonl(args.cases)
    index = _index_tasks(tasks)
    case_index = _index_tasks(cases)
    _require(set(index) == set(case_index), "public tasks and evaluator cases must have identical IDs")
    bank = _json(args.presets)
    menus = menu_type(bank)
    modes = MODES if args.mode == "both" else (args.mode,)
    selected = tasks if args.limit is None else tasks[:args.limit]
    _require(bool(selected), "no cases selected")
    scoring_tasks = [case_index[task["task_id"]] for task in selected]
    plugin_factory = None
    plugin_source = None
    if args.policy not in {"reference", "wrong-menu"}:
        module_name, separator, attribute = args.policy.partition(":")
        _require(bool(separator and module_name and attribute), "policy must be reference, wrong-menu or module:factory")
        module = importlib.import_module(module_name)
        plugin_factory = getattr(module, attribute)
        _require(callable(plugin_factory), "policy factory must be callable")
        module_path = getattr(module, "__file__", None)
        if module_path and Path(module_path).is_file():
            plugin_source = {"path": str(Path(module_path).resolve()), "sha256": _digest(Path(module_path))}
    config = {"protocol": PROTOCOL, "policy": args.policy, "modes": modes,
              "selected_task_ids": [task["task_id"] for task in selected],
              "public_tasks_sha256": _digest(args.tasks), "private_cases_sha256": _digest(args.cases),
              "public_presets_sha256": _digest(args.presets), "source_sha256": _source_provenance(),
              "policy_factory_source": plugin_source,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": os.environ.get("SLURMD_NODENAME"),
              "privileged_diagnostic": args.policy in {"reference", "wrong-menu"}}
    version_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["run_configuration_sha256"] = version_hash
    output = args.output.resolve()
    _require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
             "--output must be a fresh empty directory; each source/configuration change needs a new run")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "configuration.json", config)
    episodes = []

    def summarize_run():
        report = score_episodes(scoring_tasks, episodes, modes)
        subset = len(selected) < len(tasks)
        report.update(full_dataset_size=len(tasks), selected_cases=len(selected),
                      diagnostic_subset=subset, complete_full_dataset=report["complete"] and not subset,
                      run_configuration_sha256=version_hash)
        return report

    episode_path = output / "episodes.jsonl"
    with episode_path.open("x", encoding="utf-8") as handle:
        for task in selected:
            case = case_index[task["task_id"]]
            for mode in modes:
                step_dir = output / "steps" / hashlib.sha256(case["task_id"].encode()).hexdigest()[:24] / mode
                try:
                    if args.policy in {"reference", "wrong-menu"}:
                        policy = (reference if args.policy == "reference" else wrong)(case, menus)
                    else:
                        context = {"protocol": PROTOCOL, "mode": mode,
                                   "public_parameter_bank": copy.deepcopy(bank),
                                   "menu_builder": menu_type(copy.deepcopy(bank))}
                        policy = OfferedChoicePolicy(plugin_factory(context), menus, interface_error)
                    row = run_case(case, policy, mode=mode, menu_builder=menus,
                                   on_step=lambda step: _write_json(step_dir / f"{step['decision_index']:04d}.json", step))
                    row.update(status="ok", workflow_id=_metadata(case, "workflow_id"),
                               condition=_metadata(case, "condition"), policy=args.policy,
                               policy_identity=copy.deepcopy(getattr(policy, "identity", {})),
                               run_configuration_sha256=version_hash)
                    _validate_episode(row)
                except Exception as error:
                    row = {"task_id": task["task_id"], "mode": mode, "status": "infrastructure_error",
                           "error": f"{type(error).__name__}: {error}", "run_configuration_sha256": version_hash}
                episodes.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                # Preserve the complete selected cohort even during a partial run.
                summary = summarize_run()
                _write_json(output / "summary.json", summary)
                print(json.dumps({"completed": summary["completed_episodes"],
                                  "expected": summary["expected_episodes"], "last_status": row["status"]}), flush=True)
    return summarize_run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scorer = subparsers.add_parser("score", help="summarize native evaluator episode records without loading a simulator")
    scorer.add_argument("--episodes", required=True, type=Path)
    scorer.add_argument("--tasks", type=Path, default=DATA_ROOT / "evaluator/case-specs.jsonl")
    scorer.add_argument("--mode", choices=(*MODES, "both"), default="both")
    scorer.add_argument("--output", type=Path)
    runner = subparsers.add_parser("run", help="interactive native simulation; choices change after each action feedback")
    runner.add_argument("--policy", default="reference", help="reference or wrong-menu privileged diagnostic, or module:factory")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--tasks", type=Path, default=DATA_ROOT / "cases.jsonl")
    runner.add_argument("--cases", type=Path, default=DATA_ROOT / "evaluator/case-specs.jsonl")
    runner.add_argument("--presets", type=Path, default=DATA_ROOT / "action-presets.json")
    runner.add_argument("--mode", choices=(*MODES, "both"), default="both")
    runner.add_argument("--limit", type=int, help="positive diagnostic subset; never label subset results as full120")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _require(args.limit is None or args.limit > 0, "--limit must be positive")
            report = run(args)
        else:
            if args.output:
                _require(args.output.resolve() not in {args.episodes.resolve(), args.tasks.resolve()},
                         "--output must not overwrite an input file")
            modes = MODES if args.mode == "both" else (args.mode,)
            report = score_episodes(load_jsonl(args.tasks), load_jsonl(args.episodes), modes)
            if args.output:
                _write_json(args.output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, ImportError, AttributeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
