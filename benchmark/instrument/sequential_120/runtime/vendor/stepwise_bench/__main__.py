"""Slurm-only CLI for assisted step decisions and autonomous rollouts."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import uuid

from .policies import (AlwaysWrongPolicy, LocalActionPolicy, ResumeMismatchError,
                       fingerprint, require_allocation)

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = "instrument-stepwise-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def source_configuration(tasks, args):
    require_allocation()
    paths = sorted((PROJECT / "stepwise_bench").glob("*.py"))
    paths += sorted((PROJECT / "instrument_bench").glob("*.py"))
    paths += [PROJECT / "environment/requirements-models.lock.txt",
              PROJECT / "environment/local-models.json", PROJECT / "environment/chemlab-source.json"]
    paths += [path for path in [PROJECT / "slurm/stepwise-evaluation.sbatch"] if path.exists()]
    source = {str(path.relative_to(PROJECT)): digest(path) for path in paths}
    native_root = PROJECT / "external/ChemLab-Assets"
    native_paths = sorted((native_root / "scripts").rglob("*.py"))
    native_paths += sorted((native_root / "assets").rglob("*.json"))
    native_paths += [native_root / "catalog/catalog.csv"]
    if not native_paths or not (native_root / "scripts/basic40_runtime_thermal.py").is_file():
        raise ValueError("Native ChemLab runtime/data is missing")
    native = {str(path.relative_to(PROJECT)): digest(path) for path in native_paths}
    model = None
    if args.policy == "qwen":
        pinned = [record for record in json.loads((PROJECT / "environment/local-models.json").read_text())
                  if record.get("repository") == "Qwen/Qwen2.5-3B-Instruct"]
        if len(pinned) != 1 or pinned[0].get("revision") != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
            raise ValueError("Expected the declared pinned Qwen model in environment/local-models.json")
        expected = (PROJECT / pinned[0]["path"]).resolve()
        if Path(args.model_path).resolve() != expected:
            raise ValueError("This pilot permits only the pinned Qwen model path; custom models need a separate identity implementation")
        model_paths = sorted(path for path in expected.iterdir() if path.is_file() and
                             (path.suffix in {".json", ".safetensors", ".txt", ".model"}))
        if not any(path.suffix == ".safetensors" for path in model_paths):
            raise ValueError("Pinned Qwen checkpoint weights are missing")
        model = {"repository": pinned[0]["repository"], "revision": pinned[0]["revision"],
                 "files_sha256": {path.name: digest(path) for path in model_paths}}
    return {"protocol": PROTOCOL, "source_sha256": source,
            "native_chemlab_sha256": native, "local_model": model,
            "task_specs_sha256": fingerprint(tasks), "case_ids": [case["task_id"] for case in tasks],
            "policy": args.policy, "modes": args.modes, "seed": args.seed,
            "device": args.device if args.policy == "qwen" else None,
            "model_path": str(Path(args.model_path).resolve()) if args.policy == "qwen" else None,
            "max_context": args.max_context if args.policy == "qwen" else None,
            "max_new_tokens": args.max_new_tokens if args.policy == "qwen" else None,
            "privileged_diagnostic": args.policy == "reference",
            "primary_claim": "stepwise conditional decision accuracy" if args.modes == ["teacher_forced"]
                             else "report assisted and autonomous outcomes separately"}


class DurablePolicy:
    """Journal policy proposals before execution and replay them on resume.

    Model calls are cached by exact public context/history. The simulator is
    reconstructed on resume; cached proposals reproduce each executed prefix.
    Unknown interrupted call cost is explicitly retained in intent files.
    """

    def __init__(self, factory, directory, *, privileged=False):
        self.factory, self.directory = factory, Path(directory)
        self.privileged = bool(privileged)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.inner = None
        self.position = 0
        self.last_action_usage = {}
        self.replayed_calls = 0
        self.new_calls = 0

    def next_action(self, public_view, history):
        position = self.position
        self.position += 1
        public_hash = fingerprint({"public_view": public_view, "history": history})
        path = self.directory / f"call-{position:04d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("position") != position or saved.get("public_input_sha256") != public_hash:
                raise ResumeMismatchError("Resume public evidence/history differs; select a fresh output directory")
            self.last_action_usage = copy.deepcopy(saved["usage"])
            self.replayed_calls += 1
            if saved.get("error"):
                raise RuntimeError("Cached policy failure: " + saved["error"]["detail"])
            return copy.deepcopy(saved["action"])
        call_id = uuid.uuid4().hex
        intent_path = self.directory / f"intent-{call_id}.json"
        write_json(intent_path, {"call_id": call_id, "position": position,
                                "public_input_sha256": public_hash,
                                "status": "started", "job_id": os.environ.get("SLURM_JOB_ID"),
                                "node": os.environ.get("SLURMD_NODENAME"),
                                "time_unix_s": time.time()}, private=True)
        started = time.perf_counter()
        try:
            if self.inner is None:
                self.inner = self.factory()
            action = self.inner.next_action(copy.deepcopy(public_view), copy.deepcopy(history))
            self.last_action_usage = copy.deepcopy(getattr(self.inner, "last_action_usage", {}))
            saved = {"action": action, "error": None}
        except Exception as error:
            self.last_action_usage = copy.deepcopy(getattr(self.inner, "last_action_usage", {}))
            saved = {"action": None, "error": {"type": type(error).__name__, "detail": str(error)}}
        saved.update(position=position, call_id=call_id, public_input_sha256=public_hash,
                     usage=self.last_action_usage, call_wall_time_s=time.perf_counter() - started,
                     slurm_job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"),
                     policy_identity=copy.deepcopy(getattr(self.inner, "identity", {})))
        write_json(path, saved, private=True)
        self.new_calls += 1
        write_json(intent_path, {"call_id": call_id, "position": position, "status": "committed",
                                "record": path.name, "public_input_sha256": public_hash}, private=True)
        if saved["error"]:
            raise RuntimeError(saved["error"]["detail"])
        return copy.deepcopy(saved["action"])

    def accounting(self):
        intents = [json.loads(path.read_text()) for path in self.directory.glob("intent-*.json")]
        calls = [json.loads(path.read_text()) for path in self.directory.glob("call-*.json")]
        committed_ids = {row["call_id"] for row in calls}
        # A process may die after writing a complete call record but before
        # changing its intent; those calls have known usage and are committed.
        unknown = [row for row in intents if row["call_id"] not in committed_ids]
        usages = [row["usage"] for row in calls]
        return {"cached_proposals_replayed_this_process": self.replayed_calls,
                "new_proposals_this_process": self.new_calls,
                "committed_policy_calls": len(calls), "uncommitted_call_intents": len(unknown),
                "uncommitted_usage_known": not unknown,
                "known_input_tokens": sum(usage.get("input_tokens", 0) for usage in usages),
                "known_output_tokens": sum(usage.get("output_tokens", 0) for usage in usages),
                "input_tokens_complete": not unknown and all(usage.get("input_tokens_known", False) for usage in usages),
                "output_tokens_complete": not unknown and all(usage.get("output_tokens_known", False) for usage in usages),
                "known_gpu_seconds": sum(usage.get("gpu_seconds", 0) or 0 for usage in usages),
                "known_api_cost_usd": sum(usage.get("api_cost_usd", 0) or 0 for usage in usages),
                "committed_policy_call_wall_time_s": sum(row.get("call_wall_time_s", 0) for row in calls),
                "unknown_usage_explanation": "Interrupted calls may have consumed additional tokens/time" if unknown else None}


def summarize(reports, expected, policy):
    by_mode = {}
    for mode in ("teacher_forced", "free_run"):
        rows = [row for row in reports if row.get("mode") == mode]
        if not rows:
            continue
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("group_id") or row["case_id"]].append(row)

        def group_macro(field):
            values = []
            for members in groups.values():
                scores = [float(row[field]) for row in members if row.get(field) is not None]
                if scores:
                    values.append(sum(scores) / len(scores))
            return sum(values) / len(values) if values else None

        stage_groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            for stage, value in row.get("stage_accuracy", {}).items():
                stage_groups[stage][row.get("group_id") or row["case_id"]].append(value)
        stages = {stage: sum(sum(scores) / len(scores) for scores in members.values()) / len(members)
                  for stage, members in stage_groups.items()}
        # The paired hood variants share a workflow group, so that workflow
        # contributes the same macro weight as a single-variant workflow.
        by_mode[mode] = {"episodes": len(rows),
                         "workflow_groups": len(groups),
                         "assisted": mode == "teacher_forced",
                         "interpretation": "conditional decisions with evaluator corrections" if mode == "teacher_forced"
                                           else "uncorrected agent rollout",
                         "checkpoint_accuracy_group_macro": group_macro("checkpoint_accuracy"),
                         "stage_accuracy_group_macro": stages,
                         "machine_choice_accuracy_group_macro": group_macro("machine_choice_correct"),
                         "autonomous_success_group_macro": group_macro("autonomous_success") if mode == "free_run" else None,
                         "corrected_outcome_group_macro": group_macro("corrected_trajectory_outcome_met") if mode == "teacher_forced" else None,
                         "all_original_decisions_admissible_group_macro": group_macro("all_original_decisions_admissible"),
                         "interventions": sum(row.get("interventions", 0) for row in rows),
                         "decisions": sum(row.get("decisions", 0) for row in rows),
                         "query_attempts": sum(row.get("query_attempts", 0) for row in rows),
                         "query_valid": sum(row.get("query_valid", 0) for row in rows),
                         "false_completion_claims": sum(bool(row.get("false_completion_claim")) for row in rows),
                         "known_input_tokens": sum(row.get("durability", {}).get("known_input_tokens", 0) for row in rows),
                         "known_output_tokens": sum(row.get("durability", {}).get("known_output_tokens", 0) for row in rows),
                         "committed_policy_call_wall_time_s": sum(row.get("durability", {}).get("committed_policy_call_wall_time_s", 0) for row in rows),
                         "unknown_interrupted_call_intents": sum(row.get("durability", {}).get("uncommitted_call_intents", 0) for row in rows),
                         "engine_summaries": [{key: row.get(key) for key in (
                             "case_id", "group_id", "checkpoint_accuracy", "stage_accuracy",
                             "machine_choice_correct", "autonomous_success", "corrected_trajectory_outcome_met",
                             "all_original_decisions_admissible", "interventions", "decisions", "termination")}
                             for row in rows]}
    return {"protocol": PROTOCOL, "policy": policy, "privileged_diagnostic": policy == "reference",
            "expected_episodes": expected, "completed_episodes": len(reports),
            "complete": len(reports) == expected, "modes": by_mode,
            "expert_review_completed": False,
            "scope": "prototype curriculum; not a validated replacement for the frozen MCQ release"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Separate assisted step decisions and autonomous ChemLab rollouts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=("reference", "always-wrong", "qwen"), default="reference")
    parser.add_argument("--modes", nargs="+", choices=("teacher_forced", "free_run"),
                        default=["teacher_forced", "free_run"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default="external/models/qwen2.5-3b")
    parser.add_argument("--max-context", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    require_allocation()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if len(args.modes) != len(set(args.modes)):
        parser.error("--modes must not contain duplicates")
    from .engine import ReferencePolicy, run_case
    from .tasks import build_tasks
    tasks = build_tasks(seed=args.seed)
    if args.cases:
        available = {case["task_id"] for case in tasks}
        if set(args.cases) - available:
            parser.error("Unknown --cases: " + ", ".join(sorted(set(args.cases) - available)))
        tasks = [case for case in tasks if case["task_id"] in args.cases]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks or len({case["task_id"] for case in tasks}) != len(tasks):
        raise ValueError("Expected a nonempty curriculum with unique case IDs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another process is already using this output directory") from None
    configuration = source_configuration(tasks, args)
    config_path = output / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Source, task, or policy configuration changed; use a fresh output directory")
    elif any(path.name != ".lock" for path in output.iterdir()):
        raise ValueError("Refusing a nonempty output directory without matching provenance")
    else:
        write_json(config_path, configuration)
    private = output / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    private.chmod(0o700)
    write_json(private / "task-specs.json", tasks, private=True)
    write_json(output / "hardware.json", {"node": platform.node(), "python": platform.python_version(),
               "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_node": os.environ["SLURMD_NODENAME"]})
    shared = {}

    def model_factory():
        if "model" not in shared:
            shared["model"] = LocalActionPolicy(model_path=args.model_path, device=args.device,
                                                max_context=args.max_context,
                                                max_new_tokens=args.max_new_tokens)
        return shared["model"]

    reports = []
    for case in tasks:
        # Opaque content-addressed directories avoid interpreting user strings as
        # paths and keep fixture labels out of policy context.
        case_key = fingerprint(case)[:24]
        for mode in args.modes:
            episode_dir = private / case_key / mode
            episode_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            result_path = episode_dir / "episode.json"
            if result_path.exists():
                episode = json.loads(result_path.read_text())
            else:
                factory = (lambda case=case: ReferencePolicy(case)) if args.policy == "reference" else (
                    AlwaysWrongPolicy if args.policy == "always-wrong" else model_factory)
                policy = DurablePolicy(factory, episode_dir / "calls", privileged=args.policy == "reference")
                step_count = 0

                def on_step(record):
                    nonlocal step_count
                    write_json(episode_dir / "steps" / f"{step_count:04d}.json", record, private=True)
                    step_count += 1

                episode = run_case(case, policy, mode=mode, on_step=on_step)
                episode.update(case_id=case["task_id"], group_id=case.get("group_id"), mode=mode,
                               policy=args.policy, privileged_diagnostic=args.policy == "reference",
                               durability=policy.accounting(),
                               latency_measurement_note="Engine policy latency includes any cached proposal replay; grader latency is separate. Durable call wall time retains original committed policy-call time.")
                write_json(result_path, episode, private=True)
            reports.append(episode)
            write_json(output / "summary.json", summarize(reports, len(tasks) * len(args.modes), args.policy))
            print(json.dumps({"case_id": case["task_id"], "mode": mode,
                              "completed_episodes": len(reports),
                              "expected_episodes": len(tasks) * len(args.modes)}, sort_keys=True), flush=True)
    write_json(private / "episodes.json", reports, private=True)
    summary = summarize(reports, len(tasks) * len(args.modes), args.policy)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
