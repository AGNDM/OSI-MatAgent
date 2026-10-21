"""Release I/O and the explicit, label-free solver input boundary."""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def expand_text_record(record):
    """Reverse lossless text packing of readings and repeated error docs."""
    if "error_definitions" not in record.get("contract", {}) and "readings" not in record.get("observation", {}):
        return record
    record = copy.deepcopy(record)
    contract = record.get("contract", {})
    definitions = contract.pop("error_definitions", {})
    for capability in contract.get("capabilities", {}).values():
        errors = capability.get("errors")
        if isinstance(errors, list):
            capability["errors"] = [copy.deepcopy(definitions[error["error_ref"]])
                                     if isinstance(error, dict) and set(error) == {"error_ref"} else error
                                     for error in errors]
    observation = record.get("observation", {})
    if "readings" in observation:
        columns = observation.pop("columns")
        observation["observations"] = {key: dict(zip(columns, values))
                                        for key,values in observation.pop("readings").items()}
    return record


def solver_view(task, interactive=False):
    """Deterministic, label-free, common input projection for every solver.

    Keep every observation value/quality/time and relevant native input schemas.
    For menu decisions retrieve docs for the union of ALL proposed candidates,
    independent of correctness. Interactive control exposes all public tools.
    Full native contracts remain in the release for inspection and replay.
    """
    task = expand_text_record(task)
    fields = ("instrument_id", "question", "observation", "history", "options",
              "manual", "contract", "allowed_actions", "budget", "goal", "constraints",
              "observation_age_s", "observation_policy")
    view = {key: task[key] for key in fields if key in task}
    contract = view.get("contract", {})
    options = view.get("options", [])
    steps = [s for option in options for s in option.get("steps", [])]
    steps += list(view.get("goal", {}).get("proposed_steps", []))
    referenced = {s["capability_id"] for s in steps if s.get("kind") == "invoke"}
    capabilities = contract.get("capabilities", {})
    keep_all = interactive or not options
    view["contract"] = {
        "capabilities": {key: {name: value for name, value in cap.items()
                               if name in ("input_schema", "preconditions", "preconditions_zh", "function_zh", "description_zh", "errors", "observes")}
                         for key, cap in capabilities.items() if keep_all or key in referenced},
        "state_slots": {key: {name: value for name, value in slot.items()
                              if name in ("meaning_zh", "reference_conditions") and value is not None}
                        for key, slot in contract.get("state_slots", {}).items()}}
    observation = view.get("observation")
    if isinstance(observation, dict) and "observations" in observation:
        # Request/instance identifiers and transaction counters carry no sensor
        # information. All physical sampling values and timestamps are retained.
        view["observation"] = {key: value for key, value in observation.items()
                               if key in ("clock", "observations", "error")}
    if isinstance(view.get("goal"), dict):
        # Held-out wrappers can refer to the scenario-specific objective.
        # Preserve it even when a training phrasing repeats the same sentence.
        view["goal"] = dict(view["goal"])
    view["input_protocol"] = "public-menu-union-v2" if not interactive else "public-tools-v2"
    return view


def task_text(task, include_options=True):
    view = solver_view(task)
    if not include_options:
        view.pop("options", None)
    observation = view.get("observation", {})
    if "observations" in observation:
        # A columnar text serialization removes repeated key names without
        # dropping values, units, quality flags or physical sample timestamps.
        columns = ("value", "unit", "quality", "sample_time_s", "sample_window_s")
        view["observation"] = {
            **{k:v for k,v in observation.items() if k != "observations"},
            "columns": list(columns),
            "readings": {key: [slot.get(column) for column in columns]
                         for key,slot in observation["observations"].items()}}
    # Native capability docs repeat the same common errors many times. A
    # shared dictionary preserves every field and per-capability error order
    # while fitting the same evidence into shorter native model contexts.
    caps = view.get("contract", {}).get("capabilities", {})
    unique = {json.dumps(error, sort_keys=True, ensure_ascii=False, allow_nan=False): error
              for cap in caps.values() for error in (cap.get("errors") or [])}
    if unique:
        references = {key: f"E{index}" for index,key in enumerate(sorted(unique))}
        view["contract"]["error_definitions"] = {references[key]: unique[key] for key in sorted(unique)}
        for cap in caps.values():
            if isinstance(cap.get("errors"), list):
                cap["errors"] = [{"error_ref": references[json.dumps(error, sort_keys=True, ensure_ascii=False, allow_nan=False)]}
                                 for error in cap["errors"]]
    return json.dumps(view, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def option_text(option):
    return json.dumps({key: option[key] for key in ("text", "steps") if key in option},
                      sort_keys=True, ensure_ascii=False)


def load_release(directory):
    directory = Path(directory)
    tasks = read_jsonl(directory / "tasks.jsonl")
    oracles = read_jsonl(directory / "private" / "oracles.jsonl")
    task_ids = [task["task_id"] for task in tasks]
    oracle_ids = [oracle["task_id"] for oracle in oracles]
    if len(set(task_ids)) != len(task_ids) or len(set(oracle_ids)) != len(oracle_ids):
        raise ValueError("Release contains duplicate task/oracle identifiers")
    if set(task_ids) != set(oracle_ids):
        raise ValueError("Every public task must have exactly one private oracle")
    return tasks, {oracle["task_id"]: oracle for oracle in oracles}
