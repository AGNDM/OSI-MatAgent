"""Export the frozen MCQA public evidence without model-specific instructions.

The three projection functions below are copied from instrument_bench/io.py.
They preserve the public-menu-union-v2 input used by the recorded Qwen runs.
Answers and evaluation metadata are deliberately excluded from model prompts.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from .common import load_jsonl
except ImportError:
    from common import load_jsonl


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


def export_prompts(questions, output, split="heldout"):
    """Write task IDs alongside canonical evidence text; never open answer keys."""
    ids = [task.get("task_id") for task in questions]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise ValueError("Every question requires a nonempty task_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question task_id")
    splits = {task.get("split") for task in questions}
    if split not in {"all", "heldout"} and split not in splits:
        raise ValueError(f"Unknown split: {split}")
    if split == "all":
        selected = questions
    elif split == "heldout":
        selected = [task for task in questions if str(task.get("split", "")).startswith("test")]
    else:
        selected = [task for task in questions if task.get("split") == split]
    if not selected:
        raise ValueError(f"No questions selected for split: {split}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in selected:
            record = {"task_id": task["task_id"], "prompt": task_text(task)}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return len(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export canonical public MCQA evidence as task_id/prompt JSONL. "
                    "Add your model's system prompt, response instructions and chat template separately; "
                    "this command does not reproduce Qwen's sampling/voting procedure.")
    parser.add_argument("--questions", type=Path,
                        default=Path(__file__).resolve().parents[1] / "mcqa_388" / "questions.jsonl.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="heldout",
                        help="heldout (all test* splits; default), all, or an exact dataset split")
    args = parser.parse_args(argv)
    if args.questions.resolve() == args.output.resolve():
        parser.error("--output must differ from --questions")
    try:
        count = export_prompts(load_jsonl(args.questions), args.output, args.split)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"prompts": count, "split": args.split, "output": str(args.output),
                      "input_protocol": "public-menu-union-v2"}))


if __name__ == "__main__":
    main()
