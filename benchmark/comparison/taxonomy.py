"""Evidence-backed instrument mechanism labels; frozen task IDs stay unchanged.

These are evaluator-side analysis labels, never an additional model input.
They describe authored simulator conditions, not verified physical incidents.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

VERSION = "instrument-mechanisms-v1"
MECHANISMS = {
    "interlock_precondition": "Interlocks and operating prerequisites",
    "configuration_argument": "Configuration and argument validity",
    "timing_interruption": "Timing, completion and interruptions",
    "actuator_motor": "Actuator and motor operation",
    "sensor_calibration": "Sensors, readings and calibration",
    "power_supply": "Power and electrical supply",
    "lighting_optical": "Lighting and illumination",
    "fluid_flow": "Fluid flow and delivery",
    "thermal_control": "Thermal operation",
    "electrochemical_cell": "Electrochemical cell hardware",
    "communication_software": "Communication and software state",
    "healthy_control": "Healthy control without a declared target mechanism",
    "mixed": "Multiple explicitly declared mechanisms",
    "unknown": "Mechanism not established by available metadata",
}
FAULT_FAMILIES = {
    "precondition": "interlock_precondition", "interlock": "interlock_precondition",
    "argument_schema": "configuration_argument", "configuration": "configuration_argument",
    "operator_interruption": "timing_interruption", "timing": "timing_interruption",
    "motor_failure": "actuator_motor", "motor_blocked": "actuator_motor",
    "timed_motor_failure": "actuator_motor", "sensor_bias": "sensor_calibration",
    "sensor_failure": "sensor_calibration", "sensor_damage": "sensor_calibration",
    "sensor_rupture": "sensor_calibration", "supply": "power_supply",
    "power_loss": "power_supply", "mains_outage": "power_supply",
    "lighting": "lighting_optical", "lamp_failure": "lighting_optical",
    "cell_failure": "electrochemical_cell", "door_interruption": "timing_interruption",
    "operator_standby": "timing_interruption",
}
# Healthy and failed members of an authored pair retain the same target
# mechanism. Conditions remain a separate axis, so healthy hardware is never
# reported as a failure merely because a workflow requires several steps.
WORKFLOWS = {
    "furnace_program": "timing_interruption",
    "oven_usb_recording": "configuration_argument",
    "hotplate_unlock_stirring": "interlock_precondition",
    "pump_delivery": "fluid_flow",
    "hood_illumination": "lighting_optical",
    "pump_timed_monitoring": "actuator_motor",
    "incubator_sensor_cover_access": "interlock_precondition",
    "hydrogen_startup_diagnosis": "electrochemical_cell",
    "dewpoint_offline_calibration": "sensor_calibration",
    "hood_sash_access_resecure": "interlock_precondition",
    "furnace_door_pause_recovery": "timing_interruption",
    "oven_standby_restoration": "timing_interruption",
    "hotplate_motor_diagnostic": "actuator_motor",
    "dewpoint_startup_measurement_diagnosis": "sensor_calibration",
}


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Taxonomy construction and validation require a Slurm compute allocation")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _steps(record):
    for field in ("setup_steps", "scheduled_events"):
        for index, step in enumerate(record.get(field, [])):
            action = step.get("action", step.get("step", step))
            if isinstance(action, dict):
                yield f"{field}[{index}]", action


def _evidence(source, path, value, mechanism):
    return {"source": source, "path": path, "value": value, "mechanism": mechanism}


def _state_mechanisms(state_id):
    roots = {
        "door": "interlock_precondition", "sash": "interlock_precondition",
        "keylock": "interlock_precondition", "installation": "interlock_precondition",
        "power": "power_supply", "mains": "power_supply", "supply": "power_supply",
        "program": "timing_interruption", "timer": "timing_interruption",
        "calibration": "sensor_calibration", "sensor": "sensor_calibration",
        "monitor": "sensor_calibration", "reading": "sensor_calibration",
        "rh_sensor": "sensor_calibration", "lamp": "lighting_optical",
        "light": "lighting_optical", "lighting": "lighting_optical",
        "motor": "actuator_motor", "stirring": "actuator_motor",
        "pressure": "fluid_flow", "flow": "fluid_flow", "dispense": "fluid_flow",
        "thermal": "thermal_control", "temperature": "thermal_control",
        "usb": "configuration_argument", "serial": "communication_software",
    }
    return roots.get(state_id.split(".", 1)[0]) if isinstance(state_id, str) else None


def classify_mcqa(task, oracle=None):
    oracle = oracle or {}
    evidence, declared = [], []
    family = oracle.get("fault_family")
    if family in FAULT_FAMILIES:
        declared.append(FAULT_FAMILIES[family])
        evidence.append(_evidence("mcqa_oracle", "fault_family", family, declared[-1]))
    if task.get("decision_kind") == "argument_validity":
        # A generic default 'precondition' tag does not turn a schema question
        # into a hardware interlock case.
        declared = ["configuration_argument"]
        evidence.append(_evidence("mcqa_task", "decision_kind", "argument_validity", declared[0]))
    injected = []
    for path, step in _steps(oracle):
        if step.get("capability_id") == "fault.inject":
            name = step.get("arguments", {}).get("fault")
            if name:
                injected.append(name)
                mechanism = FAULT_FAMILIES.get(name)
                if mechanism:
                    evidence.append(_evidence("mcqa_oracle", path + ".arguments.fault", name, mechanism))
    supplementary = []
    for index, cause in enumerate(oracle.get("causes", [])):
        state_id = cause.get("state_id") if isinstance(cause, dict) else None
        mechanism = _state_mechanisms(state_id)
        if mechanism:
            supplementary.append(mechanism)
            evidence.append(_evidence("mcqa_oracle", f"causes[{index}].state_id", state_id, mechanism))
    mechanisms = sorted(set(declared + supplementary + [FAULT_FAMILIES[name] for name in injected if name in FAULT_FAMILIES]))
    primary = declared[0] if len(set(declared)) == 1 else "mixed" if declared else None
    if primary is None:
        inferred = sorted(set(FAULT_FAMILIES[name] for name in injected if name in FAULT_FAMILIES))
        primary = inferred[0] if len(inferred) == 1 else "mixed" if inferred else "unknown"
    if oracle.get("healthy_control") is True:
        condition = "healthy_control"
        if primary == "unknown":
            primary = "healthy_control"
    elif oracle.get("expected_information_request") and oracle.get("information_witness"):
        condition = "ambiguous_evidence"
    elif injected:
        condition = "injected_failure"
    elif oracle:
        condition = "authored_operating_condition"
    else:
        condition = "unknown"
    return {"task_id": task["task_id"], "track": "mcqa", "primary_failure_mode": primary,
            "mechanisms": mechanisms or [primary], "condition": condition,
            "authored_fault_family": family, "injected_faults": injected,
            "legacy_level": task.get("difficulty"), "instrument_id": task.get("instrument_id"),
            "group_id": task.get("group_id", task["task_id"]),
            "evidence": evidence}


def classify_sequential(case):
    expansion = case.get("expansion", {})
    workflow, condition = case.get("workflow_id"), expansion.get("condition")
    primary = WORKFLOWS.get(workflow, FAULT_FAMILIES.get(condition, "unknown"))
    evidence = [_evidence("sequential_case", "workflow_id", workflow, primary)]
    mechanisms, injected = {primary}, []
    for path, step in _steps(case):
        if step.get("capability_id") == "fault.inject":
            name = step.get("arguments", {}).get("fault")
            if name:
                injected.append(name)
                mechanism = FAULT_FAMILIES.get(name)
                if mechanism:
                    mechanisms.add(mechanism)
                    evidence.append(_evidence("sequential_case", path + ".arguments.fault", name, mechanism))
    for index, checkpoint in enumerate(case.get("checkpoints", [])):
        for criterion in checkpoint.get("accept", {}).get("conditions", []):
            mechanism = _state_mechanisms(criterion.get("state_id"))
            if mechanism:
                mechanisms.add(mechanism)
                evidence.append(_evidence("sequential_case", f"checkpoints[{index}].accept.conditions.state_id",
                                          criterion["state_id"], mechanism))
    if condition == "healthy":
        status = "healthy_control"
    elif injected or condition in {"cell_failure", "motor_blocked", "timed_motor_failure", "sensor_damage", "lamp_failure"}:
        status = "injected_failure"
    elif condition:
        status = "authored_operating_condition"
    else:
        status = "unknown"
    return {"task_id": case["task_id"], "track": "sequential", "primary_failure_mode": primary,
            "mechanisms": sorted(mechanisms), "condition": status,
            "authored_condition": condition, "injected_faults": injected,
            "workflow_id": workflow, "instrument_id": case.get("instrument_id"),
            "pair_id": expansion.get("pair_id"),
            "group_id": expansion.get("split_group", workflow or case.get("group_id", case["task_id"])),
            "legacy_level": case.get("difficulty"), "evidence": evidence}


def build_taxonomy(bundle, mcqa_oracles=None):
    bundle = Path(bundle).resolve()
    tasks = load_jsonl(bundle / "mcqa_388/questions.jsonl.gz")
    cases = load_jsonl(bundle / "sequential_120/evaluator/case-specs.jsonl")
    answers = {row["task_id"]: row["correct_option_ids"] for row in load_jsonl(bundle / "mcqa_388/answers.jsonl")}
    manifest = json.loads((bundle / "mcqa_388/manifest.json").read_text())
    oracle_index, source = {}, None
    if mcqa_oracles:
        digest = sha256(mcqa_oracles)
        if digest != manifest["original_oracles_sha256"]:
            raise ValueError("MCQA oracle bytes do not match the frozen release source hash")
        rows = load_jsonl(mcqa_oracles)
        oracle_index = {row["task_id"]: row for row in rows}
        if len(oracle_index) != len(rows) or set(oracle_index) != set(answers):
            raise ValueError("MCQA oracle IDs do not exactly cover the frozen release")
        if any(set(oracle_index[key]["correct_option_ids"]) != set(value) for key, value in answers.items()):
            raise ValueError("MCQA oracle answers differ from the frozen answer key")
        source = {"sha256": digest, "matched_frozen_oracle_hash": True}
    labels = [classify_mcqa(task, oracle_index.get(task["task_id"])) for task in tasks]
    labels += [classify_sequential(case) for case in cases]
    # Keep a small evaluator-side metadata projection for research metrics. It
    # replaces any dependence on the large original execution traces.
    metadata = {key: {field: row.get(field) for field in (
        "fault_family", "family_id", "base_case_id", "healthy_control", "label_class",
        "expected_information_request", "evidence_condition", "causes")}
        for key, row in oracle_index.items()}
    return {"schema_version": 1, "taxonomy_version": VERSION,
            "mcqa_questions_sha256": sha256(bundle / "mcqa_388/questions.jsonl.gz"),
            "sequential_cases_sha256": sha256(bundle / "sequential_120/cases.jsonl"),
            "oracle_source": source, "mechanism_labels": MECHANISMS,
            "counts": {track: dict(Counter(row["primary_failure_mode"] for row in labels if row["track"] == track))
                       for track in ("mcqa", "sequential")},
            "condition_counts": {track: dict(Counter(row["condition"] for row in labels if row["track"] == track))
                                 for track in ("mcqa", "sequential")},
            "definitions": {"primary_failure_mode": "Authored target mechanism; healthy members retain their matched fault-family mechanism.",
                            "mechanisms": "Overlapping declared mechanisms and prerequisite state axes; not counts of independent failures.",
                            "condition": "Healthy controls, injected failures and other authored operating conditions are separate.",
                            "legacy_level": "Original identifier preserved; L1-L5 are not an empirically established difficulty ordering.",
                            "validity": "Synthetic simulator conditions; independent instrument-expert review is pending."},
            "labels": labels, "mcqa_evaluator_metadata": metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mcqa-oracles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_allocation()
    if args.output.exists():
        parser.error("--output must name a new file")
    result = build_taxonomy(args.bundle, args.mcqa_oracles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
