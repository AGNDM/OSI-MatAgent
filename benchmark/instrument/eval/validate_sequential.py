"""Static integrity checks for the dynamic, native sequential benchmark.

This module uses only the Python standard library. It authenticates the frozen
exports and checks their public/evaluator boundary; it never runs a simulator,
loads a model, or treats the checkpoint references as fixed answer strings.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


EXPECTED_COUNTS = {"cases": 120, "instruments": 8, "workflow_templates": 14,
                   "public_commands": 146, "schema_validated_argument_presets": 1954}
_FORBIDDEN_COMMANDS = {"fault.inject", "plant.configure", "truth.inspect", "sim.step"}
_PRIVATE_FIELDS = {"case_id", "task_id", "group_id", "workflow_id", "seed", "expansion",
                   "setup_steps", "scheduled_events", "fixture_conditions", "goal_conditions",
                   "escalation_conditions", "escalation_expected", "checkpoints", "reference_action",
                   "canonical_actions", "correct_option_ids", "answer", "oracle", "fault_family",
                   "fault_type", "split", "split_group", "difficulty", "tier"}
_VIEW_FIELDS = {"allowed_action_kinds", "budget", "contract", "elapsed_sim_time_s", "goal",
                "machine_catalog", "manual", "mode", "observation", "observation_policy",
                "selected_instrument"}
_ACTION_KINDS = {"select_instrument", "describe", "invoke", "read", "wait", "finish", "escalate"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(bundle, name, files=None):
    _require(isinstance(name, str) and bool(name), "Sequential paths must be nonempty strings")
    parts = PurePosixPath(name)
    _require("\\" not in name and "\x00" not in name and not parts.is_absolute()
             and not PureWindowsPath(name).drive
             and all(part not in {"", ".", ".."} for part in name.split("/")),
             f"Unsafe sequential path: {name!r}")
    path = bundle.joinpath(*parts.parts)
    _require(not any(bundle.joinpath(*parts.parts[:end]).is_symlink()
                     for end in range(1, len(parts.parts) + 1)), f"Sequential symlink: {name}")
    try:
        path.resolve().relative_to(bundle)
    except (ValueError, OSError, RuntimeError) as error:
        raise ValueError(f"Sequential path escapes its bundle: {name}") from error
    _require(path.is_file(), f"Missing sequential file: {name}")
    if files is not None:
        _require(name in files, f"Sequential file is outside hashed inventory: {name}")
    return path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Non-finite JSON value {value} in {path.name}")))


def _lines(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"Blank sequential JSONL line: {path.name}:{index}")
        row = json.loads(line)
        _require(isinstance(row, dict), f"Sequential record must be an object: {path.name}:{index}")
        _canonical(row)  # Reject non-finite values even on permissive JSON decoders.
        rows.append(row)
    return rows


def _number(value, *, positive=False):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _public_boundary(value, location="initial_context"):
    if isinstance(value, dict):
        for key, child in value.items():
            _require(key not in _PRIVATE_FIELDS and not key.startswith("truth."),
                     f"Private evaluator field in public {location}: {key}")
            _public_boundary(child, f"{location}.{key}")
    elif isinstance(value, list):
        for child in value:
            _public_boundary(child, location)


def _schema_accepts(value, schema):
    """Check the value constraints used by the shipped public input schemas.

    Original full JSON-Schema validation is authenticated by the native receipt;
    these stdlib checks catch corrupt parameter objects without that dependency.
    """
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return False
    for keyword, condition in (("anyOf", any), ("allOf", all)):
        if keyword in schema and not condition(_schema_accepts(value, item) for item in schema[keyword]):
            return False
    if "oneOf" in schema and sum(_schema_accepts(value, item) for item in schema["oneOf"]) != 1:
        return False
    if "not" in schema and _schema_accepts(value, schema["not"]):
        return False
    if "const" in schema and _canonical(value) != _canonical(schema["const"]):
        return False
    if "enum" in schema and _canonical(value) not in {_canonical(item) for item in schema["enum"]}:
        return False
    kind = schema.get("type")
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list),
              "string": isinstance(value, str), "boolean": isinstance(value, bool),
              "null": value is None,
              "number": isinstance(value, (int, float)) and not isinstance(value, bool),
              "integer": isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value) and value == int(value)}
    if kind and not any(checks.get(item, False) for item in (kind if isinstance(kind, list) else [kind])):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not set(schema.get("required", [])) <= value.keys():
            return False
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get("maxProperties", math.inf):
            return False
        for key, item in value.items():
            subschema = properties.get(key, schema.get("additionalProperties", True))
            if not _schema_accepts(item, subschema):
                return False
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            return False
        if schema.get("uniqueItems") and len({_canonical(item) for item in value}) != len(value):
            return False
        items = schema.get("items", True)
        for index, item in enumerate(value):
            subschema = (items[index] if index < len(items) else schema.get("additionalItems", True)) if isinstance(items, list) else items
            if not _schema_accepts(item, subschema):
                return False
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            return False
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            return False
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return False
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            return False
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False
        if "multipleOf" in schema and not math.isclose(value / schema["multipleOf"], round(value / schema["multipleOf"]), abs_tol=1e-8):
            return False
    return True


def _validate_bank(bank, archive):
    _require(isinstance(bank, dict) and isinstance(bank.get("__meta__"), dict), "Missing public parameter bank metadata")
    instruments = archive.get("instruments")
    contracts = archive.get("contracts_by_sha256")
    _require(isinstance(instruments, dict) and len(instruments) == 8 and isinstance(contracts, dict),
             "Expected eight public instrument contracts")
    _require(set(bank) - {"__meta__"} == set(instruments), "Parameter bank instrument mismatch")
    _require(set(contracts) == set(instruments.values()), "Unreferenced or missing public contract")
    meta, command_count, preset_count = bank["__meta__"], 0, 0
    _require(set(meta.get("read_sets", {})) == set(instruments), "Read-set instrument mismatch")
    _require(set(meta.get("support_report", {})) == set(instruments), "Support-report instrument mismatch")
    waits = meta.get("wait_durations_s")
    _require(isinstance(waits, list) and bool(waits) and all(_number(value, positive=True) and value <= 86400 for value in waits)
             and waits == sorted(set(waits)), "Invalid shared wait durations")
    resolved = {}
    for machine, digest in instruments.items():
        contract = contracts[digest]
        _require(_fingerprint(contract) == digest, f"Public contract hash mismatch: {machine}")
        caps, slots = contract.get("capabilities"), contract.get("state_slots")
        _require(isinstance(caps, dict) and bool(caps) and isinstance(slots, dict), f"Invalid public contract: {machine}")
        _require(set(bank[machine]) == set(caps), f"Parameter bank command mismatch: {machine}")
        _require(set(meta["support_report"][machine]) == set(caps), f"Support-report command mismatch: {machine}")
        for state_id, spec in slots.items():
            _require(not state_id.startswith("truth.") and spec.get("visibility", "agent") == "agent",
                     f"Private state in public contract: {machine}/{state_id}")
        read_sets = meta["read_sets"][machine]
        _require(isinstance(read_sets, list) and bool(read_sets), f"Missing public read sets: {machine}")
        for reading in read_sets:
            _require(isinstance(reading, list) and bool(reading) and all(isinstance(item, str) for item in reading)
                     and len(reading) == len(set(reading)) and set(reading) <= set(slots), f"Invalid public read set: {machine}")
        for capability, spec in caps.items():
            _require(capability not in _FORBIDDEN_COMMANDS and not capability.startswith("truth.")
                     and spec.get("visibility", "agent") == "agent"
                     and spec.get("availability", "implemented") == "implemented", f"Private or unavailable public command: {capability}")
            values = bank[machine][capability]
            _require(isinstance(values, list) and bool(values) and all(isinstance(value, dict) for value in values),
                     f"Invalid parameter presets: {machine}/{capability}")
            _require(len({_canonical(value) for value in values}) == len(values), f"Duplicate parameter preset: {machine}/{capability}")
            _require(all(_schema_accepts(value, spec.get("input_schema", {"type": "object"})) for value in values),
                     f"Parameter preset violates public input schema: {machine}/{capability}")
            support = meta["support_report"][machine][capability]
            _require(support.get("preset_count") == len(values) and support.get("schema_validated") is True
                     and support.get("exhaustive_parameter_coverage") is False, f"Invalid preset support report: {machine}/{capability}")
            command_count += 1
            preset_count += len(values)
        resolved[machine] = contract
    _require(command_count == 146, f"Public command count mismatch: {command_count}")
    _require(preset_count == 1954, f"Parameter preset count mismatch: {preset_count}")
    return resolved


def _conditions(value, location):
    _require(isinstance(value, list), f"Missing condition list: {location}")
    for condition in value:
        _require(isinstance(condition, dict) and isinstance(condition.get("state_id"), str)
                 and condition.get("operator") in {"eq", "ne", "gt", "ge", "lt", "le", "contains", "in"}
                 and "value" in condition and isinstance(condition.get("path"), list)
                 and _number(condition.get("tolerance")), f"Invalid condition: {location}")


def _validate_cases(public, cases, bank, contracts):
    _require(len(public) == len(cases) == 120, "Sequential case count mismatch: expected 120 public and evaluator records")
    public_ids, case_ids = [row.get("task_id") for row in public], [row.get("task_id") for row in cases]
    _require(all(isinstance(value, str) and value for value in public_ids + case_ids), "Invalid sequential task ID")
    _require(len(set(public_ids)) == 120 and len(set(case_ids)) == 120, "Duplicate sequential task ID")
    _require(set(public_ids) == set(case_ids), "Public and evaluator task IDs do not match")
    by_id = {row["task_id"]: row for row in cases}
    for row in public:
        case = by_id[row["task_id"]]
        _require(set(row) == {"task_id", "initial_context", "action_presets"}, "Unexpected public task metadata")
        _require(row["action_presets"] == "action-presets.json", "Unexpected public action-presets reference")
        view = row["initial_context"]
        _require(isinstance(view, dict) and set(view) == _VIEW_FIELDS, "Unexpected public initial_context fields")
        _public_boundary(view)
        _require(view["goal"] == case.get("goal") and bool(view["goal"]), "Public/evaluator goal mismatch")
        _require(view["manual"] == case.get("manual") and isinstance(view["manual"], list), "Public/evaluator manual mismatch")
        _require(view["selected_instrument"] is None and view["contract"] == {} and view["observation"] == {}
                 and view["elapsed_sim_time_s"] == 0 and view["mode"] == "teacher_forced", "Initial public context is not an unselected zero-time state")
        _require(set(view["allowed_action_kinds"]) == _ACTION_KINDS, "Public action-kind mismatch")
        catalog = view["machine_catalog"]
        _require(isinstance(catalog, list) and len(catalog) == 8
                 and {entry.get("instrument_id") for entry in catalog} == set(contracts), "Public machine catalog mismatch")
        for entry in catalog:
            _require(set(entry) == {"instrument_id", "name", "purpose", "capabilities"}
                     and set(entry["capabilities"]) <= set(contracts[entry["instrument_id"]]["capabilities"]), "Invalid public machine catalog entry")
        budget = case.get("budget", {})
        _require(set(budget) == {"max_actions", "max_decisions", "max_queries", "max_sim_time_s"}
                 and all(type(budget[key]) is int and budget[key] > 0 for key in ("max_actions", "max_decisions", "max_queries"))
                 and _number(budget["max_sim_time_s"], positive=True), "Invalid sequential action/time budget")
        _require(view["budget"] == {**budget, "decisions_used": 0}, "Public/evaluator budget mismatch")
        _require(case.get("case_id") == case["task_id"] and case.get("instrument_id") in contracts, "Invalid evaluator case identity")
        _require(type(case.get("seed")) is int and case["seed"] >= 0, "Missing evaluator seed")
        for field in ("fixture_conditions", "goal_conditions", "escalation_conditions"):
            # Native evaluation treats omitted condition lists as empty. An
            # explicitly supplied null or other non-list value remains invalid.
            _conditions(case.get(field, []), field)
        _require(type(case.get("escalation_expected")) is bool and isinstance(case.get("observation_ids"), list)
                 and isinstance(case.get("setup_steps"), list) and isinstance(case.get("scheduled_events"), list), "Missing evaluator fixtures or schedules")
        for event in case["scheduled_events"]:
            _require(isinstance(event, dict) and _number(event.get("trigger_time_s"))
                     and isinstance(event.get("steps"), list) and bool(event["steps"]), "Invalid scheduled native event")
        nodes = case.get("checkpoints")
        _require(isinstance(nodes, list) and bool(nodes), "Missing evaluator checkpoints")
        ids = [node.get("checkpoint_id") for node in nodes]
        _require(all(isinstance(value, str) and value for value in ids) and len(ids) == len(set(ids)), "Duplicate or invalid checkpoint identity")
        done = set()
        while len(done) < len(ids):
            eligible = {node["checkpoint_id"] for node in nodes if isinstance(node.get("depends_on"), list)
                        and set(node["depends_on"]) <= done}
            _require(bool(eligible - done), "Checkpoint dependency graph has a cycle or missing dependency")
            done.update(eligible)
        for node in nodes:
            _require(isinstance(node.get("stage"), str) and bool(node["stage"]), "Missing checkpoint stage")
            _conditions(node.get("when_conditions", []), "checkpoint branch")
            accept, action = node.get("accept", {}), node.get("reference_action", {})
            _require(accept.get("kind") in {"conditions", "read", "wait", "finish", "escalate"}
                     and isinstance(accept.get("state_ids"), list) and _number(accept.get("min_dt_s")), "Invalid dynamic checkpoint acceptance rule")
            _conditions(accept.get("conditions", []), "checkpoint acceptance")
            kind = {"conditions": "invoke"}.get(accept["kind"], accept["kind"])
            _require(action.get("kind") == kind, "Checkpoint reference/acceptance kind mismatch")
            if kind == "invoke":
                machine = action.get("instrument_id", case["instrument_id"])
                values = bank.get(machine, {}).get(action.get("capability_id"), [])
                _require(_canonical(action.get("arguments", {})) in {_canonical(value) for value in values}, "Checkpoint reference is absent from shared public presets")
            if kind == "wait":
                _require(_number(action.get("dt_s"), positive=True), "Invalid checkpoint wait duration")
        expansion = case.get("expansion", {})
        _require(expansion.get("template_id") == case.get("workflow_id") == expansion.get("split_group")
                 and isinstance(expansion.get("difficulty_axes"), dict)
                 and expansion.get("declared_checkpoint_count") == len(nodes) + 1
                 and expansion.get("difficulty_is_empirically_validated") is False, "Invalid workflow grouping or difficulty provenance")
        _require("split" not in case and "split" not in expansion, "Sequential train/test split has not been assigned")
    instruments = Counter(case["instrument_id"] for case in cases)
    workflows = Counter(case["workflow_id"] for case in cases)
    _require(len(instruments) == 8 and len(workflows) == 14, "Sequential instrument/workflow count mismatch")
    return {"instrument_counts": dict(sorted(instruments.items())), "workflow_counts": dict(sorted(workflows.items()))}


def _validate_runtime(bundle, config, files, manifest):
    receipt_path = _path(bundle, "sequential_120/runtime/provenance.json", files)
    receipt = _json(receipt_path)
    entries = receipt.get("files")
    _require(isinstance(entries, dict) and bool(entries) and receipt.get("file_count") == len(entries),
             "Invalid runtime source inventory")
    expected = {**config.get("native_chemlab_sha256", {}), **config.get("source_sha256", {}),
                **config.get("choice_source_sha256", {})}
    seen, total = set(), 0
    for relative, entry in entries.items():
        path = _path(bundle, f"sequential_120/runtime/vendor/{relative}", files)
        source = entry.get("source_relative_path")
        _require(source == relative and entry.get("modified") is False,
                 f"Runtime source mapping is not an unmodified byte copy: {relative}")
        _require(type(entry.get("bytes")) is int and path.stat().st_size == entry["bytes"]
                 and _digest(path) == entry.get("sha256"), f"Runtime byte or SHA256 mismatch: {relative}")
        if relative in expected:
            _require(entry["sha256"] == expected[relative], f"Runtime differs from frozen source: {relative}")
        elif relative.endswith(".py"):
            raise ValueError(f"Runtime Python source lacks frozen experiment provenance: {relative}")
        seen.add(relative)
        total += entry["bytes"]
    for relative in ("instrument_bench/adapters.py", "stepwise_bench/engine.py", "stepwise_bench/policies.py",
                     "stepwise_choice/engine.py", "stepwise_choice/menus.py", "stepwise_choice/policies.py",
                     "stepwise_expanded/engine.py", "stepwise_expanded/tasks.py"):
        _require(relative in seen, f"Missing required frozen runtime source: {relative}")
    _require(receipt.get("total_bytes") == total, "Runtime source byte total mismatch")
    if "runtime_sources" in manifest:
        mapping = {"sequential_120/runtime/vendor/" + name: {
            "source_path": entry["source_relative_path"], "sha256": entry["sha256"],
            "bytes": entry["bytes"], "modified": False} for name, entry in entries.items()}
        _require(manifest["runtime_sources"] == mapping, "Sequential/runtime source mapping mismatch")
    return len(entries)


def validate_sequential_dataset(bundle, metadata, files=None):
    """Validate the included 120-case dataset and return its static receipt.

    ``files`` is the already authenticated parent-bundle inventory. Its absence
    permits independent validation while preserving source export hash checks.
    """
    bundle = Path(bundle).resolve()
    _require(isinstance(metadata, dict) and metadata.get("data_included") is True, "Sequential data are not included")
    names = {"public": metadata.get("cases_file", "sequential_120/cases.jsonl"),
             "evaluator": metadata.get("evaluator_cases_file", "sequential_120/evaluator/case-specs.jsonl"),
             "bank": metadata.get("action_presets_file", "sequential_120/action-presets.json"),
             "contracts": metadata.get("public_contracts_file", "sequential_120/public-contracts.json"),
             "manifest": metadata.get("manifest_file", "sequential_120/manifest.json")}
    paths = {key: _path(bundle, name, files) for key, name in names.items()}
    provenance = {key: _path(bundle, f"sequential_120/provenance/{name}", files)
                  for key, name in {"config": "source-configuration.json", "source": "source-manifest.json",
                                    "evidence": "native-evidence.json", "native": "native-validation.json",
                                    "context": "context-preflight.json", "cross": "cross-machine-preflight.json",
                                    "contract_receipt": "public-contracts-receipt.json"}.items()}
    public, cases = _lines(paths["public"]), _lines(paths["evaluator"])
    bank, archive, manifest = _json(paths["bank"]), _json(paths["contracts"]), _json(paths["manifest"])
    contracts = _validate_bank(bank, archive)
    inventory = _validate_cases(public, cases, bank, contracts)
    for key, expected in EXPECTED_COUNTS.items():
        _require(manifest.get(key) == expected, f"Sequential declared {key} count mismatch")
        if key in metadata:
            _require(metadata[key] == expected, f"Parent sequential {key} count mismatch")
    for key, observed in inventory.items():
        if key in manifest:
            _require(manifest[key] == observed, f"Sequential declared {key} mismatch")
    _require(manifest.get("train_test_split_assigned") is False, "Sequential split assignment must remain explicitly false")
    config, source = _json(provenance["config"]), _json(provenance["source"])
    _require(_fingerprint(config) == source.get("source_configuration_sha256"), "Frozen source configuration hash mismatch")
    _require(_digest(paths["public"]) == source.get("public_tasks_sha256"), "Frozen public cases hash mismatch")
    _require(_digest(paths["bank"]) == source.get("action_presets_sha256"), "Frozen parameter bank hash mismatch")
    _require(_fingerprint(cases) == config.get("full_corpus_task_specs_sha256", config.get("task_specs_sha256")), "Frozen evaluator case specifications hash mismatch")
    _require(_fingerprint(bank) == config.get("public_parameter_bank_sha256"), "Frozen canonical parameter bank hash mismatch")
    _require(config.get("case_ids") == [row["task_id"] for row in cases], "Frozen evaluator case ordering mismatch")
    contract_receipt = _json(provenance["contract_receipt"])
    _require(contract_receipt.get("archive_sha256") == _digest(paths["contracts"])
             and contract_receipt.get("instruments") == 8 and contract_receipt.get("public_commands") == 146,
             "Frozen public contract archive hash/count mismatch")
    evidence, native = _json(provenance["evidence"]), _json(provenance["native"])
    _require(_digest(provenance["native"]) == evidence.get("original_receipt_sha256"), "Native validation receipt hash mismatch")
    _require(native.get("cases") == 120 and native.get("completed_diagnostic_episodes") == 480
             and native.get("expected_diagnostic_episodes") == 480 and native.get("failures") == []
             and native.get("all_proposals_and_corrections_are_offered_choices") is True,
             "Invalid frozen native validation evidence")
    _require(evidence.get("same_case_specs_and_public_bank") is True
             and evidence.get("original_completed_native_diagnostic_episodes") == 480
             and evidence.get("native_diagnostic_episodes_rerun_in_this_job") == 0,
             "Invalid reused native validation provenance")
    source_hashes = {**config.get("source_sha256", {}), **config.get("choice_source_sha256", {})}
    protected = evidence.get("protected_python_sources_verified")
    _require(isinstance(protected, dict) and bool(protected)
             and all(source_hashes.get(name) == digest for name, digest in protected.items()), "Native protected source hashes mismatch")
    for kind in ("context", "cross"):
        receipt = _json(provenance[kind])
        _require(receipt.get("passed") is True and receipt.get("failures") == []
                 and receipt.get("model_performance_measured") is False
                 and receipt.get("neural_inference_performed") is False
                 and receipt.get("public_preset_bank_sha256") == _fingerprint(bank), f"Invalid {kind} preflight receipt")
        _require(all(source_hashes.get(name, digest) == digest for name, digest in receipt.get("source_sha256", {}).items()),
                 f"Frozen {kind} source hashes mismatch")
    runtime_files = _validate_runtime(bundle, config, files, manifest)
    example_name = "examples/sequential-case.json"
    if (bundle / example_name).exists():
        _require(_json(_path(bundle, example_name, files)) == public[0], "Sequential public example is not an exported case")
    return {"status": "passed", "counts": dict(EXPECTED_COUNTS), **inventory,
            "public_evaluator_boundary_checked": True, "source_exports_authenticated": True,
            "native_replay_performed": False, "model_performance_measured": False,
            "native_diagnostic_episodes_in_preserved_evidence": 480,
            "train_test_split_assigned": False, "unmodified_runtime_files": runtime_files}
