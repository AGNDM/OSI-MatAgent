"""Public, finite instrument menus shared by every expanded sequential case.

The preset bank is compiled once from the union of authored *public* calls and
contract schemas. It must be frozen and published with the experiment. Menu
construction never receives a private case, checkpoint, fault or grading result.
Numeric/string parameter spaces are discretized; coverage is not exhaustive.
"""
from __future__ import annotations

import copy
import itertools
import json
import math

from jsonschema import validators
from stepwise_bench.policies import ResumeMismatchError


_META = "__meta__"
_FORBIDDEN = {"fault.inject", "plant.configure", "truth.inspect", "sim.step"}
_DEFAULT_WAITS = (0.1, 1, 2, 5, 10, 30, 60, 120, 300, 600, 3600)


class MenuBankError(ResumeMismatchError):
    """A broken public menu is infrastructure failure, never a model mistake."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _unique(values):
    found = {}
    for value in values:
        found.setdefault(_canonical(value), copy.deepcopy(value))
    return [found[key] for key in sorted(found)]


def _public_capabilities(contract):
    return {name: spec for name, spec in contract.get("capabilities", {}).items()
            if spec.get("visibility", "agent") == "agent"
            and spec.get("availability", "implemented") == "implemented"
            and name not in _FORBIDDEN and not name.startswith("truth.")}


def _valid(schema, value):
    try:
        return validators.validator_for(schema)(schema).is_valid(value)
    except Exception as error:
        raise ValueError(f"Unsupported public JSON schema: {error}") from error


def _schema_values(schema, *, limit=128, depth=0):
    """Bounded deterministic representatives, including each enum/bool value.

    For objects we vary each property individually around a baseline and add a
    bounded Cartesian prefix. This avoids an exponential parameter expansion.
    Full candidate objects are validated again against the original schema.
    """
    if depth > 12 or schema is False:
        return []
    if schema is True:
        return [None]
    candidates = []
    if "default" in schema:
        candidates.append(schema["default"])
    candidates.extend(schema.get("examples", []))
    if "const" in schema:
        candidates.append(schema["const"])
    candidates.extend(schema.get("enum", []))
    for keyword in ("oneOf", "anyOf"):
        for branch in schema.get(keyword, []):
            combined = {key: value for key, value in schema.items() if key != keyword}
            combined.update(branch)
            candidates.extend(_schema_values(combined, limit=limit, depth=depth + 1))
    types = schema.get("type")
    if not types:
        types = "object" if "properties" in schema else "string"
    for kind in types if isinstance(types, list) else [types]:
        if kind == "null":
            candidates.append(None)
        elif kind == "boolean":
            candidates.extend([False, True])
        elif kind in {"number", "integer"}:
            numbers = [0, 1, -1]
            if "minimum" in schema:
                numbers.extend([schema["minimum"], schema["minimum"] + 1])
            if "maximum" in schema:
                numbers.extend([schema["maximum"], schema["maximum"] - 1])
            if "minimum" in schema and "maximum" in schema:
                numbers.append((schema["minimum"] + schema["maximum"]) / 2)
            for key, direction in (("exclusiveMinimum", 1), ("exclusiveMaximum", -1)):
                value = schema.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numbers.extend([value + direction, math.nextafter(value, math.inf * direction)])
            multiple = schema.get("multipleOf")
            if multiple:
                numbers += [multiple, 2 * multiple]
            candidates.extend(int(value) if kind == "integer" else value
                              for value in numbers if math.isfinite(value))
        elif kind == "string":
            minimum = schema.get("minLength", 0)
            candidates.extend(["", "example", "x" * min(minimum, 256)])
        elif kind == "array":
            minimum = schema.get("minItems", 0)
            maximum = schema.get("maxItems", max(minimum, 1))
            items = schema.get("items", {})
            if isinstance(items, list):
                choices = [_schema_values(item, depth=depth + 1) for item in items]
                if all(choices):
                    candidates.extend(list(values) for values in itertools.islice(itertools.product(*choices), limit))
            else:
                choices = _schema_values(items, limit=limit, depth=depth + 1)
                if minimum == 0:
                    candidates.append([])
                if minimum <= 32 and choices:
                    for length in sorted({minimum, min(maximum, max(minimum, 1))}):
                        candidates.extend([copy.deepcopy(value) for _ in range(length)] for value in choices[:limit])
        elif kind == "object":
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            choices = {key: _schema_values(value, limit=limit, depth=depth + 1)
                       for key, value in sorted(properties.items())}
            if all(choices.get(key) for key in required):
                baseline = {key: copy.deepcopy(choices[key][0]) for key in sorted(required)}
                candidates.append(baseline)
                # Every property can be set explicitly, including optional enum
                # flags. No parameter is filtered using the live machine state.
                full = {key: copy.deepcopy(values[0]) for key, values in choices.items() if values}
                candidates.append(full)
                for base in (baseline, full):
                    for key, values in choices.items():
                        for value in values:
                            candidates.append({**copy.deepcopy(base), key: copy.deepcopy(value)})
                required_keys = sorted(required)
                candidates.extend(dict(zip(required_keys, values)) for values in itertools.islice(
                    itertools.product(*(choices[key] for key in required_keys)), limit))
    return _unique(value for value in candidates if _valid(schema, value))[:limit]


def build_preset_bank(tasks, contracts):
    """Compile a shared public candidate bank; never call this per episode.

    Only implemented, agent-visible calls survive. No fixtures, private labels,
    dependency order, scheduled events or correctness flags enter the result.
    Authored calls contribute argument values in a canonical sorted union.
    """
    bank, schemas = {}, {}
    report = {}
    for machine, contract in sorted(contracts.items()):
        bank[machine], schemas[machine] = {}, {}
        for capability, spec in sorted(_public_capabilities(contract).items()):
            schema = spec.get("input_schema", {"type": "object"})
            schemas[machine][capability] = schema
            candidates = _schema_values(schema)
            if isinstance(spec.get("example_arguments"), dict):
                candidates.append(spec["example_arguments"])
            bank[machine][capability] = [value for value in candidates
                                         if isinstance(value, dict) and _valid(schema, value)]
    waits = list(_DEFAULT_WAITS)
    reads = {machine: [] for machine in bank}
    for task in tasks:
        machine = task["instrument_id"]
        public_slots = set(contracts[machine].get("state_slots", {}))
        if task.get("observation_ids") and set(task["observation_ids"]) <= public_slots:
            reads[machine].append(sorted(task["observation_ids"]))
        calls = list(task.get("setup_steps", [])) + [node["reference_action"] for node in task.get("checkpoints", [])]
        for action in calls:
            kind = action.get("kind")
            selected = action.get("instrument_id", machine)
            if kind == "invoke" and selected in bank:
                capability = action.get("capability_id")
                arguments = action.get("arguments", {})
                if capability in bank[selected] and _valid(schemas[selected][capability], arguments):
                    bank[selected][capability].append(copy.deepcopy(arguments))
            elif kind == "wait":
                value = action.get("dt_s")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 < value <= 86400:
                    waits.append(value)
            elif kind == "read" and selected in reads:
                state_ids = action.get("state_ids")
                if state_ids and set(state_ids) <= set(contracts[selected].get("state_slots", {})):
                    reads[selected].append(sorted(state_ids))
    for machine, capabilities in bank.items():
        report[machine] = {}
        for capability, values in capabilities.items():
            capabilities[capability] = _unique(values)
            report[machine][capability] = {"preset_count": len(capabilities[capability]),
                                           "schema_validated": True,
                                           "exhaustive_parameter_coverage": False}
            if not capabilities[capability]:
                raise ValueError(f"No schema-valid public preset for {machine}/{capability}; add a shared public preset")
        # Every readable slot and authored read-set is offered; read-all remains
        # a separate direct action and is not concealed by the parameter menu.
        reads[machine] = _unique(reads[machine] + [[key] for key in sorted(contracts[machine].get("state_slots", {}))])
    bank[_META] = {"wait_durations_s": sorted(set(float(value) for value in waits)),
                   "read_sets": reads, "support_report": report,
                   "preset_provenance": "Shared union of all authored public setup/checkpoint argument values plus contract schema representatives. Values are public candidates for every case; their source case, chronology and correctness are not exposed.",
                   "parameter_policy": "Finite shared schema-valid presets; not all possible numeric/string arguments.",
                   "ordering_policy": "Canonical public command and JSON argument order; never checkpoint order."}
    return bank


class MenuBuilder:
    """Generate menus using only a public view and one frozen shared bank."""

    def __init__(self, bank):
        if not isinstance(bank, dict) or not isinstance(bank.get(_META), dict):
            raise MenuBankError("Missing shared public menu metadata")
        self.bank = copy.deepcopy(bank)
        self._validate_metadata()
        report = self.bank[_META].get("support_report", {})
        for machine, capabilities in self.bank.items():
            if machine == _META:
                continue
            if not isinstance(capabilities, dict):
                raise MenuBankError(f"Missing public preset bank for {machine}")
            for capability in set(capabilities) | set(report.get(machine, {})):
                self._require_presets(machine, capability)
            self._read_sets(machine)

    def _validate_metadata(self):
        metadata = self.bank.get(_META, {})
        waits = metadata.get("wait_durations_s")
        if not isinstance(waits, list) or not waits or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= 86400 for value in waits):
            raise MenuBankError("Missing or invalid shared public wait presets")
        if not isinstance(metadata.get("read_sets"), dict):
            raise MenuBankError("Missing shared public read metadata")

    def _require_presets(self, machine, capability):
        capabilities = self.bank.get(machine)
        values = capabilities.get(capability) if isinstance(capabilities, dict) else None
        if not isinstance(values, list) or not values or not all(isinstance(value, dict) for value in values):
            raise MenuBankError(f"Missing or empty public presets for {machine}/{capability}")
        return values

    def _read_sets(self, machine):
        self._validate_metadata()
        values = self.bank[_META]["read_sets"].get(machine)
        if not isinstance(values, list) or not all(isinstance(value, list) and value
                and all(isinstance(state, str) for state in value) for value in values):
            raise MenuBankError(f"Missing or malformed public read presets for {machine}")
        return values

    def _validate_selected(self, public_view):
        machine = public_view.get("selected_instrument")
        if machine not in self.bank or machine == _META:
            raise MenuBankError(f"Selected instrument {machine} is missing from the shared public bank")
        for capability in _public_capabilities(public_view.get("contract", {})):
            self._require_presets(machine, capability)
        self._read_sets(machine)

    def action_options(self, public_view, history=None):
        selected = public_view.get("selected_instrument")
        catalog = sorted(public_view.get("machine_catalog", []), key=lambda row: row["instrument_id"])
        machine_options = [{"text": f"Select {row.get('name', row['instrument_id'])} ({row['instrument_id']})",
                            "kind": "action", "action": {"kind": "select_instrument", "instrument_id": row["instrument_id"]}}
                           for row in catalog]
        if selected is None:
            return machine_options
        self._validate_selected(public_view)
        contract = public_view.get("contract", {})
        options = []
        for capability, spec in sorted(_public_capabilities(contract).items()):
            description = spec.get("description", spec.get("description_zh", spec.get("function_zh", "")))
            options.append({"text": capability + (f" — {description}" if description else ""),
                            "kind": "command", "capability_id": capability})
        options.extend([
            {"text": "Read all public instrument readings", "kind": "action",
             "action": {"kind": "read", "instrument_id": selected, "state_ids": None}},
            {"text": "Read selected public readings", "kind": "command", "capability_id": "__read__"},
            {"text": "Wait for simulated time", "kind": "command", "capability_id": "__wait__"},
            {"text": "Describe instrument commands and readable states", "kind": "action",
             "action": {"kind": "describe", "instrument_id": selected}},
            {"text": "Finish and report the goal complete", "kind": "action", "action": {"kind": "finish"}},
            {"text": "Escalate for maintenance or operator assistance", "kind": "action", "action": {"kind": "escalate"}},
        ])
        options.extend(machine_options)
        return options

    def argument_options(self, public_view, history, option):
        if "action" in option:
            return [{"text": option["text"], "action": copy.deepcopy(option["action"])}]
        self._validate_selected(public_view)
        machine = public_view.get("selected_instrument")
        capability = option["capability_id"]
        if capability == "__wait__":
            remaining = float(public_view["budget"]["max_sim_time_s"]) - float(public_view.get("elapsed_sim_time_s", 0))
            # Keep the command selectable at an exhausted time budget. Its
            # well-formed positive duration is then rejected by the native
            # resource guard, rather than becoming an empty parameter menu.
            if remaining <= 0:
                return [{"text": "Wait 0.1 simulated seconds",
                         "action": {"kind": "wait", "dt_s": 0.1}}]
            durations = sorted(set(min(float(value), remaining, 86400.0)
                                   for value in self.bank[_META]["wait_durations_s"]))
            return [{"text": f"Wait {duration:g} simulated seconds", "action": {"kind": "wait", "dt_s": duration}}
                    for duration in durations if duration > 0]
        if capability == "__read__":
            allowed = set(public_view.get("contract", {}).get("state_slots", {}))
            presets = self._read_sets(machine)
            if not presets or any(not set(state_ids) <= allowed for state_ids in presets):
                raise MenuBankError(f"Public read presets disagree with the selected contract for {machine}")
            return [{"text": "Read " + ", ".join(state_ids),
                     "action": {"kind": "read", "instrument_id": machine, "state_ids": copy.deepcopy(state_ids)}}
                    for state_ids in presets]
        if capability not in _public_capabilities(public_view.get("contract", {})):
            raise ValueError("Requested command is not in the selected instrument's public contract")
        return [{"text": _canonical(arguments),
                 "action": {"kind": "invoke", "instrument_id": machine, "capability_id": capability,
                            "arguments": copy.deepcopy(arguments)}}
                for arguments in self._require_presets(machine, capability)]
