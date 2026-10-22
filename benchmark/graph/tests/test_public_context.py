"""Lossless public model-input factoring; run in the graph Slurm check job."""
from copy import deepcopy
import json

import pytest

from graph_local.public_context import (
    OBJECT, PROTOCOL, REFERENCE, TABLE, assert_public_context_roundtrip,
    compact_json, compact_public_context, restore_json, restore_public_context,
)


def public_context():
    command = {
        "description_zh": "Public manual: enabling power is required before configuration. " * 8,
        "input_schema": {
            "type": "object", "properties": {"temperature_K": {"type": "number", "minimum": 273.15}},
            "required": ["temperature_K"], "additionalProperties": False,
        },
        "output_schema": {"type": "object", "properties": {"accepted": {"type": "boolean"}}},
        "preconditions": ["power.enabled"], "errors": {"OFF": "Power is disabled"},
    }
    contract = {
        "capabilities": {"temperature.set": command},
        "state_slots": {"temperature.actual_K": {"unit": "K", "visibility": "agent", "access": "read"}},
    }
    observation = {"observations": {
        "temperature.actual_K": {"value": 298.15, "unit": "K", "quality": "valid", "sample_time_s": 7.0},
        "power.enabled": {"value": True, "unit": "1", "quality": "valid", "sample_time_s": 7.0},
    }, "clock": 10.0, "state_revision": 3}
    selected = {"kind": "select_instrument", "instrument_id": "public-machine"}
    response = {"status": "succeeded", "contract": deepcopy(contract), **deepcopy(observation)}
    history = [{"proposed_action": deepcopy(selected), "executed_action": deepcopy(selected),
                "corrected": False, "response": deepcopy(response)}]
    history.append({"proposed_action": {"kind": "wait", "dt_s": 1.0},
                    "executed_action": {"kind": "wait", "dt_s": 2.0},
                    "corrected": True, "response": deepcopy(response)})
    presets = [{"text": f"Temperature {temperature} K", "action": {
        "kind": "invoke", "instrument_id": "public-machine", "capability_id": "temperature.set",
        "arguments": {"temperature_K": float(temperature)},
    }} for temperature in range(300, 324)]
    return {
        "public_task": {"task_id": "opaque-public-id", "initial_context": {
            "goal": "Configure the public machine and verify its observed temperature",
            "selected_instrument": None, "contract": {}, "observation": {},
        }},
        "current_public_view": {"selected_instrument": "public-machine", "contract": deepcopy(contract),
                                "observation": deepcopy(observation), "elapsed_sim_time_s": 10.0},
        "complete_native_history": history,
        "complete_offered_commands_and_presets": [{"command": {
            "text": "temperature.set", "capability_id": "temperature.set"}, "argument_presets": presets}],
        "wrapper_revision": 2,
        "evidence_note": "Transport state does not make a cached measurement fresh.",
    }


def test_contracts_history_and_every_preset_roundtrip_after_json_transport():
    original = public_context()
    frozen = deepcopy(original)
    compacted = compact_public_context(original)
    transported = json.loads(json.dumps(compacted, sort_keys=True, ensure_ascii=False))
    restored = restore_public_context(transported)
    assert_public_context_roundtrip(original, transported)
    assert restored == original == frozen
    assert restored["complete_native_history"][1]["corrected"] is True
    assert restored["complete_native_history"][1]["proposed_action"]["dt_s"] == 1.0
    assert restored["complete_native_history"][1]["executed_action"]["dt_s"] == 2.0
    reading = restored["current_public_view"]["observation"]["observations"]["temperature.actual_K"]
    assert reading["sample_time_s"] == 7.0
    assert len(restored["complete_offered_commands_and_presets"][0]["argument_presets"]) == 24
    assert len(json.dumps(compacted)) < len(json.dumps(original))
    assert compacted["protocol"] == PROTOCOL
    assert compacted["shared"]


def test_shared_reconstruction_never_creates_mutable_aliases():
    original = public_context()
    restored = restore_public_context(compact_public_context(original))
    restored["current_public_view"]["contract"]["capabilities"].clear()
    assert restored["complete_native_history"][0]["response"]["contract"]["capabilities"]
    assert original["current_public_view"]["contract"]["capabilities"]


def test_nested_object_tables_preserve_missing_null_empty_and_numeric_types():
    common = "A repeated public description for representative parameter rows. " * 5
    records = []
    for number in range(12):
        row = {"action": {"kind": "invoke", "instrument_id": "machine", "capability_id": "public.set",
                          "arguments": {"integer": number, "float": float(number), "bool": bool(number % 2)}},
               "description": common, "empty": {}, "array": [None, {}, [], 1, 1.0, True]}
        if number % 2:
            row["nullable"] = None
        records.append(row)
    compacted = compact_json(records)
    assert TABLE in compacted["data"]
    restored = restore_json(compacted)
    assert restored == records
    assert "nullable" not in restored[0]
    assert restored[1]["nullable"] is None
    assert restored[0]["empty"] == {}
    assert type(restored[1]["action"]["arguments"]["integer"]) is int
    assert type(restored[1]["action"]["arguments"]["float"]) is float
    assert type(restored[1]["action"]["arguments"]["bool"]) is bool


@pytest.mark.parametrize("value", [
    {REFERENCE: "This is literal public content"},
    {TABLE: {"paths": "a literal manual example, not an encoded table"}},
    {OBJECT: [["literal", "value"]]},
    {REFERENCE: "literal", "another": {OBJECT: True}},
    [{REFERENCE: "literal"}, {REFERENCE: "literal"}, {REFERENCE: "literal"}],
    {"empty_key": {"": {"": None}}},
    None, [], {}, "", False, 0, 0.0, -0.0,
])
def test_literal_encoding_markers_and_json_edges_roundtrip(value):
    restored = restore_json(compact_json(value))
    assert json.dumps(restored, sort_keys=True) == json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), ("tuple",), {1: "non-string key"}, b"bytes"])
def test_non_json_inputs_fail_closed(value):
    with pytest.raises(ValueError):
        compact_json(value)


def test_source_cycles_and_non_object_context_fail_closed():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cycle"):
        compact_json(value)
    with pytest.raises(ValueError, match="must be an object"):
        compact_public_context([])
    with pytest.raises(ValueError, match="must be an object"):
        restore_public_context(compact_json([]))


def test_tampered_public_value_fingerprint_is_rejected():
    compacted = compact_public_context({"sample_time_s": 7.0})
    compacted["data"]["sample_time_s"] = 8.0
    with pytest.raises(ValueError, match="fingerprint"):
        restore_public_context(compacted)


def test_tampered_shared_catalog_and_missing_reference_are_rejected():
    compacted = compact_public_context(public_context())
    name = next(iter(compacted["shared"]))
    compacted["shared"][name] = "changed public value"
    with pytest.raises(ValueError, match="fingerprint"):
        restore_public_context(compacted)
    compacted = compact_json({"value": "original"})
    compacted["data"] = {REFERENCE: "missing"}
    with pytest.raises(ValueError, match="Missing shared"):
        restore_json(compacted)


def test_reference_cycles_and_ambiguous_markers_fail_closed():
    compacted = compact_json({"value": "original"})
    compacted["data"] = {REFERENCE: "loop"}
    compacted["shared"] = {"loop": {REFERENCE: "loop"}}
    with pytest.raises(ValueError, match="Cyclic shared"):
        restore_json(compacted)
    compacted["data"] = {REFERENCE: "loop", "other": "not a reference"}
    with pytest.raises(ValueError, match="Ambiguous"):
        restore_json(compacted)


@pytest.mark.parametrize("table, message", [
    ({"paths": [["x"]], "defaults": [], "rows": [[2, [1]]]}, "presence mask"),
    ({"paths": [["x"]], "defaults": [], "rows": [[1, []]]}, "cell count"),
    ({"paths": [["x"], ["x"]], "defaults": [], "rows": []}, "Repeated"),
    ({"paths": [["x"], ["x", "nested"]], "defaults": [], "rows": [[3, [{}, 1]]]}, "Overlapping"),
    ({"paths": [[]], "defaults": [], "rows": []}, "object path"),
    ({"paths": [], "defaults": [], "rows": [], "extra": 1}, "Invalid public object table"),
])
def test_corrupted_tables_are_rejected_before_reconstruction(table, message):
    compacted = compact_json([])
    compacted["data"] = {TABLE: table}
    with pytest.raises(ValueError, match=message):
        restore_json(compacted)


def test_original_and_compact_pair_must_describe_identical_json():
    compacted = compact_public_context({"value": 1})
    with pytest.raises(ValueError, match="differs"):
        assert_public_context_roundtrip({"value": 1.0}, compacted)
