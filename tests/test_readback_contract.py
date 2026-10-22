from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from app.models import ActionProposal, HypothesisSet, OperationOptions
from app.readback_contract import normalize_arguments, normalize_record

CAPS = {"workflow.readback": {"input_schema": {"enum": [
    {"state_ids": None}, {"state_ids": ["error.code", "lock.on", "power.on"]}]}}}


def test_reordered_fields_match_original_preset_without_mutating_response():
    args = {"state_ids": ["power.on", "lock.on", "error.code"]}
    before = deepcopy(args)
    normalized = normalize_arguments("workflow.readback", args, CAPS)
    assert normalized == CAPS["workflow.readback"]["input_schema"]["enum"][1]
    assert not list(Draft202012Validator(CAPS["workflow.readback"]["input_schema"]).iter_errors(normalized))
    assert args == before


@pytest.mark.parametrize("args", [
    {"state_ids": ["power.on"]},
    {"state_ids": ["power.on", "lock.on", "error.code", "error.code"]},
    {"state_ids": ["power.on", "lock.on", "error.code"], "extra": 1},
    {"state_ids": ["power.on", "lock.on", 3]},
])
def test_invalid_field_combinations_are_not_repaired_or_silently_dropped(args):
    assert normalize_arguments("workflow.readback", args, CAPS) == args
    assert list(Draft202012Validator(CAPS["workflow.readback"]["input_schema"]).iter_errors(args))


def test_null_and_other_capabilities_keep_their_semantics():
    assert normalize_arguments("workflow.readback", {"state_ids": None}, CAPS) == {"state_ids": None}
    args = {"state_ids": ["power.on", "lock.on", "error.code"], "segments": [2, 1]}
    assert normalize_arguments("device::thermal.configure_program", args, CAPS) == args
    assert normalize_arguments("device::workflow.readback", args, CAPS) == args


def test_candidate_and_procedure_handoff_agree_after_normalization():
    raw = {"hypotheses": [], "candidates": [{"candidate_id": "read", "intent": {
        "capability_id": "workflow.readback", "arguments": {"state_ids": ["power.on", "lock.on", "error.code"]},
        "discriminating_observation": "Read states", "postcondition": "Fresh states"},
        "rationale": "Observe", "evidence_ids": []}]}
    candidate = normalize_record(OperationOptions.model_validate(raw), CAPS).candidates[0]
    hypothesis = normalize_record(HypothesisSet(hypotheses=[], intent=raw["candidates"][0]["intent"]), CAPS)
    proposal = ActionProposal(sample_id="sample", expected_revision=1, request_id="request",
        capability_id="workflow.readback", arguments=raw["candidates"][0]["intent"]["arguments"], postcondition="Fresh states")
    proposal = normalize_record(proposal, CAPS)
    assert candidate.intent.arguments == hypothesis.intent.arguments == proposal.arguments == {"state_ids": ["error.code", "lock.on", "power.on"]}
    assert raw["candidates"][0]["intent"]["arguments"]["state_ids"][0] == "power.on"
