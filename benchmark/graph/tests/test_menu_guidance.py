"""Exercise real LMFE character constraints for the public menu-ID protocol.

No model, tokenizer or simulator is loaded. Both offered actions must remain
generatable irrespective of their usefulness; only public membership and the
declared record shape constrain this transport.
"""
from copy import deepcopy
import json
import os

import pytest

if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
    pytest.skip("Menu guidance tests require a genuine Slurm allocation", allow_module_level=True)

pytest.importorskip('lmformatenforcer', reason='Schema-guided decoding is an explicit optional transport')
from lmformatenforcer.consts import COMPLETE_ALPHABET

from app.models import OperationOptions, VerificationReport
from graph_local.guided_json import make_parser
from graph_local.menu_records import prepare_menu_request
from graph_local.public_context import restore_json


def _request(with_observations=True, role='diagnostic'):
    actions = [
        {"kind": "invoke", "instrument_id": "public-device", "capability_id": "power.set",
         "arguments": {"enabled": False}},
        {"kind": "wait", "dt_s": 0.1},
    ]
    observations = {
        "inspection": {"id": "public-inspection-7", "revision": 7},
        "measurement": {"id": "public-measurement-7", "revision": 7},
    } if with_observations else {"inspection": None, "measurement": None}
    payload = {
        "target": {"goal": "Reason from public evidence before selecting an operation"},
        "observations": observations,
        "benchmark_public_context": {
            "wrapper_revision": 7,
            "current_public_view": {"selected_instrument": "public-device",
                                    "observation": {"sample_time_s": 2.0}},
            "complete_native_history": [],
            "complete_offered_commands_and_presets": [
                {"command": {"text": action["kind"]},
                 "argument_presets": [{"text": "Literal public choice", "action": action}]}
                for action in actions
            ],
        },
    }
    original_schema = OperationOptions if role == 'diagnostic' else VerificationReport
    return prepare_menu_request(role, payload, original_schema), actions


def _record(menu_id="m0", citations=None):
    return {
        "assessment": "Public observations do not yet establish completion.",
        "candidates": [{
            "menu_action_id": menu_id, "rationale": "Inspect this offered alternative.",
            "postcondition": "Evaluate the actual native response.",
            "evidence_ids": citations if citations is not None else ["public-inspection-7"],
        }],
    }


def _accepts(schema, value):
    """Follow the actual parser's legal character transitions and terminal state."""
    parser = make_parser(schema.model_json_schema(), COMPLETE_ALPHABET)
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize("menu_id, index", [("m0", 0), ("m1", 1)])
def test_actual_parser_allows_each_offered_id_and_exact_binding(menu_id, index):
    request, actions = _request()
    value = _record(menu_id)
    assert _accepts(request.schema, value)
    bound = request.bind(request.schema.model_validate(value).model_dump())
    record = OperationOptions.model_validate(bound)
    assert record.candidates[0].intent.arguments == {"action": actions[index]}
    assert record.candidates[0].candidate_id == menu_id
    # Power-off remains legal to propose: the grammar receives no correct-action
    # label and does not remove an offered action merely for being unhelpful.
    assert request.audit["uses_reference_or_evaluator_state"] is False
    wire = restore_json(request.payload)
    assert wire["benchmark_public_context"]["offered_menu_actions"][index]["preset"]["action"] == actions[index]


@pytest.mark.parametrize("menu_id", ["m2", "old-menu-id", "power.set"])
def test_actual_parser_rejects_identifiers_outside_current_public_menu(menu_id):
    request, _ = _request()
    assert not _accepts(request.schema, _record(menu_id))


@pytest.mark.parametrize("citations", [[], ["public-inspection-7"], ["public-measurement-7"],
                                      ["public-inspection-7", "public-measurement-7"]])
def test_actual_parser_allows_only_supplied_current_citation_values(citations):
    request, _ = _request()
    assert _accepts(request.schema, _record(citations=citations))


@pytest.mark.parametrize("citations", [["invented-evidence"], ["public-inspection-6"],
                                      ["public-inspection-7", "invented-evidence"]])
def test_actual_parser_rejects_unknown_and_stale_citation_values(citations):
    request, _ = _request()
    assert not _accepts(request.schema, _record(citations=citations))


def test_actual_parser_with_no_observation_omits_citations_and_binds_empty_list():
    request, _ = _request(with_observations=False)
    value = _record(citations=[])
    value['candidates'][0].pop('evidence_ids')
    assert _accepts(request.schema, value)
    assert request.bind(request.schema.model_validate(value).model_dump())['candidates'][0]['evidence_ids'] == []
    assert not _accepts(request.schema, _record(citations=["public-inspection-7"]))


def test_real_parser_and_binder_have_distinct_duplicate_id_responsibilities():
    request, _ = _request()
    value = _record()
    value["candidates"].append(deepcopy(value["candidates"][0]))
    assert _accepts(request.schema, value)
    with pytest.raises(ValueError, match="distinct"):
        request.bind(request.schema.model_validate(value).model_dump())


def test_actual_parser_rejects_reconstructed_action_fields_in_id_record():
    request, _ = _request()
    value = _record()
    value["candidates"][0]["action"] = {"kind": "wait", "dt_s": 1000}
    assert not _accepts(request.schema, value)


@pytest.mark.parametrize('status', ['supported', 'unresolved', 'contradicted'])
@pytest.mark.parametrize('citations', [[], ['public-inspection-7'], ['public-measurement-7'],
                                     ['public-inspection-7', 'public-measurement-7']])
def test_actual_verifier_parser_permits_public_citations_without_forcing_verdict(status, citations):
    request, _ = _request(role='verification')
    value = {'status': status, 'evidence_ids': citations, 'missing_evidence': []}
    assert _accepts(request.schema, value)
    result = request.bind(request.schema.model_validate(value).model_dump(mode='json'))
    assert VerificationReport.model_validate(result).status == status


@pytest.mark.parametrize('citations', [['shared_49'], ['private-evidence'], ['public-inspection-6'],
                                     ['public-inspection-7', 'shared_49']])
def test_actual_verifier_parser_rejects_packing_refs_unknown_and_stale_citations(citations):
    request, _ = _request(role='verification')
    assert not _accepts(request.schema, {
        'status': 'supported', 'evidence_ids': citations, 'missing_evidence': []})


@pytest.mark.parametrize('status', ['supported', 'unresolved', 'contradicted'])
@pytest.mark.parametrize('missing', [[], ['A decisive native read is still needed'], ['x' * 240],
                                   ['First gap', 'Second gap', 'Third gap', 'Fourth gap']])
def test_actual_verifier_parser_allows_bounded_explanations_and_all_original_verdicts(status, missing):
    request, _ = _request(role='verification')
    value = {'status': status, 'evidence_ids': ['public-measurement-7'], 'missing_evidence': missing}
    assert _accepts(request.schema, value)
    result = request.bind(request.schema.model_validate(value).model_dump(mode='json'))
    assert VerificationReport.model_validate(result).status == status


@pytest.mark.parametrize('missing', [['x' * 241], ['Gap'] * 5])
def test_actual_verifier_parser_rejects_overlong_strings_and_more_than_four_gaps(missing):
    request, _ = _request(role='verification')
    assert not _accepts(request.schema, {
        'status': 'unresolved', 'evidence_ids': [], 'missing_evidence': missing})


def test_actual_verifier_parser_accepts_four_maximum_length_gaps_without_observations():
    request, _ = _request(with_observations=False, role='verification')
    value = {'status': 'unresolved', 'missing_evidence': ['x' * 240] * 4}
    assert _accepts(request.schema, value)
    result = request.bind(request.schema.model_validate(value).model_dump(mode='json'))
    assert VerificationReport.model_validate(result).evidence_ids == []
    assert not _accepts(request.schema, {**value, 'missing_evidence': ['x' * 241]})
    assert not _accepts(request.schema, {**value, 'missing_evidence': ['Gap'] * 5})


@pytest.mark.parametrize('status', ['supported', 'unresolved', 'contradicted'])
def test_actual_verifier_parser_without_observations_has_no_zero_length_array_schema(status):
    request, _ = _request(with_observations=False, role='verification')
    schema = request.schema.model_json_schema()
    assert 'evidence_ids' not in schema['properties']
    value = {'status': status, 'missing_evidence': []}
    assert _accepts(request.schema, value)
    result = request.bind(request.schema.model_validate(value).model_dump(mode='json'))
    assert VerificationReport.model_validate(result).evidence_ids == []
    assert not _accepts(request.schema, {**value, 'evidence_ids': ['shared_49']})
    assert not _accepts(request.schema, {**value, 'evidence_ids': []})
