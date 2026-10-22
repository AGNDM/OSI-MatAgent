"""Exercise the sent schema and fail-closed handling without real model requests."""
from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pydantic import ValidationError as ModelValidationError

from app.device_selection import DiscoveryRequest
from app.expert_config import CompatibleExpert, ExpertEndpoint
from app.local_roles import ExpertResponseError
from app.models import ActionProposal, OperationOptions, VerificationReport
from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
from app.single_instrument import InstrumentTask
from app.structured_output import output_schema


@pytest.fixture(scope='module')
def caps():
    task = InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8'))
    return Rc1mxExperiment(task.model_copy(update={'device': 'mt_rc1mxhfcal'})).capabilities()


def options(capability='mt_rc1mxhfcal::power.set', arguments=None):
    return {'hypotheses': [], 'candidates': [{'candidate_id': 'candidate',
        'intent': {'capability_id': capability, 'arguments': {'enabled': True} if arguments is None else arguments,
                   'discriminating_observation': 'Test', 'postcondition': 'Test'},
        'rationale': 'Test', 'evidence_ids': []}]}


def config(**changes):
    return ExpertEndpoint(**dict(dict(name='offline-strict', provider='openrouter',
        base_url='https://openrouter.ai/api/v1', model='deepseek/deepseek-v4.1-flash',
        api_key_env='OPENROUTER_API_KEY', output_format='json_schema',
        provider_only=['fireworks/spot']), **changes))


def test_strict_schema_preserves_every_native_argument_contract(caps):
    before = deepcopy(caps)
    document = output_schema(OperationOptions, {'capabilities': caps})
    branches = document['$defs']['Intent']['anyOf']
    assert len(branches) == len(caps)
    for branch in branches:
        cap = branch['properties']['capability_id']['enum'][0]
        ref = branch['properties']['arguments']['$ref'].rsplit('/', 1)[-1]
        assert document['$defs'][ref] == caps[cap]['input_schema']
        assert branch['additionalProperties'] is False
        assert 'arguments' in branch['required']
    assert caps == before


@pytest.mark.parametrize('capability,args,valid', [
    ('mt_rc1mxhfcal::power.set', {'enabled': True}, True),
    ('mt_rc1mxhfcal::power.set', {'enabled': False}, True),
    ('mt_rc1mxhfcal::power.set', {}, False),
    ('mt_rc1mxhfcal::power.set', {'enabled': 'true'}, False),
    ('mt_rc1mxhfcal::power.set', {'rpm': 250}, False),
    ('mt_rc1mxhfcal::power.set', {'enabled': True, 'extra': 1}, False),
    ('workflow.inspect', {}, True),
    ('workflow.inspect', {'state_ids': ['power.enabled']}, True),
    ('workflow.inspect', {'state_ids': None}, False),
    ('workflow.inspect', {'state_ids': ['imaginary']}, False),
    ('workflow.await', {'duration_s': 120}, True),
    ('workflow.await', {'duration_s': 121}, False),
    ('unknown.operation', {}, False),
])
def test_native_required_optional_types_ranges_and_capability_correlation(caps, capability, args, valid):
    validator = Draft202012Validator(output_schema(OperationOptions, {'capabilities': caps}))
    assert validator.is_valid(options(capability, args)) is valid


@pytest.mark.parametrize('fault', ['missing_arguments', 'null_arguments', 'rationale_in_intent', 'evidence_in_intent'])
def test_old_missing_parameter_and_misnested_outputs_are_invalid(caps, fault):
    raw = options()
    candidate = raw['candidates'][0]
    if fault == 'missing_arguments':
        del candidate['intent']['arguments']
    elif fault == 'null_arguments':
        candidate['intent']['arguments'] = None
    else:
        key = 'rationale' if fault.startswith('rationale') else 'evidence_ids'
        candidate['intent'][key] = candidate.pop(key)
    with pytest.raises(ValidationError):
        Draft202012Validator(output_schema(OperationOptions, {'capabilities': caps})).validate(raw)


def test_procedure_schema_uses_selected_capability(caps):
    selected = 'mt_rc1mxhfcal::power.set'
    document = output_schema(ActionProposal, {'capabilities': {selected: caps[selected]}})
    validator = Draft202012Validator(document)
    raw = dict(sample_id='instrument', expected_revision=0, request_id='request',
               capability_id=selected, arguments={'enabled': False}, postcondition='Test')
    validator.validate(raw)
    assert not validator.is_valid(dict(raw, arguments={}))
    assert not validator.is_valid(dict(raw, capability_id='workflow.inspect'))


def test_native_local_references_keep_their_scope():
    native = {'type': 'object', 'properties': {'enabled': {'$ref': '#/$defs/Flag'}},
              'required': ['enabled'], 'additionalProperties': False,
              '$defs': {'Flag': {'type': 'boolean'}}}
    doc = output_schema(OperationOptions, {'capabilities': {'test': {'input_schema': native}}})
    validator = Draft202012Validator(doc)
    validator.validate(options('test', {'enabled': False}))
    assert not validator.is_valid(options('test', {'enabled': 0}))


def test_discovery_and_verification_keep_existing_contracts():
    for model in [DiscoveryRequest, VerificationReport]:
        assert output_schema(model, {}) == model.model_json_schema()


def test_missing_capability_contract_does_not_fall_back():
    with pytest.raises(ValueError, match='requires public capability'):
        output_schema(OperationOptions, {})


def test_configuration_requires_explicit_supported_channel():
    with pytest.raises(ModelValidationError, match='explicitly selected'):
        config(provider_only=[])
    with pytest.raises(ModelValidationError, match='requires OpenRouter'):
        config(provider='local')


@pytest.mark.parametrize('bad', [False, True])
def test_sent_request_is_strict_pinned_and_locally_rechecked(caps, monkeypatch, bad):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-only-key')
    seen = []
    raw = options()
    if bad:
        raw['candidates'][0]['intent']['rationale'] = raw['candidates'][0].pop('rationale')
    async def respond(self, body, timeout):
        seen.append(body)
        assert body['response_format']['type'] == 'json_schema'
        contract = body['response_format']['json_schema']
        assert contract['strict'] is True
        assert contract['name'] == 'OperationOptions'
        assert body['provider'] == {'only': ['fireworks/spot'], 'order': ['fireworks/spot'],
                                    'allow_fallbacks': False, 'require_parameters': True}
        prompt_schema = json.loads(body['messages'][0]['content'].split('No Markdown or extra fields.\n')[1])
        assert prompt_schema == contract['schema']
        assert 'chat_template_kwargs' not in body
        return {'provider': 'Fireworks', 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(raw)}}]}
    monkeypatch.setattr(CompatibleExpert, '_request', respond)
    expert = CompatibleExpert(config())
    if bad:
        with pytest.raises(ExpertResponseError) as error:
            expert.invoke('diagnostic', 'Return JSON.', {'capabilities': caps}, OperationOptions)
        assert error.value.code == 'expert_schema_violation'
        assert expert.last_call['response_classification'] == 'expert_schema_violation'
    else:
        result = expert.invoke('diagnostic', 'Return JSON.', {'capabilities': caps}, OperationOptions)
        assert result['candidates'][0]['intent'] == raw['candidates'][0]['intent']
    assert len(seen) == 1  # No fallback, repair, or retry.
    assert 'offline-only-key' not in json.dumps(expert.last_call)


def test_provider_failure_is_not_retried_or_downgraded(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-only-key')
    seen = []
    async def reject(self, body, timeout):
        seen.append(body)
        raise RuntimeError('HTTP 400: schema unsupported')
    monkeypatch.setattr(CompatibleExpert, '_request', reject)
    with pytest.raises(RuntimeError, match='schema unsupported'):
        CompatibleExpert(config()).invoke('verification', 'Test', {}, VerificationReport)
    assert len(seen) == 1 and seen[0]['response_format']['type'] == 'json_schema'
