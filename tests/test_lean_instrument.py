from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from app.graph import build_graph, initial_state
from app.lab_adapter import GateRejected
from app.lean_instrument import LeanInstrumentProfile, essential_capability, lean_payload
from app.models import ActionProposal, Limits
from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
from app.single_instrument import InstrumentTask, InstrumentProfile
from app.trace import Trace


def lab():
    return Rc1mxExperiment(InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8')).model_copy(update={'device': 'mt_rc1mxhfcal'}))


def proposal(device, cap, args=None, request_id='query-1'):
    return ActionProposal(sample_id=device.sample_id, expected_revision=device.revision,
        request_id=request_id, capability_id=cap, arguments=args or {}, postcondition='Read public documentation')


def state_for(device):
    state=initial_state(device,'offline-contract-test',Limits())
    state['observations']=device.observe()
    return state


def test_every_original_parameter_schema_and_operating_condition_is_preserved():
    device=lab()
    state=state_for(device)
    before=deepcopy(state)
    payload=LeanInstrumentProfile.projection('diagnostic',state,'req')
    assert state==before
    assert set(payload['capabilities'])==set(device.capabilities())
    for key,original in device.capabilities().items():
        slim=payload['capabilities'][key]
        assert slim['input_schema']==original['input_schema']
        conditions=original.get('preconditions',[])
        reduced=slim.get('preconditions',[])
        if isinstance(conditions,dict):
            conditions,reduced=[conditions],[reduced]
        for condition,summary in zip(conditions,reduced):
            if isinstance(condition,dict):
                for field in ('description','description_zh','enforced_by'):
                    assert summary.get(field)==condition.get(field)
                assert 'check' not in summary
                assert not any(key.startswith(('truth.','host.','plant.','debug.'))
                    for key in summary.get('public_state_ids',[]))
        assert 'state_effects' not in slim and 'binding' not in slim and 'implementation' not in slim
    original=InstrumentProfile.projection('diagnostic',state,'req')
    assert len(json.dumps(payload)) < len(json.dumps(original))*0.35
    tr=payload['capabilities']['mt_rc1mxhfcal::thermostat.set_tr']
    assert not list(Draft202012Validator(tr['input_schema']).iter_errors(
        dict(end_k=298.15,ramp_s=0,controller='aqueous')))
    assert list(Draft202012Validator(tr['input_schema']).iter_errors(
        dict(end_k=25,ramp_s=0,controller='aqueous')))


def test_native_describe_returns_one_contract_without_advancing_or_inspecting():
    device=lab()
    before=device.session.read(instance_id=device.instance_id,state_ids=device.fields)
    counts=deepcopy(device.lifecycle.calls)
    query=proposal(device,'workflow.describe',{'capability_id':'mt_rc1mxhfcal::thermostat.set_tr'})
    event=device.execute(query)
    after=device.session.read(instance_id=device.instance_id,state_ids=device.fields)
    assert before==after
    assert device.lifecycle.elapsed==0 and device.revision==0
    assert device._inspection is None and device._measurement is None
    assert device.action_count==1 and device.action_cost(query)==1
    assert device.lifecycle.calls['invoke']==counts['invoke']
    assert device.lifecycle.calls['step']==counts['step']
    assert device.lifecycle.calls['describe']==1
    detail=event['result']['capability_description']
    assert detail['source']=='bound agent session.describe' and not detail['is_measurement']
    assert detail['contract']['input_schema']==device.capabilities()['mt_rc1mxhfcal::thermostat.set_tr']['input_schema']
    assert 'errors' in detail['contract'] and 'output_schema' in detail['contract']
    assert device.observe()['capability_detail']==detail
    # Replays do not even repeat the describe call or charge another action.
    assert device.execute(query)==event and device.action_count==1
    assert device.lifecycle.calls['describe']==1


@pytest.mark.parametrize('name',[
    'mt_rc1mxhfcal::reaction.configure','mt_rc1mxhfcal::reactor.charge',
    'mt_rc1mxhfcal::plant.configure','mt_rc1mxhfcal::sim.step',
    'mt_rc1mxhfcal::ledger.read','mt_rc1mxhfcal::rtcal.enable','unknown::power.set'])
def test_discovery_cannot_open_private_or_unimplemented_capabilities(name):
    device=lab()
    with pytest.raises(GateRejected):
        device.execute(proposal(device,'workflow.describe',{'capability_id':name}))
    assert device.lifecycle.calls.get('describe',0)==0
    assert device.lifecycle.elapsed==0 and device.action_count==0


def test_query_during_pending_preserves_pending_and_fresh_evidence():
    device=lab()
    device.execute(proposal(device,'workflow.inspect',request_id='inspect-1'))
    evidence=deepcopy(device._measurement)
    device.pending_operation={'operation_id':'test-pending','status':'running','capability':'calibration.run'}
    pending=deepcopy(device.pending_operation)
    device.execute(proposal(device,'workflow.describe',{'capability_id':'mt_rc1mxhfcal::calibration.run'}))
    assert device.pending_operation==pending and device._measurement==evidence
    assert device.lifecycle.elapsed==0 and device.revision==0


def test_tick_details_stay_in_original_log_but_not_repeated_in_model_prompt():
    original={'observations':{'last_action':{'result':{'constraint_reads':[{'temperature':298}]*60,
        'clock_steps':[{'dt_s':2}]*60,'advanced_seconds':120,'error':None}},
        'measurement':{'id':'obs-7','native':{'observations':{'reactor.indicated_k':{'value':298,'unit':'K'}}}}}}
    before=deepcopy(original)
    compact=lean_payload(original)
    assert original==before
    assert compact['observations']['measurement']==original['observations']['measurement']
    receipt=compact['observations']['last_action']['result']
    assert receipt['constraint_reads_count']==60 and receipt['clock_steps_count']==60
    assert receipt['advanced_seconds']==120 and 'constraint_reads' not in receipt


def test_retrieval_audit_query_is_omitted_but_grounding_passages_are_retained():
    payload={'retrieval_context':{'query':'Repeated state and task'},
        'retrieval_evidence':[{'id':'source-1','text':'Important operating condition'}]}
    compact=LeanInstrumentProfile.model_payload(payload)
    assert 'retrieval_context' not in compact and 'retrieval_context' in payload
    assert compact['retrieval_evidence']==payload['retrieval_evidence']


def test_existing_graph_can_query_native_description_then_stop_without_models():
    device=lab()
    class OfflineRoles:
        is_llm=False
        backend_id='offline-contract-test'
        def invoke(self,role,prompt,payload,schema):
            assert 'retrieval_context' not in payload
            if role=='diagnostic':
                queried=payload['observations'].get('capability_detail')
                cap='workflow.stop' if queried else 'workflow.describe'
                args={'reason':'Offline query test complete'} if queried else {'capability_id':'mt_rc1mxhfcal::thermostat.set_tr'}
                return schema.model_validate({'hypotheses':[],'intent':{'capability_id':cap,'arguments':args,
                    'discriminating_observation':'Public contract','postcondition':'Read documentation only'}}).model_dump()
            if role=='procedure':
                intent=payload['intent']
                return schema.model_validate({k:payload[k] for k in ('sample_id','expected_revision','request_id')}|
                    {k:intent[k] for k in ('capability_id','arguments','postcondition')}).model_dump()
            raise AssertionError(role)
    limits=Limits(lab_actions=4,graph_steps=25)
    trace=Trace('offline-query')
    graph=build_graph(device,trace,limits,roles=OfflineRoles(),profile=LeanInstrumentProfile(False))
    state=graph.invoke(initial_state(device,'offline-query',limits),{'recursion_limit':25,'configurable':{'thread_id':'offline-query'}})
    assert state['report']['llm_call_count']==0
    assert state['report']['terminal_reason']=='agent_stopped_unresolved'
    assert [e['proposal']['capability_id'] for e in trace.events if e['kind']=='tool_result']==['workflow.describe','workflow.stop']
    assert device.lifecycle.elapsed==0 and device.lifecycle.calls['invoke']==0
