from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.single_instrument import InstrumentTask, InstrumentLab, StatePredicate, evaluate, InstrumentProfile
from app.models import ActionProposal, Limits
from app.lab_adapter import GateRejected
from app.graph import build_graph, initial_state
from app.trace import Trace


def task(device='binder_fd056', **changes):
    return InstrumentTask.model_validate(dict(task_id='test', device=device, objective='Reach the public goal',
        predicates=[dict(state_id='goal', value=True, unit='1')], **changes))


class FakeNative:
    def __init__(self):
        self.time_s=0.; self.goal=False; self.calls=[]; self.polls=[]
        self.next_response={'status':'succeeded','result':{}}
        self.poll_responses=[]; self.throw=False

    def step(self, dt): self.time_s+=dt
    def host_step(self, dt, *, world_time_s):
        assert world_time_s == self.time_s+dt
        self.time_s=world_time_s
    def read(self, **kwargs):
        assert 'truth.secret' not in kwargs['state_ids']
        return {'instance_id':'test-instance','status':'succeeded','state_revision':len(self.calls),'clock':{'time_s':self.time_s},
                'observations':{'goal':{'value':self.goal,'unit':'1','quality':'valid','sample_time_s':self.time_s}}}
    def invoke(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.throw: raise RuntimeError('uncertain transport')
        return self.envelope(self.next_response)
    def poll(self, **kwargs):
        self.polls.append(deepcopy(kwargs))
        result=self.poll_responses.pop(0)
        if isinstance(result,Exception): raise result
        if result.get('status')=='succeeded':self.goal=True
        return self.envelope(result)
    def envelope(self, result):
        return {'instance_id':'test-instance','state_revision':len(self.calls),'clock':{'time_s':self.time_s},**deepcopy(result)}
    def cancel(self, **kwargs):
        return self.envelope({'status':'cancelled','operation_id':kwargs['operation_id'],'result':{'dosing.status':'cancelled'}})


def fake_lab(device='binder_fd056'):
    native=FakeNative()
    description={'state_slots':{'goal':{'unit':'1','visibility':'agent'},'truth.secret':{'unit':'1','visibility':'debug'}},
        'capabilities':{
          'act':{'visibility':'agent','availability':'implemented',
                 'input_schema':{'type':'object','properties':{'optional':{'type':'boolean'}},'additionalProperties':False},
                 'preconditions':['Native prerequisites'], 'execution':{'mode':'synchronous'}},
          'plant.configure':{'input_schema':{'type':'object'}},
          'secret':{'visibility':'debug','input_schema':{'type':'object'}},
          'unsupported':{'availability':'unsupported','input_schema':{'type':'object'}}}}
    lab=InstrumentLab(task(device),native=(native,native,'test-instance',description))
    return lab,native


def proposal(lab, name, args=None, **changes):
    return ActionProposal(**(dict(sample_id=lab.sample_id,expected_revision=lab.revision,
        capability_id=name,arguments=args or {},request_id=str(uuid4()),postcondition='test')|changes))


def execute(lab,name,args=None,**changes):return lab.execute(proposal(lab,name,args,**changes))


@pytest.mark.parametrize('field',['reference_probe','steps','tool_menu'])
def test_tasks_reject_reference_solutions(field):
    with pytest.raises(ValidationError):task(**{field:[]})


def test_public_contract_and_optional_native_arguments():
    lab,native=fake_lab();caps=lab.capabilities()
    assert set(caps)=={'workflow.inspect','workflow.readback','workflow.await','workflow.stop','binder_fd056::act'}
    assert caps['binder_fd056::act']['preconditions']==['Native prerequisites']
    execute(lab,'workflow.inspect')
    execute(lab,'binder_fd056::act')  # optional field must stay optional
    assert native.calls[-1]['arguments']=={}
    assert 'truth.secret' not in json.dumps(lab.observe())
    with pytest.raises(GateRejected):execute(lab,'binder_fd056::act',{'optional':1})
    with pytest.raises(GateRejected):execute(lab,'binder_fd056::plant.configure')


def test_identity_revision_replay_and_clock_budget():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    p=proposal(lab,'workflow.await',{'duration_s':20})
    event=lab.execute(p);before=(native.time_s,lab.revision,lab.action_count)
    assert lab.execute(p)==event
    assert before==(native.time_s,lab.revision,lab.action_count)
    assert lab._measurement['captured_at_step']>lab.last_mutation_step
    assert lab._measurement['revision']==lab.revision
    with pytest.raises(GateRejected):lab.execute(p.model_copy(update={'arguments':{'duration_s':30}}))
    with pytest.raises(GateRejected):execute(lab,'binder_fd056::act',expected_revision=0)


@pytest.mark.parametrize('status',['succeeded','failed'])
def test_pending_native_lifecycle_uses_real_operation_status(status):
    lab,native=fake_lab('mt_quantos_q2_xpe206dr');execute(lab,'workflow.inspect')
    native.next_response={'status':'running','operation_id':'op-1','result':{'dosing.status':'running'}}
    execute(lab,lab.spec['slug']+'::act')
    assert lab.pending_operation['operation_id']=='op-1'
    assert not lab.score({'status':'supported'})['goal_met']
    with pytest.raises(GateRejected):execute(lab,lab.spec['slug']+'::act')
    native.poll_responses=[{'status':'running','operation_id':'op-1','result':{'dosing.status':'running'}},
        {'status':status,'operation_id':'op-1','result':{'dosing.status':status},'error':{'code':'PARTIAL'} if status=='failed' else None}]
    execute(lab,'workflow.poll')
    assert native.time_s==0 and lab.pending_operation
    event=execute(lab,'workflow.await',{'duration_s':10})
    assert native.time_s==10 and lab.pending_operation is None
    assert lab.score({'status':'unresolved'})['goal_met'] is (status=='succeeded')
    if status=='failed':assert event['result']['error']['code']=='PARTIAL'


def test_pending_cancel_preserves_freshness_and_no_clock_advance():
    lab,native=fake_lab('mt_quantos_q2_xpe206dr');execute(lab,'workflow.inspect')
    native.next_response={'status':'accepted','operation_id':'op-1','result':{}}
    execute(lab,lab.spec['slug']+'::act');rev=lab.revision
    execute(lab,'workflow.cancel')
    assert lab.pending_operation is None and native.time_s==0 and lab.revision>rev
    assert not lab.score({'status':'supported'})['goal_met']


@pytest.mark.parametrize('response,terminal',[({'status':'accepted','operation_id':'op'},'unsupported_pending_lifecycle'),
                                            ({'status':'accepted'},'pending_missing_operation_id')])
def test_unsupported_or_missing_operation_is_never_success(response,terminal):
    lab,native=fake_lab();execute(lab,'workflow.inspect');native.next_response=response
    execute(lab,'binder_fd056::act')
    assert lab.termination_reason==terminal and lab.pending_operation
    assert not lab.score({'status':'supported'})['goal_met']


def test_poll_error_keeps_identity_and_transport_uncertainty_invalidates():
    lab,native=fake_lab('mt_quantos_q2_xpe206dr');execute(lab,'workflow.inspect')
    native.next_response={'status':'accepted','operation_id':'op-1'};execute(lab,lab.spec['slug']+'::act')
    native.poll_responses=[{'status':'rejected','error':{'code':'UNKNOWN_OPERATION'}}]
    execute(lab,'workflow.poll')
    assert lab.pending_operation['operation_id']=='op-1'
    assert lab.termination_reason=='operation_status_uncertain'
    lab,native=fake_lab();execute(lab,'workflow.inspect');native.throw=True
    p=proposal(lab,'binder_fd056::act');lab.execute(p);lab.execute(p)
    assert len(native.calls)==1 and lab._measurement is None
    assert not lab.score({'status':'supported'})['protocol_correct']


def test_public_predicate_units_paths_quality_and_finite_values():
    p=StatePredicate(state_id='timer',path=['expired'],value=True,unit='1')
    native={'status':'succeeded','observations':{'timer':{'value':{'expired':True},'unit':'1','quality':'estimated'}}}
    assert evaluate(p,native)['met']
    for key,value in [('quality','unavailable'),('unit','K'),('value',{'expired':1}),('value',{})]:
        bad=deepcopy(native);bad['observations']['timer'][key]=value
        assert not evaluate(p,bad)['met']
    with pytest.raises(ValidationError):StatePredicate(state_id='x',value=float('nan'),unit='1')
    lab,native=fake_lab();native.goal=True;execute(lab,'workflow.inspect')
    score=lab.score({'status':'unresolved'})
    assert score['goal_met'] and score['task_success'] and not score['agent_claim_supported'] # initial goal can be satisfied, no required change
    lab._invalidate();assert not lab.score({'status':'supported'})['goal_met']


class LifecycleTestRoles:
    is_llm=False
    backend_id='offline-lifecycle-test-double'
    def invoke(self,role,prompt,payload,schema):
        if role=='diagnostic':
            o=payload['observations'];m=o.get('measurement')
            name=('workflow.inspect' if not o['inspection'] else 'workflow.await' if o.get('pending_operation') else 'mt_quantos_q2_xpe206dr::act')
            value={'hypotheses':[],'intent':{'capability_id':name,'arguments':{'duration_s':10} if name=='workflow.await' else {},'discriminating_observation':'Read native state','postcondition':'Check public goal'}}
        elif role=='procedure':
            value={k:v for k,v in payload['intent'].items() if k!='discriminating_observation'}
            value.update({k:payload[k] for k in ['sample_id','expected_revision','request_id']})
        elif role=='verification':value={'status':'supported' if payload['observations']['measurement']['native']['observations']['goal']['value'] else 'contradicted','evidence_ids':[payload['observations']['measurement']['id']],'missing_evidence':[]}
        else:value={'backend_id':self.backend_id,'choice':'diagnose'}
        return schema.model_validate(value).model_dump()


def test_existing_graph_can_await_pending_instead_of_stopping_or_verifying():
    lab,native=fake_lab('mt_quantos_q2_xpe206dr')
    native.next_response={'status':'accepted','operation_id':'op-1'}
    native.poll_responses=[{'status':'succeeded','operation_id':'op-1','result':{'dosing.status':'succeeded'}}]
    trace=Trace('test');limits=Limits(lab_actions=16,graph_steps=120)
    graph=build_graph(lab,trace,limits,roles=LifecycleTestRoles(),profile=InstrumentProfile())
    state=graph.invoke(initial_state(lab,'test',limits),{'configurable':{'thread_id':'test'},'recursion_limit':120})
    assert lab.score(state['report'])['task_success']
    assert len(native.calls)==1 and len(native.polls)==1 and native.time_s==10


def test_runner_retains_structured_limits_and_no_reference_file_loading(tmp_path,monkeypatch):
    from app import instrument_cli
    lab,native=fake_lab()
    monkeypatch.setattr(instrument_cli,'InstrumentLab',lambda task:lab)
    for label,limits in [('action',Limits(lab_actions=0)),('graph',Limits(graph_steps=1)),('deadline',Limits(episode_seconds=1e-12))]:
        result=instrument_cli.run_episode(task(),tmp_path/label,LifecycleTestRoles(),limits=limits)
        assert result['report']['terminal_reason'] in {'action_budget','graph_limit','deadline'}
        assert (tmp_path/label/'result.json').exists()
    with pytest.raises(ValueError):instrument_cli.run(task(),tmp_path/'scripted',LifecycleTestRoles())


@pytest.mark.parametrize('bad',[
    {'status':'succeeded','operation_id':'op-other'},
    {'status':'failed','operation_id':'op-other'},
    {'status':'cancelled','operation_id':'op-other'},
    {'status':'succeeded','operation_id':'op-1','instance_id':'wrong-instance'},
    {'status':'failed','error':{'code':'UNKNOWN_OPERATION'}}])
def test_terminal_poll_must_confirm_correct_operation_and_instance(bad):
    lab,native=fake_lab('mt_quantos_q2_xpe206dr');execute(lab,'workflow.inspect')
    native.next_response={'status':'running','operation_id':'op-1'};execute(lab,lab.spec['slug']+'::act')
    native.poll_responses=[bad];execute(lab,'workflow.poll')
    assert lab.pending_operation['operation_id']=='op-1'
    assert not lab.score({'status':'supported'})['goal_met']
    assert lab.termination_reason in {'operation_identity_mismatch','execution_uncertain','operation_status_uncertain'}


def test_stale_native_readback_after_clock_step_is_not_relabelled_fresh():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    original=native.read
    def stale(**kwargs):
        r=original(**kwargs);r['observations']['goal']['sample_time_s']=0
        return r
    native.read=stale
    execute(lab,'workflow.await',{'duration_s':10})
    assert lab.termination_reason=='execution_uncertain' and lab._measurement is None
    with pytest.raises(ValueError, match='STALE_NATIVE_OBSERVATION'):
        lab.score({'status':'supported'})


def test_zero_and_exhausted_budget_include_real_router_attempts():
    class CountingRouter:
        is_decision_model=True
        backend_id='counting-test-router'
        def __init__(self):self.calls=0
        def decide(self,summary,allowed):
            from app.decision_backend import RulesBackend
            self.calls+=1
            return RulesBackend().decide(summary,allowed)
    for bound in [0,1]:
        lab,_=fake_lab();router=CountingRouter();trace=Trace('budget')
        limits=Limits(llm_calls=bound)
        graph=build_graph(lab,trace,limits,roles=LifecycleTestRoles(),decision_backend=router,profile=InstrumentProfile())
        state=graph.invoke(initial_state(lab,'budget',limits),{'configurable':{'thread_id':'budget'},'recursion_limit':100})
        assert router.calls==bound
        assert state['report']['terminal_reason']=='llm_budget'
        assert state['report']['llm_call_count']==bound


def test_initialization_failure_is_saved(tmp_path,monkeypatch):
    from app import instrument_cli
    def fail(_):raise RuntimeError('fixture unavailable')
    monkeypatch.setattr(instrument_cli,'InstrumentLab',fail)
    result=instrument_cli.run_episode(task(),tmp_path/'initialization',LifecycleTestRoles())
    assert result['report']['terminal_reason']=='initialization_error'
    assert json.loads((tmp_path/'initialization/result.json').read_text())['exception']['type']=='RuntimeError'


def test_no_reference_task_read_and_goal_changes_do_not_change_adapter(monkeypatch):
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        assert path.name!='binder_fd056.json','Generated reference task must not be loaded'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    base=dict(task_id='different-goal',device='binder_fd056',objective='Change a public control',
        predicates=[dict(state_id='temperature.set_k',value=321.15,unit='K')])
    a=InstrumentLab(InstrumentTask.model_validate(base))
    base['predicates'][0]['value']=330.15
    b=InstrumentLab(InstrumentTask.model_validate(base))
    assert a.capabilities()==b.capabilities()
    state=initial_state(a,'public',Limits());state['observations']=a.observe()
    payload=InstrumentProfile.projection('diagnostic',state,'request')
    text=json.dumps(payload)
    for forbidden in ['reference_probe','native_fixture_events','truth.','debug.']:
        assert forbidden not in text


@pytest.mark.parametrize('timestamp', ['missing', None, True, 0.0])
def test_evidence_requires_current_finite_native_sample_timestamp(timestamp):
    lab,native=fake_lab();execute(lab,'workflow.inspect');original=native.read
    def changed(**kwargs):
        r=original(**kwargs)
        if timestamp=='missing':r['observations']['goal'].pop('sample_time_s')
        else:r['observations']['goal']['sample_time_s']=timestamp
        return r
    native.read=changed
    execute(lab,'workflow.await',{'duration_s':10})
    assert lab.termination_reason=='execution_uncertain' and lab._measurement is None


def test_final_read_failure_keeps_pending_identity_in_saved_result(tmp_path,monkeypatch):
    from app import instrument_cli
    lab,native=fake_lab('mt_quantos_q2_xpe206dr')
    lab.pending_operation={'operation_id':'retained-op','status':'running'}
    def fail_score(report):raise ValueError('unconfirmed final read')
    monkeypatch.setattr(lab,'score',fail_score)
    monkeypatch.setattr(instrument_cli,'InstrumentLab',lambda task:lab)
    result=instrument_cli.run_episode(task(),tmp_path/'uncertain-final',LifecycleTestRoles(),limits=Limits(lab_actions=0))
    assert result['score']['pending_operation']['operation_id']=='retained-op'
    assert not result['score']['lifecycle_resolved']
