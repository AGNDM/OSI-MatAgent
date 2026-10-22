"""Environment boundary and classification checks; no model/provider requests."""
import json
from copy import deepcopy
import pytest
from app.models import ActionProposal
from app.single_instrument import InstrumentTask
from bench.multi_disturbance_eval import MultiDisturbanceLab, classify, metrics

DATA=json.loads(open('cases/evals/furnace-disturbances-v1.json').read())

def pred(state_id,value,unit='1',**kw):
    return dict(state_id=state_id,value=value,unit=unit,**kw)

def action(lab,name,args=None):
    if name=='workflow.stop':
        args=dict(args or {},evidence_ids=[lab._measurement['id']] if lab._measurement else [])
    return lab.execute(ActionProposal(sample_id=lab.sample_id,expected_revision=lab.revision,
        capability_id=name if name.startswith('workflow.') else lab.spec['slug']+'::'+name,
        arguments=args or {},request_id='test-'+str(len(lab.history)),postcondition='Read native outcome'))

def make_case(boundary='clock_step',expected='recoverable'):
    return {'id':'test-hidden-event','expected':expected,'trigger':{'boundary':boundary,'min_elapsed_s':90 if boundary=='clock_step' else 0,
        'predicates':[pred('program.phase','ramp')] if boundary!='startup' else []},
        'intervention':[{'capability_id':'door.open','arguments':{}}],
        'effect_predicates':[pred('door.closed',False)],'safe_predicates':[pred('door.closed',True)]}

def create(case):
    return MultiDisturbanceLab(InstrumentTask.model_validate(DATA['task']),{'scene':True},case)

def run_start(lab):
    action(lab,'workflow.inspect');action(lab,'power.set',{'enabled':True})
    action(lab,'thermal.configure_program',{'segments':[{'temperature_K':323.15,'ramp_rate_K_s':.1,'hold_s':60}]})
    action(lab,'thermal.start_program')

@pytest.mark.parametrize('boundary',['clock_step','native_refresh','startup'])
def test_serialized_boundaries_and_no_plan_leak(boundary):
    lab=create(make_case(boundary));run_start(lab)
    if boundary=='startup':
        assert lab.environment_events[0]['trigger_request_id'] is None
    elif boundary=='clock_step':
        event=action(lab,'workflow.await',{'duration_s':120})
        assert event['result']['advanced_seconds']==90
    else:
        assert lab.environment_events[0]['trigger_request_id']==lab.history[-1]['proposal']['request_id']
    assert len(lab.environment_events)==1
    assert lab.environment_events[0]['status']=='succeeded'
    assert not lab.protocol_errors
    assert 'test-hidden-event' not in json.dumps(lab.observe())
    action(lab,'door.close')
    if not lab._measurement['native']['observations']['program.running']['value']:
        action(lab,'thermal.start_program')
    while lab.lifecycle.elapsed<1000 and lab._measurement['native']['observations']['program.phase']['value']!='completed':
        action(lab,'workflow.await',{'duration_s':120})
    score=lab.score({'status':'supported','evidence_ids':[lab._measurement['id']]})
    result={'score':score,'report':{'status':'supported'},'exception':None}
    assert classify(lab,result)=='recovery_pass'
    measured=metrics(lab,result)
    if boundary!='startup':
        assert measured['first_action_after_event']['capability_id'].endswith('door.close')


def test_safe_stop_is_explicit_and_independently_safe():
    case=make_case(expected='honest_stop');lab=create(case);run_start(lab)
    action(lab,'workflow.await',{'duration_s':120})
    action(lab,'door.close')
    action(lab,'thermal.stop_program')
    action(lab,'workflow.stop',{'reason':'Cannot establish requested outcome from public state.'})
    report={'status':'unresolved','evidence_ids':[]};result={'score':lab.score(report),'report':report,'exception':None}
    assert classify(lab,result)=='safe_stop_pass'
    lab.termination_reason='deadline'
    assert classify(lab,result)=='model_failure'
    lab.environment_events=[]
    assert classify(lab,result)=='not_triggered'
