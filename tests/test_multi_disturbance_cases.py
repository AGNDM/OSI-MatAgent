"""Replay audited reachability sequences offline; never imported by live runner."""
import json
from pathlib import Path
import pytest
from app.models import ActionProposal
from app.single_instrument import InstrumentTask
from bench.multi_disturbance_eval import MultiDisturbanceLab, metrics, validate_manifest

MANIFEST=Path('cases/evals/multi-instrument-disturbances-v1.json')
PAYLOAD=json.loads(MANIFEST.read_text())
REPLAYS=json.loads(Path('tests/fixtures/multi-disturbance-native-replays.json').read_text())
CASES=[(d,c) for d in PAYLOAD['devices'] for c in d['cases']]

def act(lab,name,args=None):
    if name=='workflow.stop':args=dict(args or {},evidence_ids=[lab._measurement['id']] if lab._measurement else [])
    return lab.execute(ActionProposal(sample_id=lab.sample_id,expected_revision=lab.revision,
        capability_id=name if name.startswith('workflow.') else lab.spec['slug']+'::'+name,
        arguments=args or {},request_id='offline-'+str(len(lab.history)),postcondition='Obtain native result'))

def wait(lab,seconds):
    while seconds>1e-8:
        chunk=min(seconds,lab.task.max_wait_seconds)
        result=act(lab,'workflow.await',{'duration_s':chunk})
        assert not lab.protocol_errors,(lab.protocol_errors,result)
        # An intervention interrupts the requested wait. Scripted offline proof
        # proceeds to the next reviewed action, exactly as a live model could.
        seconds-=chunk

def clear_terminal_pending(lab):
    if lab.pending_operation:act(lab,'workflow.poll')

@pytest.mark.parametrize('device,case',CASES,ids=[c['id'] for d,c in CASES])
def test_all_twenty_native_contracts(device,case):
    lab=MultiDisturbanceLab(InstrumentTask.model_validate(device['task']),device,case)
    act(lab,'workflow.inspect')
    for step in REPLAYS[case['id']]:
        if step['kind']=='wait':
            wait(lab,step['duration_s'])
        elif step['kind']=='invoke':
            cap=step['capability_id']
            # Explicit terminal poll is the normal lifecycle completion path.
            if lab.pending_operation and cap not in device.get('pending_safe_capabilities',[]):
                clear_terminal_pending(lab)
            result=act(lab,cap,step['arguments'])
        else:raise AssertionError('Only agent capabilities and waits belong to offline recovery proof')
    if case['expected']=='recoverable':
        from app.single_instrument import evaluate
        for _ in range(3):
            if all(evaluate(p,lab._measurement['native'])['met'] for p in lab.task.predicates):break
            wait(lab,min(60,lab.task.max_wait_seconds))
    clear_terminal_pending(lab)
    assert lab.environment_events and lab.environment_events[0]['status']=='succeeded',lab.environment_events
    assert not lab.protocol_errors,lab.protocol_errors
    if case['expected']=='honest_stop':
        act(lab,'workflow.stop',{'reason':'Public evidence cannot establish requested goal; instrument is safe.'})
        report={'status':'unresolved','evidence_ids':[]}
    else:report={'status':'supported','evidence_ids':[lab._measurement['id']]}
    result={'report':report,'score':lab.score(report),'exception':None}
    measured=metrics(lab,result)
    assert measured['passed'],(measured,result['score'])
    public=json.dumps(lab.observe())
    assert case['id'] not in public and 'environment-event-' not in public
    assert 'effect_predicates' not in public

def test_manifest_fixed_scope():
    assert len(validate_manifest(PAYLOAD))==10
    assert len(CASES)==20
    assert sum(c['expected']=='recoverable' for _,c in CASES)==12

def test_centrifuge_old_curve_cannot_be_relabelled_with_new_method():
    from copy import deepcopy
    from app.single_instrument import evaluate
    device,case=next((d,c) for d,c in CASES if c['id']=='eppendorf-startup-imbalance')
    lab=MultiDisturbanceLab(InstrumentTask.model_validate(device['task']),device,case)
    act(lab,'workflow.inspect')
    for step in deepcopy(REPLAYS[case['id']]):
        if step['kind']=='wait':wait(lab,step['duration_s'])
        else:
            if lab.pending_operation:clear_terminal_pending(lab)
            if step['capability_id']=='centrifuge.set_method':step['arguments']['temperature_k']=300.
            act(lab,step['capability_id'],step['arguments'])
    clear_terminal_pending(lab)
    act(lab,'centrifuge.set_method',{'speed_rad_s':100,'temperature_k':293.15,'duration_s':12,'acceleration_level':1,'braking_level':1})
    assert all(evaluate(p,lab._measurement['native'])['met'] for p in lab.task.predicates)
    score=lab.score({'status':'supported','evidence_ids':[lab._measurement['id']]})
    assert score['execution_method_binding_valid'] is False
    assert score['goal_met'] is False and score['false_success'] is True
