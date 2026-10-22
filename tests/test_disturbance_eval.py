import json
from pathlib import Path
import pytest
from bench.disturbance_eval import DisturbanceLab, MANIFEST
from app.single_instrument import InstrumentLab, InstrumentTask
from app.models import ActionProposal

DATA=json.loads(MANIFEST.read_text())

def act(lab, name, args=None):
    return lab.execute(ActionProposal(sample_id=lab.sample_id,expected_revision=lab.revision,capability_id=name if name.startswith('workflow.') else lab.spec['slug']+'::'+name,arguments=args or {},postcondition='Observe native result',request_id='test-'+str(len(lab.history))))

def start(lab):
    act(lab,'workflow.inspect')
    act(lab,'power.set',{'enabled':True})
    act(lab,'thermal.configure_program',{'segments':[{'temperature_K':323.15,'ramp_rate_K_s':.1,'hold_s':60}]})
    act(lab,'thermal.start_program')

@pytest.mark.parametrize('case', DATA['cases'],ids=lambda c:c['id'])
def test_trigger_and_native_recovery(case):
    lab=DisturbanceLab(InstrumentTask.model_validate(DATA['task']),case)
    start(lab)
    while not lab.environment_events and lab.lifecycle.elapsed<800:
        event=act(lab,'workflow.await',{'duration_s':120})
        assert not event['result'].get('error')
    assert len(lab.environment_events)==1
    env=lab.environment_events[0]
    assert env['pre']['observations']['program.phase']['value'] in {'ramp','hold'}
    assert env['post']['state_revision']>env['pre']['state_revision']
    assert not lab.active_goal_process(lab._measurement['native'])
    assert lab._measurement['revision']==lab.revision
    assert not lab.protocol_errors
    # Offline reachability proof only: no scripted recovery exists in live runner.
    values={k:v['value'] for k,v in env['post']['observations'].items()}
    if not values['door.closed']:act(lab,'door.close')
    if not values['power.enabled']:act(lab,'power.set',{'enabled':True})
    if not values['program.running']:act(lab,'thermal.start_program')
    while lab._measurement['native']['observations']['program.phase']['value']!='completed' and lab.lifecycle.elapsed<1600:
        act(lab,'workflow.await',{'duration_s':120})
    score=lab.score({'status':'supported','evidence_ids':[lab._measurement['id']]})
    assert score['goal_met'] and score['protocol_correct'] and score['fresh_evidence']
    assert len(lab.environment_events)==1
    public=json.dumps(lab.observe())
    assert 'environment-' not in public and 'ramp90' not in public and 'hold20' not in public

def test_default_scene_prerequisite_and_paused_guard():
    lab=InstrumentLab(InstrumentTask.model_validate(DATA['task']))
    act(lab,'workflow.inspect')
    assert 'nabertherm_l9::door.open' not in lab.capabilities()
    response=lab.session.invoke(instance_id=lab.instance_id,capability_id='door.open',arguments={},request_id='default-scene-test')
    assert response['status']!='succeeded'
    assert not lab.active_goal_process({'observations':{'program.phase':{'value':'paused_door','quality':'valid'},'program.running':{'value':True,'quality':'valid'}}})
