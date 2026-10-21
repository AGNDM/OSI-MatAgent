"""OFFLINE TEST DOUBLES ONLY. Never imported by a real-model execution path."""
import json
from pathlib import Path
from uuid import uuid4

from app.models import ActionProposal,Limits
from app.single_instrument import InstrumentTask,InstrumentLab,verification_status
from app.instrument_cli import run_episode
from app.graph import build_graph,initial_state
from app.trace import Trace


def cases():
    return {
      'balance-filter':('mt_xpr204e',[('balance.set_filter',{'level':'High'})]),
      'ika-stirring':('ika_plate_rctdigital',[('power.set',{'on':True}),('stirring.set_speed',{'speed_rad_s':10.471975511965978})]),
      'binder-timer':('binder_fd056',[('power.set',{'on':True}),('timer.set',{'days':0,'hours':0,'minutes':1}),('workflow.await',{'duration_s':60})]),
      'furnace-program':('nabertherm_l9',[('power.set',{'enabled':True}),('thermal.configure_program',{'segments':[{'temperature_K':323.15,'ramp_rate_K_s':.1,'hold_s':10}]}),('thermal.start_program',{}),('workflow.await',{'duration_s':300})])}


class NativeTestRoles:
    is_llm=False;backend_id='offline-native-regression-double'
    def __init__(self,slug,actions):self.slug=slug;self.actions=list(actions)
    def invoke(self,role,prompt,payload,schema):
        if role=='diagnostic':
            if not payload['observations'].get('inspection'):name,args='workflow.inspect',{}
            elif self.actions:name,args=self.actions.pop(0)
            else:name,args='workflow.readback',{}
            cap=name if name.startswith('workflow.') else self.slug+'::'+name
            return {'hypotheses':[],'intent':{'capability_id':cap,'arguments':args,
                'discriminating_observation':'Offline fixture readback','postcondition':'Offline regression only'}}
        if role=='procedure':
            return {k:v for k,v in dict(payload['intent'],sample_id=payload['sample_id'],
                expected_revision=payload['expected_revision'],request_id=payload['request_id']).items() if k!='discriminating_observation'}
        if role=='verification':
            return {'status':verification_status(payload),'evidence_ids':[payload['observations']['measurement']['id']],
                    'missing_evidence':[]}
        raise AssertionError('No deliberation')


def run_checks(output):
    rows=[]
    plan=cases()
    for repeat in range(2):
        for name,(slug,actions) in plan.items():
            task=InstrumentTask.model_validate_json(Path(f'cases/acceptance/v1/{name}.json').read_text()).model_copy(update={'device':slug})
            r=run_episode(task,output/f'{name}-{repeat}',NativeTestRoles(slug,actions))
            rows.append({'id':name,'repeat':repeat,'track':'native-scripted-regression',
                         'accepted':r['score']['goal_met'] and r['score']['protocol_correct']})
        task=InstrumentTask(task_id='acceptable-control',device='mt_xpr204e',objective='Confirm the already selected native filter',
                            predicates=[{'state_id':'balance.filter_level','unit':'1','value':'Medium'}])
        r=run_episode(task,output/f'acceptable-{repeat}',NativeTestRoles('mt_xpr204e',[]))
        rows.append({'id':'acceptable','repeat':repeat,'track':'native-scripted-regression','accepted':r['score']['goal_met']})
        task=InstrumentTask(task_id='unsupported-material-state',device='auto',objective='Establish a material phase absent from native public states',
                            predicates=[{'state_id':'material.rutile_present','unit':'1','value':True}])
        r=run_episode(task,output/f'unsupported-{repeat}',NativeTestRoles('none',[]),operation=type('NeverChooser',(),{'is_llm':False,'backend_id':'never-used'})())
        rows.append({'id':'unsupported','repeat':repeat,'track':'native-scripted-regression','accepted':r['report']['terminal_reason']=='unsupported_objective'})
    wrong={
      'balance-filter':[('balance.set_filter',{'level':'Low'})],
      'ika-stirring':[('stirring.set_speed',{'speed_rad_s':0})],
      'binder-timer':[('power.set',{'on':True}),('timer.set',{'days':0,'hours':0,'minutes':0})],
      'furnace-program':[('thermal.configure_program',{'segments':[{'temperature_K':323.15,'ramp_rate_K_s':.1,'hold_s':0}]})]}
    for name,actions in wrong.items():
        slug=plan[name][0];task=InstrumentTask.model_validate_json(Path(f'cases/acceptance/v1/{name}.json').read_text()).model_copy(update={'device':slug})
        lab=InstrumentLab(task)
        for cap,args in [('workflow.inspect',{})]+actions:
            lab.execute(ActionProposal(sample_id=lab.sample_id,expected_revision=lab.revision,
                capability_id=cap if cap.startswith('workflow.') else slug+'::'+cap,arguments=args,
                request_id=str(uuid4()),postcondition='Deliberately wrong offline control'))
        m=lab.observe()['measurement'];r=lab.score({'status':'supported','evidence_ids':[m['id']]})
        rows.append({'id':'wrong-'+name,'track':'negative-control','accepted':not r['goal_met'] and r['false_success']})
    return rows
