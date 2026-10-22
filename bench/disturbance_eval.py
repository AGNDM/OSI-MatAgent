"""Five bounded development episodes; hidden host interventions, public observations."""
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
import time

from app.device_registry import get_device, open_native
from app.single_instrument import InstrumentLab, InstrumentTask
from app.models import Limits

MANIFEST=Path('cases/evals/furnace-disturbances-v1.json')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class RecordingScene:
    """Synchronous state delivery receipts; no renderer or physical scene."""
    def __init__(self):
        self.receipts=[]
    def __call__(self, instance_id, updates):
        self.receipts.append({'instance_id':instance_id,'updates':deepcopy(updates)})


class DisturbanceLab(InstrumentLab):
    scene_callback_description='Host-side synchronous recording adapter; no visual rendering or physical hardware'
    def __init__(self, task, case):
        self.case=deepcopy(case)
        self.scene=RecordingScene()
        self.environment_events=[]
        self.action_timings=[]
        self.created_wall=time.monotonic()
        native=open_native(get_device(task.device),scene_adapter=self.scene)
        super().__init__(task,native=native)
        self.host.register_principal('environment',visibility='evaluator')
        self.environment=self.host.session('environment')

    def execute(self, proposal):
        pre=deepcopy(self._measurement['native']) if self._measurement else None
        wall=time.monotonic()-self.created_wall
        event=super().execute(proposal)
        self.action_timings.append({'request_id':proposal.request_id,'wall_seconds':wall,'pre':pre,'post':deepcopy(self._measurement['native']) if self._measurement else None})
        return event

    def score(self, report):
        result=super().score(report)
        final=self._read()
        exactly_one=len(final['observations']['program.segments']['value'])==1
        result['exactly_one_segment']=exactly_one
        result['goal_met']=result['task_success']=result['goal_met'] and exactly_one
        result['false_success']=result['false_success'] or (report.get('status')=='supported' and not exactly_one)
        return result

    def after_clock_step(self):
        if self.environment_events:
            return False
        pre=self._read()
        values={k:v['value'] for k,v in pre['observations'].items()}
        phase=values['program.phase']
        hit=(phase=='ramp' and self.lifecycle.elapsed>=90) if self.case['trigger']=='ramp90' else (phase=='hold' and values['program.hold_elapsed_s']>=20)
        if not hit:
            return False
        event={'id':self.case['id'],'simulated_seconds':self.lifecycle.elapsed,'wall_seconds':time.monotonic()-self.created_wall,'action_step':self.action_count,'pre':pre,'receipts':[]}
        # One serialized event, including the compound F5, before wrapper readback.
        self.environment_events.append(event)
        for index,(cap,args) in enumerate(self.case['actions']):
            response=self.environment.invoke(instance_id=self.instance_id,capability_id=cap,arguments=deepcopy(args),request_id=f'environment-{index}')
            event['receipts'].append({'capability_id':cap,'arguments':args,'response':response})
            if response.get('status')!='succeeded':
                raise RuntimeError('Environment intervention rejected')
        event['post']=self._read()
        return True


def fingerprint():
    paths=list(Path('app').glob('*.py'))+[Path(__file__), MANIFEST,Path('configs/openrouter-deepseek-v4.1-flash.json')]
    return {str(p):sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def metrics(lab, result):
    events=lab.environment_events
    after=[h for h in lab.history if events and h['step']>events[0]['action_step']+1]
    native=[h for h in after if '::' in h['proposal']['capability_id']]
    waits=[h for h in after if h['proposal']['capability_id']=='workflow.await']
    starts=[h for h in native if h['proposal']['capability_id'].endswith('::thermal.start_program')]
    first=native[0] if native else None
    return {'triggered':bool(events),'intervention_succeeded':bool(events and all(r['response']['status']=='succeeded' for r in events[0]['receipts'])),
        'eval_success':bool(events and events[0].get('post') and all(r['response']['status']=='succeeded' for r in events[0]['receipts']) and result['score'].get('goal_met') and result['score'].get('fresh_evidence') and result['score'].get('evidence_cited') and result['score'].get('agent_claim_supported') and result['score'].get('protocol_correct') and not result.get('exception')),
        'first_action_after_event':after[0]['proposal'] if after else None,'first_native_action_after_event':first['proposal'] if first else None,
        'first_native_action_simulated_seconds':first['result'].get('native_response',{}).get('clock',{}).get('time_s') if first else None,
        'native_actions_after_event':[{'step':h['step'],'proposal':h['proposal'],'error':h['result'].get('error')} for h in native],
        'wait_count_after_event':len(waits),
        'ineffective_waits_after_event':[h['proposal']['request_id'] for h in waits if any(t['request_id']==h['proposal']['request_id'] and t['pre'] and not lab.active_goal_process(t['pre']) for t in lab.action_timings)],
        'first_native_action_wall_latency_seconds':next((t['wall_seconds']-events[0]['wall_seconds'] for t in lab.action_timings if first and t['request_id']==first['proposal']['request_id']),None),'restart_count_after_event':len(starts),
        'rejected_native_actions_after_event':sum(bool(h['result'].get('error')) for h in native),
        'simulated_seconds':lab.lifecycle.elapsed,'wall_seconds':result.get('elapsed_seconds')}


def live(output):
    from bench.acceptance import backends
    from app.instrument_cli import run_episode
    from app.token_budget import TokenLedger
    payload=json.loads(MANIFEST.read_text()); frozen=fingerprint()
    dump(output/'freeze.json',{'source_hashes':frozen,'git_head':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'scope':'Five development samples, not a statistical benchmark','manifest':payload})
    ledger=TokenLedger();ledger.enable_openrouter_record_only()
    dump(output/'budget-before.json',ledger.snapshot())
    state=ledger.snapshot()
    if state['inflight_or_crashed'] or state['accounting_violation']:
        raise RuntimeError('Unresolved ledger state; no requests sent')
    rows=[]
    for case in payload['cases']:
        if fingerprint()!=frozen:raise RuntimeError('Frozen implementation changed')
        from uuid import uuid4
        attempt=str(uuid4())
        dump(output/(case['id']+'-attempt.json'),{'id':attempt,'case':case['id'],'started':True})
        roles,router,operation=backends(ledger)
        holder=[]
        def factory(task):
            lab=DisturbanceLab(task,case);holder.append(lab);return lab
        directory=output/case['id']
        result=run_episode(InstrumentTask.model_validate(payload['task']),directory,roles,router=router,operation=operation,
            limits=Limits(lab_actions=64,llm_calls=64,episode_seconds=600,graph_steps=400),lab_factory=factory)
        lab=holder[0]
        dump(directory/'environment-events.json',lab.environment_events)
        dump(directory/'scene-receipts.json',lab.scene.receipts)
        dump(directory/'action-timings.json',lab.action_timings)
        measured=metrics(lab,result);dump(directory/'metrics.json',measured)
        row={'case':case['id'],'attempt':attempt,'metrics':measured,'result':result};rows.append(row)
        dump(output/'summary.json',{'attempted':len(rows),'runs':rows})
        dump(output/'budget-after.json',ledger.snapshot())
        print(json.dumps({'case':case['id'],'metrics':measured,'usage':result.get('usage'),'exception':result.get('exception')}),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('--execute is required for the five authorized paid episodes')
    if args.output.exists():parser.error('Use a new output directory')
    args.output.mkdir(parents=True);live(args.output)

if __name__=='__main__':main()
