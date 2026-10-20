"""Author one native tool-call task per device using a recorded reference probe.

These are generated integration fixtures, not scientific benchmark ground truth.
Every accepted action and changed public value is retained in reference_probe.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.device_registry import open_native, public_capabilities


def priority(item):
    name, cap = item
    useful = any(s in name for s in ['temperature', 'speed', 'wavelength', 'flow_rate', 'filter', 'unit', 'humidity', 'pressure'])
    scene = any(s in name for s in ['door', 'lid', 'sash', 'cover', 'position'])
    return (0 if useful else 2 if name == 'power.set' else 1 if scene else 3, len(cap.get('input_schema', {}).get('required', [])), name)


def build_task(spec):
    _, _, _, desc = open_native(spec)
    caps = public_capabilities(desc)
    candidates = [(k,v) for k,v in caps.items() if isinstance(v.get('example_arguments'),dict)
        and v.get('kind') not in {'query', 'acquisition', 'simulation'}
        and not k.endswith(('.start', '.run'))
        and v.get('execution',{}).get('mode','synchronous') == 'synchronous']
    failures = []
    for name, cap in sorted(candidates, key=priority):
        args = cap['example_arguments']
        if len(json.dumps(args)) > 1500:
            continue
        for powered in [False, True] if name != 'power.set' and 'power.set' in caps else [False]:
            try:
                host, session, instance, _ = open_native(spec)
                before = session.read(instance_id=instance)
                records = []
                if powered:
                    p = caps['power.set']['example_arguments']
                    response = session.invoke(instance_id=instance, capability_id='power.set', arguments=p, request_id='reference-power')
                    records.append({'capability_id':'power.set','arguments':p,'response':response})
                    if response.get('status') != 'succeeded':
                        continue
                response = session.invoke(instance_id=instance,capability_id=name, arguments=args,request_id='reference-target')
                records.append({'capability_id':name,'arguments':args,'response':response})
                if response.get('status') != 'succeeded':
                    failures.append({'capability_id':name,'powered':powered,'response':response})
                    continue
                after = session.read(instance_id=instance)
                delta = {k:v for k,v in after['observations'].items() if k in before['observations']
                    and v.get('value') != before['observations'][k].get('value') and v.get('quality') in {'valid','estimated'}}
                # Prefer a directly written parameter, rather than an incidental status change.
                matching = [k for k,v in delta.items() if any(type(v['value']) == type(a) and v['value'] == a
                    or isinstance(v['value'],(float,int)) and not isinstance(v['value'],bool)
                    and isinstance(a,(float,int)) and not isinstance(a,bool) and abs(v['value']-a)<1e-9 for a in args.values())]
                declared = [e.get('state_id') for e in cap.get('state_effects',[]) if isinstance(e,dict)]
                choices = [k for k in matching if k in declared] or matching
                if not choices:
                    continue
                state_id = choices[0]
                desired = deepcopy(delta[state_id]['value'])
                tool_menu = list(dict.fromkeys((["power.set"] if 'power.set' in caps else []) + [name]))
                return {'task_id':'device-'+spec['slug'], 'asset_id':spec['asset_id'],
                    'title':f"{spec['name']}: set {state_id} and verify public readback",
                    'scope':'Native control/interface smoke; not acquisition, chemistry or hardware validation',
                    'goal':{'state_id':state_id,'equals':desired,'unit':delta[state_id].get('unit'),'absolute_tolerance':1e-9},
                    'tool_menu':tool_menu,'reference_probe':{'environment_setup':host._workflow_fixture_events,'before':before,'steps':records,'after':after},
                    'candidate_rejections':failures}
            except Exception as exc:
                failures.append({'capability_id':name,'powered':powered,'error':f'{type(exc).__name__}: {exc}'})
    raise ValueError('No verified state-changing synchronous example: '+json.dumps(failures,ensure_ascii=False)[:1500])


if __name__ == '__main__':
    directory=ROOT/'cases/devices'
    data=json.loads((directory/'inventory.json').read_text())
    accepted=[]
    for spec in data['devices']:
        try:
            task=build_task(spec)
            path=directory/(spec['slug']+'.json')
            path.write_text(json.dumps(task,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            accepted.append(dict(spec,task_file=path.name))
            print(spec['slug'],task['goal']['state_id'],flush=True)
        except Exception as exc:
            data['unavailable'].append(dict(spec,error='task generation: '+str(exc)))
            print('UNAVAILABLE',spec['slug'],str(exc)[:140],flush=True)
    data['devices']=accepted
    (directory/'registry.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('TASKS',len(accepted),'UNAVAILABLE',len(data['unavailable']))
