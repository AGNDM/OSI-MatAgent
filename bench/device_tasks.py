"""Run single-device tool-call tasks through the existing LangGraph.

Default scripted mode makes no API requests. --backend deepseek is explicit,
uses the configured general expert (OpenRouter by default) and the shared durable 5M ceiling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

from app.device_lab import DeviceLab
from app.device_registry import REGISTRY, load_registry
from app.device_roles import DeviceProfile, ScriptedDeviceRoles
from app.graph import build_graph, initial_state
from app.local_decision import LocalOpenJevBackend
from app.models import Limits
from app.trace import Trace


def run_task(spec,task,output,roles=None,decision=None):
    run_id=str(uuid4())
    path=output/'traces'/(run_id+'.jsonl')
    trace=Trace(run_id,path)
    lab=DeviceLab(spec,task)
    backend=roles or ScriptedDeviceRoles(task,spec['slug'])
    limits=Limits(lab_actions=6,llm_calls=12,episode_seconds=180,graph_steps=80)
    started=time.monotonic()
    graph=build_graph(lab,trace,limits,roles=backend,decision_backend=decision,profile=DeviceProfile())
    final=graph.invoke(initial_state(lab,run_id,limits),{'configurable':{'thread_id':run_id},'recursion_limit':limits.graph_steps})
    report=final['report'];score=lab.score(report)
    trace.emit('evaluator','device_score',score=score)
    usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    complete=True
    for event in trace.events:
        m=event.get('llm_metadata') or {}
        if not m.get('request_sent'):
            continue
        if not m.get('usage'):
            complete=False
        for key in usage:
            usage[key]+=((m.get('usage') or {}).get(key) or 0)
    return {'run_id':run_id,'asset_id':spec['asset_id'],'slug':spec['slug'],'chem_id':spec['chem_id'],
        'name':spec['name'],'domain':spec['domain'],'task_id':task['task_id'],'goal':task['goal'],
        'backend':backend.backend_id,'router':decision.backend_id if decision else 'rules-v1',
        'report':report,'score':score,'usage':usage,'usage_complete':complete,
        'elapsed_seconds':time.monotonic()-started,'trace_path':str(path),
        'router_fallbacks':sum(e.get('decision',{}).get('fallback_used',False) for e in trace.events if e['kind']=='routing'),
        'gate_rejections':sum(e['kind']=='rejected' for e in trace.events)}


def write_report(output,rows,ledger=None):
    summary={'attempted':len(rows),'success':sum(r.get('score',{}).get('task_success',False) for r in rows),
        'exceptions':sum('error' in r for r in rows),'usage':{k:sum(r.get('usage',{}).get(k,0) for r in rows)
            for k in ['prompt_tokens','completion_tokens','total_tokens']},
        'budget':ledger.snapshot() if ledger else None,
        'scope':'Generated native control tasks, not scientific completion or multi-device transfers'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Native-device workflow tool-call report','',summary['scope'],'',
        f"Attempts: {summary['attempted']}; successful tasks: {summary['success']}; infrastructure exceptions: {summary['exceptions']}.",'',
        '| Device | Goal | Success | Native goal | Calls | Seconds | Tokens | Terminal |','|---|---|---|---|---|---|---|---|']
    for r in sorted(rows,key=lambda x:x['chem_id']):
        s=r.get('score',{});p=r.get('report',{})
        lines.append(f"| {r['chem_id']} {r['name']} | {r.get('goal',{}).get('state_id','')} | {s.get('task_success',False)} | {s.get('native_goal_met','')} | {p.get('llm_call_count',0)} | {r.get('elapsed_seconds',0):.2f} | {r.get('usage',{}).get('total_tokens',0)} | {p.get('terminal_reason',r.get('error',''))} |")
    if ledger:lines+=['','Budget ledger: `'+json.dumps(ledger.snapshot())+'`']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['scripted','deepseek'],default='scripted')
    p.add_argument('--router',choices=['rules','openjev'],default='rules')
    p.add_argument('--device',action='append',help='slug or CHEM ID; repeatable; default all registered tasks')
    p.add_argument('--sample',type=Path,help='Predeclared device selection JSON')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/openrouter-deepseek-v4.1-flash.json'))
    p.add_argument('--workers',type=int,choices=range(1,5),default=1)
    args=p.parse_args()
    if args.sample:
        if args.device:p.error('Use --sample or --device, not both')
        args.device=json.loads(args.sample.read_text())['devices']
        if not isinstance(args.device,list) or not args.device or any(not isinstance(s,str) for s in args.device):
            p.error('Sample must contain a nonempty list of device keys')
    if args.output.exists():p.error('Use a new output directory; attempts are never overwritten')
    if args.router=='openjev' and args.workers!=1:p.error('Local router runs sequentially')
    registry=load_registry()
    specs=[s for s in registry['devices'] if not args.device or s['slug'] in args.device or s['chem_id'] in args.device]
    if args.device and any(not any(k in {s['slug'],s['chem_id']} for s in specs) for k in args.device):
        p.error('Requested device has no registered verified task')
    ledger=config=None
    if args.backend=='deepseek':
        from app.expert_config import load_plan
        from app.token_budget import TokenLedger
        config=load_plan(args.config).reference
        ledger=TokenLedger()
        # Fail before output/threads when credentials are absent.
        from app.token_budget import BudgetedExpert
        BudgetedExpert(config,ledger)
    args.output.mkdir(parents=True)
    sources={str(f):sha256(f.read_bytes()).hexdigest() for directory in ['app','cases/devices','bench']
        for f in Path(directory).rglob('*') if f.is_file() and f.suffix in {'.py','.json'}}
    manifest={'backend':args.backend,'router':args.router,'planned':len(specs),'devices':[s['slug'] for s in specs],
        'registry':registry,'source_hashes':sources,'workers':args.workers,
        'budget_ledger':str(ledger.path) if ledger else None,'budget_at_start':ledger.snapshot() if ledger else None,
        'expert':config.model_dump() if config else None,'transport_retries':0,
        'scope':'One independently reset native control task per device; no geometry/hardware/scientific acquisition claims'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    # Freeze task inputs in the report directory as well as retaining their hashes.
    tasks={s['slug']:json.loads((REGISTRY.parent/s['task_file']).read_text()) for s in specs}
    (args.output/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2)+'\n')
    stopped=threading.Event()
    def run(spec):
        if stopped.is_set():return None
        roles=None
        if config:
            from app.token_budget import BudgetedExpert
            roles=BudgetedExpert(config,ledger)
        decision=LocalOpenJevBackend() if args.router=='openjev' else None
        try:
            row=run_task(spec,tasks[spec['slug']],args.output,roles,decision)
            if roles and (roles.last_call or {}).get('http_status') in {401,402,403,429}:
                stopped.set()
            if ledger and ledger.snapshot()['remaining_tokens']<30_000:stopped.set()
            return row
        except Exception as exc:
            if ledger:stopped.set()  # uncertain billable failure: no queued fanout
            return {k:spec[k] for k in ['slug','chem_id','name','domain','asset_id']}|{'error':str(exc)}
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,s) for s in specs]
        for future in as_completed(futures):
            row=future.result()
            if row is None:continue
            rows.append(row)
            with (args.output/'runs.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            write_report(args.output,rows,ledger)
            print(f"{len(rows)}/{len(specs)} {row['slug']}: success={row.get('score',{}).get('task_success',False)} tokens={row.get('usage',{}).get('total_tokens',0)} reason={row.get('report',{}).get('terminal_reason',row.get('error'))}",flush=True)
    write_report(args.output,rows,ledger)


if __name__=='__main__':main()
