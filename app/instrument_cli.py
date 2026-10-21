"""Run one declarative instrument task through real-model LangGraph roles.

No scripted backend, generated reference task, or task-specific action program.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from app.single_instrument import InstrumentLab, InstrumentProfile, InstrumentTask
from app.graph import build_graph, initial_state
from app.models import Limits
from app.trace import Trace


def run(task, output, roles, *, router=None, operation=None, limits=None, router_wait=True, profile=None):
    if roles is None or not roles.is_llm:
        raise ValueError('This runner requires an explicit real-model backend')
    return run_episode(task, output, roles, router=router, operation=operation, limits=limits, router_wait=router_wait,profile=profile)


def run_auto_episode(task, output, roles, *, router, operation, limits=None, router_wait=True, profile=None):
    """One shared budget/deadline for discovery, selection and bound execution."""
    from app.device_selection import select_device
    from app.device_catalog import CACHE
    if output.exists():raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    limits=limits or Limits(lab_actions=32,llm_calls=48,episode_seconds=360,graph_steps=260)
    trace=Trace(str(uuid4()),output/'selection-trace.jsonl')
    started=time.monotonic()
    initial_manifest={'task':task.model_dump(),'routing_policy':'argmax-v1','limits':limits.model_dump(),
        'expert':roles.backend_id,'router':getattr(router,'backend_id',None),'operation':operation.backend_id,
        'catalog_sha256':sha256(CACHE.read_bytes()).hexdigest(),'source_hashes':{
            str(p):sha256(p.read_bytes()).hexdigest() for p in
            list(Path('app').glob('*.py'))+list(Path('bench').glob('*.py'))+list(Path('configs').glob('*.json'))+
            [Path('cases/devices/registry.json')]},
        'scope':'One-time device selection; no cross-instrument transfer'}
    (output/'manifest.json').write_text(json.dumps(initial_manifest,indent=2)+'\n')
    try:
        selected,selection=select_device(task,roles,operation,trace,limits)
        left=limits.llm_calls-selection['model_calls']
        seconds=limits.episode_seconds-(time.monotonic()-started)
        if left<=0 or seconds<=0:raise RuntimeError('device_selection_exhausted_episode')
        result=run_episode(selected,output/'execution',roles,router=router,operation=operation,
            limits=limits.model_copy(update={'llm_calls':left,'episode_seconds':seconds}),router_wait=router_wait,
            absolute_deadline=started+limits.episode_seconds,profile=profile)
        result['selection']=selection
        result['report']['llm_call_count']=result['report'].get('llm_call_count',0)+selection['model_calls']
    except Exception as exc:
        result={'task_id':task.task_id,'report':{'status':'unresolved','evidence_ids':[],
            'terminal_reason':('unsupported_objective' if type(exc).__name__=='UnsupportedObjective' else getattr(exc,'code','device_selection_error')),'llm_call_count':sum(
                int(roles.is_llm) if e['component']=='device_discovery' else int(operation.is_llm)
                for e in trace.events if e['kind'] in {'role_output','role_error','device_selected','selection_error'})},'score':{'goal_met':False,'task_success':False,
            'protocol_correct':type(exc).__name__=='UnsupportedObjective','false_success':False},'exception':{'type':type(exc).__name__,'message':str(exc)[:1000]}}
    from app.token_budget import valid_usage
    selection_usage={k:0 for k in ['prompt_tokens','completion_tokens','total_tokens']};complete=True
    for event in trace.events:
        meta=event.get('llm_metadata') or event.get('backend_metadata') or {}
        if meta.get('request_sent'):
            if not valid_usage(meta.get('usage')):complete=False;continue
            for key in selection_usage:selection_usage[key]+=meta['usage'][key]
    result['selection_usage']=selection_usage
    result['usage']={k:result.get('usage',{}).get(k,0)+v for k,v in selection_usage.items()}
    result['usage_complete']=complete and result.get('usage_complete',True)
    result['elapsed_seconds']=time.monotonic()-started
    result['reference_scripts_used']=False
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def run_episode(task, output, roles, *, router=None, operation=None, limits=None, router_wait=True, absolute_deadline=None, profile=None):
    """Offline tests inject doubles here; public run()/CLI require real roles."""
    if task.device=='auto':
        if operation is None:raise ValueError('Auto device selection requires an explicit chooser')
        return run_auto_episode(task,output,roles,router=router,operation=operation,limits=limits,router_wait=router_wait,profile=profile)
    if output.exists():
        raise ValueError('Use a new output directory; attempts are never overwritten')
    limits = limits or Limits(lab_actions=32, llm_calls=36, episode_seconds=360, graph_steps=260)
    output.mkdir(parents=True)
    episode_deadline=absolute_deadline if absolute_deadline is not None else time.monotonic()+limits.episode_seconds
    episode = str(uuid4());trace = Trace(episode, output/'trace.jsonl')
    try:
        lab = InstrumentLab(task)
        lab.router_wait_enabled = router_wait
    except Exception as exc:
        report = {'episode_id': episode, 'status': 'unresolved', 'evidence_ids': [], 'terminal_reason': 'initialization_error'}
        result = {'episode_id': episode, 'task_id': task.task_id, 'backend': roles.backend_id, 'report': report,
                  'score': {'goal_met': False, 'task_success': False, 'protocol_correct': False, 'false_success': False},
                  'exception': {'type': type(exc).__name__, 'message': str(exc)[:1000]}, 'reference_scripts_used': False}
        trace.emit('report', 'initialization_error', **result)
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    source_hashes = {str(p): sha256(p.read_bytes()).hexdigest()
        for p in list(Path('app').glob('*.py')) + [Path(__file__), Path('cases/devices/registry.json')]}
    # Host setup receipts never enter profile projections or role payloads.
    manifest = {'task': task.model_dump(), 'expert': roles.backend_id,
        'router': getattr(router, 'backend_id', 'rules-v1'), 'operation': getattr(operation, 'backend_id', None),
        'reference_scripts_used': False, 'source_hashes': source_hashes,'routing_policy':'argmax-v1',
        'router_direct_wait': router_wait, 'expert_output_tokens': getattr(roles,'max_tokens',None),
        'native_fixture_events': getattr(lab.host, '_workflow_fixture_events', []),
        'scene_callback': 'No callback supplied; native scene prerequisites remain enforced',
        'clock_bridge': {'runtime_class': lab.spec['class'], **lab.lifecycle.manifest()},
        'decision_manifests': {name:backend.manifest() for name,backend in [('router',router),('operation',operation)] if hasattr(backend,'manifest')},
        'limits': limits.model_dump()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True)+'\n')
    (output/'public-contract.json').write_text(json.dumps(lab.description, indent=2, ensure_ascii=True)+'\n')
    trace.emit('native', 'capability_discovery', capabilities=lab.capabilities(), state_ids=lab.fields)
    graph = None
    started = time.monotonic();config={'configurable': {'thread_id': episode}, 'recursion_limit': limits.graph_steps}
    error=None
    try:
        graph = build_graph(lab, trace, limits, roles=roles, decision_backend=router,
                            profile=profile or InstrumentProfile(), operation_backend=operation)
        initial=initial_state(lab,episode,limits)
        initial['deadline']=min(initial['deadline'],episode_deadline)
        state = graph.invoke(initial, config)
        report=state['report']
    except Exception as exc:
        error={'type': type(exc).__name__, 'message': str(exc)[:1000]}
        reason='graph_limit' if isinstance(exc, GraphRecursionError) else 'infrastructure_error'
        try:
            checkpoint = graph.get_state(config).values
        except Exception:
            checkpoint = {}
        report={'episode_id': episode, 'sample_id': lab.sample_id, 'status': 'unresolved', 'evidence_ids': [],
                'terminal_reason': reason, 'action_count': lab.action_count,
                'llm_call_count': checkpoint.get('llm_call_count', 0)}
        trace.emit('report', 'bounded_failure', report=report, error=error,
                   pending_operation=lab.pending_operation)
    try:
        score=lab.score(report)
    except Exception as exc:
        score={'goal_met': False, 'task_success': False, 'protocol_correct': False,
               'false_success': report.get('status')=='supported', 'terminal_reason':'final_read_uncertain',
               'error': type(exc).__name__+': '+str(exc)[:1000],
               'pending_operation': lab.pending_operation, 'lifecycle_resolved': False,
               'native_io_counts': dict(lab.lifecycle.calls)}
    trace.emit('evaluator', 'instrument_score', score=score)
    from app.token_budget import valid_usage
    usage={k:0 for k in ['prompt_tokens','completion_tokens','total_tokens']};complete=True
    for event in trace.events:
        metadata=event.get('llm_metadata') or event.get('backend_metadata') or {}
        if metadata.get('request_sent'):
            if not valid_usage(metadata.get('usage')):
                complete=False
                continue
            for k in usage:usage[k] += metadata['usage'][k]
    result={'episode_id':episode,'task_id':task.task_id,'backend':roles.backend_id,
            'report':report,'score':score,'usage':usage,'usage_complete':complete,
            'elapsed_seconds':time.monotonic()-started,'exception':error,
            'trace_sha256':sha256((output/'trace.jsonl').read_bytes()).hexdigest(),
            'reference_scripts_used':False}
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['deepseek','local'],required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/openrouter-deepseek-v4.1-flash.json'))
    parser.add_argument('--router',choices=['rules','openjev','openrouter-jev'],default='openjev')
    parser.add_argument('--operation',choices=['disabled','openjev','openrouter-jev'],default='openjev')
    parser.add_argument('--max-output-tokens',type=int,default=6144,metavar='256..8192')
    parser.add_argument('--knowledge',choices=['off','on'],default='off',help='Task-conditioned instrument knowledge; never an execution recipe')
    parser.add_argument('--no-router-wait',action='store_true',help='Retain the previous flag-only routing path for comparisons')
    parser.add_argument('--actions',type=int,default=32)
    parser.add_argument('--model-calls',type=int,default=36)
    parser.add_argument('--seconds',type=float,default=360)
    parser.add_argument('--graph-steps',type=int,default=260)
    args=parser.parse_args()
    if not 256 <= args.max_output_tokens <= 8192:
        parser.error('max-output-tokens must be between 256 and 8192')
    task=InstrumentTask.model_validate_json(args.task.read_text())
    from app.expert_config import load_plan, CompatibleExpert
    plan=load_plan(args.config);ledger=None
    plan.reference.max_tokens=args.max_output_tokens
    if args.backend=='deepseek':
        from app.token_budget import BudgetedExpert,TokenLedger
        ledger=TokenLedger();roles=BudgetedExpert(plan.reference,ledger)
    else:
        if plan.reference.provider!='local':parser.error('Local backend requires a configured local expert')
        roles=CompatibleExpert(plan.reference)
    router=operation=None
    if args.router=='openjev':
        from app.local_decision import LocalOpenJevBackend
        router=LocalOpenJevBackend()
    if args.operation=='openjev':
        from app.operation_decision import LocalOperationBackend
        operation=LocalOperationBackend()
    if args.router=='openrouter-jev' or args.operation=='openrouter-jev':
        from app.token_budget import TokenLedger
        from app.openrouter_jev import OpenRouterJevBackend,OpenRouterJevOperationBackend
        ledger=ledger or TokenLedger()
        if args.router=='openrouter-jev':router=OpenRouterJevBackend(ledger)
        if args.operation=='openrouter-jev':operation=OpenRouterJevOperationBackend(ledger)
    if args.router=='openjev' or args.operation=='openjev':
        import httpx
        try:
            httpx.get('http://127.0.0.1:8792/health', timeout=5, trust_env=False).raise_for_status()
        except httpx.HTTPError as exc:
            parser.error('Local OpenJev service is unavailable; no expert requests were sent: '+type(exc).__name__)
    before=ledger.snapshot() if ledger else None
    from app.material_retrieval import KnowledgeInstrumentProfile
    result=run(task,args.output,roles,router=router,operation=operation,router_wait=not args.no_router_wait,
        profile=KnowledgeInstrumentProfile(True) if args.knowledge=='on' else None,
        limits=Limits(lab_actions=args.actions,llm_calls=args.model_calls,episode_seconds=args.seconds,graph_steps=args.graph_steps))
    if ledger:
        result.update(budget_before=before,budget_after=ledger.snapshot())
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'task_id':task.task_id,'report':result['report'],'score':result['score'],'usage':result.get('usage')},indent=2))


if __name__=='__main__':main()
