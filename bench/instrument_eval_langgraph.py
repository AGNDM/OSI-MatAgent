"""Run the repository LangGraph as a policy for the released native evaluator."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
from uuid import uuid4

from jsonschema import Draft202012Validator


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


class BenchmarkLab:
    """Public RPC boundary: the benchmark, not a second simulator, executes tools."""
    manages_pending_operations=True
    count_router_calls=True
    router_wait_enabled=False
    routing_policy='argmax-v1'
    pending_operation=None
    termination_reason=None
    sample_id='benchmark-native-session'

    def __init__(self,menus,project):
        self.menus,self.project=menus,project
        self.revision=self.action_count=0
        self.outbound=None;self.finish_requested=False;self.history=[];self.requests={}

    @staticmethod
    def encode(action):
        kind=action['kind']
        if kind=='invoke':return action['instrument_id']+'::'+action['capability_id'],action['arguments']
        if kind=='wait':return 'workflow.await',{'duration_s':action['dt_s']}
        if kind=='read':return 'workflow.readback',{'state_ids':action.get('state_ids')}
        if kind=='select_instrument':return 'benchmark.select_instrument',{'instrument_id':action['instrument_id']}
        return 'benchmark.'+kind,{}

    def refresh(self,view,history):
        self.view=deepcopy(view);self.native_history=deepcopy(history)
        self.public=self.project(view,history)
        self.revision+=1;self.outbound=None;self.finish_requested=False;self.capmap={};self.actionmap={}
        for option in self.menus.action_options(view,history):
            presets=[option] if 'action' in option else self.menus.argument_options(view,history,option)
            for preset in presets:
                action=preset['action'];name,args=self.encode(action)
                self.actionmap[(name,canonical(args))]=deepcopy(action)
                entry=self.capmap.setdefault(name,{'input_schema':{'enum':[]},'source':option['text']})
                if args not in entry['input_schema']['enum']:entry['input_schema']['enum'].append(deepcopy(args))

    def capabilities(self):return deepcopy(self.capmap)
    def target(self):
        return {'objective':self.view['goal'],'scope':'Achieve the public native benchmark goal; only the independent evaluator assigns correctness.'}
    def observe(self):
        obs={'id':'public-'+str(self.revision),'sample_id':self.sample_id,'revision':self.revision,
             'captured_at_step':self.action_count+1,'quality':'valid','public_evidence':deepcopy(self.public)}
        inspection={k:v for k,v in obs.items() if k!='public_evidence'}
        return {'revision':self.revision,'sample_id':self.sample_id,'inspection':inspection,'measurement':obs,
                'last_mutation_step':self.action_count,'pending_operation':None,
                'last_action':deepcopy(self.native_history[-1]) if self.native_history else None}
    def validate(self,proposal,pending_operation=None):
        from app.lab_adapter import GateRejected
        key=proposal.request_id
        if key in self.requests:
            if self.requests[key]!=proposal.model_dump():raise GateRejected('REQUEST_CONFLICT')
            return
        if proposal.sample_id!=self.sample_id:raise GateRejected('WRONG_DEVICE')
        if proposal.expected_revision!=self.revision:raise GateRejected('STALE_PROPOSAL')
        if (proposal.capability_id,canonical(proposal.arguments)) not in self.actionmap:raise GateRejected('UNAVAILABLE_PUBLIC_PRESET')
    def execute(self,proposal):
        self.validate(proposal)
        if proposal.request_id in self.requests:
            raise ValueError('Duplicate execution handoff; native executor must not run twice')
        self.requests[proposal.request_id]=proposal.model_dump();self.action_count+=1
        self.outbound=deepcopy(self.actionmap[(proposal.capability_id,canonical(proposal.arguments))])
        self.history.append(proposal.model_dump())
        return {'step':self.action_count,'proposal':proposal.model_dump(),'result':{
            'submitted_native_action':deepcopy(self.outbound),'execution_status':'awaiting_benchmark_native_executor'},
            'source':'LangGraph-authorized handoff; native result is supplied on next evaluator callback'}
    def route_context(self,state,summary,allowed):
        summary=deepcopy(summary)
        summary['instrument_context']={'objective':self.view['goal'],'public_evidence':self.public,
            'selected_intent':state.get('intent'),'selected_intent_current':bool(state.get('intent')),
            'last_error':state.get('last_error'),'wait_offers':{},'last_verification':state.get('last_verification_feedback')}
        if self.finish_requested and 'verify' in allowed:return summary,['verify']
        if not self.view.get('selected_instrument'):allowed=[r for r in allowed if r!='verify']
        return summary,allowed
    def operation_context(self,context,candidates):
        context=deepcopy(context);context.pop('capabilities',None)
        return context
    def verification_signature(self,state):
        return sha256(canonical({'goal':self.view['goal'],'machine':self.view.get('selected_instrument'),
            'observation':self.view.get('observation'),'last_native_response':self.native_history[-1].get('response') if self.native_history else None}).encode()).hexdigest()


class BenchmarkProfile:
    prompts={
      'diagnostic':'Use the supplied public benchmark goal, current-instrument manual, complete native feedback and history to plan one native action. All public commands and their finite shared argument presets are supplied. Do not invent parameters, setup fixtures or private faults. A command receipt is not timed completion. Public evidence already serves as inspection. Propose genuine alternatives briefly; do not rank them. Use benchmark.finish only when evidence establishes the goal, and benchmark.escalate for an evidenced irreparable fault. Cite ONLY observations.measurement.id in supporting_ids/evidence_ids. Embedded history response_ IDs are not observation IDs for the output schema. Leave material_evidence_ids empty unless citing the retrieved benchmark-public-protocol document. Never execute code.',
      'procedure':'Copy the chosen intent to one ActionProposal, copying sample_id, revision, request_id, capability and arguments exactly. Do not change the selected native action.',
      'verification':'Assess the public benchmark goal using actual executed history and native readings, timestamps, errors and manuals. Report supported only if public evidence establishes the stated goal; a command being accepted alone is not completion. Otherwise report contradicted or unresolved with the actual missing evidence. Cite ONLY observations.measurement.id, not response_ references embedded in public history. Never use hidden grading criteria.',
      'deliberation':'Choose a supplied specialist route only.'}
    candidate_prompt=prompts['diagnostic']+' Return hypotheses and 1-4 distinct schema-valid candidates; keep prose brief.'
    def __init__(self,lab):self.lab=lab
    @staticmethod
    def projection(role,state,request_id):
        from app.roles import projection
        payload=projection(role,state,request_id)
        if role in {'diagnostic','verification'}:
            payload['citation_contract']={'observation_ids':[state['observations']['measurement']['id']],
                'retrieved_document_ids':['benchmark-public-protocol'],
                'rule':'Do not put observation IDs in material_evidence_ids; embedded response_ history references are not output observation IDs.'}
        return payload
    @staticmethod
    def retrieve(role):return [{'id':'benchmark-public-protocol','text':'Use only public evidence and native feedback. Teacher corrections are assistance, not your own successful decisions.'}]
    def validate_role_contract(self,role,record,payload):
        if role=='diagnostic' and record.intent:
            cap=payload['capabilities'].get(record.intent.capability_id)
            if cap is None:raise ValueError('Use an offered public capability')
            error=next(Draft202012Validator(cap['input_schema']).iter_errors(record.intent.arguments),None)
            if error:raise ValueError('Arguments must match an offered shared preset: '+error.message)
        elif role=='procedure':
            from app.graph import validate_role_contract
            validate_role_contract(role,record,payload)
        elif role=='verification':
            current=payload['observations']['measurement']['id']
            if current not in record.evidence_ids:raise ValueError('Cite current public context packet')
            if record.status!='supported':self.lab.finish_requested=False


class LangGraphPolicy:
    privileged=False
    def __init__(self,menus,output,project,expert,router,operation,error_type=RuntimeError):
        from app.graph import build_graph
        from app.models import Limits
        from app.trace import Trace
        self.lab=BenchmarkLab(menus,project);self.roles=expert;self.error_type=error_type
        self.trace=Trace(str(uuid4()),output/'langgraph.jsonl');self.limits=Limits(lab_actions=64,llm_calls=384,episode_seconds=7200,graph_steps=3000)
        self.builder=build_graph;self.router=router;self.operation=operation;self.graph=None
        self.config={'configurable':{'thread_id':self.trace.episode_id},'recursion_limit':3000}
        self.started=False;self.last_action_usage={};self.cursor=0
        self.identity={'method':'repository-langgraph-native-policy','privileged':False,'graph_source':'app/graph.py',
            'expert':expert.backend_id,'router':router.backend_id,'operation':operation.backend_id,
            'execution_boundary':'yield after gated tool handoff; benchmark native executor owns simulation',
            'context':'Provided public projection, complete executed history; no case specifications or answer keys'}
    def _next_action(self,view,history):
        from app.graph import initial_state
        self.lab.refresh(view,history)
        self.trace.emit('benchmark','native_feedback',public_view=view,executed_history=history)
        self.lab.deadline=time.monotonic()+7200
        if not self.started:
            self.limits=self.limits.model_copy(update={'lab_actions':view['budget']['max_decisions']})
            self.graph=self.builder(self.lab,self.trace,self.limits,roles=self.roles,decision_backend=self.router,
                operation_backend=self.operation,profile=BenchmarkProfile(self.lab),yield_after_execute=True)
            state=initial_state(self.lab,self.trace.episode_id,self.limits);self.started=True
        else:
            self.graph.update_state(self.config,{'capabilities':self.lab.capabilities(),'target':self.lab.target(),
                'observation_revision':self.lab.revision,'report':None,'terminal_reason':None,'verification':None,'last_error':(history[-1].get('response') or {}).get('error') if history else None},as_node='execute')
            state=None
        result=self.graph.invoke(state,self.config)
        # Explicit finish proposals run through this same graph's verification
        # before being sent to the native evaluator as a completion claim.
        while self.lab.outbound and self.lab.outbound['kind']=='finish' and not result.get('report'):
            self.lab.outbound=None;self.lab.finish_requested=True
            result=self.graph.invoke(None,self.config)
        action=deepcopy(self.lab.outbound)
        if action is None:
            report=result.get('report') or {}
            if report.get('status')=='supported':action={'kind':'finish'}
            else:raise self.error_type('LangGraph stopped without an executable action: '+str(report))
        return action

    def next_action(self,view,history):
        self.last_action_usage={}
        try:
            return self._next_action(view,history)
        finally:
            self.collect_usage()

    def collect_usage(self):
        events=self.trace.events[self.cursor:];self.cursor=len(self.trace.events)
        unique={}
        for event in events:
            meta=event.get('llm_metadata') or event.get('backend_metadata') or {}
            if meta.get('budget_reservation_id'):unique[meta['budget_reservation_id']]=meta
        self.last_action_usage={'input_tokens':sum((m.get('usage') or {}).get('prompt_tokens',0) for m in unique.values()),
            'output_tokens':sum((m.get('usage') or {}).get('completion_tokens',0) for m in unique.values()),
            'input_tokens_known':all(m.get('usage') for m in unique.values()),'output_tokens_known':all(m.get('usage') for m in unique.values()),
            'api_cost_usd':None,'gpu_seconds':0,'selection_events':[{'kind':e['kind']} for e in events if e['kind']=='operation_selected'],
            'router_calls':sum(e['kind']=='routing' for e in events),'model_requests':len(unique)}



class SharedRequestPacer:
    """Process-safe request spacing for a provider's published free-model RPM."""
    def __init__(self,path,interval):
        self.path=Path(path);self.interval=interval
        self.path.parent.mkdir(parents=True,exist_ok=True)

    async def wait(self):
        import asyncio
        import fcntl
        with self.path.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            stream.seek(0);raw=stream.read().strip();now=time.time()
            slot=max(now,float(raw) if raw else now)
            stream.seek(0);stream.truncate();stream.write(str(slot+self.interval));stream.flush()
            fcntl.flock(stream,fcntl.LOCK_UN)
        delay=max(0.,slot-time.time())
        await asyncio.sleep(delay)
        return delay


def main():
    import argparse
    import os
    import shutil
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--case-id',action='append')
    selection.add_argument('--all',action='store_true')
    parser.add_argument('--continue-on-error',action='store_true')
    parser.add_argument('--expert-config',type=Path,default=Path('configs/openrouter-minimax-m3.1-flash-low.json'))
    parser.add_argument('--decision-model',choices=['typesafe/jev-1.14','openai/gpt-6-luna-decisions'],default='typesafe/jev-1.14')
    parser.add_argument('--expert-timeout',type=float,default=120)
    parser.add_argument('--rate-state',type=Path)
    parser.add_argument('--rate-interval',type=float,default=0)
    parser.add_argument('--ledger',type=Path)
    parser.add_argument('--provider-stop',type=Path)
    parser.add_argument('--mode',choices=['free_run','teacher_forced','both'],default='free_run')
    parser.add_argument('--execution',choices=['local-cpu'],required=True)
    args=parser.parse_args();source=args.bundle.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Use a fresh output directory')
    if not 0<args.expert_timeout<=1800:raise ValueError('Expert timeout must be in (0,1800]')
    if args.rate_interval<0 or bool(args.rate_state)!=(args.rate_interval>0):
        raise ValueError('Positive rate interval and shared state path must be supplied together')
    manifest=json.loads((source/'manifest.json').read_text())
    assert all(sha256((source/name).read_bytes()).hexdigest()==m['sha256'] for name,m in manifest['files'].items())
    output.mkdir(parents=True);release=output/'release';shutil.copytree(source,release)
    import ast
    patches=[]
    for module,function in [('policies.py','require_allocation'),('engine.py','require_compute_node')]:
        guard=release/'sequential_120/runtime/vendor/stepwise_bench'/module
        before=guard.read_bytes();needle='def '+function+'():\n'
        assert guard.read_text().count(needle)==1
        guard.write_text(guard.read_text().replace(needle,needle+'    if os.environ.get("INSTRUMENT_EVAL_EXECUTION") == "local-cpu":\n        return  # Explicit user-authorized local CPU deployment; no fake Slurm identity.\n',1))
        def remainder(code):
            tree=ast.parse(code)
            tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name!=function]
            return ast.dump(tree,include_attributes=False)
        assert remainder(before)==remainder(guard.read_bytes())
        patches.append({'path':str(guard.relative_to(release)),'function':function,
            'original_sha256':sha256(before).hexdigest(),'local_sha256':sha256(guard.read_bytes()).hexdigest(),
            'all_non_scheduler_code_ast_identical':True})
    changed=[name for name,m in manifest['files'].items() if sha256((release/name).read_bytes()).hexdigest()!=m['sha256']]
    assert set(changed)=={row['path'] for row in patches}
    os.environ['INSTRUMENT_EVAL_EXECUTION']='local-cpu'
    sys.path.insert(0,str(release))
    from sequential_120.runtime.bootstrap import activate
    activate()
    from stepwise_choice.engine import run_case,MenuProtocolError
    from stepwise_choice.menus import MenuBuilder
    from stepwise_choice.policies import project_public_input
    from eval.common import load_jsonl
    from eval.sequential import score_episodes,_validate_episode,OfferedChoicePolicy
    # Load benchmark runtimes before app.native, preventing mixed native imports.
    from app.expert_config import load_plan
    from app.token_budget import TokenLedger,BudgetedExpert
    from app.openrouter_jev import OpenRouterJevBackend,OpenRouterJevOperationBackend
    tasks=load_jsonl(release/'sequential_120/cases.jsonl')
    specs=load_jsonl(release/'sequential_120/evaluator/case-specs.jsonl')
    taskmap={q['task_id']:q for q in tasks};specmap={q['task_id']:q for q in specs}
    if args.all:args.case_id=sorted(taskmap)
    if len(args.case_id)!=len(set(args.case_id)):raise ValueError('Duplicate case ID')
    if not set(args.case_id)<=set(taskmap):raise ValueError('Unknown public case ID')
    selected=[specmap[k] for k in args.case_id]
    bank=json.loads((release/'sequential_120/action-presets.json').read_text())
    modes=['teacher_forced','free_run'] if args.mode=='both' else [args.mode]
    config={'dataset_commit':'22a8192e4e40f3a7d37e1164a8755b006f932d60','public_selected_ids':args.case_id,'modes':modes,
        'execution':'explicit local CPU; no Slurm impersonation','scheduler_patches':patches,
        'release_files_verified':len(manifest['files']),'grading_data_runtime_unchanged_except_scheduler_guard':True,
        'source_hashes':{str(p):sha256(p.read_bytes()).hexdigest() for p in list(Path('app').glob('*.py'))+[Path(__file__)]},
        'expert':load_plan(args.expert_config).reference.model,'router_and_operation':args.decision_model,
        'graph':'app.graph.build_graph with persistent checkpoint and yield_after_execute',
        'oracle_boundary':'Only native evaluator sees case specs/checkpoints; model policy receives public view/history/menu',
        'filter':'No irrelevant questions removed; full cohort' if len(selected)==len(tasks) else 'Explicit subset/shard; parent cohort recorded separately',
        'expert_timeout_seconds':args.expert_timeout,'continue_on_error':args.continue_on_error,
        'request_pacing_seconds':args.rate_interval,'complete_selected_dataset':len(selected)==len(tasks),
        'expert_configuration':load_plan(args.expert_config).reference.model_copy(update={'timeout_seconds':args.expert_timeout}).model_dump()}
    (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    for raw,digest in config['source_hashes'].items():
        p=Path(raw);assert sha256(p.read_bytes()).hexdigest()==digest
        rel=p.relative_to(Path.cwd()) if p.is_absolute() else p
        target=output/'frozen-source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    ledger=TokenLedger(args.ledger) if args.ledger else TokenLedger()
    ledger.enable_openrouter_record_only();before_budget=ledger.snapshot()
    pacer=SharedRequestPacer(args.rate_state,args.rate_interval) if args.rate_state else None
    class PacedExpert(BudgetedExpert):
        async def _request(self,body,timeout):
            if pacer:self.last_call['rate_wait_seconds']=await pacer.wait()
            return await super()._request(body,timeout)
    failures=0
    episodes=[]
    for key in args.case_id:
        for mode in modes:
            if args.provider_stop and args.provider_stop.exists():return 2
            episode_dir=output/key/mode;episode_dir.mkdir(parents=True)
            menus=MenuBuilder(deepcopy(bank))
            expert=PacedExpert(load_plan(args.expert_config).reference.model_copy(update={'timeout_seconds':args.expert_timeout}),ledger)
            router=OpenRouterJevBackend(ledger,context_byte_limit=262144-8192,model=args.decision_model,force_singleton=True)
            operation=OpenRouterJevOperationBackend(ledger,context_byte_limit=262144-8192,model=args.decision_model,force_singleton=True)
            policy=LangGraphPolicy(menus,episode_dir,project_public_input,expert,router,operation,error_type=MenuProtocolError)
            try:
                row=run_case(specmap[key],OfferedChoicePolicy(policy,menus,MenuProtocolError),mode=mode,menu_builder=menus,
                    on_step=lambda step:(episode_dir/f"step-{step['decision_index']:03d}.json").write_text(json.dumps(step,indent=2,ensure_ascii=False)+'\n'))
                row.update(status='ok',policy_identity=policy.identity)
                (episode_dir/'native-episode.json').write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n')
                _validate_episode(row)
            except Exception as exc:
                errors=[e for e in policy.trace.events if e['kind'] in {'role_error','routing_error','operation_error'}]
                row={'task_id':key,'mode':mode,'status':'infrastructure_error','error':type(exc).__name__+': '+str(exc)[:2000],
                     'failure_details':[{k:e.get(k) for k in ['component','kind','error','repair_attempt','will_repair']} for e in errors],
                     'native_actions_before_abort':policy.lab.action_count}
                if args.provider_stop:
                    for event in policy.trace.events:
                        meta=event.get('llm_metadata') or event.get('backend_metadata') or {}
                        status=meta.get('http_status');text=str(meta.get('response','')).lower()
                        if status in {402,404} or (status==429 and any(word in text for word in ['per-day','per day','daily'])):
                            flag={'http_status':status,'task_id':key,'reason':'Provider availability/daily quota requires intervention; remaining cases are unattempted.'}
                            temp=args.provider_stop.with_suffix('.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(flag));os.replace(temp,args.provider_stop)
                            break
            episodes.append(row)
            with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            summary=score_episodes(selected,episodes,modes)
            summary.update(diagnostic_subset=len(selected)!=len(tasks),complete_full_dataset=len(selected)==len(tasks) and len(episodes)==len(selected)*len(modes),full_dataset_size=len(tasks),budget_before=before_budget,budget_after=ledger.snapshot())
            (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'case':key,'mode':mode,'status':row['status'],'goal':row.get('trajectory_outcome_met'),'error':row.get('error')}),flush=True)
            if row['status']=='infrastructure_error':
                failures+=1
                if not args.continue_on_error:return 1
    return int(failures>0)


if __name__=='__main__':raise SystemExit(main())
