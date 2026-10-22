from copy import deepcopy
import json

import pytest

from app.models import RouteDecision,OperationDecision
from bench.instrument_eval_langgraph import LangGraphPolicy


class Menus:
    def action_options(self,view,history):
        if not view['selected_instrument']:
            return [{'text':'Select device','action':{'kind':'select_instrument','instrument_id':'device'}}]
        return [{'text':'Set power','capability_id':'power.set'}, {'text':'Finish','action':{'kind':'finish'}}]
    def argument_options(self,view,history,option):
        return [{'action':{'kind':'invoke','instrument_id':'device','capability_id':'power.set','arguments':{'on':on}}} for on in [False,True]]


class Expert:
    backend_id='offline-graph-integration-double';is_llm=False
    def invoke(self,role,prompt,payload,schema):
        if role=='diagnostic':
            view=payload['observations']['measurement']['public_evidence']['environment']
            if not view['selected_instrument']:name,args='benchmark.select_instrument',{'instrument_id':'device'}
            elif not view['observation']['powered']:name,args='device::power.set',{'on':True}
            else:name,args='benchmark.finish',{}
            return {'hypotheses':[],'candidates':[{'candidate_id':'one','intent':{'capability_id':name,'arguments':args,
                'discriminating_observation':'Public native feedback','postcondition':'Goal progress'},'rationale':'Offline handoff check','evidence_ids':[]}]}
        if role=='procedure':
            x=deepcopy(payload['intent']);x.pop('discriminating_observation')
            return dict(x,**{k:payload[k] for k in ['sample_id','expected_revision','request_id']})
        if role=='verification':
            return {'status':'supported','evidence_ids':[payload['observations']['measurement']['id']],'missing_evidence':[]}
        raise AssertionError(role)


class Router:
    is_decision_model=True;backend_id='offline-router'
    def decide(self,summary,allowed):
        choice='verify' if allowed==['verify'] else 'prepare_action' if summary['has_intent'] else 'diagnose'
        return RouteDecision(backend_id=self.backend_id,choice=choice,probabilities={r:float(r==choice) for r in allowed})


class Chooser:
    is_llm=False;backend_id='offline-chooser'
    def choose(self,context,candidates):
        return OperationDecision(backend_id=self.backend_id,choice=candidates[0]['candidate_id'],
            probabilities={c['candidate_id']:float(i==0) for i,c in enumerate(candidates)})


def test_persistent_repository_graph_yields_native_actions_and_consumes_actual_feedback(tmp_path):
    project=lambda v,h:{'environment':deepcopy(v),'executed_history':deepcopy(h)}
    policy=LangGraphPolicy(Menus(),tmp_path,project,Expert(),Router(),Chooser())
    view={'goal':'Power on device','selected_instrument':None,'observation':{},'budget':{'max_decisions':8}}
    first=policy.next_action(view,[])
    assert first=={'kind':'select_instrument','instrument_id':'device'}
    assert policy.graph.get_state(policy.config).next==('observe',)
    assert 'public_evidence' in policy.lab.observe()['measurement']
    view.update(selected_instrument='device',observation={'powered':False})
    history=[{'executed_action':first,'response':{'status':'succeeded'},'corrected':False}]
    second=policy.next_action(view,history)
    assert second['kind']=='invoke' and second['arguments']=={'on':True}
    assert policy.lab.view['observation']['powered'] is False  # No parallel simulator mutated here.
    view['observation']['powered']=True
    history.append({'proposed_action':second,'executed_action':second,'corrected':False,'response':{'status':'succeeded'}})
    assert policy.next_action(view,history)=={'kind':'finish'}
    assert sum(e['component']=='verification' and e['kind']=='role_output' for e in policy.trace.events)==1
    assert sum(e['kind']=='accepted' for e in policy.trace.events)==3
    assert any(e['component']=='benchmark' and e.get('executed_history')==history for e in policy.trace.events)


def test_exact_minimax_endpoint_uses_required_reasoning_and_record_only(tmp_path,monkeypatch):
    from app.expert_config import load_plan
    from app.token_budget import TokenLedger,BudgetedExpert
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-test-placeholder')
    config=load_plan(__import__('pathlib').Path('configs/openrouter-minimax-m3.1-flash-free.json')).reference
    ledger=TokenLedger(tmp_path/'ledger.jsonl',limit=100)
    expert=BudgetedExpert(config,ledger)
    body=expert.prepare_request({'model':config.model,'messages':[{'role':'user','content':'public'}],'max_tokens':8192,'stream':False})
    assert body['reasoning']=={'exclude':True}
    key=ledger.reserve(body,provider='openrouter')
    ledger.settle(key,{'prompt_tokens':1000,'completion_tokens':1000,'total_tokens':2000})
    assert ledger.snapshot()['budget_charged_tokens']==0


def test_jev_accepts_larger_public_context_only_when_explicitly_configured(tmp_path):
    from app.token_budget import TokenLedger
    ledger=TokenLedger(tmp_path/'ledger.jsonl');ledger.enable_openrouter_record_only()
    body={'model':'typesafe/jev-1.14','state':'x'*30000,'questions':{'next':{'type':'choice','criteria':{'A':'a','B':'b'}}}}
    with pytest.raises(ValueError):ledger.reserve_decision(body,provider='openrouter')
    key=ledger.reserve_decision(body,provider='openrouter',context_byte_limit=262144-8192)
    ledger.settle(key,{'prompt_tokens':10000,'completion_tokens':1,'total_tokens':10001})


def test_teacher_corrected_finish_resumes_from_actual_feedback_instead_of_old_report(tmp_path):
    project=lambda v,h:{'environment':deepcopy(v),'executed_history':deepcopy(h)}
    policy=LangGraphPolicy(Menus(),tmp_path,project,Expert(),Router(),Chooser())
    view={'goal':'Power on device','selected_instrument':'device','observation':{'powered':True},'budget':{'max_decisions':8}}
    assert policy.next_action(view,[])=={'kind':'finish'}
    view['observation']['powered']=False
    history=[{'proposed_action':{'kind':'finish'},'executed_action':{'kind':'read'},'corrected':True,
              'response':{'status':'succeeded','powered':False}}]
    assert policy.next_action(view,history)=={'kind':'invoke','instrument_id':'device','capability_id':'power.set','arguments':{'on':True}}


def test_graph_failure_raises_native_protocol_error_and_keeps_failed_call_accounting(tmp_path):
    class ProtocolError(RuntimeError):pass
    class Broken(Expert):
        def invoke(self,*args):raise ValueError('invalid provider record')
    policy=LangGraphPolicy(Menus(),tmp_path,lambda v,h:{'environment':v},Broken(),Router(),Chooser(),error_type=ProtocolError)
    view={'goal':'Power on','selected_instrument':None,'observation':{},'budget':{'max_decisions':8}}
    with pytest.raises(ProtocolError,match='LangGraph stopped'):
        policy.next_action(view,[])
    assert policy.last_action_usage['router_calls']==1
    assert policy.cursor==len(policy.trace.events)


def test_minimax_provider_maximum_replaces_local_8k_cap(tmp_path,monkeypatch):
    from app.expert_config import load_plan
    from app.token_budget import TokenLedger,BudgetedExpert
    from pathlib import Path
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-test-placeholder')
    config=load_plan(Path('configs/openrouter-minimax-m3.1-flash-free.json')).reference
    assert config.max_tokens==524288 and config.timeout_seconds==900
    ledger=TokenLedger(tmp_path/'ledger.jsonl',limit=100)
    expert=BudgetedExpert(config,ledger)
    body=expert.prepare_request({'model':config.model,'messages':[{'role':'user','content':'public'}],
                                'max_tokens':config.max_tokens,'stream':False})
    assert body['max_tokens']==524288
    assert body['reasoning']=={'exclude':True}  # Enabled by provider, no reasoning token cap.
    key=ledger.reserve(body,provider='openrouter')
    ledger.settle(key,{'prompt_tokens':500,'completion_tokens':20000,'total_tokens':20500})
    assert ledger.snapshot()['reported_tokens']==20500
    assert ledger.snapshot()['budget_charged_tokens']==0


def test_deepseek_benchmark_has_reasoning_and_large_record_only_allowance(tmp_path, monkeypatch):
    from pathlib import Path
    from app.expert_config import load_plan
    from app.token_budget import TokenLedger, BudgetedExpert
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    config=load_plan(Path("configs/openrouter-deepseek-v4.1-flash-benchmark.json")).reference
    ledger=TokenLedger(tmp_path / "deepseek.jsonl")
    expert=BudgetedExpert(config, ledger)
    body=expert.prepare_request({"model":config.model,"messages":[{"role":"user","content":"test"}],"max_tokens":config.max_tokens,"stream":False})
    assert body["reasoning"] == {"exclude":True,"effort":"low"}
    assert body["max_tokens"] == 524288
    key=ledger.reserve(body,provider="openrouter")
    ledger.settle(key,{"prompt_tokens":100,"completion_tokens":20000,"total_tokens":20100})
    assert ledger.snapshot()["charged_tokens"] == 20100
