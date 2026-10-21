import json
from copy import deepcopy
import pytest

from app.device_catalog import load_catalog,index,search,describe
from app.device_selection import select_device
from app.single_instrument import InstrumentTask
from app.models import Limits,RouteDecision,OperationDecision
from app.decision_backend import select_route
from app.token_budget import TokenLedger,TokenBudgetExceeded
from app.trace import Trace


def test_catalog_pages_cover_all_registered_devices_without_reference_fields():
    catalog=load_catalog();rows=[];offset=0
    while offset is not None:
        page=search(catalog,offset=offset,limit=37);rows+=page['items'];offset=page['next_offset']
    assert len(rows)==194 and len({r['id'] for r in rows})==194
    assert rows==index(catalog)
    for d in catalog['devices']:
        assert not {'module','class','task_file','reference_probe','steps','tool_menu'} & set(d)
        assert all(v.get('visibility','agent')=='agent' for v in d['state_slots'].values())
        assert not any(k.startswith(('truth.','debug.','host.','plant.')) for k in d['state_slots'])
        assert not any(k.startswith(('sim.','host.','plant.')) for k in d['capabilities'])


def test_unknown_device_paths_are_not_resolved():
    c=load_catalog()
    for id in ['../../etc/passwd','asset_runtime.custom','unknown']:
        with pytest.raises(ValueError):describe(c,id)


def test_expert_directory_candidates_feed_jev_without_changing_oracle():
    class Expert:
        is_llm=True;backend_id='offline-expert';last_call=None;count=0
        def invoke(self,role,prompt,payload,schema):
            assert len(payload['device_directory'])==194
            assert 'reference_probe' not in json.dumps(payload)
            self.count+=1
            if self.count==1:return {'action':'describe','device_ids':['mt_xpr204e','binder_fd056']}
            return {'action':'propose','candidates':[
                {'candidate_id':'scale','device_id':'mt_xpr204e','rationale':'Public filter control'},
                {'candidate_id':'oven','device_id':'binder_fd056','rationale':'Compare public contract'}]}
    class Chooser:
        is_llm=True;backend_id='offline-jev';last_call=None
        def choose(self,context,candidates):
            assert len(candidates)==2 and 'device_directory' not in context
            return OperationDecision(backend_id=self.backend_id,choice='scale',probabilities={'scale':.51,'oven':.49})
    task=InstrumentTask(task_id='filter',device='auto',objective='Configure balance filter',
        predicates=[{'state_id':'balance.filter_level','unit':'1','value':'High'}])
    before=task.model_dump();trace=Trace('discovery')
    selected,info=select_device(task,Expert(),Chooser(),trace,Limits(llm_calls=8))
    assert task.model_dump()==before and selected.device=='mt_xpr204e'
    assert selected.predicates==task.predicates and info['model_calls']==3
    assert not any(e['kind']=='tool_result' for e in trace.events)


@pytest.mark.parametrize('probabilities',[{'diagnose':.51,'verify':.49},{'diagnose':.5,'verify':.5}])
def test_argmax_has_no_low_confidence_review(probabilities):
    class Router:
        backend_id='offline-jev'
        def decide(self,*args):return RouteDecision(backend_id=self.backend_id,choice='diagnose',probabilities=probabilities)
    decision,error=select_route(Router(),{},list(probabilities))
    assert error is None and decision.choice=='diagnose' and decision.probabilities==probabilities


def test_argmax_failure_stops_without_rules_fallback():
    class Router:
        backend_id='offline-jev'
        def decide(self,*args):raise ValueError('bad distribution')
    decision,error=select_route(Router(),{},['diagnose','verify'])
    assert error and decision.choice=='stop' and not decision.fallback_used


def test_budget_window_survives_restart_and_directory_changes(tmp_path):
    path=tmp_path/'ledger.jsonl';ledger=TokenLedger(path,window_id='accepted-work',window_limit=1000)
    request=ledger._reserve_bound('test',600);ledger.settle(request,{'prompt_tokens':100,'completion_tokens':100,'total_tokens':200})
    restarted=TokenLedger(path,window_id='accepted-work',window_limit=1000)
    assert restarted.snapshot()['window']['charged_tokens']==200
    with pytest.raises(TokenBudgetExceeded):restarted._reserve_bound('test',801)
    r=restarted._reserve_bound('test',800);restarted.settle(r,None)
    assert restarted.snapshot()['window']['remaining_tokens']==0
    with pytest.raises(TokenBudgetExceeded):ledger._reserve_bound('test',1)
    with pytest.raises(ValueError):TokenLedger(path,window_id='accepted-work',window_limit=2000)


def test_pending_window_reservation_is_conservative(tmp_path):
    ledger=TokenLedger(tmp_path/'ledger.jsonl',window_id='work',window_limit=1000)
    ledger._reserve_bound('test',700)
    other=TokenLedger(ledger.path,window_id='work',window_limit=1000)
    with pytest.raises(TokenBudgetExceeded):other._reserve_bound('test',301)
    assert other.snapshot()['inflight_or_crashed']==1


def test_any_tied_maximum_is_accepted():
    class Router:
        backend_id='offline-jev'
        def decide(self,*args):return RouteDecision(backend_id=self.backend_id,choice='verify',probabilities={'diagnose':.5,'verify':.5})
    decision,error=select_route(Router(),{},['diagnose','verify'])
    assert error is None and decision.choice=='verify'


def test_new_unknown_usage_blocks_the_next_request_but_history_does_not(tmp_path):
    path=tmp_path/'ledger.jsonl';old=TokenLedger(path)
    key=old._reserve_bound('old',100);old.settle(key,None)
    ledger=TokenLedger(path,window_id='new-work',window_limit=10000)
    key=ledger._reserve_bound('new',100);ledger.settle(key,None)
    assert ledger.snapshot()['window']['unknown_usage_detected']
    with pytest.raises(TokenBudgetExceeded):ledger._reserve_bound('new',1)


def test_operation_budget_refusal_is_not_mislabeled_as_model_error():
    from app.graph import build_graph,initial_state
    from app.single_instrument import InstrumentProfile
    from tests.test_single_instrument import fake_lab
    from tests.test_router_wait import TestRoles,TestChooser
    class Denied(TestChooser):
        def choose(self,*args):raise TokenBudgetExceeded('No request sent')
    lab,_=fake_lab();trace=Trace('budget-refusal');limits=Limits()
    final=build_graph(lab,trace,limits,roles=TestRoles(),profile=InstrumentProfile(),operation_backend=Denied()).invoke(
        initial_state(lab,'budget-refusal',limits),{'configurable':{'thread_id':'budget-refusal'}})
    assert final['report']['terminal_reason']=='token_budget_exceeded'
    assert not any(e['kind']=='tool_result' for e in trace.events)


def test_openrouter_record_only_is_provider_scoped_and_keeps_other_budget(tmp_path):
    ledger=TokenLedger(tmp_path/'ledger.jsonl',limit=100,window_id='work',window_limit=100)
    old=ledger._reserve_bound('old',50);ledger.settle(old,{'prompt_tokens':25,'completion_tokens':25,'total_tokens':50})
    ledger.enable_openrouter_record_only()
    key=ledger._reserve_bound('typesafe/jev-1.14',300000,provider='openrouter')
    ledger.settle(key,{'prompt_tokens':200,'completion_tokens':100,'total_tokens':300})
    state=ledger.snapshot()
    assert state['charged_tokens']==350 and state['budget_charged_tokens']==50 and state['remaining_tokens']==50
    assert state['window']['remaining_tokens']==50
    assert state['requests_reserved']==state['requests_settled']==2
    for provider,model in [('siliconflow','typesafe/jev-1.14'),(None,'typesafe/jev-1.14'),('openrouter','other:free')]:
        with pytest.raises(TokenBudgetExceeded):ledger._reserve_bound(model,51,provider=provider)


def test_record_only_new_phase_remains_finite_and_does_not_erase_history(tmp_path):
    ledger=TokenLedger(tmp_path/'ledger.jsonl',window_id='work',window_limit=1000)
    for _ in range(4):ledger.claim_episode('acceptance')
    for _ in range(2):ledger.claim_episode('retrieval')
    ledger.enable_openrouter_record_only()
    for _ in range(4):ledger.claim_episode('knowledge-record-only-v4')
    with pytest.raises(TokenBudgetExceeded):ledger.claim_episode('knowledge-record-only-v4')
    entries=[json.loads(x) for x in ledger.path.read_text().splitlines()]
    assert sum(e['kind']=='episode_start' for e in entries)==10
    assert sum(e['kind']=='policy' for e in entries)==1
    assert entries[0]=={'kind':'limit','limit':5000000}
