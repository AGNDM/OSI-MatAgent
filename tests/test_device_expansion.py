from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.device_lab import DeviceLab
from app.device_registry import load_task
from app.device_roles import DeviceProfile
from app.lab_adapter import GateRejected
from app.models import ActionProposal, HypothesisSet
from app.token_budget import TokenBudgetExceeded, TokenLedger
from bench.device_tasks import run_task


def lab_fixture():
    spec,task=load_task('binder_fd056')
    return DeviceLab(spec,task),spec,task


def proposal(lab,cap,args=None,**changes):
    values=dict(sample_id=lab.sample_id,expected_revision=lab.revision,capability_id=cap,
        arguments=args or {},request_id=str(uuid4()),postcondition='test')
    values.update(changes)
    return ActionProposal(**values)


def test_native_task_runs_through_existing_graph_and_requires_evidence(tmp_path):
    lab,spec,task=lab_fixture()
    row=run_task(spec,task,tmp_path)
    assert row['score']['task_success']
    assert row['report']['llm_call_count']==0
    assert row['score']['accepted_native_action']
    assert not lab.score({'status':'supported','evidence_ids':[]})['task_success']
    assert lab.score({'status':'supported','evidence_ids':[]})['false_success']
    trace=[json.loads(x) for x in Path(row['trace_path']).read_text().splitlines()]
    assert any(e['kind']=='tool_result' and 'native_response' in e['result'] for e in trace)


def test_device_gate_identity_revision_schema_and_hidden_controls():
    lab,_,task=lab_fixture()
    native=lab.spec['slug']+'::'+task['reference_probe']['steps'][-1]['capability_id']
    args=task['reference_probe']['steps'][-1]['arguments']
    before=deepcopy(lab._read_native())
    for p,code in [(proposal(lab,native,args,sample_id='device:other'),'WRONG_DEVICE'),
        (proposal(lab,native,args,expected_revision=99),'STALE_PROPOSAL'),
        (proposal(lab,'binder_fd056::plant.configure',{}),'UNSUPPORTED_CAPABILITY'),
        (proposal(lab,'binder_fd056::sim.step',{'dt_s':3600}),'UNSUPPORTED_CAPABILITY'),
        (proposal(lab,native,dict(args,unexpected=True)),'INVALID_ARGUMENT'),
        (proposal(lab,native,args),'INSPECTION_REQUIRED')]:
        with pytest.raises(GateRejected,match=code):lab.execute(p)
    assert lab._read_native()==before
    assert lab.action_count==0


def test_device_requests_are_idempotent_and_conflicts_fail():
    lab,_,_=lab_fixture()
    p=proposal(lab,'workflow.inspect')
    first=lab.execute(p)
    assert lab.execute(p)==first and lab.action_count==1
    with pytest.raises(GateRejected,match='REQUEST_CONFLICT'):
        lab.execute(p.model_copy(update={'postcondition':'changed'}))
    projected=lab.target()|lab.observe()
    text=json.dumps(projected)
    assert 'reference_probe' not in text and 'candidate_rejections' not in text


def test_device_profile_rejects_invalid_intent_before_procedure():
    lab,_,_=lab_fixture()
    record=HypothesisSet(hypotheses=[],intent={'capability_id':'workflow.inspect','arguments':{'extra':1},
        'postcondition':'x','discriminating_observation':'x'})
    with pytest.raises(ValueError):DeviceProfile.validate_role_contract('diagnostic',record,{'capabilities':lab.capabilities()})


def body(text='x'):
    return {'model':'debug','messages':[{'role':'user','content':text}],
        'max_tokens':256,'enable_thinking':False,'stream':False}


def test_budget_concurrent_reservations_cannot_overspend(tmp_path):
    ledger=TokenLedger(tmp_path/'ledger.jsonl',limit=20_000)
    def reserve(_):
        try:return ledger.reserve(body())
        except TokenBudgetExceeded:return None
    with ThreadPoolExecutor(max_workers=8) as pool:ids=list(pool.map(reserve,range(8)))
    assert len([x for x in ids if x])==2
    assert ledger.snapshot()['charged_tokens']<=20_000
    for key in filter(None,ids):ledger.settle(key,{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150})
    assert ledger.snapshot()['reported_tokens']==300
    assert TokenLedger(ledger.path,20_000).snapshot()['charged_tokens']==300
    with pytest.raises(ValueError):TokenLedger(ledger.path,30_000)


def test_budget_unknown_usage_and_crashes_retain_full_reservation(tmp_path):
    ledger=TokenLedger(tmp_path/'ledger.jsonl',20_000)
    key=ledger.reserve(body())
    reserved=ledger.snapshot()['charged_tokens']
    ledger.settle(key,None)
    assert ledger.snapshot()['charged_tokens']==reserved
    assert ledger.snapshot()['unmetered_requests']==1
    ledger.reserve(body())  # simulate process death before settle
    assert TokenLedger(ledger.path,20_000).snapshot()['inflight_or_crashed']==1
    with pytest.raises(TokenBudgetExceeded):ledger.reserve(body())


def test_budget_rejects_unbounded_requests_and_usage_anomalies(tmp_path):
    with pytest.raises(ValueError):TokenLedger(tmp_path/'too-large',5_000_001)
    ledger=TokenLedger(tmp_path/'ledger.jsonl',20_000)
    with pytest.raises(ValueError):ledger.reserve(dict(body(),enable_thinking=True))
    key=ledger.reserve(body())
    with pytest.raises(TokenBudgetExceeded):ledger.settle(key,{'prompt_tokens':19000,'completion_tokens':1,'total_tokens':19001})
    with pytest.raises(TokenBudgetExceeded):ledger.reserve(body())


def test_successful_native_write_returns_fresh_evidence_and_replays_once():
    lab,_,task=lab_fixture()
    lab.execute(proposal(lab,'workflow.inspect'))
    for step in task['reference_probe']['steps']:
        p=proposal(lab,lab.spec['slug']+'::'+step['capability_id'],step['arguments'])
        before=lab.action_count
        event=lab.execute(p)
        assert lab.action_count==before+2
        m=lab.observe()['measurement']
        assert m['captured_at_step']>lab.last_mutation_step
        assert m['revision']==lab.revision
        assert event['result']['post_action_readback']==m
        assert lab.execute(p)==event
        assert lab.action_count==before+2
    m=lab.observe()['measurement']
    assert lab.score({'status':'supported','evidence_ids':[m['id']]})['task_success']


def test_gate_reserves_room_for_post_command_readback(tmp_path):
    from app.graph import build_graph,initial_state
    from app.device_roles import ScriptedDeviceRoles
    from app.models import Limits
    from app.trace import Trace
    lab,spec,task=lab_fixture()
    limits=Limits(lab_actions=2)
    trace=Trace('budget-test')
    graph=build_graph(lab,trace,limits,roles=ScriptedDeviceRoles(task,spec['slug']),profile=DeviceProfile())
    result=graph.invoke(initial_state(lab,'budget-test',limits),{'configurable':{'thread_id':'budget-test'}})
    assert result['report']['terminal_reason']=='action_budget'
    assert lab.action_count==1
    assert not any('native_response' in e['result'] for e in lab.history)


def test_budget_rejects_request_before_transport(monkeypatch,tmp_path):
    import asyncio
    from app.expert_config import CompatibleExpert,ExpertEndpoint
    from app.token_budget import BudgetedExpert
    monkeypatch.setenv('DEVICE_TEST_KEY','fake-test-key')
    config=ExpertEndpoint(name='debug',provider='siliconflow',base_url='https://api.siliconflow.cn/v1',
        model='deepseek-ai/DeepSeek-V4.1-Flash',api_key_env='DEVICE_TEST_KEY')
    called=[]
    async def transport(self,body,timeout):
        called.append(True)
        return {'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
    monkeypatch.setattr(CompatibleExpert,'_request',transport)
    roles=BudgetedExpert(config,TokenLedger(tmp_path/'ledger',100))
    roles.last_call={}
    with pytest.raises(TokenBudgetExceeded):asyncio.run(roles._request(body(),10))
    assert called==[]
    assert roles.last_call['request_sent'] is False
