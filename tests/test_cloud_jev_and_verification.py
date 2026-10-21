import asyncio
from copy import deepcopy

import httpx
import pytest

from app.openrouter_jev import OpenRouterJevBackend,OpenRouterJevOperationBackend,MODEL,ENDPOINT
from app.token_budget import TokenLedger
from app.local_roles import LocalLlamaRoles,ExpertResponseError
from app.models import VerificationReport,Limits
from app.single_instrument import InstrumentProfile,verification_status
from app.graph import initial_state,build_graph
from app.trace import Trace
from tests.test_single_instrument import fake_lab,execute,LifecycleTestRoles


def cloud(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','test-key')
    return OpenRouterJevBackend(TokenLedger(tmp_path/'ledger.jsonl'))


def test_typed_cloud_probabilities_and_budget(tmp_path,monkeypatch):
    backend=cloud(tmp_path,monkeypatch)
    async def post(self,url,**kw):
        assert url==ENDPOINT and kw['json']['model']==MODEL
        assert 'max_tokens' not in kw['json']
        return httpx.Response(200,json={'model':MODEL+'-20261014','answers':{'next_action':{
            'type':'choice','choice':'A','probabilities':{'A':.8,'B':.2},'confidence':.6}},
            'usage':{'input_tokens':10,'output_tokens':12}})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    decision=backend.decide({'has_intent':False},['diagnose','verify'])
    assert decision.choice=='diagnose' and decision.probabilities=={'diagnose':.8,'verify':.2}
    assert backend.ledger.snapshot()['charged_tokens']==22
    assert backend.last_call['response']['model']==MODEL+'-20261014'
    # An invalid response is still charged before structural validation fails.
    backend.last_call={}
    with pytest.raises(ValueError):asyncio.run(backend._infer_options('choose',{'A':'only'},1))
    assert backend.ledger.snapshot()['charged_tokens']==44


def test_cloud_timeout_and_oversize_are_conservative(tmp_path,monkeypatch):
    backend=cloud(tmp_path,monkeypatch)
    async def post(*args,**kwargs):raise httpx.ReadTimeout('test timeout')
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    backend.last_call={}
    with pytest.raises(httpx.ReadTimeout):asyncio.run(backend._infer_options('choose',{'A':'only'},1))
    snap=backend.ledger.snapshot()
    assert snap['unmetered_requests']==1 and snap['charged_tokens']>262144 and snap['inflight_or_crashed']==0
    backend.last_call={}
    with pytest.raises(ValueError):asyncio.run(backend._infer_options('x'*24001,{'A':'only'},1))
    assert backend.ledger.snapshot()['requests_reserved']==1


def test_cloud_operation_reuses_validated_candidate_selection(tmp_path,monkeypatch):
    base=cloud(tmp_path,monkeypatch)
    backend=OpenRouterJevOperationBackend(base.ledger)
    async def post(*args,**kwargs):
        return httpx.Response(200,json={'model':MODEL,'answers':{'next_action':{
            'type':'choice','choice':'B','probabilities':{'A':.1,'B':.9}}},
            'usage':{'input_tokens':100,'output_tokens':10}})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    candidates=[{'candidate_id':i,'intent':{'capability_id':'test::act','arguments':{'value':n},
        'discriminating_observation':'test','postcondition':'test'},'rationale':'test','evidence_ids':[]}
        for n,i in enumerate(['first','second'])]
    decision=backend.choose({},candidates)
    assert decision.choice=='second' and decision.probabilities=={'first':.1,'second':.9}
    assert backend.last_call['usage']['total_tokens']==110


def test_cloud_failure_does_not_execute_rules_fallback(tmp_path,monkeypatch):
    backend=cloud(tmp_path,monkeypatch)
    def fail(*args):raise TimeoutError('test')
    monkeypatch.setattr(backend,'decide',fail)
    lab,_=fake_lab();trace=Trace('unavailable');limits=Limits()
    graph=build_graph(lab,trace,limits,roles=LifecycleTestRoles(),profile=InstrumentProfile(),decision_backend=backend)
    result=graph.invoke(initial_state(lab,'unavailable',limits),{'configurable':{'thread_id':'unavailable'}})
    assert result['report']['terminal_reason']=='decision_model_error'
    assert not any(e['kind'] in {'tool_result','role_start'} for e in trace.events)


def verification_payload():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    state=initial_state(lab,'policy',Limits());state['observations']=lab.observe()
    return InstrumentProfile.projection('verification',state,'test')


@pytest.mark.parametrize('quality,value,expected',[
    ('valid',False,'contradicted'),('estimated',False,'contradicted'),
    ('valid',True,'supported'),('estimated',True,'supported'),
    ('unavailable',True,'unresolved')])
def test_shared_verification_quality_policy(quality,value,expected):
    payload=verification_payload();slot=payload['observations']['measurement']['native']['observations']['goal']
    slot.update(quality=quality,value=value)
    assert verification_status(payload)==expected
    assert payload['verification_policy']['accepted_native_qualities']==['valid','estimated']
    record=VerificationReport(status=expected,evidence_ids=['obs-1'],missing_evidence=[])
    InstrumentProfile.validate_role_contract('verification',record,payload)
    wrong=record.model_copy(update={'status':'unresolved' if expected!='unresolved' else 'supported'})
    with pytest.raises(ValueError):InstrumentProfile.validate_role_contract('verification',wrong,payload)
    assert wrong.status!=expected  # validator never rewrites the model's verdict


@pytest.mark.parametrize('change',['stale','identity','pending'])
def test_invalid_evidence_remains_unresolved(change):
    p=verification_payload();obs=p['observations'];m=obs['measurement']
    if change=='stale':m['captured_at_step']=obs['last_mutation_step']
    if change=='identity':m['sample_id']='other'
    if change=='pending':obs['pending_operation']={'operation_id':'active'}
    assert verification_status(p)=='unresolved'


@pytest.mark.parametrize('choice,code',[
    ({'finish_reason':'length','message':{'content':None}},'expert_output_truncated'),
    ({'finish_reason':'length','message':{'content':'{"partial":'}},'expert_output_truncated'),
    ({'finish_reason':'stop','message':{'content':None}},'expert_empty_output'),
    ({'finish_reason':'stop','message':{'content':'   '}},'expert_empty_output')])
def test_response_limits_are_not_json_format_errors(choice,code,monkeypatch):
    roles=LocalLlamaRoles(max_tokens=6144)
    async def request(*args):return {'choices':[choice],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
    monkeypatch.setattr(roles,'_request',request)
    with pytest.raises(ExpertResponseError) as exc:roles.invoke('verification','test',{},VerificationReport)
    assert exc.value.code==code and roles.last_call['response_classification']==code
    assert not isinstance(exc.value,ValueError)


def test_truncated_output_does_not_trigger_generic_format_retry():
    class Truncated:
        is_llm=True;backend_id='offline-truncation-double';calls=0
        def invoke(self,*args):
            self.calls+=1
            raise ExpertResponseError('expert_output_truncated','test')
    lab,_=fake_lab();roles=Truncated();trace=Trace('truncated');limits=Limits()
    graph=build_graph(lab,trace,limits,roles=roles,profile=InstrumentProfile())
    result=graph.invoke(initial_state(lab,'truncated',limits),{'configurable':{'thread_id':'truncated'}})
    assert roles.calls==1 and result['report']['terminal_reason']=='expert_output_truncated'
