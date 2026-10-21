import asyncio
import httpx
import pytest
from app.expert_config import ExpertEndpoint
from app.token_budget import BudgetedExpert, TokenLedger


def test_openrouter_uses_requested_model_and_shared_bounded_transport(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','test-key-never-log')
    config=ExpertEndpoint(name='openrouter-debug',provider='openrouter',base_url='https://openrouter.ai/api/v1',
        model='deepseek/deepseek-v4.1-flash',api_key_env='OPENROUTER_API_KEY')
    ledger=TokenLedger(tmp_path/'ledger.jsonl');expert=BudgetedExpert(config,ledger)
    body=expert.prepare_request({'model':config.model,'messages':[{'role':'user','content':'test'}],
        'max_tokens':100,'stream':False,'chat_template_kwargs':{'enable_thinking':False}})
    assert body['reasoning']=={'enabled':False} and 'chat_template_kwargs' not in body
    with pytest.raises(ValueError):ledger.reserve({**body,'enable_thinking':True})
    async def post(self,url,**kwargs):
        assert url=='https://openrouter.ai/api/v1/chat/completions'
        assert kwargs['headers']['Authorization']=='Bearer test-key-never-log'
        return httpx.Response(200,json={'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    expert.last_call={}
    asyncio.run(expert._request(body,1))
    assert ledger.snapshot()['charged_tokens']==15
    async def fail(*args,**kwargs):raise httpx.ReadTimeout('test timeout')
    monkeypatch.setattr(httpx.AsyncClient,'post',fail)
    with pytest.raises(httpx.ReadTimeout):asyncio.run(expert._request(body,1))
    state=ledger.snapshot()
    assert state['unmetered_requests']==1 and state['inflight_or_crashed']==0 and state['charged_tokens']>15


def test_other_openrouter_models_are_not_implicitly_authorized(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENROUTER_API_KEY','test')
    config=ExpertEndpoint(name='other',provider='openrouter',base_url='https://openrouter.ai/api/v1',
        model='other/model',api_key_env='OPENROUTER_API_KEY')
    with pytest.raises(ValueError):BudgetedExpert(config,TokenLedger(tmp_path/'ledger.jsonl'))
