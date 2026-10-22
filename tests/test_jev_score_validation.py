"""Regression coverage for raw Jev option values; every HTTP call is mocked."""
from copy import deepcopy
import json

import httpx
import pytest

from app.decision_backend import select_route
from app.graph import build_graph, initial_state
from app.models import Limits
from app.openrouter_jev import OpenRouterJevBackend, OpenRouterJevOperationBackend, MODEL
from app.operation_decision import validate_decision
from app.single_instrument import InstrumentProfile
from app.token_budget import TokenLedger
from app.trace import Trace
from tests.test_router_wait import fake_lab, WaitingRouter, TestChooser, TestRoles


def candidates():
    return [{'candidate_id': name, 'intent': {'capability_id': cap, 'arguments': args,
        'discriminating_observation': 'Fresh public state.', 'postcondition': 'Check current state.'},
        'rationale': 'Compare public options.', 'evidence_ids': []}
        for name,cap,args in [
            ('await-120','workflow.await',{'duration_s':120}),
            ('readback-now','workflow.readback',{}),
            ('describe-await','workflow.describe',{'capability_id':'workflow.await'}),
            ('describe-stirrer','workflow.describe',{'capability_id':'mt_rc1mxhfcal::stirrer.set'})]]


def install_response(monkeypatch, probabilities, choice='A'):
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-test-key')
    reply = {'model':MODEL+'-20261014','answers':{'next_action':{
        'type':'choice','choice':choice,'probabilities':deepcopy(probabilities),'confidence':.91}},
        'usage':{'input_tokens':12,'output_tokens':4}}
    async def post(*args,**kwargs):
        # Serialize explicitly so NaN/Infinity fixtures reach the value checker.
        return httpx.Response(200,content=json.dumps(reply),headers={'content-type':'application/json'})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    return reply


@pytest.mark.parametrize('values',[
    {'A':.93,'B':.04,'C':0,'D':.02},  # Exact trial02 response: total .99.
    {'A':.8,'B':.7,'C':.6,'D':.5},   # No upper bound on the aggregate.
    {'A':.2,'B':.1,'C':0,'D':0},     # No near-one rounding tolerance.
    {'A':.4,'B':.4,'C':0,'D':0},     # Tied maxima remain valid.
])
def test_cloud_choices_and_routes_preserve_nonunit_totals(tmp_path,monkeypatch,values):
    reply=install_response(monkeypatch,values)
    ledger=TokenLedger(tmp_path/'ledger.jsonl')
    chooser=OpenRouterJevOperationBackend(ledger)
    options=candidates()
    decision=validate_decision(chooser.choose({},options),options)
    expected={candidate['candidate_id']:value for candidate,value in zip(options,values.values())}
    assert decision.choice=='await-120' and decision.probabilities==expected
    assert chooser.last_call['response']==reply
    assert chooser.last_call['native_probability_sum']==pytest.approx(sum(values.values()))
    router=OpenRouterJevBackend(ledger)
    routes=['diagnose','prepare_action','verify','deliberate']
    routed,error=select_route(router,{'has_intent':False},routes)
    assert error is None and routed.choice=='diagnose'
    assert routed.probabilities==dict(zip(routes,values.values()))
    assert ledger.snapshot()['charged_tokens']==32


@pytest.mark.parametrize('values,choice',[
    ({'A':.93,'B':.04,'C':0},'A'),
    ({'A':.93,'B':.04,'C':0,'D':.02,'E':.01},'A'),
    ({'A':-.1,'B':.04,'C':0,'D':.02},'B'),
    ({'A':1.1,'B':.04,'C':0,'D':.02},'A'),
    ({'A':float('nan'),'B':.04,'C':0,'D':.02},'A'),
    ({'A':float('inf'),'B':.04,'C':0,'D':.02},'A'),
    ({'A':True,'B':.04,'C':0,'D':.02},'A'),
    ({'A':.93,'B':.04,'C':0,'D':.02},'B'),
    ({'A':.93,'B':.04,'C':0,'D':.02},'E'),
])
def test_cloud_still_rejects_invalid_values_missing_options_and_wrong_choice(tmp_path,monkeypatch,values,choice):
    install_response(monkeypatch,values,choice)
    backend=OpenRouterJevOperationBackend(TokenLedger(tmp_path/'ledger.jsonl'))
    with pytest.raises(ValueError):
        backend.choose({},candidates())
    assert backend.ledger.snapshot()['charged_tokens']==16


def test_nonunit_scores_pass_graph_selection_handoff_and_direct_wait():
    lab,native=fake_lab()
    original_step=native.step
    def step(dt):
        original_step(dt)
        native.goal=native.time_s>=30
    native.step=step
    class RawRouter(WaitingRouter):
        def decide(self,summary,allowed):
            result=super().decide(summary,allowed)
            return result.model_copy(update={'probabilities':{
                option:.93 if option==result.choice else .02 for option in allowed}})
    class RawChooser(TestChooser):
        def choose(self,context,options):
            result=super().choose(context,options)
            return result.model_copy(update={'probabilities':{
                option:.93 if option==result.choice else .02 for option in result.probabilities}})
    trace=Trace('nonunit-choice-regression')
    limits=Limits(lab_actions=12,llm_calls=12,graph_steps=100)
    graph=build_graph(lab,trace,limits,roles=TestRoles(),decision_backend=RawRouter(lab),
        operation_backend=RawChooser(),profile=InstrumentProfile())
    state=graph.invoke(initial_state(lab,trace.episode_id,limits),
        {'configurable':{'thread_id':trace.episode_id},'recursion_limit':100})
    assert lab.score(state['report'])['goal_met'] and native.time_s==120
    assert [e['proposal']['capability_id'] for e in trace.events if e['kind']=='tool_result']==[
        'workflow.inspect','workflow.await']
    assert not any(e['kind'] in {'operation_error','selection_rejected','rejected'} for e in trace.events)
    selected=next(e for e in trace.events if e['kind']=='operation_selected')
    assert selected['decision']['probabilities']=={'inspect':.93}
