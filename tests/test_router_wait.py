from collections import Counter
from copy import deepcopy
import json

import pytest

from app.graph import build_graph, initial_state
from app.models import Limits, RouteDecision, OperationDecision, HypothesisSet
from app.single_instrument import InstrumentProfile
from app.local_decision import option_prompt
from app.trace import Trace
from tests.test_single_instrument import fake_lab as base_fake_lab, execute, LifecycleTestRoles


def fake_lab():
    lab,native=base_fake_lab()
    original=native.read
    native.running=True
    def read(**kwargs):
        result=original(**kwargs)
        result["observations"]["goal.running"]={"value":native.running,"quality":"valid","unit":"1","sample_time_s":native.time_s}
        return result
    native.read=read
    return lab,native


class WaitingRouter:
    is_decision_model=True
    backend_id='offline-rich-router-test-double'
    def __init__(self,lab):self.lab=lab;self.calls=[]
    def decide(self,summary,allowed):
        self.calls.append(deepcopy(summary))
        if summary['has_intent']:choice='prepare_action'
        elif not summary['has_inspection']:choice='diagnose'
        elif 'wait_long' in allowed:choice='wait_long'
        elif 'wait_short' in allowed:choice='wait_short'
        else:choice='verify'
        return RouteDecision(backend_id=self.backend_id,choice=choice,
            probabilities={r:float(r==choice) for r in allowed})


class TestRoles(LifecycleTestRoles):
    __test__=False
    def invoke(self,role,prompt,payload,schema):
        if role=='diagnostic' and schema.__name__=='OperationOptions':
            value=super().invoke(role,prompt,payload,HypothesisSet)
            return schema.model_validate({'hypotheses':value['hypotheses'], 'candidates':[{
                'candidate_id':'inspect','intent':value['intent'],'rationale':'Get current public state.','evidence_ids':[]}]}).model_dump()
        return super().invoke(role,prompt,payload,schema)


class TestChooser:
    __test__=False
    is_llm=False
    backend_id='offline-chooser-test-double'
    def choose(self,context,candidates):
        return OperationDecision(backend_id=self.backend_id,choice=candidates[0]['candidate_id'],
                                 probabilities={c['candidate_id']:float(i==0) for i,c in enumerate(candidates)})


def test_wait_bypasses_experts_and_chooser_but_uses_gate_executor_and_fresh_read():
    lab,native=fake_lab()
    original=native.step
    def step(dt):original(dt);native.goal=native.time_s>=30
    native.step=step
    trace=Trace('direct-wait');limits=Limits(lab_actions=12,llm_calls=12,graph_steps=100)
    router=WaitingRouter(lab)
    graph=build_graph(lab,trace,limits,roles=TestRoles(),decision_backend=router,
                      operation_backend=TestChooser(),profile=InstrumentProfile())
    state=graph.invoke(initial_state(lab,'direct-wait',limits),{'configurable':{'thread_id':'direct-wait'},'recursion_limit':100})
    assert lab.score(state['report'])['goal_met']
    assert state['report']['router_wait_count']==1
    assert [e['proposal']['capability_id'] for e in trace.events if e['kind']=='tool_result']==['workflow.inspect','workflow.await']
    assert Counter(e['component'] for e in trace.events if e['kind']=='role_start')=={'diagnostic':1,'procedure':1,'verification':1}
    assert sum(e['kind']=='operation_start' for e in trace.events)==1
    assert sum(e['kind']=='accepted' for e in trace.events)==2
    assert native.time_s==120 and not native.calls
    assert any(s['instrument_context']['goal_checks'][0]['met'] for s in router.calls)
    assert all(len(e['allowed_routes'])<=4 for e in trace.events if e['kind']=='routing')


def test_wait_context_contains_current_public_state_goal_history_and_budgets():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    state=initial_state(lab,'context',Limits());state['observations']=lab.observe()
    summary,allowed=lab.route_context(state,{'has_intent':False},['diagnose','deliberate','verify'])
    context=summary['instrument_context']
    assert context['objective']==lab.task.objective
    assert context['public_state']['goal']['value'] is False
    assert context['recent_actions'][-1]['capability_id']=='workflow.inspect'
    assert set(context['wait_offers'])=={'wait_short','wait_long'}
    text,_=option_prompt(summary,allowed)
    assert 'duration_s=120.0' in text and 'intermediate read alone' in text
    for forbidden in ['reference_probe','native_fixture_events','truth.','debug.']:
        assert forbidden not in text
    native.goal=True;execute(lab,'workflow.readback');state['observations']=lab.observe()
    assert lab.router_wait_offers(state)=={}


def test_wait_menu_bounds_freshness_errors_and_legacy_disable():
    lab,native=fake_lab();state=initial_state(lab,'offers',Limits());state['observations']=lab.observe()
    assert not lab.router_wait_offers(state)
    execute(lab,'workflow.inspect');state['observations']=lab.observe()
    lab.lifecycle.elapsed=lab.task.max_simulated_seconds-5
    assert lab.router_wait_offers(state)=={'wait_short':5}
    state['last_error']={'code':'FAIL'};assert not lab.router_wait_offers(state)
    state['last_error']=None;state['observations']['measurement']['revision']=-1
    assert not lab.router_wait_offers(state)
    lab.router_wait_enabled=False
    assert lab.route_context(state,{},['diagnose','deliberate','verify'])==({},['diagnose','deliberate','verify'])


def test_stale_router_wait_cannot_advance_clock():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    class StaleRouter(WaitingRouter):
        def decide(self,summary,allowed):
            value=super().decide(summary,allowed)
            self.lab.revision+=1
            return value
    trace=Trace('stale');limits=Limits()
    graph=build_graph(lab,trace,limits,roles=TestRoles(),decision_backend=StaleRouter(lab),profile=InstrumentProfile())
    state=initial_state(lab,'stale',limits)
    final=graph.invoke(state,{'configurable':{'thread_id':'stale'},'recursion_limit':100})
    assert final['report']['terminal_reason']=='stale_router_wait'
    assert native.time_s==0


def test_direct_wait_obeys_action_budget_before_native_step():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    trace=Trace('budget');limits=Limits(lab_actions=2)
    graph=build_graph(lab,trace,limits,roles=TestRoles(),decision_backend=WaitingRouter(lab),profile=InstrumentProfile())
    final=graph.invoke(initial_state(lab,'budget',limits),{'configurable':{'thread_id':'budget'},'recursion_limit':100})
    assert final['report']['terminal_reason']=='action_budget' and native.time_s==0
    assert not any(e['kind']=='tool_result' for e in trace.events)


def test_idle_process_and_unrelated_fan_cannot_offer_wait():
    lab,native=fake_lab();native.running=False
    execute(lab,'workflow.inspect')
    state=initial_state(lab,'idle',Limits());state['observations']=lab.observe()
    state['observations']['measurement']['native']['observations']['fan.running']={'value':True,'quality':'valid'}
    assert not lab.router_wait_offers(state)
    native.running=True;execute(lab,'workflow.readback');state['observations']=lab.observe()
    assert lab.router_wait_offers(state)
    state['observations']['measurement']['native']['observations']['goal.phase']={'value':'paused','quality':'valid'}
    assert not lab.router_wait_offers(state)


def test_router_receives_concrete_current_intent_and_stale_or_error_context():
    lab,native=fake_lab();execute(lab,'workflow.inspect')
    state=initial_state(lab,'selected-intent',Limits());state['observations']=lab.observe()
    state['observation_revision']=lab.revision
    state['intent']={'capability_id':'binder_fd056::act','arguments':{'optional':True},
        'discriminating_observation':'Current public readback','postcondition':'Change goal'}
    summary,allowed=lab.route_context(state,{'has_intent':True},['diagnose','prepare_action','verify'])
    context=summary['instrument_context']
    assert context['selected_intent']==state['intent'] and context['selected_intent_current']
    assert not context['wait_offers']
    prompt,_=option_prompt(summary,allowed)
    assert 'ALREADY been selected' in prompt and 'binder_fd056::act' in prompt and '"optional": true' in prompt
    state['observation_revision']-=1
    stale,allowed=lab.route_context(state,{'has_intent':True},allowed)
    assert not stale['instrument_context']['selected_intent_current']
    assert 'ALREADY been selected' not in option_prompt(stale,allowed)[0]
    state['observation_revision']=lab.revision;state['last_error']={'code':'NEW_ERROR'}
    failed,allowed=lab.route_context(state,{'has_intent':True},allowed)
    assert 'ALREADY been selected' not in option_prompt(failed,allowed)[0]
