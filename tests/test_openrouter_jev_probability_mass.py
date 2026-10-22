"""Hosted Jev scores retain native rounding through every decision boundary."""
import httpx
import pytest

from app.decision_backend import select_route
from app.graph import build_graph, initial_state
from app.models import Limits
from app.openrouter_jev import MODEL, OpenRouterJevBackend, OpenRouterJevOperationBackend
from app.operation_decision import validate_decision
from app.single_instrument import InstrumentProfile
from app.token_budget import TokenLedger
from app.trace import Trace
from tests.test_router_wait import WaitingRouter, TestRoles, TestChooser, fake_lab


@pytest.mark.parametrize('model', [MODEL, 'openai/gpt-6-luna-decisions'])
@pytest.mark.parametrize('operation', [False, True])
def test_native_rounded_scores_pass_downstream_validation(tmp_path, monkeypatch, operation, model):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    scores = {'A': .05, 'B': .93, 'C': .01}

    async def post(*args, **kwargs):
        return httpx.Response(200, json={'model': model, 'answers': {'next_action': {
            'type': 'choice', 'choice': 'B', 'probabilities': scores}},
            'usage': {'input_tokens': 10, 'output_tokens': 2}})

    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    backend_type = OpenRouterJevOperationBackend if operation else OpenRouterJevBackend
    backend = backend_type(TokenLedger(tmp_path / 'ledger.jsonl'), model=model)
    assert backend.manifest()['model'] == model
    options = ['diagnose', 'verify', 'deliberate']
    if operation:
        candidates = [{'candidate_id': name, 'intent': {'capability_id': 'workflow.inspect',
            'arguments': {}, 'postcondition': 'Inspect.', 'discriminating_observation': 'Read state.'},
            'rationale': 'Check evidence.', 'evidence_ids': []} for name in options]
        decision = validate_decision(backend.choose({}, candidates), candidates)
    else:
        decision, error = select_route(backend, {}, options)
        assert error is None
    assert decision.choice == 'verify'
    assert decision.probabilities == dict(zip(options, scores.values()))
    assert backend.ledger.snapshot()['charged_tokens'] == 12


def test_rounded_hosted_jev_scores_pass_wait_authorization():
    class RoundedRouter(WaitingRouter):
        backend_id = OpenRouterJevBackend.backend_id

        def decide(self, summary, allowed):
            decision = super().decide(summary, allowed)
            decision.probabilities[decision.choice] = .99
            return decision

    lab, native = fake_lab()
    original = native.step

    def step(dt):
        original(dt)
        native.goal = native.time_s >= 30

    native.step = step
    limits = Limits(lab_actions=12, llm_calls=12, graph_steps=100)
    trace = Trace('rounded-wait')
    graph = build_graph(lab, trace, limits, roles=TestRoles(), decision_backend=RoundedRouter(lab),
                        operation_backend=TestChooser(), profile=InstrumentProfile())
    state = graph.invoke(initial_state(lab, 'rounded-wait', limits),
                         {'configurable': {'thread_id': 'rounded-wait'}, 'recursion_limit': 100})
    assert lab.score(state['report'])['goal_met']
    assert state['report']['router_wait_count'] == 1
    assert any(e['kind'] == 'tool_result' and e['proposal']['capability_id'] == 'workflow.await'
               for e in trace.events)


@pytest.mark.parametrize('scores,choice', [
    ({'A': -.01, 'B': .93, 'C': .01}, 'B'),
    ({'A': .05, 'B': 1.01, 'C': .01}, 'B'),
    ({'A': float('inf'), 'B': .93, 'C': .01}, 'B'),
    ({'A': float('nan'), 'B': .93, 'C': .01}, 'B'),
    ({'A': .05, 'B': .93}, 'B'),
    ({'A': .05, 'B': .93, 'C': .01}, 'A'),
])
def test_rounded_scores_still_require_valid_complete_argmax(tmp_path, monkeypatch, scores, choice):
    import asyncio
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    backend = OpenRouterJevBackend(TokenLedger(tmp_path / 'ledger.jsonl'))
    backend.last_call = {}

    async def post(*args, **kwargs):
        response = httpx.Response(200)
        monkeypatch.setattr(response, 'json', lambda: {'model': MODEL, 'answers': {'next_action': {
            'type': 'choice', 'choice': choice, 'probabilities': scores}},
            'usage': {'input_tokens': 10, 'output_tokens': 2}})
        return response

    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    with pytest.raises(ValueError):
        asyncio.run(backend._infer_options('choose', {'A': 'first', 'B': 'second', 'C': 'third'}, 1))
    assert backend.ledger.snapshot()['charged_tokens'] == 12


@pytest.mark.parametrize("model", [MODEL, "openai/gpt-6-luna-decisions"])
def test_single_legal_option_is_forced_without_api(tmp_path, monkeypatch, model):
    import asyncio
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    async def forbidden(*args, **kwargs):
        raise AssertionError("Singleton must not call the model API")
    monkeypatch.setattr(httpx.AsyncClient, "post", forbidden)
    backend=OpenRouterJevBackend(TokenLedger(tmp_path / "ledger.jsonl"), model=model,force_singleton=True)
    backend.last_call={}
    assert asyncio.run(backend._infer_options("choose", {"A":"diagnose"}, 1)) == {"diagnose":1.0}
    assert backend.last_call["decision_source"] == "single_legal_option"
    assert backend.last_call["request_sent"] is False
