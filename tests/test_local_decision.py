import asyncio
import json
import math
import time

import httpx
import pytest

from app.decision_backend import RulesBackend, select_route, summarize
from app.graph import initial_state
from app.lab_adapter import LabAdapter
from app.local_decision import LocalOpenJevBackend, option_prompt, read_probabilities
from app.models import Limits
from app.scenarios import case_by_id
from bench.run import run_episode, summarize_runs


def raw_probabilities(values):
    return {"probs": [{"top_probs": [
        {"token": letter, "bytes": [ord(letter)], "prob": prob}
        for letter, prob in values.items()]}]}


def test_real_full_vocabulary_logprobs_are_conditioned_on_candidates():
    details = {}
    response = {"completion_probabilities": [{"top_logprobs": [
        {"token": "A", "bytes": [65], "logprob": math.log(.2)},
        {"token": "B", "bytes": [66], "logprob": math.log(.7)},
        {"token": "E", "bytes": [69], "logprob": math.log(.1)},
    ]}]}
    result = read_probabilities(response, {"A": "diagnose", "B": "verify"}, details=details)
    assert result == {"diagnose": pytest.approx(2/9), "verify": pytest.approx(7/9)}
    assert details["option_probability_mass"] == pytest.approx(.9)
    assert details["readout_source"] == "logprobs"


def test_readout_requires_every_real_letter_probability():
    options = {"A": "diagnose", "B": "prepare_action", "C": "verify"}
    assert read_probabilities(raw_probabilities({"A": .2, "B": .7, "C": .1}), options) == {
        "diagnose": pytest.approx(.2), "prepare_action": pytest.approx(.7), "verify": pytest.approx(.1)}
    with pytest.raises(ValueError, match="Incomplete"):
        read_probabilities(raw_probabilities({"A": 1}), options)
    assert read_probabilities(raw_probabilities({"A": .1, "B": .1, "C": .1}), options) == {
        "diagnose": pytest.approx(1/3), "prepare_action": pytest.approx(1/3), "verify": pytest.approx(1/3)}
    assert read_probabilities(raw_probabilities({"A": .2, "B": .6, "C": .1, "X": .1}), options)["prepare_action"] == pytest.approx(2/3)
    with pytest.raises(ValueError, match="Invalid"):
        read_probabilities(raw_probabilities({"A": 1, "X": float("nan")}), {"A": "diagnose"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_probability_is_rejected(value):
    with pytest.raises(ValueError, match="Invalid"):
        read_probabilities(raw_probabilities({"A": value}), {"A": "diagnose"})


def test_local_model_can_only_target_loopback():
    with pytest.raises(ValueError, match="loopback"):
        LocalOpenJevBackend("https://example.com")
    with pytest.raises(ValueError, match="loopback"):
        LocalOpenJevBackend("http://127.0.0.1@evil.example:8792")


def test_prompt_reorders_labels_together_with_routes():
    text, options = option_prompt({"has_intent": True}, ["verify", "prepare_action"])
    assert options == {"A": "verify", "B": "prepare_action"}
    assert "[B] prepare_action" in text


def test_backend_uses_one_token_and_preserves_actual_probabilities(monkeypatch):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "model formatted request"})
        assert body["n_predict"] == 1 and body["post_sampling_probs"] is False
        assert body["temperature"] < 0
        assert body["grammar"] == "root ::= [ABC]"
        # Even if sampling emitted A, this backend chooses the maximum-probability option B.
        return httpx.Response(200, json=dict(raw_probabilities({"A": .1, "B": .8, "C": .1}), content="A"))

    client_type = httpx.AsyncClient
    monkeypatch.setattr("app.local_decision.httpx.AsyncClient", lambda **kwargs:
        client_type(transport=httpx.MockTransport(respond), **kwargs))
    backend = LocalOpenJevBackend()
    result = backend.decide({"has_intent": True}, ["diagnose", "prepare_action", "verify"])
    assert result.choice == "prepare_action"
    assert result.probabilities["prepare_action"] == .8
    assert [path for path, _ in calls] == ["/apply-template", "/completion"]
    assert backend.last_call["response"]["content"] == "A"


def test_missing_probability_causes_explicit_rules_fallback(monkeypatch):
    async def fail(self, summary, allowed, timeout):
        raise ValueError("Incomplete option readout")

    monkeypatch.setattr(LocalOpenJevBackend, "_infer", fail)
    lab = LabAdapter(case_by_id("dev-01"))
    state = initial_state(lab, "readout-test", Limits())
    state["observations"] = lab.observe()
    backend = LocalOpenJevBackend()
    decision, error = select_route(backend, summarize(state), ["diagnose", "deliberate"])
    assert decision.backend_id == "rules-v1" and decision.fallback_used
    assert decision.probabilities is None
    assert "Incomplete" in error


def test_local_inference_has_a_total_deadline(monkeypatch):
    async def blocked(self, summary, allowed, timeout):
        await asyncio.sleep(10)

    monkeypatch.setattr(LocalOpenJevBackend, "_infer", blocked)
    backend = LocalOpenJevBackend(timeout_seconds=.02)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.decide({}, ["diagnose"])
    assert time.monotonic() - start < .5


def test_router_comparisons_keep_denominators_separate():
    first = run_episode(case_by_id("dev-03"))
    second = dict(first, decision_backend="another-router", score=dict(first["score"], task_success=False))
    rows = [r for r in summarize_runs([first, second]) if r["family"] == "overall"]
    assert len(rows) == 2 and {r["task_success_rate"] for r in rows} == {0, 1}
