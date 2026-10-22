"""Adapter contract checks; execute only inside an actual Slurm allocation.

Fakes isolate transport/schema contracts; real-model graph smoke runs belong to
the integration job and are never represented by these tests.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
import time

import pytest

from app.models import VerificationReport
from graph_local.backends import LocalDecisionBackend, LocalHFRoles


class FakeHF:
    max_new_tokens = 4096
    identity = {"backend": "hf", "weight_sha256": {"weights": "test-fixture-only"}}

    def __init__(self, raw=None, generated=15, error=None):
        self.raw = raw if raw is not None else '{"status":"unresolved","evidence_ids":[],"missing_evidence":["fresh measurement"]}'
        self.generated, self.error = generated, error
        self.model = SimpleNamespace(generation_config=SimpleNamespace(max_time=None))
        self.last_generation_receipt = {}
        self.calls = []

    def _generate(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), "kwargs": kwargs,
                           "max_time": self.model.generation_config.max_time})
        if self.error:
            raise self.error
        self.last_generation_receipt = {"input_tokens": 30, "output_tokens": self.generated,
                                        "generation_max_time_s": self.model.generation_config.max_time}
        return self.raw, 30, self.generated


class FakeChooser:
    identity = {"backend": "test-double-not-a-model"}

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def decide(self, messages, labels, **kwargs):
        self.calls.append({"messages": deepcopy(messages), "labels": labels, "kwargs": kwargs})
        return deepcopy(self.result or {"status": "ok", "selected_option_id": labels[-1],
                                       "probabilities": {label: float(label == labels[-1]) for label in labels},
                                       "input_tokens": 45, "output_tokens": 0})


def role_backend(fake=None):
    return LocalHFRoles({"model_path": "/fixture/not-loaded"}, backend=fake or FakeHF())


def operation_candidates():
    return [{"candidate_id": f"candidate-{index}",
             "intent": {"capability_id": capability, "arguments": {},
                        "discriminating_observation": "Read public observation obs-7.",
                        "postcondition": "Observe the requested state change."},
             "rationale": "Compare the exact public goal and current evidence.",
             "evidence_ids": ["obs-7"]}
            for index, capability in enumerate(["workflow.inspect", "power.on"])]


def test_schema_json_uses_actual_role_contract_and_accounts_tokens():
    fake = FakeHF()
    roles = role_backend(fake)
    payload = {"observations": {"revision": 7, "very_old_observation": {"value": "keep this evidence"}}}
    result = roles.invoke("verification", "Verify using only observations.", payload, VerificationReport)
    assert result["status"] == "unresolved"
    assert "keep this evidence" in fake.calls[0]["messages"][1]["content"]
    assert '"additionalProperties":false' in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[0]["kwargs"] == {"sample": False, "seed": 0}
    assert 0 < fake.calls[0]["max_time"] <= 120
    assert fake.model.generation_config.max_time is None
    assert roles.last_call["usage"] == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
    assert roles.last_call["request_sent"]
    assert roles.last_call["response_classification"] == "valid_schema_json"
    assert roles.manifest()["context_policy"] == "error_without_truncation"


@pytest.mark.parametrize("raw", [
    '```json\n{"status":"unresolved","evidence_ids":[],"missing_evidence":[]}\n```',
    '{"status":"unresolved","evidence_ids":[],"missing_evidence":[],"hidden":1}',
    '{"status":"supported","status":"unresolved","evidence_ids":[],"missing_evidence":[]}',
    '{"status":"unresolved","evidence_ids":[],"missing_evidence":[NaN]}',
    '[{"status":"unresolved","evidence_ids":[],"missing_evidence":[]}]',
])
def test_no_json_repair_or_unvalidated_specialist_fallback(raw):
    fake = FakeHF(raw)
    roles = role_backend(fake)
    with pytest.raises(ValueError):
        roles.invoke("verification", "Verify.", {}, VerificationReport)
    assert len(fake.calls) == 1
    assert roles.last_call["error"]
    assert fake.model.generation_config.max_time is None


def test_output_limit_is_explicit_truncation_even_if_suffix_parses():
    fake = FakeHF(generated=4096)
    roles = role_backend(fake)
    with pytest.raises(RuntimeError) as caught:
        roles.invoke("verification", "Verify.", {}, VerificationReport)
    assert caught.value.code == "expert_output_truncated"
    assert roles.last_call["response_classification"] == "expert_output_truncated"
    assert roles.last_call["response"]["choices"][0]["finish_reason"] == "length"


def test_context_overflow_is_preserved_without_input_truncation():
    fake = FakeHF(error=ValueError("Public input exceeds context; evidence was not truncated"))
    roles = role_backend(fake)
    with pytest.raises(ValueError, match="evidence was not truncated"):
        roles.invoke("verification", "Verify.", {"history": list(range(300))}, VerificationReport)
    assert json.loads(fake.calls[0]["messages"][1]["content"])["history"] == list(range(300))
    assert fake.model.generation_config.max_time is None


def test_expired_deadline_prevents_generation():
    fake = FakeHF()
    roles = role_backend(fake)
    roles.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        roles.invoke("verification", "Verify.", {}, VerificationReport)
    assert fake.calls == []
    assert roles.last_call["request_sent"] is False


def test_generation_deadline_settings_restore_on_failure():
    fake = FakeHF(error=RuntimeError("actual generation failed"))
    fake.model.generation_config.max_time = 9.5
    roles = role_backend(fake)
    with pytest.raises(RuntimeError, match="actual generation failed"):
        roles.invoke("verification", "Verify.", {}, VerificationReport)
    assert fake.model.generation_config.max_time == 9.5


def test_completion_after_episode_deadline_is_rejected_with_usage_retained():
    fake = FakeHF()
    roles = role_backend(fake)
    generate = fake._generate
    def complete_after_deadline(*args, **kwargs):
        result = generate(*args, **kwargs)
        roles.deadline = time.monotonic() - 1
        return result
    fake._generate = complete_after_deadline
    with pytest.raises(TimeoutError, match="completion rejected"):
        roles.invoke("verification", "Verify.", {}, VerificationReport)
    assert roles.last_call["usage"]["total_tokens"] == 45
    assert roles.last_call["response_classification"] == "expert_deadline_exceeded"
    assert fake.model.generation_config.max_time is None


def test_shared_backend_budget_must_match_and_lock_is_shared():
    fake = FakeHF()
    roles = role_backend(fake)
    chooser = LocalDecisionBackend({"kind": "hf", "model_path": "/fixture/not-loaded"}, backend=fake)
    assert roles._model_lock is chooser._model_lock
    with pytest.raises(ValueError, match="declared JSON generation budget"):
        LocalHFRoles({"model_path": "/fixture/not-loaded"}, backend=fake, max_new_tokens=128)


def test_shared_backend_cannot_silently_change_declared_weights():
    fake = FakeHF()
    fake.identity = {"model_path": "/fixture/different-local-weights"}
    with pytest.raises(ValueError, match="declared local artifact"):
        role_backend(fake)
    with pytest.raises(ValueError, match="declared local artifact"):
        LocalDecisionBackend({"kind": "hf", "model_path": "/fixture/not-loaded"}, backend=fake)


def test_real_route_prompt_is_retained_and_probabilities_map_to_routes():
    fake = FakeChooser()
    chooser = LocalDecisionBackend({"kind": "hf", "model_path": "/fixture/not-loaded"}, backend=fake)
    summary = {"has_intent": True, "observations": {"state": "obs-7"}}
    routes = ["diagnose", "prepare_action"]
    result = chooser.decide(summary, routes)
    assert result.choice == "prepare_action"
    assert result.probabilities == {"diagnose": 0., "prepare_action": 1.}
    assert result.fallback_used is False
    assert "obs-7" in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[0]["kwargs"]["method"] == "logprobs"
    assert set(fake.calls[0]["kwargs"]["option_descriptions"]) == {"A", "B"}
    assert chooser.last_call["usage"]["total_tokens"] == 45


@pytest.mark.parametrize("kind,expected_method", [("hf", "logprobs"), ("native", "native"), ("features", "features")])
def test_operation_candidates_and_full_public_evidence_preserved(kind, expected_method):
    fake = FakeChooser()
    chooser = LocalDecisionBackend({"kind": kind, "model_path": "/fixture/not-loaded"}, backend=fake)
    context = {"observations": {"current": "obs-7", "previous": "obs-2"},
               "target": {"door.closed": True}, "retrieval_evidence": [{"id": "manual-1", "text": "full manual text"}]}
    candidates = operation_candidates()
    result = chooser.choose(context, candidates)
    assert result.choice == "candidate-1"
    assert result.probabilities == {"candidate-0": 0., "candidate-1": 1.}
    call = fake.calls[0]
    assert call["kwargs"]["method"] == expected_method
    text = call["messages"][0]["content"]
    assert all(value in text for value in ["obs-7", "obs-2", "full manual text", "workflow.inspect", "power.on"])
    assert chooser.last_call["public_input"]["context"] == context
    assert set(call["kwargs"]["option_descriptions"]) == {"A", "B"}
    assert "evidence_ids" in call["kwargs"]["option_descriptions"]["B"]


@pytest.mark.parametrize("result", [
    {"status": "ok", "selected_option_id": "A", "probabilities": {"A": 1.}},
    {"status": "ok", "selected_option_id": "A", "probabilities": {"A": .8, "B": .8}},
    {"status": "ok", "selected_option_id": "A", "probabilities": {"A": .1, "B": .9}},
    {"status": "ok", "selected_option_id": "C", "probabilities": {"A": .1, "B": .9}},
    {"status": "ok", "selected_option_id": "B", "probabilities": {"A": float("nan"), "B": 1.}},
    {"status": "ok", "selected_option_id": "B", "probabilities": {"A": False, "B": True}},
    {"status": "error", "selected_option_id": None, "probabilities": None, "error": "Native context limit exceeded"},
])
def test_no_incomplete_vectors_nonargmax_or_error_fallback(result):
    fake = FakeChooser(result)
    chooser = LocalDecisionBackend({"kind": "native", "model_path": "/fixture/not-loaded"}, backend=fake)
    with pytest.raises((ValueError, RuntimeError)):
        chooser.choose({"observations": {}}, operation_candidates())
    assert len(fake.calls) == 1
    assert chooser.last_call["error"]
    assert chooser.allow_rules_fallback is False


def test_chooser_deadline_prevents_inference():
    fake = FakeChooser()
    chooser = LocalDecisionBackend({"kind": "hf", "model_path": "/fixture/not-loaded"}, backend=fake)
    chooser.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        chooser.choose({}, operation_candidates())
    assert fake.calls == []


def test_decision_computed_after_deadline_is_rejected():
    fake = FakeChooser()
    chooser = LocalDecisionBackend({"kind": "hf", "model_path": "/fixture/not-loaded"}, backend=fake)
    infer = fake.decide
    def late_decision(*args, **kwargs):
        result = infer(*args, **kwargs)
        chooser.deadline = time.monotonic() - 1
        return result
    fake.decide = late_decision
    with pytest.raises(TimeoutError, match="decision rejected"):
        chooser.choose({}, operation_candidates())
    assert chooser.last_call["usage"]["total_tokens"] == 45


def test_episode_deadline_is_forwarded_to_subprocess_compatible_backend():
    fake = FakeChooser()
    fake.deadline = float("inf")
    chooser = LocalDecisionBackend({"kind": "native", "family": "apus", "model_path": "/fixture/not-loaded"},
                                   backend=fake, timeout_seconds=60)
    chooser.deadline = time.monotonic() + 15
    chooser.choose({}, operation_candidates())
    # Deriving the remaining call budget can conservatively shorten it by the
    # elapsed clock-read time. The child must receive a live finite deadline
    # that never exceeds its episode deadline.
    assert time.monotonic() < fake.deadline <= chooser.deadline


def test_no_generated_probability_or_hosted_model_kind():
    with pytest.raises(ValueError, match="actual complete logprobs"):
        LocalDecisionBackend({"kind": "hf", "method": "verbalized"}, backend=FakeChooser())
    with pytest.raises(ValueError, match="local HF"):
        LocalDecisionBackend({"kind": "openrouter"}, backend=FakeChooser())
    with pytest.raises(ValueError, match="offline Hugging Face"):
        LocalHFRoles({"backend": "openrouter"}, backend=FakeHF())


def test_allocation_guard_survives_backend_injection(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        role_backend(FakeHF())
    with pytest.raises(RuntimeError, match="Slurm"):
        LocalDecisionBackend({"kind": "hf"}, backend=FakeChooser())
