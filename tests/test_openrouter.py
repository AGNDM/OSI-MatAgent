from io import BytesIO
import asyncio
import json
import time

import httpx

import pytest

from app.models import VerificationReport
from app.openrouter import OpenRouterRoles, is_free


def catalog(pricing=None):
    return [{"id": "test/model:free", "pricing": pricing or {"prompt": "0", "completion": "0"},
        "supported_parameters": ["temperature", "response_format", "reasoning"]}]


@pytest.mark.parametrize("price", ["0.01", "NaN", "Infinity", "-1"])
def test_nonzero_or_invalid_price_is_refused(monkeypatch, price):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    with pytest.raises(ValueError, match="not zero-priced"):
        OpenRouterRoles("test/model:free", catalog({"prompt": price, "completion": "0"}))


def test_paid_id_is_refused_before_any_network_access(monkeypatch):
    monkeypatch.setattr("app.openrouter.model_catalog", lambda: pytest.fail("must not fetch catalog"))
    with pytest.raises(ValueError, match="explicit :free"):
        OpenRouterRoles("test/paid-model")
    assert not is_free({"id": "test/model:free", "pricing": {"prompt": "0", "completion": "0", "request": "0.1"}})


def test_zero_price_request_records_usage_without_leaking_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    OpenRouterRoles._next_request_at = 0
    seen = []

    async def respond(self, body, timeout):
        seen.append(body)
        return httpx.Response(200,request=httpx.Request('POST','https://openrouter.ai/api/v1/chat/completions'),json={"id": "test-generation", "model": "test/model:free",
            "choices": [{"message": {"content": '{"status":"unresolved","evidence_ids":[],"missing_evidence":["mass"]}',
                                      "reasoning": "do not persist private reasoning"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0}})

    monkeypatch.setattr(OpenRouterRoles, "_completion", respond)
    backend = OpenRouterRoles("test/model:free", catalog())
    result = backend.invoke("verification", "prompt", {}, VerificationReport)
    assert result["status"] == "unresolved"
    body = seen[0]
    assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0}
    assert body["model"] == "test/model:free"
    assert "fake-test-key" not in json.dumps(backend.last_call)
    assert "private reasoning" not in json.dumps(backend.last_call)
    assert backend.last_call["usage"]["total_tokens"] == 30


def test_transport_failure_is_redacted_and_never_retried(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    OpenRouterRoles._next_request_at = 0
    attempts = []

    async def fail(self, body, timeout):
        attempts.append(body)
        return httpx.Response(429, request=httpx.Request('POST','https://openrouter.ai/api/v1/chat/completions'),json={"error":"fake-test-key echoed"})

    monkeypatch.setattr(OpenRouterRoles, "_completion", fail)
    backend = OpenRouterRoles("test/model:free", catalog())
    with pytest.raises(RuntimeError, match="HTTP 429") as exc:
        backend.invoke("verification", "prompt", {}, VerificationReport)
    assert len(attempts) == 1
    assert "fake-test-key" not in str(exc.value) + json.dumps(backend.last_call)
    assert backend.last_call["http_status"] == 429


def test_whole_request_deadline_cancels_keepalive_or_stalled_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")
    OpenRouterRoles._next_request_at = 0
    cancelled = []

    async def keepalive(self, body, timeout):
        try:
            while True:
                await asyncio.sleep(0.002)  # socket activity would never hit a read timeout
        finally:
            cancelled.append(True)

    monkeypatch.setattr(OpenRouterRoles, "_completion", keepalive)
    backend = OpenRouterRoles("test/model:free", catalog(), timeout_seconds=0.03)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.invoke("verification", "prompt", {}, VerificationReport)
    assert time.monotonic() - started < 0.5
    assert cancelled == [True]
