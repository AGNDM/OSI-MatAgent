import asyncio
import json
import time

import pytest

from app.local_roles import LocalLlamaRoles
from app.models import VerificationReport


def test_local_specialists_use_json_and_record_real_usage_without_reasoning(monkeypatch):
    async def respond(self, body, timeout):
        assert body["response_format"] == {"type": "json_object"}
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        return {"choices": [{"message": {"content": '{"status":"unresolved","evidence_ids":[],"missing_evidence":["mass"]}', "reasoning_content": "private"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    monkeypatch.setattr(LocalLlamaRoles, "_request", respond)
    b = LocalLlamaRoles()
    assert b.invoke("verification", "protocol", {}, VerificationReport)["status"] == "unresolved"
    assert b.last_call["usage"]["completion_tokens"] == 20
    assert "private" not in json.dumps(b.last_call)


def test_local_specialist_total_timeout_and_external_url_rejection(monkeypatch):
    with pytest.raises(ValueError, match="loopback"):
        LocalLlamaRoles("http://example.com")
    async def block(self, body, timeout):
        await asyncio.sleep(10)
    monkeypatch.setattr(LocalLlamaRoles, "_request", block)
    b = LocalLlamaRoles(timeout_seconds=.02)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        b.invoke("verification", "protocol", {}, VerificationReport)
    assert time.monotonic() - start < .5
