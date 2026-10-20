import json
import httpx
import pytest

from app.expert_config import CompatibleExpert, ExpertEndpoint
from app.models import VerificationReport


def configuration(**kwargs):
    return ExpertEndpoint(name="sf", provider="siliconflow", base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V4.1-Flash", api_key_env="SILICONFLOW_API_KEY", **kwargs)


def test_siliconflow_requires_exact_authorized_endpoint():
    for url in ["https://evil.example/v1", "https://api.siliconflow.cn.evil.example/v1", "http://api.siliconflow.cn/v1"]:
        with pytest.raises(ValueError, match="explicitly authorized"):
            ExpertEndpoint(name="bad", provider="siliconflow", base_url=url, model="general", api_key_env="KEY")


def test_cloud_request_has_single_v1_and_provider_native_thinking_switch(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-secret")
    def respond(request):
        assert str(request.url) == "https://api.siliconflow.cn/v1/chat/completions"
        body = json.loads(request.content)
        assert body["enable_thinking"] is False and "chat_template_kwargs" not in body
        assert body["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
        assert request.headers["Authorization"] == "Bearer fake-secret"
        return httpx.Response(200, json={"model": body["model"], "choices": [{"message": {
            "content": '{"status":"unresolved","evidence_ids":[],"missing_evidence":["mass"]}',
            "reasoning_content": "private"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    original = httpx.AsyncClient
    monkeypatch.setattr("app.expert_config.httpx.AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    b = CompatibleExpert(configuration())
    assert b.invoke("verification", "prompt", {}, VerificationReport)["status"] == "unresolved"
    assert "fake-secret" not in json.dumps(b.last_call) and "private" not in json.dumps(b.last_call)


def test_provider_errors_are_recorded_and_credentials_redacted_without_retry(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-secret")
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(402, text="fake-secret echoed")
    original = httpx.AsyncClient
    monkeypatch.setattr("app.expert_config.httpx.AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    b = CompatibleExpert(configuration())
    with pytest.raises(RuntimeError, match="HTTP 402") as exc:
        b.invoke("verification", "prompt", {}, VerificationReport)
    assert len(calls) == 1 and b.last_call["http_status"] == 402
    assert "fake-secret" not in str(exc.value) + json.dumps(b.last_call)
