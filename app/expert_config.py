"""Configured general specialists are separate from the pinned OpenJev router."""
import json
import os
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, model_validator

from app.local_decision import LocalOpenJevBackend
from app.local_roles import LocalLlamaRoles
from app.models import Record


class ExpertEndpoint(Record):
    name: str = Field(min_length=1)
    base_url: str | None = None
    model: str | None = None
    revision: str | None = None
    api_key_env: str | None = None
    max_tokens: int = Field(default=1536, ge=64, le=524288)
    timeout_seconds: float = Field(default=45, gt=0, le=1800)
    reasoning_enabled: bool = False
    reasoning_effort: Literal["low", "medium", "max"] | None = None
    provider: Literal["local", "siliconflow", "openrouter"] = "local"

    @model_validator(mode="after")
    def authorized_endpoint(self):
        if self.reasoning_effort is not None and not (self.provider=="openrouter" and self.model in {"minimax/minimax-m3.1-flash", "minimax/minimax-m3.1-flash:free", "deepseek/deepseek-v4.1-flash"}):
            raise ValueError("Explicit reasoning effort is configured for the authorized benchmark expert endpoints only")
        if self.max_tokens>8192 and not (self.provider=="openrouter" and self.model in {"minimax/minimax-m3.1-flash", "minimax/minimax-m3.1-flash:free", "deepseek/deepseek-v4.1-flash"}):
            raise ValueError("Expanded completion allowance is authorized for the benchmark expert endpoints only")
        if self.base_url:
            if self.provider == "siliconflow":
                if self.base_url.rstrip("/") != "https://api.siliconflow.cn/v1":
                    raise ValueError("SiliconFlow requires the explicitly authorized https://api.siliconflow.cn/v1 endpoint")
                if not self.api_key_env:
                    raise ValueError("SiliconFlow requires an environment credential name")
            elif self.provider == "openrouter":
                if self.base_url.rstrip("/") != "https://openrouter.ai/api/v1" or self.api_key_env != "OPENROUTER_API_KEY":
                    raise ValueError("OpenRouter requires the authorized endpoint and OPENROUTER_API_KEY")
            else:
                LocalOpenJevBackend(self.base_url)  # validates loopback/root URL, never calls it
        if self.model and ((self.model.endswith(":free") and (self.provider,self.model) !=
                ("openrouter","minimax/minimax-m3.1-flash:free")) or "openjev" in self.model.lower()):
            raise ValueError("Formal specialists must be general models; free OpenRouter and OpenJev exploratory experts are excluded")
        return self

    @property
    def configured(self):
        return bool(self.base_url and self.model)


class ExpertPlan(Record):
    reference: ExpertEndpoint
    contrast: ExpertEndpoint

    @model_validator(mode="after")
    def distinct(self):
        if self.reference.name == self.contrast.name:
            raise ValueError("Expert configuration names must differ")
        if self.reference.configured and self.contrast.configured and (
            self.reference.base_url, self.reference.model) == (self.contrast.base_url, self.contrast.model):
            raise ValueError("Two aliases of the same endpoint/model are not independent expert models")
        return self


def load_plan(path: Path) -> ExpertPlan:
    return ExpertPlan.model_validate_json(path.read_text())


class CompatibleExpert(LocalLlamaRoles):
    """Configured local or explicitly authorized SiliconFlow/OpenRouter expert transport."""
    def __init__(self, config: ExpertEndpoint):
        if not config.configured:
            raise ValueError(f"Expert {config.name} is not configured; supply a local endpoint and model ID")
        self.timeout_seconds, self.max_tokens = config.timeout_seconds, config.max_tokens
        self.deadline = float("inf")
        self.last_call = None
        self.config = config
        self.model_id = config.model
        self.backend_id = f"compatible/{config.name}/{config.model}@{config.revision or 'revision-unavailable'}"
        self._key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if config.api_key_env and not self._key:
            raise ValueError(f"Credential environment variable {config.api_key_env} is missing")

    def manifest(self):
        return {"backend_id": self.backend_id, "configuration": self.config.model_dump(),
            "identity_status": "User-declared model/revision; actual response model retained, alias equivalence not verified",
            "temperature": 0, "output_mode": "JSON object + common Pydantic validation",
            "transport_retries": 0, "purpose": "general specialist", "is_openjev_router": False}

    def prepare_request(self, body):
        if self.config.provider == "siliconflow":
            body.pop("chat_template_kwargs", None)
            body["enable_thinking"] = False
        elif self.config.provider == "openrouter":
            body.pop("chat_template_kwargs", None)
            body["reasoning"] = ({"exclude": True} if self.config.model in {"minimax/minimax-m3.1-flash", "minimax/minimax-m3.1-flash:free"} or self.config.reasoning_enabled or self.config.reasoning_effort is not None
                                 else {"enabled": False})
            if self.config.reasoning_effort is not None:
                body["reasoning"]["effort"] = self.config.reasoning_effort
        return body

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]") if self._key else value
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        return value

    async def _request(self, body, timeout):
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        suffix = "/chat/completions" if self.config.provider in {"siliconflow", "openrouter"} else "/v1/chat/completions"
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(self.config.base_url.rstrip("/") + suffix, json=body, headers=headers)
            if response.is_error:
                self.last_call.update(http_status=response.status_code, response=self.redact(response.text))
                raise RuntimeError(f"Expert HTTP {response.status_code}: {self.redact(response.text)[:1200]}")
            raw = response.json()
        return self.redact(raw)
