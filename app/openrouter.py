"""Metered OpenRouter role backend with an enforced free-only model boundary."""
import argparse
import asyncio
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
import time
from urllib.request import urlopen

import httpx
from pydantic import BaseModel

API = "https://openrouter.ai/api/v1"


def model_catalog() -> list[dict]:
    with urlopen(f"{API}/models", timeout=20) as response:
        return json.load(response)["data"]


def is_free(model: dict) -> bool:
    try:
        p = model["pricing"]
        prices = [Decimal(p["prompt"]), Decimal(p["completion"]), Decimal(p.get("request", "0"))]
        return model["id"].endswith(":free") and all(v.is_finite() and v == 0 for v in prices)
    except (KeyError, InvalidOperation, TypeError):
        return False


class OpenRouterRoles:
    is_llm = True
    _next_request_at = 0.0

    def __init__(self, model_id: str, catalog: list[dict] | None = None,
                 timeout_seconds: float = 30, max_tokens: int = 1536):
        # Validate ID before any network request or reading credentials.
        if not model_id.endswith(":free"):
            raise ValueError("Only an explicit :free model ID is allowed")
        self.model = next((m for m in (catalog if catalog is not None else model_catalog())
                           if m["id"] == model_id), None)
        if not self.model or not is_free(self.model):
            raise ValueError("Requested model is absent or not zero-priced in the current catalog")
        self._api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is not exported in this process")
        self.model_id = model_id
        self.backend_id = f"openrouter/{model_id}"
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.deadline = float("inf")
        self.last_call: dict | None = None

    def manifest(self) -> dict:
        return {"backend_id": self.backend_id, "model_id": self.model_id,
            "usage_policy": "debug-only; not a formal evaluation backend",
            "canonical_slug": self.model.get("canonical_slug"), "catalog_pricing": self.model["pricing"],
            "catalog_checked_at_unix": time.time(), "max_tokens": self.max_tokens,
            "temperature": 0 if "temperature" in self.model.get("supported_parameters", []) else None,
            "output_mode": "common JSON + shared Pydantic validation",
            "timeout_seconds": self.timeout_seconds, "transport_retries": 0,
            "provider_max_price": {"prompt": 0, "completion": 0, "request": 0}}

    def _redact(self, value):
        # Defense against an upstream error echoing an auth token. Headers are never logged.
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()
                    if k not in {"reasoning", "reasoning_details"}}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        return value

    async def _completion(self, body: dict, timeout: float) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(f"{API}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                         "X-OpenRouter-Title": "Decision Routed Lab development test"})

    async def _bounded_completion(self, body: dict, timeout: float) -> httpx.Response:
        # Whole-request cancellation also bounds a response kept open by heartbeat bytes.
        return await asyncio.wait_for(self._completion(body, timeout), timeout=timeout)

    def invoke(self, role: str, prompt: str, payload: dict, schema: type[BaseModel]) -> dict:
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        body = {"model": self.model_id, "stream": False, "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": prompt + "\nReturn one JSON object matching this schema exactly. No Markdown or extra fields.\n" + schema_json},
                {"role": "user", "content": json.dumps(payload, allow_nan=False)},
            ],
            "provider": {"max_price": {"prompt": 0, "completion": 0, "request": 0}}}
        supported = self.model.get("supported_parameters", [])
        if "response_format" in supported:
            body["response_format"] = {"type": "json_object"}
        if "temperature" in supported:
            body["temperature"] = 0
        if "reasoning" in supported and not (self.model.get("reasoning") or {}).get("mandatory"):
            body["reasoning"] = {"enabled": False}
        wait = max(0, OpenRouterRoles._next_request_at - time.monotonic())
        self.last_call = {"model_id": self.model_id, "role": role,
            "schema_hash": sha256(schema_json.encode()).hexdigest(),
            "prompt_hash": sha256(body["messages"][0]["content"].encode()).hexdigest(),
            "request": body, "usage": None, "response": None, "request_sent": False}
        if time.monotonic() + wait >= self.deadline:
            raise TimeoutError("Episode deadline would expire during free-tier pacing")
        if wait:
            time.sleep(wait)
        OpenRouterRoles._next_request_at = time.monotonic() + 3.2
        timeout = min(self.timeout_seconds, self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Episode deadline expired")
        started = time.monotonic()
        self.last_call["request_sent"] = True
        try:
            response = asyncio.run(self._bounded_completion(body, timeout))
            response.raise_for_status()
            result = self._redact(response.json())
            self.last_call["response"] = result
            self.last_call["usage"] = result.get("usage")
            if result.get("error"):
                raise RuntimeError(json.dumps(result["error"]))
            content = result["choices"][0]["message"]["content"]
            return schema.model_validate_json(content).model_dump()
        except httpx.HTTPStatusError as exc:
            detail = self._redact(exc.response.text)
            code = exc.response.status_code
            self.last_call.update(http_status=code, response=detail)
            raise RuntimeError(f"OpenRouter HTTP {code}: {detail[:1200]}") from None
        finally:
            self.last_call["elapsed_ms"] = (time.monotonic() - started) * 1000


def main():
    parser = argparse.ArgumentParser(description="List currently zero-priced explicit OpenRouter model IDs")
    parser.parse_args()
    for model in model_catalog():
        if is_free(model):
            print(model["id"])


if __name__ == "__main__":
    main()
