"""Exploratory full-JSON specialist calls to the local model, distinct from one-token routing."""
import asyncio
from hashlib import sha256
import json
import time

import httpx

from app.local_decision import LocalOpenJevBackend


class ExpertResponseError(RuntimeError):
    def __init__(self,code,message):
        self.code=code
        super().__init__(message)


def completion_content(response):
    if response.get('error'):
        raise ExpertResponseError('expert_provider_error','Provider returned an error envelope')
    choices=response.get('choices') or []
    if not choices:
        raise ExpertResponseError('expert_invalid_envelope','Provider returned no completion choices')
    choice=choices[0]
    if choice.get('finish_reason')=='length':
        raise ExpertResponseError('expert_output_truncated','Completion reached its output-token limit; not a JSON formatting error')
    content=(choice.get('message') or {}).get('content')
    if not isinstance(content,str) or not content.strip():
        raise ExpertResponseError('expert_empty_output','Provider returned no nonempty text completion')
    return content


class LocalLlamaRoles:
    is_llm = True
    backend_id = "local/OpenJev-Flash-9B-Q4_K_M-specialists-exploratory-v1"

    def __init__(self, base_url="http://127.0.0.1:8792", timeout_seconds=45, max_tokens=1024):
        self.endpoint = LocalOpenJevBackend(base_url)  # same loopback-only endpoint validation
        self.timeout_seconds, self.max_tokens = timeout_seconds, max_tokens
        self.model_id = "openjev-flash-9b-q4"
        self.deadline = float("inf")
        self.last_call = None

    def manifest(self):
        return {"backend_id": self.backend_id, "model": self.endpoint.manifest(),
            "temperature": 0, "max_tokens": self.max_tokens, "timeout_seconds": self.timeout_seconds,
            "output_mode": "JSON object + common Pydantic validation", "transport_retries": 0,
            "limitations": "Decision-tuned model used experimentally as a generative specialist; not a second independent LLM"}

    async def _request(self, body, timeout):
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(self.endpoint.base_url + "/v1/chat/completions", json=body)
            response.raise_for_status()
            return response.json()

    def prepare_request(self, body):
        return body

    def invoke(self, role, prompt, payload, schema):
        system = prompt + "\nReturn one JSON object matching this schema exactly. No Markdown or extra fields.\n" + json.dumps(schema.model_json_schema(), sort_keys=True)
        body = {"model": self.model_id, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}],
            "max_tokens": self.max_tokens, "temperature": 0, "stream": False,
            "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}}
        body = self.prepare_request(body)
        self.last_call = {"model_id": self.model_id, "role": role, "request": body,
            "prompt_hash": sha256(system.encode()).hexdigest(), "schema_hash": sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode()).hexdigest(),
            "request_sent": False, "response": None, "usage": None}
        timeout = min(self.timeout_seconds, self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Episode deadline expired before local specialist call")
        start = time.monotonic()
        async def bounded():
            return await asyncio.wait_for(self._request(body, timeout), timeout)
        try:
            self.last_call["request_sent"] = True
            response = asyncio.run(bounded())
            # Do not persist separate private reasoning fields from any completion API.
            response = {**response, "choices": [{**c, "message": {k: v for k, v in (c.get("message") or {}).items()
                if k not in {"reasoning", "reasoning_content", "reasoning_details"}}} for c in response.get("choices", [])]}
            self.last_call.update(response=response, usage=response.get("usage"))
            try:
                content=completion_content(response)
            except ExpertResponseError as exc:
                self.last_call['response_classification']=exc.code
                raise
            return schema.model_validate_json(content).model_dump()
        finally:
            self.last_call["elapsed_ms"] = (time.monotonic() - start) * 1000
