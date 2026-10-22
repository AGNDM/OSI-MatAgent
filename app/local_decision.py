"""OpenJev Flash 9B GGUF on llama.cpp: genuine one-token option probabilities.

This is openjev/OpenJev-Flash-9B, NOT ejhshen/OpenJev-4B. Readout is a
conditional canonical-letter distribution from T=1 logits, without laboratory calibration.
"""
import asyncio
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx

from app.models import RouteDecision

MODEL_REVISION = "a3acd96966b06c2f161209f888b66235029f9ac1"
MODEL_SHA = "6209cf51aec005b00cb80fdfef1b961adde57d5a8f181a377bd3882a4668b91d"
CRITERIA = {
    "diagnose": "Inspect evidence and identify the next check or correction. Use when no actionable intent or fresh measurement exists, or verification needs another diagnosis.",
    "prepare_action": "Convert the existing diagnostic intent into a validated action proposal. Use when has_intent is true.",
    "verify": "Assess the available fresh measurement against the requested outcome. Use when a fresh measurement exists and no intent is waiting.",
    "wait_short": "Directly advance the supported native process by the offered short duration, then read fresh state. No specialist is needed just to wait.",
    "wait_long": "Directly advance the supported native process by the offered long duration, then read fresh state. No specialist is needed just to wait.",
    "deliberate": "Escalate ambiguous or conflicting routing information when choosing a specialist needs deliberation.",
}


def option_prompt(summary: dict, allowed: list[str]) -> tuple[str, dict[str, str]]:
    if not 1 <= len(allowed) <= 4 or len(set(allowed)) != len(allowed) or set(allowed) - set(CRITERIA):
        raise ValueError("Invalid allowed route set")
    options = {chr(65 + i): route for i, route in enumerate(allowed)}
    lines = [f"[{letter}] {route}: {CRITERIA[route]}" for letter, route in options.items()]
    instruction = ""
    if 'instrument_context' in summary:
        rich = dict(CRITERIA,
            diagnose='Inspect or plan when no current selected intent exists, or a new error/stale selection requires revision.',
            prepare_action='Convert the already selected intent into a proposal; use when has_intent is true.',
            verify='Verify the FINAL objective when its public goal checks hold with fresh evidence and no pending operation; not for routine intermediate progress.')
        lines = [f"[{letter}] {route}: {rich[route]}" for letter, route in options.items()]
        instruction = ("Choose a routing or bounded wait action. Use the objective, predicate checks, current native state and recent actions. "
            "Command acceptance is not process completion. If an intent is ready, prepare_action. "
            "If all public goal conditions hold with fresh evidence and no pending operation, verify. "
            "For a running process whose goal is not yet met and with no error, choose a wait duration from the offered menu. "
            "If configuration, starting, correction or investigation is required, diagnose. "
            "A fresh intermediate read alone is not a reason to verify the final goal. "
            "Use only offered options; no host truth or reference action script is provided.\n")
        context = summary['instrument_context']
        if context.get('selected_intent_current') and not context.get('last_error'):
            instruction = ("A concrete operation has ALREADY been selected for the current revision; choose prepare_action. "
                "Read selected_intent for its exact capability and arguments. Unmet goal checks describe what this operation "
                "is intended to change; they are not a reason to repeat diagnosis. Replan only for a new error or stale selection.\n" + instruction)
        lines = [line + (" duration_s=" + str(summary['instrument_context']['wait_offers'][route])
                 if route in summary['instrument_context']['wait_offers'] else "")
                 for line, route in zip(lines, allowed)]
    question = ("Which routing or bounded waiting action should happen next?" if instruction else
                "Which specialist should handle the next step of this laboratory workflow?")
    text = (instruction + "State:\n" + json.dumps(summary, sort_keys=True, ensure_ascii=False) +
            "\n\nQuestion: " + question + "\nOptions:\n" +
            "\n".join(lines) + "\n\nAnswer with the letter of the best option only.")
    return text, options


def read_probabilities(response: dict, options: dict[str, str], *, details: dict | None = None) -> dict[str, float]:
    positions = response.get("probs", response.get("completion_probabilities", []))
    if len(positions) != 1:
        raise ValueError("Expected exactly one token probability record")
    logarithmic = "top_logprobs" in positions[0]
    entries = positions[0].get("top_logprobs" if logarithmic else "top_probs")
    if not entries:
        raise ValueError("Missing token probabilities")
    weights = {}
    for entry in entries:
        if logarithmic:
            logp = float(entry["logprob"])
            if math.isnan(logp) or logp > 0:
                raise ValueError("Invalid token log-probability")
            p = math.exp(logp)
        else:
            p = float(entry["prob"])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid option probability")
        token = bytes(entry["bytes"]).decode("utf-8", errors="replace") if entry.get("bytes") is not None else entry["token"]
        if token not in options:
            continue
        weights[token] = weights.get(token, 0.0) + p
    if set(weights) != set(options):
        raise ValueError("Incomplete option readout; do not fabricate missing probabilities")
    mass = sum(weights.values())
    if not 0 < mass <= 1.0001:
        raise ValueError("Invalid total option probability mass")
    if details is not None:
        details.update(option_probability_mass=mass, readout_source="logprobs" if logarithmic else "probabilities")
    return {route: weights[letter] / mass for letter, route in options.items()}


class LocalOpenJevBackend:
    backend_id = f"openjev/OpenJev-Flash-9B-GGUF@{MODEL_REVISION}:Q4_K_M"
    is_decision_model = True

    def __init__(self, base_url="http://127.0.0.1:8792", timeout_seconds=45):
        parsed = urlparse(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("Local OpenJev endpoint must be loopback HTTP")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.deadline = float("inf")
        self.last_call = None

    def manifest(self):
        root = Path.home() / ".local/share/osi-hack"
        model_path = root / "models/openjev-flash-9b-q4/deployment-model.json"
        runtime_path = root / "llama-b11147/deployment-runtime.json"
        server_arguments = None
        try:
            pid = int((root / "openjev-server.pid").read_text())
            server_arguments = Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
        except (OSError, ValueError):
            pass
        return {"backend_id": self.backend_id, "base_url": self.base_url,
            "model_sha256": MODEL_SHA, "llama_cpp": "b11147", "quantization": "Q4_K_M",
            "probability_semantics": "Full-vocabulary T=1 log-probabilities renormalized over allowed canonical letter tokens; argmax route; not calibrated on lab tasks",
            "source": "https://huggingface.co/openjev/OpenJev-Flash-9B-GGUF",
            "model_artifact": json.loads(model_path.read_text()) if model_path.exists() else None,
            "runtime_artifact": json.loads(runtime_path.read_text()) if runtime_path.exists() else None,
            "server_arguments": server_arguments, "request_timeout_seconds": self.timeout_seconds}

    async def _infer(self, summary, allowed, timeout):
        text, options = option_prompt(summary, allowed)
        probabilities = await self._infer_options(text, options, timeout)
        return RouteDecision(backend_id=self.backend_id,
            choice=max(probabilities, key=probabilities.get), probabilities=probabilities)

    async def _infer_options(self, text, options, timeout):
        """Shared one-token inference; caller owns the question and answer semantics."""
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            templated = await client.post(self.base_url + "/apply-template", json={
                "messages": [{"role": "user", "content": text}],
                "chat_template_kwargs": {"enable_thinking": False}})
            templated.raise_for_status()
            prompt = templated.json()["prompt"]
            request = {"prompt": prompt, "n_predict": 1, "n_probs": 64,
                "post_sampling_probs": False, "temperature": -1.0, "samplers": ["temperature"],
                "grammar": 'root ::= [' + "".join(options) + ']',
                "seed": 0, "cache_prompt": True, "stream": False}
            self.last_call.update(prompt=text, formatted_prompt=prompt, options=options, request=request)
            response = await client.post(self.base_url + "/completion", json=request)
            response.raise_for_status()
            raw = response.json()
            self.last_call["response"] = raw
            return read_probabilities(raw, options, details=self.last_call)

    async def _bounded(self, summary, allowed, timeout):
        return await asyncio.wait_for(self._infer(summary, allowed, timeout), timeout)

    def decide(self, summary: dict, allowed_routes: list[str]) -> RouteDecision:
        start = time.monotonic()
        timeout = min(self.timeout_seconds, self.deadline - start)
        self.last_call = {"backend_id": self.backend_id, "summary": summary,
            "allowed_routes": list(allowed_routes), "probability_semantics": self.manifest()["probability_semantics"]}
        if timeout <= 0:
            raise TimeoutError("Episode deadline expired before decision inference")
        try:
            result = asyncio.run(self._bounded(summary, allowed_routes, timeout))
            result.latency_ms = (time.monotonic() - start) * 1000
            return result
        except Exception as exc:
            self.last_call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.last_call["elapsed_ms"] = (time.monotonic() - start) * 1000
