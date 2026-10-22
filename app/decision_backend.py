"""Typed routing boundary. Rules are a published baseline, not a decision model."""
import math
import time
from typing import Protocol

from app.models import RouteDecision


class DecisionBackend(Protocol):
    backend_id: str

    def decide(self, summary: dict, allowed_routes: list[str]) -> RouteDecision: ...


def summarize(state: dict) -> dict:
    obs = state["observations"]
    measurement = obs.get("measurement")
    return {"stage": "prepare" if state.get("intent") else "verify" if measurement else "diagnose",
        "has_intent": bool(state.get("intent")), "has_proposal": bool(state.get("proposal")),
        "has_inspection": bool(obs.get("inspection")), "has_measurement": bool(measurement),
        "measurement_quality": measurement["quality"] if measurement else "unknown",
        "measurement_fresh": bool(measurement and measurement["revision"] == obs["revision"]
            and measurement["captured_at_step"] > obs["last_mutation_step"]),
        "pending_operation": bool(state.get("pending_operation")),
        "verifier_disagreement": bool(state.get("verification") and state["verification"]["status"] != "supported"),
        "actions_used": state["action_count"], "llm_calls_used": state["llm_call_count"]}


class RulesBackend:
    backend_id = "rules-v1"

    def decide(self, summary: dict, allowed_routes: list[str]) -> RouteDecision:
        route = ("prepare_action" if summary["has_intent"] else
                 "diagnose" if summary["verifier_disagreement"] else
                 "verify" if summary["has_measurement"] else "diagnose")
        if route not in allowed_routes:
            route = next(r for r in ("diagnose", "verify", "prepare_action") if r in allowed_routes)
        return RouteDecision(backend_id=self.backend_id, choice=route)


def select_route(backend: DecisionBackend, summary: dict, allowed: list[str], *, policy="argmax-v1") -> tuple[RouteDecision, str | None]:
    if policy not in {"argmax-v1", "legacy-threshold-v1"}:
        raise ValueError("Unknown routing policy")
    start = time.monotonic()
    error = None
    try:
        decision = RouteDecision.model_validate(backend.decide(summary, allowed))
        if decision.choice not in allowed:
            raise ValueError("route not in allowed set")
        p = decision.probabilities
        if p is not None:
            if (set(p) != set(allowed) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values())
                    or (decision.backend_id not in {"openrouter-decisions/typesafe/jev-1.14", "openrouter-decisions/openai/gpt-6-luna-decisions"}
                        and abs(sum(p.values()) - 1) > 1e-5)):
                raise ValueError("invalid route probabilities")
            ordered = sorted(p, key=p.get, reverse=True)
            if p[decision.choice] != max(p.values()):
                raise ValueError("choice disagrees with probability maximum")
            if policy == "legacy-threshold-v1":
                values = sorted(p.values(), reverse=True)
                if "deliberate" in allowed and (values[0] < .6 or (len(values)>1 and values[0]-values[1]<.15)):
                    decision = decision.model_copy(update={"choice":"deliberate"})
    except Exception as exc:
        # The logged fallback has no synthetic probabilities and a truthful backend ID.
        error = f"{type(exc).__name__}: {exc}"
        decision = (RulesBackend().decide(summary, allowed).model_copy(update={"fallback_used": True})
                    if policy == "legacy-threshold-v1" else RouteDecision(backend_id=backend.backend_id,choice="stop"))
    decision.latency_ms = (time.monotonic() - start) * 1000
    return decision, error
