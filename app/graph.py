"""Explicit sequential LangGraph; hidden evaluator state is never a state channel."""
import time
import math
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.decision_backend import RulesBackend, select_route, summarize
from app.lab_adapter import GateRejected, LabAdapter
from app.models import ActionProposal, HypothesisSet, LabState, Limits, RouteDecision, VerificationReport
from app.roles import PROMPTS, ScriptedRoles, projection
from app.trace import Trace


class EvidenceReferenceError(ValueError):
    pass


def validate_role_contract(role, record, payload):
    """Check public capability and handoff contracts before passing records onward."""
    if role == "diagnostic" and record.intent:
        intent = record.intent
        capability = payload["capabilities"].get(intent.capability_id)
        if capability is None:
            raise ValueError("Intent must use a listed capability")
        if set(intent.arguments) != set(capability["arguments"]):
            raise ValueError(f"{intent.capability_id} arguments must have exactly these keys: {list(capability['arguments'])}")
        if intent.capability_id == "temperature.set":
            value = intent.arguments["temperature_k"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or abs(value - payload["target"]["temperature_k"]) > 1e-9):
                raise ValueError("temperature_k must equal the supplied public target.temperature_k in Kelvin")
    elif role == "procedure":
        for field in ["sample_id", "expected_revision", "request_id"]:
            if getattr(record, field) != payload[field]:
                raise ValueError(f"Copy the supplied {field} exactly")
        if record.capability_id != payload["intent"]["capability_id"] or record.arguments != payload["intent"]["arguments"]:
            raise ValueError("Proposal capability and arguments must match the validated diagnostic intent exactly")


def validate_evidence(role, record, payload):
    if role not in {"diagnostic", "verification"}:
        return
    observations = payload.get("observations", {})
    known = {o["id"] for key in ["inspection", "measurement"] if (o := observations.get(key))}
    for h in payload.get("previous_hypotheses", payload.get("hypotheses", [])):
        known.update(h["supporting_ids"] + h["contradicting_ids"])
    cited = (record.evidence_ids if role == "verification" else
             [e for h in record.hypotheses for e in h.supporting_ids + h.contradicting_ids])
    if set(cited) - known:
        raise EvidenceReferenceError("Cite only IDs present in the supplied observations/hypotheses; unknown IDs: "
                                     + ", ".join(sorted(set(cited) - known)))


def initial_state(lab: LabAdapter, episode_id: str, limits: Limits) -> LabState:
    return LabState(episode_id=episode_id, sample_id=lab.sample_id,
        observation_revision=lab.revision, observations={}, capabilities=lab.capabilities(),
        target=lab.target(), hypotheses=[], evidence_refs=[], intent=None, proposal=None,
        pending_operation=None, verification=None, route="diagnose", action_count=0,
        llm_call_count=0, role_call_count=0, deadline=time.monotonic() + limits.episode_seconds,
        terminal_reason=None, last_error=None)


def build_graph(lab: LabAdapter, trace: Trace, limits: Limits | None = None,
                roles=None, decision_backend=None, checkpointer=None, profile=None):
    limits = limits or Limits()
    roles = roles or ScriptedRoles()
    decision_backend = decision_backend or RulesBackend()

    def stop_reason(state, for_model=False, for_action=False):
        if state.get("terminal_reason"):
            return state["terminal_reason"]
        if time.monotonic() >= state["deadline"]:
            return "deadline"
        if state.get("pending_operation"):
            return "pending_operation_unresolved"  # selected native capabilities are synchronous
        if for_action and state["action_count"] >= limits.lab_actions:
            return "action_budget"
        if for_model and roles.is_llm and state["llm_call_count"] >= limits.llm_calls:
            return "llm_budget"
        return None

    def observe(state):
        observations = lab.observe()
        trace.emit("observe", "observation", observations=observations)
        return {"observations": observations, "observation_revision": lab.revision,
                "action_count": lab.action_count}

    def router(state):
        reason = stop_reason(state)
        if reason:
            return {"route": "stop", "terminal_reason": reason}
        allowed = ["diagnose", "deliberate"]
        if state.get("intent"):
            allowed.append("prepare_action")
        if state["observations"].get("measurement"):
            allowed.append("verify")
        summary = summarize(state)
        trace.emit("router", "routing_start", backend_id=decision_backend.backend_id)
        if hasattr(decision_backend, "deadline"):
            decision_backend.deadline = state["deadline"]
        decision, error = select_route(decision_backend, summary, allowed)
        trace.emit("router", "routing", summary=summary, allowed_routes=allowed,
                   decision=decision.model_dump(), backend_metadata=getattr(decision_backend, "last_call", None),
                   error=error, error_code="E6" if error else None)
        return {"route": decision.choice}

    def call_role(role, state, schema, payload=None):
        reason = stop_reason(state, for_model=True)
        if reason:
            return None, {"terminal_reason": reason}
        project = profile.projection if profile else projection
        payload = payload if payload is not None else project(role, state, str(uuid4()))
        if role != "deliberation":
            from app.retrieval import retrieve
            payload = dict(payload, retrieval_evidence=profile.retrieve(role) if profile else retrieve(role))
        counts = {"role_call_count": state["role_call_count"], "llm_call_count": state["llm_call_count"]}
        attempts = 1 + (limits.format_repairs if roles.is_llm else 0)
        for attempt in range(attempts):
            reason = stop_reason(dict(state, **counts), for_model=True)
            if reason:
                return None, dict(counts, terminal_reason=reason)
            counts["role_call_count"] += 1
            counts["llm_call_count"] += int(roles.is_llm)
            start, raw = time.monotonic(), None
            trace.emit(role, "role_start", backend_id=roles.backend_id, is_llm=roles.is_llm,
                repair_attempt=attempt, llm_call_count=counts["llm_call_count"])
            if hasattr(roles, "deadline"):
                roles.deadline = state["deadline"]
            try:
                raw = roles.invoke(role, (profile.prompts if profile else PROMPTS)[role], payload, schema)
                value = schema.model_validate(raw)
                validate_evidence(role, value, payload)
                (profile.validate_role_contract if profile else validate_role_contract)(role, value, payload)
            except Exception as exc:
                repair = isinstance(exc, ValueError) and attempt + 1 < attempts
                metadata = getattr(roles, "last_call", None)
                trace.emit(role, "role_error", backend_id=roles.backend_id, input=payload,
                    raw_output=raw, error=f"{type(exc).__name__}: {exc}", repair_attempt=attempt,
                    will_repair=repair, llm_metadata=metadata,
                    error_code="E1" if isinstance(exc, EvidenceReferenceError) else "E4" if isinstance(exc, ValueError) else "E9")
                if repair:
                    payload = dict(payload, format_repair={"instruction": "Fix only the JSON/schema error; return the requested record.",
                        "validation_error": str(exc)[:2000], "previous_response": raw if raw is not None else (metadata or {}).get("response")})
                    continue
                return None, dict(counts, terminal_reason="role_error")
            trace.emit(role, "role_output", backend_id=roles.backend_id, is_llm=roles.is_llm,
                input=payload, output=value.model_dump(), llm_metadata=getattr(roles, "last_call", None),
                repair_attempt=attempt, elapsed_ms=(time.monotonic() - start) * 1000)
            if time.monotonic() >= state["deadline"]:
                return None, dict(counts, terminal_reason="deadline")
            return value, counts

    def diagnostic(state):
        value, update = call_role("diagnostic", state, HypothesisSet)
        if value:
            update.update(hypotheses=[h.model_dump() for h in value.hypotheses],
                intent=value.intent.model_dump() if value.intent else None,
                evidence_refs=list(dict.fromkeys(e for h in value.hypotheses for e in h.supporting_ids)),
                proposal=None, verification=None)
        return update

    def procedure(state):
        value, update = call_role("procedure", state, ActionProposal)
        if value:
            update["proposal"] = value.model_dump()
        return update

    def gate(state):
        reason = stop_reason(state, for_action=True)
        if reason:
            return {"gate_result": "stop", "terminal_reason": reason}
        try:
            proposal = ActionProposal.model_validate(state["proposal"])
            if hasattr(lab, 'action_cost') and lab.action_count + lab.action_cost(proposal) > limits.lab_actions:
                return {"gate_result": "stop", "terminal_reason": "action_budget"}
            lab.validate(proposal, state.get("pending_operation"))
        except (GateRejected, ValueError) as exc:
            code = getattr(exc, "code", "INVALID_SCHEMA")
            trace.emit("gate", "rejected", error=code, error_code="E5" if code == "STALE_PROPOSAL" else "E4")
            return {"gate_result": "reject", "proposal": None, "intent": None,
                    "last_error": {"code": code}}
        trace.emit("gate", "accepted", proposal=proposal.model_dump())
        return {"gate_result": "execute"}

    def execute(state):
        # Recheck at execution time: graph routing and elapsed time cannot authorize a tool.
        reason = stop_reason(state, for_action=True)
        if reason:
            return {"terminal_reason": reason}
        proposal = ActionProposal.model_validate(state["proposal"])
        try:
            event = lab.execute(proposal)
        except Exception as exc:
            trace.emit("execute", "execution_error", error=f"{type(exc).__name__}: {exc}", error_code="E9")
            return {"terminal_reason": "execution_uncertain", "proposal": None, "intent": None,
                    "action_count": lab.action_count}
        trace.emit("execute", "tool_result", **event)
        error = event["result"].get("error")
        return {"intent": None, "proposal": None, "verification": None,
                "action_count": lab.action_count, "last_error": error}

    def verification(state):
        value, update = call_role("verification", state, VerificationReport)
        if value:
            update["verification"] = value.model_dump()
        return update

    def termination(state):
        reason = stop_reason(state)
        report = state.get("verification")
        measurement = state["observations"].get("measurement")
        # A reliable measurement contradicting recovery can lead to another diagnosis.
        # Unavailable evidence supports bounded abstention; no looping on an offline sensor.
        if (not reason and report and report["status"] != "supported" and measurement
                and measurement["quality"] == "valid"):
            reason = stop_reason(state, for_model=True, for_action=True)
            if not reason:
                return {"stop_result": "continue"}
        if not reason and report:
            reason = {"supported": "agent_supported", "unresolved": "agent_unresolved",
                      "contradicted": "agent_contradicted"}[report["status"]]
        return {"stop_result": "stop", "terminal_reason": reason or "no_verification"}

    def deliberation(state):
        allowed = ["diagnose"]
        if state.get("intent"):
            allowed.append("prepare_action")
        if state["observations"].get("measurement"):
            allowed.append("verify")
        value, update = call_role("deliberation", state, RouteDecision,
            {"summary": summarize(state), "allowed_routes": allowed})
        if value and value.choice in allowed:
            update["route"] = value.choice
        else:
            update.update(route="stop", terminal_reason=update.get("terminal_reason", "invalid_deliberation"))
        return update

    def report(state):
        value = {"episode_id": state["episode_id"], "sample_id": state["sample_id"],
            "status": (state.get("verification") or {}).get("status", "unresolved"),
            "evidence_ids": (state.get("verification") or {}).get("evidence_ids", []),
            "hypotheses": state.get("hypotheses", []), "terminal_reason": state["terminal_reason"],
            "action_count": state["action_count"], "llm_call_count": state["llm_call_count"],
            "role_call_count": state["role_call_count"]}
        trace.emit("report", "agent_report", report=value)
        return {"report": value}

    graph = StateGraph(LabState)
    nodes = {"observe": observe, "router": router, "diagnostic": diagnostic, "procedure": procedure,
        "gate": gate, "execute": execute, "verification": verification, "termination": termination,
        "deliberation": deliberation, "report": report}
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "router")
    destinations = {"diagnose": "diagnostic", "prepare_action": "procedure",
        "verify": "verification", "deliberate": "deliberation", "stop": "report"}
    graph.add_conditional_edges("router", lambda s: s["route"], destinations)
    graph.add_edge("diagnostic", "router")
    graph.add_edge("procedure", "gate")
    graph.add_conditional_edges("gate", lambda s: s["gate_result"],
        {"execute": "execute", "reject": "router", "stop": "report"})
    graph.add_edge("execute", "observe")
    graph.add_conditional_edges("deliberation", lambda s: s["route"],
        {k: v for k, v in destinations.items() if k != "deliberate"})
    graph.add_edge("verification", "termination")
    graph.add_conditional_edges("termination", lambda s: s["stop_result"], {"stop": "report", "continue": "router"})
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
