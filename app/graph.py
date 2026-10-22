"""Explicit sequential LangGraph; hidden evaluator state is never a state channel."""
import time
import math
from copy import deepcopy
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.decision_backend import RulesBackend, select_route, summarize
from app.lab_adapter import GateRejected, LabAdapter
from app.models import ActionProposal, HypothesisSet, LabState, Limits, RouteDecision, VerificationReport, OperationOptions
from app.operation_decision import CANDIDATE_INSTRUCTION, operation_payload, validate_decision
from app.roles import PROMPTS, ScriptedRoles, projection
from app.trace import Trace
from app.readback_contract import normalize_record


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
    if isinstance(record, OperationOptions):
        cited += [e for c in record.candidates for e in c.evidence_ids]
    records=record.hypotheses if role=='diagnostic' else []
    materials=[e for h in records for e in h.material_evidence_ids]
    if isinstance(record,OperationOptions):
        materials += [e for c in record.candidates for e in c.material_evidence_ids]
    known_materials={d['id'] for d in payload.get('retrieval_evidence',[])}
    if set(materials)-known_materials:
        raise EvidenceReferenceError('Material citations must identify supplied retrieval passages')
    if set(cited) - known:
        raise EvidenceReferenceError("Cite only IDs present in the supplied observations/hypotheses; unknown IDs: "
                                     + ", ".join(sorted(set(cited) - known)))


def initial_state(lab: LabAdapter, episode_id: str, limits: Limits) -> LabState:
    return LabState(episode_id=episode_id, sample_id=lab.sample_id,
        observation_revision=lab.revision, observations={}, capabilities=lab.capabilities(),
        target=lab.target(), hypotheses=[], evidence_refs=[], intent=None, proposal=None,
        pending_operation=None, verification=None, verified_evidence_signatures=[],
        last_verification_feedback=None, route="diagnose", action_count=0,
        candidates=[], operation_decision=None, operation_revision=None, operation_call_count=0,
        llm_call_count=0, role_call_count=0, router_call_count=0, router_wait_count=0, router_wait_authorization=None, deadline=time.monotonic() + limits.episode_seconds,
        terminal_reason=None, last_error=None)


def build_graph(lab: LabAdapter, trace: Trace, limits: Limits | None = None,
                roles=None, decision_backend=None, checkpointer=None, profile=None, operation_backend=None):
    limits = limits or Limits()
    roles = roles or ScriptedRoles()
    decision_backend = decision_backend or RulesBackend()

    def stop_reason(state, for_model=False, for_action=False):
        if state.get("terminal_reason"):
            return state["terminal_reason"]
        if time.monotonic() >= state["deadline"]:
            return "deadline"
        if state.get("pending_operation") and not getattr(lab, "manages_pending_operations", False):
            return "pending_operation_unresolved"  # selected native capabilities are synchronous
        if for_action and state["action_count"] >= limits.lab_actions:
            return "action_budget"
        if for_model and roles.is_llm and state["llm_call_count"] >= limits.llm_calls:
            return "llm_budget"
        return None

    def observe(state):
        if hasattr(lab, "deadline") or getattr(lab, "manages_pending_operations", False):
            lab.deadline = state["deadline"]
        observations = lab.observe()
        trace.emit("observe", "observation", observations=observations)
        update = {"observations": observations, "observation_revision": lab.revision, "action_count": lab.action_count}
        if getattr(lab, "manages_pending_operations", False):
            update["pending_operation"] = deepcopy(lab.pending_operation)
            if lab.termination_reason:
                update["terminal_reason"] = lab.termination_reason
        return update

    def verification_signature(state):
        if hasattr(lab, 'verification_signature'):
            return lab.verification_signature(state)
        return None

    def verification_allowed(state):
        signature = verification_signature(state)
        return not signature or signature not in state.get('verified_evidence_signatures', [])

    def router(state):
        reason = stop_reason(state)
        if reason:
            return {"route": "stop", "terminal_reason": reason}
        count_router = (getattr(lab, "count_router_calls", False) and
                        (getattr(decision_backend, "is_decision_model", False) or getattr(decision_backend, "is_llm", False)))
        if count_router and state["llm_call_count"] >= limits.llm_calls:
            return {"route": "stop", "terminal_reason": "llm_budget"}
        allowed = ["diagnose"]
        if getattr(lab,"routing_policy","argmax-v1") == "legacy-threshold-v1":
            allowed.append("deliberate")
        if state.get("intent"):
            allowed.append("prepare_action")
        if state["observations"].get("measurement") and verification_allowed(state):
            allowed.append("verify")
        summary = summarize(state)
        if getattr(lab, "router_wait_enabled", False):
            summary["budget_remaining"] = {"model_calls": limits.llm_calls-state["llm_call_count"],
                "lab_actions": limits.lab_actions-lab.action_count,
                "wall_seconds": max(0., state["deadline"]-time.monotonic())}
        if hasattr(lab, "route_context"):
            summary, allowed = lab.route_context(state, summary, allowed)
        trace.emit("router", "routing_start", backend_id=decision_backend.backend_id)
        if hasattr(decision_backend, "deadline"):
            decision_backend.deadline = state["deadline"]
        decision, error = select_route(decision_backend, summary, allowed, policy=getattr(lab,"routing_policy","argmax-v1"))
        trace.emit("router", "routing", summary=summary, allowed_routes=allowed,
                   routing_policy=getattr(lab,"routing_policy","argmax-v1"),
                   decision=decision.model_dump(), backend_metadata=getattr(decision_backend, "last_call", None),
                   error=error, error_code="E6" if error else None)
        update = {"route": decision.choice, "router_wait_authorization": None}
        if count_router:
            update.update(llm_call_count=state["llm_call_count"]+1, router_call_count=state.get("router_call_count", 0)+1)
        if error and (getattr(lab,'routing_policy','argmax-v1') == 'argmax-v1' or not getattr(decision_backend, 'allow_rules_fallback', True)):
            return dict(update, route='stop', terminal_reason='token_budget_exceeded' if (error or '').startswith('TokenBudgetExceeded:') else 'decision_model_error')
        if decision.choice in {'wait_short', 'wait_long'}:
            if error or decision.fallback_used or not hasattr(lab, 'router_wait_offers'):
                return dict(update, route='stop', terminal_reason='invalid_router_wait')
            offers = lab.router_wait_offers(state)
            if (decision.choice not in offers or lab.revision != state['observation_revision']
                    or lab.sample_id != state['sample_id']):
                return dict(update, route='stop', terminal_reason='stale_router_wait')
            proposal = ActionProposal(sample_id=lab.sample_id, expected_revision=lab.revision,
                capability_id='workflow.await', arguments={'duration_s':offers[decision.choice]},
                request_id=str(uuid4()), postcondition='Advance the native clock and obtain fresh public process state.')
            update.update(proposal=proposal.model_dump(), intent={
                'capability_id':proposal.capability_id, 'arguments':proposal.arguments,
                'discriminating_observation':'Fresh native process state after bounded wait.',
                'postcondition':proposal.postcondition}, candidates=[], operation_decision=None,
                operation_revision=None, verification=None,
                router_wait_authorization={'decision':decision.model_dump(), 'proposal':proposal.model_dump(),
                    'revision':lab.revision, 'sample_id':lab.sample_id})
            trace.emit('router', 'direct_wait_proposed', decision=decision.model_dump(), proposal=proposal.model_dump())
        return update

    def call_role(role, state, schema, payload=None):
        reason = stop_reason(state, for_model=True)
        if reason:
            return None, {"terminal_reason": reason}
        project = profile.projection if profile else projection
        payload = payload if payload is not None else project(role, state, str(uuid4()))
        if role != "deliberation":
            from app.retrieval import retrieve
            if hasattr(profile,'retrieve_context'):
                retrieval=profile.retrieve_context(role,payload)
                payload=dict(payload,retrieval_evidence=retrieval['hits'],
                    retrieval_context={k:v for k,v in retrieval.items() if k!='hits'})
                trace.emit('retrieval','knowledge_lookup',role=role,**retrieval)
            else:
                payload = dict(payload, retrieval_evidence=profile.retrieve(role) if profile else retrieve(role))
        if role == "diagnostic" and operation_backend is not None:
            inspection = payload["observations"].get("inspection")
            payload = dict(payload, operation_context={
                "inspection_present": bool(inspection),
                "inspection_id": inspection["id"] if inspection else None,
                "inspection_current": bool(inspection and inspection["revision"] == state["observation_revision"]),
                "last_mutation_step": payload["observations"].get("last_mutation_step", 0),
                "current_aliquot_id": inspection.get("aliquot_id") if inspection else None,
                "current_thermal_run": inspection.get("current_thermal_run") if inspection else None,
                "current_scan_method": inspection.get("scan_method") if inspection else None,
                "instruction": "Use the CURRENT inspection above; do not restart the episode or assume it is absent."})
        counts = {"role_call_count": state["role_call_count"], "llm_call_count": state["llm_call_count"]}
        if role=='diagnostic':counts['retrieval_evidence']=deepcopy(payload.get('retrieval_evidence',[]))
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
                prompt = (profile.prompts if profile else PROMPTS)[role]
                if role == "diagnostic" and operation_backend is not None:
                    import json
                    prompt = (getattr(profile, "candidate_prompt", prompt) + CANDIDATE_INSTRUCTION
                              + "\nCURRENT PUBLIC STATE FOR THIS CALL (not the episode initial state):\n"
                              + json.dumps(payload["operation_context"], sort_keys=True, allow_nan=False))
                raw = roles.invoke(role, prompt, payload, schema)
                value = schema.model_validate(raw)
                value = normalize_record(value, payload.get("capabilities", {}))
                validate_evidence(role, value, payload)
                validator = profile.validate_role_contract if profile else validate_role_contract
                if isinstance(value, OperationOptions):
                    for candidate in value.candidates:
                        validator(role, HypothesisSet(hypotheses=value.hypotheses, intent=candidate.intent), payload)
                else:
                    validator(role, value, payload)
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
                return None, dict(counts, terminal_reason=getattr(exc,"code","role_error"))
            trace.emit(role, "role_output", backend_id=roles.backend_id, is_llm=roles.is_llm,
                input=payload, output=value.model_dump(), llm_metadata=getattr(roles, "last_call", None),
                repair_attempt=attempt, elapsed_ms=(time.monotonic() - start) * 1000)
            if time.monotonic() >= state["deadline"]:
                return None, dict(counts, terminal_reason="deadline")
            return value, counts

    def diagnostic(state):
        schema = OperationOptions if operation_backend is not None else HypothesisSet
        value, update = call_role("diagnostic", state, schema)
        if value:
            update.update(hypotheses=[h.model_dump() for h in value.hypotheses],
                evidence_refs=list(dict.fromkeys(e for h in value.hypotheses for e in h.supporting_ids)),
                proposal=None, verification=None, operation_decision=None, operation_revision=None)
            if operation_backend is not None:
                update.update(intent=None, candidates=[c.model_dump() for c in value.candidates])
            else:
                update.update(intent=value.intent.model_dump() if value.intent else None)
        return update

    def operation_decision(state):
        reason = stop_reason(state, for_action=True)
        if reason:
            return {"terminal_reason": reason}
        if operation_backend.is_llm and state["llm_call_count"] >= limits.llm_calls:
            return {"terminal_reason": "llm_budget"}
        counts = {"operation_call_count": state.get("operation_call_count", 0)+1,
                  "llm_call_count": state["llm_call_count"]+int(operation_backend.is_llm)}
        context, candidates = operation_payload(state), deepcopy(state["candidates"])
        if hasattr(lab, "operation_context"):
            context = lab.operation_context(context, candidates)
        trace.emit("operation_decision", "operation_start", backend_id=operation_backend.backend_id,
                   is_llm=operation_backend.is_llm)
        if hasattr(operation_backend, "deadline"):
            operation_backend.deadline = state["deadline"]
        try:
            if not candidates:
                raise ValueError("Operation selection requires diagnostic candidates")
            decision = validate_decision(operation_backend.choose(deepcopy(context), deepcopy(candidates)), candidates)
            if decision.backend_id != operation_backend.backend_id:
                raise ValueError("Operation backend identity mismatch")
            if time.monotonic() >= state["deadline"]:
                raise TimeoutError("Operation decision exceeded episode deadline")
            if lab.revision != state["observation_revision"] or lab.sample_id != state["sample_id"]:
                raise ValueError("Operation decision refers to stale world state")
            selected = next(c for c in candidates if c["candidate_id"] == decision.choice)
        except Exception as exc:
            trace.emit("operation_decision", "operation_error", input=context, candidates=candidates,
                error=f"{type(exc).__name__}: {exc}", error_code="E6",
                backend_metadata=getattr(operation_backend, "last_call", None),
                llm_metadata=(getattr(operation_backend, "last_call", None)
                    if (getattr(operation_backend, "last_call", None) or {}).get("request_sent") else None))
            return dict(counts, intent=None, proposal=None, operation_decision=None,
                        terminal_reason="deadline" if time.monotonic() >= state["deadline"] else getattr(exc,"code","operation_decision_error"))
        trace.emit("operation_decision", "operation_selected", input=context, candidates=candidates,
            decision=decision.model_dump(), selected=selected,
            backend_metadata=getattr(operation_backend, "last_call", None),
            llm_metadata=(getattr(operation_backend, "last_call", None)
                    if (getattr(operation_backend, "last_call", None) or {}).get("request_sent") else None))
        return dict(counts, intent=deepcopy(selected["intent"]), operation_decision=decision.model_dump(),
                    operation_revision=state["observation_revision"])

    def check_operation_selection(state):
        authorization = state.get('router_wait_authorization')
        if authorization:
            decision = RouteDecision.model_validate(authorization['decision'])
            offers = lab.router_wait_offers(state)
            proposal = ActionProposal.model_validate(authorization['proposal'])
            p = decision.probabilities
            if (decision.backend_id != decision_backend.backend_id or decision.fallback_used
                    or decision.choice not in offers or not p or decision.choice not in p
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values())
                    or (decision.backend_id != "openrouter-decisions/typesafe/jev-1.14"
                        and abs(sum(p.values())-1)>1e-5)
                    or p[decision.choice] != max(p.values())
                    or authorization['revision'] != lab.revision or authorization['sample_id'] != lab.sample_id
                    or proposal.sample_id != lab.sample_id or proposal.expected_revision != lab.revision
                    or proposal.capability_id != 'workflow.await'
                    or proposal.arguments != {'duration_s':offers[decision.choice]}
                    or state.get('proposal') != proposal.model_dump()
                    or state.get('intent',{}).get('capability_id') != 'workflow.await'
                    or state.get('intent',{}).get('arguments') != proposal.arguments):
                raise ValueError('Invalid or stale router wait authorization')
            return
        if operation_backend is None:
            return
        if state.get("operation_revision") != lab.revision or not state.get("operation_decision"):
            raise ValueError("A fresh operation decision is required before procedure/execution")
        decision = validate_decision(state["operation_decision"], state["candidates"])
        if decision.backend_id != operation_backend.backend_id:
            raise ValueError("Operation backend identity mismatch")
        selected = next(c for c in state["candidates"] if c["candidate_id"] == decision.choice)
        if state.get("intent") != selected["intent"]:
            raise ValueError("Intent differs from the selected candidate")

    def procedure(state):
        try:
            check_operation_selection(state)
        except ValueError as exc:
            trace.emit("procedure", "selection_rejected", error=str(exc), error_code="E5")
            return {"terminal_reason": "invalid_operation_handoff"}
        value, update = call_role("procedure", state, ActionProposal)
        if value:
            update["proposal"] = value.model_dump()
        return update

    def gate(state):
        reason = stop_reason(state, for_action=True)
        if reason:
            return {"gate_result": "stop", "terminal_reason": reason}
        try:
            check_operation_selection(state)
            proposal = ActionProposal.model_validate(state["proposal"])
            if operation_backend is not None and (proposal.capability_id != state["intent"]["capability_id"]
                    or proposal.arguments != state["intent"]["arguments"]):
                raise ValueError("Proposal must match the selected operation")
            if hasattr(lab, 'action_cost') and lab.action_count + lab.action_cost(proposal) > limits.lab_actions:
                return {"gate_result": "stop", "terminal_reason": "action_budget"}
            lab.validate(proposal, state.get("pending_operation"))
        except (GateRejected, ValueError) as exc:
            code = getattr(exc, "code", "INVALID_SCHEMA")
            trace.emit("gate", "rejected", error=code, error_code="E5" if code == "STALE_PROPOSAL" else "E4")
            return {"gate_result": "reject", "proposal": None, "intent": None,
                    "candidates": [], "operation_decision": None, "operation_revision": None,
                    "router_wait_authorization": None, "last_error": {"code": code}}
        trace.emit("gate", "accepted", proposal=proposal.model_dump())
        return {"gate_result": "execute"}

    def execute(state):
        # Recheck at execution time: graph routing and elapsed time cannot authorize a tool.
        reason = stop_reason(state, for_action=True)
        if reason:
            return {"terminal_reason": reason}
        proposal = ActionProposal.model_validate(state["proposal"])
        try:
            check_operation_selection(state)
            event = lab.execute(proposal)
        except Exception as exc:
            trace.emit("execute", "execution_error", error=f"{type(exc).__name__}: {exc}", error_code="E9")
            return {"terminal_reason": "execution_uncertain", "proposal": None, "intent": None,
                    "action_count": lab.action_count}
        trace.emit("execute", "tool_result", **event)
        error = event["result"].get("error")
        return {"intent": None, "proposal": None, "verification": None,
                "candidates": [], "operation_decision": None, "operation_revision": None,
                "action_count": lab.action_count, "last_error": error, "router_wait_authorization": None,
                "router_wait_count": state.get("router_wait_count", 0)+int(bool(state.get("router_wait_authorization")))}

    def verification(state):
        if not verification_allowed(state):
            return {"terminal_reason": "duplicate_verification_evidence"}
        signature = verification_signature(state)
        value, update = call_role("verification", state, VerificationReport)
        if value:
            update["verification"] = value.model_dump()
            if signature:
                update['verified_evidence_signatures'] = list(dict.fromkeys(
                    state.get('verified_evidence_signatures', []) + [signature]))
                measurement = state['observations'].get('measurement') or {}
                update['last_verification_feedback'] = dict(value.model_dump(),
                    evidence_signature=signature, observation_id=measurement.get('id'))
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
        if state["observations"].get("measurement") and verification_allowed(state):
            allowed.append("verify")
        summary = summarize(state)
        if hasattr(lab, 'route_context'):
            summary, _ = lab.route_context(state, summary, list(allowed))
        value, update = call_role("deliberation", state, RouteDecision,
            {"summary": summary, "allowed_routes": allowed})
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
            "role_call_count": state["role_call_count"],
            "operation_call_count": state.get("operation_call_count", 0),
            "operation_backend": operation_backend.backend_id if operation_backend is not None else None}
        if getattr(lab, "count_router_calls", False):
            value["router_call_count"] = state.get("router_call_count", 0)
            value["router_wait_count"] = state.get("router_wait_count", 0)
        trace.emit("report", "agent_report", report=value)
        return {"report": value}

    graph = StateGraph(LabState)
    nodes = {"observe": observe, "router": router, "diagnostic": diagnostic, "procedure": procedure,
        "gate": gate, "execute": execute, "verification": verification, "termination": termination,
        "deliberation": deliberation, "report": report, "operation_decision": operation_decision}
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "router")
    destinations = {"diagnose": "diagnostic", "prepare_action": "procedure",
        "verify": "verification", "deliberate": "deliberation", "stop": "report",
        "wait_short": "gate", "wait_long": "gate"}
    graph.add_conditional_edges("router", lambda s: s["route"], destinations)
    graph.add_conditional_edges("diagnostic",
        lambda s: "operation_decision" if operation_backend is not None and not s.get("terminal_reason") else "router",
        {"operation_decision": "operation_decision", "router": "router"})
    graph.add_edge("operation_decision", "router")
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
