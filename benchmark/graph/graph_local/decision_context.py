"""Deterministic public views for routing and choosing supplied operations.

This is an explicit decision-input protocol, not a lossless replacement for the
specialist context or trace. Old history is represented by public action counts
and hashes; recent, failed and candidate-related receipts are retained. Oversize
views fail before inference rather than cutting text or changing offered actions.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json


PROTOCOL = "decision-view-v1"
DEFAULT_MAX_BYTES = 16000
PURPOSES = {"routing", "operational_action_selection"}
GRAPH_FIELDS = (
    "stage", "has_intent", "has_proposal", "has_inspection", "has_measurement",
    "measurement_quality", "measurement_fresh", "pending_operation",
    "verifier_disagreement", "actions_used", "llm_calls_used", "budget_remaining",
)
PRIVATE_FIELDS = {
    "evaluator", "private", "oracle", "answer_key", "correct_option_ids",
    "checkpoints", "checkpoint_id", "grader_reason", "proposal_correct",
    "future_faults", "scheduled_events", "reference_action", "reference_actions",
}
CANDIDATE_FIELDS = {
    "candidate_id", "intent", "rationale", "evidence_ids", "material_evidence_ids",
}
INTENT_FIELDS = {
    "capability_id", "arguments", "discriminating_observation", "postcondition",
}


class DecisionContextError(ValueError):
    """Public decision input is malformed or crosses the declared boundary."""


class DecisionContextLimitError(DecisionContextError):
    """The complete projected view exceeds its explicit byte budget."""


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DecisionContextError("Decision context must be finite JSON") from error


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _public(value):
    """Reject evaluator fields in retained state/receipts, without guessing values."""
    if isinstance(value, dict):
        bad = set(value) & PRIVATE_FIELDS
        if bad:
            raise DecisionContextError("Non-public fields in retained input: " + ", ".join(sorted(bad)))
        return {key: _public(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return deepcopy(value)


def _candidates(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise DecisionContextError("Operation decisions require all 1-4 offered candidates")
    ids = []
    for value in values:
        if not isinstance(value, dict) or set(value) - CANDIDATE_FIELDS:
            raise DecisionContextError("Candidate contains unsupported fields")
        identifier = value.get("candidate_id")
        if not isinstance(identifier, str) or not identifier:
            raise DecisionContextError("Candidate IDs must be nonempty strings")
        intent = value.get("intent")
        if not isinstance(intent, dict) or set(intent) - INTENT_FIELDS:
            raise DecisionContextError("Candidate intent contains unsupported fields")
        if not isinstance(intent.get("capability_id"), str) or not isinstance(intent.get("arguments"), dict):
            raise DecisionContextError("Candidate needs exact capability and arguments")
        for field in ("evidence_ids", "material_evidence_ids"):
            citations = value.get(field, [])
            if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
                raise DecisionContextError("Candidate citations must be string arrays")
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise DecisionContextError("Candidate IDs must be distinct")
    # Actions are already validated by the graph. Never rewrite their arguments,
    # relabel their identifiers, infer legality or add an evaluator ranking here.
    return deepcopy(values)


def _native_action(intent):
    if not isinstance(intent, dict):
        return None
    arguments = intent.get("arguments", {})
    action = arguments.get("action") if isinstance(arguments, dict) else None
    if isinstance(action, dict):
        return action
    if isinstance(intent.get("capability_id"), str):
        return {"kind": "invoke", "capability_id": intent["capability_id"]}
    return None


def _command(action):
    if not isinstance(action, dict):
        return None
    return (action.get("instrument_id"), action.get("kind"), action.get("capability_id"))


def _history_view(history, actions, observation, *, exact_observation_refs=False):
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise DecisionContextError("Public native history must be an array of receipts")
    relevant = {_command(action) for action in actions if action}
    latest = {}
    counts = Counter()
    kinds = Counter()
    selected = set(range(max(0, len(history) - 2), len(history)))
    last_failure = last_selection = None
    corrections = 0
    for index, row in enumerate(history):
        action = row.get("executed_action")
        if isinstance(action, dict):
            kinds[str(action.get("kind", "unspecified"))] += 1
            key = _command(action)
            if key in relevant:
                latest[key] = index
                counts[key] += 1
            if action.get("kind") == "select_instrument":
                last_selection = index
        corrections += row.get("corrected") is True
        response = row.get("response", {})
        if isinstance(response, dict) and (response.get("error") or response.get("status") in {"rejected", "error", "failed"}):
            last_failure = index
    selected.update(latest.values())
    selected.update(index for index in (last_failure, last_selection) if index is not None)
    receipts = []
    for index in sorted(selected):
        row = history[index]
        receipt = {key: _public(row[key]) for key in
                   ("proposed_action", "executed_action", "corrected") if key in row}
        response = row.get("response", {})
        if not isinstance(response, dict):
            raise DecisionContextError("Native response must be an object")
        response = deepcopy(response)
        omitted = {}
        if "contract" in response:
            omitted["contract"] = {"sha256": _hash(response.pop("contract")),
                "representation": "Current candidate capability contracts are supplied separately; the historical whole-device contract is not repeated."}
        response = _public(response)
        # This reference is lossless: equality includes values, quality and times.
        if (isinstance(observation, dict) and "observations" in observation
                and ((_json(response.get("observations")) == _json(observation["observations"]))
                     if exact_observation_refs else response.get("observations") == observation["observations"])):
            response["observations"] = {"$decision_ref": "#/state/current_observation/observations"}
        receipt["response"] = response
        receipts.append({"history_index": index, "receipt": receipt,
                         "projected_response_fields": omitted})
    return {
        "count": len(history), "sha256": _hash(history),
        "selection_policy": "Last two receipts, latest failure, latest instrument selection and latest receipt for each candidate/selected command; older receipts remain in the raw trace.",
        "omitted_receipt_count": len(history) - len(receipts),
        "executed_action_kind_counts": dict(sorted(kinds.items())),
        "public_correction_count": corrections,
        "related_command_counts": [
            {"command": list(key), "count": counts[key], "last_history_index": latest.get(key)}
            for key in sorted(relevant, key=_json)
        ],
        "receipts": receipts,
    }


def _contract_view(view, actions, goal):
    contract = view.get("contract", {})
    if not isinstance(contract, dict):
        raise DecisionContextError("Public contract must be an object")
    capabilities = contract.get("capabilities", {})
    names = {action.get("capability_id") for action in actions
             if isinstance(action, dict) and action.get("kind") == "invoke"}
    retained = {key: _public(value) for key, value in capabilities.items() if key in names}
    slots = contract.get("state_slots", {})
    needed = set()
    def visit(value):
        if isinstance(value, str) and value in slots:
            needed.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit([actions, goal, retained])
    return {"capabilities": retained,
            "state_slots": {key: _public(slots[key]) for key in sorted(needed)},
            "selection_policy": "Only candidate/selected native capability contracts and explicitly referenced public state-slot definitions; no entire menu or preset bank.",
            "source_sha256": _hash(contract)}


def build_decision_context(public_input, *, purpose, candidates=None,
                           allowed_routes=None, max_bytes=DEFAULT_MAX_BYTES,
                           exact_observation_refs=False):
    """Return the complete bounded decision view, or fail before model inference.

    The caller keeps the original input and this view's audit in the raw trace.
    A byte cap supplements, rather than replaces, the backend's exact tokenizer
    context check. No claim of calibrated confidence is added to this protocol.
    """
    if purpose not in PURPOSES or not isinstance(public_input, dict):
        raise DecisionContextError("Expected a supported purpose and public input object")
    if type(max_bytes) is not int or max_bytes < 1:
        raise DecisionContextError("max_bytes must be a positive integer")
    original = deepcopy(public_input)
    encoded_original = _json(original).encode("utf-8")
    source = original.get("benchmark_public_context", {})
    if not isinstance(source, dict):
        raise DecisionContextError("benchmark_public_context must be an object")
    view = source.get("current_public_view", {})
    if not isinstance(view, dict):
        raise DecisionContextError("current_public_view must be an object")
    task = source.get("public_task", {})
    if not isinstance(task, dict):
        raise DecisionContextError("public_task must be an object")
    initial = task.get("initial_context", {})
    goal = view.get("goal", task.get("goal", initial.get("goal") if isinstance(initial, dict) else None))
    instrument = original.get("instrument_context", {})
    instrument = instrument if isinstance(instrument, dict) else {}
    intent = original.get("selected_intent", original.get("intent", instrument.get("selected_intent")))
    values = _candidates(candidates) if purpose == "operational_action_selection" else []
    if purpose == "routing":
        if (not isinstance(allowed_routes, list) or not 1 <= len(allowed_routes) <= 4
                or len(set(allowed_routes)) != len(allowed_routes)
                or any(not isinstance(item, str) or not item for item in allowed_routes)):
            raise DecisionContextError("Routing requires the exact distinct offered route list")
        if candidates is not None:
            raise DecisionContextError("Routing must not add operation candidates")
    actions = [_native_action(value["intent"]) for value in values]
    selected_action = _native_action(intent)
    if selected_action:
        actions.append(selected_action)
    observation = _public(view.get("observation", {}))
    state = {
        "graph": {key: _public(original[key]) for key in GRAPH_FIELDS if key in original},
        "goal": _public(goal), "target": _public(original.get("target")),
        "selected_instrument": view.get("selected_instrument"),
        "selected_intent": _public(intent),
        "current_observation": observation,
        "observation_wrappers": _public(original.get("observations", {})),
        "last_error": _public(original.get("last_error", instrument.get("last_error"))),
        "hypotheses": _public(original.get("hypotheses", [])),
        "verification": _public(original.get("verification")),
        "progress": {key: _public(view[key]) for key in
                     ("budget", "elapsed_sim_time_s", "mode", "observation_policy") if key in view},
        "wrapper_revision": source.get("wrapper_revision"),
    }
    history = source.get("complete_native_history", [])
    result = {
        "protocol": PROTOCOL, "purpose": purpose, "state": state,
        "history": _history_view(history, actions, observation,
                                 exact_observation_refs=exact_observation_refs),
        "candidate_contracts": _contract_view(view, actions, goal),
        "manual": _public(view.get("manual", [])),
        "retrieval_evidence": _public(original.get("retrieval_evidence", [])),
        "audit": {
            "input_sha256": hashlib.sha256(encoded_original).hexdigest(),
            "input_bytes": len(encoded_original),
            "projection": "Explicit current-decision role view; full original context remains in the trace and specialist inputs.",
            "excluded_sections": ["public_task.initial_context", "machine_catalog", "complete_offered_commands_and_presets", "unrelated_capability_contracts", "older_history_receipts"],
            "no_oracle_or_reference_ranking": True,
            "overflow_policy": "error_without_text_truncation",
        },
    }
    if purpose == "routing":
        result["allowed_routes"] = deepcopy(allowed_routes)
    else:
        result["candidates"] = values
    size = len(_json(result).encode("utf-8"))
    if size > max_bytes:
        raise DecisionContextLimitError(
            f"Complete {PROTOCOL} view needs {size} bytes, exceeding {max_bytes}; no evidence, candidates or text were truncated")
    return result


def serialize_decision_context(public_input, **kwargs):
    """Build and serialize a view with deterministic compact UTF-8 JSON."""
    return _json(build_decision_context(public_input, **kwargs))


def decision_model_content(view):
    """V3 model text omits provenance bookkeeping retained in the call trace.

    Native observations, their quality/times, actual response receipts, selected
    intents, exact actions and capability constraints remain unchanged. Only
    explicitly generated hashes and descriptions of the projection are omitted.
    """
    result = deepcopy(view)
    result.pop('audit', None)
    history = result.get('history', {})
    for key in ('sha256', 'selection_policy'):
        history.pop(key, None)
    for entry in history.get('receipts', []):
        entry.pop('projected_response_fields', None)
    contracts = result.get('candidate_contracts', {})
    for key in ('source_sha256', 'selection_policy'):
        contracts.pop(key, None)
    return result


def decision_model_content_v4(view):
    """Project schema-free menu-choice text while preserving exact public data.

    The unchanged full view and omitted parameter-schema hashes stay in the
    trace. Native/app validators and exact candidate criteria are unchanged.
    Receipt-local references replace only canonical-JSON-identical actions.
    """
    result = decision_model_content(view)
    omitted, omitted_output = {}, {}
    for name, capability in result.get('candidate_contracts', {}).get('capabilities', {}).items():
        if 'input_schema' in capability:
            omitted[name] = _hash(capability.pop('input_schema'))
        if 'output_schema' in capability:
            omitted_output[name] = _hash(capability.pop('output_schema'))
    action_refs = []
    for entry in result.get('history', {}).get('receipts', []):
        receipt = entry['receipt']
        if ('proposed_action' in receipt and 'executed_action' in receipt
                and _json(receipt['proposed_action']) == _json(receipt['executed_action'])):
            receipt.pop('proposed_action')
            receipt['proposed_action_ref'] = 'executed_action'
            action_refs.append(entry['history_index'])
    return result, {'projection': 'Input/output payload schemas omitted only from chooser model text; exact candidate arguments, native observations/receipts and native validation unchanged.',
        'omitted_input_schema_sha256_by_capability': omitted,
        'omitted_output_schema_sha256_by_capability': omitted_output,
        'proposed_action_reference_history_indices': action_refs,
        'observation_reference_policy': 'Only canonical-JSON-identical values, including numeric types, quality and timestamps.'}
