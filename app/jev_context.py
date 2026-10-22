"""Small typed Jev requests; source observations and candidates stay immutable."""
from copy import deepcopy
import json

from app.action_memory import MEMORY_INSTRUCTION


POLICY_VERSION = 'jev-essential-context-v1'
DECISION_INSTRUCTION = (
    'Choose the best next candidate for the stated objective using public observations, '
    'constraints, operation parameters and prerequisites. Hypotheses and rationales are proposals, '
    'not verified facts. Respect snapshot quality, timestamps, revisions, errors and pending operations. '
    'A snapshot_ref points to the identical snapshot elsewhere in observations. '
    'A command receipt alone does not establish process completion. '
    'The candidate operations are supplied once in the question criteria. '
    'Select one supplied choice; do not invent parameters or execute tools.'
) + ' ' + MEMORY_INSTRUCTION


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def essential_contract(cap):
    """Keep executable constraints and state meanings; exclude the global manual."""
    from app.lean_instrument import essential_capability
    result = essential_capability(cap)
    meaning = cap.get('function_zh') or cap.get('description') or cap.get('source')
    if meaning:
        result['description'] = deepcopy(meaning)
    # Already-lean contracts use public_state_ids rather than native state_ids.
    if isinstance(cap.get('preconditions'), list):
        for source, target in zip(cap['preconditions'], result.get('preconditions', [])):
            if isinstance(source, dict) and isinstance(target, dict) and source.get('public_state_ids'):
                target['public_state_ids'] = deepcopy(source['public_state_ids'])
    for name in ('output_schema', 'observes'):
        if name in cap:
            result[name] = deepcopy(cap[name])
    return result


def compact_context(context, candidates):
    result = deepcopy(context)
    selected = {c['intent']['capability_id'] for c in candidates}
    # Only public contracts for operations being compared, never the whole menu.
    contracts = {}
    for name in ('capabilities', 'candidate_capabilities'):
        for capability_id, cap in (result.pop(name, None) or {}).items():
            if capability_id in selected:
                contracts[capability_id] = essential_contract(cap)
    obs = result.get('observations')
    if isinstance(obs, dict):
        obs.pop('instrument_contract', None)
        snapshots = {}
        for name in ('inspection', 'measurement'):
            snapshot = obs.get(name)
            if not isinstance(snapshot, dict):
                continue
            # Equal IDs alone are insufficient: times/quality/values must match too.
            identical = next((old for old, value in snapshots.items() if value == snapshot), None)
            if identical:
                obs[name] = {'snapshot_ref': identical}
            else:
                snapshots[name] = snapshot
        detail = obs.get('capability_detail')
        if isinstance(detail, dict) and isinstance(detail.get('contract'), dict):
            capability_id = detail.get('capability_id')
            minimal = essential_contract(detail['contract'])
            if capability_id in contracts:
                # Preserve any queried output/state definitions without repeating inputs.
                contracts[capability_id].update(minimal)
                detail['contract'] = {'contract_ref': 'candidate_capabilities.' + capability_id}
            else:
                detail['contract'] = minimal
    if contracts:
        result['candidate_capabilities'] = contracts
    return result


def operation_request(context, candidates, model):
    if not 1 <= len(candidates) <= 4:
        raise ValueError('Operational decision supports 1-4 candidates')
    projected = compact_context(context, candidates)
    criteria = {chr(65+i): compact_json(candidate) for i, candidate in enumerate(candidates)}
    return {'model': model,
        'state': DECISION_INSTRUCTION + '\nPublic context:\n' + compact_json(projected),
        'questions': {'next_action': {'type': 'choice',
            'instructions': 'Select the candidate best supported by the public context and objective.',
            'criteria': criteria}}}, projected
