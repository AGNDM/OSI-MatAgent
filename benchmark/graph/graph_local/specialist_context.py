"""Readable public specialist role views for explicit menu-ids-v2/v3 transport.

This projection keeps literal cached readings and native feedback separate. It
does not infer current physical state, rank menu actions, refresh a reading, or
consult evaluator data. The full source payload remains in the graph trace.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json

from graph_local.decision_context import _public


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def readable_payload_json(value):
    """Preserve the readable view's deliberate feedback-before-cache field order."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _contract(contract):
    """Keep current command semantics/argument schemas and public slot metadata."""
    if not isinstance(contract, dict):
        raise ValueError('Current public contract must be an object')
    capabilities = {}
    for name, value in contract.get('capabilities', {}).items():
        capabilities[name] = {key: _public(value[key]) for key in
            ('description', 'description_zh', 'preconditions', 'preconditions_zh', 'observes', 'input_schema')
            if key in value}
    return {'capabilities': capabilities,
        'state_slots': {name: {key: _public(value[key]) for key in
            ('unit', 'origin', 'visibility', 'access') if key in value}
            for name, value in contract.get('state_slots', {}).items()}}


def readable_specialist_payload(payload, *, role, menu=None):
    """Return a declared role view and an audit; never alter literal menu actions.

    Recent receipts retain their exact actions, responses, errors and clocks.
    Historical whole-device contracts are omitted explicitly because the current
    selected contract is supplied once. Older receipts remain in the raw trace.
    """
    public = payload.get('benchmark_public_context')
    if not isinstance(public, dict):
        raise ValueError('menu-ids-v2 requires the public benchmark context')
    view = public.get('current_public_view', {})
    if not isinstance(view, dict):
        raise ValueError('Current public view must be an object')
    history = public.get('complete_native_history', [])
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError('Public native history must be an array of receipts')
    selected = set(range(max(0, len(history) - 6), len(history)))
    last_failure = last_read = last_success = None
    latest_command, successful_command = {}, {}
    command_counts = Counter()
    for index, row in enumerate(history):
        action = row.get('executed_action')
        if isinstance(action, dict):
            command_counts[_json(_public(action))] += 1
            command = (action.get('instrument_id'), action.get('kind'), action.get('capability_id'))
            latest_command[command] = index
        response = row.get('response', {})
        if not isinstance(response, dict):
            raise ValueError('Native receipt response must be an object')
        if response.get('error') or response.get('status') in {'rejected', 'failed', 'error'}:
            last_failure = index
        elif response.get('status') in {'succeeded', 'success', 'ok'}:
            last_success = index
            if isinstance(action, dict):
                successful_command[command] = index
        if isinstance(response.get('observations'), dict) and response['observations']:
            last_read = index
    selected.update(index for index in (last_failure, last_read, last_success) if index is not None)
    selected.update(latest_command.values())
    selected.update(successful_command.values())
    receipts, omitted_contracts = [], []
    for index in sorted(selected):
        source = history[index]
        receipt = {key: _public(source[key]) for key in
                   ('proposed_action', 'executed_action', 'corrected') if key in source}
        response = deepcopy(source.get('response', {}))
        if 'contract' in response:
            omitted_contracts.append({'history_index': index, 'sha256': _hash(response.pop('contract'))})
            receipt['contract_projection_note'] = 'Historical whole-device contract omitted; current selected contract is supplied below.'
        receipt['response'] = _public(response)
        receipts.append({'history_index': index, **receipt})
    observation = _public(view.get('observation', {}))
    native_values = observation.get('observations', observation) if isinstance(observation, dict) else observation
    last_values = history[last_read]['response']['observations'] if last_read is not None else None
    task = public.get('public_task') or {}
    initial = task.get('initial_context') or {}
    target = payload.get('target') or {}
    goal = view.get('goal', target.get('goal', task.get('goal', initial.get('goal'))))
    by_index = {item['history_index']: item for item in receipts}
    result = {
        'protocol': 'menu-ids-v2', 'role': role, 'goal': _public(goal),
        'selected_instrument': view.get('selected_instrument'),
        'latest_native_receipt': deepcopy(by_index[len(history) - 1]) if history else None,
        'latest_successful_native_receipt': deepcopy(by_index[last_success]) if last_success is not None else None,
        'current_public_wrapper_ids_and_metadata': _public(payload.get('observations', {})),
        'cached_native_observation': observation,
        'cached_reading_provenance': {
            'latest_native_receipt_with_observations_index': last_read,
            'cached_values_equal_that_receipt': native_values == last_values if last_read is not None else None,
            'receipts_after_that_reading': len(history) - last_read - 1 if last_read is not None else None,
            'interpretation': 'Readings are cached values with their original quality and sample times. Later receipts can report newer command outcomes even at the same simulation time. Wrapper revision does not refresh cached physical readings.'},
        'graph_last_error': _public(payload.get('last_error')),
        'last_native_response_error': _public(history[-1].get('response', {}).get('error')) if history else None,
        'native_receipts_in_execution_order': receipts,
        'public_history_progress': {'native_decision_count': len(history),
            'omitted_older_receipt_count': len(history) - len(receipts),
            'identical_executed_action_counts': [
                {'action': json.loads(action), 'count': count}
                for action, count in sorted(command_counts.items())]},
        'simulation_progress': {key: _public(view[key]) for key in
            ('budget', 'elapsed_sim_time_s', 'mode', 'observation_policy') if key in view},
        'public_manual': _public(view.get('manual', [])),
        'current_public_contract': _contract(view.get('contract', {})),
    }
    if not view.get('selected_instrument'):
        result['available_instruments'] = _public(view.get('machine_catalog', []))
    if role == 'diagnostic':
        if not isinstance(menu, list) or not menu:
            raise ValueError('Diagnostic role needs every current offered menu action')
        result['offered_menu_actions'] = _public(menu)
    audit = {'input_projection': 'readable-current-specialist-view-v2',
        'source_payload_sha256': _hash(payload),
        'native_history_sha256': _hash(history), 'native_history_count': len(history),
        'retained_history_indices': sorted(selected), 'historical_contracts_omitted': omitted_contracts,
        'history_projection': 'Last six receipts, latest receipt and latest success for each public native command, latest failure and latest receipt with native observations.',
        'current_contract_sha256': _hash(view.get('contract', {})),
        'contract_projection': 'Current descriptions, preconditions, observes and input schemas; slot units/origins/access. Output/value schemas omitted.',
        'excluded_source_sections': ['public_task.initial_context', 'duplicated_full_history',
            'previous_model_hypotheses', 'unselected_instrument_catalog_after_selection',
            'duplicate_command_preset_wrappers', 'historical_whole_device_contracts'],
        'menu_for_verification': 'omitted; verifier proposes no actions' if role == 'verification' else 'all distinct offered literal actions retained',
        'inferred_physical_state': False, 'ranked_or_filtered_actions': False,
        'serialized_payload_sha256': hashlib.sha256(readable_payload_json(result).encode()).hexdigest()}
    return result, audit


def _same_json(left, right):
    """Compare JSON values including numeric types, sample times and quality."""
    return _json(left) == _json(right)


def grouped_menu_actions(menu, *, omission_audit=None):
    """Factor only identical literal native action prefixes; retain every choice.

    A choice inherits its group's common_action, then supplies every remaining
    native field literally. No action is filtered or ordered by usefulness.
    Preset text is omitted only for exact JSON repetition or the public menu's
    mechanically derived Read/Wait labels. Original text stays in the raw trace.
    """
    groups, by_prefix, omitted_text = [], {}, []
    for item in menu:
        action = item['action']
        if not isinstance(action, dict):
            raise ValueError('A public menu action must be an object')
        reserved = {'menu_action_id', 'preset_text'} & set(action)
        if reserved:
            raise ValueError('Native action fields collide with menu metadata')
        common = {key: deepcopy(action[key]) for key in
                  ('kind', 'instrument_id', 'capability_id') if key in action}
        signature = _json(common)
        if signature not in by_prefix:
            group = {'common_action': common, 'choices': []}
            by_prefix[signature] = group
            groups.append(group)
        choice = {'menu_action_id': item['menu_action_id'],
                  **{key: deepcopy(value) for key, value in action.items() if key not in common}}
        text = item.get('preset_text')
        redundant = False
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                redundant = (_same_json(parsed, action) or
                             ('arguments' in action and _same_json(parsed, action['arguments'])))
            except (ValueError, TypeError):
                pass
        if redundant:
            omitted_text.append(item['menu_action_id'])
        elif (isinstance(text, str) and
              ((action.get('kind') == 'read' and isinstance(action.get('state_ids'), list)
                and all(isinstance(name, str) for name in action['state_ids'])
                and text == 'Read ' + ', '.join(action['state_ids']))
               or (action.get('kind') == 'wait' and type(action.get('dt_s')) in (int, float)
                   and text == f"Wait {action['dt_s']:g} simulated seconds"))):
            if omission_audit is not None:
                omission_audit.setdefault('derived_preset_text_omitted_ids', []).append(item['menu_action_id'])
        elif 'preset_text' in item:
            choice['preset_text'] = deepcopy(text)
        by_prefix[signature]['choices'].append(choice)
    # Validate the representation before it can reach a specialist.
    expected = {item['menu_action_id']: item['action'] for item in menu}
    restored = reconstruct_menu_actions(groups)
    if len(expected) != len(menu) or not _same_json(restored, expected):
        raise ValueError('Grouped public menu did not preserve every exact action')
    return groups, omitted_text


def reconstruct_menu_actions(groups):
    """Reconstruct exact native actions from the v3 human-readable menu groups."""
    actions = {}
    for group in groups:
        common = group['common_action']
        if not isinstance(common, dict):
            raise ValueError('Menu common_action must be an object')
        for choice in group['choices']:
            identifier = choice['menu_action_id']
            fields = {key: value for key, value in choice.items()
                      if key not in ('menu_action_id', 'preset_text')}
            if identifier in actions or set(common) & set(fields):
                raise ValueError('Menu action IDs and native fields must be unambiguous')
            actions[identifier] = deepcopy({**common, **fields})
    return actions


def resolve_specialist_contract(contract):
    """Restore exact repeated precondition lists from named capability references."""
    result = deepcopy(contract)
    capabilities = result.get('capabilities', {})
    for name, capability in capabilities.items():
        if 'preconditions_ref' not in capability:
            continue
        reference = capability.pop('preconditions_ref')
        if ('preconditions' in capability or reference == name or reference not in capabilities
                or 'preconditions_ref' in capabilities[reference]
                or not isinstance(capabilities[reference].get('preconditions'), list)):
            raise ValueError('Unknown or ambiguous preconditions capability reference')
        capability['preconditions'] = deepcopy(capabilities[reference]['preconditions'])
    return result


def resolve_specialist_action_counts(view):
    """Restore exact count-metadata actions from this diagnostic view's menu IDs."""
    rows = deepcopy(view['public_history_progress']['identical_executed_action_counts'])
    actions = reconstruct_menu_actions(view.get('offered_action_groups', []))
    for row in rows:
        if 'action_from_menu_id' in row:
            identifier = row.pop('action_from_menu_id')
            if identifier not in actions or 'action' in row:
                raise ValueError('Unknown or ambiguous action-count menu reference')
            row['action'] = deepcopy(actions[identifier])
    return rows


def resolve_specialist_receipt(view, receipt):
    """Resolve only declared v3 receipt/observation reference positions.

    Native values are never recursively interpreted as references. Thus a
    literal native object containing a similarly named key remains literal.
    """
    if receipt is None:
        return None
    result = deepcopy(receipt)
    seen = set()
    while 'receipt_ref' in result:
        name = result['receipt_ref']
        if name in seen or name not in ('latest_native_receipt', 'latest_successful_native_receipt'):
            raise ValueError('Unknown or cyclic specialist receipt reference')
        seen.add(name)
        index = result['history_index']
        result = deepcopy(view[name])
        if result is None or result['history_index'] != index:
            raise ValueError('Specialist receipt reference changed its history index')
    response = result.get('response', {})
    if 'proposed_action_ref' in result:
        if result.pop('proposed_action_ref') != 'executed_action' or 'proposed_action' in result:
            raise ValueError('Unknown or ambiguous proposed action reference')
        result['proposed_action'] = deepcopy(result['executed_action'])
    if 'observations_ref' in response and 'observations' not in response:
        name = response.pop('observations_ref')
        if name == 'cached_native_observation.observations':
            response['observations'] = deepcopy(view['cached_native_observation']['observations'])
        elif name == 'cached_native_observation':
            response['observations'] = deepcopy(view['cached_native_observation'])
        else:
            raise ValueError('Unknown specialist observation reference')
    if 'observations_from_cached' in response:
        names = response.pop('observations_from_cached')
        cache = view['cached_native_observation']
        values = cache['observations'] if isinstance(cache, dict) and 'observations' in cache else cache
        observations = response.get('observations', {})
        if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or not isinstance(values, dict)
                or not isinstance(observations, dict) or set(names) & set(observations)
                or any(name not in values for name in names)):
            raise ValueError('Unknown or ambiguous cached observation entries')
        response['observations'] = {**observations, **{name: deepcopy(values[name]) for name in names}}
    return result


def compact_readable_specialist_payload(payload, *, role, menu=None):
    """V3 keeps native evidence and exact menu choices with duplicate references.

    The cache is shown once. A response uses a named cache reference only for
    byte-equivalent canonical JSON values, including numeric types and metadata.
    Each retained receipt payload is shown once; history references preserve its
    original index and order. Parameter schemas stay in the source/native
    validator, with omitted-schema hashes audited. V1/v2 and actions stay intact.
    """
    result, audit = readable_specialist_payload(payload, role=role, menu=menu)
    result['protocol'] = 'menu-ids-v3'
    omitted_input_schemas = {}
    first_preconditions, precondition_refs = {}, {}
    for capability_id, capability in result['current_public_contract']['capabilities'].items():
        if 'input_schema' in capability:
            omitted_input_schemas[capability_id] = _hash(capability.pop('input_schema'))
        conditions = capability.get('preconditions')
        if isinstance(conditions, list):
            signature = _json(conditions)
            if signature in first_preconditions:
                if 'preconditions_ref' in capability:
                    raise ValueError('Public capability collides with precondition reference metadata')
                capability.pop('preconditions')
                capability['preconditions_ref'] = first_preconditions[signature]
                precondition_refs[capability_id] = first_preconditions[signature]
            else:
                first_preconditions[signature] = capability_id
    if role == 'diagnostic' and payload.get('last_verification_feedback') is not None:
        result['historical_last_verification_feedback'] = {
            'report': _public(payload['last_verification_feedback']),
            'meaning': 'Previous verifier assessment with its original observation ID and evidence signature. '
                       'Its cited evidence is historical; this field does not assert current or fresh native evidence.'}
    if 'format_repair' in payload:
        # Preserve invalid prior output too: its fields need not match the public
        # native schema, and must remain available for the actual repair attempt.
        result['format_repair'] = deepcopy(payload['format_repair'])
        result['format_repair_semantics'] = (
            'Actual JSON/schema validation feedback for the preceding model response. '
            'previous_response is model output, not a native observation or an executed action.')
    cache = result['cached_native_observation']
    has_map = isinstance(cache, dict) and 'observations' in cache
    values = cache['observations'] if has_map else cache
    cache_path = 'cached_native_observation.observations' if has_map else 'cached_native_observation'
    deduplicated, deduplicated_actions, entry_references = [], [], []
    receipts = result['native_receipts_in_execution_order']
    for receipt in receipts:
        if ('proposed_action' in receipt and 'executed_action' in receipt
                and _same_json(receipt['proposed_action'], receipt['executed_action'])):
            receipt.pop('proposed_action')
            receipt['proposed_action_ref'] = 'executed_action'
            deduplicated_actions.append(receipt['history_index'])
        response = receipt.get('response', {})
        if 'observations' in response and _same_json(response['observations'], values):
            if 'observations_ref' in response:
                raise ValueError('Native response collides with specialist reference metadata')
            response.pop('observations')
            response['observations_ref'] = cache_path
            deduplicated.append(receipt['history_index'])
        elif isinstance(response.get('observations'), dict) and isinstance(values, dict):
            observations = response['observations']
            names = [name for name, value in observations.items()
                     if name in values and _same_json(value, values[name])]
            if names:
                if 'observations_from_cached' in response:
                    raise ValueError('Native response collides with cached-entry reference metadata')
                response['observations'] = {name: value for name, value in observations.items() if name not in names}
                response['observations_from_cached'] = names
                entry_references.append({'history_index': receipt['history_index'], 'state_ids': names})
    by_index = {item['history_index']: item for item in receipts}
    latest = result['latest_native_receipt']
    success = result['latest_successful_native_receipt']
    latest_index = latest['history_index'] if latest is not None else None
    success_index = success['history_index'] if success is not None else None
    result['latest_native_receipt'] = deepcopy(by_index[latest_index]) if latest is not None else None
    if success is not None:
        result['latest_successful_native_receipt'] = (
            {'history_index': success_index, 'receipt_ref': 'latest_native_receipt'}
            if success_index == latest_index else deepcopy(by_index[success_index]))
    for index, receipt in enumerate(receipts):
        target = ('latest_native_receipt' if receipt['history_index'] == latest_index else
                  'latest_successful_native_receipt' if receipt['history_index'] == success_index else None)
        if target:
            receipts[index] = {'history_index': receipt['history_index'], 'receipt_ref': target}
    public = payload['benchmark_public_context']
    last_read = result['cached_reading_provenance']['latest_native_receipt_with_observations_index']
    if last_read is not None:
        last_values = public['complete_native_history'][last_read]['response']['observations']
        result['cached_reading_provenance']['cached_values_equal_that_receipt'] = _same_json(values, last_values)
    omitted_text, menu_omission_audit = [], {}
    count_menu_refs = []
    if role == 'diagnostic':
        original_menu = result.pop('offered_menu_actions')
        groups, omitted_text = grouped_menu_actions(original_menu, omission_audit=menu_omission_audit)
        result['offered_action_groups'] = groups
        exact_actions = {_json(item['action']): item['menu_action_id'] for item in original_menu}
        for index, row in enumerate(result['public_history_progress']['identical_executed_action_counts']):
            identifier = exact_actions.get(_json(row['action']))
            if identifier is not None:
                row.pop('action')
                row['action_from_menu_id'] = identifier
                count_menu_refs.append({'count_index': index, 'menu_action_id': identifier})
    audit.update(input_projection='readable-current-specialist-view-v3',
        observation_reference_history_indices=deduplicated,
        observation_entry_references=entry_references,
        proposed_action_reference_history_indices=deduplicated_actions,
        proposed_action_reference_policy='Only canonical-JSON-identical proposed/executed native actions; receipt-local executed_action is kept literally.',
        observation_reference_policy='Only canonical-JSON-identical current cached observation maps or individual state-ID readings; every differing value/type/quality/time remains literal.',
        receipt_reference_policy='Latest and latest successful receipts appear once; ordered history retains explicit indexed references.',
        menu_projection='Literal common native kind/instrument/capability plus every remaining native field; all actions exactly reconstructable.',
        redundant_json_preset_text_omitted_ids=omitted_text,
        redundant_derived_preset_text_omitted_ids=menu_omission_audit.get('derived_preset_text_omitted_ids', []),
        derived_preset_text_omission_policy='Only exact Read+joined-state-IDs and Wait+g-formatted-dt_s public menu labels; all native IDs/fields/values/order retained.',
        omitted_input_schema_sha256_by_capability=omitted_input_schemas,
        preconditions_reference_by_capability=precondition_refs,
        preconditions_reference_policy='Only entire canonical-JSON-identical precondition lists, including every check, description and state ID; first list remains literal.',
        action_count_menu_references=count_menu_refs,
        contract_projection='Current descriptions, preconditions and observes; slot units/origins/access. Parameter input schemas omitted from specialist model input: exact offered menu actions and arguments remain complete, and native/app validation is unchanged.',
        serialized_payload_sha256=hashlib.sha256(readable_payload_json(result).encode()).hexdigest())
    return result, audit
