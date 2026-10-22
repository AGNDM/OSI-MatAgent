"""Public operation memory and snapshot freshness, with no runtime reads."""
from copy import deepcopy
import math


POLICY_VERSION = 'public-action-memory-v1'
RECENT_ACTIONS = 6
MEMORY_INSTRUCTION = (
    'Use operation_memory for past actions and native receipts; it survives intervening waits. '
    'snapshot_status marks historical observations that cannot establish current state. '
    'A receipt reports an operation outcome, not a fresh sensor reading or reaction completion. '
    'Choose whether, when and which states to read yourself.'
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _small_result(result):
    """Keep returned identifiers/scalars; name omitted bulk arrays instead of guessing."""
    if not isinstance(result, dict):
        return {}, []
    kept, omitted = {}, []
    for key, value in result.items():
        if value is None or isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= 256):
            kept[key] = deepcopy(value)
        else:
            omitted.append(key)
    return kept, omitted


def operation_memory(history, current_time_s, recent_limit=RECENT_ACTIONS):
    # In this workflow only await advances time. Use actual advanced_seconds,
    # including waits cut short by a boundary, not the requested duration.
    advanced = sum((e.get('result') or {}).get('advanced_seconds', 0) for e in history)
    clock = current_time_s - advanced
    entries, latest_receipt = [], {}
    for event in history:
        proposal, result = event['proposal'], event.get('result') or {}
        clock += result.get('advanced_seconds', 0)
        row = dict(step=event['step'], time_s=clock, capability_id=proposal['capability_id'],
                   arguments=deepcopy(proposal['arguments']))
        native = result.get('native_response')
        if isinstance(native, dict):
            receipt = {key: deepcopy(native[key]) for key in (
                'status', 'request_id', 'operation_id', 'state_revision', 'error') if key in native}
            native_time = (native.get('clock') or {}).get('time_s')
            if _number(native_time):
                row['time_s'] = native_time
            returned, omitted = _small_result(native.get('result'))
            if returned:
                receipt['result'] = returned
            if omitted:
                receipt['omitted_result_fields'] = omitted
            row['receipt'] = receipt
            latest_receipt[proposal['capability_id']] = row
        elif result.get('error'):
            row['error'] = deepcopy(result['error'])
        elif 'advanced_seconds' in result:
            row['advanced_seconds'] = result['advanced_seconds']
        elif isinstance(result.get('measurement'), dict):
            snapshot = result['measurement']
            row['observation_id'] = snapshot['id']
            row['observation_revision'] = snapshot.get('revision')
        elif isinstance(result.get('capability_description'), dict):
            row['described_capability'] = result['capability_description'].get('capability_id')
        elif result.get('stopped'):
            row['stopped'] = True
            row['reason'] = result.get('reason')
        entries.append(row)
    recent = entries[-recent_limit:]
    recent_steps = {row['step'] for row in recent}
    earlier = sorted((row for row in latest_receipt.values() if row['step'] not in recent_steps),
                     key=lambda row: row['step'])
    return dict(policy=POLICY_VERSION, total_actions=len(history),
        recent_actions=recent, earlier_latest_receipts=earlier,
        meaning='Recent 6 actions plus the latest native receipt per capability beyond that window. '
                'Historical receipts, not current sensor values. Full event data remains in the run log.')


def snapshot_status(observations):
    now, revision = observations.get('simulated_seconds'), observations.get('revision')
    mutation = observations.get('last_mutation_step', 0)
    result = {}
    for name in ('inspection', 'measurement'):
        snapshot = observations.get(name)
        if not isinstance(snapshot, dict):
            result[name] = {'status': 'missing'}
            continue
        native = snapshot.get('native') or {}
        read_time = (native.get('clock') or {}).get('time_s')
        reasons = []
        age = now - read_time if _number(now) and _number(read_time) else None
        if snapshot.get('revision') != revision:
            reasons.append('world_revision_changed_since_read')
        if snapshot.get('captured_at_step', -1) <= mutation:
            reasons.append('state_changing_action_after_read')
        if age is not None and age > 1e-9:
            reasons.append('simulation_advanced_since_read')
        status = 'stale' if reasons else 'current'
        if age is None or age < -1e-9 or snapshot.get('revision') is None:
            status = 'unknown'
            reasons.append('missing_or_inconsistent_clock_or_revision')
        result[name] = dict(status=status, observation_id=snapshot.get('id'),
            read_time_s=read_time, current_time_s=now, age_s=age,
            read_revision=snapshot.get('revision'), current_revision=revision, reasons=reasons)
    return result


def attach_public_memory(observations, history):
    result = deepcopy(observations)
    result['operation_memory'] = operation_memory(history, result['simulated_seconds'])
    result['snapshot_status'] = snapshot_status(result)
    return result
