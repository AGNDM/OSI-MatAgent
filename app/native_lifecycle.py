"""Native lifecycle bindings inferred from API contracts, never device identities."""
from copy import deepcopy
import inspect
import math


def clock_binding(host):
    """Recognize the pinned runtimes' relative/absolute host clock protocols."""
    for name in ('step', 'host_step', 'host_advance', 'host_set_clock'):
        method = getattr(host, name, None)
        if not callable(method):
            continue
        signature = inspect.signature(method)
        params = list(signature.parameters.values())
        if not params:
            continue
        first = params[0].name
        if first in {'dt', 'dt_s'}:
            mode = 'delta_world' if 'world_time_s' in signature.parameters else 'delta'
        elif first in {'time_s', 'world_time_s'}:
            mode = 'absolute'
        else:
            continue
        kwargs = {first: 1.0}
        if mode == 'delta_world':
            kwargs['world_time_s'] = 1.0
        try:
            signature.bind(**kwargs)
        except TypeError:
            continue
        return {'method': name, 'mode': mode, 'time_parameter': first,
                'signature': str(signature),
                'effect': 'clock_only' if name == 'host_set_clock' else 'native_dynamics'}
    return None


def operation_binding(session, name, instance):
    method = getattr(session, name, None)
    if not callable(method):
        return None
    signature = inspect.signature(method)
    kwargs = {'operation_id': 'operation-probe'}
    if name == 'cancel':
        kwargs['request_id'] = 'request-probe'
    if 'instance_id' in signature.parameters or any(
            p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        kwargs['instance_id'] = instance
    try:
        signature.bind(**kwargs)
    except TypeError:
        return None
    return {'method': name, 'with_instance': 'instance_id' in kwargs}


class NativeLifecycle:
    def __init__(self, spec, host, session, instance, description=None):
        self.host, self.session, self.instance = host, session, instance
        self.clock = clock_binding(host)
        self.can_step = self.clock is not None
        self.elapsed = self.initial_clock = 0.0
        self.calls = dict(read=0, invoke=0, step=0, poll=0, cancel=0)
        if description is None:
            description = session.describe(instance_id=instance)
        self.async_capabilities = [name for name, cap in description.get('capabilities', {}).items()
            if cap.get('availability', 'implemented') == 'implemented'
            and cap.get('visibility', 'agent') == 'agent'
            and any(isinstance(cap.get(field), dict) and cap[field].get('mode') == 'asynchronous'
                    for field in ('execution', 'execution_notes'))]
        # Placeholder poll methods on synchronous runtimes are not async support.
        self.poll_binding = operation_binding(session, 'poll', instance) if self.async_capabilities else None
        self.cancel_binding = operation_binding(session, 'cancel', instance) if self.async_capabilities else None
        self.can_poll = self.poll_binding is not None
        self.can_cancel = self.cancel_binding is not None

    def manifest(self):
        return {'step': self.can_step, 'poll': self.can_poll, 'cancel': self.can_cancel,
                'clock_binding': deepcopy(self.clock), 'async_capabilities': self.async_capabilities,
                'discovery': 'Native method signatures and declared asynchronous capability contracts; no device whitelist'}

    def step(self, duration):
        if not self.can_step:
            raise ValueError('NATIVE_CLOCK_API_ABSENT')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('INVALID_CLOCK_DURATION')
        self.calls['step'] += 1
        binding = self.clock
        target = self.initial_clock + self.elapsed + duration
        kwargs = {binding['time_parameter']: target if binding['mode'] == 'absolute' else duration}
        if binding['mode'] == 'delta_world':
            kwargs['world_time_s'] = target
        getattr(self.host, binding['method'])(**kwargs)
        self.elapsed += duration

    def _operation(self, binding, operation_id, request_id=None):
        if binding is None:
            raise ValueError('NATIVE_OPERATION_API_ABSENT')
        name = binding['method']
        kwargs = {'operation_id': operation_id}
        if binding['with_instance']:
            kwargs['instance_id'] = self.instance
        if name == 'cancel':
            kwargs['request_id'] = request_id
        self.calls[name] += 1
        response = getattr(self.session, name)(**kwargs)
        if not isinstance(response, dict):
            raise ValueError('INVALID_NATIVE_OPERATION_RESPONSE')
        # Some native sessions return only operation_id/status/result. Preserve
        # that payload and obtain missing envelope fields from a fresh read of
        # the same bound session; never replace contradictory native fields.
        if any(k not in response for k in ('instance_id', 'clock', 'state_revision')):
            raw = deepcopy(response)
            self.calls['read'] += 1
            receipt = self.session.read(instance_id=self.instance, state_ids=[])
            response = deepcopy(response)
            for field in ('instance_id', 'clock', 'state_revision'):
                if field not in response:
                    response[field] = deepcopy(receipt.get(field))
            response['native_operation_response'] = raw
            response['envelope_source'] = 'Fresh read from same native session; original operation fields retained'
        return response

    def poll(self, operation_id):
        return self._operation(self.poll_binding, operation_id)

    def cancel(self, operation_id, request_id):
        return self._operation(self.cancel_binding, operation_id, request_id)

    def operation_status(self, response):
        return response.get('status')
