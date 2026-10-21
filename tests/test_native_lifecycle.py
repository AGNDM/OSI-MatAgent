from pathlib import Path
from uuid import uuid4

import pytest

from app.native_lifecycle import NativeLifecycle
from app.device_registry import get_device, open_native
from app.single_instrument import InstrumentLab, InstrumentTask
from tests.test_single_instrument import execute


class Session:
    def __init__(self, host): self.host = host
    def read(self, **kwargs):
        return {'instance_id': 'instance', 'state_revision': 0,
                'clock': {'time_s': self.host.time_s}, 'status': 'succeeded'}


@pytest.mark.parametrize('protocol', ['delta', 'absolute', 'delta_world', 'advance', 'clock_only'])
def test_unknown_device_uses_native_protocol_and_nonzero_clock(protocol):
    class Host:
        time_s = 20.0
    host = Host()
    if protocol == 'delta':
        def step(dt_s): host.time_s += dt_s
        host.step = step
    elif protocol in {'absolute', 'advance'}:
        def absolute(time_s): host.time_s = time_s
        setattr(host, 'host_step' if protocol == 'absolute' else 'host_advance', absolute)
    elif protocol == 'delta_world':
        def combined(dt_s, *, world_time_s):
            assert world_time_s == host.time_s + dt_s
            host.time_s = world_time_s
        host.host_step = combined
    else:
        def set_clock(world_time_s): host.time_s = world_time_s
        host.host_set_clock = set_clock
    bridge = NativeLifecycle({'module': 'never-listed', 'class': 'NewDevice'}, host, Session(host), 'instance', {})
    bridge.initial_clock = 20
    bridge.step(4); bridge.step(6)
    assert host.time_s == 30 and bridge.elapsed == 10
    assert bridge.clock['effect'] == ('clock_only' if protocol == 'clock_only' else 'native_dynamics')


def test_partial_operation_receipt_uses_real_read_and_keeps_wrong_identity():
    class Host: time_s = 0
    class BareSession(Session):
        def poll(self, operation_id):
            return {'operation_id': operation_id, 'status': 'running', 'result': {'progress': .2}}
        def cancel(self, *, operation_id, request_id):
            return {'instance_id': 'WRONG', 'operation_id': operation_id, 'status': 'cancelled'}
    host = Host();session = BareSession(host)
    desc = {'capabilities': {'run': {'execution': {'mode': 'asynchronous'}}}}
    bridge = NativeLifecycle({}, host, session, 'instance', desc)
    receipt = bridge.poll('op')
    assert receipt['instance_id'] == 'instance' and receipt['status'] == 'running'
    assert receipt['native_operation_response'] == {'operation_id': 'op', 'status': 'running', 'result': {'progress': .2}}
    assert bridge.cancel('op', 'request')['instance_id'] == 'WRONG'
    assert bridge.calls['read'] == 2


def test_real_balance_waits_with_native_clock_and_keeps_setpoint():
    task = InstrumentTask.model_validate_json(Path('cases/acceptance/v1/balance-filter.json').read_text()).model_copy(update={'device':'mt_xpr204e'})
    lab = InstrumentLab(task)
    assert 'workflow.await' in lab.capabilities()
    execute(lab, 'workflow.inspect')
    execute(lab, 'mt_xpr204e::balance.set_filter', {'level':'High'})
    execute(lab, 'workflow.await', {'duration_s':10})
    score = lab.score({'status':'unresolved'})
    assert score['goal_met'] and score['protocol_correct'] and lab.lifecycle.elapsed == 10


def test_real_bare_poll_async_measurement_completes_through_generic_adapter():
    # Explicit developer fixture only: no setup/actions are loaded by the solver.
    spec = get_device('mt_hx204');host,session,instance,description = open_native(spec)
    host.invoke('power.set', enabled=True)
    host.invoke('lid.set_position', position_m=.1085)
    host.invoke('sample.configure', species_kg={'ethanol':.001}, residue_kg=.004)
    host.invoke('lid.set_position', position_m=0)
    host.invoke('method.configure', mode='standard', duration_s=60, basis='wet')
    task = InstrumentTask(task_id='async-native', device=spec['slug'], objective='Complete the native configured acquisition',
        predicates=[{'state_id':'cycle.phase','unit':'1','value':'succeeded'}])
    # Use actual public acquisition state ID from this pinned contract.
    phase = next(k for k in description['state_slots'] if k.endswith('.phase'))
    task.predicates[0].state_id = phase
    lab = InstrumentLab(task, native=(host,session,instance,description))
    assert lab.lifecycle.can_poll and lab.lifecycle.can_cancel
    execute(lab, 'workflow.inspect')
    execute(lab, spec['slug']+'::acquisition.start')
    assert lab.pending_operation and lab.termination_reason is None
    event = execute(lab, 'workflow.poll')
    assert event['result']['native_response']['native_operation_response']['status'] == 'running'
    assert lab.lifecycle.elapsed == 0
    execute(lab, 'workflow.await', {'duration_s':60})
    assert lab.pending_operation is None and lab.termination_reason is None
    assert lab.score({'status':'unresolved'})['goal_met']
