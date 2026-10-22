import json

import pytest

from app.models import ActionProposal
from app.single_instrument import InstrumentTask, StatePredicate
from app.rc1mx_experiment import Rc1mxExperiment, TASK_PATH
from app.lab_adapter import GateRejected


def new_lab():
    return Rc1mxExperiment(InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8')).model_copy(update={'device': 'mt_rc1mxhfcal'}))


def action(lab, cap, arguments=None):
    return ActionProposal(sample_id=lab.sample_id, expected_revision=lab.revision,
        request_id=f'test-{lab.action_count}-{cap}', capability_id=cap,
        arguments=arguments or {}, postcondition='Test stopping invariant')


def test_initial_conditions_preserve_agent_evaluator_separation():
    lab = new_lab()
    assert lab.lifecycle.elapsed == 0
    assert lab.termination_reason is None
    assert lab.samples[0]['conversion'] == 0
    assert 24 <= lab.samples[0]['reactor_c'] <= 26
    assert lab.monitor_summary()['temperature_band_c'] == [34, 36]
    assert lab.monitor_summary()['warmup_temperature_band_c'] == [24, 36]
    assert lab.hold_started_s is None
    assert all(not key.startswith('truth.') for key in lab.fields)
    assert 'mt_rc1mxhfcal::reaction.configure' not in lab.capabilities()
    lab.execute(action(lab, 'workflow.inspect'))
    assert 'truth.composition_mol' not in json.dumps(lab.observe())


def test_first_failed_constraint_interrupts_long_wait_at_two_seconds():
    lab = new_lab()
    lab.execute(action(lab, 'workflow.inspect'))
    # Deliberately impossible boundary checks monitor behavior, not a control policy.
    lab.task.constraints = [StatePredicate(state_id='reactor.indicated_k', operator='ge', value=350, unit='K')]
    event = lab.execute(action(lab, 'workflow.await', {'duration_s': 120}))
    assert event['result']['advanced_seconds'] == 2
    assert lab.lifecycle.elapsed == 2
    assert lab.termination_reason == 'constraint_violated'
    assert lab.boundary['simulated_seconds'] == 2
    with pytest.raises(GateRejected):
        lab.execute(action(lab, 'workflow.await', {'duration_s': 120}))
    assert lab.lifecycle.elapsed == 2


def test_completion_interrupts_wait_without_advancing_to_requested_end(monkeypatch):
    import app.rc1mx_experiment as module
    lab = new_lab()
    monkeypatch.setattr(module, 'MIN_CONVERSION', 0.0)
    lab.task.predicates = [StatePredicate(state_id='clock.time_s', operator='ge', value=2, unit='s')]
    lab.execute(action(lab, 'workflow.inspect'))
    event = lab.execute(action(lab, 'workflow.await', {'duration_s': 120}))
    assert lab.termination_reason == 'task_completed'
    assert event['result']['advanced_seconds'] == 2
    assert lab.score({'status':'unresolved'})['task_success']


def test_native_rejection_latches_boundary_without_retry():
    lab = new_lab()
    lab.execute(action(lab, 'workflow.inspect'))
    event = lab.execute(action(lab, 'mt_rc1mxhfcal::thermostat.set_tr',
        {'end_k':400.0, 'ramp_s':0.0, 'controller':'aqueous'}))
    assert event['result']['error']['code'] == 'OUTSIDE_SAFETY_LIMITS'
    assert lab.termination_reason == 'native_boundary'
    assert lab.lifecycle.elapsed == 0
    assert not lab.score({'status':'unresolved'})['task_success']


def test_attempt_to_change_batch_stops_at_gate():
    lab = new_lab()
    with pytest.raises(GateRejected):
        lab.validate(action(lab, 'mt_rc1mxhfcal::dosing.start', {}))
    assert lab.termination_reason == 'capability_boundary'
    assert lab.boundary['detail']['code'] == 'FIXED_BATCH_COMPOSITION'
    assert lab.lifecycle.elapsed == 0


def scripted_temperature_monitor(monkeypatch, lab):
    from copy import deepcopy
    from app.autonomous_instrument import AutonomousInstrumentLab
    public = AutonomousInstrumentLab._read(lab)
    monkeypatch.setattr(AutonomousInstrumentLab, '_read', lambda self: deepcopy(public))
    def observe(celsius, seconds):
        public['observations']['reactor.indicated_k']['value'] = celsius + 273.15
        lab.lifecycle.elapsed = seconds
        lab._read()
    return observe


def test_warmup_enters_holding_once_and_cannot_relax_back(monkeypatch):
    lab = new_lab()
    observe = scripted_temperature_monitor(monkeypatch, lab)
    observe(33.9, 2)
    assert lab.termination_reason is None and lab.hold_started_s is None
    observe(34.0, 4)
    assert lab.termination_reason is None and lab.hold_started_s == 4
    assert lab.samples[-1]['temperature_phase'] == 'holding'
    observe(33.9, 6)
    assert lab.termination_reason == 'constraint_violated'
    assert lab.boundary['simulated_seconds'] == 6
    observe(35.0, 8)
    assert lab.termination_reason == 'constraint_violated'
    assert lab.boundary['simulated_seconds'] == 6 and lab.hold_started_s == 4


@pytest.mark.parametrize('temperature', [23.9, 36.1])
def test_warmup_global_boundaries_remain_enforced(monkeypatch, temperature):
    lab = new_lab()
    observe = scripted_temperature_monitor(monkeypatch, lab)
    observe(temperature, 2)
    assert lab.termination_reason == 'constraint_violated'
    assert lab.boundary['simulated_seconds'] == 2


def test_high_conversion_cannot_complete_below_target_temperature(monkeypatch):
    lab = new_lab()
    truth = lab._truth()
    truth['truth.composition_mol']['acetic_anhydride'] = lab.initial_anhydride_mol * 0.001
    truth['truth.composition_mol']['acetic_acid'] = lab.initial_acid_mol + 2 * lab.initial_anhydride_mol * 0.999
    monkeypatch.setattr(lab, '_truth', lambda: truth)
    observe = scripted_temperature_monitor(monkeypatch, lab)
    observe(25.0, 2)
    assert lab.termination_reason is None and lab.hold_started_s is None
    observe(35.0, 4)
    assert lab.termination_reason == 'task_completed'
    assert lab.boundary['simulated_seconds'] == 4


def test_holding_phase_upper_boundary_remains_enforced(monkeypatch):
    lab = new_lab()
    observe = scripted_temperature_monitor(monkeypatch, lab)
    observe(35.0, 2)
    observe(36.1, 4)
    assert lab.termination_reason == 'constraint_violated'
    assert lab.boundary['simulated_seconds'] == 4
