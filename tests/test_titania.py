"""Acquisition, causal controls, and evidence contracts for the titania substep."""
from copy import deepcopy
from uuid import uuid4

import pytest

from app.graph import build_graph, initial_state
from app.lab_adapter import GateRejected
from app.models import ActionProposal, Limits
from app.titania import TitaniaLab, TitaniaProfile, ScriptedTitaniaRoles, analyze
from app.trace import Trace


def proposal(lab, name, args=None, **overrides):
    fields = dict(sample_id=lab.sample_id, expected_revision=lab.revision, capability_id=name,
                  arguments=args or {}, request_id=str(uuid4()), postcondition='Acquire evidence.')
    return ActionProposal(**(fields | overrides))


def execute(lab, name, args=None):
    return lab.execute(proposal(lab, name, args))


@pytest.mark.parametrize('branch,actions', [
    ('thermal_history', ['workflow.inspect', 'thermal.prepare_reference', 'xrd.acquire']),
    ('scan_coverage', ['workflow.inspect', 'xrd.acquire'])])
def test_same_graph_differentiates_branches_and_acquires(branch, actions):
    lab = TitaniaLab(branch)
    assert not lab.initial_analysis['qualitative_goal_met']
    limits = Limits(lab_actions=8, graph_steps=120)
    trace = Trace(str(uuid4()))
    final = build_graph(lab, trace, limits, profile=TitaniaProfile(), roles=ScriptedTitaniaRoles()).invoke(
        initial_state(lab, trace.episode_id, limits),
        {'configurable': {'thread_id': trace.episode_id}, 'recursion_limit': 120})
    assert lab.score(final['report'])['task_success']
    assert [e['proposal']['capability_id'] for e in lab.history] == actions
    assert lab.last_scan_states[0]['scan.status'] == 'running'
    assert any(0 < s['scan.completed_points'] < 801 for s in lab.last_scan_states)
    assert lab.last_scan_states[-1]['scan.completed_points'] == 801
    assert lab.furnace.time_s == lab.xrd.time_s == lab.time_s
    assert lab._pattern['synthetic'] is True
    assert lab._pattern['synthetic_parts']
    assert all(r['completion']['program.phase'] == 'completed' and
               r['completion']['program.hold_elapsed_s'] == 7200 and
               r['transfer_temperature_k'] <= 313.15 and
               r['start_temperature']['thermal.indicated_temperature_k'] <= 298.65
               for r in lab.thermal_runs)
    public_inputs = [e['input'] for e in trace.events if e['kind'] == 'role_output']
    # No authored phase fractions, branch label or host sample truth in specialist inputs.
    import json
    encoded = json.dumps(public_inputs)
    for hidden in ['weight_fraction', 'template_sample_truth', '"branch":']:
        assert hidden not in encoded
    denied = lab.x.read(instance_id='xrd-001', state_ids=['template_sample_truth'])
    assert denied['error']['code'] == 'ACCESS_DENIED'
    m = lab._measurement
    optimistic = {'status': 'supported', 'evidence_ids': [m['id']]}
    for key, wrong in [('aliquot_id', 'wrong'), ('sample_id', 'wrong'), ('quality', 'unavailable'),
                       ('revision', -1), ('captured_at_step', lab.last_mutation_step)]:
        lab._measurement = deepcopy(m)
        lab._measurement[key] = wrong
        assert not lab.score(optimistic)['task_success']
    lab._measurement = m
    assert not lab.score({'status': 'supported', 'evidence_ids': []})['task_success']


@pytest.mark.parametrize('branch', ['thermal_history', 'scan_coverage'])
def test_wrong_repair_cannot_pass_even_with_optimistic_verifier(branch):
    lab = TitaniaLab(branch)
    execute(lab, 'workflow.inspect')
    if branch == 'scan_coverage':
        execute(lab, 'thermal.prepare_reference')
    execute(lab, 'xrd.acquire', {'stop_deg': 60 if branch == 'thermal_history' else 26})
    score = lab.score({'status': 'supported', 'evidence_ids': [lab._measurement['id']]})
    assert not score['task_success'] and score['false_success']
    assert not analyze(lab._pattern)['qualitative_goal_met']
    if branch == 'thermal_history':
        assert analyze(lab._pattern)['required_coverage']
        assert not analyze(lab._pattern)['matched_reference_degrees']['rutile']
    else:
        assert not analyze(lab._pattern)['required_coverage']


def test_fresh_rescan_cannot_excuse_unnecessary_synthesis():
    lab = TitaniaLab('scan_coverage')
    execute(lab, 'workflow.inspect')
    execute(lab, 'thermal.prepare_reference')
    execute(lab, 'xrd.acquire', {'stop_deg': 60})
    score = lab.score({'status': 'supported', 'evidence_ids': [lab._measurement['id']]})
    assert score['acquired_phase_evidence'] and not score['branch_policy_met']
    assert not score['task_success']


def test_timeout_produces_no_valid_measurement_and_stops_native_scan():
    lab = TitaniaLab('scan_coverage', scan_timeout_s=1)
    execute(lab, 'workflow.inspect')
    event = execute(lab, 'xrd.acquire', {'stop_deg': 60})
    assert event['result']['error']['code'] == 'XRD_SCAN_TIMEOUT_NO_VALID_MEASUREMENT'
    assert lab.observe()['measurement'] is None
    assert lab._read('xrd', ['scan.status'])['scan.status'] == 'stopped'
    assert not lab.score({'status': 'supported', 'evidence_ids': []})['task_success']


def test_identity_revision_and_idempotency_guard_native_side_effects():
    lab = TitaniaLab('thermal_history')
    p = proposal(lab, 'workflow.inspect')
    event = lab.execute(p)
    before = (lab.action_count, lab.time_s, len(lab.native_events))
    assert lab.execute(p) == event
    assert before == (lab.action_count, lab.time_s, len(lab.native_events))
    for overrides in [dict(sample_id='wrong'), dict(expected_revision=99), dict(arguments={'extra': True})]:
        with pytest.raises(GateRejected):
            lab.validate(proposal(lab, 'workflow.inspect', **overrides))
    with pytest.raises(GateRejected):
        lab.validate(p.model_copy(update={'postcondition': 'Conflicting replay.'}))
