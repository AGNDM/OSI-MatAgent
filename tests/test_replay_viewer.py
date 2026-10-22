import json
from pathlib import Path

import pytest

from app.replay_viewer import build_replay, response_for


def fixture_run(tmp_path, events, monitor=()):
    run = tmp_path / 'trial'
    run.mkdir()
    (run / 'result.json').write_text(json.dumps({'selection': {'selected_device': 'mt_rc1mxhfcal'}}))
    for i, event in enumerate(events):
        event.setdefault('timestamp', f'2026-01-01T00:00:{i:02}Z')
    (run / 'trace.jsonl').write_text('\n'.join(map(json.dumps, events)))
    if monitor:
        (run / 'monitor.jsonl').write_text('\n'.join(map(json.dumps, monitor)))
    return run


def read_event(seconds=0, slots=None, revision=0, mid='read1'):
    return {'kind': 'tool_result', 'component': 'execute', 'revision': revision,
            'proposal': {'capability_id': 'workflow.readback'},
            'result': {'measurement': {'id': mid, 'revision': revision, 'native': {
                'clock': {'time_s': seconds}, 'observations': slots or {
                    'reactor.indicated_k': {'value': 298.15, 'sample_time_s': seconds, 'unit': 'K'}}}}}}


def model_event():
    return {'kind': 'role_output', 'component': 'diagnostic',
            'llm_metadata': {'request_sent': True, 'model_id': 'deepseek/test',
                             'request': {'messages': [{'role': 'user', 'content': 'Read temperature?'}]}}}


def test_wait_advances_truth_but_does_not_refresh_agent_reading(tmp_path):
    fixture_run(tmp_path, [read_event(), {'kind': 'tool_result', 'component': 'execute', 'revision': 1,
                'proposal': {'capability_id': 'workflow.await'}, 'result': {'advanced_seconds': 120}}, model_event()],
                [{'time_s': 0, 'true_reactor_c': 25}, {'time_s': 120, 'true_reactor_c': 32},
                 {'time_s': 122, 'true_reactor_c': 33}])
    replay = build_replay(tmp_path, 'trial')
    assert [s['time_s'] for s in replay['steps']] == [0, 120, 120]
    assert replay['steps'][-1]['monitor']['true_reactor_c'] == 32
    slot = replay['steps'][-1]['readings']['reactor.indicated_k']
    assert slot['value'] == 298.15 and slot['time_s'] == 0 and slot['evidence_id'] == 'read1'


def test_partial_reads_preserve_each_fields_actual_age(tmp_path):
    fixture_run(tmp_path, [read_event(), read_event(120, {'stirrer.indicated_rpm': {'value': 400,
                           'sample_time_s': 120}}, revision=1, mid='read2')])
    step = build_replay(tmp_path, 'trial')['steps'][-1]
    assert step['readings']['reactor.indicated_k']['time_s'] == 0
    assert step['readings']['stirrer.indicated_rpm']['time_s'] == 120
    assert step['monitor'] is None


def test_future_same_time_control_state_is_not_used_and_failed_command_has_no_effect(tmp_path):
    def start(status):
        return {'kind': 'tool_result', 'component': 'execute', 'proposal': {'capability_id': 'mt_rc1mxhfcal::experiment.start'},
                'result': {'native_response': {'status': status, 'clock': {'time_s': 0}}}}
    fixture_run(tmp_path, [model_event(), start('failed'), start('succeeded')],
                [{'time_s': 0, 'true_reactor_c': 25, 'experiment_running': False},
                 {'time_s': 0, 'true_reactor_c': 26, 'experiment_running': True}])
    steps = build_replay(tmp_path, 'trial')['steps']
    assert steps[0]['monitor']['true_reactor_c'] == 25
    assert 'experiment.running' not in steps[1]['controls']
    assert steps[2]['controls']['experiment.running']['value'] is True


def test_visual_state_comes_from_receipt_not_proposed_action(tmp_path):
    event = model_event()
    event['output'] = {'capability_id': 'mt_rc1mxhfcal::stirrer.set', 'arguments': {'rpm': 400}}
    fixture_run(tmp_path, [event])
    assert build_replay(tmp_path, 'trial')['steps'][0]['controls'] == {}


def test_details_preserve_original_request_and_redact_secrets(tmp_path):
    event = model_event()
    event['llm_metadata']['request']['messages'][0]['content'] += ' Bearer private-token'
    fixture_run(tmp_path, [event])
    detail = build_replay(tmp_path, 'trial', detail_index=0)
    text = detail['event']['llm_metadata']['request']['messages'][0]['content']
    assert text == 'Read temperature? [redacted]'
    with pytest.raises(FileNotFoundError):
        build_replay(tmp_path, 'trial', detail_index=1)


def test_routes_are_allowlisted_and_leave_original_viewer_routes_alone(tmp_path):
    fixture_run(tmp_path, [model_event()])
    assert response_for('/', tmp_path, tmp_path, tmp_path) is None
    assert response_for('/viewer.js', tmp_path, tmp_path, tmp_path) is None
    for path in ['/replay/../.env', '/replay/vendor/three/../../.env', '/api/replay/run/../.env']:
        assert response_for(path, tmp_path, tmp_path, tmp_path)[0] in (400, 404)
    code, _, body = response_for('/api/replay/run/trial', tmp_path, tmp_path, tmp_path)
    assert code == 200 and json.loads(body)['supported_model'] is True


def test_final_boundary_uses_actual_advance_not_requested_wait(tmp_path):
    fixture_run(tmp_path, [read_event(), {'kind': 'tool_result', 'component': 'execute',
                'proposal': {'capability_id': 'workflow.await', 'arguments': {'duration_s': 120}},
                'result': {'advanced_seconds': 18}}, {'kind': 'agent_report', 'component': 'report'}],
                [{'time_s': 0, 'true_reactor_c': 25}, {'time_s': 18, 'true_reactor_c': 36.1}])
    steps = build_replay(tmp_path, 'trial')['steps']
    assert steps[-1]['time_s'] == 18 and steps[-1]['monitor']['true_reactor_c'] == 36.1


def test_observation_duplicate_does_not_overwrite_successful_receipt(tmp_path):
    read = read_event(slots={'experiment.running': {'value': False, 'sample_time_s': 0}})
    fixture_run(tmp_path, [read, {'kind': 'tool_result', 'component': 'execute',
                 'proposal': {'capability_id': 'mt_rc1mxhfcal::experiment.start'},
                 'result': {'native_response': {'status': 'succeeded'}}},
                 {'kind': 'observation', 'observations': {'measurement': read['result']['measurement']}}, model_event()])
    assert build_replay(tmp_path, 'trial')['steps'][-1]['controls']['experiment.running']['value'] is True


def test_ramp_target_is_separate_from_starting_setpoint_and_ignores_rejection(tmp_path):
    def ramp(target, status):
        return {'kind': 'tool_result', 'component': 'execute',
                'proposal': {'capability_id': 'mt_rc1mxhfcal::thermostat.set_tj',
                             'arguments': {'end_k': target, 'ramp_s': 600}},
                'result': {'native_response': {'status': status,
                           'result': {'tj_setpoint_k': 298.15}}}}
    fixture_run(tmp_path, [ramp(308.15, 'succeeded'), ramp(323.15, 'failed')])
    controls = build_replay(tmp_path, 'trial')['steps'][-1]['controls']
    assert controls['thermostat.tj_setpoint_k']['value'] == 298.15
    assert controls['thermostat.tj_target_k']['value'] == 308.15
    assert controls['thermostat.mode']['value'] == 'tj'
