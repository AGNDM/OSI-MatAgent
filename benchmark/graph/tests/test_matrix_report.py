"""Matrix/report contracts. Run in the actual Slurm validation allocation."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    pytest.skip('Report computation requires Slurm', allow_module_level=True)

from graph_local.reporting import matrix_report as report


def summary(*, completed=120, attempted=120, goals=60):
    return {'n': 120, 'unique_cases': 120, 'completed': completed,
        'coverage': completed / 120, 'attempt_coverage': attempted / 120,
        'missing': 120 - attempted, 'infrastructure_errors': attempted - completed,
        'cohort_goal_completion': goals / 120, 'goal_completion_ci95': {'low': .2, 'high': .7},
        'observed_control_accuracy_micro': .8 if completed else None,
        'observed_control_accuracy_episode_macro': .7 if completed else None,
        'control_accuracy_ci95': {'low': .6, 'high': .9}, 'control_decisions_observed': completed * 2,
        'interventions': 3 if completed else 0, 'breakdowns': {}}


def configuration():
    # Names deliberately do not encode either axis.
    return {'analysis': {'matrix_rows': ['Expert-X', 'Expert-Y'], 'matrix_columns': ['Chooser-P', 'Chooser-Q']},
        'experiments': [{'name': 'opaque-' + str(i), 'expert_row': row, 'chooser_column': column,
                         'architecture': 'repository_langgraph', 'seed': 0}
                        for i, (row, column) in enumerate((r, c) for r in ['Expert-X', 'Expert-Y'] for c in ['Chooser-P', 'Chooser-Q'])]}


def combined(config=None):
    config = config or configuration()
    return {'experiments': [{'name': spec['name'], 'specification': spec,
                            'sequential': {mode: summary() for mode in report.MODES}, 'sources': []}
                           for spec in config['experiments']]}


def test_matrix_axes_come_only_from_explicit_frozen_metadata():
    matrix = report.build_matrix(configuration(), combined())
    assert matrix['rows'] == ['Expert-X', 'Expert-Y']
    assert matrix['columns'] == ['Chooser-P', 'Chooser-Q']
    assert matrix['cells'][0]['name'] == 'opaque-0'
    assert matrix['cells'][0]['modes']['free_run']['goals_verified'] == 60
    assert matrix['cells'][0]['modes']['free_run']['goal_completion_ci95'] == {'low': .2, 'high': .7}


def test_aborted_workflows_have_zero_verified_goals_but_no_invented_decision_accuracy():
    value = combined()
    value['experiments'][0]['sequential']['teacher_forced'] = summary(completed=0, attempted=120, goals=0)
    value['experiments'][0]['sequential']['free_run'] = summary(completed=0, attempted=120, goals=0)
    value['experiments'] = value['experiments'][:1]
    matrix = report.build_matrix(configuration(), value)
    unsupported = matrix['cells'][0]['modes']['teacher_forced']
    pending = matrix['cells'][1]['modes']['free_run']
    assert unsupported['control_accuracy_display'] is None
    assert unsupported['goal_rate_display'] == 0 and unsupported['aborted'] == 120
    assert matrix['cells'][0]['modes']['free_run']['goal_rate_display'] == 0
    assert pending['attempted'] == 0 and pending['unattempted'] == 120
    assert pending['goal_rate_display'] is None


@pytest.mark.parametrize('mutation', ['duplicate', 'axis_missing', 'denominator', 'specification'])
def test_incompatible_matrix_or_report_rejected(mutation):
    config, value = configuration(), combined()
    if mutation == 'duplicate':
        config['experiments'][1]['chooser_column'] = config['experiments'][0]['chooser_column']
    elif mutation == 'axis_missing':
        del config['experiments'][0]['expert_row']
    elif mutation == 'denominator':
        value['experiments'][0]['sequential']['free_run']['n'] = 119
    else:
        value['experiments'][0]['specification'] = {**value['experiments'][0]['specification'], 'seed': 99}
    with pytest.raises(ValueError):
        report.build_matrix(config, value)


def event(index=1, *, kind='role_output', usage=None, elapsed=123., error=None):
    metadata = {'backend_id': 'fixture-only', 'request_sent': True, 'request': {'messages': 'fixture-public-only'},
                'elapsed_ms': elapsed, 'usage': usage if usage is not None else {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}}
    return {'event_id': 'episode:' + str(index), 'component': 'diagnostic', 'kind': kind,
            'llm_metadata': metadata, **({'error': error} if error else {})}


def ledger(events):
    return report.call_ledger(events, experiment='fixture', task_id='case-1', mode='free_run')


def test_call_events_deduplicate_mirrored_records_without_collapsing_real_repeats():
    first = event()
    mirror = {**deepcopy(first), 'event_id': 'episode:2', 'kind': 'role_error'}
    second_real = event(3, elapsed=123.0007)  # Same prompt/answer, separately measured real attempt.
    rows, _ = ledger([first, deepcopy(first), mirror, second_real])
    assert len(rows) == 2
    assert report.summarize_calls(rows)['total_input_tokens'] == 20


def test_one_event_with_both_metadata_views_is_one_actual_call():
    value = event(kind='operation_selected')
    value['backend_metadata'] = deepcopy(value['llm_metadata'])
    rows, _ = ledger([value])
    assert len(rows) == 1


def test_missing_usage_and_timing_remain_unknown_with_known_subtotals():
    first = event()
    unknown = event(2, usage={}, elapsed=None)
    rows, _ = ledger([first, unknown])
    value = report.summarize_calls(rows)
    assert value['calls'] == 2 and value['usage_coverage'] == .5
    assert value['known_input_tokens'] == 10 and value['total_input_tokens'] is None
    assert value['known_model_call_wall_s'] == .123 and value['total_model_call_wall_s'] is None
    assert report.summarize_calls([])['known_input_tokens'] is None


@pytest.mark.parametrize('message,category', [
    ('EvidenceReferenceError: unknown observation id', 'model_invalid_citation'),
    ('ValueError: Action is absent from the complete public native menu', 'model_invalid_preset'),
    ('expert_output_truncated', 'model_output_token_limit'),
    ('JSONDecodeError: Expecting value', 'model_json_schema'),
    ('Actual LangGraph stopped: llm_budget', 'model_call_budget'),
    ('Episode deadline expired', 'episode_deadline'),
    ('ContextTooLong: state is 8900 tokens, limit 4096', 'unsupported_context'),
    ('BrokenPipeError: Local APUS IPC failure', 'runtime_provider_ipc'),
])
def test_abort_causes_distinguish_model_outputs_from_context_and_transport(message, category):
    assert report.classify_abort(message) == category


def test_failed_format_attempt_is_counted_but_not_automatically_an_aborted_episode():
    rows, errors = ledger([event(kind='role_error', error='JSONDecodeError: invalid response')])
    assert rows[0]['error_category'] == 'model_json_schema'
    assert errors[0]['category'] == 'model_json_schema'
    assert report.summarize_calls(rows)['failed_call_categories'] == {'model_json_schema': 1}


def test_conflicting_reuse_of_one_event_id_is_rejected():
    first, conflict = event(), event()
    conflict['llm_metadata']['elapsed_ms'] = 999
    with pytest.raises(ValueError, match='conflicting'):
        ledger([first, conflict])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(report.canonical(row) + '\n' for row in rows))


def create_small_full_cohort_fixture(tmp_path):
    study = tmp_path / 'frozen-study'
    bundle = study / 'snapshot/instrument'
    (bundle / 'sequential_120').mkdir(parents=True)
    report.write_json(bundle / 'manifest.json', {'test_fixture': True})
    cases = [{'task_id': 'case-' + str(i)} for i in range(120)]
    write_jsonl(bundle / 'sequential_120/cases.jsonl', cases)
    spec = {'name': 'opaque-one', 'expert_row': 'Expert-X', 'chooser_column': 'Chooser-P',
            'architecture': 'repository_langgraph', 'seed': 0}
    config = {'analysis': {'matrix_rows': ['Expert-X'], 'matrix_columns': ['Chooser-P']}, 'experiments': [spec]}
    report.write_json(study / 'configuration.json', config)
    report.write_json(study / 'jobs.json', [{'experiment': spec['name'], 'shard': 0}])
    manifest = {path.relative_to(study / 'snapshot').as_posix(): report.digest(path)
                for path in (study / 'snapshot').rglob('*') if path.is_file()}
    report.write_json(study / 'source-manifest.json', manifest)
    report.write_json(study / 'prepared.json', {
        'configuration_sha256': report.digest(study / 'configuration.json'),
        'jobs_sha256': report.digest(study / 'jobs.json'),
        'source_manifest_sha256': report.digest(study / 'source-manifest.json')})
    source = study / 'runs/shard-000'
    data = source / spec['name'] / 'sequential'
    episodes = [{'task_id': 'case-0', 'mode': mode, 'status': 'ok', 'trajectory_outcome_met': True,
                 'decision_records': [{'decision_index': 1, 'proposal_correct': True, 'auxiliary_query': False,
                                       'corrected': False, 'checkpoint_id': 'fixture', 'stage': 'operation'}]}
                for mode in report.MODES]
    write_jsonl(data / 'episodes.jsonl', episodes)
    for mode in report.MODES:
        trace = data / 'steps' / hashlib.sha256(b'case-0').hexdigest()[:24] / mode / 'graph.jsonl'
        write_jsonl(trace, [event()])
    value = {'bundle_manifest_sha256': report.digest(bundle / 'manifest.json'), 'experiments': [
        {'name': spec['name'], 'specification': spec, 'sources': [str(source)],
         'sequential': {mode: summary(completed=1, attempted=1, goals=1) for mode in report.MODES}}]}
    value_path = tmp_path / 'combined-report.json'
    report.write_json(value_path, value)
    return study, value_path, value


def test_raw_trace_join_preserves_original_cohort_and_source_binding(tmp_path):
    study, path, value = create_small_full_cohort_fixture(tmp_path)
    config, cases = report.authenticate(study, value)
    matrix = report.build_matrix(config, value)
    calls, episodes, files = report.attach_ledgers(matrix, value, study, cases, [])
    assert len(calls) == 2 and len(episodes) == 2
    assert len(files) == 3
    assert matrix['cells'][0]['modes']['free_run']['goals_verified'] == 1
    assert matrix['cells'][0]['modes']['free_run']['unattempted'] == 119
    with (study / 'configuration.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='configuration'):
        report.authenticate(study, value)


def test_distinct_frozen_studies_append_explicit_expert_rows(tmp_path):
    first, _, first_report = create_small_full_cohort_fixture(tmp_path / 'first')
    second, _, second_report = create_small_full_cohort_fixture(tmp_path / 'second')
    config = json.loads((second / 'configuration.json').read_text())
    config['analysis']['matrix_rows'] = ['Stronger-Expert-Z']
    config['experiments'][0].update(name='different-opaque-cell', expert_row='Stronger-Expert-Z')
    report.write_json(second / 'configuration.json', config)
    prepared = json.loads((second / 'prepared.json').read_text())
    prepared['configuration_sha256'] = report.digest(second / 'configuration.json')
    report.write_json(second / 'prepared.json', prepared)
    second_report['experiments'][0].update(name='different-opaque-cell', specification=config['experiments'][0])
    value = {**first_report, 'experiments': first_report['experiments'] + second_report['experiments']}
    matrix, cases, provenance = report.join_studies([first, second], value)
    assert matrix['rows'] == ['Expert-X', 'Stronger-Expert-Z']
    assert matrix['columns'] == ['Chooser-P']
    assert len(matrix['cells']) == 2 and len(cases) == 120
    assert matrix['cells'][1]['study'] == str(second)
    assert len(provenance) == 2


def test_two_versions_of_one_expert_row_cannot_be_silently_pooled(tmp_path):
    first, _, value = create_small_full_cohort_fixture(tmp_path / 'first')
    second, _, _ = create_small_full_cohort_fixture(tmp_path / 'second')
    with pytest.raises(ValueError, match='repeat an expert row'):
        report.join_studies([first, second], value)


def test_rendered_artifacts_are_standalone_and_keep_micro_and_macro_intervals_separate(tmp_path):
    study, path, value = create_small_full_cohort_fixture(tmp_path)
    output = tmp_path / 'matrix-output'
    assert report.main(['--study', str(study), '--report', str(path), '--output', str(output)]) == 0
    page = (output / 'index.html').read_text()
    assert 'data:image/png;base64,' in page
    assert '58.3%' in page and 'not a paired comparison' in page
    assert 'Episode mean 70.0%' in page
    assert 'not the micro percentage' in page
    assert len(list((output / 'plots').glob('*.pdf'))) == 3
    assert len(list((output / 'plots').glob('*.svg'))) == 3
    assert len(list((output / 'plots').glob('*.png'))) == 3
    matrix = json.loads((output / 'matrix.json').read_text())
    assert matrix['all_expected_outputs_present'] is False
    assert matrix['call_accounting']['total_input_tokens'] == 20
    assert (output / 'call-ledger.jsonl').read_text().count('\n') == 2


def test_authentic_partial_controls_have_separate_exact_denominator_without_terminal_outcome(tmp_path):
    study, path, value = create_small_full_cohort_fixture(tmp_path)
    entry = value['experiments'][0]
    data = Path(entry['sources'][0]) / entry['name'] / 'sequential'
    episodes = [json.loads(line) for line in (data / 'episodes.jsonl').read_text().splitlines()]
    episodes[0]['decision_records'].append({'decision_index': 2, 'proposal_correct': False,
        'auxiliary_query': False, 'corrected': False, 'checkpoint_id': 'fixture-2', 'stage': 'operation'})
    episodes.append({'task_id': 'case-1', 'mode': 'free_run', 'status': 'infrastructure_error',
                     'error': 'Expert JSONDecodeError: invalid model JSON',
                     'partial_native_decisions_observed': 2})
    write_jsonl(data / 'episodes.jsonl', episodes)
    partial_dir = data / 'steps' / hashlib.sha256(b'case-1').hexdigest()[:24] / 'free_run'
    partial_dir.mkdir(parents=True)
    for index, auxiliary, correct in [(1, True, True), (2, False, False)]:
        report.write_json(partial_dir / (f'{index:04d}.json'), {'decision_index': index,
            'auxiliary_query': auxiliary, 'proposal_correct': correct, 'corrected': False})
    entry['sequential']['free_run'] = summary(completed=1, attempted=2, goals=1)
    config, cases = report.authenticate(study, value)
    matrix = report.build_matrix(config, value)
    _, diagnostics, hashes = report.attach_ledgers(matrix, value, study, cases, [])
    metric = matrix['cells'][0]['modes']['free_run']
    assert metric['all_observed_native_control_correct'] == 1
    assert metric['all_observed_native_control_count'] == 3
    assert metric['all_observed_control_accuracy_micro'] == 1 / 3
    assert metric['partial_native_control_count'] == 1
    assert metric['partial_native_control_correct'] == 0
    assert metric['cohort_goal_completion'] == 1 / 120
    aborted = next(row for row in diagnostics if row['task_id'] == 'case-1')
    assert aborted['primary_abort_category'] == 'model_json_schema'
    assert aborted['observed_native_grades'][1]['proposal_correct'] is False
    assert 'trajectory_outcome_met' not in aborted
    assert str(partial_dir / '0002.json') in hashes


def test_raw_native_coverage_must_agree_with_combined_report_completion(tmp_path):
    study, path, value = create_small_full_cohort_fixture(tmp_path)
    value['experiments'][0]['sequential']['free_run'] = summary(completed=120, attempted=120, goals=60)
    config, cases = report.authenticate(study, value)
    matrix = report.build_matrix(config, value)
    warnings = []
    report.attach_ledgers(matrix, value, study, cases, warnings)
    metric = matrix['cells'][0]['modes']['free_run']
    assert metric['raw_attempted'] == 1 and metric['raw_completed'] == 1
    assert metric['source_summary_consistent'] is False
    assert warnings


def test_slurm_guard_is_preserved(monkeypatch):
    monkeypatch.delenv('SLURMD_NODENAME', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        report.require_slurm()
