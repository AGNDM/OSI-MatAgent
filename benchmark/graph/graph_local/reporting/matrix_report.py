"""Auditable expert-by-chooser report for the actual local repository graph.

This reader never executes a policy or regrades a native action. It joins the
frozen explicit matrix with comparison.study_report summaries and actual graph
call ledgers. Run rendering, parsing and validation inside Slurm.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path


VERSION = 'actual-langgraph-expert-chooser-matrix-v1'
MODES = ('free_run', 'teacher_forced')
TERMINAL_CALL_EVENTS = {'role_output', 'role_error', 'routing', 'operation_selected', 'operation_error'}
ABORT_LABELS = {
    'model_invalid_citation': 'Model citation / evidence contract',
    'model_invalid_preset': 'Model action / preset contract',
    'model_output_token_limit': 'Model output token limit',
    'model_json_schema': 'Model JSON / schema',
    'model_call_budget': 'Model call budget',
    'episode_deadline': 'Episode or inference deadline',
    'unsupported_context': 'Unsupported full context',
    'runtime_provider_ipc': 'Runtime / process / provider',
    'other_policy_abort': 'Other graph abort',
}
MODEL_RESPONSE_ABORTS = {'model_invalid_citation', 'model_invalid_preset', 'model_output_token_limit', 'model_json_schema'}


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Matrix analysis and rendering require a genuine Slurm allocation')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def authenticate(study, report):
    """Bind the frozen plan and source bytes before interpreting any scores."""
    prepared = json.loads((study / 'prepared.json').read_text())
    for key, filename in [('configuration_sha256', 'configuration.json'), ('jobs_sha256', 'jobs.json'),
                          ('source_manifest_sha256', 'source-manifest.json')]:
        if prepared.get(key) != digest(study / filename):
            raise ValueError('Frozen matrix study changed: ' + filename)
    manifest = json.loads((study / 'source-manifest.json').read_text())
    snapshot = study / 'snapshot'
    actual = {}
    for path in snapshot.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            actual[path.relative_to(snapshot).as_posix()] = digest(path)
    if actual != manifest:
        raise ValueError('Frozen matrix source inventory differs from its manifest')
    bundle_manifest = snapshot / 'instrument/manifest.json'
    if report.get('bundle_manifest_sha256') != digest(bundle_manifest):
        raise ValueError('Combined report describes a different instrument benchmark release')
    public_cases = [json.loads(line) for line in (snapshot / 'instrument/sequential_120/cases.jsonl').read_text().splitlines() if line.strip()]
    if len(public_cases) != 120 or len({case['task_id'] for case in public_cases}) != 120:
        raise ValueError('Actual graph matrix must retain the original 120-case cohort')
    return json.loads((study / 'configuration.json').read_text()), public_cases


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def classify_abort(value):
    """Descriptive error taxonomy; never converts an abort into a native grade."""
    text = (value if isinstance(value, str) else canonical(value)).lower()
    if any(term in text for term in ('exceeds context', 'context limit', 'exceeds runtime limit', 'maximum context',
                                     'contexttoolong', 'max_state', 'evidence was not truncated', 'input too long',
                                     'input is too long', 'state is too long', 'tokens, limit')):
        return 'unsupported_context'
    if any(term in text for term in ('brokenpipe', 'broken pipe', 'ipc ', 'ipc_', 'subprocess', 'provider_error',
                                     'provider returned', 'modulenotfound', 'importerror', 'out of memory',
                                     'cuda error', 'cuda unavailable', 'process startup', 'python prefix', 'model_load_error')):
        return 'runtime_provider_ipc'
    if any(term in text for term in ('expert_output_truncated', 'output-token limit', 'output token limit', 'max_new_tokens')):
        return 'model_output_token_limit'
    if any(term in text for term in ('llm_budget', 'model_call_budget', 'model call budget', 'token_budget_exceeded')):
        return 'model_call_budget'
    if any(term in text for term in ('deadline', 'timed out', 'timeouterror', 'episode_seconds')):
        return 'episode_deadline'
    if any(term in text for term in ('evidencereferenceerror', 'unknown evidence', 'unknown observation',
                                     'completion claim must cite', 'invalid evidence', 'unsupported citation')):
        return 'model_invalid_citation'
    if any(term in text for term in ('absent from the complete public native menu', 'absent from the public menu',
                                     'procedure changed', 'capability alias', 'intent arguments must',
                                     'must copy', 'preset', 'not offered', 'not a native action', 'invalid public native action')):
        return 'model_invalid_preset'
    if any(term in text for term in ('expert_schema_error', 'validationerror', 'jsondecodeerror', 'invalid json',
                                     'json object', 'duplicate json', 'json/schema', 'expert_empty_output',
                                     'schema error', 'field required', 'extra_forbidden', 'json_invalid')):
        return 'model_json_schema'
    return 'other_policy_abort'


def _iter_jsonl(path, warnings):
    with path.open() as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):
                    warnings.append('Unfinished live JSON line excluded: ' + str(path) + ':' + str(index))
                    return
                raise


def call_ledger(events, *, experiment, task_id, mode):
    """One row per actual terminal model-call event, including failed calls.

    Both metadata views on an operation event refer to one call. Explicit
    provider/process request IDs take precedence; otherwise the complete call
    metadata (which includes measured elapsed_ms) identifies mirrored events.
    Identical deterministic prompts on separate real calls are not collapsed.
    """
    seen_events, seen_calls, rows, errors = {}, set(), [], []
    for event in events:
        event_id = event.get('event_id')
        if event_id:
            signature = fingerprint(event)
            if event_id in seen_events:
                if seen_events[event_id] != signature:
                    raise ValueError('A graph event ID refers to conflicting call records')
                continue
            seen_events[event_id] = signature
        if event.get('error'):
            errors.append({'event_id': event_id, 'kind': event.get('kind'),
                           'category': classify_abort(event['error']), 'error': str(event['error'])[:2000]})
        if event.get('kind') not in TERMINAL_CALL_EVENTS:
            continue
        metadata = event.get('llm_metadata') or event.get('backend_metadata') or {}
        if metadata.get('request_sent') is not True:
            continue
        response = metadata.get('response') or {}
        result_metadata = response.get('metadata', {}) if isinstance(response, dict) else {}
        request_id = metadata.get('call_id') or metadata.get('request_id') or result_metadata.get('subprocess_request_id')
        if not request_id and isinstance(response, dict):
            request_id = response.get('id')
        call_key = ('request', str(request_id)) if request_id else ('metadata', fingerprint(metadata))
        if call_key in seen_calls:
            continue
        seen_calls.add(call_key)
        usage = metadata.get('usage') or {}
        input_tokens = usage.get('prompt_tokens') if _count(usage.get('prompt_tokens')) else None
        output_tokens = usage.get('completion_tokens') if _count(usage.get('completion_tokens')) else None
        total_tokens = usage.get('total_tokens') if _count(usage.get('total_tokens')) else None
        if input_tokens is not None and output_tokens is not None and total_tokens != input_tokens + output_tokens:
            # Preserve individual known counters, but do not claim a complete
            # provider usage record when its accounting identity is invalid.
            total_tokens = None
        elapsed = metadata.get('elapsed_ms')
        elapsed_s = float(elapsed) / 1000 if _number(elapsed) and elapsed >= 0 else None
        error = event.get('error') or metadata.get('error')
        rows.append({'experiment': experiment, 'task_id': task_id, 'mode': mode,
            'event_id': event_id, 'call_id': str(request_id) if request_id else None,
            'call_metadata_sha256': fingerprint(metadata), 'component': event.get('component'),
            'kind': event.get('kind'), 'backend_id': metadata.get('backend_id', metadata.get('model_id')),
            'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens,
            'usage_complete': total_tokens is not None and input_tokens is not None and output_tokens is not None,
            'model_call_wall_s': elapsed_s, 'error': str(error)[:2000] if error else None,
            'error_category': classify_abort(error) if error else None,
            'response_classification': metadata.get('response_classification')})
    return rows, errors


def summarize_calls(rows):
    n = len(rows)
    result = {'calls': n, 'calls_with_complete_usage': sum(row['usage_complete'] for row in rows),
              'usage_coverage': sum(row['usage_complete'] for row in rows) / n if n else None,
              'failed_call_categories': dict(Counter(row['error_category'] for row in rows if row['error_category'])),
              'component_calls': dict(Counter(str(row.get('component')) for row in rows))}
    for field in ('input_tokens', 'output_tokens', 'model_call_wall_s'):
        known = [row[field] for row in rows if row[field] is not None]
        result['known_' + field] = sum(known) if known else None
        result['total_' + field] = sum(known) if n and len(known) == n else None
        result[field + '_known_calls'] = len(known)
    return result


def read_raw(experiment, study, valid_ids, warnings):
    episodes, ledgers, trace_errors, files = {}, [], {}, {}
    roots = experiment.get('sources', [])
    for source in roots:
        source = Path(source).resolve()
        if not source.is_relative_to((study / 'runs').resolve()):
            raise ValueError('Matrix raw source is outside the declared full study runs: ' + str(source))
        data = source / experiment['name'] / 'sequential/episodes.jsonl'
        if not data.is_file():
            continue
        files[str(data)] = digest(data)
        for episode in _iter_jsonl(data, warnings):
            key = (episode.get('task_id'), episode.get('mode'))
            if key[0] not in valid_ids or key[1] not in MODES:
                raise ValueError('Unknown task or mode in raw graph cohort')
            if key in episodes:
                raise ValueError('Duplicate raw graph episode for a matrix cell')
            episodes[key] = episode
            trace = data.parent / 'steps' / hashlib.sha256(key[0].encode()).hexdigest()[:24] / key[1] / 'graph.jsonl'
            if not trace.is_file():
                warnings.append('Graph trace unavailable: ' + str(trace))
                trace_errors[key] = []
                continue
            files[str(trace)] = digest(trace)
            calls, errors = call_ledger(_iter_jsonl(trace, warnings), experiment=experiment['name'], task_id=key[0], mode=key[1])
            ledgers.extend(calls)
            trace_errors[key] = errors
    diagnostics = []
    for (task_id, mode), episode in episodes.items():
        aborted = episode.get('status') != 'ok'
        records = episode.get('decision_records', []) if not aborted else []
        record_source = 'completed_native_episode'
        if aborted:
            record_source = 'partial_native_step_files'
            # Preserve authentic native grades from on_step artifacts.  No
            # missing action or terminal native outcome is reconstructed.
            for source in roots:
                data = Path(source).resolve() / experiment['name'] / 'sequential'
                partial_dir = data / 'steps' / hashlib.sha256(task_id.encode()).hexdigest()[:24] / mode
                for path in sorted(partial_dir.glob('*.json')):
                    if path.stem.isdigit():
                        records.append(json.loads(path.read_text()))
                        files[str(path)] = digest(path)
        grades = []
        for index, record in enumerate(records, 1):
            if (record.get('decision_index') != index or type(record.get('proposal_correct')) is not bool
                    or type(record.get('auxiliary_query')) is not bool):
                raise ValueError('Noncontiguous or untyped authentic native decision records')
            grades.append({key: record.get(key) for key in
                ('decision_index', 'proposal_correct', 'auxiliary_query', 'corrected', 'checkpoint_id', 'stage')})
        controls = [record for record in grades if not record['auxiliary_query']]
        evidence = [episode.get('error', ''), episode.get('graph_terminal_reason', ''),
                    episode.get('model_response_classifications', []), episode.get('recorded_policy_errors', [])]
        categories = Counter(error['category'] for error in trace_errors.get((task_id, mode), []))
        primary = classify_abort(evidence) if aborted else None
        if aborted and primary == 'other_policy_abort' and categories:
            # The last native/graph failure is the terminal evidence; earlier
            # repaired model-format errors are retained separately below.
            primary = trace_errors[(task_id, mode)][-1]['category']
        diagnostics.append({'experiment': experiment['name'], 'task_id': task_id, 'mode': mode,
            'native_status': episode.get('status'), 'aborted': aborted, 'primary_abort_category': primary,
            'model_response_abort': primary in MODEL_RESPONSE_ABORTS if primary else False,
            'observed_error_event_categories': dict(categories),
            'native_partial_decisions': episode.get('partial_native_decisions_observed',
                episode.get('partial_decisions_observed', 0)) if aborted else None,
            'native_partial_correct_original_proposals': episode.get('partial_native_correct_original_proposals') if aborted else None,
            'native_record_source': record_source, 'observed_native_grades': grades,
            'observed_native_control_count': len(controls),
            'observed_native_control_correct': sum(record['proposal_correct'] for record in controls),
            'partial_records_are_completed_episode_metrics': False,
            'graph_terminal_reason': episode.get('graph_terminal_reason'),
            'model_response_classifications': episode.get('model_response_classifications', []),
            'error': str(episode.get('error', ''))[:3000] if aborted else None})
    return episodes, ledgers, diagnostics, files


def _empty_summary(n):
    return {'n': n, 'unique_cases': n, 'completed': 0, 'coverage': 0., 'attempt_coverage': 0.,
            'missing': n, 'infrastructure_errors': 0, 'cohort_goal_completion': 0.,
            'observed_control_accuracy_micro': None, 'observed_control_accuracy_episode_macro': None,
            'control_decisions_observed': 0, 'interventions': 0, 'breakdowns': {}}


def build_matrix(config, report, *, cohort_size=120):
    """Join cells using declared metadata only; never parse names into axes."""
    specifications = config.get('experiments', [])
    rows = config.get('analysis', {}).get('matrix_rows', [])
    columns = config.get('analysis', {}).get('matrix_columns', [])
    if not rows or not columns or len(rows) != len(set(rows)) or len(columns) != len(set(columns)):
        raise ValueError('Frozen configuration must declare distinct matrix rows and columns')
    index = {}
    for entry in report.get('experiments', []):
        if entry.get('name') in index:
            raise ValueError('Combined report repeats an experiment name')
        index[entry.get('name')] = entry
    seen, cells = set(), []
    for specification in specifications:
        axis = (specification.get('expert_row'), specification.get('chooser_column'))
        if axis[0] not in rows or axis[1] not in columns or axis in seen:
            raise ValueError('Missing, duplicate or undeclared explicit matrix coordinates')
        seen.add(axis)
        name = specification['name']
        entry = index.get(name)
        if entry and entry.get('specification') is not None and fingerprint(entry['specification']) != fingerprint(specification):
            raise ValueError('Combined report experiment differs from frozen matrix specification: ' + name)
        modes = {}
        for mode in MODES:
            summary = entry.get('sequential', {}).get(mode, _empty_summary(cohort_size)) if entry else _empty_summary(cohort_size)
            if summary.get('n') != cohort_size or summary.get('unique_cases', cohort_size) != cohort_size:
                raise ValueError('Matrix summary changed the original per-cell cohort denominator')
            completed = summary.get('completed', 0)
            attempt_coverage = summary.get('attempt_coverage', 0)
            attempted = round(attempt_coverage * cohort_size)
            completion = summary.get('cohort_goal_completion')
            goals = round(completion * cohort_size) if _number(completion) else None
            if not 0 <= completed <= attempted <= cohort_size or goals is not None and not 0 <= goals <= completed:
                raise ValueError('Inconsistent attempted, valid or successful matrix coverage')
            modes[mode] = {**summary, 'attempted': attempted, 'goals_verified': goals,
                           'aborted': summary.get('infrastructure_errors', attempted - completed),
                           'unattempted': cohort_size - attempted,
                           'scope': 'full' if completed == cohort_size else 'partial' if completed else 'no_valid_episodes',
                           'goal_rate_display': completion if attempted else None,
                           'control_accuracy_display': summary.get('observed_control_accuracy_micro') if completed else None}
        cells.append({'name': name, 'expert_row': axis[0], 'chooser_column': axis[1],
                      'architecture': specification.get('architecture'), 'seed': specification.get('seed'),
                      'modes': modes, 'sources': entry.get('sources', []) if entry else [],
                      'specification_sha256': fingerprint(specification),
                      'report_entry_present': entry is not None})
    if seen != {(row, column) for row in rows for column in columns}:
        raise ValueError('Declared crossed matrix has missing cells')
    return {'rows': rows, 'columns': columns, 'cells': cells, 'cohort_size': cohort_size}


def attach_ledgers(matrix, report, study, cases, warnings):
    index = {entry['name']: entry for entry in report.get('experiments', [])}
    valid_ids = {case['task_id'] for case in cases}
    all_calls, all_episodes, files = [], [], {}
    for cell in matrix['cells']:
        entry = index.get(cell['name'], {'name': cell['name'], 'sources': []})
        cell_study = Path(cell.get('study', study)).resolve()
        episodes, calls, diagnostics, cell_files = read_raw(entry, cell_study, valid_ids, warnings)
        files.update(cell_files)
        all_calls.extend(calls)
        all_episodes.extend(diagnostics)
        for mode in MODES:
            summary = cell['modes'][mode]
            actual = [episode for (task_id, found_mode), episode in episodes.items() if found_mode == mode]
            raw_completed = sum(episode.get('status') == 'ok' for episode in actual)
            consistent = len(actual) == summary['attempted'] and raw_completed == summary['completed']
            native_completed = [episode for episode in actual if episode.get('status') == 'ok']
            if any(type(episode.get('trajectory_outcome_met')) is not bool for episode in native_completed):
                consistent = False
            raw_goals = sum(episode.get('trajectory_outcome_met') is True for episode in native_completed)
            consistent &= raw_goals == summary['goals_verified']
            summary.update(raw_attempted=len(actual), raw_completed=raw_completed,
                           raw_goals_verified=raw_goals, source_summary_consistent=consistent)
            if not consistent:
                warnings.append('Raw/report attempt coverage changed: ' + cell['name'] + ' / ' + mode)
            subset = [row for row in diagnostics if row['mode'] == mode]
            summary['abort_categories'] = dict(Counter(row['primary_abort_category'] for row in subset if row['aborted']))
            summary['model_response_aborts'] = sum(row['model_response_abort'] for row in subset)
            summary['observed_error_event_categories'] = dict(sum((Counter(row['observed_error_event_categories']) for row in subset), Counter()))
            summary['call_accounting'] = summarize_calls([row for row in calls if row['mode'] == mode])
            summary['native_partial_decisions_observed'] = sum(row['native_partial_decisions'] or 0 for row in subset if row['aborted'])
            observed = sum(row['observed_native_control_count'] for row in subset)
            correct = sum(row['observed_native_control_correct'] for row in subset)
            summary.update(all_observed_native_control_count=observed,
                all_observed_native_control_correct=correct,
                all_observed_control_accuracy_micro=correct / observed if observed else None,
                partial_native_control_count=sum(row['observed_native_control_count'] for row in subset if row['aborted']),
                partial_native_control_correct=sum(row['observed_native_control_correct'] for row in subset if row['aborted']),
                all_observed_control_accuracy_ci95=None,
                all_observed_control_scope='Authentic graded controls from completed episodes and partial on_step records; no unobserved actions or terminal outcomes inferred.')
    return all_calls, all_episodes, files


def join_studies(studies, report):
    """Append explicitly declared expert rows from separately frozen studies."""
    matrix, cohort, provenance = None, None, []
    seen_names = set()
    for study in studies:
        study = Path(study).resolve()
        config, cases = authenticate(study, report)
        current = build_matrix(config, report)
        if matrix is None:
            matrix = {**current, 'rows': [], 'cells': []}
            cohort = cases
        elif current['columns'] != matrix['columns']:
            raise ValueError('Separate matrix studies declare different chooser columns or ordering')
        if fingerprint(cases) != fingerprint(cohort):
            raise ValueError('Separate matrix studies do not share the same original case cohort')
        if set(current['rows']) & set(matrix['rows']):
            raise ValueError('Separate matrix studies repeat an expert row; select one frozen version')
        for cell in current['cells']:
            if cell['name'] in seen_names:
                raise ValueError('Separate matrix studies repeat an experiment name')
            seen_names.add(cell['name'])
            cell['study'] = str(study)
        matrix['rows'].extend(current['rows'])
        matrix['cells'].extend(current['cells'])
        provenance.append({'study': str(study),
            'configuration_sha256': digest(study / 'configuration.json'),
            'source_manifest_sha256': digest(study / 'source-manifest.json'),
            'expert_rows': current['rows'], 'cells': len(current['cells'])})
    if matrix is None:
        raise ValueError('At least one frozen matrix study is required')
    return matrix, cohort, provenance


def pct(value):
    return f'{value * 100:.1f}%' if _number(value) else '—'


def interval(value):
    if isinstance(value, dict) and _number(value.get('low')) and _number(value.get('high')):
        return '[' + pct(value['low']) + ', ' + pct(value['high']) + ']'
    return 'unavailable'


def _save_plot(fig, directory, name, caption):
    import matplotlib.pyplot as plt
    paths = {}
    for suffix in ('pdf', 'svg', 'png'):
        path = directory / (name + '.' + suffix)
        fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
        paths[suffix] = path
    plt.close(fig)
    return {'name': name, 'caption': caption, 'paths': paths}


def publication_plots(matrix, directory):
    require_slurm()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    index = {(cell['expert_row'], cell['chooser_column']): cell for cell in matrix['cells']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    plots = []
    for mode, metric, name, title in [
        ('free_run', 'goal_rate_display', 'autonomous-goals', 'Autonomous native outcomes · all 120 cases per cell'),
        ('teacher_forced', 'control_accuracy_display', 'teacher-forced-controls', 'Teacher-forced controls · completed native episodes only'),
        ('teacher_forced', 'all_observed_control_accuracy_micro', 'teacher-forced-all-observed-controls', 'Teacher-forced controls · all authentic observed native grades')]:
        values = np.array([[index[(row, column)]['modes'][mode][metric]
                            if index[(row, column)]['modes'][mode][metric] is not None else np.nan
                            for column in matrix['columns']] for row in matrix['rows']])
        fig, ax = plt.subplots(figsize=(max(12, len(matrix['columns']) * 1.65), 4.1))
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_bad('#e6e9ef')
        image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap, aspect='auto')
        for y, row in enumerate(matrix['rows']):
            for x, column in enumerate(matrix['columns']):
                summary = index[(row, column)]['modes'][mode]
                value = summary[metric]
                if value is None:
                    text = 'pending' if summary['attempted'] == 0 else 'no observed\nscore'
                elif mode == 'free_run':
                    text = pct(value) + '\n' + str(summary['goals_verified']) + '/120 goals'
                else:
                    count = summary.get('all_observed_native_control_count', 0) if metric == 'all_observed_control_accuracy_micro' else summary.get('control_decisions_observed', 0)
                    text = pct(value) + '\nn=' + str(count) + ' controls'
                text += '\nv=' + str(summary['completed']) + '/120'
                if not summary['completed']:
                    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor='#e6e9ef', edgecolor='none', zorder=1))
                ax.text(x, y, text, ha='center', va='center', fontsize=8,
                        color='white' if value is not None and value > .62 and summary['completed'] else '#15213a')
                if summary['scope'] != 'full':
                    ax.add_patch(Rectangle((x - .49, y - .49), .98, .98, fill=False, edgecolor='#b47726', lw=1.6))
        ax.set_xticks(range(len(matrix['columns'])), [label.replace('-', '\n') for label in matrix['columns']], fontsize=8)
        ax.set_yticks(range(len(matrix['rows'])), matrix['rows'])
        ax.set(ylabel='Expert model', xlabel='Chooser model / method', title=title)
        fig.colorbar(image, ax=ax, fraction=.025, pad=.02, label='Rate')
        caption = ('Goal yield retains all 120 original cases, including model-response aborts. Zero verified goals after attempts is 0/120; '
                   'context/runtime limitations remain explicit. Grey means no complete native episode; amber means incomplete coverage.'
                   if mode == 'free_run' else
                   'Micro accuracy uses every authentic observed native control grade, including partial records from aborted episodes. '
                   'The denominator is observed controls only; terminal outcomes and absent actions are not inferred. No matching CI is supplied.'
                   if metric == 'all_observed_control_accuracy_micro' else
                   'Micro accuracy uses original controls from completed native episodes only and may have survivorship bias. '
                   'The separate all-observed figure includes authentic partial controls. Corrections and assisted goal completion remain separate.')
        fig.text(.02, -.04, caption, fontsize=8, wrap=True)
        plots.append(_save_plot(fig, directory, name, caption))
    for axis, filename, title in [('primary_failure_mode', 'failure-mechanisms', 'Autonomous goals by authored instrument failure mechanism'),
                                  ('condition', 'conditions', 'Autonomous goals by healthy / fault condition')]:
        labels = sorted({label for cell in matrix['cells']
                         for label in cell['modes']['free_run'].get('breakdowns', {}).get(axis, {})})
        if not labels:
            continue
        values = []
        for cell in matrix['cells']:
            groups = cell['modes']['free_run'].get('breakdowns', {}).get(axis, {})
            values.append([groups.get(label, {}).get('cohort_goal_completion')
                           if groups.get(label, {}).get('attempt_coverage', 0) else np.nan for label in labels])
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.15), max(7, len(matrix['cells']) * .38)))
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_bad('#e6e9ef')
        image = ax.imshow(np.array(values, dtype=float), vmin=0, vmax=1, cmap=cmap, aspect='auto')
        ax.set_yticks(range(len(matrix['cells'])), [cell['expert_row'] + ' → ' + cell['chooser_column'] for cell in matrix['cells']], fontsize=8)
        ax.set_xticks(range(len(labels)), [label.replace('_', '\n') for label in labels], fontsize=7)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=.025, pad=.02, label='Goals / original subgroup cohort')
        plots.append(_save_plot(fig, directory, filename, 'Each subgroup keeps its original case denominator, including attempted model-response aborts. Grey means no attempted episodes in that subgroup.'))
    return plots


def render_html(matrix, plots, provenance, warnings):
    escape = lambda value: html.escape(str(value))
    index = {(cell['expert_row'], cell['chooser_column']): cell for cell in matrix['cells']}
    tables = []
    for mode, heading, metric in [
            ('free_run', 'Autonomous verified goal completion · original 120-case cohort', 'goal_rate_display'),
            ('teacher_forced', 'Teacher-forced controls · completed episodes only', 'control_accuracy_display'),
            ('teacher_forced', 'Teacher-forced controls · all authentic observed native grades', 'all_observed_control_accuracy_micro')]:
        rows = []
        for expert in matrix['rows']:
            cells = ['<th scope="row">' + escape(expert) + '</th>']
            for chooser in matrix['columns']:
                cell = index[(expert, chooser)]
                summary = cell['modes'][mode]
                value = summary[metric]
                main = ('Pending' if not summary['attempted'] else 'No valid episodes') if value is None else pct(value)
                if mode == 'free_run' and value is not None:
                    main += '<small>' + str(summary['goals_verified']) + '/120 verified goals</small>'
                elif mode == 'teacher_forced' and value is not None:
                    count = summary.get('all_observed_native_control_count', 0) if metric == 'all_observed_control_accuracy_micro' else summary.get('control_decisions_observed', 0)
                    main += '<small>' + str(count) + ' observed controls</small>'
                ci = ('Workflow CI ' + interval(summary.get('goal_completion_ci95')) if mode == 'free_run' and value is not None else
                      'No supplied CI for all-observed micro' if metric == 'all_observed_control_accuracy_micro' else
                      'Episode mean ' + pct(summary.get('observed_control_accuracy_episode_macro')) + ' · CI ' + interval(summary.get('control_accuracy_ci95'))
                      if mode == 'teacher_forced' and value is not None else 'Score unavailable')
                cells.append('<td class="' + summary['scope'] + '"><a href="#cell-' + escape(fingerprint(cell['name'])[:12]) + '"><strong>' + main + '</strong></a>'
                    + '<small>attempted ' + str(summary['attempted']) + '/120 · valid ' + str(summary['completed']) + '/120</small>'
                    + '<small>aborted ' + str(summary['aborted']) + ' · pending ' + str(summary['unattempted']) + '</small>'
                    + '<small>' + escape(ci) + '</small></td>')
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        tables.append('<section><h2>' + heading + '</h2><div class="scroll"><table class="matrix"><thead><tr><th>Expert ↓ / chooser →</th>'
            + ''.join('<th>' + escape(value) + '</th>' for value in matrix['columns']) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>')
    details = []
    for cell in matrix['cells']:
        summaries = []
        for mode in MODES:
            summary = cell['modes'][mode]
            accounting = summary.get('call_accounting', {})
            causes = ', '.join(ABORT_LABELS.get(key, key) + ': ' + str(value) for key, value in summary.get('abort_categories', {}).items()) or 'None recorded'
            known_tokens = accounting.get('known_input_tokens')
            total_tokens = accounting.get('total_input_tokens')
            token_text = str(total_tokens) if total_tokens is not None else ('known subtotal ' + str(known_tokens) if known_tokens is not None else 'unknown')
            wall = accounting.get('total_model_call_wall_s')
            known_wall = accounting.get('known_model_call_wall_s')
            wall_text = f'{wall:.1f} s' if wall is not None else f'known subtotal {known_wall:.1f} s' if known_wall is not None else 'unknown'
            summaries.append('<tr><th>' + escape(mode) + '</th><td>' + str(summary['goals_verified']) + '/120</td><td>'
                + pct(summary.get('observed_control_accuracy_micro')) + '</td><td>'
                + pct(summary.get('all_observed_control_accuracy_micro')) + '<small>'
                + str(summary.get('all_observed_native_control_correct', 0)) + '/'
                + str(summary.get('all_observed_native_control_count', 0)) + ' graded controls</small></td><td>' + str(summary.get('interventions', 0))
                + '</td><td>' + str(accounting.get('calls', 0)) + '</td><td>' + escape(token_text) + '<small>usage coverage '
                + pct(accounting.get('usage_coverage')) + '</small></td><td>' + escape(wall_text) + '</td><td>' + escape(causes) + '</td></tr>')
        details.append('<details id="cell-' + fingerprint(cell['name'])[:12] + '"><summary>' + escape(cell['expert_row'] + ' → ' + cell['chooser_column'])
            + '</summary><div class="scroll"><table><thead><tr><th>Mode</th><th>Goals /120</th><th>Completed-only controls</th><th>All observed controls</th><th>Corrections</th><th>Model-call attempts</th><th>Input tokens</th><th>Call wall time</th><th>Abort causes</th></tr></thead><tbody>'
            + ''.join(summaries) + '</tbody></table></div><p class="note">Partial native records from aborted episodes remain in the raw ledger. '
            + 'They are excluded from completed-only accuracy and included in the separately labelled all-observed control denominator. '
            + 'Model-call attempts include failed context preflights; they do not establish that a neural forward executed. Unknown usage is never zero-filled.</p></details>')
    figures = []
    for plot in plots:
        encoded = base64.b64encode(plot['paths']['png'].read_bytes()).decode()
        figures.append('<figure><img alt="' + escape(plot['caption']) + '" src="data:image/png;base64,' + encoded + '"><figcaption>'
            + escape(plot['caption']) + ' <a href="plots/' + plot['paths']['pdf'].name + '">PDF</a> · <a href="plots/' + plot['paths']['svg'].name + '">SVG</a></figcaption></figure>')
    all_modes = [cell['modes'][mode] for cell in matrix['cells'] for mode in MODES]
    attempted = sum(summary['attempted'] for summary in all_modes)
    completed = sum(summary['completed'] for summary in all_modes)
    expected = len(matrix['cells']) * 240
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Instrument benchmark · expert × chooser</title><style>
:root{color-scheme:light;--ink:#17263f;--muted:#5b6b83;--line:#dce3ec;--blue:#184f8b}*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:var(--ink);font:15px/1.6 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:36px 24px 80px}h1{font-size:35px;letter-spacing:-1px;line-height:1.2;margin:8px 0 16px}h2{font-size:22px;margin:0 0 16px}p{max-width:1100px}.eyebrow{color:var(--blue);font-weight:750;letter-spacing:1px;font-size:12px;text-transform:uppercase}.lede{font-size:18px;color:var(--muted)}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:25px 0}.card{flex:1;min-width:170px;padding:18px;background:white;border:1px solid var(--line);border-radius:10px}.card strong{font-size:27px;display:block}.card span,.note,small{color:var(--muted)}section,details,figure{background:white;border:1px solid var(--line);border-radius:10px;padding:22px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{border:1px solid var(--line);padding:12px;vertical-align:top;text-align:left}th{background:#f5f7fb}small{display:block;font-size:11px;line-height:1.5;margin-top:5px}.matrix td{min-width:160px}.matrix strong{font-size:21px;color:var(--ink)}.matrix strong small{font-size:12px;font-weight:500}.matrix a{text-decoration:none}.partial{box-shadow:inset 0 3px #bc842e}.no_valid_episodes{background:#eef1f5;box-shadow:inset 0 3px #aeb8c8}.full{box-shadow:inset 0 3px #367e69}.reference{border-left:4px solid #8094ac;padding:14px 20px;background:#edf2f8}summary{cursor:pointer;font-weight:650}figure img{width:100%;height:auto}figcaption{font-size:12px;color:var(--muted)}a{color:#185b9f}code{font-size:12px;word-break:break-all}.note{font-size:13px}.footer{font-size:12px;color:var(--muted);margin-top:35px}</style><main>
<div class="eyebrow">Actual repository LangGraph · local models only</div><h1>Expert × chooser on instrument failures</h1>
<p class="lede">The same instrument goal persists across actions: generate alternatives, choose an operation, pass the typed action gate, observe the native simulator, then continue.</p>
<div class="cards"><div class="card"><strong>''' + str(len(matrix['rows'])) + ''' × ''' + str(len(matrix['columns'])) + '''</strong><span>Explicit expert / chooser matrix</span></div><div class="card"><strong>120</strong><span>Original cases per cell and mode</span></div><div class="card"><strong>''' + str(attempted) + '/' + str(expected) + '''</strong><span>Attempted episodes · both modes</span></div><div class="card"><strong>''' + str(completed) + '/' + str(expected) + '''</strong><span>Valid complete native episodes</span></div><div class="card"><strong>0</strong><span>Hosted inference calls</span></div></div>
<p class="reference"><strong>Context reference:</strong> the supplied hosted-model PDF reports 70/120 outcomes (58.3%). Models, public adapters and budgets differ. This reference is not a paired comparison, reproduced run, API probability readout, or significance claim.</p>
<p class="note">Every autonomous denominator remains 120, including graph aborts. Zero verified outcomes after attempted model-response failures is 0/120. Green indicates all native episodes completed; amber indicates partial coverage; grey indicates no complete native episodes. Context/runtime limitations remain explicit. Missing actions have no invented decision accuracy. Teacher-forced corrections change execution while preserving the original grade.</p>
''' + ''.join(tables) + '''<p class="note">The autonomous interval describes full-cohort pipeline outcome yield and resamples whole workflows. Teacher-forced matrix percentages are micro accuracy over observed original control decisions from completed episodes. Their supplied workflow interval applies to the separately labelled episode mean, not the micro percentage. Corrected goal completion is assisted execution, not autonomous success or decision accuracy.</p>
<h2>Publication figures</h2>''' + ''.join(figures) + '''<h2>Calls, aborts and corrected outcomes</h2>''' + ''.join(details) + '''
<details><summary>Methods, provenance and limitations</summary><p>Rows and columns come from the frozen study configuration. Native grades, workflow intervals and breakdowns are reused from the authenticated combined report without regrading. Failure mechanisms and healthy/fault conditions remain separate axes. Trace call attempts are deduplicated; unknown token counters and timing remain unknown. Completed-only step accuracy can have survivorship bias. All-observed native control accuracy separately includes genuine partial on_step grades, while unobserved actions and terminal outcomes stay unknown. Model-call wall time is not GPU kernel time and mixes hardware unless stratified in the combined report.</p><p>Aborts distinguish model JSON/schema, citation, action-preset and output-budget failures from unsupported context, runtime/process/transport errors, and episode limits. Repaired error events are counted separately from terminal episode aborts.</p><p>Independent expert review remains pending. Cases are authored instrument simulations; coupled material chemistry and physical hardware are outside this evaluation. Decision probabilities are conditional on the offered menu and are not established as calibrated safety scores.</p><pre><code>''' + escape(json.dumps(provenance, indent=2, sort_keys=True)) + '''</code></pre></details>
<details><summary>Coverage warnings (''' + str(len(warnings)) + ''')</summary><ul>''' + ''.join('<li>' + escape(value) + '</li>' for value in warnings) + '''</ul></details>
<p><a href="matrix.json">Matrix JSON</a> · <a href="matrix.csv">Summary CSV</a> · <a href="call-ledger.jsonl">Actual call ledger</a> · <a href="episode-diagnostics.jsonl">Episode abort diagnostics</a></p><div class="footer">''' + VERSION + ' · ' + escape(provenance['generated_at']) + '</div></main></html>'


def main(argv=None):
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, action='append', required=True,
                        help='Frozen study; repeat to append distinct explicit expert rows')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    studies, report_path, output = [path.resolve() for path in args.study], args.report.resolve(), args.output.resolve()
    if len(studies) != len(set(studies)):
        raise ValueError('The same frozen study was supplied more than once')
    if output.exists():
        raise FileExistsError('Matrix report output must be fresh: ' + str(output))
    report = json.loads(report_path.read_text())
    matrix, cases, study_provenance = join_studies(studies, report)
    warnings = []
    calls, diagnostics, files = attach_ledgers(matrix, report, studies[0], cases, warnings)
    provenance = {'version': VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'studies': study_provenance,
        'combined_report': str(report_path), 'combined_report_sha256': digest(report_path),
        'matrix_report_source_sha256': digest(__file__), 'compute_node': os.environ['SLURMD_NODENAME'],
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hosted_inference_calls': 0,
        'reference_pdf': {'outcomes': 70, 'cohort': 120, 'rate': 70 / 120, 'scope': 'context only; different hosted models and budgets; no paired inference'}}
    result = {**matrix, 'provenance': provenance, 'warnings': warnings,
        'source_files_sha256': files, 'call_accounting': summarize_calls(calls),
        'abort_category_definitions': ABORT_LABELS,
        'all_expected_outputs_present': all(cell['modes'][mode]['attempted'] == 120
            and cell['modes'][mode]['raw_attempted'] == 120 and cell['modes'][mode]['source_summary_consistent']
            for cell in matrix['cells'] for mode in MODES),
        'all_native_episodes_valid': all(cell['modes'][mode]['completed'] == 120
            and cell['modes'][mode]['raw_completed'] == 120 and cell['modes'][mode]['source_summary_consistent']
            for cell in matrix['cells'] for mode in MODES)}
    output.mkdir(parents=True)
    plots = publication_plots(matrix, output / 'plots')
    write_json(output / 'matrix.json', result)
    fields = ['expert_row', 'chooser_column', 'experiment', 'mode', 'cohort', 'attempted', 'valid', 'aborted', 'missing',
              'goals', 'cohort_goal_completion', 'control_accuracy_micro', 'control_accuracy_episode_macro',
              'observed_controls', 'all_observed_control_accuracy_micro', 'all_observed_native_control_correct',
              'all_observed_native_control_count', 'partial_native_control_count',
              'corrections', 'model_response_aborts', 'calls', 'token_usage_coverage',
              'known_input_tokens', 'total_input_tokens', 'known_model_call_wall_s', 'total_model_call_wall_s']
    with (output / 'matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in matrix['cells']:
            for mode in MODES:
                value = cell['modes'][mode]
                accounting = value['call_accounting']
                writer.writerow({'expert_row': cell['expert_row'], 'chooser_column': cell['chooser_column'],
                    'experiment': cell['name'], 'mode': mode, 'cohort': 120, 'attempted': value['attempted'],
                    'valid': value['completed'], 'aborted': value['aborted'], 'missing': value['unattempted'],
                    'goals': value['goals_verified'], 'cohort_goal_completion': value['goal_rate_display'],
                    'control_accuracy_micro': value['control_accuracy_display'],
                    'control_accuracy_episode_macro': value.get('observed_control_accuracy_episode_macro'),
                    'observed_controls': value.get('control_decisions_observed'), 'corrections': value.get('interventions'),
                    **{key: value[key] for key in ('all_observed_control_accuracy_micro',
                       'all_observed_native_control_correct', 'all_observed_native_control_count', 'partial_native_control_count')},
                    'model_response_aborts': value.get('model_response_aborts'), 'calls': accounting['calls'],
                    'token_usage_coverage': accounting['usage_coverage'],
                    **{key: accounting[key] for key in ('known_input_tokens', 'total_input_tokens', 'known_model_call_wall_s', 'total_model_call_wall_s')}})
    for filename, rows in [('call-ledger.jsonl', calls), ('episode-diagnostics.jsonl', diagnostics)]:
        with (output / filename).open('w') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
    (output / 'index.html').write_text(render_html(matrix, plots, provenance, warnings))
    print(canonical({'status': 'matrix_report_written', 'output': str(output), 'cells': len(matrix['cells']),
                     'actual_model_calls': len(calls), 'all_expected_outputs_present': result['all_expected_outputs_present'],
                     'all_native_episodes_valid': result['all_native_episodes_valid']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
