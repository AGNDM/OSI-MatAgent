"""Read-only workflow reachability appendix for an authenticated matrix report.

The appendix separates routing from operational selection. It never awards an
action grade or terminal outcome, and it does not change the primary report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path


MODES = ('free_run', 'teacher_forced')
STAGES = ('router_entered', 'diagnostic_entered', 'operation_chooser_entered',
          'operation_choice_valid', 'procedure_entered', 'procedure_schema_valid',
          'typed_gate_accepted', 'native_handoff_reached', 'native_grading_reached')


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Workflow appendix analysis requires a genuine Slurm allocation')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Unfinished or invalid appendix input: {path}:{number}') from error


def stage_evidence(events, calls, diagnostics):
    """Project stage evidence without interpreting missing actions as failures."""
    result = {stage: False for stage in STAGES}
    result.update(router_call_attempts=0, nontrivial_router_call_attempts=0,
        operation_call_attempts=0, nontrivial_operation_call_attempts=0,
        nontrivial_operation_valid_selections=0, procedure_call_attempts=0)
    seen = set()
    for event in events:
        event_id = event.get('event_id')
        if event_id and event_id in seen:
            continue
        if event_id:
            seen.add(event_id)
        kind, component = event.get('kind'), event.get('component')
        metadata = event.get('llm_metadata') or event.get('backend_metadata') or {}
        options = metadata.get('options', {})
        if component == 'router' and kind in {'routing_start', 'routing'}:
            result['router_entered'] = True
        if component == 'diagnostic' and kind in {'role_start', 'role_output', 'role_error'}:
            result['diagnostic_entered'] = True
        if component == 'operation_decision' and kind in {'operation_start', 'operation_selected', 'operation_error'}:
            result['operation_chooser_entered'] = True
        if kind == 'operation_selected':
            result['operation_choice_valid'] = True
            count = len(options) if options else len(event.get('candidates', []))
            if count >= 2:
                result['nontrivial_operation_valid_selections'] += 1
        if component == 'procedure' and kind in {'role_start', 'role_output', 'role_error'}:
            result['procedure_entered'] = True
        if component == 'procedure' and kind == 'role_output':
            result['procedure_schema_valid'] = True
        if component == 'gate' and kind == 'accepted':
            result['typed_gate_accepted'] = True
        if component == 'native_bridge' and kind == 'action_handoff':
            result['native_handoff_reached'] = True
        # Count nontrivial menu attempts from the exact canonical option map,
        # including errors. A native single-option shortcut is not a learned
        # operational choice even when its boundary was reached.
        if kind == 'routing' and metadata.get('request_sent') is True and len(options) >= 2:
            result['nontrivial_router_call_attempts'] += 1
        if kind in {'operation_selected', 'operation_error'} and metadata.get('request_sent') is True:
            count = len(options) if options else len(event.get('candidates', []))
            if count >= 2:
                result['nontrivial_operation_call_attempts'] += 1
    # Terminal call rows were already deduplicated by the validated matrix
    # reader. These counts include failed preflight attempts, not guaranteed
    # neural forward passes.
    for call in calls:
        kind, component = call.get('kind'), call.get('component')
        if kind == 'routing':
            result['router_call_attempts'] += 1
            result['router_entered'] = True
        if kind in {'operation_selected', 'operation_error'}:
            result['operation_call_attempts'] += 1
            result['operation_chooser_entered'] = True
            result['operation_choice_valid'] |= kind == 'operation_selected'
        if component == 'diagnostic':
            result['diagnostic_entered'] = True
        if component == 'procedure' and kind in {'role_output', 'role_error'}:
            result['procedure_call_attempts'] += 1
            result['procedure_entered'] = True
            result['procedure_schema_valid'] |= kind == 'role_output'
    native_grades = diagnostics.get('observed_native_grades', []) if diagnostics else []
    result['native_grading_reached'] = bool(native_grades)
    # An authentic native grade proves that an offered action, including a
    # terminal completion claim, reached the evaluator. No final success follows.
    result['native_handoff_reached'] |= bool(native_grades)
    result['nontrivial_operation_choice_observed'] = result['nontrivial_operation_valid_selections'] > 0
    return result


def barrier(row):
    """Farthest observed boundary; not a claim about a model's capability."""
    if row['native_grading_reached']:
        return 'native_grades_observed'
    if row['native_handoff_reached']:
        return 'native_handoff_without_grade'
    if row['typed_gate_accepted']:
        return 'after_gate_before_native_handoff'
    if row['procedure_schema_valid']:
        return 'procedure_valid_before_gate'
    if row['procedure_entered']:
        return 'procedure_not_validated'
    if row['operation_choice_valid']:
        return 'operation_selected_before_procedure'
    if row['operation_chooser_entered']:
        return 'operation_choice_not_validated'
    if row['diagnostic_entered']:
        return 'expert_before_operation_chooser'
    if row['router_entered']:
        return 'router_before_expert'
    return 'before_graph_or_not_attempted'


def summarize(rows):
    n = len(rows)
    return {'cohort': n, 'attempted_episodes': sum(row['episode_record_present'] for row in rows),
        'trace_coverage': sum(row['trace_present'] for row in rows),
        **{stage + '_episodes': sum(row[stage] for row in rows) for stage in STAGES},
        **{stage + '_cohort_rate': sum(row[stage] for row in rows) / n if n else None for stage in STAGES},
        'nontrivial_operation_choice_episodes': sum(row['nontrivial_operation_choice_observed'] for row in rows),
        'router_call_attempts': sum(row['router_call_attempts'] for row in rows),
        'nontrivial_router_call_attempts': sum(row['nontrivial_router_call_attempts'] for row in rows),
        'operation_call_attempts': sum(row['operation_call_attempts'] for row in rows),
        'nontrivial_operation_call_attempts': sum(row['nontrivial_operation_call_attempts'] for row in rows),
        'nontrivial_operation_valid_selections': sum(row['nontrivial_operation_valid_selections'] for row in rows),
        'procedure_call_attempts': sum(row['procedure_call_attempts'] for row in rows),
        'barriers': dict(Counter(row['barrier'] for row in rows)),
        'terminal_abort_categories': dict(Counter(row['primary_abort_category'] for row in rows if row['primary_abort_category'])),
        'operational_chooser_attribution': ('No valid nontrivial operational choice was observed; the backend may still have affected routing.'
            if not any(row['nontrivial_operation_choice_observed'] for row in rows)
            else 'Operational chooser was exercised on a subset; report this coverage alongside whole-workflow outcomes.')}


def analyze(matrix, diagnostics, calls):
    index, call_index = {}, defaultdict(list)
    valid_names = {cell['name'] for cell in matrix['cells']}
    for diagnostic in diagnostics:
        key = (diagnostic['experiment'], diagnostic['task_id'], diagnostic['mode'])
        if key in index or key[0] not in valid_names or key[2] not in MODES:
            raise ValueError('Duplicate, unknown or invalid episode diagnostic')
        index[key] = diagnostic
    for call in calls:
        key = (call['experiment'], call['task_id'], call['mode'])
        if key[0] not in valid_names or key[2] not in MODES:
            raise ValueError('Unknown or invalid graph call ledger row')
        call_index[key].append(call)
    rows, summaries, trace_files = [], [], {}
    protocol_hashes = set()
    for cell in matrix['cells']:
        study = Path(cell['study']).resolve()
        prepared = json.loads((study / 'prepared.json').read_text())
        manifest_path = study / 'source-manifest.json'
        if digest(manifest_path) != prepared['source_manifest_sha256']:
            raise ValueError('Study source manifest changed since execution')
        source_manifest = json.loads(manifest_path.read_text())
        case_path = study / 'snapshot/instrument/sequential_120/cases.jsonl'
        if digest(case_path) != source_manifest['instrument/sequential_120/cases.jsonl']:
            raise ValueError('Original public cohort changed')
        cases = list(jsonl(case_path))
        if len(cases) != 120 or len({case['task_id'] for case in cases}) != 120:
            raise ValueError('Reachability must retain all original 120 cases')
        configuration_path = study / 'configuration.json'
        if digest(configuration_path) != prepared['configuration_sha256']:
            raise ValueError('Matrix study configuration changed')
        config = json.loads(configuration_path.read_text())
        spec = next(item for item in config['experiments'] if item['name'] == cell['name'])
        protocol_hashes.add(spec.get('graph_source_manifest_sha256', prepared['source_manifest_sha256']))
        for mode in MODES:
            cohort_rows = []
            for case in cases:
                key = (cell['name'], case['task_id'], mode)
                diagnostic = index.get(key)
                events, present = [], False
                for source in cell.get('sources', []):
                    source = Path(source).resolve()
                    if not source.is_relative_to(study / 'runs'):
                        raise ValueError('Trace source is outside the frozen study run root')
                    trace = source / cell['name'] / 'sequential/steps' / hashlib.sha256(case['task_id'].encode()).hexdigest()[:24] / mode / 'graph.jsonl'
                    if not trace.is_file():
                        continue
                    if present:
                        raise ValueError('More than one trace for one cell/task/mode')
                    expected = matrix.get('source_files_sha256', {}).get(str(trace))
                    actual = digest(trace)
                    if expected is None or expected != actual:
                        raise ValueError('Raw graph trace differs from authenticated matrix evidence: ' + str(trace))
                    events = list(jsonl(trace))
                    present = True
                    trace_files[str(trace)] = actual
                evidence = stage_evidence(events, call_index.get(key, []), diagnostic)
                row = {'experiment': cell['name'], 'expert_row': cell['expert_row'], 'chooser_column': cell['chooser_column'],
                    'task_id': case['task_id'], 'mode': mode, 'episode_record_present': diagnostic is not None,
                    'trace_present': present, 'primary_abort_category': diagnostic.get('primary_abort_category') if diagnostic else None,
                    **evidence}
                row['barrier'] = barrier(row)
                rows.append(row)
                cohort_rows.append(row)
            summaries.append({'experiment': cell['name'], 'expert_row': cell['expert_row'],
                'chooser_column': cell['chooser_column'], 'mode': mode, **summarize(cohort_rows)})
    return {'cells': summaries, 'episode_reachability': rows, 'trace_files_sha256': trace_files,
        'graph_source_protocol_hashes': sorted(protocol_hashes), 'homogeneous_graph_source_protocol': len(protocol_hashes) == 1,
        'protocol_warning': None if len(protocol_hashes) == 1 else
            'Different graph source protocols occur across cells. Keep protocol versions in separate primary matrices; do not pool this appendix as one controlled comparison.'}


def render_html(result):
    escape = lambda value: html.escape(str(value))
    rows = []
    for cell in result['cells']:
        counts = [cell[key + '_episodes'] for key in ('router_entered', 'operation_chooser_entered',
            'operation_choice_valid', 'procedure_entered', 'typed_gate_accepted', 'native_handoff_reached', 'native_grading_reached')]
        rows.append('<tr><th>' + escape(cell['expert_row'] + ' → ' + cell['chooser_column']) + '<small>' + escape(cell['mode'])
            + '</small></th>' + ''.join('<td>' + str(value) + '/120</td>' for value in counts)
            + '<td>' + str(cell['nontrivial_operation_choice_episodes']) + '/120</td><td>'
            + escape(', '.join(key + ': ' + str(value) for key, value in cell['barriers'].items())) + '</td></tr>')
    warning = result.get('protocol_warning') or 'One graph source protocol across the included cells.'
    return ('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Workflow reachability</title>
<style>body{font:14px/1.6 system-ui;color:#17263f;background:#f6f8fb;margin:30px}main{max-width:1600px;margin:auto}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #dce3ec;padding:10px;text-align:left;vertical-align:top}th{background:#edf2f8}small{display:block;color:#607088}section{overflow:auto}p{max-width:1100px}.note{padding:14px;background:#e9f0f8}a{color:#185b9f}</style><main><h1>Which workflow boundaries were reached?</h1>
<p>Whole-workflow failures before operational selection provide no direct evidence about the operation chooser. The same backend can still affect routing. Each denominator below is the original 120 cases in one cell and mode.</p><p class="note">'''
        + escape(warning) + '''</p><section><table><thead><tr><th>Expert → chooser / mode</th><th>Router</th><th>Operation chooser entered</th><th>Valid operation selected</th><th>Procedure</th><th>Typed gate accepted</th><th>Native handoff</th><th>Native grade</th><th>Nontrivial operation choice</th><th>Farthest observed barrier</th></tr></thead><tbody>'''
        + ''.join(rows) + '''</tbody></table></section><p>“Nontrivial” requires at least two offered routes or operation candidates. A single-option native shortcut does not establish a learned operational choice. Native handoff is established by its explicit graph event or an authentic native grade, including terminal completion claims. No final outcome is inferred from partial records. Reaching a boundary once is different from reaching it at every action.</p><p>Model-call counts in JSON/CSV are recorded attempts, including failed context preflights. Procedure entry and typed gate acceptance use actual graph events; token generation is not assumed. Sources are verified against the frozen matrix trace hashes. Native grades and primary report scores remain unchanged.</p><p><a href="reachability.json">JSON</a> · <a href="coverage.csv">Cell coverage CSV</a> · <a href="episode-reachability.jsonl">Per-case boundary evidence</a></p></main></html>''')


def main(argv=None):
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True, help='matrix/matrix.json from the validated primary reporter')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    matrix_path, output = args.matrix.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Reachability appendix output must be fresh')
    matrix = json.loads(matrix_path.read_text())
    diagnostics_path = matrix_path.with_name('episode-diagnostics.jsonl')
    calls_path = matrix_path.with_name('call-ledger.jsonl')
    result = analyze(matrix, list(jsonl(diagnostics_path)), list(jsonl(calls_path)))
    result['provenance'] = {'generated_at': datetime.now(timezone.utc).isoformat(),
        'matrix_sha256': digest(matrix_path), 'episode_diagnostics_sha256': digest(diagnostics_path),
        'call_ledger_sha256': digest(calls_path), 'appendix_source_sha256': digest(__file__),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'compute_node': os.environ['SLURMD_NODENAME'],
        'modifies_primary_scores': False, 'hosted_inference_calls': 0}
    output.mkdir(parents=True)
    (output / 'reachability.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    flat = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in result['cells']]
    with (output / 'coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
    with (output / 'episode-reachability.jsonl').open('w') as stream:
        for row in result['episode_reachability']:
            stream.write(canonical(row) + '\n')
    (output / 'index.html').write_text(render_html(result))
    print(canonical({'status': 'reachability_appendix_written', 'output': str(output),
        'cells_and_modes': len(result['cells']), 'homogeneous_graph_source_protocol': result['homogeneous_graph_source_protocol']}), flush=True)
    return 0


if __name__ == '__main__':
    main()
