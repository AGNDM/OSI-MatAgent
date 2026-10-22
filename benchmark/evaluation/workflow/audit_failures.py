"""Slurm-only authenticated audit of all22 saved production episode failures.

No scorer, evaluator, simulator, inference, retries or source changes are used.
Error classifications inspect actual error fields and emitted call metadata,
never arbitrary prompt text. Categories describe observed evidence, not a claim
that every unknown failure has an established model or infrastructure cause.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

STUDY = None
CONTEXT_PATTERNS = [
    re.compile(r'Public input (\d+) \+ output reserve (\d+) exceeds context (\d+)', re.I),
    re.compile(r'Input of (\d+) tokens exceeds (?:native )?context\s*(\d+)', re.I)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def categories(error, classification=None):
    """Use only actual exception text, never nested payload/prompt strings."""
    text = error.lower()
    found = set()
    if (any(pattern.search(error) for pattern in CONTEXT_PATTERNS)
            or any(value in text for value in ('tokenbudgetexceeded:', 'decisioncontextlimiterror:',
                'exceeds context', 'exceeds native context', 'context length exceeded'))):
        found.add('context_overflow')
    if 'out of memory' in text and ('cuda' in text or 'gpu' in text or 'outofmemoryerror' in text):
        found.add('cuda_out_of_memory')
    elif 'cuda error' in text or 'cublas' in text or 'cudnn' in text:
        found.add('cuda_runtime_error')
    if any(value in text for value in ('brokenpipeerror', 'eoferror', 'connectionreseterror',
                                       'connection refused', 'subprocess worker', 'worker process died')):
        found.add('ipc_or_worker_failure')
    if any(value in text for value in ('filenotfounderror', 'no such file or directory',
                                       'artifact mismatch', 'fitted artifact', 'manifest mismatch')):
        found.add('missing_file_or_artifact_identity')
    if any(value in text for value in ('importerror', 'modulenotfounderror', 'undefined symbol',
                                       'unexpected keyword argument', 'has no attribute')):
        found.add('library_or_api_compatibility')
    if classification in {'expert_output_truncated', 'expert_empty_output', 'expert_schema_error'}:
        found.add(classification)
    if any(value in text for value in ('validationerror', 'validation error', 'jsondecodeerror',
                                       'invalid json', 'schema validation', 'duplicate menu action')):
        found.add('generated_record_or_schema_error')
    if 'graphrecursionerror' in text or 'recursion limit' in text:
        found.add('graph_recursion_limit')
    if 'timeouterror' in text or 'timeout' in text or 'deadline' in text:
        found.add('timeout_or_deadline')
    return found


def error_evidence(event, trace, line):
    error = event.get('error')
    if not isinstance(error, str) or not error:
        return None
    metadata = event.get('llm_metadata') or event.get('backend_metadata') or {}
    if not isinstance(metadata, dict):
        metadata = {}
    classification = metadata.get('response_classification')
    generated = {}
    choices = (metadata.get('response') or {}).get('choices') or []
    if choices:
        raw = (choices[0].get('message') or {}).get('content')
        if isinstance(raw, str):
            generated = {'raw_response_sha256': sha(raw.encode()), 'raw_response_characters': len(raw)}
            try:
                wire = json.loads(raw)
                if isinstance(wire, dict):
                    ids = [value.get('menu_action_id') for value in wire.get('candidates', []) if isinstance(value, dict)]
                    generated.update(candidate_menu_action_ids=ids, evidence_ids=wire.get('evidence_ids'),
                        duplicate_candidate_ids=bool(ids) and len(ids) != len(set(ids)))
            except (ValueError, TypeError):
                generated['invalid_json'] = True
    context = []
    for pattern in CONTEXT_PATTERNS:
        match = pattern.search(error)
        if match:
            values = [int(value) for value in match.groups()]
            context.append({'input_tokens': values[0], 'output_reserve': values[1] if len(values) == 3 else None,
                'context_limit': values[-1]})
    payload = event.get('input')
    encoded_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return {'trace_path': str(trace), 'line': line, 'event_id': event.get('event_id'),
        'component': event.get('component'), 'kind': event.get('kind'),
        'error': error[:2000], 'error_truncated_in_audit': len(error) > 2000,
        'full_error_sha256': sha(error.encode()), 'classification': classification,
        'categories': sorted(categories(error, classification)), 'context_limits': context,
        'request_sent': metadata.get('request_sent'), 'usage': metadata.get('usage'),
        'generated_response': generated,
        'input_payload_sha256': sha(encoded_payload), 'input_payload_bytes': len(encoded_payload),
        'will_repair': event.get('will_repair'), 'repair_attempt': event.get('repair_attempt')}


def main():
    global STUDY
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    STUDY = args.study.resolve()
    if args.output.resolve().is_relative_to(STUDY):
        raise ValueError('Write new audit evidence outside the frozen study')
    sys.path[:0] = [str(STUDY / 'snapshot'), str(STUDY / 'snapshot/upstream')]
    from graph_local.run import authenticate
    authenticate(STUDY)
    prepared = read(STUDY / 'prepared.json')
    for filename, key in [('configuration.json', 'configuration_sha256'), ('jobs.json', 'jobs_sha256')]:
        if sha((STUDY / filename).read_bytes()) != prepared[key]:
            raise ValueError('Frozen input changed: ' + filename)
    config, jobs = read(STUDY / 'configuration.json'), read(STUDY / 'jobs.json')
    public_path = STUDY / 'snapshot/instrument/sequential_120/cases.jsonl'
    public = [json.loads(line) for line in public_path.read_text().splitlines() if line.strip()]
    specs = {row['name']: row for row in config['experiments']}
    cells = {name: {'experiment': name, 'expert': spec['expert_row'], 'chooser': spec['chooser_column'],
        'batch': spec['deadline_batch'], 'planned': 120, 'closed': 0, 'native_completed': 0,
        'assisted_goals': 0, 'aborted': 0, 'missing': 0, 'in_progress_unclosed': 0, 'unusable': 0,
        'abort_categories': Counter(), 'abort_error_messages': Counter(), 'terminal_reasons': Counter(), 'failures': []} for name, spec in specs.items()}
    read_errors, source_files = [], []
    for index, job in enumerate(jobs):
        name = job['experiment']
        if name not in specs:
            continue
        cell = cells[name]
        worker = STUDY / 'runs' / name / 'sequential' / f"shard-{job['shard_index']:03d}"
        episode_path = worker / name / 'sequential/episodes.jsonl'
        if not episode_path.exists():
            cell['missing'] += 1
            continue
        try:
            data = episode_path.read_bytes()
            lines = data.splitlines(keepends=True)
            if not lines or (len(lines) == 1 and not lines[0].endswith(b'\n')):
                cell['missing'] += 1
                cell['in_progress_unclosed'] += 1
                continue
            if len(lines) != 1:
                raise ValueError('Expected at most one completely written closed episode')
            row = json.loads(lines[0])
            task = public[job['shard_index']]
            task_id = task['task_id']
            worker_config = read(worker / 'configuration.json')
            if (row.get('task_id') != task_id or row.get('mode') != 'teacher_forced'
                    or worker_config.get('selected_experiments') != [name]
                    or worker_config.get('selected_task_ids') != [task_id]
                    or worker_config.get('source_sha256', {}).get('configuration_input') != prepared['configuration_sha256']):
                raise ValueError('Production identity differs from frozen selection')
            if row.get('status', 'ok') not in {'ok', 'infrastructure_error'}:
                raise ValueError('Unknown closed episode status')
            source_files.append({'path': str(episode_path), 'sha256': sha(data), 'bytes': len(data)})
            cell['closed'] += 1
            if row.get('status', 'ok') == 'ok':
                cell['native_completed'] += 1
                cell['assisted_goals'] += row.get('trajectory_outcome_met') is True
                continue
            cell['aborted'] += 1
            terminal = row.get('graph_terminal_reason') or 'unspecified'
            cell['terminal_reasons'][terminal] += 1
            trace = episode_path.parent / 'steps' / sha(task_id.encode())[:24] / 'teacher_forced/graph.jsonl'
            errors = []
            trace_errors = []
            verification_outputs = []
            handoffs = Counter()
            if trace.exists():
                trace_data = trace.read_bytes()
                source_files.append({'path': str(trace), 'sha256': sha(trace_data), 'bytes': len(trace_data)})
                for number, line in enumerate(trace_data.splitlines(), 1):
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError) as error:
                        trace_errors.append({'line': number, 'error': str(error)})
                        continue
                    evidence = error_evidence(event, trace, number)
                    if evidence:
                        errors.append(evidence)
                    if event.get('kind') == 'role_output' and event.get('component') == 'verification':
                        report = event.get('output') or {}
                        if isinstance(report, dict):
                            verification_outputs.append({'trace_path': str(trace), 'line': number,
                                'event_id': event.get('event_id'), 'status': report.get('status'),
                                'evidence_ids': report.get('evidence_ids')})
                    if event.get('kind') == 'action_handoff':
                        handoffs[json.dumps(event.get('native_action'), sort_keys=True, separators=(',', ':'))] += 1
            else:
                trace_errors.append({'error': 'Aborted graph trace is absent'})
            observed = set().union(*(set(error['categories']) for error in errors)) if errors else set()
            observed.update(categories(row.get('error', '')))
            if any(error['generated_response'].get('duplicate_candidate_ids') for error in errors):
                observed.add('generated_duplicate_candidate_ids')
            if terminal in {'deadline', 'action_budget', 'llm_budget', 'token_budget_exceeded'}:
                observed.add('terminal_' + terminal)
            if terminal == 'duplicate_verification_evidence':
                observed.add('repeated_verification_without_new_evidence')
            if not observed:
                observed.add('unknown')
            for value in observed:
                cell['abort_categories'][value] += 1
            for value in {error['error'] for error in errors if error['kind'] != 'policy_abort'}:
                cell['abort_error_messages'][value] += 1
            failure = {'job_index': index, 'shard_index': job['shard_index'], 'task_id': task_id,
                'public_instrument': task.get('instrument_id'), 'episode_path': str(episode_path),
                'episode_sha256': sha(data), 'error': row.get('error'), 'terminal_reason': terminal,
                'observed_categories': sorted(observed),
                'model_response_classifications': row.get('model_response_classifications'),
                'partial_native_decisions': row.get('partial_native_decisions_observed'),
                'partial_native_correct_original_proposals': row.get('partial_native_correct_original_proposals'),
                'graph_error_events': errors, 'trace_read_errors': trace_errors,
                'verification_outputs': verification_outputs,
                'repeated_identical_proposed_handoffs_max': max(handoffs.values(), default=0),
                'repetition_note': 'Descriptive handoff repetition only; repetition alone is not classified as the cause of abort.'}
            cell['failures'].append(failure)
        except (OSError, ValueError, TypeError, KeyError) as error:
            cell['unusable'] += 1
            read_errors.append({'path': str(episode_path), 'error': f'{type(error).__name__}: {error}'})
    result = {'schema_version': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'study': str(STUDY), 'mode': 'teacher_forced', 'seed': 0,
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'compute_node': os.environ['SLURMD_NODENAME'],
        'scope': 'All22 declared production cells and their currently closed production episodes; saved records only.',
        'source_manifest_sha256': prepared['source_manifest_sha256'], 'configuration_sha256': prepared['configuration_sha256'],
        'jobs_sha256': prepared['jobs_sha256'], 'audit_helper': str(Path(__file__)),
        'audit_helper_sha256': sha(Path(__file__).read_bytes()), 'public_cases_sha256': sha(public_path.read_bytes()),
        'source_files': source_files, 'read_errors': read_errors, 'cells': list(cells.values()),
        'trace_read_error_count': sum(len(failure['trace_read_errors']) for cell in cells.values() for failure in cell['failures']),
        'method': 'Authenticate frozen source; verify declared worker identities; inspect actual episode/error-event fields. No scorer or evaluator imported, no regrading/model calls.',
        'interpretation': ['Categories overlap and count affected aborted episodes, not individual repeated repair events.',
            'Context overflow requires an actual emitted exception/error field with numeric token/context evidence or TokenBudgetExceeded; prompt content is never searched.',
            'Timeout/deadline/budget observations do not establish whether model behavior, speed or hardware was the ultimate cause.',
            'Generated schema/truncation evidence is retained separately from technical context/CUDA/IPC/artifact/compatibility errors.',
            'Saved valid native/model aborts are closed study results and are not selected for reruns.',
            'A zero category count is exhaustive only for the saved closed records/traces successfully read at this audit timestamp. Missing or unreadable traces remain explicit.']}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'read_errors': len(read_errors), 'trace_read_error_count': result['trace_read_error_count'], 'cells': [
        {key: cell[key] for key in ('experiment', 'closed', 'native_completed', 'assisted_goals', 'aborted', 'missing', 'in_progress_unclosed', 'unusable', 'abort_categories', 'terminal_reasons', 'abort_error_messages')}
        for cell in cells.values()]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
