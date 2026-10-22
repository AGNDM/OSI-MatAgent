"""Export a Slurm-only saved-record progress snapshot of the full rescue grid.

Only exact declared production run paths in the frozen mode are inspected.
Development smokes, other studies and other modes cannot enter the result.
No model inference, native simulation or regrading is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

# Explicitly supplied by the CLI or by an importing auditor. Recovery evidence
# remains bounded to this root; relocation never rewrites frozen input bytes.
EVIDENCE_ROOT = None


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + os.environ['SLURM_JOB_ID'] + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def count(value):
    return type(value) is int and value >= 0


def top(counter, limit=20):
    return [{'value': key, 'count': value} for key, value in counter.most_common(limit)]


def usage_summary(episodes):
    wall = []
    policy_wall = []
    tokens = {field: {'known_partial_total': 0, 'observed_n': 0, 'exact_episode_n': 0}
              for field in ('input', 'output')}
    calls = []
    partial_decisions = 0
    for row in episodes:
        if finite(row.get('recorded_episode_wall_s')):
            wall.append(row['recorded_episode_wall_s'])
        if row.get('status', 'ok') == 'ok':
            if finite(row.get('policy_latency_s')):
                policy_wall.append(row['policy_latency_s'])
            for field in tokens:
                key = field + '_tokens'
                if row.get('token_accounting_complete') is True and count(row.get(key)):
                    tokens[field]['known_partial_total'] += row[key]
                    tokens[field]['observed_n'] += 1
                    tokens[field]['exact_episode_n'] += 1
                else:
                    observed = [step.get('usage', {}) for step in row.get('decision_records', [])
                                if step.get('usage', {}).get(key + '_known') is True
                                and count(step.get('usage', {}).get(key))]
                    if observed:
                        tokens[field]['known_partial_total'] += sum(value[key] for value in observed)
                        tokens[field]['observed_n'] += 1
                    if row.get('decision_records') and len(observed) == len(row['decision_records']):
                        tokens[field]['exact_episode_n'] += 1
            observed_calls = [step.get('usage', {}).get('graph_model_calls')
                              for step in row.get('decision_records', [])]
            if observed_calls and all(count(value) for value in observed_calls):
                calls.append(sum(observed_calls))
        else:
            # This already sums the whole graph trace. last_policy_call_usage is
            # a subset and must never be added to it.
            usage = row.get('observed_graph_model_usage') or {}
            for field in tokens:
                key = field + '_tokens'
                value = usage.get(key)
                if count(value) and (usage.get(key + '_known') is True or value > 0):
                    tokens[field]['known_partial_total'] += value
                    tokens[field]['observed_n'] += 1
                if usage.get(key + '_known') is True and count(value):
                    tokens[field]['exact_episode_n'] += 1
            if count(usage.get('graph_model_calls')):
                calls.append(usage['graph_model_calls'])
            partial_decisions += row.get('partial_native_decisions_observed', 0) or 0
    n = len(episodes)
    for value in tokens.values():
        value['known_partial_total'] = value['known_partial_total'] if value['observed_n'] else None
        value['exact_episode_coverage'] = value['exact_episode_n'] / n if n else None
        value['exact_total'] = value['known_partial_total'] if n and value['exact_episode_n'] == n else None
    return {'recorded_episode_wall_seconds': sum(wall) if wall else None,
            'recorded_episode_wall_n': len(wall),
            'recorded_policy_wall_seconds_completed_episodes': sum(policy_wall) if policy_wall else None,
            'recorded_policy_wall_n': len(policy_wall),
            'tokens': tokens, 'recorded_graph_model_calls': sum(calls) if calls else None,
            'graph_model_call_episode_n': len(calls),
            'aborted_partial_native_decisions': partial_decisions,
            'hosted_api_cost_usd': 0.0, 'gpu_active_seconds': None, 'actual_compute_cost': None,
            'note': 'Saved episode records only; in-progress/unwritten episodes and model initialization outside episode timing are not invented. Known partial token sums are lower bounds when exact coverage is incomplete. Wall seconds are not GPU kernel time.'}


def original_action_metrics(named_episodes, aborted_corrections):
    """Observed native proposals in all closed attempts; never invent missing steps."""
    proposals = correct = corrections = proposal_episodes = correction_episodes = 0
    completed_proposals = completed_correct = completed_corrections = completed = 0
    for name, row in named_episodes:
        if row.get('status', 'ok') == 'ok':
            steps = row['decision_records']
            observed, accepted = len(steps), sum(step['proposal_correct'] for step in steps)
            corrected = sum(step['corrected'] for step in steps)
            completed += 1
            completed_proposals += observed
            completed_correct += accepted
            completed_corrections += corrected
        else:
            observed = row.get('partial_native_decisions_observed')
            accepted = row.get('partial_native_correct_original_proposals')
            if not (count(observed) and count(accepted) and accepted <= observed):
                continue
            corrected = 0 if observed == 0 else aborted_corrections.get((name, row['task_id']))
        proposals += observed
        correct += accepted
        proposal_episodes += 1
        if count(corrected):
            corrections += corrected
            correction_episodes += 1
    total = len(named_episodes)
    all_proposals_known = proposal_episodes == total
    all_corrections_known = correction_episodes == total
    return {'closed_attempts': total, 'completed_native_episodes': completed,
        'aborted_closed_attempts': total - completed,
        'completed_episode_original_proposals': completed_proposals,
        'completed_episode_correct_original_proposals': completed_correct,
        'completed_episode_reference_corrections': completed_corrections,
        'observed_original_proposals': proposals if all_proposals_known else None,
        'correct_original_proposals': correct if all_proposals_known else None,
        'original_proposal_accuracy_micro': correct / proposals if proposals and all_proposals_known else None,
        'known_partial_observed_original_proposals': proposals,
        'known_partial_correct_original_proposals': correct,
        'closed_attempts_with_proposal_counts': proposal_episodes,
        'proposal_count_coverage': proposal_episodes / total if total else None,
        'all_closed_attempt_proposal_counts_known': all_proposals_known,
        'reference_corrections': corrections if all_corrections_known else None,
        'known_partial_reference_corrections': corrections,
        'closed_attempts_with_correction_counts': correction_episodes,
        'correction_count_coverage': correction_episodes / total if total else None,
        'all_closed_attempt_correction_counts_known': all_corrections_known,
        'definition': 'Native original proposals from completed decision_records plus saved abort partial counters, including auxiliary queries. Corrections in aborted attempts require matching saved partial step JSON; zero observed steps contribute zero corrections. Missing/unwritten episodes have no invented action denominator. Partial abort actions do not become completed-episode or outcome metrics.'}


def read_abort_corrections(episode_path, row, mode, name):
    """Reconcile actual saved partial steps before counting their corrections."""
    expected = row.get('partial_native_decisions_observed')
    accepted = row.get('partial_native_correct_original_proposals')
    if not (count(expected) and count(accepted) and accepted <= expected) or expected == 0:
        return None, [], None
    folder = episode_path.parent / 'steps' / sha(row['task_id'].encode())[:24] / mode
    files, steps = [], []
    try:
        paths = sorted(path for path in folder.glob('*.json') if re.fullmatch(r'\d{4}\.json', path.name))
        for position, path in enumerate(paths, 1):
            data = path.read_bytes()
            files.append({'path': str(path), 'sha256': sha(data), 'bytes': len(data),
                          'experiment': name, 'task_id': row['task_id']})
            step = json.loads(data)
            if (path.name != f'{position:04d}.json' or step.get('decision_index') != position
                    or type(step.get('proposal_correct')) is not bool or type(step.get('corrected')) is not bool
                    or step['corrected'] != (mode == 'teacher_forced' and not step['proposal_correct'])):
                raise ValueError('Partial step identity or original-grade/correction flags are inconsistent')
            steps.append(step)
        if len(steps) != expected or sum(step['proposal_correct'] for step in steps) != accepted:
            raise ValueError('Saved partial steps do not reconcile with the closed abort counters')
        return sum(step['corrected'] for step in steps), files, None
    except (OSError, ValueError, TypeError, KeyError) as error:
        return None, files, {'path': str(folder), 'error': f'{type(error).__name__}: {error}'}


def parse_slot_selection(selection, job_count):
    slots = set()
    for part in selection.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        if not match:
            raise ValueError('Expected explicit array indices or ranges')
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if not 0 <= first <= last < job_count:
            raise ValueError('Submitted selection contains an undeclared worker')
        values = set(range(first, last + 1))
        if slots & values:
            raise ValueError('Submitted selection repeats a worker slot')
        slots.update(values)
    return slots


def submitted_arrays(text, jobs):
    """Bind every scheduler array to its exact declared global worker slots."""
    arrays = {}
    owners = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 4:
            raise ValueError('Malformed submitted.tsv row')
        batch, submitted_id, selection, deadline = fields
        array_id = submitted_id.split(';')[0]
        if not re.fullmatch(r'\d+', array_id) or array_id in arrays:
            raise ValueError('Malformed or repeated submitted array ID')
        slots = parse_slot_selection(selection, len(jobs))
        for slot in slots:
            if slot in owners:
                raise ValueError('Submitted arrays declare overlapping worker slots')
            owners[slot] = array_id
        arrays[array_id] = {'batch': batch, 'slots': slots, 'deadline': deadline}
    if not arrays:
        raise ValueError('No parseable submitted arrays')
    return arrays


def recovery_ownership(study, jobs, original_arrays, original_submitted_sha256, *,
                       stage='initial', supplemental_to=None):
    """Authenticate explicit pre-Python/unstarted replacements without new retries."""
    if stage not in {'initial', 'late'}:
        raise ValueError('Only the explicitly declared initial and late recovery stages are supported')
    manifest_path = study / ('scheduler-recovery/manifest.json' if stage == 'initial' else 'scheduler-recovery-late/manifest.json')
    submitted_path = study / ('recovery-submitted.tsv' if stage == 'initial' else 'recovery-late-submitted.tsv')
    if not manifest_path.exists() and not submitted_path.exists():
        return {}, {}, {'status': 'not_present'}
    data, submitted = manifest_path.read_bytes(), submitted_path.read_bytes()
    manifest = json.loads(data)
    if stage == 'late':
        parent = manifest.get('supplemental_to') or {}
        if (not supplemental_to or supplemental_to.get('status') != 'authenticated'
                or Path(parent['manifest_path']).resolve() != Path(supplemental_to['manifest']['path']).resolve()
                or parent['manifest_sha256'] != supplemental_to['manifest']['sha256']
                or Path(parent['mapping_path']).resolve() != Path(supplemental_to['submission_file']['path']).resolve()
                or parent['mapping_sha256'] != supplemental_to['submission_file']['sha256']):
            raise ValueError('Late recovery does not authenticate the unchanged initial mapping')
    prepared = read_json(study / 'prepared.json')
    if (manifest.get('schema_version') != 1 or manifest.get('status') != 'submitted'
            or Path(manifest['study']).resolve() != study
            or manifest.get('original_submitted_sha256') != original_submitted_sha256
            or manifest.get('recovery_submitted_sha256') != sha(submitted)
            or manifest.get('frozen_files_modified') is not False):
        raise ValueError('Recovery declaration is incomplete or belongs to another study/submission')
    for key in ('source_manifest_sha256', 'configuration_sha256', 'jobs_sha256', 'deadline_scope_sha256'):
        if manifest.get(key) != prepared[key]:
            raise ValueError('Recovery changed frozen study identity: ' + key)
    evidence_cache = {}
    def verified(path, expected):
        path = Path(path).resolve()
        if EVIDENCE_ROOT is None or not path.is_relative_to(EVIDENCE_ROOT):
            raise ValueError('Recovery evidence escaped the explicitly authorized evidence root')
        if path not in evidence_cache:
            evidence_cache[path] = path.read_bytes()
        if sha(evidence_cache[path]) != expected:
            raise ValueError('Recovery evidence changed: ' + str(path))
        return evidence_cache[path]
    for path_key, sha_key in (('recovery_helper', 'recovery_helper_sha256'), ('wrapper', 'wrapper_sha256')):
        verified(manifest[path_key], manifest[sha_key])
    arrays = submitted_arrays(submitted.decode(), jobs)
    if set(arrays) & set(original_arrays):
        raise ValueError('Recovery reused an original array ID')
    eligible = {}
    for row in manifest['eligible_slots']:
        index, old_array = row['job_index'], row['original_array_id']
        if (type(index) is not int or index in eligible or old_array not in original_arrays
                or index not in original_arrays[old_array]['slots']
                or row['original_job_id'] != f'{old_array}_{index}'
                or row['batch'] != original_arrays[old_array]['batch']
                or any(row.get(key) is not True for key in
                       ('closed_episode_absent', 'dispatch_absent', 'worker_configuration_absent'))):
            raise ValueError('Recovery eligibility has invalid original ownership or absence attestations')
        evidence = [(Path(item['path']), verified(item['path'], item['sha256'])) for item in row['evidence']]
        snapshots = [body for path, body in evidence if path.name == 'original-sacct.txt']
        if len(snapshots) != 1:
            raise ValueError('Recovery lacks one authenticated original scheduler snapshot')
        observations = {tuple(line.split('|')) for line in snapshots[0].decode().splitlines()
                        if line.startswith(row['original_job_id'] + '|')}
        if len(observations) != 1:
            raise ValueError('Recovery eligibility original scheduler row is missing or conflicting')
        fields = next(iter(observations))
        if len(fields) != 5 or fields[1] != row['original_state'] or fields[2] != row['original_exit_code']:
            raise ValueError('Recovery eligibility differs from its original scheduler evidence')
        reason, state = row['eligibility_reason'], row['original_state'].split()[0]
        if reason == 'original_pending_cancelled_for_cpu_binding_recovery':
            if state != 'CANCELLED' or fields[3] not in ('None', 'Unknown', ''):
                raise ValueError('A replacement cancelled worker had already started')
        elif reason == 'proven_pre_python_cpu_binding_failure':
            logs = [body.decode() for path, body in evidence if path.suffix == '.log']
            if (state != 'FAILED' or not any('CPU binding outside of job step allocation' in body
                    and 'Unable to satisfy cpu bind request' in body and 'Traceback' not in body
                    and 'Loading checkpoint' not in body for body in logs)):
                raise ValueError('Replacement does not prove the pre-Python CPU binding failure')
        else:
            raise ValueError('Unapproved recovery eligibility reason')
        eligible[index] = row
    if manifest.get('eligible_n') != len(eligible):
        raise ValueError('Recovery eligibility count differs from its declared inventory')
    bindings = {}
    for array_id, declaration in arrays.items():
        for index in declaration['slots']:
            if index not in eligible or declaration['batch'] != eligible[index]['batch']:
                raise ValueError('A replacement array owns an ineligible or wrong-batch slot')
            if declaration['deadline'] != original_arrays[eligible[index]['original_array_id']]['deadline']:
                raise ValueError('Recovery changed the original scheduler deadline')
            bindings[str(index)] = {'original_array_id': eligible[index]['original_array_id'],
                'recovery_stage': stage,
                'original_job_id': eligible[index]['original_job_id'], 'replacement_array_id': array_id,
                'replacement_job_id': f'{array_id}_{index}', 'eligibility_reason': eligible[index]['eligibility_reason'],
                'original_state_at_selection': eligible[index]['original_state'],
                'original_exit_code_at_selection': eligible[index]['original_exit_code'],
                'closed_episode_absent_at_selection': True, 'dispatch_absent_at_selection': True,
                'worker_configuration_absent_at_selection': True}
    if set(map(int, bindings)) != set(eligible):
        raise ValueError('Submitted recovery arrays do not exactly cover the eligible inventory')
    declared = {str(row['replacement_array_id']): row for row in manifest['submissions']}
    if set(declared) != set(arrays):
        raise ValueError('Recovery TSV array IDs differ from the manifest')
    for array_id, declaration in arrays.items():
        row = declared[array_id]
        if (row['batch'] != declaration['batch'] or row['deadline'] != declaration['deadline']
                or parse_slot_selection(row['indices'], len(jobs)) != declaration['slots']
                or row['count'] != len(declaration['slots'])
                or any(eligible[index]['original_array_id'] != row['original_array_id'] for index in declaration['slots'])):
            raise ValueError('Recovery TSV ownership differs from its manifest submission')
    return arrays, bindings, {'status': 'authenticated', 'stage': stage, 'manifest': {'path': str(manifest_path), 'sha256': sha(data)},
        'submission_file': {'path': str(submitted_path), 'sha256': sha(submitted)},
        'eligible_slots': len(eligible), 'eligibility_counts': dict(Counter(row['eligibility_reason'] for row in eligible.values())),
        'ownership_policy': 'Each declared slot has one effective scheduler owner. Only explicitly attested pre-Python/unstarted slots switch to their replacement; original allocation states remain audit data.',
        'absence_attestations_scope': 'At recovery selection time, before replacement execution; replacement outputs may now exist.',
        'by_slot': bindings}


def parse_sacct_rows(stdout, arrays):
    """Accept complete task IDs only; publish no map until all rows validate.

    sacct can repeat an identical allocation row. That is one observation of
    one allocation, not a second owner. Conflicting duplicate rows, abbreviated
    IDs and unexpanded array summaries are rejected rather than guessed.
    """
    by_slot = {}
    exact_duplicates = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split('|')
        if len(fields) != 10:
            raise ValueError('Malformed sacct row: expected ten complete fields')
        raw, display, state, partition, elapsed, tres, node, start, end, exit_code = fields
        # Allocation-only queries may include the array's parent summary.
        if display in arrays:
            continue
        match = re.fullmatch(r'(\d+)_(\d+)', display)
        if not match or match.group(1) not in arrays or not raw.isdigit():
            raise ValueError('sacct returned an abbreviated, compressed or unexpected task ID: ' + display)
        array_id, index = match.group(1), int(match.group(2))
        if index not in arrays[array_id]['slots']:
            raise ValueError('sacct task does not belong to its submitted array range: ' + display)
        if not elapsed.isdigit() or not state or '+' in state:
            raise ValueError('sacct returned an incomplete state or elapsed value: ' + display)
        for value in (start, end):
            if value not in ('', 'None', 'Unknown') and not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                raise ValueError('sacct returned an incomplete timestamp: ' + display)
        row = {'job_id_raw': raw, 'job_id': display, 'array_id': array_id,
               'state': state.split()[0], 'state_detail': state, 'partition': partition,
               'elapsed_seconds': int(elapsed), 'allocated_tres': tres, 'node': node,
               'start_utc': start + '+00:00' if start not in ('', 'None', 'Unknown') else None,
               'end_utc': end + '+00:00' if end not in ('', 'None', 'Unknown') else None,
               'exit_code': exit_code}
        key = str(index)
        if key in by_slot:
            if by_slot[key] != row:
                raise ValueError('Conflicting scheduler observations or allocations for declared slot ' + key)
            exact_duplicates += 1
            continue
        by_slot[key] = row
    return by_slot, exact_duplicates


def combined_recovery_ownership(study, jobs, arrays, submitted_sha256):
    initial_arrays, initial_bindings, initial = recovery_ownership(study, jobs, arrays, submitted_sha256)
    late_arrays, late_bindings, late = recovery_ownership(study, jobs, arrays, submitted_sha256,
                                                        stage='late', supplemental_to=initial)
    if late.get('status') == 'not_present':
        return initial_arrays, initial_bindings, initial
    if set(initial_arrays) & set(late_arrays) or set(initial_bindings) & set(late_bindings):
        raise ValueError('Late recovery overlaps an initial replacement array or slot')
    bindings = {**initial_bindings, **late_bindings}
    counts = Counter(initial['eligibility_counts']) + Counter(late['eligibility_counts'])
    summary = {**initial, 'eligible_slots': len(bindings), 'eligibility_counts': dict(counts),
        'initial_eligible_slots': len(initial_bindings), 'late_eligible_slots': len(late_bindings),
        'stages': {'initial': initial, 'late': late}, 'by_slot': bindings}
    return {**initial_arrays, **late_arrays}, bindings, summary


def scheduler_snapshot(study, jobs, enabled):
    result = {'requested': enabled, 'source': 'sacct read-only', 'states': {}, 'by_slot': {}}
    submitted = study / 'submitted.tsv'
    if not enabled:
        result['status'] = 'not_queried'
        return result
    if not submitted.exists():
        result['status'] = 'no_submission_file_present'
        return result
    try:
        data = submitted.read_bytes()
        result['submission_file'] = {'path': str(submitted), 'sha256': sha(data)}
        arrays = submitted_arrays(data.decode(), jobs)
        replacements, bindings, recovery = combined_recovery_ownership(study, jobs, arrays, sha(data))
        result['recovery'] = recovery
        result['queries'] = []
        def query_arrays(declarations, cohort):
            command = ['sacct', '-X', '--array', '-n', '-P', '-j', ','.join(declarations), '-o',
                       'JobIDRaw%40,JobID%200,State%100,Partition%100,ElapsedRaw,AllocTRES%4096,NodeList%4096,Start,End,ExitCode']
            query = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False,
                                   env={**os.environ, 'TZ': 'UTC', 'SLURM_TIME_FORMAT': 'standard'})
            result['queries'].append({'cohort': cohort, 'command': command, 'returncode': query.returncode,
                'stdout_sha256': sha(query.stdout.encode()), 'stderr': query.stderr})
            result.update(command=command, returncode=query.returncode,
                stdout_sha256=sha(query.stdout.encode()), stderr=query.stderr,
                status='available' if query.returncode == 0 else 'query_failed')
            return None if query.returncode else parse_sacct_rows(query.stdout, declarations)
        original_answer = query_arrays(arrays, 'original')
        if original_answer is None:
            return result
        original, duplicates = original_answer
        by_slot = {key: dict(row, owner_source='original') for key, row in original.items()}
        replacement_rows = {}
        if replacements:
            replacement_answer = query_arrays(replacements, 'replacement')
            if replacement_answer is None:
                return result
            replacement_rows, duplicate_replacements = replacement_answer
            duplicates += duplicate_replacements
            for key, binding in bindings.items():
                old = original.get(key)
                if (old is None or old['job_id'] != binding['original_job_id']
                        or old['state'] not in {'FAILED', 'CANCELLED'}
                        or old['exit_code'] != binding['original_exit_code_at_selection']):
                    raise ValueError('Replacement original owner is not a matching terminated allocation: ' + key)
                by_slot.pop(key, None)
                if key in replacement_rows:
                    new = replacement_rows[key]
                    if new['job_id'] != binding['replacement_job_id']:
                        raise ValueError('Replacement observation differs from explicit recovery ownership')
                    by_slot[key] = dict(new, owner_source='replacement', original_job_id=old['job_id'])
        result.update(by_slot=by_slot,
            states=dict(Counter(row['state'] for row in by_slot.values())),
            matched_declared_slots=len(by_slot), exact_duplicate_rows_collapsed=duplicates,
            allocated_worker_count=sum(row['start_utc'] is not None for row in by_slot.values()),
            workers_without_allocation_start=sum(row['start_utc'] is None for row in by_slot.values()))
        if replacements:
            result.update(original_by_slot=original, replacement_by_slot=replacement_rows,
                original_states=dict(Counter(row['state'] for row in original.values())),
                replacement_states=dict(Counter(row['state'] for row in replacement_rows.values())),
                original_allocated_worker_count=sum(row['start_utc'] is not None for row in original.values()))
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
        # Never leak the prefix of an invalid ownership map into case records.
        result.update(status='unavailable', error=f'{type(error).__name__}: {error}', states={}, by_slot={})
        for field in ('original_by_slot', 'replacement_by_slot', 'original_states', 'replacement_states',
                      'matched_declared_slots', 'allocated_worker_count', 'workers_without_allocation_start'):
            result.pop(field, None)
    return result


def main():
    global EVIDENCE_ROOT
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True,
                        help='Authorized parent of the study and its recorded recovery evidence')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--stopped-study', action='store_true',
                        help='Require complete terminal accounting and label this as a final stopped-study snapshot')
    parser.add_argument('--report-launcher', type=Path, required=True,
                        help='The actual reporting wrapper, included by path and hash in provenance')
    args = parser.parse_args()
    EVIDENCE_ROOT = args.evidence_root.resolve()
    study = args.study.resolve()
    if not study.is_relative_to(EVIDENCE_ROOT):
        raise ValueError('Study escaped the explicitly authorized evidence root')
    launcher = args.report_launcher.resolve()
    if not launcher.is_file():
        raise ValueError('The actual reporting launcher must exist')
    prepared = read_json(study / 'prepared.json')
    documents = {}
    for filename, key in (('configuration.json', 'configuration_sha256'), ('jobs.json', 'jobs_sha256'),
                          ('source-manifest.json', 'source_manifest_sha256'),
                          ('deadline-scope.json', 'deadline_scope_sha256')):
        data = (study / filename).read_bytes()
        if sha(data) != prepared[key]:
            raise ValueError('Frozen preparation changed: ' + filename)
        documents[filename] = json.loads(data)
    config, jobs, manifest, scope = (documents[name] for name in
        ('configuration.json', 'jobs.json', 'source-manifest.json', 'deadline-scope.json'))
    mode = scope['mode']
    if len(config['experiments']) != 22 or len(jobs) != 2640 or mode not in {'free_run', 'teacher_forced'} or scope['seed'] != 0:
        raise ValueError('Expected the declared 22-cell seed-zero grid in one explicit mode')
    output = args.output.resolve()
    other_name = {'free_run': 'workflow-teacher-forced-results.json',
                  'teacher_forced': 'workflow-deadline-results.json'}[mode]
    if output.name == other_name or (output.exists() and read_json(output).get('mode') != mode):
        raise ValueError('Cannot overwrite an export for the other evaluation mode')
    if output.is_relative_to(study):
        raise ValueError('Write exports outside the authenticated study')
    rate_key = 'autonomous_success_rate' if mode == 'free_run' else 'assisted_goal_completion_rate'
    snapshot = study / 'snapshot'
    required = ('instrument/sequential_120/cases.jsonl',
                'instrument/sequential_120/evaluator/case-specs.jsonl',
                'instrument/eval/sequential.py', 'instrument/eval/common.py',
                'provenance/application.json')
    reporting_inputs = {}
    for relative in required:
        data = (snapshot / relative).read_bytes()
        if manifest.get(relative) != sha(data):
            raise ValueError('Frozen reporting input changed: ' + relative)
        reporting_inputs[relative] = sha(data)
    public = [json.loads(line) for line in (snapshot / required[0]).read_text().splitlines() if line.strip()]
    cases = [json.loads(line) for line in (snapshot / required[1]).read_text().splitlines() if line.strip()]
    case_index = {row['task_id']: row for row in cases}
    if len(public) != 120 or len(case_index) != 120 or {row['task_id'] for row in public} != set(case_index):
        raise ValueError('Frozen public and evaluator case inventories differ')
    # Import only the frozen standard-library saved-record scorer, not a runtime.
    sys.path.insert(0, str(snapshot / 'instrument/eval'))
    spec = importlib.util.spec_from_file_location('deadline_saved_episode_scorer', snapshot / 'instrument/eval/sequential.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    by_name = {row['name']: row for row in config['experiments']}
    declared = {(row['experiment'], row['shard_index']) for row in jobs}
    if len(declared) != 2640 or declared != {(name, index) for name in by_name for index in range(120)}:
        raise ValueError('Jobs do not contain each declared experiment/case exactly once')
    scheduler = scheduler_snapshot(study, jobs, args.scheduler)
    if args.stopped_study:
        terminal = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
                    'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}
        if (scheduler.get('status') != 'available'
                or scheduler.get('matched_declared_slots') != len(jobs)
                or set(scheduler['states']) - terminal):
            raise ValueError('Stopped-study export requires complete terminal scheduler accounting')
    records = defaultdict(dict)
    case_presence = defaultdict(dict)
    source_files = []
    read_errors = []
    trailing_fragments = []
    for index, job in enumerate(jobs):
        name, shard = job['experiment'], job['shard_index']
        task_id = public[shard]['task_id']
        if job['shards'] != 120 or job['mode'] != mode:
            raise ValueError('A declared worker changed its one-case selection or frozen mode')
        worker = study / 'runs' / name / 'sequential' / f'shard-{shard:03d}'
        episode_file = worker / name / 'sequential/episodes.jsonl'
        presence = {'task_id': task_id, 'job_index': index, 'record_status': 'missing',
                    'native_outcome_met': None, 'worker_configuration_present': (worker / 'configuration.json').exists()}
        if str(index) in scheduler['by_slot']:
            presence['slurm'] = scheduler['by_slot'][str(index)]
        if str(index) in scheduler.get('recovery', {}).get('by_slot', {}):
            presence['scheduler_recovery'] = scheduler['recovery']['by_slot'][str(index)]
            if str(index) in scheduler.get('original_by_slot', {}):
                presence['slurm_original'] = scheduler['original_by_slot'][str(index)]
        skip = study / 'deadline-skips' / (str(index) + '.json')
        if skip.exists():
            skipped = read_json(skip)
            if skipped.get('job_index') != index or skipped.get('job') != job:
                raise ValueError('Deadline skip belongs to another declared worker')
            presence['record_status'] = 'deadline_not_started'
        if episode_file.exists():
            data = episode_file.read_bytes()
            source_files.append({'path': str(episode_file), 'sha256': sha(data), 'bytes': len(data),
                                 'experiment': name, 'task_id': task_id})
            lines = data.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b'\n'):
                lines.pop()
                trailing_fragments.append(str(episode_file))
            try:
                configuration = read_json(worker / 'configuration.json')
                if (configuration.get('selected_experiments') != [name]
                        or configuration.get('selected_task_ids') != [task_id]
                        or configuration.get('shards') != 120 or configuration.get('shard_index') != shard
                        or configuration.get('mode') != mode or configuration.get('limit') is not None
                        or configuration.get('source_sha256', {}).get('configuration_input') != prepared['configuration_sha256']):
                    raise ValueError('Production worker configuration differs from the frozen one-case declaration')
                if len(lines) > 1:
                    raise ValueError('A one-case worker contains multiple episode records')
                if lines:
                    row = json.loads(lines[0])
                    if row.get('task_id') != task_id or row.get('mode') != mode:
                        raise ValueError('Record does not match the declared production case/mode')
                    scorer._validate_episode(row)
                    records[name][task_id] = row
                    presence.update(record_status=row.get('status', 'ok'),
                        native_outcome_met=row['trajectory_outcome_met'] if row.get('status', 'ok') == 'ok' else None,
                        episode_source=str(episode_file))
            except (OSError, ValueError, TypeError, KeyError) as error:
                presence['record_status'] = 'unusable_record'
                read_errors.append({'path': str(episode_file), 'error': f'{type(error).__name__}: {error}'})
        case_presence[name][task_id] = presence

    cells = []
    all_rows = []
    all_named_rows = []
    aborted_corrections = {}
    partial_step_files, partial_step_errors = [], []
    cohort_groups = {field: dict(sorted(Counter(scorer._metadata(case, field) for case in cases).items()))
                     for field in ('workflow_id', 'instrument_id', 'condition')}
    for name, experiment in by_name.items():
        episodes = list(records[name].values())
        score = scorer.score_episodes(cases, episodes, modes=(mode,))['by_mode'][mode]
        failures = [row for row in episodes if row.get('status', 'ok') != 'ok']
        for row in failures:
            episode_path = Path(case_presence[name][row['task_id']]['episode_source'])
            corrected, files, error = read_abort_corrections(episode_path, row, mode, name)
            aborted_corrections[(name, row['task_id'])] = corrected
            partial_step_files.extend(files)
            if error:
                partial_step_errors.append(error)
        action_metrics = original_action_metrics([(name, row) for row in episodes], aborted_corrections)
        if (action_metrics['completed_episode_original_proposals'] != score['decisions_observed']
                or action_metrics['completed_episode_correct_original_proposals'] != score['correct_original_proposals']
                or action_metrics['completed_episode_reference_corrections'] != score['interventions']):
            raise ValueError('Completed-only action metrics differ from the frozen native scorer')
        reasons = Counter(row.get('graph_terminal_reason') or 'unspecified' for row in failures)
        classifications = Counter(value for row in failures for value in row.get('model_response_classifications', []))
        errors = Counter(row.get('error', 'unspecified') for row in failures)
        counts = Counter(value['record_status'] for value in case_presence[name].values())
        cells.append({'experiment': name, 'expert': experiment.get('expert_row'),
            'chooser': experiment.get('chooser_column'), 'batch': experiment.get('deadline_batch'),
            'seed': 0, 'mode': mode, 'expected_cases': 120,
            'episode_records': len(episodes), 'valid_native_episodes': score['completed'],
            'verified_outcome_successes': score['goals_reached'], rate_key: score[rate_key],
            'aborted_episodes': len(failures), 'missing_episode_records': score['missing'],
            'deadline_not_started': counts['deadline_not_started'], 'unusable_records': counts['unusable_record'],
            'raw_record_presence_counts': dict(counts), 'native_score': score,
            'original_action_metrics_all_closed_attempts': action_metrics,
            'usage': usage_summary(episodes), 'graph_terminal_reasons': dict(reasons),
            'model_response_classifications': dict(classifications), 'top_abort_errors': top(errors),
            'configuration': {key: experiment.get(key) for key in
                ('max_model_calls', 'episode_seconds', 'call_timeout_seconds', 'graph_steps', 'expert', 'backend')},
            'cases': [case_presence[name][case['task_id']] for case in public]})
        all_rows.extend(episodes)
        all_named_rows.extend((name, row) for row in episodes)
    expected = len(jobs)
    progress = {'declared_episode_count': expected, 'episode_records': len(all_rows),
        'attempt_record_coverage': len(all_rows) / expected,
        'valid_native_episodes': sum(cell['valid_native_episodes'] for cell in cells),
        'verified_outcome_successes': sum(cell['verified_outcome_successes'] for cell in cells),
        'aborted_episodes': sum(cell['aborted_episodes'] for cell in cells),
        'missing_episode_records': sum(cell['missing_episode_records'] for cell in cells),
        'deadline_not_started': sum(cell['deadline_not_started'] for cell in cells),
        'unusable_records': sum(cell['unusable_records'] for cell in cells)}
    progress['cohort_' + rate_key] = progress['verified_outcome_successes'] / expected
    progress['observed_original_proposals'] = sum(cell['native_score']['decisions_observed'] for cell in cells)
    progress['correct_original_proposals'] = sum(cell['native_score']['correct_original_proposals'] for cell in cells)
    progress['observed_original_proposal_accuracy_micro'] = (progress['correct_original_proposals'] / progress['observed_original_proposals']
        if progress['observed_original_proposals'] else None)
    progress['reference_corrections'] = sum(cell['native_score']['interventions'] for cell in cells)
    progress['original_proposal_metric_scope'] = 'Completed native episodes only; see original_action_metrics_all_closed_attempts for completed plus aborted closed attempts.'
    progress['reference_corrections_scope'] = 'Completed native episodes only.'
    progress['reference_corrections_closed_attempt_coverage'] = progress['valid_native_episodes'] / len(all_rows) if all_rows else None
    progress['valid_native_episode_coverage'] = progress['valid_native_episodes'] / expected
    if progress['episode_records'] + progress['missing_episode_records'] != expected:
        raise ValueError('Full-grid denominator reconciliation failed')
    application = read_json(snapshot / 'provenance/application.json')
    by_batch = {}
    for batch in dict.fromkeys(cell['batch'] for cell in cells):
        selected_cells = [cell for cell in cells if cell['batch'] == batch]
        expected_batch = sum(cell['expected_cases'] for cell in selected_cells)
        batch_counts = {field: sum(cell[field] for cell in selected_cells) for field in
            ('episode_records', 'valid_native_episodes', 'verified_outcome_successes',
             'aborted_episodes', 'missing_episode_records', 'deadline_not_started', 'unusable_records')}
        by_batch[batch] = {'cells': len(selected_cells), 'declared_episode_count': expected_batch,
                          **batch_counts, rate_key: batch_counts['verified_outcome_successes'] / expected_batch}
    failures = [row for row in all_rows if row.get('status', 'ok') != 'ok']
    result = {'schema_version': 1, 'benchmark': 'Instrument sequential workflow',
        'mode': mode,
        'outcome_semantics': ('Assisted native goal completion after grading original proposals and applying reference corrections; not autonomous success.'
                              if mode == 'teacher_forced' else 'Autonomous native goal completion without reference corrections.'),
        'protocol': scope['protocol'], 'study': str(study),
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'compute_node': os.environ['SLURMD_NODENAME'],
        'status': ('stopped_incomplete_grid' if len(all_rows) != expected else 'stopped_all_records_present')
                  if args.stopped_study else ('all_episode_records_present' if len(all_rows) == expected else 'provisional_partial_coverage'),
        'study_stopped': args.stopped_study,
        'cohort': {'cells': 22, 'cases_per_cell': 120, 'declared_episode_count': 2640, 'seed': 0,
            'mode': mode, 'deadline_utc': scope['deadline_utc'], 'group_counts_per_cell': cohort_groups,
            'train_test_split_assigned': False, 'development_smokes_included': False},
        'progress': progress, 'by_batch': by_batch, 'usage': usage_summary(all_rows),
        'original_action_metrics_all_closed_attempts': original_action_metrics(all_named_rows, aborted_corrections),
        'graph_terminal_reasons': dict(Counter(row.get('graph_terminal_reason') or 'unspecified' for row in failures)),
        'model_response_classifications': dict(Counter(value for row in failures for value in row.get('model_response_classifications', []))),
        'top_abort_errors': top(Counter(row.get('error', 'unspecified') for row in failures)),
        'scheduler': scheduler, 'cells': cells,
        'provenance': {'app_revision': application.get('git_commit'), 'application': application,
            'source_manifest_sha256': prepared['source_manifest_sha256'],
            'configuration_sha256': prepared['configuration_sha256'], 'jobs_sha256': prepared['jobs_sha256'],
            'deadline_scope_sha256': prepared['deadline_scope_sha256'], 'reporting_inputs_sha256': reporting_inputs,
            'export_helper': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
            'report_launcher': {'path': str(launcher), 'sha256': sha(launcher.read_bytes())},
            'source_episode_files': source_files, 'neural_inference_performed': False,
            'source_aborted_partial_step_files': partial_step_files,
            'full_frozen_inventory_reverified': False,
            'integrity_scope': 'Frozen config/jobs/deadline scope and reporting input bytes authenticated; workers authenticate their complete frozen snapshot before execution.'},
        'source_read_errors': read_errors, 'source_aborted_partial_step_read_errors': partial_step_errors,
        'in_progress_trailing_fragments': trailing_fragments,
        'definitions': {'success': 'status ok and native trajectory_outcome_met is true; no model completion claim is independently awarded credit',
            'denominator': '120 cases per cell and 2640 declared episodes overall; aborted, unrecorded and deadline-not-started cases remain in cohort denominators',
            'valid_native_episode': 'Native runner returned a structurally consistent status-ok episode, which may still fail its outcome',
            'aborted': 'Saved infrastructure_error envelope; may represent model/protocol/context/budget/runtime failures, not necessarily scheduler or hardware failure',
            'missing': 'No usable closed episode record; includes queued/running/unsubmitted workers, deadline skips and unusable records',
            'offline_trust': 'Frozen scorer checks consistency and aggregates saved native grades; it does not re-simulate arbitrary reported outcomes'},
        'interpretation': [('This study was stopped; its incomplete cohort rates describe only this terminated study and are not full-grid model performance.'
                            if args.stopped_study else 'Partial-coverage cohort rates are provisional lower bounds and must be read with attempt/native completion coverage.'),
            'Only exact production paths under runs/ in the declared frozen mode are included; development-smokes, development-qualification, older protocols and other modes are excluded.',
            'Original proposals are graded before any teacher-forced correction. Assisted goal completion is distinct from autonomous success. native_score retains original-proposal, control/query, checkpoint, stage and intervention metrics.',
            'Grouped scores retain every declared case in their original workflow/instrument/condition denominator.',
            'No wall time, token count or native outcome is invented for an unwritten episode. No new model calls or significance tests occur.']}
    write_atomic(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output), 'progress': progress,
                      'source_read_error_count': len(read_errors), 'scheduler_status': scheduler['status']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
