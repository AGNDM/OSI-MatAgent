"""Independently reconcile frozen E's saved final results; never run a model.

Only this audit's own JSON artifacts are written. The pinned reporter, monitor,
canonical result/status inputs, native episodes and all model sources are read.
Valid native/model/budget aborts remain closed results, not missing episodes.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
STUDY = MAIN = STATUS = PUBLIC_AUDIT = FAILURE_AUDIT = None
TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}
VALID_FAILURE_CATEGORIES = {'expert_schema_error', 'expert_output_truncated', 'expert_empty_output',
    'generated_duplicate_candidate_ids', 'generated_record_or_schema_error',
    'repeated_verification_without_new_evidence', 'timeout_or_deadline',
    'terminal_llm_budget', 'terminal_deadline', 'terminal_action_budget', 'terminal_token_budget_exceeded'}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def file_proof(path):
    data = Path(path).read_bytes()
    return {'path': str(path), 'sha256': sha(data), 'bytes': len(data)}


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def same(left, right):
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) is float or type(right) is float:
        return number(left) and number(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def differences(actual, expected, path=''):
    """Compare every nested published field, preserving nulls and numeric types."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{'field': path, 'error': 'Expected an object'}]
        result = []
        for key in sorted(set(actual) | set(expected)):
            location = path + '.' + str(key) if path else str(key)
            if key not in actual or key not in expected:
                result.append({'field': location, 'error': 'Missing or unexpected field'})
            else:
                result.extend(differences(actual[key], expected[key], location))
        return result
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [{'field': path, 'error': 'List length or type differs'}]
        return [issue for index, value in enumerate(expected)
                for issue in differences(actual[index], value, f'{path}[{index}]')]
    return [] if same(actual, expected) else [{'field': path, 'raw': expected, 'public': actual}]


def native_case_outcome(episode):
    """Use the pinned reporter's exact completed/abort/default-status semantics."""
    return episode['trajectory_outcome_met'] if episode.get('status', 'ok') == 'ok' else None


def case_outcome_errors(published_case, expected):
    observed = ({'native_outcome_met': published_case['native_outcome_met']}
                if 'native_outcome_met' in published_case else {})
    return differences(observed, {'native_outcome_met': expected}, 'case')


def verify_source_bindings(recorded_sources, expected_sources):
    """Bind final failure evidence to exactly the current raw episodes/traces."""
    errors, recorded = [], {}
    for proof in recorded_sources:
        path = Path(proof['path']).resolve()
        if not path.is_relative_to(STUDY / 'runs') or str(path) in recorded:
            errors.append({'path': str(path), 'error': 'Unexpected or duplicate failure-proof source'})
            continue
        recorded[str(path)] = proof
    if set(recorded) != set(expected_sources):
        errors.append({'error': 'Failure-proof source paths differ from all current episodes and aborted traces',
            'missing': sorted(set(expected_sources) - set(recorded)),
            'unexpected': sorted(set(recorded) - set(expected_sources))})
    for path in set(recorded) & set(expected_sources):
        try:
            current = file_proof(path)
            if current != recorded[path] or current != expected_sources[path]:
                errors.append({'path': path, 'error': 'Failure proof is stale or differs from current raw bytes'})
        except (OSError, ValueError, TypeError) as error:
            errors.append({'path': path, 'error': f'{type(error).__name__}: {error}'})
    return errors


def failure_evidence_errors(document, cells, expected_hashes):
    """Require full final coverage and evidenced model/budget failures only."""
    errors = []
    if (document.get('study') != str(STUDY) or document.get('mode') != 'teacher_forced'
            or document.get('seed') != 0
            or any(document.get(key) != value for key, value in expected_hashes.items() if key != 'deadline_scope_sha256')
            or document.get('read_errors') != [] or document.get('trace_read_error_count') != 0):
        errors.append({'error': 'Failure evidence identity/readability differs from frozen full study'})
    rows = document.get('cells', [])
    declared = {row.get('experiment'): row for row in rows}
    if len(rows) != 22 or len(declared) != 22 or set(declared) != set(cells):
        return errors + [{'error': 'Failure evidence does not contain exactly the22 declared cells'}]
    if sum(row.get('closed', 0) for row in rows) != 2640:
        errors.append({'error': 'Failure evidence does not cover2640 closed production episodes'})
    for name, cell in cells.items():
        row = declared[name]
        if cell.get('explicit_initialization_failures', 0):
            errors.append({'experiment': name, 'error': 'Explicit model initialization failures prevent final completion'})
        if (row.get('closed') != cell['closed'] or cell['closed'] != 120
                or row.get('native_completed') != cell['native_completed']
                or row.get('assisted_goals') != cell['assisted_goals']
                or row.get('aborted') != cell['aborted'] or row.get('missing') != 0
                or row.get('unusable') != 0 or row.get('in_progress_unclosed') != 0):
            errors.append({'experiment': name, 'error': 'Failure evidence cell counts are incomplete or differ from raw records'})
        failures = row.get('failures', [])
        expected = {proof['task_id']: proof for proof in cell['source_cases']
                    if proof.get('record_status') == 'infrastructure_error'}
        actual = {failure.get('task_id'): failure for failure in failures}
        if len(failures) != cell['aborted'] or len(actual) != len(failures) or set(actual) != set(expected):
            errors.append({'experiment': name, 'error': 'Failure evidence does not identify every exact aborted case'})
        category_counts = Counter()
        for task_id, failure in actual.items():
            proof = expected.get(task_id)
            if (proof is None or failure.get('job_index') != proof['index']
                    or failure.get('episode_path') != proof['episode']['path']
                    or failure.get('episode_sha256') != proof['episode']['sha256']
                    or failure.get('trace_read_errors') != []):
                errors.append({'experiment': name, 'task_id': task_id, 'error': 'Exact aborted-case source/read proof differs'})
            categories = failure.get('observed_categories')
            if (not isinstance(categories, list) or not categories
                    or any(not isinstance(value, str) for value in categories)
                    or len(set(categories)) != len(categories)):
                errors.append({'experiment': name, 'task_id': task_id, 'error': 'Failure categories absent or malformed'})
                continue
            category_counts.update(categories)
            if set(categories) - VALID_FAILURE_CATEGORIES:
                errors.append({'experiment': name, 'task_id': task_id,
                    'error': 'Technical or unknown failure prevents final completion',
                    'categories': sorted(set(categories) - VALID_FAILURE_CATEGORIES)})
            for event in failure.get('graph_error_events', []):
                event_categories = set(event.get('categories', []))
                generated = event.get('generated_response') or {}
                if event_categories - VALID_FAILURE_CATEGORIES:
                    errors.append({'experiment': name, 'task_id': task_id, 'event_id': event.get('event_id'),
                        'error': 'Actual error event has technical or unknown categories',
                        'categories': sorted(event_categories - VALID_FAILURE_CATEGORIES)})
                if event.get('kind') != 'policy_abort' and not event_categories:
                    errors.append({'experiment': name, 'task_id': task_id, 'event_id': event.get('event_id'),
                        'error': 'Unmapped actual non-wrapper error event prevents final completion'})
                if event_categories & {'expert_schema_error', 'generated_record_or_schema_error',
                                        'expert_output_truncated', 'expert_empty_output'}:
                    minimum = 0 if 'expert_empty_output' in event_categories else 1
                    if (event.get('request_sent') is not True
                            or not count(generated.get('raw_response_characters'))
                            or generated['raw_response_characters'] < minimum
                            or not re.fullmatch(r'[a-f0-9]{64}', generated.get('raw_response_sha256', ''))):
                        errors.append({'experiment': name, 'task_id': task_id, 'event_id': event.get('event_id'),
                            'error': 'Model-output classification lacks actual generated-response proof'})
            if (set(categories) & {'expert_schema_error', 'generated_record_or_schema_error',
                                   'expert_output_truncated', 'expert_empty_output'}
                    and not failure.get('graph_error_events')):
                errors.append({'experiment': name, 'task_id': task_id,
                    'error': 'Model-output classification has no emitted error event'})
        if dict(category_counts) != row.get('abort_categories', {}):
            errors.append({'experiment': name, 'error': 'Failure-category totals differ from exact aborted cases'})
    return errors


def count(value):
    return type(value) is int and value >= 0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_projection(spec, wrapper):
    """Check loaded identity, allowing an explicitly unexercised native head."""
    model = wrapper.get('model')
    if not isinstance(model, dict):
        raise ValueError('Loaded model identity is absent')
    kind = spec['kind']
    if kind in {'hf', 'native'}:
        if Path(model.get('model_path', '')).resolve() != Path(spec['model_path']).resolve():
            raise ValueError('Loaded checkpoint path differs from frozen specification')
        if model.get('dtype') != spec.get('dtype', 'bfloat16'):
            raise ValueError('Loaded model dtype differs from frozen specification')
        if kind == 'hf':
            if (model.get('backend') != 'hf' or model.get('weights_frozen') is not True
                    or model.get('local_files_only') is not True or not model.get('weight_sha256')
                    or model.get('context_extension') != spec.get('context_extension')
                    or model.get('max_context') != spec.get('max_context')):
                raise ValueError('HF identity/context/weight evidence differs from frozen profile')
            return {key: model.get(key) for key in ('backend', 'model_path', 'dtype', 'max_context',
                'context_extension', 'weight_sha256', 'config_sha256', 'tokenizer_sha256', 'effective_rope_scaling')}
        if model.get('family') != spec.get('family', 'openjev') or model.get('effort') != spec.get('effort', 'high'):
            raise ValueError('Native family/effort differs from frozen specification')
        return {key: model.get(key) for key in ('kind', 'model_path', 'dtype', 'family', 'effort', 'mechanism', 'local_files_sha256')}
    if kind == 'features':
        if (model.get('backend') != 'frozen_features'
                or Path(model.get('artifact_path', '')).resolve() != Path(spec['artifact_path']).resolve()
                or not model.get('artifact_sha256') or not model.get('manifest_sha256')):
            raise ValueError('Fitted feature identity differs from frozen specification')
        encoder = model.get('encoder', {})
        identity = encoder.get('model_identity', {})
        if identity.get('kind') != spec.get('encoder_kind', 'embedding'):
            raise ValueError('Feature encoder kind differs from frozen specification')
        if identity.get('kind') == 'hf_hidden':
            backend = identity.get('backend', {})
            if (Path(backend.get('model_path', '')).resolve() != Path(spec['model_path']).resolve()
                    or backend.get('dtype') != spec['dtype'] or not backend.get('weight_sha256')):
                raise ValueError('Feature HF encoder checkpoint/dtype differs')
            encoder_identity = {key: backend.get(key) for key in ('weight_sha256', 'config_sha256', 'tokenizer_sha256', 'dtype')}
        else:
            if (Path(encoder.get('local_path', '')).resolve() != Path(spec['model_path']).resolve()
                    or not identity.get('model_files_sha256')):
                raise ValueError('Embedding encoder checkpoint differs')
            encoder_identity = identity
        return {key: model.get(key) for key in ('backend', 'artifact_path', 'artifact_sha256', 'manifest_sha256', 'method')} | {
            'encoder_identity': encoder_identity, 'model_dtype': encoder.get('model_dtype')}
    raise ValueError('Unexpected runtime backend kind')


def memory_mib(value):
    match = re.fullmatch(r'(\d+)([KMGT]?)', str(value))
    if not match:
        raise ValueError('Unrecognized allocation memory: ' + str(value))
    return int(match[1]) * {'': 1, 'K': 1 / 1024, 'M': 1, 'G': 1024, 'T': 1024 ** 2}[match[2]]


def resource_override_proofs(prepared, specs, jobs):
    """Load only exact verified resource updates bound to frozen E declarations."""
    allowed, evidence, errors = {}, [], []
    for directory in sorted((STUDY / 'hardware-scheduling').iterdir()):
        if not directory.is_dir() or not (directory / 'plan.json').exists():
            continue
        verified_paths = sorted(directory.glob('chunk-*/verified-before-release.json'))
        if (directory / 'verified-before-release.json').exists():
            verified_paths.append(directory / 'verified-before-release.json')
        if not verified_paths:
            continue
        try:
            plan = read(directory / 'plan.json')
            if (any(plan.get(key) != prepared[key] for key in ('source_manifest_sha256', 'configuration_sha256', 'jobs_sha256'))
                    or plan.get('models_or_protocols_modified') is not False
                    or plan.get('array_ownership_modified') is not False):
                raise ValueError('Resource controller is not bound to the unchanged frozen model/array profile')
            selected = {row['job_id']: row for row in plan['selected']}
            memory_proof = plan['memory_proof']
            if sha(Path(memory_proof['path']).read_bytes()) != memory_proof['sha256']:
                raise ValueError('Recorded memory proof bytes changed')
            evidence.extend([file_proof(directory / 'plan.json'), file_proof(memory_proof['path'])])
            for path in verified_paths:
                verified = read(path)
                records = ({row['job_id']: row['control'] for row in verified['updated']}
                    if 'updated' in verified else verified)
                for alias, control in records.items():
                    selected_row = selected[alias]
                    index = selected_row['index']
                    job = jobs[index]
                    spec = specs[job['experiment']]
                    if (selected_row['job'] != job or sha(encoded(spec)) != selected_row['specification_sha256']
                            or alias != selected_row['job_id'] or alias != control.get('resolved_job_alias', alias)
                            or control.get('ArrayJobId') != alias.split('_')[0]
                            or control.get('ArrayTaskId') != str(index)
                            or control.get('CPUs/Task') != '8' or control.get('NumCPUs') != '8'
                            or memory_mib(control.get('MinMemoryNode')) != 47104
                            or set(control.get('Partition', '').split(',')) - {'l40s', 'h100'}
                            or not control.get('Partition')):
                        raise ValueError('Verified exact-task profile/specification differs')
                    allowed[alias] = {'verified_update': file_proof(path), 'controller_plan': str(directory / 'plan.json'),
                        'index': index, 'profile': 'documented8CPU46G',
                        'note': 'Actual allocation is checked; stale TresPerTask/oldMinCPUsNode request fields are not actual CPU counts.'}
                evidence.append(file_proof(path))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append({'path': str(directory), 'error': f'{type(error).__name__}: {error}'})
    return allowed, evidence, errors


def hardware(owner, specification, overrides, baseline):
    tres = dict(part.split('=', 1) for part in owner.get('allocated_tres', '').split(',') if '=' in part)
    types = [key.split(':', 1)[1].lower() for key in tres if key.startswith('gres/gpu:') and tres[key] == '1']
    family = next((value for value in ('h100', 'l40s') if any(value in item for item in types)), None)
    if (tres.get('gres/gpu') != '1' or family is None or not owner.get('node')
            or not owner.get('start_utc') or not tres.get('cpu') or not tres.get('mem')):
        raise ValueError('Actual one-GPU H100/L40S allocation metadata is incomplete')
    original = baseline[specification['deadline_batch']]
    ordinary = (family in original['partitions'] and owner.get('partition') in original['partitions']
        and tres['cpu'] == original['cpus']
        and memory_mib(tres['mem']) == memory_mib(original['memory']))
    override = overrides.get(owner['job_id'])
    changed = (override is not None and owner.get('partition') in {'l40s', 'h100'}
        and tres['cpu'] == '8' and memory_mib(tres['mem']) == 47104)
    if not ordinary and not changed:
        raise ValueError('Actual CPU/memory/GPU profile has no matching original or exact-alias documented override')
    return {'gpu_family': family, 'typed_gpus': types, 'partition': owner.get('partition'),
        'cpus': tres['cpu'], 'memory': tres['mem'], 'node': owner['node'], 'allocated_tres': owner['allocated_tres'],
        'profile_evidence': 'frozen_launch_batch' if ordinary else override}


def main():
    global STUDY, MAIN, STATUS, PUBLIC_AUDIT, FAILURE_AUDIT
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--failure-audit', type=Path, required=True)
    parser.add_argument('--failure-helper', type=Path, default=HERE / 'audit_failures.py',
                        help='Exact classifier source that generated the supplied failure audit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    STUDY, MAIN, STATUS, FAILURE_AUDIT, PUBLIC_AUDIT = (
        value.resolve() for value in (args.study, args.main, args.status, args.failure_audit, args.output))
    if (not STUDY.is_relative_to(args.evidence_root.resolve())
            or PUBLIC_AUDIT.is_relative_to(STUDY)
            or PUBLIC_AUDIT in {MAIN, STATUS, FAILURE_AUDIT}):
        raise ValueError('Audit output must be separate; study must belong to the explicit evidence root')
    started = datetime.now(timezone.utc).isoformat()
    sys.path[:0] = [str(STUDY / 'snapshot'), str(STUDY / 'snapshot/upstream')]
    from graph_local.run import authenticate, digest
    snapshot, inventory = authenticate(STUDY)
    prepared, config, jobs, scope = (read(STUDY / name) for name in
        ('prepared.json', 'configuration.json', 'jobs.json', 'deadline-scope.json'))
    if digest(STUDY / 'deadline-scope.json') != prepared['deadline_scope_sha256']:
        raise ValueError('Frozen deadline scope changed')
    for name, expected in scope['helper_sha256'].items():
        if Path(name).name != name or digest(STUDY / 'rescue-helpers' / name) != expected:
            raise ValueError('Frozen production helper changed: ' + name)
    public_path = snapshot / 'instrument/sequential_120/cases.jsonl'
    public = [json.loads(line) for line in public_path.read_text().splitlines() if line.strip()]
    private_path = snapshot / 'instrument/sequential_120/evaluator/case-specs.jsonl'
    private_cases = [json.loads(line) for line in private_path.read_text().splitlines() if line.strip()]
    if len(private_cases) != 120 or {row['task_id'] for row in private_cases} != {row['task_id'] for row in public}:
        raise ValueError('Frozen public/evaluator case inventories differ')
    specs = {row['name']: row for row in config['experiments']}
    if (len(specs) != 22 or len(public) != 120 or len({row['task_id'] for row in public}) != 120
            or len({(row['expert_row'], row['chooser_column']) for row in specs.values()}) != 22
            or scope.get('mode') != 'teacher_forced' or scope.get('seed') != 0
            or any(row.get('seed') != 0 for row in specs.values())
            or len(jobs) != 2640 or {(row['experiment'], row['shard_index'], row['shards'], row['mode']) for row in jobs}
                != {(name, index, 120, 'teacher_forced') for name in specs for index in range(120)}):
        raise ValueError('Frozen22×120 model-seed-zero TF declarations do not reconcile')
    baseline = {}
    for line in (STUDY / 'rescue-helpers/launch-batches.tsv').read_text().splitlines():
        batch, _, partitions, cpus, memory, *_ = line.split('\t')
        baseline[batch] = {'partitions': partitions.split(','), 'cpus': cpus, 'memory': memory}
    overrides, override_evidence, override_errors = resource_override_proofs(prepared, specs, jobs)
    reporter = load_module('independent_final_report_ownership', HERE / 'report_grid.py')
    reporter.EVIDENCE_ROOT = args.evidence_root.resolve()
    sys.path.insert(0, str(snapshot / 'instrument/eval'))
    scorer = load_module('independent_final_saved_validator', snapshot / 'instrument/eval/sequential.py')
    compact_projector = load_module('independent_final_compact_projection', HERE / 'compact_status.py')
    scheduler = reporter.scheduler_snapshot(STUDY, jobs, True)
    main_bytes, status_bytes = MAIN.read_bytes(), STATUS.read_bytes()
    report, status = json.loads(main_bytes), json.loads(status_bytes)
    issues = defaultdict(list)
    issues['resource_override_provenance'].extend(override_errors)
    gates = {}
    gates['frozen_source_and_exact_declarations'] = True
    gates['effective_2640_scheduler_owners_available'] = scheduler.get('status') == 'available' and scheduler.get('matched_declared_slots') == 2640
    gates['all_effective_owners_terminal'] = gates['effective_2640_scheduler_owners_available'] and set(scheduler.get('states', {})) <= TERMINAL
    for label, document in (('main', report), ('status', status)):
        if document.get('study') != str(STUDY) or document.get('mode') != 'teacher_forced':
            issues['public_identity'].append(label)
    expected_hashes = {key: prepared[key] for key in ('source_manifest_sha256', 'configuration_sha256', 'jobs_sha256', 'deadline_scope_sha256')}
    if any(report.get('provenance', {}).get(key) != value for key, value in expected_hashes.items()):
        issues['public_identity'].append('main frozen hashes')
    if any(status.get('source', {}).get(key) != value for key, value in expected_hashes.items()):
        issues['public_identity'].append('status frozen hashes')
    if (status.get('source', {}).get('public_report_sha256') != sha(main_bytes)
            or status.get('report_generated_utc') != report.get('generated_utc')):
        issues['public_pair_coherence'].append('compact status does not describe these exact canonical main bytes/timestamp')
    try:
        if datetime.fromisoformat(status['generated_utc']) < datetime.fromisoformat(report['generated_utc']):
            issues['public_pair_coherence'].append('compact status predates its main input')
    except (ValueError, KeyError, TypeError):
        issues['public_pair_coherence'].append('invalid public timestamps')
    if (report.get('cohort', {}).get('seed') != 0 or report.get('cohort', {}).get('development_smokes_included') is not False
            or report.get('cohort', {}).get('declared_episode_count') != 2640):
        issues['public_identity'].append('cohort seed/production scope')
    cells = {name: dict(experiment=name, expert=spec['expert_row'], chooser=spec['chooser_column'],
        expected=120, closed=0, native_completed=0, assisted_goals=0, aborted=0, missing=0,
        proposals=0, correct=0, corrections=0, proposal_known=0, correction_known=0,
        recorded_wall_n=0, recorded_wall_s=0.0, explicit_initialization_failures=0,
        hardware=Counter(), token_coverage={}, source_cases=[]) for name, spec in specs.items()}
    raw_by_name = defaultdict(list)
    raw_abort_corrections = {}
    expected_failure_sources = {}
    model_signatures = defaultdict(set)
    checkpoint_signatures = defaultdict(set)
    owned = scheduler.get('by_slot', {}) if gates['effective_2640_scheduler_owners_available'] else {}
    for index, job in enumerate(jobs):
        name, shard = job['experiment'], job['shard_index']
        cell, spec = cells[name], specs[name]
        task_id = public[shard]['task_id']
        worker = STUDY / 'runs' / name / 'sequential' / f'shard-{shard:03d}'
        experiment = worker / name
        episode_path = experiment / 'sequential/episodes.jsonl'
        if not episode_path.exists():
            cell['missing'] += 1
            issues['missing_case_slots'].append({'index': index, 'experiment': name, 'task_id': task_id})
            continue
        proof = {'index': index, 'task_id': task_id, 'episode': file_proof(episode_path)}
        try:
            data = episode_path.read_bytes()
            lines = data.splitlines(keepends=True)
            if not lines or (len(lines) == 1 and not lines[0].endswith(b'\n')):
                cell['missing'] += 1
                issues['missing_case_slots'].append({'index': index, 'experiment': name, 'task_id': task_id,
                    'reason': 'in_progress_unclosed_record'})
                cell['source_cases'].append(proof)
                continue
            if len(lines) != 1 or not lines[0].endswith(b'\n'):
                raise ValueError('Expected exactly one completely written JSON record; no discarded trailing fragments')
            row = json.loads(lines[0])
            proof['episode'] = {'path': str(episode_path), 'sha256': sha(data), 'bytes': len(data)}
            expected_failure_sources[str(episode_path)] = proof['episode']
            if row.get('task_id') != task_id or row.get('mode') != 'teacher_forced':
                raise ValueError('Episode task/mode differs from exact frozen shard')
            scorer._validate_episode(row)
            worker_config = read(worker / 'configuration.json')
            proof['worker_configuration'] = file_proof(worker / 'configuration.json')
            if (worker_config.get('selected_experiments') != [name] or worker_config.get('selected_task_ids') != [task_id]
                    or worker_config.get('shards') != 120 or worker_config.get('shard_index') != shard
                    or worker_config.get('mode') != 'teacher_forced' or worker_config.get('limit') is not None
                    or worker_config.get('source_sha256') != {**inventory, 'configuration_input': prepared['configuration_sha256']}
                    or worker_config.get('resolved_configuration') != config or worker_config.get('hosted_inference_calls') != 0):
                raise ValueError('Worker declaration/source/seed/hosted-call evidence differs')
            cell['closed'] += 1
            is_ok = row.get('status', 'ok') == 'ok'
            proof['record_status'] = row.get('status', 'ok')
            proof['native_outcome_met'] = native_case_outcome(row)
            cell['native_completed'] += is_ok
            cell['aborted'] += not is_ok
            cell['assisted_goals'] += is_ok and row['trajectory_outcome_met'] is True
            identity = read(experiment / 'identity.json')
            proof['runtime_identity'] = file_proof(experiment / 'identity.json')
            try:
                if (identity.get('graph_source_manifest_sha256') != prepared['source_manifest_sha256']
                        or identity.get('hosted_inference_calls') != 0):
                    raise ValueError('Runtime identity has wrong source/hosted-call scope')
                initialization_error = identity.get('initialization_error')
                if initialization_error:
                    if is_ok or row.get('partial_native_decisions_observed') != 0:
                        raise ValueError('Initialization failure conflicts with observed native trajectory')
                    cell['explicit_initialization_failures'] += 1
                    proof['initialization_error'] = initialization_error
                else:
                    if identity.get('shared_hf_instance') != (spec['expert'] == spec['backend']):
                        raise ValueError('Shared expert/chooser identity differs from declared cell')
                    if identity.get('chooser', {}).get('rules_fallback') is not False:
                        raise ValueError('Chooser fallback policy is not explicitly disabled')
                    if (identity.get('expert', {}).get('record_protocol') != spec['expert'].get('record_protocol')
                            or identity.get('chooser', {}).get('context_protocol') != spec['backend'].get('context_protocol')):
                        raise ValueError('Runtime public-interface protocol differs from frozen profile')
                    for role, role_spec in (('expert', spec['expert']), ('chooser', spec['backend'])):
                        projection = model_projection(role_spec, identity.get(role, {}))
                        signature = sha(encoded(projection))
                        proof[role + '_identity_sha256'] = signature
                        # Unloaded native heads omit file hashes; track that explicitly.
                        if projection.get('kind') != 'native' or projection.get('local_files_sha256'):
                            model_signatures[(role, name)].add(signature)
                        weights = projection.get('weight_sha256') or projection.get('local_files_sha256')
                        if weights:
                            checkpoint_signatures[role_spec['model_path']].add(sha(encoded(weights)))
            except (ValueError, TypeError, KeyError) as error:
                issues['runtime_identity'].append({'index': index, 'error': str(error)})
            owner = owned.get(str(index))
            if owner is not None:
                try:
                    allocation = hardware(owner, spec, overrides, baseline)
                    if (str(worker_config.get('slurm_job_id')) != owner.get('job_id_raw')
                            or worker_config.get('compute_node') != owner.get('node')):
                        raise ValueError('Worker raw job ID/node differs from effective scheduler owner')
                    cell['hardware'][allocation['gpu_family']] += 1
                    proof['actual_allocation'] = allocation | {'job_id': owner['job_id'], 'job_id_raw': owner['job_id_raw'], 'state': owner['state']}
                except (ValueError, TypeError, KeyError) as error:
                    issues['hardware_or_owner'].append({'index': index, 'error': str(error)})
            else:
                issues['hardware_or_owner'].append({'index': index, 'error': 'Effective owner metadata unavailable'})
            if number(row.get('recorded_episode_wall_s')):
                cell['recorded_wall_n'] += 1
                cell['recorded_wall_s'] += row['recorded_episode_wall_s']
            if is_ok:
                steps = row['decision_records']
                observed, correct = len(steps), sum(step['proposal_correct'] for step in steps)
            else:
                observed, correct = row.get('partial_native_decisions_observed'), row.get('partial_native_correct_original_proposals')
                folder = episode_path.parent / 'steps' / sha(task_id.encode())[:24] / 'teacher_forced'
                paths = sorted(path for path in folder.glob('*.json') if path.stem.isdigit())
                steps = [read(path) for path in paths]
                proof['aborted_partial_step_files'] = [file_proof(path) for path in paths]
                graph_path = folder / 'graph.jsonl'
                expected_failure_sources[str(graph_path)] = file_proof(graph_path)
                if not (count(observed) and count(correct) and correct <= observed and len(steps) == observed):
                    raise ValueError('Aborted saved steps do not reconcile with original-proposal counters')
                if sum(step['proposal_correct'] is True for step in steps) != correct:
                    raise ValueError('Aborted original correct count disagrees with saved steps')
            for position, step in enumerate(steps, 1):
                if (step.get('decision_index') != position or type(step.get('proposal_correct')) is not bool
                        or type(step.get('corrected')) is not bool or step['corrected'] != (not step['proposal_correct'])
                        or not isinstance(step.get('proposed_action'), dict)
                        or 'executed_action' not in step
                        or (step.get('executed_action') is not None and not isinstance(step['executed_action'], dict))
                        or not isinstance(step.get('response'), dict)):
                    raise ValueError('Saved decision does not retain original proposal and consistent TF correction')
            cell['proposals'] += observed
            cell['correct'] += correct
            cell['corrections'] += sum(step['corrected'] for step in steps)
            cell['proposal_known'] += 1
            cell['correction_known'] += 1
            raw_by_name[name].append(row)
            if not is_ok:
                raw_abort_corrections[(name, task_id)] = sum(step['corrected'] for step in steps)
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues['raw_record_or_metrics'].append({'index': index, 'path': str(episode_path), 'error': f'{type(error).__name__}: {error}'})
        cell['source_cases'].append(proof)
    report_cells = {cell.get('experiment'): cell for cell in report.get('cells', [])}
    status_cells = {cell.get('experiment'): cell for cell in status.get('cells', [])}
    if set(report_cells) != set(cells) or set(status_cells) != set(cells):
        issues['public_identity'].append('Public cell names do not match all22 frozen cells')
    if len(report.get('cells', [])) != 22 or len(status.get('cells', [])) != 22:
        issues['public_identity'].append('Duplicate or extra public cells are not allowed')
    expected_scores = {}
    for name, cell in cells.items():
        published = report_cells.get(name, {})
        compact = status_cells.get(name, {})
        expected_score = scorer.score_episodes(private_cases, raw_by_name[name], modes=('teacher_forced',))['by_mode']['teacher_forced']
        expected_scores[name] = expected_score
        issues['native_saved_score_reconciliation'].extend(
            dict(experiment=name, **issue) for issue in
            differences(published.get('native_score'), expected_score, 'native_score'))
        expected_actions = reporter.original_action_metrics([(name, row) for row in raw_by_name[name]], raw_abort_corrections)
        issues['cell_metric_reconciliation'].extend(dict(experiment=name, **issue) for issue in
            differences(published.get('original_action_metrics_all_closed_attempts'), expected_actions,
                        'original_action_metrics_all_closed_attempts'))
        issues['usage_metric_reconciliation'].extend(dict(experiment=name, **issue) for issue in
            differences(published.get('usage'), reporter.usage_summary(raw_by_name[name]), 'usage'))
        if (published.get('seed') != 0 or published.get('mode') != 'teacher_forced' or published.get('expected_cases') != 120
                or published.get('expert') != cell['expert'] or published.get('chooser') != cell['chooser']
                or compact.get('expert') != cell['expert'] or compact.get('chooser') != cell['chooser']):
            issues['public_identity'].append({'experiment': name, 'error': 'Cell identity/seed/denominator differs'})
        if cell['closed'] == 120 and (published.get('deadline_not_started') != 0 or published.get('unusable_records') != 0):
            issues['cell_metric_reconciliation'].append({'experiment': name,
                'error': 'A full usable closed cell cannot also contain skipped or unusable records'})
        public_cases = published.get('cases', [])
        case_map = {case.get('task_id'): case for case in public_cases}
        if len(public_cases) != 120 or set(case_map) != {case['task_id'] for case in public}:
            issues['public_identity'].append({'experiment': name, 'error': 'Public case set is not exact120'})
        for proof in cell['source_cases']:
            if 'record_status' not in proof:
                continue
            case = case_map.get(proof['task_id'], {})
            if (case.get('job_index') != proof['index'] or case.get('record_status') != proof['record_status']
                    or case.get('episode_source') != proof['episode']['path']):
                issues['cell_metric_reconciliation'].append({'index': proof['index'], 'error': 'Published exact-case/source identity differs'})
            issues['cell_metric_reconciliation'].extend(dict(index=proof['index'], **issue)
                for issue in case_outcome_errors(case, proof['native_outcome_met']))
            actual = owned.get(str(proof['index']), {})
            saved_owner = case.get('slurm', {})
            if any(saved_owner.get(field) != actual.get(field) for field in
                   ('job_id', 'job_id_raw', 'partition', 'allocated_tres', 'node', 'start_utc')) or not saved_owner:
                issues['public_hardware_provenance'].append({'index': proof['index'], 'error': 'Published allocation missing or differs from effective owner'})
        checks = {'episode_records': cell['closed'], 'valid_native_episodes': cell['native_completed'],
            'verified_outcome_successes': cell['assisted_goals'], 'aborted_episodes': cell['aborted'],
            'missing_episode_records': 120 - cell['closed'],
            'assisted_goal_completion_rate': cell['assisted_goals'] / 120}
        for field, expected in checks.items():
            if published.get(field) != expected:
                issues['cell_metric_reconciliation'].append({'experiment': name, 'field': field, 'raw': expected, 'public': published.get(field)})
        metrics = published.get('original_action_metrics_all_closed_attempts', {})
        raw_metrics = {'closed_attempts': cell['closed'], 'known_partial_observed_original_proposals': cell['proposals'],
            'known_partial_correct_original_proposals': cell['correct'], 'known_partial_reference_corrections': cell['corrections'],
            'closed_attempts_with_proposal_counts': cell['proposal_known'], 'closed_attempts_with_correction_counts': cell['correction_known']}
        if cell['proposal_known'] == cell['closed']:
            raw_metrics.update(observed_original_proposals=cell['proposals'], correct_original_proposals=cell['correct'],
                original_proposal_accuracy_micro=cell['correct'] / cell['proposals'] if cell['proposals'] else None)
        if cell['correction_known'] == cell['closed']:
            raw_metrics['reference_corrections'] = cell['corrections']
        for field, expected in raw_metrics.items():
            if not same(metrics.get(field), expected):
                issues['cell_metric_reconciliation'].append({'experiment': name, 'field': field, 'raw': expected, 'public': metrics.get(field)})
        if (compact.get('processed') != published.get('episode_records') or compact.get('aborted') != published.get('aborted_episodes')
                or compact.get('assisted_goals') != published.get('verified_outcome_successes')
                or compact.get('original', {}).get('observed') != metrics.get('observed_original_proposals')
                or compact.get('original', {}).get('correct') != metrics.get('correct_original_proposals')
                or not same(compact.get('original', {}).get('accuracy'), metrics.get('original_proposal_accuracy_micro'))
                or compact.get('original', {}).get('corrections') != metrics.get('reference_corrections')):
            issues['compact_metric_reconciliation'].append(name)
        usage = published.get('usage', {})
        if (usage.get('recorded_episode_wall_n') != cell['recorded_wall_n']
                or not same(usage.get('recorded_episode_wall_seconds'), cell['recorded_wall_s'] if cell['recorded_wall_n'] else None)):
            issues['usage_metric_reconciliation'].append({'experiment': name, 'field': 'recorded episode wall',
                'note': 'Different current raw/public snapshot populations are expected while workers remain active.'})
        for token, record in usage.get('tokens', {}).items():
            cell['token_coverage'][token] = record
            published_n = published.get('episode_records', 0)
            if ((record.get('exact_episode_n', 0) < published_n and record.get('exact_total') is not None)
                    or record.get('exact_episode_n', 0) > published_n):
                issues['usage_honesty'].append({'experiment': name, 'field': token, 'error': 'Unknown token coverage was promoted to an exact total'})
    totals = {field: sum(cell[field] for cell in cells.values()) for field in
        ('closed', 'native_completed', 'assisted_goals', 'aborted', 'missing', 'proposals', 'correct', 'corrections',
         'proposal_known', 'correction_known', 'explicit_initialization_failures')}
    if totals['proposals'] != totals['correct'] + totals['corrections']:
        issues['raw_record_or_metrics'].append({'error': 'Original correct plus TF corrections differs from proposals'})
    global_metrics = report.get('original_action_metrics_all_closed_attempts', {})
    expected_actions = reporter.original_action_metrics(
        [(name, row) for name in cells for row in raw_by_name[name]], raw_abort_corrections)
    issues['global_metric_reconciliation'].extend(differences(global_metrics, expected_actions,
        'original_action_metrics_all_closed_attempts'))
    for field, raw in (('closed_attempts', totals['closed']), ('known_partial_observed_original_proposals', totals['proposals']),
                       ('known_partial_correct_original_proposals', totals['correct']), ('known_partial_reference_corrections', totals['corrections'])):
        if global_metrics.get(field) != raw:
            issues['global_metric_reconciliation'].append({'field': field, 'raw': raw, 'public': global_metrics.get(field)})
    if totals['proposal_known'] == totals['closed']:
        for field, raw in (('observed_original_proposals', totals['proposals']), ('correct_original_proposals', totals['correct']),
                           ('original_proposal_accuracy_micro', totals['correct'] / totals['proposals'] if totals['proposals'] else None)):
            if not same(global_metrics.get(field), raw):
                issues['global_metric_reconciliation'].append({'field': field, 'raw': raw, 'public': global_metrics.get(field)})
    if totals['correction_known'] == totals['closed'] and global_metrics.get('reference_corrections') != totals['corrections']:
        issues['global_metric_reconciliation'].append({'field': 'reference_corrections'})
    progress = report.get('progress', {})
    compact_totals = status.get('totals', {})
    for main_field, compact_field, raw_field in (('episode_records', 'processed', 'closed'),
            ('valid_native_episodes', 'native_completed', 'native_completed'), ('verified_outcome_successes', 'assisted_goals', 'assisted_goals'),
            ('aborted_episodes', 'aborted', 'aborted')):
        if progress.get(main_field) != totals[raw_field]:
            issues['global_metric_reconciliation'].append({'field': main_field, 'raw': totals[raw_field], 'public': progress.get(main_field)})
        if compact_totals.get(compact_field) != progress.get(main_field):
            issues['compact_metric_reconciliation'].append({'field': compact_field, 'scope': 'global'})
    completed_proposals = sum(score['decisions_observed'] for score in expected_scores.values())
    completed_correct = sum(score['correct_original_proposals'] for score in expected_scores.values())
    expected_progress = {'declared_episode_count': 2640, 'episode_records': totals['closed'],
        'valid_native_episodes': totals['native_completed'], 'verified_outcome_successes': totals['assisted_goals'],
        'aborted_episodes': totals['aborted'], 'missing_episode_records': 2640 - totals['closed'],
        'attempt_record_coverage': totals['closed'] / 2640,
        'valid_native_episode_coverage': totals['native_completed'] / 2640,
        'cohort_assisted_goal_completion_rate': totals['assisted_goals'] / 2640,
        'observed_original_proposals': completed_proposals, 'correct_original_proposals': completed_correct,
        'observed_original_proposal_accuracy_micro': completed_correct / completed_proposals if completed_proposals else None,
        'reference_corrections': sum(score['interventions'] for score in expected_scores.values()),
        'reference_corrections_closed_attempt_coverage': totals['native_completed'] / totals['closed'] if totals['closed'] else None}
    issues['completed_only_progress_reconciliation'].extend(differences(
        {key: progress.get(key) for key in expected_progress}, expected_progress, 'progress'))
    all_saved_rows = [row for name in cells for row in raw_by_name[name]]
    issues['usage_metric_reconciliation'].extend(differences(
        report.get('usage'), reporter.usage_summary(all_saved_rows), 'usage.global'))
    expected_batches = {}
    for batch in baseline:
        group = [cell for name, cell in cells.items() if specs[name]['deadline_batch'] == batch]
        denominator = 120 * len(group)
        expected_batches[batch] = {'cells': len(group), 'declared_episode_count': denominator,
            'episode_records': sum(cell['closed'] for cell in group),
            'valid_native_episodes': sum(cell['native_completed'] for cell in group),
            'verified_outcome_successes': sum(cell['assisted_goals'] for cell in group),
            'aborted_episodes': sum(cell['aborted'] for cell in group),
            'missing_episode_records': denominator - sum(cell['closed'] for cell in group),
            'deadline_not_started': sum(report_cells[cell['experiment']].get('deadline_not_started', 0) for cell in group),
            'unusable_records': sum(report_cells[cell['experiment']].get('unusable_records', 0) for cell in group),
            'assisted_goal_completion_rate': sum(cell['assisted_goals'] for cell in group) / denominator}
    issues['global_metric_reconciliation'].extend(differences(report.get('by_batch'), expected_batches, 'by_batch'))
    raw_failures = [row for row in all_saved_rows if row.get('status', 'ok') != 'ok']
    for field, expected in (
        ('graph_terminal_reasons', dict(Counter(row.get('graph_terminal_reason') or 'unspecified' for row in raw_failures))),
        ('model_response_classifications', dict(Counter(value for row in raw_failures for value in row.get('model_response_classifications', [])))),
        ('top_abort_errors', reporter.top(Counter(row.get('error', 'unspecified') for row in raw_failures)))):
        issues['global_metric_reconciliation'].extend(differences(report.get(field), expected, field))
    try:
        expected_compact = compact_projector.compact(report, STUDY)
        issues['compact_metric_reconciliation'].extend(differences(
            {key: status.get(key) for key in expected_compact if key != 'source'},
            {key: value for key, value in expected_compact.items() if key != 'source'}, 'compact'))
    except (KeyError, ValueError, TypeError) as error:
        issues['compact_metric_reconciliation'].append({'error': f'{type(error).__name__}: {error}'})
    if any(len(signatures) > 1 for signatures in model_signatures.values()):
        issues['runtime_identity'].append('Loaded model/artifact fingerprint changed within one frozen cell/role')
    if any(len(signatures) > 1 for signatures in checkpoint_signatures.values()):
        issues['runtime_identity'].append('One declared checkpoint path has inconsistent saved weight-file hashes')
    if report.get('scheduler', {}).get('status') != 'available' or report.get('scheduler', {}).get('matched_declared_slots') != 2640:
        issues['public_hardware_provenance'].append('Public report lacks all2640 effective scheduler owners')
    for key in ('source_read_errors', 'source_aborted_partial_step_read_errors', 'in_progress_trailing_fragments'):
        if report.get(key):
            issues['public_source_readiness'].append(key)
    if totals['closed'] == 2640 and (progress.get('deadline_not_started') != 0 or progress.get('unusable_records') != 0):
        issues['public_source_readiness'].append('Full usable cohort cannot contain skipped or unusable records')
    gates['all_22_cells_have_120_usable_closed_TF_records'] = all(cell['closed'] == 120 for cell in cells.values()) and not issues['raw_record_or_metrics']
    gates['all_current_closed_original_and_correction_counts_known'] = totals['proposal_known'] == totals['correction_known'] == totals['closed']
    gates['no_explicit_model_initialization_failures'] = totals['explicit_initialization_failures'] == 0
    for gate, issue_key in (('runtime_identity_consistent', 'runtime_identity'), ('actual_hardware_and_owner_known', 'hardware_or_owner'),
                           ('resource_changes_have_exact_alias_provenance', 'resource_override_provenance'),
                           ('actual_hardware_in_public_provenance', 'public_hardware_provenance'),
                           ('public_study_identity_exact', 'public_identity'), ('canonical_pair_coherent', 'public_pair_coherence'),
                           ('cell_metrics_reconcile_with_raw', 'cell_metric_reconciliation'), ('global_metrics_reconcile_with_raw', 'global_metric_reconciliation'),
                           ('native_saved_scores_including_groups_reconcile', 'native_saved_score_reconciliation'),
                           ('completed_only_progress_reconciles', 'completed_only_progress_reconciliation'),
                           ('compact_metrics_reconcile', 'compact_metric_reconciliation'), ('unknown_usage_not_invented', 'usage_honesty'),
                           ('usage_metrics_reconcile_with_current_raw_snapshot', 'usage_metric_reconciliation'),
                           ('no_discarded_trailing_fragments_or_read_errors', 'public_source_readiness')):
        gates[gate] = not issues[issue_key]
    gates['public_inputs_unchanged_during_audit'] = sha(MAIN.read_bytes()) == sha(main_bytes) and sha(STATUS.read_bytes()) == sha(status_bytes)
    failure_proof = None
    if FAILURE_AUDIT.exists():
        failure_audit = read(FAILURE_AUDIT)
        issues['final_failure_evidence'].extend(failure_evidence_errors(failure_audit, cells, expected_hashes))
        issues['final_failure_source_binding'].extend(verify_source_bindings(
            failure_audit.get('source_files', []), expected_failure_sources))
        audit_helper = Path(failure_audit.get('audit_helper', '')).resolve()
        if (audit_helper != args.failure_helper.resolve()
                or failure_audit.get('audit_helper_sha256') != digest(audit_helper)
                or failure_audit.get('public_cases_sha256') != digest(public_path)):
            issues['final_failure_evidence'].append({'error': 'Failure audit helper/public-case provenance changed'})
        gates['final_all22_failure_evidence_present'] = not issues['final_failure_evidence']
        gates['final_failure_sources_match_current_raw_bytes'] = not issues['final_failure_source_binding']
        failure_proof = file_proof(FAILURE_AUDIT)
    else:
        gates['final_all22_failure_evidence_present'] = False
        gates['final_failure_sources_match_current_raw_bytes'] = False
        issues['required_missing_fields'].append({'path': str(FAILURE_AUDIT), 'reason': 'Final saved-error evidence is queued after the final export'})
    ready = all(gates.values())
    result = {'schema_version': 1, 'study': str(STUDY), 'mode': 'teacher_forced', 'model_seed': 0,
        'status': 'passed_final_completion_audit' if ready else 'not_ready', 'goal_complete': ready,
        'started_utc': started, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'compute_node': os.environ['SLURMD_NODENAME'],
        'scope': 'Read-only independent frozen-source/raw-record/original-grade/identity/hardware/public-artifact reconciliation; no simulation, inference, regrading or production mutation.',
        'gates': gates, 'required_missing_or_failed_gates': [key for key, value in gates.items() if not value],
        'observed_raw_totals': totals, 'issues': dict(issues), 'cells': list(cells.values()),
        'scheduler': scheduler, 'provenance': expected_hashes | {'full_frozen_inventory_reverified': True,
            'audit_source': file_proof(Path(__file__)), 'reporter_source': file_proof(HERE / 'report_grid.py'),
            'public_main': {'path': str(MAIN), 'sha256': sha(main_bytes), 'generated_utc': report.get('generated_utc')},
            'public_status': {'path': str(STATUS), 'sha256': sha(status_bytes), 'generated_utc': status.get('generated_utc')},
            'public_cases': file_proof(public_path), 'final_all22_failure_audit': failure_proof,
            'resource_override_sources': override_evidence,
            'checkpoint_weight_fingerprints': {key: sorted(value) for key, value in checkpoint_signatures.items()}},
        'interpretation': ['Completion requires all2640 declared closed records and complete saved original/correction accounting; it does not require2640 goals.',
            'Valid native/model/budget aborts count as closed results. Explicit initialization failures remain technical failures, not model proposal scores.',
            'Case-construction seeds remain exactly as frozen; model seed0 does not overwrite native scenario seeds.',
            'Native saved original grades are checked for consistency, not rerun or replaced. Corrected completion remains assisted performance.',
            'Unknown usage stays null/partial. Episode wall time excludes model initialization and is not GPU-active time.',
            'An incomplete or concurrently changing current snapshot must fail readiness and be audited again after final export/status alignment.']}
    output = PUBLIC_AUDIT
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    with output.open('xb') as stream:
        stream.write(data)
    print(json.dumps({'output': str(output), 'public_audit': str(PUBLIC_AUDIT), 'status': result['status'],
        'raw_totals': totals, 'required_missing_or_failed_gates': result['required_missing_or_failed_gates'],
        'issue_counts': {key: len(value) for key, value in issues.items()}}, indent=2), flush=True)
    if args.require_complete and not ready:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
