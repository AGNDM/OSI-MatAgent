"""Run frozen, disjoint sequential shards through the actual repository graph."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def require_slurm():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A genuine Slurm compute allocation is required')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def authenticate(study):
    snapshot = study / 'snapshot'
    preparation = json.loads((study / 'prepared.json').read_text())
    for name, field in (('configuration.json', 'configuration_sha256'),
                        ('jobs.json', 'jobs_sha256'), ('source-manifest.json', 'source_manifest_sha256')):
        if digest(study / name) != preparation.get(field):
            raise ValueError('Frozen graph study configuration changed: ' + name)
    manifest = json.loads((study / 'source-manifest.json').read_text())
    actual_paths = set()
    for path in snapshot.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        relative = path.relative_to(snapshot).as_posix()
        actual_paths.add(relative)
        if manifest.get(relative) != digest(path):
            raise ValueError('Frozen graph study source changed: ' + relative)
    if actual_paths != set(manifest):
        raise ValueError('Frozen graph study source inventory changed')
    return snapshot, manifest


def recover_closed_child(child, spec, *, python, comparison_root, stderr_path,
                         timeout_seconds, factory=None):
    """Recover only between independent episodes; never retry a failed call."""
    if not child.closed:
        return child, False
    if factory is None:
        from graph_local.process_backend import LocalProcessBackend
        factory = LocalProcessBackend
    return factory(copy.deepcopy(spec), python=python, comparison_root=comparison_root,
                   stderr_path=stderr_path, timeout_seconds=timeout_seconds), True


def current_identity(roles, chooser, specification, source_manifest_sha256, *, shared=False,
                     initialization_error=None):
    identity = {'architecture': 'repository_langgraph',
        'expert': roles.manifest() if roles is not None else {'specification': specification['expert']},
        'chooser': chooser.manifest() if chooser is not None else {'specification': specification['backend']},
        'shared_hf_instance': shared, 'graph_source_manifest_sha256': source_manifest_sha256,
        'context_policy': 'Complete supplied public evidence and menus; no truncation',
        'hosted_inference_calls': 0}
    if initialization_error is not None:
        identity['initialization_error'] = initialization_error
    return identity


def abort_diagnostics(policy, partial):
    """Describe actual partial records without inventing a completed episode."""
    events = policy.trace.events if policy is not None else []
    failures = [event for event in events if event.get('kind') in
                {'role_error', 'operation_error', 'policy_abort', 'execution_error'}]
    classifications = sorted({metadata['response_classification'] for event in failures
        if isinstance(metadata := (event.get('llm_metadata') or event.get('backend_metadata')), dict)
        and isinstance(metadata.get('response_classification'), str)})
    return {'partial_native_decisions_observed': len(partial),
        'partial_native_correct_original_proposals': sum(record.get('proposal_correct') is True for record in partial),
        'partial_records_are_completed_episode_metrics': False,
        'observed_graph_model_usage': policy._usage(0) if policy is not None else None,
        'last_policy_call_usage': getattr(policy, 'last_action_usage', None),
        'model_response_classifications': classifications,
        'recorded_policy_errors': [{key: event.get(key) for key in
            ('component', 'kind', 'error', 'error_code', 'will_repair', 'repair_attempt') if key in event}
            for event in failures],
        'graph_terminal_reason': getattr(policy, 'terminal_reason', None)}


def run(args):
    require_slurm()
    study = args.study.resolve()
    snapshot, inventory = authenticate(study)
    sys.path[:0] = [str(snapshot), str(snapshot / 'upstream')]
    config_file = study / 'configuration.json'
    config = json.loads(config_file.read_text())
    specification = next(item for item in config['experiments'] if item['name'] == args.experiment)
    if not 0 <= args.shard_index < args.shards:
        raise ValueError('Invalid disjoint shard')
    from comparison.run import fingerprint, load_bundle
    helpers = load_bundle(snapshot / 'instrument')
    helpers['validate'].validate_bundle(snapshot / 'instrument')
    sequential = helpers['sequential']
    _, _, native_run_case, menu_type, interface_error = sequential._runtime()
    # app.native prepends its hydrated vendor path. Prime the authenticated
    # benchmark modules first, then preserve its mixed-import protection.
    for name in ('asset_runtime', 'asset_runtime.binder_fd056', 'basic40_runtime', 'basic40_runtime_thermal'):
        importlib.import_module(name)
    from app.models import Limits
    from graph_local.backends import LocalHFRoles, LocalDecisionBackend
    from graph_local.policy import GraphNativePolicy
    importlib.import_module('sequential_120.runtime.bootstrap').activate()

    public_tasks = read_jsonl(snapshot / 'instrument/sequential_120/cases.jsonl')
    private_index = {item['task_id']: item for item in read_jsonl(snapshot / 'instrument/sequential_120/evaluator/case-specs.jsonl')}
    if set(private_index) != {item['task_id'] for item in public_tasks}:
        raise ValueError('Public and evaluator cohorts differ')
    selected = public_tasks[args.shard_index::args.shards]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('Diagnostic limit must be positive')
        selected = selected[:args.limit]
    modes = ['free_run', 'teacher_forced'] if args.mode == 'both' else [args.mode]
    menus = menu_type(json.loads((snapshot / 'instrument/sequential_120/action-presets.json').read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiment_dir = output / specification['name']
    episode_dir = experiment_dir / 'sequential'
    episode_dir.mkdir(parents=True)
    configuration = {
        'resolved_configuration': config, 'selected_experiments': [specification['name']],
        'track': 'sequential', 'limit': args.limit, 'selected_task_ids': [item['task_id'] for item in selected],
        'mode': args.mode, 'shard_index': args.shard_index, 'shards': args.shards,
        'source_sha256': {**inventory, 'configuration_input': digest(config_file)},
        'source_inventory_sha256': fingerprint(inventory),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'compute_node': os.environ['SLURMD_NODENAME'],
        'hosted_inference_calls': 0,
    }
    write(output / 'configuration.json', configuration)
    episodes, child, roles, chooser = [], None, None, None
    model_load_error = None
    shared = False
    try:
        expert_spec = specification['expert']
        roles = LocalHFRoles(expert_spec, timeout_seconds=specification['call_timeout_seconds'],
                             max_new_tokens=expert_spec['max_new_tokens'])
        chooser_spec = specification['backend']
        shared = chooser_spec == expert_spec
        if chooser_spec.get('kind') == 'native' and chooser_spec.get('family') == 'apus':
            from graph_local.process_backend import LocalProcessBackend
            child = LocalProcessBackend(chooser_spec,
                python=Path(specification['apus_python']), comparison_root=snapshot,
                stderr_path=experiment_dir / 'apus-child.stderr',
                timeout_seconds=specification['call_timeout_seconds'])
        chooser = LocalDecisionBackend(chooser_spec,
            backend=child if child is not None else roles.backend if shared else None,
            timeout_seconds=specification['call_timeout_seconds'])
    except Exception as error:
        model_load_error = f'{type(error).__name__}: {error}'
    identity = current_identity(roles, chooser, specification, digest(study / 'source-manifest.json'),
                                shared=shared, initialization_error=model_load_error)
    write(experiment_dir / 'identity-initial.json', identity)
    write(experiment_dir / 'identity.json', identity)

    try:
        with (episode_dir / 'episodes.jsonl').open('x') as handle:
            for public in selected:
                for mode in modes:
                    task_id = public['task_id']
                    case = private_index[task_id]
                    step_dir = episode_dir / 'steps' / hashlib.sha256(task_id.encode()).hexdigest()[:24] / mode
                    policy = None
                    partial = []
                    started = time.perf_counter()
                    try:
                        if model_load_error:
                            raise RuntimeError(model_load_error)
                        if child is not None and child.closed:
                            child, recreated = recover_closed_child(child, specification['backend'],
                                python=Path(specification['apus_python']), comparison_root=snapshot,
                                stderr_path=experiment_dir / ('apus-child-' + task_id + '-' + mode + '.stderr'),
                                timeout_seconds=specification['call_timeout_seconds'])
                            chooser = LocalDecisionBackend(specification['backend'], backend=child,
                                timeout_seconds=specification['call_timeout_seconds'])
                            with (experiment_dir / 'process-reinitializations.jsonl').open('a') as ledger:
                                ledger.write(json.dumps({'task_id': task_id, 'mode': mode,
                                    'explicit_new_episode_process': recreated, 'failed_call_retried': False,
                                    'identity': child.identity}, allow_nan=False) + '\n')
                        budget = public.get('initial_context', {}).get('budget', {})
                        limits = Limits(lab_actions=budget.get('max_actions', budget.get('max_decisions', 64)),
                            llm_calls=specification['max_model_calls'], episode_seconds=specification['episode_seconds'],
                            graph_steps=specification['graph_steps'], format_repairs=1)
                        policy = GraphNativePolicy(copy.deepcopy(public), menus, roles, chooser,
                            mode=mode, trace_path=step_dir / 'graph.jsonl', limits=limits)

                        def on_step(record):
                            partial.append(copy.deepcopy(record))
                            write(step_dir / (f"{record['decision_index']:04d}.json"), record)

                        class NativePolicyBoundary:
                            privileged = False
                            menu_builder = menus

                            @property
                            def last_action_usage(self):
                                return policy.last_action_usage

                            def next_action(self, view, history):
                                try:
                                    return policy.next_action(view, history)
                                except Exception as error:
                                    # The PDF graph aborts exhausted policy/schema
                                    # failures. Preserve that boundary rather than
                                    # inventing a reference action inside a graph.
                                    raise interface_error(f'Graph policy stopped: {type(error).__name__}: {error}') from error

                        episode = native_run_case(case, NativePolicyBoundary(), mode=mode, on_step=on_step, menu_builder=menus)
                        episode.update(status='ok', workflow_id=public.get('workflow_id', case.get('workflow_id')),
                            condition=public.get('condition', case.get('condition')),
                            policy='repository_langgraph_local', policy_identity=policy.identity)
                        sequential._validate_episode(episode)
                    except Exception as error:
                        episode = {'task_id': task_id, 'mode': mode, 'status': 'infrastructure_error',
                            'error': f'{type(error).__name__}: {error}',
                            'policy_identity': getattr(policy, 'identity', None),
                            **abort_diagnostics(policy, partial),
                            'hosted_inference_calls': 0}
                    write(experiment_dir / 'identity.json', current_identity(roles, chooser, specification,
                        digest(study / 'source-manifest.json'), shared=shared, initialization_error=model_load_error))
                    episode['recorded_episode_wall_s'] = time.perf_counter() - started
                    episodes.append(episode)
                    handle.write(json.dumps(episode, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    selected_cases = [private_index[item['task_id']] for item in selected]
                    scores = sequential.score_episodes(selected_cases, episodes, modes)
                    scores.update(diagnostic_only=args.limit is not None, full_case_count=len(public_tasks),
                                  cohort_definition='Disjoint release-order shard; free_run and teacher_forced are separate')
                    write(episode_dir / 'summary.json', scores)
                    print(json.dumps({'experiment': specification['name'], 'task_id': task_id, 'mode': mode,
                        'submitted': len(episodes), 'expected': len(selected) * len(modes),
                        'status': episode['status'], 'trajectory_outcome_met': episode.get('trajectory_outcome_met')}), flush=True)
        authenticate(study)
        write(output / 'completion.json', {'all_expected_outputs_present': len(episodes) == len(selected) * len(modes),
            'valid_native_episodes': sum(item['status'] == 'ok' for item in episodes),
            'submitted': len(episodes), 'expected': len(selected) * len(modes), 'hosted_inference_calls': 0})
    finally:
        if child is not None:
            child.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=12)
    parser.add_argument('--mode', choices=['both', 'free_run', 'teacher_forced'], default='both')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
