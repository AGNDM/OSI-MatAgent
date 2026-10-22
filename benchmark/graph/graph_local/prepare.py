"""Prepare an immutable local graph study from explicit repository inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

from graph_local.run import digest, require_slurm, write

APP_DIRECTORIES = ('app', 'corpus', 'configs', 'cases', 'vendor')
LOCAL_KINDS = {'hf', 'huggingface', 'native', 'features'}
DEFAULTS = {'max_model_calls': 256, 'episode_seconds': 1800, 'graph_steps': 4096,
            'call_timeout_seconds': 180, 'seed': 0}


def ignored(_directory, names):
    return [name for name in names if name in {'__pycache__', '.git', '.cache', '.pytest_cache',
        'output', '.venv'} or name.endswith('.pyc')]


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def label(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Labels and experiment names must be safe nonempty identifiers')
    return value


def resolve_spec(spec, plan_directory, model_root=None, artifact_root=None):
    spec = copy.deepcopy(spec)
    kind = spec.get('kind', spec.get('backend'))
    if kind not in LOCAL_KINDS:
        raise ValueError('Only local HF, native and feature backends are accepted')
    spec['kind'] = kind
    for field in ('model_path', 'artifact_path'):
        if field not in spec:
            continue
        value = str(spec[field])
        for placeholder, root in (('{MODEL_ROOT}', model_root), ('{ARTIFACT_ROOT}', artifact_root)):
            if placeholder in value:
                if root is None:
                    raise ValueError(placeholder + ' requires its explicit CLI root')
                value = value.replace(placeholder, str(Path(root).expanduser().resolve()))
        if '{' in value or '}' in value or '://' in value:
            raise ValueError('Unresolved or nonlocal artifact path: ' + value)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = plan_directory / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        spec[field] = str(path)
    if kind in {'hf', 'huggingface', 'native'} and ('model_path' not in spec or not Path(spec['model_path']).is_dir()):
        raise ValueError('Neural backends require an existing local model_path directory')
    if kind == 'features' and ('artifact_path' not in spec or
            not (Path(spec['artifact_path']) / 'manifest.json').is_file() or
            not (Path(spec['artifact_path']) / 'decision.pkl').is_file()):
        raise ValueError('Feature backends require a local artifact directory with decision.pkl and manifest.json')
    if kind == 'native' and spec.get('family', 'openjev') not in {'openjev', 'apus'}:
        raise ValueError('Native family must be actual OpenJev or APUS')
    return spec


def experiments_from_plan(plan, *, plan_directory, apus_python, model_root=None, artifact_root=None):
    defaults = {**DEFAULTS, **plan.get('defaults', {})}
    if 'experiments' in plan:
        if 'experts' in plan or 'choosers' in plan:
            raise ValueError('Use explicit experiments or a crossed experts/choosers matrix')
        experiments = copy.deepcopy(plan['experiments'])
    else:
        experts, choosers = plan.get('experts', []), plan.get('choosers', [])
        if not experts or not choosers:
            raise ValueError('A plan requires explicit experiments or nonempty experts and choosers')
        if len({label(row['label']) for row in experts}) != len(experts):
            raise ValueError('Expert labels must be unique')
        if len({label(row['label']) for row in choosers}) != len(choosers):
            raise ValueError('Chooser labels must be unique')
        experiments = []
        for expert in experts:
            for chooser in choosers:
                name = 'graph-' + label(expert['label']) + '-to-' + label(chooser['label'])
                experiments.append({**defaults, 'name': name + '-seed' + str(defaults['seed']),
                    'analysis_family': name, 'expert_row': expert['label'],
                    'chooser_column': chooser['label'], 'expert': copy.deepcopy(expert['backend']),
                    'backend': copy.deepcopy(chooser['backend']), 'method': chooser.get('method')})
    if not experiments:
        raise ValueError('The declared experiment list is empty')
    resolved = []
    for supplied in experiments:
        experiment = {**defaults, **supplied}
        if type(experiment['seed']) is not int or experiment['seed'] < 0:
            raise ValueError('Seed must be a nonnegative integer')
        label(experiment['name'])
        experiment['architecture'] = 'repository_langgraph'
        experiment['tracks'] = ['sequential']
        for field in ('expert', 'backend'):
            experiment[field] = resolve_spec(experiment[field], plan_directory, model_root, artifact_root)
        if experiment['expert']['kind'] not in {'hf', 'huggingface'}:
            raise ValueError('Specialists require an actual local generative HF expert')
        expert_tokens = experiment['expert'].get('max_new_tokens', 4096)
        if type(expert_tokens) is not int or expert_tokens < 1:
            raise ValueError('Expert max_new_tokens must be positive')
        experiment['expert']['max_new_tokens'] = expert_tokens
        kind = experiment['backend']['kind']
        method = {'hf': 'logprobs', 'huggingface': 'logprobs', 'native': 'native', 'features': 'features'}[kind]
        allowed_methods = {None, method, 'native'} if kind == 'features' else {None, method}
        if experiment.get('method') not in allowed_methods:
            raise ValueError('Chooser method differs from its actual local readout')
        experiment['method'] = method
        if kind in {'hf', 'huggingface'}:
            experiment['backend']['max_new_tokens'] = expert_tokens
        if kind == 'native' and experiment['backend'].get('family') == 'apus':
            if apus_python is None or not Path(apus_python).expanduser().absolute().is_file():
                raise ValueError('APUS requires an explicit existing --apus-python interpreter')
            experiment['apus_python'] = str(Path(apus_python).expanduser().absolute())
        else:
            experiment['apus_python'] = str(Path(apus_python).expanduser().absolute()) if apus_python else None
        for field in ('max_model_calls', 'graph_steps'):
            if type(experiment[field]) is not int or experiment[field] < 1:
                raise ValueError(field + ' must be a positive integer')
        for field in ('episode_seconds', 'call_timeout_seconds'):
            if type(experiment[field]) not in (int, float) or not 0 < experiment[field] < float('inf'):
                raise ValueError(field + ' must be finite and positive')
        resolved.append(experiment)
    if len({item['name'] for item in resolved}) != len(resolved):
        raise ValueError('Experiment names must be unique')
    return resolved


def prepare(args):
    require_slurm()
    source = args.source.expanduser().resolve()
    app_source = args.app_source.expanduser().resolve()
    comparison_source = args.comparison_source.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    environment_path = args.environment_receipt.expanduser().resolve()
    plan = read_json(plan_path)
    environment = read_json(environment_path)
    if environment.get('status') != 'validated':
        raise ValueError('A validated local graph environment receipt is required')
    experiments = experiments_from_plan(plan, plan_directory=plan_path.parent,
        apus_python=args.apus_python, model_root=args.model_root, artifact_root=args.artifact_root)
    if args.shards < 1:
        raise ValueError('Shards must be positive')
    if not (source / 'graph_local/run.py').is_file() or not (app_source / 'app/graph.py').is_file():
        raise ValueError('Source must contain graph_local; app-source must contain the actual app/graph.py')
    for directory in APP_DIRECTORIES:
        if not (app_source / directory).is_dir():
            raise FileNotFoundError(app_source / directory)
    if not (comparison_source / 'comparison/native.py').is_file() or not (comparison_source / 'instrument/manifest.json').is_file():
        raise ValueError('comparison-source must contain sibling comparison/ and instrument/ directories')
    import sys
    sys.path.insert(0, str(comparison_source))
    from comparison.policies import load_bundle
    helpers = load_bundle(comparison_source / 'instrument')
    helpers['validate'].validate_bundle(comparison_source / 'instrument')
    public_cases = list(map(json.loads,
        (comparison_source / 'instrument/sequential_120/cases.jsonl').read_text().splitlines()))
    if len(public_cases) != 120 or len({case['task_id'] for case in public_cases}) != 120:
        raise ValueError('The primary graph study requires the original 120-case cohort')
    result = subprocess.run(['git', '-C', str(app_source), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False)
    app_commit = result.stdout.strip() if result.returncode == 0 else None
    if args.app_revision is not None and app_commit != args.app_revision:
        raise ValueError('Application checkout differs from --app-revision')
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / 'snapshot'
    snapshot.mkdir()
    shutil.copytree(source / 'graph_local', snapshot / 'graph_local', ignore=ignored)
    for directory in ('comparison', 'instrument'):
        shutil.copytree(comparison_source / directory, snapshot / directory, ignore=ignored)
    for directory in APP_DIRECTORIES:
        shutil.copytree(app_source / directory, snapshot / 'upstream' / directory, ignore=ignored)
    package = snapshot / 'package'
    package.mkdir()
    for directory in ('tools', 'examples', 'docs'):
        if (source / directory).is_dir():
            shutil.copytree(source / directory, package / directory, ignore=ignored)
    for filename in ('README.md', 'requirements-graph.txt'):
        if (source / filename).is_file():
            shutil.copy2(source / filename, package / filename)
    provenance = snapshot / 'provenance'
    provenance.mkdir()
    shutil.copy2(plan_path, provenance / 'input-plan.json')
    shutil.copy2(environment_path, provenance / 'environment.json')
    write(provenance / 'application.json', {'source_path': str(app_source),
        'git_commit': app_commit, 'expected_revision': args.app_revision,
        'source_is_versioned_checkout': app_commit is not None,
        'authority': 'Exact per-file snapshot hashes; git commit is supplementary provenance'})
    inventory = {path.relative_to(snapshot).as_posix(): digest(path)
        for path in sorted(snapshot.rglob('*')) if path.is_file()}
    write(destination / 'source-manifest.json', inventory)
    source_hash = digest(destination / 'source-manifest.json')
    for experiment in experiments:
        experiment['graph_source_manifest_sha256'] = source_hash
    config = {'schema_version': 1, 'bundle': str(snapshot / 'instrument'), 'experiments': experiments,
        'tracks': {'sequential': {'mode': args.mode}},
        'analysis': {**plan.get('analysis', {}),
            'primary': {
                'free_run': 'All120 autonomous free_run',
                'teacher_forced': 'All120 teacher_forced original-proposal accuracy and reference corrections; goal completion is assisted',
                'both': 'All120 autonomous free_run; teacher_forced is a separate diagnostic',
            }[args.mode],
            'classification': 'Instrument failure mechanism, condition and workflow',
            'matrix_rows': list(dict.fromkeys(item.get('expert_row', item['name']) for item in experiments)),
            'matrix_columns': list(dict.fromkeys(item.get('chooser_column', item['name']) for item in experiments)),
            'pdf_relationship': 'Actual unchanged app.graph plus a new public benchmark adapter; not exact reproduction',
            'hosted_inference_calls': 0, 'independent_expert_review_completed': False},
        'preparation_inputs': {'input_plan_sha256': digest(plan_path),
            'environment_receipt_sha256': digest(environment_path), 'application_git_commit': app_commit}}
    write(destination / 'configuration.json', config)
    jobs = [{'experiment': experiment['name'], 'shard_index': index, 'shards': args.shards, 'mode': args.mode}
        for experiment in experiments for index in range(args.shards)]
    write(destination / 'jobs.json', jobs)
    import os
    write(destination / 'prepared.json', {'source_manifest_sha256': source_hash,
        'configuration_sha256': digest(destination / 'configuration.json'),
        'jobs_sha256': digest(destination / 'jobs.json'), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'compute_node': os.environ['SLURMD_NODENAME'], 'experiments': len(experiments),
        'shards': args.shards, 'workers': len(jobs),
        'episodes_per_experiment': 120 * (2 if args.mode == 'both' else 1), 'jobs_submitted': False})
    print(json.dumps({'prepared': str(destination), 'workers': len(jobs), 'source_manifest_sha256': source_hash}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--app-source', type=Path, required=True)
    parser.add_argument('--app-revision', help='Optional exact git revision to require before snapshotting')
    parser.add_argument('--comparison-source', type=Path, required=True)
    parser.add_argument('--environment-receipt', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--apus-python', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('both', 'free_run', 'teacher_forced'), default='both')
    parser.add_argument('--shards', type=int, default=12)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
