"""Authenticate and sanitize the two exact historical static inference profiles.

Run only in a real Slurm allocation. No weights, fitted pickles or original
files are modified; no model, simulator, scorer or pickle loader is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


MODEL_KEYS = {
    'qwen2.5-3b': 'MODEL_QWEN25_3B',
    'qwen3-4b-instruct-2507': 'MODEL_QWEN3_4B',
    'openjev8b': 'MODEL_OPENJEV8B',
    '4B-5949': 'MODEL_APUS_REFLEX4B',
    'minilm-l6': 'MODEL_MINILM_L6',
}
ARTIFACT_KEYS = {
    'minilm-cosine-seed0': 'ARTIFACT_MINILM_COSINE',
    'minilm-linear-seed0': 'ARTIFACT_MINILM_LINEAR',
    'minilm-knn-seed0': 'ARTIFACT_MINILM_KNN',
    'qwen25-hidden-linear-seed0': 'ARTIFACT_QWEN25_HIDDEN_LINEAR',
    'reference-majority-seed0': 'ARTIFACT_MAJORITY',
}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def proof(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_profile(study, label, expected_count, static):
    study = study.resolve()
    metadata = read(study / 'study.json')
    inventory = metadata['snapshot_source_sha256']
    require(fingerprint(inventory) == metadata['snapshot_source_inventory_sha256'],
            'Stored source inventory digest differs: ' + label)
    root = study / 'snapshot'
    for relative, expected in inventory.items():
        path = root / ('configuration.json' if relative == 'configuration_input' else relative)
        require(sha(path) == expected, 'Historical source/configuration changed: ' + str(path))
    config = read(root / 'configuration.json')
    require(fingerprint(config) == metadata['configuration_sha256'],
            'Historical configuration binding differs: ' + label)
    require(config['tracks']['mcqa']['split'] == 'all' and metadata['shards'] == 8,
            'Expected all388 with eight original shards: ' + label)
    require(len(config['experiments']) == expected_count, 'Unexpected profile size: ' + label)
    require(sha(root / 'instrument/manifest.json') == static['bundle_sha256'],
            'Historical profile uses another instrument release: ' + label)
    return {'label': label, 'study': study, 'metadata': metadata, 'config': config,
            'experiments': {row['name']: row for row in config['experiments']},
            'manifest': {'profile': label, 'historical_study_manifest': proof(study / 'study.json'),
                'historical_configuration': proof(root / 'configuration.json'),
                'configuration_canonical_sha256': metadata['configuration_sha256'],
                'snapshot_source_inventory_sha256': metadata['snapshot_source_inventory_sha256'],
                'original_snapshot_source_sha256': inventory,
                'historical_tracks': config['tracks'], 'original_shards': metadata['shards'],
                'experiments': [row['name'] for row in config['experiments']]}}


def sanitize(config, *, label, originals):
    result = deepcopy(config)
    result['bundle'] = '${INSTRUMENT_BUNDLE}'
    # This scope change excludes historical sequential work; MCQA inputs stay exact.
    result['tracks'] = {'mcqa': deepcopy(config['tracks']['mcqa'])}
    result.setdefault('analysis', {}).update({
        'portable_profile': label,
        'execution_scope': 'MCQA only; use --track mcqa. Original sequential settings are retained only in provenance.',
        'historical_source_identity': 'See mcqa-used-profiles-provenance.json; source profiles remain separate.',
    })
    result['analysis'].pop('sequential', None)
    for row in result['experiments']:
        for field in ('backend', 'expert'):
            backend = row.get(field)
            if not backend:
                continue
            for path_key in ('model_path', 'tokenizer_path'):
                if path_key not in backend:
                    continue
                path = Path(backend[path_key]).resolve()
                key = MODEL_KEYS[path.name]
                require(key not in originals or originals[key] == str(path),
                        'One binding refers to different historical models: ' + key)
                originals[key] = str(path)
                backend[path_key] = '${' + key + '}'
            if 'artifact_path' in backend:
                key = ARTIFACT_KEYS[row['name']]
                originals[key] = str(Path(backend['artifact_path']).resolve())
                backend['artifact_path'] = '${' + key + '}'
    return result


def main():
    require(os.environ.get('SLURMD_NODENAME') and os.environ.get('SLURM_JOB_ID'),
            'A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--static-snapshot', type=Path, required=True)
    parser.add_argument('--r2-study', type=Path, required=True)
    parser.add_argument('--feature-study', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), 'Profile output must be a fresh directory')
    for source in (args.r2_study.resolve(), args.feature_study.resolve(),
                   args.static_snapshot.resolve().parent, repository / 'benchmark/instrument',
                   repository / 'results/388', repository / 'results/120'):
        require(not output.is_relative_to(source), 'Output is inside retained inputs: ' + str(source))
    static = read(args.static_snapshot)
    profile_metadata = read(repository / 'benchmark/evaluation/configs/mcqa-overall-metadata.json')
    require(sha(args.static_snapshot) == profile_metadata['expected_snapshot_sha256'],
            'The static snapshot differs from the published metadata profile')
    require(static['all_applicable_mcqa_attempts_present'] is True,
            'Static snapshot is incomplete')
    require(sha(repository / 'benchmark/instrument/manifest.json') == static['bundle_sha256'],
            'The receiving instrument bundle differs from the static snapshot')
    profiles = [load_profile(args.r2_study, 'mcqa-r2-22', 22, static),
                load_profile(args.feature_study, 'mcqa-features-12', 12, static)]
    owners = {}
    for profile in profiles:
        for name in profile['experiments']:
            require(name not in owners, 'Profiles overlap: ' + name)
            owners[name] = profile
    require(set(owners) == {row['experiment'] for row in static['methods']} and len(owners) == 34,
            'Used configurations do not exactly cover the published34 static methods')
    counts = Counter()
    shard_indices = {}
    identities, source_rows, ollama_receipts = {}, [], []
    for source in static['sources']:
        mcqa = {relative: digest for relative, digest in source['files'].items()
                if relative.endswith('/mcqa/predictions.jsonl')}
        if not mcqa:
            continue
        run = Path(source['path']).resolve()
        config_path = run / 'configuration.json'
        require(sha(config_path) == source['configuration_sha256'],
                'Saved MCQA worker configuration changed: ' + str(run))
        worker = read(config_path)
        require(worker['track'] == 'mcqa' and worker['limit'] is None
                and worker['selection']['shard_count'] == 8
                and worker['selection'].get('task_ids') is None,
                'Unexpected historical MCQA selection: ' + str(run))
        for relative in mcqa:
            name = Path(relative).parts[0]
            owner = owners[name]
            require(run.is_relative_to(owner['study'] / 'runs')
                    and source['comparison_source_inventory_sha256']
                    == owner['metadata']['snapshot_source_inventory_sha256']
                    and worker['selected_experiments'] == [name],
                    'Worker source/profile binding differs: ' + str(run))
            experiment = next(row for row in worker['resolved_configuration']['experiments']
                              if row['name'] == name)
            require(experiment == owner['experiments'][name],
                    'Executed experiment differs from historical input plan: ' + name)
            identity_relative = name + '/identity.json'
            identity_path = run / identity_relative
            require(sha(identity_path) == source['files'][identity_relative],
                    'Saved model identity changed: ' + str(identity_path))
            identities.setdefault(name, {'source': proof(identity_path),
                                         'historical_loaded_identity': read(identity_path)})
            if experiment['backend'].get('kind') == 'ollama':
                receipt_path = owner['study'] / 'snapshot/output' / (
                    'comparison-ollama-runtime-' + source['slurm_job_id'] + '.json')
                runtime = read(receipt_path)
                loaded = read(identity_path)
                require(runtime['runtime_version'] == loaded['server_version'],
                        'Worker runtime receipt differs from loaded Ollama identity: ' + name)
                ollama_receipts.append({'experiment': name, 'source': proof(receipt_path),
                                        'recorded_runtime': runtime})
            counts[name] += 1
            shard_indices.setdefault(name, set()).add(worker['selection']['shard_index'])
            source_rows.append({'experiment': name, 'profile': owner['label'],
                'configuration': proof(config_path), 'shard_index': worker['selection']['shard_index'],
                'identity': proof(identity_path),
                'predictions': {'path': str(run / relative), 'recorded_sha256': mcqa[relative]}})
    require(len(source_rows) == 272 and set(counts.values()) == {8},
            'Expected exactly eight authenticated MCQA workers for each of34 configurations')
    require(all(indices == set(range(8)) for indices in shard_indices.values()),
            'Historical MCQA shard selections are incomplete or duplicated')
    originals = {'INSTRUMENT_BUNDLE': str(repository / 'benchmark/instrument')}
    portable = {profile['label']: sanitize(profile['config'], label=profile['label'], originals=originals)
                for profile in profiles}
    renderer_path = repository / 'benchmark/evaluation/workflow/render_plan.py'
    spec = importlib.util.spec_from_file_location('static_profile_path_renderer', renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    sys.path.insert(0, str(repository / 'benchmark'))
    from comparison.run import validate_config
    for profile in profiles:
        rendered = renderer.render(portable[profile['label']], originals)
        validate_config(rendered)
        require(rendered['experiments'] == profile['config']['experiments'],
                'Sanitization changed inference fields: ' + profile['label'])
    artifacts = {}
    for key in sorted(set(ARTIFACT_KEYS.values())):
        directory = Path(originals[key])
        filenames = ('majority.json', 'majority-manifest.json') if key == 'ARTIFACT_MAJORITY' else ('decision.pkl', 'manifest.json')
        paths = {name: proof(directory / name) for name in filenames}
        manifest = read(directory / filenames[1])
        require(paths[filenames[0]]['sha256'] == manifest['artifact_sha256'],
                'Historical fitted artifact checksum differs: ' + key)
        artifacts[key] = {'files': paths, 'manifest': manifest,
                          'unpickled_or_retrained': False}
    r2 = profiles[0]
    ollama_binary = Path(r2['metadata']['ollama_bin'])
    lock = r2['study'] / 'snapshot/comparison/ollama-runtime.lock.json'
    binary_proof = proof(ollama_binary)
    lock_proof = proof(lock)
    for row in ollama_receipts:
        runtime = row['recorded_runtime']
        require(runtime['executable_sha256'] == binary_proof['sha256']
                and runtime['runtime_lock_sha256'] == lock_proof['sha256'],
                'Current staged Ollama binary/lock differs from its historical worker receipt')
    require(len(ollama_receipts) == 48, 'Expected six Ollama configurations with eight worker runtime receipts each')
    bindings = {key: '/absolute/path/to/' + key.lower() for key in sorted(originals)}
    bindings.update(OLLAMA_BIN='/absolute/path/to/ollama',
                    OLLAMA_MODELS='/absolute/path/to/dedicated-ollama-model-store')
    output.mkdir(parents=True)
    files = {}
    for label, config in portable.items():
        path = output / (label + '-plan.json')
        write(path, config)
        files[path.name] = proof(path)
    write(output / 'mcqa-bindings.example.json', bindings)
    manifest = {'schema_version': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'compute_node': os.environ['SLURMD_NODENAME'],
        'generator': proof(__file__), 'static_snapshot': proof(args.static_snapshot),
        'historical_profiles': [profile['manifest'] for profile in profiles],
        'portable_profiles': files,
        'historical_path_bindings': originals,
        'loaded_model_identities_by_experiment': identities,
        'fitted_artifacts': artifacts, 'mcqa_worker_sources': source_rows,
        'ollama_staging': {'historical_executable': binary_proof,
            'historical_model_store': r2['metadata']['ollama_models'],
            'runtime_lock': {'file': lock_proof, 'document': read(lock)},
            'worker_runtime_receipts': ollama_receipts,
            'runtime_receipt_scope': 'Supporting original worker receipts, matched to the published worker identities; their original bytes were not independently pinned by the static snapshot.',
            'required_tag': 'qwen25-3b-local:imported',
            'identity_authority': 'Retain the exact server version, tag digest, model details, tokenizer and template hashes recorded above. Importing or quantizing anew is a new runtime variant.'},
        'scope': {'track': 'mcqa', 'questions': 388, 'training': 232, 'validation': 42, 'heldout': 114,
                  'configuration_count': 34, 'profile_sizes': [22, 12], 'shards_per_configuration': 8,
                  'sequential_work_included': False,
                  'T1_reference': 'Restricted to applicable T1 rows; preserve original experiment reporting_scope.'},
        'validation': {'historical_source_and_configuration_hashes_match': True,
            'executed_worker_experiments_match_used_plans': True,
            'exact34_name_union_and_eight_shards_each': True,
            'placeholder_roundtrip_preserves_all_experiment_fields': True,
            'artifact_files_match_manifests': True,
            'prediction_hashes': 'Recorded authenticated snapshot hashes copied; predictions not rescored or reread.',
            'neural_inference_performed': False, 'training_performed': False,
            'originals_modified': False},
        'interpretation': 'Two historical source identities remain separate. These path-bound plans reproduce declared MCQA inference settings, not byte-identical stochastic outputs or historical sequential results. New execution records require fresh source/runtime identities.'}
    write(output / 'mcqa-used-profiles-provenance.json', manifest)
    documentation = '''# Exact used static inference profiles

These two profiles preserve the actual 22 + 12 configuration inputs behind the
published 34-configuration MCQA snapshot. They retain separate historical source
identities in mcqa-used-profiles-provenance.json; do not describe them as one
historical source freeze or as the later repository graph protocol.

Fill a COPY of mcqa-bindings.example.json with absolute staged paths. The existing
evaluation/workflow/render_plan.py resolves either profile with those explicit
bindings. This only changes paths and removes historical sequential track settings;
all experiment inference fields are checked against authenticated worker inputs.

Run all preparation, hashing and Python below inside real Slurm allocations.
For EACH profile, render a fresh configuration with render_plan.py --plan PROFILE
--bindings YOUR_BINDINGS --output FRESH_CONFIG. Then from benchmark/ prepare a
separate fresh study:

    "$PYTHON" -m comparison.study prepare --config "$FRESH_CONFIG" \\
      --output "$FRESH_STUDY" --track mcqa --shards 8 --max-parallel 24 \\
      --hf-python "$HF_PYTHON" --apus-python "$APUS_PYTHON"

The 22-row R2 profile additionally requires --ollama-bin "$OLLAMA_BIN" and
--ollama-models "$OLLAMA_MODELS". Stage the pinned Ollama executable and dedicated
store with the exact qwen25-3b-local:imported tag/digest and recorded F16/model,
tokenizer and template identities. No pull/import is automatic; recreating a tag
or changing conversion settings requires a new runtime identity. Read the original
binary/runtime lock and per-experiment identities in the provenance manifest.

Execute each generated submit.sh only after inspecting its fresh study.json.
For direct diagnostics use comparison.run --track mcqa explicitly. Neither plan
authorizes or implies any 120-case sequential workload. All 388 MCQA rows include
232 train and 42 validation diagnostics; report the 114 held-out rows separately.

Preserve strict ten-draw sampling with declared seeds 0/1/2, cited-confidence
variants, original menu seed 42, native effort depths, the four comparison-owned
shortlist pairs (384-token expert, max 4 candidates), direct 192-token caps and
restricted T1 reference. The 12-row profile requires externally staged, trusted
frozen feature artifacts and their complete original manifests; no pickle is
bundled or loaded by this generator. Retraining or changing library/source/encoder
identities defines another artifact. This is not the v11 teacher-forced graph grid.
'''
    (output / 'mcqa-used-profiles.md').write_text(documentation)
    print(json.dumps({'status': 'passed', 'output_dir': str(output),
        'profiles': list(portable), 'configurations': len(owners),
        'authenticated_mcqa_workers': len(source_rows), 'neural_inference_performed': False}), flush=True)


if __name__ == '__main__':
    main()
