"""Regenerate the original 388 cases from an authenticated portable runtime.

The upstream generate_dataset wrapper requires a Git checkout. The portable
release has no .git directory, so this tool authenticates all published runtime
bytes instead, then calls its unchanged generate_records implementation.
"""
import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Regeneration requires a genuine Slurm compute allocation')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Regeneration output must be fresh')
    sys.path.insert(0, str(bundle / 'eval'))
    loader = importlib.util.spec_from_file_location('instrument_release_validation', bundle / 'eval/validate.py')
    validation = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(validation)
    validation.validate_bundle(bundle)
    sys.path.insert(0, str(bundle))
    from sequential_120.runtime.bootstrap import activate
    vendor = activate()
    from instrument_bench.generator import generate_records, audit_release_files
    from instrument_bench.schema import DatasetConfig
    original = json.loads((bundle / 'mcqa_388/source-release-manifest.json').read_text())
    release = json.loads((bundle / 'mcqa_388/manifest.json').read_text())
    for relative, expected in original['generator_sha256'].items():
        if sha(vendor / relative) != expected:
            raise ValueError('Vendored generator differs from original release: ' + relative)
    allowed = {field.name for field in fields(DatasetConfig)}
    if set(original['config']) - allowed:
        raise ValueError('Original dataset configuration contains unsupported fields')
    config = DatasetConfig(**original['config'])
    output.mkdir(parents=True)
    private = output / 'private'
    private.mkdir(mode=0o700)
    task_path, oracle_path = output / 'tasks.jsonl', private / 'oracles.jsonl'
    count = 0
    with task_path.open('w') as task_file, oracle_path.open('w') as oracle_file:
        oracle_path.chmod(0o600)
        for task, oracle in generate_records(config):
            task_file.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + '\n')
            oracle_file.write(json.dumps(oracle, ensure_ascii=False, allow_nan=False) + '\n')
            count += 1
    actual = {'tasks': sha(task_path), 'oracles': sha(oracle_path)}
    expected = {'tasks': release['original_tasks_uncompressed_sha256'],
                'oracles': release['original_oracles_sha256']}
    audit = audit_release_files(task_path, oracle_path)
    identical = count == release['questions'] and actual == expected
    write(output / 'receipt.json', {'status': 'identical' if identical else 'mismatch',
        'task_count': count, 'expected_task_count': release['questions'],
        'actual_sha256': actual, 'expected_sha256': expected, 'audit': audit,
        'bundle': str(bundle), 'bundle_manifest_sha256': sha(bundle / 'manifest.json'),
        'source_release_manifest_sha256': sha(bundle / 'mcqa_388/source-release-manifest.json'),
        'generator_source_mode': 'Complete authenticated portable release, unchanged generate_records; no Git metadata required',
        'tool_sha256': sha(__file__), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'compute_node': os.environ['SLURMD_NODENAME']})
    validation.validate_bundle(bundle)
    print(json.dumps({'status': 'identical' if identical else 'mismatch', 'count': count, 'output': str(output)}), flush=True)
    if not identical:
        raise RuntimeError('Regenerated records differ from frozen release; original bundle was not modified')


if __name__ == '__main__':
    main()
