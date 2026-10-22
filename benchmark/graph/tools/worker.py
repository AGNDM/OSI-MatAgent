"""Dispatch exactly one authenticated declared job from a frozen graph study."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--job-index', type=int)
    args = parser.parse_args()
    study = args.study.expanduser().resolve()
    sys.path[:0] = [str(study / 'snapshot'), str(study / 'snapshot/upstream')]
    from graph_local.run import authenticate, digest, run, write
    authenticate(study)
    receipt = json.loads((study / 'snapshot/provenance/environment.json').read_text())
    expected = {**receipt.get('packages', {}), **receipt.get('pinned_graph_closure', {}),
                **receipt.get('model_stack', {})}
    if not expected:
        raise ValueError('Environment receipt contains no authenticated package versions')
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError(f'Frozen environment changed: {name} {actual} != {version}')
    index = args.job_index
    if index is None:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    jobs = json.loads((study / 'jobs.json').read_text())
    if not 0 <= index < len(jobs):
        raise ValueError('Unknown frozen array job index')
    job = jobs[index]
    output = study / 'runs' / job['experiment'] / 'sequential' / f"shard-{job['shard_index']:03d}"
    write(study / 'launch-provenance' / (os.environ['SLURM_JOB_ID'] + '.json'), {
        'job': job, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'compute_node': os.environ['SLURMD_NODENAME'], 'launcher_sha256': digest(__file__),
        'study_source_manifest_sha256': digest(study / 'source-manifest.json'),
        'python': sys.executable, 'hosted_inference_calls': 0})
    run(SimpleNamespace(study=study, experiment=job['experiment'],
        shard_index=job['shard_index'], shards=job['shards'], mode=job['mode'],
        limit=None, output=output))


if __name__ == '__main__':
    main()
