"""Copy authenticated historical MCQA snapshot methods into the schema1.0 export.

This reconstructs the overall input to export_groups/enrich without inference.
The explicit metadata profile binds the historical dataset/model declarations;
it is not a way to infer a different experiment's models or evaluation scope.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    profile = json.loads(args.metadata.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != profile['expected_snapshot_sha256']:
        raise ValueError('Snapshot differs from the explicitly declared historical metadata profile')
    document = deepcopy(profile['document'])
    methods = [{key: value for key, value in row.items()
                if key not in {'teacher_forced_attempts', 'free_run_attempts'}}
               for row in snapshot['methods']]
    if (len(methods) != 34 or len({row['experiment'] for row in methods}) != 34
            or snapshot['all_applicable_mcqa_attempts_present'] is not True):
        raise ValueError('Expected the authenticated complete historical34-configuration MCQA snapshot')
    document.update(methods=methods, snapshot_generated_utc=snapshot['generated_utc'],
        all_applicable_mcqa_attempts_present=snapshot['all_applicable_mcqa_attempts_present'])
    document['provenance'] = {
        'source_snapshot': str(args.snapshot.resolve()), 'source_snapshot_sha256': digest,
        'source_slurm_job_id': snapshot['slurm_job_id'], 'source_compute_node': snapshot['compute_node'],
        'source_snapshot_generated_utc': snapshot['generated_utc'],
        'export': 'Stored MCQA metrics copied without inference or score recomputation.',
        'exporter': str(Path(__file__).resolve()),
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'export_slurm_job_id': os.environ['SLURM_JOB_ID']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'method_count': len(methods),
                      'neural_inference_performed': False}), flush=True)


if __name__ == '__main__':
    main()
