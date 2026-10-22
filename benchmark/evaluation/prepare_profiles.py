"""Capture used-source provenance and sanitize explicit historical profiles in Slurm.

No model loading, fitting, simulation or score computation. Original artifacts
are read only; new portable configuration/provenance files are written separately.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path


ORIGINALS = {
    'workflow/report_grid.py': 'results/.deadline-rescue/report_grid.py',
    'workflow/compact_status.py': 'results/.deadline-rescue/compact_status.py',
    'workflow/audit_failures.py': 'results/.deadline-rescue/audit_all_closed_failures.py',
    'workflow/final_audit.py': 'results/.deadline-rescue/final_audit.py',
    'tests/test_final_audit.py': 'results/.deadline-rescue/test_final_audit.py',
    'tests/test_compact_status.py': 'results/.deadline-rescue/check_compact_status.py',
    'mcqa/export_groups.py': 'results/.mcqa-group-export/export.py',
    'mcqa/enrich.py': 'results/.mcqa-metrics-export/enrich.py',
    'mcqa/metrics.py': 'results/.mcqa-metrics-export/metrics.py',
    'mcqa/accounting.py': 'results/.mcqa-metrics-export/accounting.py',
    'configs/site-tariffs-20261022.yaml': 'results/.mcqa-metrics-export/site-tariffs.yaml',
}
MODEL_KEYS = {
    'qwen25-14b': 'MODEL_QWEN25_14B', 'deepseek-v2-lite': 'MODEL_DEEPSEEK_V2_LITE',
    'qwen25': 'MODEL_QWEN25_3B', 'qwen3': 'MODEL_QWEN3_4B',
    'qwen25-14b-self': 'MODEL_QWEN25_14B', 'deepseek-v2-lite-self': 'MODEL_DEEPSEEK_V2_LITE',
    'qwen25-self': 'MODEL_QWEN25_3B', 'qwen3-self': 'MODEL_QWEN3_4B',
    'qwen25-hf': 'MODEL_QWEN25_3B', 'qwen3-hf': 'MODEL_QWEN3_4B',
    'openjev8b': 'MODEL_OPENJEV8B', 'apus-low': 'MODEL_APUS_REFLEX4B',
    'apus-high': 'MODEL_APUS_REFLEX4B', 'minilm-cosine': 'MODEL_MINILM_L6',
    'minilm-linear': 'MODEL_MINILM_L6', 'minilm-knn': 'MODEL_MINILM_L6',
    'qwen25-hidden-linear': 'MODEL_QWEN25_3B',
}
ARTIFACT_KEYS = {'minilm-cosine': 'ARTIFACT_MINILM_COSINE',
    'minilm-linear': 'ARTIFACT_MINILM_LINEAR', 'minilm-knn': 'ARTIFACT_MINILM_KNN',
    'qwen25-hidden-linear': 'ARTIFACT_QWEN25_HIDDEN_LINEAR'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--mcqa-base', type=Path, required=True,
                        help='Original schema1.0 overall export, preserved before augmentation')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repo, study, target = (path.resolve() for path in (args.repository, args.study, args.output_dir))
    main_report = json.loads((repo / 'results/120/workflow-teacher-forced-results.json').read_text())
    mcqa = json.loads((repo / 'results/388/mcqa-results.json').read_text())
    audit = json.loads((repo / 'results/120/workflow-teacher-forced-final-audit.json').read_text())
    status = json.loads((repo / 'results/120/workflow-teacher-forced-status.json').read_text())
    expected = {'workflow/report_grid.py': main_report['provenance']['export_helper']['sha256'],
        'workflow/compact_status.py': status['source']['status_exporter_sha256'],
        'workflow/final_audit.py': audit['provenance']['audit_source']['sha256']}
    failure = json.loads(Path(audit['provenance']['final_all22_failure_audit']['path']).read_text())
    expected['workflow/audit_failures.py'] = failure['audit_helper_sha256']
    expected.update({'mcqa/' + name: digest for name, digest in mcqa['metrics_augmentation']['scripts_sha256'].items()})
    source_rows = []
    for destination, relative in ORIGINALS.items():
        original = repo / relative
        actual = sha(original)
        if destination in expected and actual != expected[destination]:
            raise ValueError('Original helper differs from final public provenance: ' + relative)
        source_rows.append({'portable_path': destination, 'original_path_relative_to_repository': relative,
            'original_sha256': actual, 'original_bytes': original.stat().st_size,
            'publicly_pinned_original': destination in expected})
    config = json.loads((study / 'configuration.json').read_text())
    if (main_report['study'] != str(study) or main_report['cohort']['cells'] != 22
            or main_report['mode'] != 'teacher_forced' or len(config['experiments']) != 22):
        raise ValueError('Expected exactly the final historical E teacher-forced study')
    experiments = deepcopy(config['experiments'])
    keys = set()
    for row in experiments:
        row.pop('graph_source_manifest_sha256')
        row.pop('apus_python', None)  # Supplied explicitly to graph prepare as --apus-python.
        for field, label in (('expert', row['expert_row']), ('backend', row['chooser_column'])):
            key = MODEL_KEYS[label]
            keys.add(key)
            row[field]['model_path'] = '${' + key + '}'
            if 'artifact_path' in row[field]:
                key = ARTIFACT_KEYS[label]
                keys.add(key)
                row[field]['artifact_path'] = '${' + key + '}'
    bindings = {key: '/absolute/path/to/' + key.lower() for key in sorted(keys)}
    write(target / 'workflow-e-bindings.example.json', bindings)
    write(target / 'workflow-e-plan.json', {'schema_version': 1, 'experiments': experiments,
        'analysis': {'protocol': main_report['protocol'], 'mode': 'teacher_forced',
            'primary': 'Original-action accuracy and corrections; goal completion is assisted.',
            'source_configuration_sha256': sha(study / 'configuration.json'),
            'note': 'Portable E model/protocol recipe only; prepare fresh source/configuration bindings and retain mode/case scope. Historical scheduler and evidence identities are not regenerated.'}})
    base = json.loads(args.mcqa_base.read_text())
    snapshot = Path(base['provenance']['source_snapshot'])
    if base['schema_version'] != '1.0' or len(base['methods']) != 34:
        raise ValueError('Expected the unaugmented historical overall export')
    metadata = {key: value for key, value in base.items()
                if key not in {'methods', 'provenance', 'snapshot_generated_utc'}}
    write(target / 'mcqa-overall-metadata.json', {'expected_snapshot_sha256': sha(snapshot), 'document': metadata})
    write(target / 'original-source-provenance.json', {'schema_version': 1,
        'source_files': source_rows, 'workflow_mode': 'teacher_forced',
        'workflow_cohort': main_report['cohort'],
        'workflow_frozen_hashes': {key: main_report['provenance'][key] for key in
            ('source_manifest_sha256', 'configuration_sha256', 'jobs_sha256', 'deadline_scope_sha256')},
        'workflow_reporting_inputs_sha256': main_report['provenance']['reporting_inputs_sha256'],
        'app_revision': main_report['provenance']['app_revision'],
        'mcqa_snapshot_sha256': sha(snapshot),
        'adaptation': 'Narrow explicit path/CLI parameterization; metrics, source checks and E closure gates retained. New helper hashes/timestamps differ from the historical exports.'})
    print(json.dumps({'output_dir': str(target), 'source_helpers': len(source_rows),
                      'workflow_experiments': len(experiments), 'neural_inference_performed': False}), flush=True)


if __name__ == '__main__':
    main()
