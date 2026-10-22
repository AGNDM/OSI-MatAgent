"""Exercise portable preparation against a real authenticated instrument bundle."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import subprocess
import sys

import pytest

if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
    pytest.skip('Portability checks require a genuine Slurm allocation', allow_module_level=True)

from graph_local.prepare import experiments_from_plan, prepare, read_json, resolve_spec
from graph_local.run import authenticate


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "repository with space and apostrophe '"
    root.mkdir()
    for name in ('app', 'corpus', 'configs', 'cases', 'vendor'):
        (root / name).mkdir()
    # Explicitly artificial app source: preparation copies bytes and never
    # imports this fixture as an actual application implementation.
    (root / 'app/graph.py').write_text('# Portability fixture, no model or graph execution.\n')
    (root / 'app/provenance.txt').write_text('Exact original application fixture bytes.\n')
    (root / 'app/__pycache__').mkdir()
    (root / 'app/__pycache__/ignored.pyc').write_bytes(b'not executable')
    model = root / 'models/local-expert'
    model.mkdir(parents=True)
    plan = {'experts': [{'label': 'expert', 'backend': {'kind': 'hf', 'model_path': 'models/local-expert'}}],
        'choosers': [{'label': 'same', 'backend': {'kind': 'hf', 'model_path': 'models/local-expert'}}]}
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    receipt = root / 'environment.json'
    receipt.write_text(json.dumps({'status': 'validated', 'hosted_inference_calls': 0,
        'purpose': 'Explicit environment fixture; no real environment validation claim'}))
    package = Path(__file__).resolve().parents[1]
    comparison_source = Path(os.environ['GRAPH_PACKAGE_TEST_COMPARISON'])
    args = SimpleNamespace(source=package, app_source=root, comparison_source=comparison_source,
        environment_receipt=receipt, plan=plan_path, model_root=None, artifact_root=None,
        apus_python=None, app_revision=None, output=tmp_path / 'fresh study', shards=3, mode='both')
    return root, plan, args


def test_relocated_paths_freeze_real_bundle_and_only_snapshot_application(fixture):
    root, plan, args = fixture
    prepare(args)
    snapshot, inventory = authenticate(args.output)
    config = read_json(args.output / 'configuration.json')
    jobs = read_json(args.output / 'jobs.json')
    assert len(jobs) == 3 and {job['mode'] for job in jobs} == {'both'}
    assert read_json(args.output / 'prepared.json')['episodes_per_experiment'] == 240
    assert config['experiments'][0]['expert']['model_path'] == str((root / 'models/local-expert').resolve())
    assert (snapshot / 'upstream/app/provenance.txt').read_bytes() == (root / 'app/provenance.txt').read_bytes()
    assert not any('__pycache__' in path for path in inventory)
    assert (snapshot / 'provenance/input-plan.json').read_bytes() == args.plan.read_bytes()
    assert (snapshot / 'provenance/environment.json').read_bytes() == args.environment_receipt.read_bytes()
    assert not (args.source / 'upstream').exists()
    assert (snapshot / 'graph_local/reporting/matrix_report.py').is_file()
    assert (snapshot / 'package/docs/provenance/matrix-report-4635920/receipt.json').is_file()


@pytest.mark.parametrize('mode', ['free_run', 'teacher_forced', 'both'])
def test_preparation_primary_description_matches_requested_mode(fixture, mode):
    _, _, args = fixture
    args.mode = mode
    prepare(args)
    primary = read_json(args.output / 'configuration.json')['analysis']['primary']
    if mode == 'teacher_forced':
        assert 'teacher_forced' in primary and 'assisted' in primary and 'autonomous' not in primary
    else:
        assert 'autonomous free_run' in primary


@pytest.mark.parametrize('archive,implementation,test_name', [
    ('matrix-report-4635920', 'matrix_report.py', 'test_matrix_report.py'),
    ('reachability-4635941', 'reachability.py', 'test_reachability.py'),
])
def test_reporting_receipts_authenticate_original_source_and_archived_validation(archive, implementation, test_name):
    package = Path(__file__).resolve().parents[1]
    directory = package / 'docs/provenance' / archive
    receipt = read_json(directory / 'receipt.json')
    locations = {implementation: package / 'graph_local/reporting' / implementation,
        test_name: directory / (test_name + '.txt'), 'run.sbatch': directory / 'run.sbatch.txt'}
    for name, expected in receipt['source_sha256'].items():
        assert hashlib.sha256(locations[name].read_bytes()).hexdigest() == expected


def test_legacy_report_modules_keep_cli_and_import_entrypoints(tmp_path):
    from graph_local.reporting import matrix_report, reachability
    package = Path(__file__).resolve().parents[1]
    for relative, implementation in (
        ('tools/reporting/matrix_report.py', matrix_report),
        ('tools/reporting/reachability/reachability.py', reachability),
    ):
        path = package / relative
        spec = importlib.util.spec_from_file_location('legacy_' + implementation.__name__.rsplit('.', 1)[-1], path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        assert legacy.main is implementation.main
        subprocess.run([sys.executable, str(path), '--help'], cwd=tmp_path, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_frozen_report_launcher_finds_runtime_and_legacy_modules_without_checkout(fixture, tmp_path):
    _, _, args = fixture
    prepare(args)
    snapshot = args.output / 'snapshot'
    path = snapshot / 'package/tools/report.py'
    spec = importlib.util.spec_from_file_location('frozen_graph_report_wrapper', path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert wrapper.reporting_directory() == snapshot / 'graph_local/reporting'
    environment = {**os.environ, 'PYTHONPATH': ''}
    for relative in ('tools/reporting/matrix_report.py', 'tools/reporting/reachability/reachability.py'):
        subprocess.run([sys.executable, str(snapshot / 'package' / relative), '--help'],
            cwd=tmp_path, env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_common_launcher_discards_controller_cpu_mask():
    package = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'GRAPH_PACKAGE': str(package), 'SLURM_CPU_BIND': 'mask_cpu:0xFFFF',
        'SLURM_CPU_BIND_TYPE': 'mask_cpu', 'SLURM_CPU_BIND_LIST': '0xFFFF', 'SLURM_CPU_BIND_VERBOSE': 'verbose'}
    result = subprocess.run(['bash', '-c',
        'source "$1"; printf "%s\\n" "$SLURM_CPU_BIND" "${SLURM_CPU_BIND_TYPE-unset}" "${SLURM_CPU_BIND_LIST-unset}" "${SLURM_CPU_BIND_VERBOSE-unset}"',
        'graph-common-check', str(package / 'slurm/common.sh')],
        env=environment, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ['none', 'unset', 'unset', 'unset']


@pytest.mark.parametrize('relative', ['provenance/input-plan.json', 'provenance/environment.json', 'upstream/app/graph.py'])
def test_snapshot_input_and_application_tampering_detected(fixture, relative):
    _, _, args = fixture
    prepare(args)
    path = args.output / 'snapshot' / relative
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='source changed'):
        authenticate(args.output)


def test_existing_output_refused_without_changing_files(fixture):
    _, _, args = fixture
    args.output.mkdir()
    marker = args.output / 'keep.txt'
    marker.write_text('preserved')
    with pytest.raises(FileExistsError):
        prepare(args)
    assert marker.read_text() == 'preserved'


def test_local_matrix_crosses_rows_and_columns_without_mutating_plan(fixture):
    root, plan, _ = fixture
    plan['experts'].append({'label': 'second', 'backend': {'kind': 'hf', 'model_path': 'models/local-expert'}})
    plan['choosers'].append({'label': 'other', 'backend': {'kind': 'hf', 'model_path': 'models/local-expert'}})
    original = deepcopy(plan)
    rows = experiments_from_plan(plan, plan_directory=root, apus_python=None)
    assert len(rows) == 4
    assert {(row['expert_row'], row['chooser_column']) for row in rows} == {
        ('expert', 'same'), ('expert', 'other'), ('second', 'same'), ('second', 'other')}
    assert plan == original
    assert all(row['expert']['max_new_tokens'] == row['backend']['max_new_tokens'] == 4096 for row in rows)


@pytest.mark.parametrize('kind', ['openrouter', 'openai', 'remote', 'ollama', 'reference'])
def test_hosted_and_unsupported_provider_kind_rejected_before_snapshot(fixture, kind):
    _, plan, args = fixture
    plan['experts'][0]['backend']['kind'] = kind
    args.plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Only local'):
        prepare(args)
    assert not args.output.exists()


def test_apus_requires_explicit_separate_interpreter(fixture):
    root, plan, _ = fixture
    plan['choosers'][0]['backend'] = {'kind': 'native', 'family': 'apus',
        'model_path': 'models/local-expert', 'effort': 'low'}
    with pytest.raises(ValueError, match='apus-python'):
        experiments_from_plan(plan, plan_directory=root, apus_python=None)


def test_explicit_model_root_is_resolved_without_shell_expansion(fixture):
    root, _, _ = fixture
    spec = {'kind': 'hf', 'model_path': '{MODEL_ROOT}/local-expert'}
    with pytest.raises(ValueError, match='explicit CLI root'):
        resolve_spec(spec, root)
    resolved = resolve_spec(spec, root, model_root=root / 'models')
    assert resolved['model_path'] == str((root / 'models/local-expert').resolve())


def test_nonlocal_artifact_and_missing_model_are_rejected(fixture):
    root, _, _ = fixture
    with pytest.raises(ValueError, match='nonlocal'):
        resolve_spec({'kind': 'hf', 'model_path': 'https://example.com/model'}, root)
    with pytest.raises(ValueError, match='model_path'):
        resolve_spec({'kind': 'hf'}, root)


def test_duplicate_plan_keys_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"experts": [], "experts": []}')
    with pytest.raises(ValueError, match='Duplicate JSON field'):
        read_json(path)


def load_report_wrapper():
    path = Path(__file__).resolve().parents[1] / 'tools/report.py'
    spec = importlib.util.spec_from_file_location('portable_graph_report_wrapper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_protocol_fixture(study, suffix='v2'):
    from graph_local.reporting.matrix_report import digest, write_json
    for name in ('graph_local/profile.py', 'graph_local/policy.py',
                 'graph_local/backends.py', 'upstream/app/graph.py'):
        path = study / 'snapshot' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Explicit protocol fixture ' + suffix + '\n')
    manifest = {str(path.relative_to(study / 'snapshot')): digest(path)
        for path in (study / 'snapshot').rglob('*') if path.is_file()}
    write_json(study / 'source-manifest.json', manifest)
    preparation = read_json(study / 'prepared.json')
    preparation['source_manifest_sha256'] = digest(study / 'source-manifest.json')
    write_json(study / 'prepared.json', preparation)


def test_report_wrapper_rejects_pooling_different_protocols(tmp_path):
    from test_matrix_report import create_small_full_cohort_fixture
    left, _, _ = create_small_full_cohort_fixture(tmp_path / 'left')
    right, _, _ = create_small_full_cohort_fixture(tmp_path / 'right')
    add_protocol_fixture(left, 'v2')
    add_protocol_fixture(right, 'v3')
    with pytest.raises(ValueError, match='Do not pool different graph protocols'):
        load_report_wrapper().validate_protocols([left, right])


def test_default_packaged_report_entrypoint_renders_authenticated_fixture(tmp_path):
    from test_matrix_report import create_small_full_cohort_fixture
    study, combined, _ = create_small_full_cohort_fixture(tmp_path / 'fixture')
    add_protocol_fixture(study)
    package = Path(__file__).resolve().parents[1]
    reporter = package / 'graph_local/reporting/matrix_report.py'
    assert hashlib.sha256(reporter.read_bytes()).hexdigest() == '673999624748da641434ed688f32da346f377750dd90fcee7b64c786ee5a456a'
    output = tmp_path / 'rendered report with spaces'
    subprocess.run([sys.executable, str(package / 'tools/report.py'), '--study', str(study),
        '--combined-report', str(combined), '--output', str(output)], check=True)
    assert 'data:image/png;base64,' in (output / 'index.html').read_text()
    assert len(list((output / 'plots').glob('*.pdf'))) == 3
    assert len(list((output / 'plots').glob('*.svg'))) == 3
    assert len(list((output / 'plots').glob('*.png'))) == 3
    assert read_json(output / 'matrix.json')['all_expected_outputs_present'] is False
    assert (output / 'reachability/index.html').is_file()


def test_shipped_small_and_stronger_matrices_resolve_without_provider_substitution(tmp_path):
    package = Path(__file__).resolve().parents[1]
    models, artifacts = tmp_path / 'models', tmp_path / 'artifacts'
    for name in ('qwen2.5-3b', 'qwen3-4b-instruct-2507', 'qwen2.5-14b',
                 'deepseek-v2-lite-chat', 'openjev8b', 'apus-reflex/4B-5949', 'minilm-l6'):
        (models / name).mkdir(parents=True)
    for name in ('minilm/cosine', 'minilm/linear', 'minilm/knn', 'qwen25-hidden/linear'):
        path = artifacts / name
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{"explicit_planning_fixture": true}')
        (path / 'decision.pkl').write_bytes(b'not deserialized by the planner')
    for filename, expected in [('matrix.json', {'qwen25', 'qwen3'}),
                               ('stronger-matrix.json', {'qwen25-14b', 'deepseek-v2-lite'})]:
        rows = experiments_from_plan(read_json(package / 'examples' / filename),
            plan_directory=package / 'examples', model_root=models, artifact_root=artifacts,
            apus_python=sys.executable)
        assert len(rows) == 18 and {row['expert_row'] for row in rows} == expected
        assert len({row['chooser_column'] for row in rows}) == 9
    assert read_json(package / 'examples/provider-design.json')['executable'] is False


def test_portable_worker_dispatches_frozen_shard_and_mode_without_model_execution(fixture, monkeypatch):
    root, _, args = fixture
    receipt = read_json(args.environment_receipt)
    receipt['packages'] = {'transformers': '4.57.1'}
    args.environment_receipt.write_text(json.dumps(receipt))
    prepare(args)
    path = args.output / 'snapshot/package/tools/worker.py'
    spec = importlib.util.spec_from_file_location('portable_graph_worker_test', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    from graph_local import run as run_module
    calls = []
    monkeypatch.setattr(run_module, 'run', calls.append)
    monkeypatch.setattr(sys, 'argv', [str(path), '--study', str(args.output), '--job-index', '1'])
    worker.main()
    assert len(calls) == 1
    assert calls[0].shard_index == 1 and calls[0].shards == 3
    assert calls[0].mode == 'both' and calls[0].limit is None
    assert 'shard-001' in str(calls[0].output)
    assert not calls[0].output.exists()  # The injected dispatcher never runs inference.
