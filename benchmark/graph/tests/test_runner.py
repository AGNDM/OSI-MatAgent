"""Freeze, model provenance and independent-episode recovery contracts.

These are controller tests, not model-performance measurements.  Execute only
through the compute-node test allocation alongside the actual graph checks.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if not os.environ.get('SLURMD_NODENAME') or not os.environ.get('SLURM_JOB_ID'):
    pytest.skip('Runner checks require an actual Slurm allocation', allow_module_level=True)

from graph_local import run


@pytest.fixture
def frozen_study(tmp_path):
    study = tmp_path / 'study with spaces'
    source = study / 'snapshot/graph_local/policy.py'
    source.parent.mkdir(parents=True)
    source.write_text('# Explicit test fixture, not benchmark implementation.\n')
    run.write(study / 'source-manifest.json', {'graph_local/policy.py': run.digest(source)})
    run.write(study / 'configuration.json', {'experiments': [{'name': 'fixture-only', 'seed': 0}]})
    run.write(study / 'jobs.json', [{'experiment': 'fixture-only', 'shard_index': 0, 'shards': 1}])
    run.write(study / 'prepared.json', {
        'configuration_sha256': run.digest(study / 'configuration.json'),
        'jobs_sha256': run.digest(study / 'jobs.json'),
        'source_manifest_sha256': run.digest(study / 'source-manifest.json')})
    return study


def test_authenticated_study_binds_configuration_job_plan_and_source(frozen_study):
    snapshot, manifest = run.authenticate(frozen_study)
    assert snapshot == frozen_study / 'snapshot'
    assert manifest == {'graph_local/policy.py': run.digest(snapshot / 'graph_local/policy.py')}


@pytest.mark.parametrize('filename', ['configuration.json', 'jobs.json', 'source-manifest.json'])
def test_configuration_and_jobs_cannot_change_after_prepare(frozen_study, filename):
    target = frozen_study / filename
    target.write_text(target.read_text() + '\n')  # Even semantic-preserving byte changes must be detected.
    with pytest.raises(ValueError):
        run.authenticate(frozen_study)


def test_authenticated_source_still_rejects_changed_code(frozen_study):
    (frozen_study / 'snapshot/graph_local/policy.py').write_text('# Changed after the freeze.\n')
    with pytest.raises(ValueError):
        run.authenticate(frozen_study)


def test_authenticated_source_still_rejects_foreign_files(frozen_study):
    (frozen_study / 'snapshot/unlisted.py').write_text('# Unlisted implementation.\n')
    with pytest.raises(ValueError):
        run.authenticate(frozen_study)


def test_source_inventory_is_checked_even_when_manifest_hash_matches_prepared(frozen_study):
    manifest_path = frozen_study / 'source-manifest.json'
    run.write(manifest_path, {'graph_local/policy.py': 'wrong bytes'})
    prepared = json.loads((frozen_study / 'prepared.json').read_text())
    prepared['source_manifest_sha256'] = run.digest(manifest_path)
    run.write(frozen_study / 'prepared.json', prepared)
    with pytest.raises(ValueError):
        run.authenticate(frozen_study)


class FakeChild:
    """Counts attempts to prove recovery never retries an inference call."""
    def __init__(self, closed=False):
        self.closed = closed
        self.decide_calls = 0

    def decide(self, *_args, **_kwargs):
        self.decide_calls += 1
        raise AssertionError('Recovery must not invoke model inference')


def recovery_arguments(tmp_path):
    return {'python': Path('/fixture/pinned-apus/bin/python'),
            'comparison_root': tmp_path / 'frozen comparison',
            'stderr_path': tmp_path / 'next-independent-episode.stderr',
            'timeout_seconds': 180}


def test_live_child_is_preserved_without_constructor_or_model_call(tmp_path):
    child = FakeChild()
    def must_not_construct(*_args, **_kwargs):
        raise AssertionError('Live native process must be reused')
    returned, recreated = run.recover_closed_child(child,
        {'kind': 'native', 'family': 'apus', 'model_path': '/fixture/native'},
        **recovery_arguments(tmp_path), factory=must_not_construct)
    assert returned is child and recreated is False
    assert child.decide_calls == 0


def test_closed_child_recreated_once_with_exact_original_backend_and_no_retry(tmp_path):
    old = FakeChild(closed=True)
    old.decide_calls = 1  # The previous episode's failed call remains its sole attempt.
    new = FakeChild()
    spec = {'kind': 'native', 'family': 'apus', 'model_path': '/fixture/native',
            'effort': 'low', 'device': 'cuda:0', 'declared_nested_fixture': {'seed': 0}}
    original = deepcopy(spec)
    kwargs = recovery_arguments(tmp_path)
    calls = []
    def factory(actual_spec, **actual_kwargs):
        calls.append((deepcopy(actual_spec), actual_kwargs))
        return new
    returned, recreated = run.recover_closed_child(old, spec, **kwargs, factory=factory)
    assert returned is new and recreated is True
    assert calls == [(original, kwargs)] and spec == original
    assert old.decide_calls == 1 and new.decide_calls == 0


def test_recovery_construction_failure_does_not_retry_or_fallback(tmp_path):
    calls = []
    def failing_factory(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError('Explicit native process startup failure')
    old = FakeChild(closed=True)
    with pytest.raises(RuntimeError, match='startup failure'):
        run.recover_closed_child(old, {'kind': 'native', 'family': 'apus'},
            **recovery_arguments(tmp_path), factory=failing_factory)
    assert len(calls) == 1 and old.decide_calls == 0


class LazyManifest:
    def __init__(self):
        self.identity = {'family': 'openjev', 'loaded': False}

    def manifest(self):
        return {'model': deepcopy(self.identity)}


def test_current_identity_captures_real_lazy_loaded_model_hashes():
    roles = SimpleNamespace(manifest=lambda: {'model': {'weight_sha256': {'expert': 'expert-fixed-bytes'}}})
    chooser = LazyManifest()
    spec = {'expert': {'model_path': '/fixture/expert'}, 'backend': {'model_path': '/fixture/native'}}
    before = run.current_identity(roles, chooser, spec, 'source-manifest-fixture-sha', shared=False)
    chooser.identity.update(loaded=True, local_files_sha256={'native-weights': 'native-loaded-real-hash-fixture'})
    after = run.current_identity(roles, chooser, spec, 'source-manifest-fixture-sha', shared=False)
    assert 'native-loaded-real-hash-fixture' not in json.dumps(before)
    assert 'native-loaded-real-hash-fixture' in json.dumps(after)
    assert 'expert-fixed-bytes' in json.dumps(after)
    assert 'source-manifest-fixture-sha' in json.dumps(after)


def test_initialization_failure_identity_retains_original_specs_and_error():
    spec = {'expert': {'model_path': '/fixture/expert'}, 'backend': {'model_path': '/fixture/native'}}
    identity = run.current_identity(None, None, spec, 'source-sha',
                                   initialization_error='Pinned native dependency unavailable')
    text = json.dumps(identity)
    assert '/fixture/expert' in text and '/fixture/native' in text
    assert 'Pinned native dependency unavailable' in text
