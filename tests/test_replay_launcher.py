"""Portability and read-only boundaries for the standalone replay entry point."""
import importlib.util
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launch_replay', ROOT / 'tools/launch_replay.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_bundle_has_complete_playable_history(tmp_path):
    assert launcher.install_example(tmp_path)
    replay = launcher.replay_viewer.build_replay(tmp_path, launcher.SAMPLE_NAME)
    assert replay['supported_model']
    assert len(replay['steps']) == 257
    assert replay['samples']
    assert any(step['category'] == 'model' for step in replay['steps'])
    original = (tmp_path / launcher.SAMPLE_NAME / 'result.json').read_bytes()
    assert not launcher.install_example(tmp_path)
    assert (tmp_path / launcher.SAMPLE_NAME / 'result.json').read_bytes() == original


def test_existing_history_is_preserved(tmp_path):
    folder = tmp_path / 'my-run'
    folder.mkdir()
    (folder / 'result.json').write_text('{}')
    assert not launcher.install_example(tmp_path)
    assert not (tmp_path / launcher.SAMPLE_NAME).exists()


def test_unexpected_archive_is_rejected(tmp_path, monkeypatch):
    archive = tmp_path / 'examples/replay/rc1mx-hydrolysis.zip'
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('../escape.json', '{}')
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Unexpected'):
        launcher.install_example(tmp_path / 'results')
    assert not (tmp_path / 'escape.json').exists()


def test_http_routes_and_instance_identity(tmp_path):
    server = launcher.ReplayHTTPServer(('127.0.0.1', 0), launcher.handler_for(tmp_path))
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert launcher.existing_server(port, launcher.instance_id(tmp_path))
        assert not launcher.existing_server(port, 'some-other-folder')
        for path in ['/replay', '/', '/api/replay/runs', '/api/runs', '/replay/model/rc1mx.glb']:
            with opener.open(f'http://127.0.0.1:{port}{path}') as response:
                assert response.status == 200
                assert response.read()
        for path in ['/.env', '/api/state', '/api/run', '/../../.env']:
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(f'http://127.0.0.1:{port}{path}')
            assert error.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(f'http://127.0.0.1:{port}/api/run', data=b'{}')
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_existing_port_cannot_be_claimed_twice(tmp_path):
    # Match the legacy demo's reusable socket: the new listener must still fail.
    with launcher.ThreadingHTTPServer(('127.0.0.1', 0), launcher.handler_for(tmp_path)) as occupied:
        with pytest.raises(OSError):
            launcher.ReplayHTTPServer(('127.0.0.1', occupied.server_port), launcher.handler_for(tmp_path))
