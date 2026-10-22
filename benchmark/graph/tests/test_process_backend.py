"""Real IPC/process contract tests using explicitly fake native decisions.

No model performance is measured here.  Execute through the Slurm test job; the
fake child inherits the real allocation and never fabricates Slurm variables.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import graph_local.process_backend as ipc


MOCK_CHILD = r'''
import json, os, sys, time
import graph_local.process_backend as ipc

def interpreter():
    return {"executable":sys.executable,"executable_realpath":str(ipc.Path(sys.executable).resolve()),
            "executable_sha256":ipc._sha256_file(sys.executable),"version":sys.version,
            "prefix":sys.prefix,"base_prefix":sys.base_prefix,"transformers":"5.16.1",
            "torch":"explicit test double, not a loaded model", "test_double":True}

mode = os.environ.get("MOCK_APUS_MODE", "ok")
if mode == "actual_serve_with_fake_native":
    ipc._python_identity = interpreter
    ipc.serve(sys.argv[1])
    sys.exit(0)

for raw in sys.stdin:
    request=json.loads(raw)
    envelope={"protocol":ipc.PROTOCOL,"request_id":request["request_id"],"input_sha256":request["input_sha256"]}
    if request["operation"] == "initialize":
        spec=request["spec"]
        identity={"family":"apus","model_path":spec["model_path"],"effort":spec.get("effort","high"),
                  "mechanism":"trained_depth_candidate_projection","test_double":True}
        if mode == "wrong_model": identity["model_path"]="/different/test/weights"
        response={**envelope,"status":"ok","identity":identity,"interpreter":interpreter()}
    elif mode == "malformed":
        print("non-JSON diagnostics on the protocol channel",flush=True)
        continue
    elif mode == "partial_line":
        sys.stdout.write('{"unterminated":');sys.stdout.flush()
        time.sleep(20)
        continue
    elif mode == "exit":
        sys.exit(3)
    else:
        ids=request["option_ids"]
        result={"status":"ok","selected_option_id":ids[-1],"probabilities":{key:float(key==ids[-1]) for key in ids},
                "input_tokens":45,"output_tokens":0,"confidence":1.,"metadata":{"test_double":True,
                "received_messages":request["messages"],"received_descriptions":request["option_descriptions"]}}
        if mode == "native_error":
            result={"status":"error","selected_option_id":None,"probabilities":None,
                    "input_tokens":None,"output_tokens":None,"error":"Released native context limit exceeded",
                    "error_type":"infrastructure_error","metadata":{"error_kind":"infrastructure"}}
        response={**envelope,"status":"ok","result":result}
        if mode == "wrong_hash":response["input_sha256"]="wrong"
        if mode == "wrong_id":response["request_id"]="stale-or-replayed-request"
    print(json.dumps(response),flush=True)
'''


@pytest.fixture
def mock_factory(monkeypatch, tmp_path):
    real_popen = subprocess.Popen
    created = []
    captured = []
    native_root = tmp_path / "comparison source with spaces"
    (native_root / "comparison").mkdir(parents=True)
    (native_root / "comparison/__init__.py").write_text("")
    (native_root / "comparison/native.py").write_text('''
import os
class NativeBackend:
    def __init__(self, spec):
        self.identity={"family":"apus","model_path":spec["model_path"],
            "mechanism":"trained_depth_candidate_projection","test_double":True}
    def _load(self):
        print("native Python startup diagnostic")
        os.write(1,b"native C/extension startup diagnostic\\n")
    def decide(self,messages,option_ids,option_descriptions,**kwargs):
        print("native Python inference diagnostic")
        os.write(1,b"native C/extension inference diagnostic\\n")
        return {"status":"ok","selected_option_id":option_ids[-1],
            "probabilities":{key:float(key==option_ids[-1]) for key in option_ids},
            "input_tokens":33,"output_tokens":0,"metadata":{"test_double":True}}
''')

    def create(mode="ok", timeout=3):
        def mock_popen(command, **kwargs):
            captured.append({"command": command, "environment": kwargs["env"]})
            kwargs["env"] = {**kwargs["env"], "MOCK_APUS_MODE": mode}
            child = real_popen([sys.executable, "-u", "-c", MOCK_CHILD, str(native_root)], **kwargs)
            created.append(child)
            return child
        monkeypatch.setattr(ipc.subprocess, "Popen", mock_popen)
        backend = ipc.LocalProcessBackend(
            {"kind": "native", "family": "apus", "model_path": "/fixture/explicit-test-double",
             "effort": "high", "device": "cuda:0"},
            python=sys.executable, comparison_root=native_root,
            stderr_path=tmp_path / (mode + "-stderr.log"), timeout_seconds=timeout)
        return backend

    yield create, captured, created
    for child in created:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def decide(backend):
    return backend.decide(
        [{"role": "user", "content": "Current obs-7; previous obs-2; exact public manual and material observation."}],
        ["A", "B"], option_descriptions={"A": "Inspect the machine", "B": "Close the door"})


def test_persistent_child_roundtrips_real_ipc_and_usage(mock_factory):
    create, captured, children = mock_factory
    backend = create()
    try:
        first, second = decide(backend), decide(backend)
        assert first["status"] == second["status"] == "ok"
        assert first["probabilities"] == {"A": 0., "B": 1.}
        assert first["input_tokens"] == 45 and first["output_tokens"] == 0
        assert "previous obs-2" in first["metadata"]["received_messages"][0]["content"]
        assert first["metadata"]["received_descriptions"] == {"A": "Inspect the machine", "B": "Close the door"}
        assert first["metadata"]["subprocess_request_id"] != second["metadata"]["subprocess_request_id"]
        assert len(children) == 1
        identity = backend.identity["subprocess"]
        assert identity["interpreter"]["test_double"] is True
        assert identity["requested_executable_sha256"] == identity["interpreter"]["executable_sha256"]
        assert identity["slurm_job_id"] == os.environ["SLURM_JOB_ID"]
        assert captured[0]["environment"]["PYTHONNOUSERSITE"] == "1"
        assert len(captured[0]["environment"]["PYTHONPATH"].split(os.pathsep)) == 2
        assert captured[0]["command"][1:4] == ["-u", "-m", "graph_local.process_backend"]
    finally:
        backend.close()
    assert backend.closed and children[0].poll() is not None


def test_serve_isolates_python_and_native_fd_stdout(mock_factory):
    create, _, _ = mock_factory
    with create("actual_serve_with_fake_native") as backend:
        result = decide(backend)
        assert result["status"] == "ok" and result["input_tokens"] == 33
        log = backend.stderr_path.read_text()
        assert "Python startup diagnostic" in log and "C/extension startup diagnostic" in log
        assert "Python inference diagnostic" in log and "C/extension inference diagnostic" in log


@pytest.mark.parametrize("mode", ["malformed", "wrong_hash", "wrong_id", "exit"])
def test_bad_protocol_never_returns_a_model_choice_and_closes_child(mock_factory, mode):
    create, _, children = mock_factory
    backend = create(mode)
    result = decide(backend)
    assert result["status"] == "error" and result["selected_option_id"] is None
    assert result["probabilities"] is None and result["error_type"] == "infrastructure_error"
    assert result["metadata"]["automatic_retry"] is False
    assert backend.closed and children[0].poll() is not None
    assert decide(backend)["status"] == "error"


def test_partial_line_cannot_defeat_deadline(mock_factory):
    create, _, children = mock_factory
    backend = create("partial_line", timeout=2)
    backend.deadline = time.monotonic() + .15
    started = time.monotonic()
    result = decide(backend)
    assert time.monotonic() - started < 3
    assert result["status"] == "error" and "TimeoutError" in result["error"]
    assert backend.closed and children[0].poll() is not None


def test_expired_episode_deadline_sends_no_decision(mock_factory):
    create, _, _ = mock_factory
    backend = create()
    requests = backend._requests
    backend.deadline = time.monotonic() - 1
    result = decide(backend)
    assert result["status"] == "error" and backend._requests == requests
    assert backend.closed


def test_native_limit_error_is_preserved_without_poisoning_valid_transport(mock_factory):
    create, _, children = mock_factory
    with create("native_error") as backend:
        first, second = decide(backend), decide(backend)
        assert first["status"] == second["status"] == "error"
        assert first["error"] == "Released native context limit exceeded"
        assert not backend.closed and children[0].poll() is None


def test_handshake_model_mismatch_is_rejected(mock_factory):
    create, _, children = mock_factory
    with pytest.raises(ValueError, match="different model artifacts"):
        create("wrong_model")
    assert children[0].poll() is not None


def test_protocol_hash_binds_request_context_and_exact_options():
    request = {"protocol": ipc.PROTOCOL, "request_id": "test-only", "operation": "decide",
               "messages": [{"role": "user", "content": "public obs-7"}], "option_ids": ["A", "B"]}
    first = ipc._input_hash(request)
    request["option_ids"] = ["B", "A"]
    assert ipc._input_hash(request) != first
    with pytest.raises(ValueError, match="Duplicate"):
        ipc._load_json('{"request_id":"a","request_id":"b"}')


def test_allocation_guard_prevents_startup(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        ipc.LocalProcessBackend({"kind": "native", "family": "apus"}, sys.executable,
                                tmp_path, tmp_path / "stderr", timeout_seconds=1)
