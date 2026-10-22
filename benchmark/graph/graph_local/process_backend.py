"""Persistent, allocation-local IPC for APUS's separately pinned interpreter.

There is no HTTP server or remote endpoint.  The master graph keeps its HF
expert environment while this child uses the actual APUS environment and native
head on the same allocated CUDA device.  Stdout is exclusively a checked JSON
protocol; all native runtime diagnostics go to the caller's stderr artifact.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import uuid


PROTOCOL = "graph-local-apus-ipc-v1"
MAX_LINE = 32 * 1024 * 1024


def _allocation():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Local model subprocesses require a real Slurm compute allocation")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hash(request):
    return hashlib.sha256(_json({key: value for key, value in request.items() if key != "input_sha256"}).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate IPC JSON field: " + key)
        result[key] = value
    return result


def _load_json(raw):
    def reject_constant(value):
        raise ValueError("Nonfinite IPC constant: " + value)
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    _json(value)  # Also reject floating-point overflow inside Any-valued fields.
    if not isinstance(value, dict):
        raise ValueError("IPC envelope must be a JSON object")
    return value


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    return float(value)


def _python_identity():
    executable = Path(sys.executable)
    return {"executable": str(executable), "executable_realpath": str(executable.resolve()),
            "executable_sha256": _sha256_file(executable), "version": sys.version,
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch")}


class LocalProcessBackend:
    """A comparison-compatible backend backed by one persistent APUS child.

    A timeout, invalid envelope or broken pipe poisons and closes the child.  The
    failed call is returned as an infrastructure error, with no retry, fabricated
    distribution, text-model substitution or implicit restart.  A subsequent
    episode may explicitly construct a fresh proxy.
    """

    def __init__(self, spec, python, comparison_root, stderr_path, timeout_seconds=180):
        _allocation()
        self.spec = dict(spec)
        if self.spec.get("kind", self.spec.get("backend")) != "native" or self.spec.get("family") != "apus":
            raise ValueError("Separate-process adapter requires the actual native APUS backend")
        if self.spec.get("method", "native") != "native":
            raise ValueError("Separate APUS process supports native readout only")
        self.timeout_seconds = _positive(timeout_seconds)
        # Keep the venv executable path: resolving its symlink would select the
        # base Python prefix and lose the pinned APUS environment.
        self.python = Path(python).expanduser().absolute()
        if not self.python.is_file():
            raise FileNotFoundError(self.python)
        self.comparison_root = Path(comparison_root).expanduser().resolve()
        if not (self.comparison_root / "comparison/native.py").is_file():
            raise FileNotFoundError("comparison_root must contain comparison/native.py")
        self.stderr_path = Path(stderr_path).expanduser().resolve()
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.deadline = float("inf")
        self._lock = threading.RLock()
        self._buffer = bytearray()
        self._closed = False
        self._failure = None
        self._requests = 0
        self._process = None
        self.identity = {"kind": "native", "family": "apus", "model_path": self.spec["model_path"],
                         "transport": PROTOCOL}
        graph_source = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = os.pathsep.join([str(graph_source), str(self.comparison_root)])
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["VIRTUAL_ENV"] = str(self.python.parent.parent)
        self._stderr = self.stderr_path.open("ab", buffering=0)
        try:
            self._process = subprocess.Popen(
                [str(self.python), "-u", "-m", "graph_local.process_backend", "--serve",
                 "--comparison-root", str(self.comparison_root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                env=environment, cwd=str(graph_source), bufsize=0, close_fds=True)
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            handshake = self._exchange("initialize", {"spec": self.spec}, time.monotonic() + self.timeout_seconds)
            if handshake.get("status") != "ok" or not isinstance(handshake.get("identity"), dict):
                raise RuntimeError("APUS child startup failed: " + str(handshake.get("error", "invalid handshake")))
            identity = handshake["identity"]
            interpreter = handshake.get("interpreter")
            expected_sha = _sha256_file(self.python)
            if not isinstance(interpreter, dict) or interpreter.get("executable_sha256") != expected_sha:
                raise ValueError("APUS child interpreter bytes differ from the requested interpreter")
            if Path(interpreter.get("prefix", "")).resolve() != self.python.parent.parent.resolve():
                raise ValueError("APUS child Python prefix differs from the requested environment")
            if identity.get("family") != "apus" or identity.get("mechanism") != "trained_depth_candidate_projection":
                raise ValueError("APUS child did not load the actual native candidate head")
            if interpreter.get("transformers") != "5.16.1":
                raise ValueError("APUS child is not using its pinned Transformers==5.16.1")
            if Path(identity.get("model_path", "")).resolve() != Path(self.spec["model_path"]).expanduser().resolve():
                raise ValueError("APUS child loaded different model artifacts")
            self.identity = {**identity, "transport": PROTOCOL,
                             "subprocess": {"interpreter": interpreter,
                                            "requested_python": str(self.python),
                                            "requested_executable_sha256": expected_sha,
                                            "requested_pyvenv_cfg_sha256": _sha256_file(self.python.parent.parent / "pyvenv.cfg")
                                            if (self.python.parent.parent / "pyvenv.cfg").is_file() else None,
                                            "comparison_root": str(self.comparison_root),
                                            "comparison_native_sha256": _sha256_file(self.comparison_root / "comparison/native.py"),
                                            "ipc_source_sha256": _sha256_file(Path(__file__)),
                                            "stderr_path": str(self.stderr_path),
                                            "child_pid": self._process.pid,
                                            "slurm_job_id": os.environ["SLURM_JOB_ID"],
                                            "slurmd_nodename": os.environ["SLURMD_NODENAME"],
                                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                            "same_allocation": True, "hosted_api_calls": 0,
                                            "automatic_retries": 0}}
        except Exception:
            self.close()
            raise

    @property
    def closed(self):
        return self._closed

    def _poll(self, descriptor, events, deadline):
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("Local APUS IPC deadline expired")
        poller = select.poll()
        poller.register(descriptor, events | select.POLLERR | select.POLLHUP)
        ready = poller.poll(max(1, math.ceil(timeout * 1000)))
        if not ready:
            raise TimeoutError("Local APUS IPC response timed out")
        return ready[0][1]

    def _write(self, payload, deadline):
        descriptor = self._process.stdin.fileno()
        data = memoryview(payload)
        while data:
            events = self._poll(descriptor, select.POLLOUT, deadline)
            if events & (select.POLLERR | select.POLLHUP):
                raise BrokenPipeError("APUS child closed its input pipe")
            try:
                written = os.write(descriptor, data[:65536])
            except BlockingIOError:
                continue
            if written <= 0:
                raise BrokenPipeError("APUS child accepted no input bytes")
            data = data[written:]

    def _read(self, deadline):
        descriptor = self._process.stdout.fileno()
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return line
            if len(self._buffer) > MAX_LINE:
                raise ValueError("APUS IPC response exceeds maximum line size")
            self._poll(descriptor, select.POLLIN, deadline)
            try:
                block = os.read(descriptor, 65536)
            except BlockingIOError:
                continue
            if not block:
                raise BrokenPipeError("APUS child closed its output pipe")
            self._buffer.extend(block)

    def _exchange(self, operation, payload, deadline):
        if self._closed:
            raise RuntimeError("APUS proxy is closed; construct a fresh proxy for another episode")
        self._requests += 1
        request = {"protocol": PROTOCOL, "request_id": str(self._requests) + "-" + uuid.uuid4().hex,
                   "operation": operation, **payload}
        request["input_sha256"] = _input_hash(request)
        encoded = (_json(request) + "\n").encode()
        if len(encoded) > MAX_LINE:
            raise ValueError("APUS IPC request exceeds maximum line size; input was not truncated")
        self._write(encoded, deadline)
        response = _load_json(self._read(deadline))
        if (response.get("protocol") != PROTOCOL or response.get("request_id") != request["request_id"]
                or response.get("input_sha256") != request["input_sha256"]):
            raise ValueError("APUS IPC protocol, request ID or input hash mismatch")
        if time.monotonic() >= deadline:
            raise TimeoutError("APUS IPC response arrived after its deadline")
        return response

    def decide(self, messages, option_ids, *, option_descriptions=None, method="native", samples=10,
               seed=0, temperature=.8, **kwargs):
        _allocation()
        start = time.monotonic()
        try:
            if kwargs:
                raise ValueError("Unknown APUS subprocess readout arguments: " + ", ".join(sorted(kwargs)))
            with self._lock:
                timeout = min(self.timeout_seconds, self.deadline - time.monotonic())
                if timeout <= 0:
                    raise TimeoutError("APUS subprocess episode deadline expired before inference")
                response = self._exchange("decide", {"messages": messages, "option_ids": list(option_ids),
                    "option_descriptions": option_descriptions, "kwargs": {"method": method,
                    "samples": samples, "seed": seed, "temperature": temperature}}, time.monotonic() + timeout)
                if response.get("status") != "ok" or not isinstance(response.get("result"), dict):
                    raise RuntimeError("APUS child rejected inference: " + str(response.get("error", "malformed result")))
                result = response["result"]
                # Model errors (including a released context limit) remain the
                # original native error envelope and do not poison IPC.
                result.setdefault("metadata", {}).update(subprocess_transport=PROTOCOL,
                    ipc_roundtrip_wall_time_s=time.monotonic() - start,
                    subprocess_request_id=response["request_id"],
                    subprocess_input_sha256=response["input_sha256"])
                return result
        except Exception as error:
            self._failure = f"{type(error).__name__}: {error}"
            self.close()
            return {"status": "error", "selected_option_id": None, "probabilities": None,
                    "confidence": None, "input_tokens": None, "output_tokens": None,
                    "causes_cited": [], "latency_s": time.monotonic() - start,
                    "error_type": "infrastructure_error", "error": self._failure,
                    "metadata": {"error_kind": "infrastructure", "subprocess_transport": PROTOCOL,
                                 "subprocess_closed": True, "automatic_retry": False,
                                 "backend_identity": self.identity}}

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        stderr = getattr(self, "_stderr", None)
        if stderr is not None:
            stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _emit(stream, request, **payload):
    stream.write(_json({"protocol": PROTOCOL, "request_id": request.get("request_id"),
                       "input_sha256": request.get("input_sha256"), **payload}) + "\n")
    stream.flush()


def serve(comparison_root):
    _allocation()
    root = Path(comparison_root).resolve()
    if not (root / "comparison/native.py").is_file():
        raise FileNotFoundError("Missing pinned comparison native backend")
    sys.path.insert(0, str(root))
    # Retain a dedicated protocol descriptor and redirect fd 1 itself.  This
    # captures both Python print() and native-extension/C stdout diagnostics.
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    backend = None
    seen_requests = set()
    while True:
        raw = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not raw:
            break
        if len(raw) > MAX_LINE:
            raise ValueError("IPC request exceeded maximum line size")
        request = _load_json(raw)
        if (request.get("protocol") != PROTOCOL or not isinstance(request.get("request_id"), str)
                or request["request_id"] in seen_requests or request.get("input_sha256") != _input_hash(request)):
            raise ValueError("Invalid or replayed APUS IPC request")
        seen_requests.add(request["request_id"])
        try:
            if request.get("operation") == "initialize":
                if backend is not None:
                    raise ValueError("APUS child already initialized")
                spec = request["spec"]
                if spec.get("kind", spec.get("backend")) != "native" or spec.get("family") != "apus":
                    raise ValueError("Child accepts actual native APUS only")
                with redirect_stdout(sys.stderr):
                    from comparison.native import NativeBackend
                    backend = NativeBackend(spec)
                    backend._load()
                    interpreter = _python_identity()
                _emit(protocol_output, request, status="ok", identity=backend.identity, interpreter=interpreter)
            elif request.get("operation") == "decide":
                if backend is None:
                    raise ValueError("APUS child has not initialized")
                with redirect_stdout(sys.stderr):
                    result = backend.decide(request["messages"], request["option_ids"],
                        option_descriptions=request["option_descriptions"], **request["kwargs"])
                _emit(protocol_output, request, status="ok", result=result)
            else:
                raise ValueError("Unknown APUS IPC operation")
        except Exception as error:
            _emit(protocol_output, request, status="error", error=f"{type(error).__name__}: {error}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--comparison-root", required=True)
    arguments = parser.parse_args()
    serve(arguments.comparison_root)


if __name__ == "__main__":
    main()
