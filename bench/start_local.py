"""Start one owned local model service and measure readiness without an autostart service."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx

from app.scenarios import ROOT
from bench.probe_local import resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new startup directory to retain previous logs")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid TCP port")
    try:
        httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=1, trust_env=False)
        parser.error("Port already serves HTTP; stop the previous service before measuring startup")
    except httpx.RequestError:
        pass
    args.output.mkdir(parents=True)
    command = [sys.executable, str(ROOT / "tools/serve_openjev.py"), "--port", str(args.port)]
    if args.cpu:
        command.append("--cpu")
    result = {"label": "CPU" if args.cpu else "Vulkan", "command": command,
        "resources_before": resources(), "ready": False,
        "note": "Process startup includes model integrity check and loading; OS/file/shader caches may already be warm."}
    start = time.monotonic()
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result["pid"] = process.pid
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() - start < 180:
            if process.poll() is not None:
                result["error"] = f"Model service exited {process.returncode}; inspect server.log"
                break
            try:
                response = client.get(f"http://127.0.0.1:{args.port}/health")
                pid_file = Path.home() / ".local/share/osi-hack/openjev-server.pid"
                own_pid = int(pid_file.read_text()) if pid_file.exists() else None
                if response.status_code == 200 and own_pid == process.pid and process.poll() is None:
                    result["ready"] = True
                    break
            except httpx.RequestError:
                pass
            time.sleep(.5)
    if not result["ready"] and process.poll() is None:
        result["error"] = "Startup exceeded 180 seconds"
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    result["startup_ms"] = (time.monotonic() - start) * 1000
    result["resources_after"] = resources()
    (args.output / "startup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
