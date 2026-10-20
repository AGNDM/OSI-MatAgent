"""Record real local OpenJev decisions, latency, identity and memory usage."""
import argparse
import json
from pathlib import Path
import statistics

import httpx

from app.local_decision import LocalOpenJevBackend


def resources():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:")):
            key, value = line.split(":", 1)
            info[key + "_GiB"] = int(value.split()[0]) / 1024**2
    pid_path = Path.home() / ".local/share/osi-hack/openjev-server.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        try:
            info["server_pid"] = pid
            info["server_arguments"] = Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmSwap:")):
                    key, value = line.split(":", 1)
                    info[key + "_GiB"] = int(value.split()[0]) / 1024**2
        except (FileNotFoundError, PermissionError):
            info["server_process_readable"] = False
    for kind in ["vram", "gtt"]:
        path = Path(f"/sys/class/drm/card0/device/mem_info_{kind}_used")
        if path.exists():
            info[f"GPU_{kind}_used_GiB"] = int(path.read_text()) / 1024**3
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="Vulkan")
    parser.add_argument("--base-url", default="http://127.0.0.1:8792")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error("Use a new output file to retain earlier measurements")
    backend = LocalOpenJevBackend(args.base_url)
    identity = httpx.get(args.base_url + "/v1/models", timeout=5, trust_env=False)
    identity.raise_for_status()
    models = identity.json()
    if not any(m.get("id") == "openjev-flash-9b-q4" for m in models.get("data", [])):
        raise RuntimeError("Server alias differs from the pinned deployment")
    base = {"stage": "diagnose", "has_intent": False, "has_proposal": False,
        "has_inspection": False, "has_measurement": False, "measurement_quality": "unknown",
        "measurement_fresh": False, "pending_operation": False, "verifier_disagreement": False,
        "actions_used": 0, "llm_calls_used": 0}
    samples = [
        ("diagnose", base),
        ("prepare_action", dict(base, stage="prepare", has_inspection=True, has_intent=True, actions_used=1)),
        ("verify", dict(base, stage="verify", has_measurement=True, measurement_fresh=True,
                       measurement_quality="valid", actions_used=3)),
    ]
    results = []
    before = resources()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for expected, summary in samples:
            record = {"expected": expected, "repeat": repeat}
            try:
                decision = backend.decide(summary, ["diagnose", "prepare_action", "verify", "deliberate"])
                record.update(decision=decision.model_dump(), correct=decision.choice == expected)
                print(f"{args.label} {expected}: {decision.choice}, {decision.latency_ms:.1f} ms, p={decision.probabilities}", flush=True)
            except Exception as exc:
                record.update(error=f"{type(exc).__name__}: {exc}", correct=False)
                print(record["error"], flush=True)
            record["raw"] = backend.last_call
            results.append(record)
    times = [r["decision"]["latency_ms"] for r in results if "decision" in r]
    output = {"label": args.label, "backend": backend.manifest(), "server_identity": models,
        "resources_before": before, "resources_after": resources(),
        "attempted": len(results), "completed": len(times), "correct": sum(r["correct"] for r in results),
        "median_latency_ms": statistics.median(times) if times else None,
        "note": "Three simple routing states, not a laboratory benchmark; first request and cached requests retained separately. GPU and RSS memory may overlap.",
        "records": results}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output.resolve()}")
    if len(times) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
