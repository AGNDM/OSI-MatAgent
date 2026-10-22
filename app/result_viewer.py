"""Read-only views over completed native workflow records. No model calls."""
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TRACE_BYTES = 64 * 1024 * 1024
SECRET = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._-]+", re.I)
HIDDEN_KEYS = {"authorization", "api_key", "openrouter_api_key", "access_token",
               "reasoning_content", "reasoning_details"}
TRACE_PATHS = ("selection-trace.jsonl", "execution/trace.jsonl", "trace.jsonl")


def sanitize(value):
    if isinstance(value, dict):
        return {k: ("[redacted]" if k.lower() in HIDDEN_KEYS else sanitize(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return SECRET.sub("[redacted]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_text(path, *, max_bytes=None):
    limit = MAX_FILE_BYTES if max_bytes is None else max_bytes
    if path.stat().st_size > limit:
        raise ValueError(f"Record exceeds {limit // (1024 * 1024)} MB and was not loaded.")
    return path.read_text(encoding="utf-8-sig")


def read_json(path):
    value = json.loads(read_text(path))
    if not isinstance(value, dict):
        raise ValueError("Record must be a JSON object.")
    return value


def run_directory(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", run_id):
        raise ValueError("Invalid run ID.")
    directory = (root / run_id).resolve()
    if directory.parent != root.resolve() or not (directory / "result.json").is_file():
        raise FileNotFoundError("Run record does not exist.")
    if not (directory / "result.json").resolve().is_relative_to(directory):
        raise ValueError("Result path is outside the run directory.")
    return directory


def read_events(directory):
    events, warnings = [], []
    for relative in TRACE_PATHS:
        path = directory / relative
        if not path.is_file():
            continue
        # A symlink must not let the read-only endpoint escape this run.
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Record path is outside the run directory.")
        for number, line in enumerate(read_text(path, max_bytes=MAX_TRACE_BYTES).splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError()
                event["_source"] = relative
                event["_line"] = number
                events.append(event)
            except ValueError:
                warnings.append(f"{relative} line {number} could not be parsed and was skipped.")
    events.sort(key=lambda event: str(event.get("timestamp", "")))
    return events, warnings


def metadata(event):
    return event.get("llm_metadata") or event.get("backend_metadata") or {}


def brief(directory, result):
    score, report = result.get("score", {}), result.get("report", {})
    return {"id": directory.name, "task_id": result.get("task_id", "Untitled task"),
            "device": result.get("selection", {}).get("selected_device") or
                      report.get("sample_id", "").removeprefix("instrument:"),
            "success": score.get("task_success"), "status": report.get("status"),
            "model_calls": report.get("llm_call_count"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "total_tokens": result.get("usage", {}).get("total_tokens"),
            "has_trace": any((directory / path).is_file() for path in TRACE_PATHS)}


def list_runs(root):
    rows = []
    if not root.is_dir():
        return rows
    for directory in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
        if not directory.is_dir() or not (directory / "result.json").is_file():
            continue
        try:
            directory = run_directory(root, directory.name)
            rows.append(brief(directory, read_json(directory / "result.json")))
        except (OSError, ValueError):
            continue
    return sanitize(rows)


def load_run(root, run_id):
    directory = run_directory(root, run_id)
    result_path = directory / "result.json"
    if not result_path.resolve().is_relative_to(directory):
        raise ValueError("Result path is outside the run directory.")
    result = read_json(result_path)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() and manifest_path.resolve().is_relative_to(directory) else {}
    events, warnings = read_events(directory)
    requests, entries = {}, []
    snapshot, snapshot_source, snapshot_step = None, None, None
    for event in events:
        meta = metadata(event)
        sent = bool(meta.get("request_sent"))
        if sent:
            request_id = meta.get("budget_reservation_id") or event.get("event_id")
            requests[request_id] = meta.get("usage") or {}
        kind = event.get("kind")
        measurement = (event.get("observations") or {}).get("measurement") if kind == "observation" else None
        if kind == "tool_result":
            measurement = (event.get("result") or {}).get("measurement") or measurement
        if measurement and measurement.get("native"):
            snapshot = measurement
            snapshot_source = {"file": event["_source"], "line": event["_line"], "timestamp": event.get("timestamp")}
            if kind == "observation" and entries and entries[-1]["category"] == "action":
                snapshot_step = entries[-1]["index"]
                # The graph emits the fresh readback immediately after execution.
                # Associate it with that action, keeping its own trace provenance.
                entries[-1].update(snapshot=snapshot, snapshot_source=snapshot_source, snapshot_step=snapshot_step)
            elif kind == "tool_result":
                snapshot_step = len(entries)
        if sent or kind in {"tool_result", "instrument_score", "agent_report", "rejected", "role_error", "bounded_failure"}:
            category = "model" if sent else "action" if kind == "tool_result" else "result" if kind in {"instrument_score", "agent_report"} else "error"
            entries.append({"index": len(entries), "category": category, "event": event,
                            "snapshot": snapshot, "snapshot_source": snapshot_source, "snapshot_step": snapshot_step})
    costs = [usage.get("cost") for usage in requests.values()]
    reported = [cost for cost in costs if type(cost) in (float, int) and math.isfinite(cost)]
    return sanitize({
        "summary": {**brief(directory, result), "logged_model_calls": len(requests),
                    "cost_usd": sum(reported) if reported else None,
                    "cost_complete": bool(costs) and len(reported) == len(costs),
                    "event_count": len(events), "display_event_count": len(entries)},
        "task": manifest.get("task", {}), "result": result, "entries": entries,
        "warnings": warnings,
        "files": {"directory": str(directory), "result": str(result_path),
                  "traces": [name for name in TRACE_PATHS if (directory / name).is_file()]},
    })


def response_for(url, root, web_root):
    """Return (status, content type, bytes), or None for the legacy routes."""
    path = unquote(urlsplit(url).path)
    assets = {"/": ("result-viewer.html", "text/html; charset=utf-8"),
              "/results": ("result-viewer.html", "text/html; charset=utf-8"),
              "/viewer.css": ("result-viewer.css", "text/css; charset=utf-8"),
              "/viewer.js": ("result-viewer.js", "text/javascript; charset=utf-8")}
    if path in assets:
        name, content_type = assets[path]
        return 200, content_type, (web_root / name).read_bytes()
    if path != "/api/runs" and not path.startswith("/api/runs/"):
        return None
    try:
        payload = {"runs": list_runs(root)} if path == "/api/runs" else load_run(root, path.removeprefix("/api/runs/"))
        status = 200
    except FileNotFoundError:
        status, payload = 404, {"error": "Run record was not found."}
    except (ValueError, OSError) as exc:
        status, payload = 400, {"error": "Run record could not be loaded.", "detail": type(exc).__name__}
    return status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
