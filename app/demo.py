"""Loopback-only live demonstration. No cloud backend is available from this UI."""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4

from app.decision_backend import RulesBackend
from app.expert_config import CompatibleExpert, load_plan
from app.local_decision import LocalOpenJevBackend
from app.roles import ScriptedRoles
from app.scenarios import ROOT, case_by_id, load_cases
from bench.replay import replay
from bench.run import manifest, run_episode
from app.models import Limits


class DemoController:
    def __init__(self, output, decision_url="http://127.0.0.1:8792", expert=None, exploratory=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.decision_url, self.expert = decision_url, expert
        self.lock = Lock()
        self.modes = {"scripted-rules": "Scripted specialists + rules router (regression)",
            "scripted-openjev": "Scripted specialists + local OpenJev router"}
        if expert and expert.configured:
            self.modes["configured-openjev"] = "Configured general specialists + local OpenJev router"
        if exploratory:
            self.modes["exploratory-openjev"] = "Exploratory: OpenJev used for specialists and routing"
        self.latest = {"running": False, "events": [], "record": None, "display_mode": "idle"}
        self.history = {}
        for directory in sorted(self.output.glob("demo-*"), key=lambda p: p.stat().st_mtime)[-20:]:
            try:
                record = json.loads((directory / "runs.jsonl").read_text().splitlines()[-1])
                metadata = json.loads((directory / "demo.json").read_text())
                self.history[directory.name] = {"directory": directory, "record": record, **metadata}
            except (OSError, ValueError, IndexError):
                continue

    def state(self):
        with self.lock:
            return {**deepcopy(self.latest), "modes": self.modes,
                "cases": [c.case_id for c in load_cases("dev")],
                "history": [{"id": key, "case_id": v["case_id"], "mode": v["mode"]}
                            for key, v in self.history.items()]}

    @staticmethod
    def public_event(event):
        return {k: v for k, v in event.items() if k not in {"llm_metadata", "backend_metadata", "raw_output", "input"}}

    def start(self, case_id, mode):
        case_by_id(case_id)  # validate before creating any output or thread
        if mode not in self.modes:
            raise ValueError("Unsupported/unconfigured demo mode; no cloud backend is available")
        with self.lock:
            if self.latest["running"]:
                raise ValueError("An episode is running; wait for its terminal result")
            key = "demo-" + uuid4().hex[:12]
            self.latest = {"id": key, "case_id": case_id, "mode": mode,
                "running": True, "display_mode": "live", "events": [], "record": None}
        Thread(target=self._run, args=(key, case_id, mode), daemon=True).start()
        return key

    def _run(self, key, case_id, mode):
        directory = self.output / key
        try:
            directory.mkdir()
            roles = ScriptedRoles()
            if mode == "configured-openjev":
                roles = CompatibleExpert(self.expert)
            elif mode == "exploratory-openjev":
                from app.local_roles import LocalLlamaRoles
                roles = LocalLlamaRoles(self.decision_url)
            decision = RulesBackend() if mode == "scripted-rules" else LocalOpenJevBackend(self.decision_url)
            metadata = {"case_id": case_id, "mode": mode}
            (directory / "demo.json").write_text(json.dumps(metadata) + "\n")
            provenance = manifest(Limits(), "dev", 1, [roles], [case_id], decision)
            provenance["demo_mode"] = self.modes[mode]
            (directory / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
            def observer(event):
                with self.lock:
                    self.latest["events"].append(self.public_event(event))
            result = run_episode(case_by_id(case_id), directory, roles=roles, decision_backend=decision, observer=observer)
            with self.lock:
                self.latest.update(record=result, running=False)
                self.history[key] = {"directory": directory, "record": result, **metadata}
                while len(self.history) > 20:
                    del self.history[next(iter(self.history))]
        except Exception as exc:
            with self.lock:
                self.latest.update(running=False, error=f"{type(exc).__name__}: {exc}")

    def view(self, key):
        with self.lock:
            if self.latest["running"]:
                raise ValueError("Cannot replace the live view while an episode is running")
            item = self.history[key]
            path = item["directory"] / "traces" / f"{item['record']['run_id']}.jsonl"
            events = [self.public_event(json.loads(s)) for s in path.read_text().splitlines()]
            self.latest = {"id": key, "case_id": item["case_id"], "mode": item["mode"],
                "running": False, "display_mode": "recorded", "record": item["record"], "events": events}

    def replay(self, key):
        with self.lock:
            if self.latest["running"]:
                raise ValueError("Wait for the live episode before replay")
            item = self.history[key]
            path = item["directory"] / "traces" / f"{item['record']['run_id']}.jsonl"
            matched = replay(item["case_id"], path)
            result = {"recording_id": key, "run_id": item["record"]["run_id"], "case_id": item["case_id"],
                "matched_actions": matched, "world_reset": True, "model_calls": 0,
                "note": "Full reset/action replay, not checkpoint resume or a new model run"}
            self.latest["replay_result"] = result
            return result

    def reset(self):
        with self.lock:
            if self.latest["running"]:
                raise ValueError("Cannot reset a running episode")
            self.latest = {"running": False, "events": [], "record": None, "display_mode": "idle"}


def handler_for(controller):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = (ROOT / "web/demo.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self.respond(200, controller.state())
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected = f"http://{self.headers.get('Host')}"
            if origin and origin != expected:
                self.respond(403, {"error": "Cross-origin commands are disabled"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Use JSON commands")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid command size")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/run":
                    result = {"id": controller.start(body["case_id"], body["mode"])}
                elif self.path == "/api/view":
                    controller.view(body["id"])
                    result = {"ok": True}
                elif self.path == "/api/replay":
                    result = controller.replay(body["id"])
                elif self.path == "/api/reset":
                    controller.reset()
                    result = {"ok": True}
                else:
                    self.respond(404, {"error": "Not found"})
                    return
                self.respond(200, result)
            except (ValueError, KeyError, StopIteration) as exc:
                self.respond(400, {"error": str(exc)})
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8793)
    p.add_argument("--output", type=Path, default=ROOT / "results/live-demo")
    p.add_argument("--decision-url", default="http://127.0.0.1:8792")
    p.add_argument("--expert-config", type=Path)
    p.add_argument("--allow-exploratory", action="store_true")
    args = p.parse_args()
    LocalOpenJevBackend(args.decision_url)  # validate, no request
    expert = load_plan(args.expert_config).reference if args.expert_config else None
    controller = DemoController(args.output, args.decision_url, expert, args.allow_exploratory)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(controller))
    print(f"Live local demo: http://127.0.0.1:{args.port} (no cloud backend)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
