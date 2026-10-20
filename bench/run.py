import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

from langgraph.errors import GraphRecursionError

from app.decision_backend import RulesBackend
from app.graph import build_graph, initial_state
from app.lab_adapter import LabAdapter
from app.models import Limits
from app.roles import ScriptedRoles
from app.scenarios import ROOT, CaseFixture, load_cases
from app.trace import Trace
from bench.score import score


def hashes(directory: str, pattern: str) -> dict:
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / directory).rglob(pattern))}


def manifest(limits: Limits, split: str, repeats: int, backends=None, selected_cases=None, decision_backend=None) -> dict:
    from app.roles import PROMPTS
    from app.retrieval import manifest as retrieval_manifest
    backends = backends or [ScriptedRoles()]
    decision_backend = decision_backend or RulesBackend()
    stage = ("live LLM development pilot" if any(b.is_llm for b in backends) else "scripted role integration test")
    return {"implementation_stage": stage + " with " + decision_backend.backend_id,
        "role_backends": [b.manifest() if hasattr(b, "manifest") else {"backend_id": b.backend_id} for b in backends],
        "prompt_hash": sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
        "retrieval": retrieval_manifest(),
        "decision_backend": decision_backend.manifest() if hasattr(decision_backend, "manifest") else {"backend_id": decision_backend.backend_id},
        "limits": limits.model_dump(), "split": split, "repeats": repeats,
        "selected_cases": selected_cases or [c.case_id for c in load_cases(split)],
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["langgraph", "pydantic"]},
        "source_hashes": {**hashes("app", "*.py"), **hashes("bench", "*.py")},
        "scenario_hashes": hashes(f"cases/{split}", "*.json"),
        "native_provenance": json.loads((ROOT / "vendor/chemlab/manifest.json").read_text()),
        "limitations": ["OpenJev conditional letter probabilities are not calibrated on laboratory tasks", "Balance is an authored sensor, not native ChemLab",
            "No async operations in the selected capability subset", "Full reset/replay only; no world-aware mid-run resume",
            "Lexical protocol retrieval only; no Materials Encoder or model-ranking claims"]}


def run_episode(fixture: CaseFixture, output_dir: Path | None = None, limits: Limits | None = None,
                roles=None, decision_backend=None, repeat: int = 0, observer=None) -> dict:
    limits = limits or Limits()
    roles = roles or ScriptedRoles()
    decision_backend = decision_backend or RulesBackend()
    episode_id = str(uuid4())
    path = output_dir / "traces" / f"{episode_id}.jsonl" if output_dir else None
    trace = Trace(episode_id, path, observer)
    start = time.monotonic()
    completed, report, world, outcome = False, None, None, None
    try:
        lab = LabAdapter(fixture)
        graph = build_graph(lab, trace, limits, roles, decision_backend)
        state = initial_state(lab, episode_id, limits)
        config = {"configurable": {"thread_id": episode_id}, "recursion_limit": limits.graph_steps}
        try:
            final = graph.invoke(state, config)
            report = final["report"]
            completed = True
        except GraphRecursionError:
            snapshot = graph.get_state(config).values
            report = {"episode_id": episode_id, "sample_id": lab.sample_id, "status": "unresolved",
                "terminal_reason": "graph_budget", "evidence_ids": [],
                "action_count": lab.action_count, "llm_call_count": snapshot.get("llm_call_count", 0),
                "role_call_count": snapshot.get("role_call_count", 0), "hypotheses": []}
            trace.emit("harness", "graph_budget", error_code="E8")
        world = lab.evaluator_snapshot()
        outcome = score(fixture, world, report)
    except Exception as exc:
        trace.emit("harness", "infrastructure_error", error=f"{type(exc).__name__}: {exc}", error_code="E9")
    if report is None:
        report = {"episode_id": episode_id, "status": "unresolved", "terminal_reason": "infrastructure_error",
                  "action_count": 0, "llm_call_count": 0, "role_call_count": 0}
    record = {"run_id": episode_id, "case_id": fixture.case_id, "family": fixture.family,
        "repeat": repeat, "seed": fixture.seed, "role_backend": roles.backend_id,
        "decision_backend": decision_backend.backend_id, "is_llm": roles.is_llm,
        "attempted": True, "completed": completed, "elapsed_ms": (time.monotonic() - start) * 1000,
        "fallback_count": sum(bool(e.get("decision", {}).get("fallback_used")) for e in trace.events),
        "report": report, "score": outcome or {"task_success": False, "false_success": False},
        "trace_path": str(path.resolve()) if path else None}
    record["observer_error_count"] = trace.observer_error_count
    calls = [e["llm_metadata"] for e in trace.events if e.get("llm_metadata")]
    usages = [c["usage"] for c in calls if c.get("usage")]
    record["usage"] = {k: sum(u.get(k, 0) for u in usages) for k in ["prompt_tokens", "completion_tokens", "total_tokens"]} if usages else None
    record["usage_complete"] = bool(calls) and len(usages) == len(calls)
    record["usage_cost_currency"] = "USD" if roles.backend_id.startswith("openrouter/") else None
    record["reported_cost_usd"] = sum(u["cost"] for u in usages) if record["usage_cost_currency"] == "USD" and record["usage_complete"] and all(u.get("cost") is not None for u in usages) else None
    record["http_request_count"] = sum(c.get("request_sent", False) for c in calls)
    record["provider_failure_count"] = sum(e["kind"] == "role_error" and e.get("error_code") == "E9" for e in trace.events)
    routing = [e for e in trace.events if e["kind"] == "routing"]
    record["decision_call_count"] = len(routing)
    record["decision_elapsed_ms"] = sum(e["decision"]["latency_ms"] for e in routing)
    record["decision_model_exercised"] = bool(getattr(decision_backend, "is_decision_model", False)
        and any((e.get("backend_metadata") or {}).get("response") for e in routing))
    record["decision_integration_ok"] = (record["decision_model_exercised"] and record["fallback_count"] == 0
                                          if getattr(decision_backend, "is_decision_model", False) else None)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Evaluator-only material is kept separate from public graph traces.
        private_dir = output_dir / "evaluator"
        private_dir.mkdir(exist_ok=True)
        (private_dir / f"{episode_id}.json").write_text(json.dumps({"fixture": fixture.model_dump(), "world": world,
            "score": record["score"]}, indent=2) + "\n")
        with (output_dir / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    return record


def _summarize_one_backend(records: list[dict]) -> list[dict]:
    rows = []
    for family in sorted({r["family"] for r in records}) + ["overall"]:
        selected = [r for r in records if family == "overall" or r["family"] == family]
        n = len(selected)
        claimed = sum(r["report"]["status"] == "supported" for r in selected)
        recovery_n = sum(r["family"] in {"process", "measurement"} for r in selected)
        pairs = {}
        for r in selected:
            pairs.setdefault(r["case_id"], []).append(r)
        repeated = [rs for rs in pairs.values() if len(rs) >= 2]
        false_n = sum(r["score"].get("false_success", False) for r in selected)
        rows.append({"role_backend": selected[0]["role_backend"], "decision_backend": selected[0]["decision_backend"], "family": family, "attempted": n, "completed": sum(r["completed"] for r in selected),
            "task_success_count": sum(r["score"]["task_success"] for r in selected),
            "task_success_rate": sum(r["score"]["task_success"] for r in selected) / n,
            "expected_cause_covered_count": sum(r["score"].get("expected_cause_covered", r["score"].get("diagnosis_correct", False)) for r in selected),
            "expected_cause_coverage": sum(r["score"].get("expected_cause_covered", r["score"].get("diagnosis_correct", False)) for r in selected) / n,
            "recovery_cases": recovery_n,
            "recovery_count": sum(r["score"].get("recovered", False) for r in selected),
            "recovery_rate": sum(r["score"].get("recovered", False) for r in selected) / recovery_n if recovery_n else "",
            "success_claims": claimed, "false_success_count": false_n,
            "false_success_rate": false_n / claimed if claimed else "",
            "appropriate_abstention_count": sum(r["score"].get("appropriate_abstention", False) for r in selected),
            "repeated_cases": len(repeated),
            "repeat_consistency": sum(all(r["score"]["task_success"] for r in rs) for rs in repeated) / len(repeated) if repeated else "",
            "mean_actions": sum(r["report"]["action_count"] for r in selected) / n,
            "mean_role_calls": sum(r["report"].get("role_call_count", 0) for r in selected) / n,
            "mean_elapsed_ms": sum(r["elapsed_ms"] for r in selected) / n,
            "llm_calls": sum(r["report"]["llm_call_count"] for r in selected),
            "provider_failures": sum(r.get("provider_failure_count", 0) for r in selected),
            "decision_calls": sum(r.get("decision_call_count", 0) for r in selected),
            "decision_elapsed_ms": sum(r.get("decision_elapsed_ms", 0) for r in selected),
            "reported_cost_usd": sum(r["reported_cost_usd"] for r in selected) if all(r.get("reported_cost_usd") is not None for r in selected) else "",
            "fallback_count": sum(r["fallback_count"] for r in selected)})
    return rows


def summarize_runs(records: list[dict]) -> list[dict]:
    return [row for roles, decision in sorted({(r["role_backend"], r["decision_backend"]) for r in records})
            for row in _summarize_one_backend([r for r in records if (r["role_backend"], r["decision_backend"]) == (roles, decision)])]


def main():
    parser = argparse.ArgumentParser(description="Offline skeleton regression suite; not a model benchmark")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", help="Explicit free OpenRouter IDs; omit for offline roles")
    parser.add_argument("--local-roles", action="store_true", help="Experimental local generative specialists")
    parser.add_argument("--cases", nargs="+", help="Optional development smoke-test subset; omit for a full suite")
    parser.add_argument("--router", choices=["rules", "openjev"], default="rules")
    parser.add_argument("--decision-url", default="http://127.0.0.1:8792")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.models and args.local_roles:
        parser.error("Choose either cloud models or local roles")
    cases = load_cases(args.split)
    if args.cases:
        if set(args.cases) - {c.case_id for c in cases} or len(set(args.cases)) != len(args.cases):
            parser.error("--cases must contain unique IDs from the selected split")
        cases = [c for c in cases if c.case_id in args.cases]
    if args.models and (args.split != "dev" or len(cases) != 1 or len(args.models) != 1 or args.repeats != 1):
        parser.error("OpenRouter free models are debug-only: one model, one development case, one attempt; batch evaluation is disabled")
    output = args.output or ROOT / "results" / f"offline-{uuid4().hex[:10]}"
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty, to preserve earlier results")
    output.mkdir(parents=True, exist_ok=True)
    limits = Limits()
    decision = RulesBackend()
    if args.router == "openjev":
        from app.local_decision import LocalOpenJevBackend
        decision = LocalOpenJevBackend(args.decision_url)
    backends = [ScriptedRoles()]
    if args.models:
        from app.openrouter import OpenRouterRoles, model_catalog
        catalog = model_catalog()
        backends = [OpenRouterRoles(model, catalog=catalog) for model in args.models]
    elif args.local_roles:
        from app.local_roles import LocalLlamaRoles
        backends = [LocalLlamaRoles(args.decision_url)]
    (output / "manifest.json").write_text(json.dumps(manifest(limits, args.split, args.repeats, backends, [c.case_id for c in cases], decision), indent=2) + "\n")
    records = []
    for repeat in range(args.repeats):
        for case in cases:
            # Alternate model order on matched cases/repeats; each episode resets the world.
            order = backends if (repeat + case.seed) % 2 == 0 else list(reversed(backends))
            for backend in order:
                record = run_episode(case, output, limits, roles=backend, decision_backend=decision, repeat=repeat)
                records.append(record)
                print(f"{case.case_id} {backend.backend_id} repeat={repeat} success={record['score']['task_success']} actions={record['report']['action_count']} reason={record['report']['terminal_reason']}", flush=True)
    rows = summarize_runs(records)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Artifacts: {output.resolve()}")
    if args.router == "openjev" and not all(r["decision_integration_ok"] for r in records):
        print("Local decision-model integration did not pass: inspect fallback/readout errors.")
        raise SystemExit(2)
    if not all(r["score"]["task_success"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
