"""Frozen proposal experiment A/B: two LLMs, learned routing, matched rule baseline."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from app.decision_backend import RulesBackend
from app.expert_config import CompatibleExpert, load_plan
from app.local_decision import LocalOpenJevBackend
from app.models import Limits
from app.scenarios import load_cases
from bench.run import hashes, manifest, run_episode, summarize_runs


def fingerprint():
    from app.retrieval import manifest as retrieval_manifest
    return {"sources": {**hashes("app", "*.py"), **hashes("bench", "*.py")},
        "dev_cases": hashes("cases/dev", "*.json"), "test_cases": hashes("cases/test", "*.json"),
        "retrieval": retrieval_manifest()}


def schedule(models, cases, repeats):
    configs = [(models[0], "openjev", "A-reference"), (models[1], "openjev", "A-contrast"),
               (models[0], "rules", "B-baseline")]
    return [{"case_id": c.case_id, "repeat": repeat, "model": model, "router": router, "arm": arm}
        for repeat in range(repeats) for index, c in enumerate(cases)
        for model, router, arm in configs[(index + repeat) % 3:] + configs[:(index + repeat) % 3]]


def write_summary(output, records):
    rows = summarize_runs(records)
    if not rows:
        return
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs=2, help="Retired free-model interface; execution is prohibited")
    p.add_argument("--config", type=Path, help="ExpertPlan JSON for two general local model endpoints")
    p.add_argument("--repeats", type=int, choices=[1, 2], default=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--freeze-only", action="store_true", help="Save an unexecuted plan without cloud access")
    p.add_argument("--execute", action="store_true", help="Explicitly execute configured local expert endpoints; default is plan only")
    args = p.parse_args()
    if args.models:
        p.error("Free OpenRouter batches are retired: debug only via app.cli; use --config for general expert endpoints")
    if not args.config:
        p.error("--config is required; no expert models are assumed")
    if args.execute and args.freeze_only:
        p.error("Choose plan-only or --execute")
    try:
        expert_plan = load_plan(args.config)
    except ValueError as exc:
        p.error(str(exc))
    specifications = [expert_plan.reference, expert_plan.contrast]
    models = [s.name for s in specifications]
    frozen = fingerprint()
    plan = schedule(models, load_cases("test"), args.repeats)
    config = {"experts": expert_plan.model_dump(), "repeats": args.repeats, "fingerprint": frozen, "plan": plan,
        "label": "48-episode benchmark" if args.repeats == 2 else "24-episode single-repeat pilot",
        "routing_policy": "argmax-v1"}
    records = []
    if args.resume:
        previous = json.loads((args.output / "freeze.json").read_text())
        if previous != config:
            p.error("Frozen sources, cases or configuration changed; resume refused")
        records = [json.loads(s) for s in (args.output / "runs.jsonl").read_text().splitlines()] if (args.output / "runs.jsonl").exists() else []
    elif args.output.exists():
        p.error("Use a new output directory or --resume")
    if not args.execute:
        if args.resume:
            p.error("--freeze-only is for a new experiment")
        args.output.mkdir(parents=True)
        (args.output / "freeze.json").write_text(json.dumps(config, indent=2) + "\n")
        (args.output / "status.json").write_text(json.dumps({"planned": len(plan), "attempted": 0,
            "matrix_complete": False, "state": "plan only; general experts must be configured before execution",
            "unconfigured_experts": [s.name for s in specifications if not s.configured]}, indent=2) + "\n")
        print(f"Frozen {len(plan)} episodes without network calls: {args.output}")
        return
    # Never inspect OpenRouter or send any request while preparing an experiment.
    try:
        roles = {s.name: CompatibleExpert(s) for s in specifications}
    except ValueError as exc:
        p.error(str(exc))
    decisions = {"rules": RulesBackend(), "openjev": LocalOpenJevBackend()}
    if not args.resume:
        args.output.mkdir(parents=True)
        (args.output / "freeze.json").write_text(json.dumps(config, indent=2) + "\n")
    if not (args.output / "manifest.json").exists():
        provenance = manifest(Limits(), "test", args.repeats, list(roles.values()), decision_backend=decisions["openjev"])
        provenance.update(experiment=config["label"], frozen_at=datetime.now(timezone.utc).isoformat(),
            baseline=decisions["rules"].backend_id, freeze_sha256=sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())
        (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    completed_keys = {(r["case_id"], r["repeat"], r["role_backend"], r["decision_backend"]) for r in records}
    by_case = {c.case_id: c for c in load_cases("test")}
    infrastructure_streak = 0
    try:
        for item in plan:
            backend, decision = roles[item["model"]], decisions[item["router"]]
            key = (item["case_id"], item["repeat"], backend.backend_id, decision.backend_id)
            if key in completed_keys:
                continue
            if fingerprint() != frozen:
                raise RuntimeError("Frozen implementation changed during experiment; stopping before next episode")
            r = run_episode(by_case[item["case_id"]], args.output, Limits(), backend, decision, item["repeat"])
            records.append(r)
            write_summary(args.output, records)
            print(f"{len(records)}/{len(plan)} {item['arm']} {item['case_id']} success={r['score']['task_success']} reason={r['report']['terminal_reason']} fallback={r['fallback_count']}", flush=True)
            infrastructure_streak = infrastructure_streak + 1 if r["provider_failure_count"] else 0
            if infrastructure_streak >= 3:
                raise RuntimeError("Three consecutive provider-failed episodes; preserving partial results, use --resume when service recovers")
    finally:
        (args.output / "status.json").write_text(json.dumps({"planned": len(plan), "attempted": len(records),
            "matrix_complete": len(records) == len(plan), "successful": sum(r["score"]["task_success"] for r in records),
            "note": "All attempts retained. Completion is not task success; infrastructure failures stay in denominator."}, indent=2) + "\n")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
