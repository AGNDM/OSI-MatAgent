"""Exactly three preselected development attempts, never the full benchmark matrix."""
import argparse
import json
from pathlib import Path
import time

import httpx

from app.expert_config import CompatibleExpert, load_plan
from app.local_decision import LocalOpenJevBackend
from app.models import Limits
from app.scenarios import case_by_id
from bench.annotate import analyze
from bench.matrix import fingerprint, write_summary
from bench.run import manifest, run_episode


def write_report(output, records):
    tokens = {name: sum((r.get("usage") or {}).get(name, 0) for r in records)
              for name in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    lines = ["# SiliconFlow DeepSeek V4.1 Flash + local OpenJev: three-trajectory development pilot", "",
        "Expert roles: DeepSeek V4.1 Flash on SiliconFlow. Router: pinned local OpenJev Q4. No scripted expert or rules substitution; fallback is explicitly recorded if it occurs.", "",
        "Three development attempts, one per selected case. Not a model ranking or a formal test-suite comparison. No between-case prompt tuning or best-of retries.", "",
        "| Case | Task success | Expected cause covered (not diagnosis accuracy) | Actions | Expert calls | Router calls | Fallbacks | Seconds | Input tokens | Output tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        usage = r.get("usage") or {}
        lines.append(f"| {r['case_id']} | {r['score']['task_success']} | {r['score'].get('expected_cause_covered')} | {r['report']['action_count']} | {r['report']['llm_call_count']} | {r['decision_call_count']} | {r['fallback_count']} | {r['elapsed_ms']/1000:.2f} | {usage.get('prompt_tokens','unknown')} | {usage.get('completion_tokens','unknown')} |")
    lines += ["", f"Reported usage totals: {json.dumps(tokens)}. Total tokens in millions: {tokens['total_tokens']/1e6:.6f}.",
        f"Usage complete for all attempts: {all(r['usage_complete'] for r in records)}. Failed/unmetered requests may make these totals incomplete.",
        "Cost is unknown unless the provider explicitly returns a billable amount and currency; token counts are not a billing statement. No USD amount is inferred from an unspecified provider cost field."]
    for r in records:
        lines += ["", f"## {r['case_id']} / {r['run_id']}", "",
            f"Terminal reason: {r['report']['terminal_reason']}; official score: `{json.dumps(r['score'])}`."]
        events = [json.loads(s) for s in Path(r["trace_path"]).read_text().splitlines()]
        for e in events:
            prefix = f"- `{e['event_id']}` "
            if e["kind"] == "role_output":
                out = e["output"]
                value = out.get("intent") if e["component"] == "diagnostic" else out
                lines.append(prefix + e["component"] + ": `" + json.dumps(value, ensure_ascii=False) + "`")
            elif e["kind"] == "tool_result":
                lines.append(prefix + "executed " + e["proposal"]["capability_id"] + ": `" + json.dumps(e["result"], ensure_ascii=False) + "`")
            elif e["kind"] in {"rejected", "role_error", "execution_error"}:
                lines.append(prefix + e["kind"] + ": " + str(e.get("error")))
        lines.append(f"\nRaw trace: `traces/{r['run_id']}.jsonl`. Semantic/causal review remains separate from automatic suggestions.")
    (output / "review.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decision-url", default="http://127.0.0.1:8792")
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new directory; previous attempts are never replaced")
    config = load_plan(args.config).reference
    if config.provider != "siliconflow" or config.model != "deepseek-ai/DeepSeek-V4.1-Flash":
        p.error("This explicitly authorized pilot is limited to SiliconFlow DeepSeek V4.1 Flash")
    roles = CompatibleExpert(config)
    decision = LocalOpenJevBackend(args.decision_url)
    health = httpx.get(args.decision_url + "/health", timeout=5, trust_env=False)
    health.raise_for_status()
    cases = [case_by_id(key) for key in ["dev-01", "dev-02", "dev-03"]]
    frozen = fingerprint()
    args.output.mkdir(parents=True)
    provenance = manifest(Limits(), "dev", 1, [roles], [c.case_id for c in cases], decision)
    provenance.update(experiment="exactly three user-authorized development trajectories",
        frozen_fingerprint=frozen, reasoning="SiliconFlow enable_thinking=false", transport_retries=0,
        provider_docs="https://docs.siliconflow.cn/docs/api/chat-completions-post")
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records, start = [], time.monotonic()
    try:
        for case in cases:
            if fingerprint() != frozen:
                raise RuntimeError("Sources changed during frozen pilot; stopping")
            print(f"Starting {case.case_id}", flush=True)
            def observer(event):
                if event["kind"] == "routing":
                    print("  route -> " + event["decision"]["choice"], flush=True)
                elif event["kind"] == "role_output":
                    out = event["output"]
                    detail = (out.get("intent") or {}).get("capability_id") if event["component"] == "diagnostic" else out.get("capability_id", out.get("status"))
                    print(f"  {event['component']} -> {detail}", flush=True)
                elif event["kind"] in {"role_error", "rejected"}:
                    print(f"  {event['kind']}: {event.get('error')}", flush=True)
            r = run_episode(case, args.output, roles=roles, decision_backend=decision, observer=observer)
            records.append(r)
            write_summary(args.output, records)
            write_report(args.output, records)
            print(f"Completed {case.case_id}: success={r['score']['task_success']} reason={r['report']['terminal_reason']} tokens={r['usage']}", flush=True)
            if any(e.get("llm_metadata", {}).get("http_status") in {401, 402, 403} for e in
                   map(json.loads, Path(r["trace_path"]).read_text().splitlines()) if e.get("llm_metadata")):
                raise RuntimeError("Provider authorization/balance error; no repeated billable attempts")
    finally:
        (args.output / "status.json").write_text(json.dumps({"planned": 3, "attempted": len(records),
            "pilot_complete": len(records) == 3, "task_success_count": sum(r["score"]["task_success"] for r in records),
            "elapsed_seconds": time.monotonic()-start}, indent=2) + "\n")
        if records:
            analyze(args.output)


if __name__ == "__main__":
    main()
